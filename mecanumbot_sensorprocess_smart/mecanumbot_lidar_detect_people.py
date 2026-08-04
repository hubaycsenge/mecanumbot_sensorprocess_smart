#!/usr/bin/env python3
import os
import math
import time
from contextlib import nullcontext

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    DurabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

# Removed interp1d, using pure numpy indexing for speed
from scipy.ndimage import median_filter, binary_dilation
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

import torch
from dr_spaam.detector import Detector
from ament_index_python.packages import get_package_share_directory

# ---- 1. Determine Device Dynamically ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 2. Dynamic Monkey-patch for torch.load ----
_original_torch_load = torch.load
np.int = int
np.float = float
np.bool = bool


def processor_load(path, *args, **kwargs):
    return _original_torch_load(path, map_location=DEVICE)


class Track:
    """Represents a single tracked person."""

    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([detection[0], detection[1], 0.0, 0.0]).reshape(4, 1)

        # F is rebuilt on every predict() from the measured time step, because the
        # network no longer runs once per scan: the interval between two tracker
        # updates depends on the inference rate cap, not on the LiDAR rate.
        self.kf.F = np.eye(4)

        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.kf.P *= 10.0
        self.kf.R *= 0.5
        self.kf.Q *= 0.01

        self.time_since_update = 0.0
        self.hits = 1
        self.has_moved = False
        self.speed_thresh = 0.1

    def predict(self, dt):
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt
        self.kf.predict()
        self.time_since_update += dt
        return self.kf.x[:2].reshape(-1)

    def update(self, detection):
        self.kf.update(detection.reshape(2, 1))
        self.time_since_update = 0.0
        self.hits += 1

        vx = self.kf.x[2, 0]
        vy = self.kf.x[3, 0]
        speed = np.hypot(vx, vy)

        if speed > self.speed_thresh:
            self.has_moved = True


class MultiObjectTracker:
    """Manages all active tracks and matches new detections.

    Ages tracks in seconds rather than in frames so that the behaviour does not
    change when the detector runs at a lower rate than the LiDAR.
    """

    def __init__(self, max_distance=0.5, max_missed_time=0.4, min_hits=2):
        self.max_distance = max_distance
        self.max_missed_time = max_missed_time
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0

    def _confirmed_positions(self):
        valid_positions = [
            t.kf.x[:2].reshape(-1)
            for t in self.tracks
            if t.hits >= self.min_hits and t.has_moved
        ]
        return (
            np.array(valid_positions) if len(valid_positions) > 0 else np.empty((0, 2))
        )

    def predict_only(self, dt):
        """Advance the motion model without a measurement.

        Used on scans where the detector was skipped, so that the published
        detections keep moving at LiDAR rate instead of freezing between
        inferences.
        """
        for track in self.tracks:
            track.predict(dt)
        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_missed_time
        ]
        return self._confirmed_positions()

    def update(self, detections, dt):
        if len(self.tracks) == 0:
            predicted_positions = np.empty((0, 2))
        else:
            predicted_positions = np.array([track.predict(dt) for track in self.tracks])

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.linalg.norm(
                predicted_positions[:, None, :] - detections[None, :, :], axis=2
            )
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(track_indices, det_indices):
                if cost_matrix[t_idx, d_idx] < self.max_distance:
                    matched_indices.append((t_idx, d_idx))
                    unmatched_detections.remove(d_idx)
                    unmatched_tracks.remove(t_idx)

        for t_idx, d_idx in matched_indices:
            self.tracks[t_idx].update(detections[d_idx])

        for d_idx in unmatched_detections:
            self.tracks.append(Track(detections[d_idx], self.next_id))
            self.next_id += 1

        self.tracks = [
            t for t in self.tracks if t.time_since_update <= self.max_missed_time
        ]

        return self._confirmed_positions()


class DrSpaamNode(Node):
    """ROS2 node performing people detection using DR-SPAAM on 2D LiDAR."""

    def __init__(self):
        super().__init__("mecanumbot_lidar_detect_people")
        torch.load = processor_load
        self.get_logger().info(f"DEVICE: {DEVICE}")

        self.declare_parameter("weight_file", "dr_spaam_5_on_frog.pth")
        self.declare_parameter("conf_thresh", 0.45)
        self.declare_parameter("stride", 2)
        self.declare_parameter("scan_topic", "/mecanumbot/scan")
        self.declare_parameter("detections_topic", "dets")
        self.declare_parameter("rviz_topic", "dets_marker")
        self.declare_parameter("leading_mode", True)
        self.declare_parameter("obstacle_exclusion_radius", 0.2)
        self.declare_parameter("detection_frame", "base_scan")
        # --- GPU load control (see scan_callback) ---
        self.declare_parameter("max_inference_rate", 5.0)
        self.declare_parameter("publish_on_skipped_scans", True)
        self.declare_parameter("use_amp", True)
        self.declare_parameter("cudnn_benchmark", True)
        self.declare_parameter("torch_threads", 2)
        self.declare_parameter("idle_skip_range", 0.0)
        self.declare_parameter("expected_points", 240)
        self.declare_parameter("angle_increment", 0.026)
        self.declare_parameter("perf_log_period", 0.0)
        # --- tracker ---
        self.declare_parameter("track_max_distance", 0.5)
        self.declare_parameter("track_max_missed_time", 0.4)
        self.declare_parameter("track_min_hits", 2)

        self.weight_file = self.get_parameter("weight_file").value
        self.conf_thresh = self.get_parameter("conf_thresh").value
        self.stride = self.get_parameter("stride").value
        self.leading_mode = self.get_parameter("leading_mode").value
        self.exclusion_radius = self.get_parameter("obstacle_exclusion_radius").value
        self.detection_frame = str(self.get_parameter("detection_frame").value)

        max_inference_rate = float(self.get_parameter("max_inference_rate").value)
        self.min_inference_period = (
            1.0 / max_inference_rate if max_inference_rate > 0.0 else 0.0
        )
        self.publish_on_skipped_scans = bool(
            self.get_parameter("publish_on_skipped_scans").value
        )
        self.idle_skip_range = float(self.get_parameter("idle_skip_range").value)
        self.expected_points = int(self.get_parameter("expected_points").value)
        self.angle_increment = float(self.get_parameter("angle_increment").value)
        self.perf_log_period = float(self.get_parameter("perf_log_period").value)

        self.use_gpu = DEVICE.type == "cuda"
        self.use_amp = bool(self.get_parameter("use_amp").value) and self.use_gpu

        torch_threads = int(self.get_parameter("torch_threads").value)
        if torch_threads > 0:
            # The cutout tensor is tiny, so intra-op threading buys nothing and
            # only steals CPU from the scan pre-processing and the ROS executor.
            torch.set_num_threads(torch_threads)

        if self.use_gpu and bool(self.get_parameter("cudnn_benchmark").value):
            # Every inference has exactly the same shape, so let cuDNN autotune
            # its 1D convolution kernels once and reuse the choice.
            torch.backends.cudnn.benchmark = True

        if self.detection_frame not in ("base_scan", "map"):
            self.get_logger().warn(
                f"Invalid detection_frame '{self.detection_frame}', defaulting to 'base_scan'."
            )
            self.detection_frame = "base_scan"

        pkg_share = get_package_share_directory("mecanumbot_sensorprocess_smart")
        weight_path = os.path.join(pkg_share, "models", self.weight_file)
        self.pose_out = None
        self.last_pose_out = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if not os.path.isfile(weight_path):
            self.get_logger().error(f"DR-SPAAM model file not found: {weight_path}")
            raise FileNotFoundError(weight_path)

        self.detector = Detector(
            model_name="DR-SPAAM",
            ckpt_file=weight_path,
            gpu=self.use_gpu,
            stride=self.stride,
        )
        self.detector.set_laser_spec(
            angle_inc=self.angle_increment, num_pts=self.expected_points
        )
        self._validate_amp_and_warm_up()

        self.tracker = MultiObjectTracker(
            max_distance=float(self.get_parameter("track_max_distance").value),
            max_missed_time=float(self.get_parameter("track_max_missed_time").value),
            min_hits=int(self.get_parameter("track_min_hits").value),
        )

        # Inference scheduling / perf bookkeeping
        self.last_inference_time = None
        self.last_scan_time = None
        self.inference_count = 0
        self.skipped_count = 0
        self.inference_time_sum = 0.0
        self.last_perf_log_time = None

        # ---- Map State Data ----
        self.map_data = None
        self.extended_map = None
        self.map_resolution = 0.05
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_width = 0
        self.map_height = 0

        self.dets_pub = self.create_publisher(
            PoseArray, self.get_parameter("detections_topic").value, 10
        )

        # Flag to track if rviz visualization is needed to avoid useless computation
        self.publish_rviz = False
        # self.rviz_pub = self.create_publisher(Marker, self.get_parameter("rviz_topic").value, 10)

        if self.leading_mode:
            self.subject_pub = self.create_publisher(PoseStamped, "subject_pose", 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.get_parameter("scan_topic").value,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, "keepout_filter_mask", self.map_callback, map_qos
        )

        self.get_logger().info("DR-SPAAM ROS2 detector node started.")

    def _inference_context(self):
        """Half-precision autocast when enabled, otherwise a no-op context.

        `model.half()` is deliberately not used. _SpatialAttention builds its
        neighbour mask lazily at runtime instead of registering it as a buffer,
        so `.half()` never reaches it and it stays FP32. That FP32 mask promotes
        the masked softmax back to FP32, and the weighted-average matmul that
        follows then fails outright with
        "expected m1 and m2 to have the same dtype, but got: float != c10::Half"
        (and the `1e10` masking constant is `inf` in FP16 anyway). Autocast
        instead leaves that arithmetic alone and only runs the convolutions --
        which is where the time actually goes -- in FP16.
        """
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _run_detector(self, scan):
        with self._inference_context():
            return self.detector(scan)

    def _validate_amp_and_warm_up(self):
        """Warm the model up, and verify half precision actually behaves.

        Two passes are needed: the first only populates the auto-regressive
        feature template, the second is the one that goes through the spatial
        attention gate, which is the part that half precision can break. Doing
        this at start-up also pays the cuDNN autotuning and CUDA context cost
        here instead of on the first real scan.
        """
        dummy = np.full(self.expected_points, 5.0, dtype=float)
        middle = self.expected_points // 2
        dummy[middle : middle + 8] = (
            2.0  # a person-sized dip, so the graph sees realistic input
        )

        while True:
            self.detector._fea = None
            try:
                for _ in range(2):
                    _, dets_cls, _ = self._run_detector(dummy)
                if not np.all(np.isfinite(np.asarray(dets_cls, dtype=np.float64))):
                    raise ValueError("non-finite confidences")
            except Exception as exc:
                if self.use_amp:
                    self.get_logger().warn(
                        f"Half-precision inference failed ({exc}); falling back to FP32."
                    )
                    self.use_amp = False
                    continue
                self.get_logger().error(f"Model warm-up failed: {exc}")
                raise
            break

        self.detector._fea = None
        self.get_logger().info(
            f"DR-SPAAM ready: device={DEVICE.type}, stride={self.stride}, "
            f"points={self.expected_points}, cutouts={len(range(0, self.expected_points, self.stride))}, "
            f"precision={'fp16-autocast' if self.use_amp else 'fp32'}, "
            f"max_inference_rate="
            f"{'unlimited' if self.min_inference_period <= 0.0 else f'{1.0 / self.min_inference_period:.1f} Hz'}"
        )

    def map_callback(self, msg: OccupancyGrid):
        """Updates internal static map grid and generates the obstacle exclusion zone."""
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.map_width = msg.info.width
        self.map_height = msg.info.height

        raw_map = np.array(msg.data, dtype=np.int8).reshape(
            (self.map_height, self.map_width)
        )
        self.map_data = raw_map

        obstacles = raw_map > 65
        radius_px = int(math.ceil(self.exclusion_radius / self.map_resolution))

        if radius_px > 0:
            y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
            circular_footprint = x**2 + y**2 <= radius_px**2
            self.extended_map = binary_dilation(obstacles, structure=circular_footprint)
        else:
            self.extended_map = obstacles

        self.get_logger().info(
            f"Occupancy map received. Extended exclusion zone built (Radius: {radius_px}px)."
        )

    def _filter_detections_by_map(self, dets_xy, transform):
        """Vectorized filtering of detections that fall within the inflated map occupancy."""
        if self.extended_map is None or transform is None or len(dets_xy) == 0:
            return dets_xy

        # Extract translation
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        # Convert Quaternion to Yaw angle (Euler)
        q = transform.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # 1. Transform all local sensor coordinates into global map coordinates simultaneously
        local_x = dets_xy[:, 1]
        local_y = dets_xy[:, 0]

        global_x = tx + (local_x * cos_yaw) - (local_y * sin_yaw)
        global_y = ty + (local_x * sin_yaw) + (local_y * cos_yaw)

        # 2. Convert physical global coordinates to map pixel indices
        px = ((global_x - self.map_origin_x) / self.map_resolution).astype(int)
        py = ((global_y - self.map_origin_y) / self.map_resolution).astype(int)

        # 3. Create a boolean mask to keep valid (safe) points
        safe_mask = np.ones(len(dets_xy), dtype=bool)

        # Check if points are within map bounds
        in_bounds_mask = (
            (px >= 0) & (px < self.map_width) & (py >= 0) & (py < self.map_height)
        )

        # Of the points that are in bounds, reject the ones inside the exclusion zone
        # (~ inverts the map boolean so True means safe)
        safe_mask[in_bounds_mask] = ~self.extended_map[
            py[in_bounds_mask], px[in_bounds_mask]
        ]

        return dets_xy[safe_mask]

    def scan_callback(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = (
            0.1 if self.last_scan_time is None else max(1e-3, now - self.last_scan_time)
        )
        self.last_scan_time = now

        scan = np.array(msg.ranges)
        scan = preprocess_lidar(scan, target_len=self.expected_points, max_range=10.0)

        if self._should_skip_inference(now, scan):
            self.skipped_count += 1
            if self.publish_on_skipped_scans:
                # Keep `dets` and `subject_pose` flowing at LiDAR rate by
                # extrapolating the existing tracks instead of re-running the net.
                self._publish_tracks(
                    self.tracker.predict_only(dt),
                    msg,
                    self._lookup_map_tf(msg) if self.leading_mode else None,
                )
            self._log_perf(now)
            return

        inference_dt = (
            dt
            if self.last_inference_time is None
            else max(1e-3, now - self.last_inference_time)
        )
        self.last_inference_time = now

        inference_start = time.perf_counter()
        dets_xy, dets_cls, _ = self._run_detector(scan)
        self.inference_time_sum += time.perf_counter() - inference_start
        self.inference_count += 1

        dets_cls = np.asarray(dets_cls, dtype=np.float32)
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = np.asarray(dets_xy, dtype=np.float64)[conf_mask]
        dets_xy = -1 * dets_xy

        # --- OPTIMIZATION: Fetch TF ONCE per frame ---
        tf_map_to_sensor = None
        if self.extended_map is not None or self.leading_mode:
            tf_map_to_sensor = self._lookup_map_tf(msg)

        # Apply the static map filter using the fetched TF
        dets_xy = self._filter_detections_by_map(dets_xy, tf_map_to_sensor)

        # Filter the raw network detections through the Kalman tracker
        tracked_xy = self.tracker.update(dets_xy, inference_dt)

        self._publish_tracks(tracked_xy, msg, tf_map_to_sensor)
        self._log_perf(now)

    def _should_skip_inference(self, now, scan):
        """Decide whether this scan can be served without touching the GPU.

        Two independent gates. The rate cap is the important one: people do not
        move far in 200 ms, so running DR-SPAAM at a fraction of the LiDAR rate
        cuts GPU time proportionally while the Kalman tracker covers the gaps.
        The idle gate additionally skips scans with nothing in range at all,
        where there is provably nothing to detect.
        """
        if (
            self.min_inference_period > 0.0
            and self.last_inference_time is not None
            and now - self.last_inference_time < self.min_inference_period
        ):
            return True

        if self.idle_skip_range > 0.0 and not np.any(scan < self.idle_skip_range):
            return True

        return False

    def _lookup_map_tf(self, msg):
        try:
            return self.tf_buffer.lookup_transform(
                "map", msg.header.frame_id, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF error: {e}", throttle_duration_sec=2.0)
            return None

    def _publish_tracks(self, tracked_xy, msg, tf_map_to_sensor=None):
        dets_msg = self._dets_to_pose_array(tracked_xy)
        dets_msg.header = msg.header
        self.dets_pub.publish(dets_msg)

        # Use the already fetched TF to avoid a second lookup
        if (
            self.leading_mode
            and len(dets_msg.poses) > 0
            and tf_map_to_sensor is not None
        ):
            self.pose_out = self._parse_subject_pose(dets_msg, tf_map_to_sensor)

        if self.leading_mode:
            if self.pose_out is not None:
                self.last_pose_out = self.pose_out
                self.subject_pub.publish(self.pose_out)
            elif self.last_pose_out is not None:
                self.subject_pub.publish(self.last_pose_out)

        # Avoid generating heavy marker logic if we aren't publishing it
        if self.publish_rviz:
            marker_msg = self._dets_to_marker(tracked_xy)
            marker_msg.header = msg.header
            self.rviz_pub.publish(marker_msg)

    def _log_perf(self, now):
        """Report the measured inference rate, so the GPU saving is verifiable."""
        if self.perf_log_period <= 0.0:
            return
        if self.last_perf_log_time is None:
            self.last_perf_log_time = now
            return

        elapsed = now - self.last_perf_log_time
        if elapsed < self.perf_log_period:
            return

        total = self.inference_count + self.skipped_count
        mean_ms = (
            1e3 * self.inference_time_sum / self.inference_count
            if self.inference_count
            else 0.0
        )
        self.get_logger().info(
            f"DR-SPAAM: {self.inference_count / elapsed:.1f} inferences/s "
            f"({self.inference_count}/{total} scans, {mean_ms:.1f} ms mean, "
            f"~{100.0 * self.inference_time_sum / elapsed:.0f}% busy)"
        )

        self.last_perf_log_time = now
        self.inference_count = 0
        self.skipped_count = 0
        self.inference_time_sum = 0.0

    def _parse_subject_pose(self, dets_msg, transform):
        """Calculates pose using the transform passed down from scan_callback."""
        ps_msg = Pose()
        ps_msg.position.x = dets_msg.poses[0].position.x
        ps_msg.position.y = dets_msg.poses[0].position.y
        ps_msg.position.z = 0.0

        pose_out = PoseStamped()
        pose_out.header.stamp = self.get_clock().now().to_msg()
        pose_out.header.frame_id = "map"
        pose_out.pose = tf2_geometry_msgs.do_transform_pose(ps_msg, transform)
        return pose_out

    def _dets_to_pose_array(self, dets_xy):
        msg = PoseArray()
        for xy in dets_xy:
            p = Pose()
            p.position.x = xy[1]
            p.position.y = xy[0]
            p.position.z = 0.0
            msg.poses.append(p)
        return msg

    def _dets_to_marker(self, dets_xy):
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = "dr_spaam"
        msg.id = 0
        msg.type = Marker.LINE_LIST
        msg.scale.x = 0.03
        msg.color.r = 1.0
        msg.color.a = 1.0

        r = 0.2
        ang = np.linspace(0, 2 * np.pi, 20)
        xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

        for d_xy in dets_xy:
            for i in range(len(xy_offsets) - 1):
                p0 = Point()
                p1 = Point()

                p0.x = d_xy[1] + xy_offsets[i][0]
                p0.y = d_xy[0] + xy_offsets[i][1]

                p1.x = d_xy[1] + xy_offsets[i + 1][0]
                p1.y = d_xy[0] + xy_offsets[i + 1][1]

                msg.points.append(p0)
                msg.points.append(p1)

        return msg


def preprocess_lidar(scan, target_len=240, max_range=10.0):
    scan = np.array(scan, dtype=float)
    invalid = (scan <= 0.01) | np.isinf(scan) | np.isnan(scan)
    scan[invalid] = max_range
    scan = median_filter(scan, size=3)

    if len(scan) != target_len:
        # OPTIMIZATION: Replaced slow interp1d with native fast indexing
        indices = np.round(np.linspace(0, len(scan) - 1, target_len)).astype(int)
        scan = scan[indices]

    return scan


def main(args=None):
    rclpy.init(args=args)
    node = DrSpaamNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
