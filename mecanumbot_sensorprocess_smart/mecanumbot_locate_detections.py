import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, ReliabilityPolicy
from mecanumbot_msgs.msg import CamPersonDetectionArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (
    PoseArray,
    Pose,
    PoseStamped,
    Point,
    PoseWithCovarianceStamped,
    Quaternion,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
from transforms3d.euler import euler2quat, quat2euler
from rclpy.duration import Duration
import math
import threading
import numpy as np
import copy

from mecanumbot_sensorprocess_smart.person_tracking import (
    PersonTracker,
    TrackerConfig,
    combine_measurements,
)


class PersonLocateNode(Node):
    def __init__(self):
        super().__init__("mecanumbot_locate_detections")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.namespace = self.get_namespace().strip("/")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_cb_group = MutuallyExclusiveCallbackGroup()
        # Declare parameter for the "X" meter offset behind the obstacle
        self.declare_parameter("obstacle_buffer_x", 0.5)
        self.declare_parameter("debug_mode", False)
        # A camera detection older than this is not evidence about now. Without
        # the check the last CamPersonDetectionArray was kept for ever and its
        # bearings were re-used against fresh LiDAR scans, so a person who had
        # left was still being placed wherever the scan happened to hit inside
        # a bearing wedge measured minutes earlier.
        self.declare_parameter("cam_detection_timeout", 0.6)
        # --- map-frame tracking (see person_tracking.py) ---
        # The camera gate rejects real people -- a long skirt gives the pose
        # network no skeleton to find -- and every rejection used to empty
        # `people_fusion`, which the behaviour layer reads as the person being
        # gone. The tracker carries the estimate across those gaps instead.
        self.declare_parameter("tracking.enabled", True)
        self.declare_parameter("tracking.publish_rate", 10.0)
        self.declare_parameter("tracking.max_association_distance", 0.9)
        self.declare_parameter("tracking.min_hits", 2)
        self.declare_parameter("tracking.max_coast_time", 1.2)
        self.declare_parameter("tracking.max_uncorroborated_time", 4.0)
        self.declare_parameter("tracking.measurement_noise", 0.25)
        self.declare_parameter("tracking.process_noise", 0.08)
        self.declare_parameter("tracking.max_reported_speed", 2.5)

        # Publishers
        self.people_pub = self.create_publisher(PoseArray, "people_fusion", 10)
        self.debug_mode = self.get_parameter("debug_mode").value

        # Subscribers
        self.cam_people_sub = self.create_subscription(
            CamPersonDetectionArray,
            "cam_people_detections",
            self.cam_people_callback,
            10,
            callback_group=self.camera_cb_group,
        )
        self.laser_people_sub = self.create_subscription(
            PoseArray, "dets", self.lidar_people_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "scan", self.scan_callback, qos
        )

        # Map sub uses Transient Local QoS because maps are usually published once
        map_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, map_qos
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.amcl_callback, 10
        )

        # State variables
        self.laser_detections = []
        self.laser_angles = []
        self.cam_detections = []
        self.scan_data = None
        self.map_data = None
        self.map_array = None
        self.amcl_pose = None
        self.last_published_time = None
        self.cam_stamp = None
        self.cam_received_time = None
        self.fused_poses = PoseArray()
        # merge_detections() runs on the LiDAR callback and publish_tracks() on
        # a timer; under the multi-threaded executor those are different
        # threads, so the handover between them is the one place that locks.
        self._measurement_lock = threading.Lock()
        self._pending_measurements = None

        self.tracking_enabled = bool(self.get_parameter("tracking.enabled").value)
        self.cam_detection_timeout = float(
            self.get_parameter("cam_detection_timeout").value
        )
        self.tracker = PersonTracker(self._build_tracker_config())
        if self.tracking_enabled:
            # `people_fusion` is published on this timer rather than on arrival
            # of a detection, so it keeps flowing -- with a fresh stamp and a
            # coasted position -- through a run of camera rejections. Publishing
            # only on arrival is what made a dropped frame indistinguishable
            # from an empty room.
            rate = float(self.get_parameter("tracking.publish_rate").value)
            self.publish_timer = self.create_timer(
                1.0 / max(rate, 1.0), self.publish_tracks
            )

        if self.debug_mode:
            self.people_left_FOV = PoseArray()
            self.people_left_FOV.header.frame_id = (
                "map" if self.get_namespace().strip("/") else "map"
            )
            self.people_right_FOV = PoseArray()
            self.people_right_FOV.header.frame_id = (
                "map" if self.get_namespace().strip("/") else "map"
            )
            self.people_left_FOV_pub = self.create_publisher(
                PoseArray, "cam_people_detections/left_FOV", 10
            )
            self.people_right_FOV_pub = self.create_publisher(
                PoseArray, "cam_people_detections/right_FOV", 10
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.trans = None

        self.get_logger().info("Person Locate Node has started.")

    def cam_people_callback(self, msg):
        self.cam_stamp = msg.header.stamp
        self.cam_detections = msg.people
        self.cam_received_time = self.get_clock().now()

    def _camera_is_fresh(self):
        """Say whether the last camera detections are recent enough to use.

        They are kept between messages on purpose - the camera runs slower than
        the LiDAR and a bearing from the previous frame is still the best guess
        for this one - but only for `cam_detection_timeout`. Past that the
        person may simply have left, and re-using the bearing would keep
        placing them wherever the scan happens to hit inside a wedge that no
        longer means anything.
        """
        if not self.cam_detections or self.cam_received_time is None:
            return False
        age = (self.get_clock().now() - self.cam_received_time).nanoseconds / 1e9
        return age <= self.cam_detection_timeout

    def scan_callback(self, msg):
        self.scan_data = msg

    def amcl_callback(self, msg):
        self.amcl_pose = msg.pose.pose

    def map_callback(self, msg):
        self.map_data = msg
        # Convert map 1D array to 2D numpy array for fast spatial lookups
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )

    def fill_bound_angle(self, X_min, X_max):
        if self.amcl_pose is None:
            self.get_logger().warn(
                "AMCL pose is not available yet. Cannot fill FOV bounds."
            )
            return

        t = self.tf_buffer.lookup_transform(
            "map", "mecanumbot/head_link", rclpy.time.Time()
        )
        q_msg = t.transform.rotation
        q_map_to_base = Quaternion(w=q_msg.w, x=q_msg.x, y=q_msg.y, z=q_msg.z)
        # 1. Use deepcopy so we don't accidentally modify the actual amcl_pose
        min_pose = copy.deepcopy(self.amcl_pose)
        max_pose = copy.deepcopy(self.amcl_pose)

        # 5. Convert back to quaternions: returns [w, x, y, z]
        q_min = euler2quat(0, 0, X_min)
        q_min = Quaternion(w=q_min[0], x=q_min[1], y=q_min[2], z=q_min[3])
        q_max = euler2quat(0, 0, X_max)
        q_max = Quaternion(w=q_max[0], x=q_max[1], y=q_max[2], z=q_max[3])

        # 1. Create Scipy Rotation objects directly from Euler angles
        # 'xyz' means extrinsic rotations; 'z' is yaw.
        r_min_base = R.from_euler("xyz", [0, 0, X_min])
        r_max_base = R.from_euler("xyz", [0, 0, X_max])

        # 2. Create the Map-to-Base Rotation object
        # (Assuming you extracted the quaternion [x, y, z, w] from your TF tree)
        r_map_to_base = R.from_quat([q_msg.x, q_msg.y, q_msg.z, q_msg.w])

        # 3. Multiply them (Scipy supports the * operator)
        r_min_map = r_map_to_base * r_min_base
        r_max_map = r_map_to_base * r_max_base

        # 4. Convert back to a raw array [x, y, z, w] and build your final object
        q_final_array = r_min_map.as_quat()
        q_min_map = Quaternion(
            x=q_final_array[0],
            y=q_final_array[1],
            z=q_final_array[2],
            w=q_final_array[3],
        )
        # 3. Multiply them (Scipy supports the * operator)
        r_max_map = r_max_base * r_map_to_base

        # 4. Convert back to a raw array [x, y, z, w] and build your final object
        q_final_array = r_max_map.as_quat()
        q_max_map = Quaternion(
            x=q_final_array[0],
            y=q_final_array[1],
            z=q_final_array[2],
            w=q_final_array[3],
        )

        # 6. Assign the new values back to our copied ROS poses (W is q[0])
        min_pose.orientation.w = q_min_map.w
        min_pose.orientation.x = q_min_map.x
        min_pose.orientation.y = q_min_map.y
        min_pose.orientation.z = q_min_map.z

        max_pose.orientation.w = q_max_map.w
        max_pose.orientation.x = q_max_map.x
        max_pose.orientation.y = q_max_map.y
        max_pose.orientation.z = q_max_map.z

        # 7. Append to your PoseArrays
        self.people_left_FOV.poses.append(min_pose)
        self.people_right_FOV.poses.append(max_pose)

    def lidar_people_callback(self, msg):
        try:
            # Safely fetch transform, defaulting to base_scan if header is missing
            source_frame = (
                msg.header.frame_id if msg.header.frame_id else "mecanumbot/base_scan"
            )
            transform = self.tf_buffer.lookup_transform(
                "mecanumbot/base_link", source_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return

        # Extract translation and yaw
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        self.laser_detections = []
        self.laser_angles = []

        # Fast native math transformation (avoids tf2_geometry_msgs overhead)
        for pose in msg.poses:
            global_x = tx + (pose.position.x * cos_yaw) - (pose.position.y * sin_yaw)
            global_y = ty + (pose.position.x * sin_yaw) + (pose.position.y * cos_yaw)

            new_pose = Pose()
            new_pose.position.x = global_x
            new_pose.position.y = global_y
            new_pose.position.z = 0.0

            self.laser_detections.append(new_pose)
            self.laser_angles.append(math.atan2(global_y, global_x))

        self.merge_detections()

    def arrange_with_scan_dets(self, person):
        # Ensure correct min/max bounds even if wrapped
        ang_min = min(person.bound_angle_min.data, person.bound_angle_max.data)
        ang_max = max(person.bound_angle_min.data, person.bound_angle_max.data)

        # Of the LiDAR people inside the camera's bearing wedge, the one the
        # camera is actually looking at is the one nearest the middle of it.
        # Taking the first in list order made the fused position jump between
        # two people standing side by side from one frame to the next, which
        # the tracker downstream would then have to smooth away.
        centre = (ang_min + ang_max) / 2.0
        best_pose = None
        best_offset = None
        for laser_pose, angle in zip(self.laser_detections, self.laser_angles):
            if ang_min <= angle <= ang_max:
                offset = abs(angle - centre)
                if best_offset is None or offset < best_offset:
                    best_offset = offset
                    best_pose = laser_pose

        return best_pose

    def extrap_from_raw_scan(self, person):
        if self.scan_data is None:
            return None

        ranges = np.array(self.scan_data.ranges)
        ang_min_scan = self.scan_data.angle_min
        ang_inc = self.scan_data.angle_increment

        # Order the person bounding angles correctly
        p_min = min(person.bound_angle_min.data, person.bound_angle_max.data)
        p_max = max(person.bound_angle_min.data, person.bound_angle_max.data)

        # Calculate indices and clamp them to array bounds to prevent IndexError
        idx_min = int((p_min - ang_min_scan) / ang_inc)
        idx_max = int((p_max - ang_min_scan) / ang_inc)

        idx_min = max(0, min(idx_min, len(ranges) - 1))
        idx_max = max(0, min(idx_max, len(ranges) - 1))

        if idx_min >= idx_max:
            return None

        # Extract distances and filter out inf, nan, and out-of-range limits
        slice_ranges = ranges[idx_min : idx_max + 1]
        valid_mask = (
            (slice_ranges > self.scan_data.range_min)
            & (slice_ranges < self.scan_data.range_max)
            & ~np.isinf(slice_ranges)
            & ~np.isnan(slice_ranges)
        )
        valid_ranges = slice_ranges[valid_mask]
        # valid_ranges = np.round(valid_ranges, 1)  # Round to 3 decimal places for stability

        if len(valid_ranges) == 0:
            return None

        # Use median to ignore background laser hits
        dist_median = float(
            np.percentile(valid_ranges, 20)
        )  # float(np.median(valid_ranges))
        center_angle = p_min + (p_max - p_min) / 2.0

        x = dist_median * math.cos(center_angle)
        y = dist_median * math.sin(center_angle)
        # self.get_logger().info(f"Extrapolated detection at local coordinates: ({x:.2f}, {y:.2f})")
        return Pose(position=Point(x=x, y=y, z=0.0))

    def handle_map_occlusion(self, local_pose):
        """Checks if the proposed local point lands in a map obstacle and extrudes it."""
        if self.map_data is None or self.map_array is None or self.amcl_pose is None:
            return local_pose  # Missing data, return standard point safely

        local_x = local_pose.position.x
        local_y = local_pose.position.y

        # 1. Get Robot global pose in map
        rx = self.amcl_pose.position.x
        ry = self.amcl_pose.position.y
        q = self.amcl_pose.orientation
        ryaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        # 2. Transform hit location to global map frame
        hit_x = rx + local_x * math.cos(ryaw) - local_y * math.sin(ryaw)
        hit_y = ry + local_x * math.sin(ryaw) + local_y * math.cos(ryaw)

        # 3. Convert to map grid coordinates
        res = self.map_data.info.resolution
        ox = self.map_data.info.origin.position.x
        oy = self.map_data.info.origin.position.y
        width = self.map_data.info.width
        height = self.map_data.info.height

        gx = int((hit_x - ox) / res)
        gy = int((hit_y - oy) / res)

        if not (0 <= gx < width and 0 <= gy < height):
            self.get_logger().warn(
                "Detection is out of map bounds, skipping occlusion check."
            )
            return local_pose  # Out of map bounds

        if self.map_array[gy, gx] > 50:
            # self.get_logger().info("Wall occlusion detected! Tracing back of wall...")

            # Global ray angle from robot
            ray_yaw = ryaw + math.atan2(local_y, local_x)
            step_size = res / 2.0  # Sub-cell stepping to ensure we don't jump gaps

            curr_dist = math.hypot(local_x, local_y)  # Distance from robot to wall hit
            max_dist = (
                curr_dist + 4.0
            )  # Limit tracing to prevent infinite loops (max 4m thick wall)

            # Trace until free space is found
            while curr_dist < max_dist:
                curr_x_map = rx + curr_dist * math.cos(ray_yaw)
                curr_y_map = ry + curr_dist * math.sin(ray_yaw)

                cgx = int((curr_x_map - ox) / res)
                cgy = int((curr_y_map - oy) / res)

                if not (0 <= cgx < width and 0 <= cgy < height):
                    break  # Ray left the map

                cell_val = self.map_array[cgy, cgx]
                if cell_val < 50 and cell_val != -1:
                    # Found free space behind wall (ignoring unknown space (-1))
                    break

                curr_dist += step_size

            # 5. Apply the +X offset from behind the wall
            x_offset = self.get_parameter("obstacle_buffer_x").value
            final_dist = curr_dist + x_offset

            # 6. Re-calculate returning pose in robot's local base_link frame
            local_angle = math.atan2(local_y, local_x)
            corrected_pose = Pose()
            corrected_pose.position.x = final_dist * math.cos(local_angle)
            corrected_pose.position.y = final_dist * math.sin(local_angle)
            corrected_pose.position.z = 0.0

            self.get_logger().info(
                f"Corrected pose shifted +{x_offset}m behind obstacle to "
                f"({corrected_pose.position.x:.2f}, {corrected_pose.position.y:.2f})"
            )
            return corrected_pose
        # self.get_logger().info("No wall occlusion detected, keeping original pose.")
        return local_pose

    def _build_tracker_config(self):
        """Assemble the map-frame tracker settings from the ROS parameters."""

        def track(name):
            return self.get_parameter(f"tracking.{name}").value

        return TrackerConfig(
            max_association_distance=float(track("max_association_distance")),
            min_hits=int(track("min_hits")),
            max_coast_time=float(track("max_coast_time")),
            max_uncorroborated_time=float(track("max_uncorroborated_time")),
            measurement_noise=float(track("measurement_noise")),
            process_noise=float(track("process_noise")),
            max_reported_speed=float(track("max_reported_speed")),
        )

    def _map_transform(self):
        """Return the latest ``map <- base_link`` transform, or None."""
        try:
            return self.tf_buffer.lookup_transform(
                "map", "mecanumbot/base_link", rclpy.time.Time()
            )
        except Exception as error:
            self.get_logger().warn(
                f"map transform unavailable: {error}", throttle_duration_sec=2.0
            )
            return None

    def _locate_camera_person(self, person):
        """Place one camera detection in the base_link frame, or return None.

        The camera gives a bearing and no range, so the range is looked up in
        whatever the LiDAR has inside that bearing: a DR-SPAAM person first,
        the raw scan second.
        """
        person_pose = self.arrange_with_scan_dets(person)
        if person_pose is None:
            person_pose = self.extrap_from_raw_scan(person)
        if person_pose is None:
            return None
        return self.handle_map_occlusion(person_pose)

    def merge_detections(self):
        """Turn this round's detections into map-frame measurements.

        Called from the LiDAR callback, which is the faster of the two inputs.
        Nothing is published here: with tracking on, the measurements are left
        for :meth:`publish_tracks` to fold in on its timer, so `people_fusion`
        runs at a steady rate whether or not a detection arrived. Publishing
        on arrival is what used to make a rejected camera frame look exactly
        like an empty room.
        """
        transform = self._map_transform()
        if transform is None:
            return
        self.trans = transform

        camera_points = []
        if self._camera_is_fresh():
            for person in self.cam_detections:
                person_pose = self._locate_camera_person(person)
                if person_pose is not None:
                    mapped = do_transform_pose(person_pose, transform)
                    camera_points.append((mapped.position.x, mapped.position.y))
                if self.debug_mode:
                    self.fill_bound_angle(
                        person.bound_angle_min.data, person.bound_angle_max.data
                    )

        if not self.tracking_enabled:
            self._publish_untracked(camera_points)
            self._publish_debug_fov()
            return

        # LiDAR people the camera did not vouch for. These may sustain a track
        # the camera has lost - the whole point, for a person the pose network
        # cannot skeletonise - but never start one, because a leg-sized return
        # on its own is not evidence of a person.
        lidar_points = []
        for pose in self.laser_detections:
            mapped = do_transform_pose(pose, transform)
            lidar_points.append((mapped.position.x, mapped.position.y))

        measurements = combine_measurements(
            camera_points,
            lidar_points,
            self.tracker.config.max_association_distance,
        )
        with self._measurement_lock:
            self._pending_measurements = measurements
        self._publish_debug_fov()

    def publish_tracks(self):
        """Advance the tracker and publish where every confirmed person is.

        The tracker is stepped only from here, on the timer, so it is touched
        by one thread despite the multi-threaded executor, and so it advances
        on a clock rather than on whether a detection happened to arrive.
        """
        with self._measurement_lock:
            measurements = self._pending_measurements
            self._pending_measurements = None

        now = self.get_clock().now()
        tracks = self.tracker.step(measurements or [], now.nanoseconds * 1e-9)
        if not tracks:
            return

        fused = PoseArray()
        fused.header.stamp = now.to_msg()
        fused.header.frame_id = "map"
        for track in tracks:
            x, y = track.position
            fused.poses.append(Pose(position=Point(x=x, y=y, z=0.0)))
        self.fused_poses = fused
        self.people_pub.publish(fused)

    def _publish_untracked(self, camera_points):
        """Publish raw fused points, the way the node behaved before tracking.

        Kept for `tracking.enabled: false`, which is how a run is compared
        against the unfiltered pipeline.
        """
        if not camera_points or self.cam_stamp == self.last_published_time:
            return
        fused = PoseArray()
        fused.header.stamp = self.cam_stamp
        fused.header.frame_id = "map"
        for x, y in camera_points:
            fused.poses.append(Pose(position=Point(x=x, y=y, z=0.0)))
        self.fused_poses = fused
        self.people_pub.publish(fused)
        self.last_published_time = self.cam_stamp

    def _publish_debug_fov(self):
        """Publish the bearing-bound pose arrays the RViz overlay draws."""
        if not self.debug_mode:
            return
        stamp = self.cam_stamp or self.get_clock().now().to_msg()
        if self.people_left_FOV.poses:
            self.people_left_FOV.header.stamp = stamp
            self.people_left_FOV_pub.publish(self.people_left_FOV)
            self.people_left_FOV.poses.clear()
        if self.people_right_FOV.poses:
            self.people_right_FOV.header.stamp = stamp
            self.people_right_FOV_pub.publish(self.people_right_FOV)
            self.people_right_FOV.poses.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PersonLocateNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
