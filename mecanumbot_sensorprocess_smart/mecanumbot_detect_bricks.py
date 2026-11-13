#!/usr/bin/env python3
"""
The object-choice task's brick detector: coloured cubes, found and placed in the map.

Publishes the same two shapes the fetch game's ball goes out in, one brick per
colour, with the colour as the label:

    cam_brick_boxes      vision_msgs/Detection2DArray   pixel boxes, full frame
    brick_detections     vision_msgs/Detection3DArray   map frame, with height

The colour finding and the gates are `brick_detection.py`; the camera model,
the neck model and the per-object tracker are `ball_locating.py`'s, because a
brick on the floor has exactly the ball's problem -- the LiDAR plane passes over
it -- and the same answer: bearing and range both from the camera.

## It owns the camera

The object-choice tree needs two things off one webcam: skeletons, to read where
the person points, and bricks. The pose detector is a DeepStream network and the
bricks are a colour threshold, and the camera can be opened once. So with
`camera_source: device` (the default) this node opens it -- with OpenCV, MJPG,
at the frame size every other node is told about -- and republishes every frame
on `frames_topic`, which the pose detector reads with `camera_source:=topic`.
Colour detection then costs no decode at all. `camera_source: topic` reads the
same topic instead, for when something else owns the device.

That republishing makes this node the only thing in the perception package that
starts a compressed camera stream since `2f7aade`, and only when the
object-choice launch asks for it (`perception.launch.py bricks:=true`).

## Calibrating the colours

The HSV ranges are parameters, and a room's light moves them. `calibrate: true`
logs the median HSV of a box in the middle of the frame every two seconds: hold
each brick there, read its numbers, and set the colour's range around them.
`debug_mode` publishes the frame with every blob drawn, accepted ones in their
colour and refused ones with the check they failed.
"""

import math
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from mecanumbot_msgs.msg import OpenCRState
from mecanumbot_sensorprocess_smart import ball_locating
from mecanumbot_sensorprocess_smart import brick_detection as bd

SOURCE_DEVICE = "device"
SOURCE_TOPIC = "topic"

# BGR, for drawing each colour's accepted boxes on the debug image.
DRAW_COLOURS = {
    "red": (40, 40, 230),
    "yellow": (30, 220, 230),
    "green": (60, 200, 60),
    "blue": (230, 120, 40),
}
REJECT_DRAW = (128, 128, 128)

# A colour named in `colours.names` with no range set gets this one, which
# matches nothing (full saturation AND full value at one hue), and a warning.
MATCH_NOTHING = [0, 0, 255, 255, 255, 255]


class BrickDetectNode(Node):
    def __init__(self):
        super().__init__("mecanumbot_detect_bricks")

        # ---- where frames come from, and whether they go out again ----
        self.declare_parameter("camera_source", SOURCE_DEVICE)
        self.declare_parameter("webcam_device", "/dev/video0")
        self.declare_parameter("camera_fps", 15.0)
        self.declare_parameter("camera_topic", "/camera/image_raw/compressed")
        self.declare_parameter("publish_frames", True)
        self.declare_parameter("frames_topic", "/camera/image_raw/compressed")
        self.declare_parameter("frames_frame_id", "camera_link")
        self.declare_parameter("jpeg_quality", 80)

        # ---- the lens: must agree with every other camera node ----
        self.declare_parameter("camera_params.camera_width", 1280)
        self.declare_parameter("camera_params.camera_height", 720)
        self.declare_parameter("camera_params.camera_fov", math.radians(51.0))
        self.declare_parameter("camera_params.camera_vfov", 0.0)
        # Unlike the fetch detector's balls, nothing has been measured about
        # this one, so it is a parameter: put a brick on the robot's LEFT and
        # brick_detections must place it at positive y in the base frame.
        self.declare_parameter("mirror_image_x", False)

        # ---- how often, and at what size, frames are searched ----
        self.declare_parameter("process_rate", 8.0)
        # Search a frame this many times smaller; boxes still come out in full
        # frame pixels. 2 keeps a brick at 4 m about 10 px wide.
        self.declare_parameter("process_scale", 2.0)
        self.declare_parameter("open_kernel", 3)

        # ---- the brick, and where the camera sits (see ball_locating) ----
        self.declare_parameter("brick.size", bd.BRICK_SIZE)
        # ground_plane, not size: a cube's apparent width changes by sqrt(2)
        # as it turns, a ball's does not.
        self.declare_parameter("brick.range_source", ball_locating.SOURCE_GROUND)
        self.declare_parameter("brick.camera_x", 0.128)
        self.declare_parameter("brick.camera_z", 0.2245)
        self.declare_parameter("brick.camera_pitch_deg", 0.0)
        self.declare_parameter("brick.floor_z", -0.01)
        self.declare_parameter("brick.min_range", 0.15)
        self.declare_parameter("brick.max_range", 5.0)

        # ---- the neck, frame by frame: ball.neck.* in the locate node ----
        self.declare_parameter("neck.enabled", True)
        self.declare_parameter("neck.topic", "opencr_state")
        self.declare_parameter("neck.stale_s", 0.5)
        self.declare_parameter("neck.pivot_x", 0.1063)
        self.declare_parameter("neck.pivot_z", 0.170)
        self.declare_parameter("neck.lever_x", 0.022)
        self.declare_parameter("neck.lever_z", 0.0545)
        self.declare_parameter("neck.level_ticks", 600.0)
        self.declare_parameter("neck.rad_per_tick", 0.005061)
        self.declare_parameter("neck.pitch_at_level_deg", 27.5)

        # ---- what a blob has to be ----
        defaults = bd.BrickGate()
        for field in (
            "min_area",
            "min_aspect",
            "max_aspect",
            "min_fill",
            "min_size_ratio",
            "max_size_ratio",
        ):
            self.declare_parameter(f"gate.{field}", float(getattr(defaults, field)))

        # ---- the colours ----
        self.declare_parameter("colours.names", list(bd.DEFAULT_COLOURS))
        names = [str(name) for name in self.get_parameter("colours.names").value]
        table = {}
        for name in names:
            default = list(bd.DEFAULT_COLOURS.get(name, MATCH_NOTHING))
            self.declare_parameter(f"colours.{name}", default)
            table[name] = list(self.get_parameter(f"colours.{name}").value)
            if name not in bd.DEFAULT_COLOURS and table[name] == MATCH_NOTHING:
                self.get_logger().warn(
                    f"colour '{name}' has no HSV range (colours.{name}); it will "
                    "never be found"
                )
        self.colours = bd.colours_from_table(table)

        # ---- tracking, one tracker per colour ----
        self.declare_parameter("tracking.max_association_distance", 0.5)
        self.declare_parameter("tracking.min_hits", 3)
        # Longer than the ball's 1 s: a brick does not roll away, and the robot's
        # own body hides it in the last few decimetres of an approach.
        self.declare_parameter("tracking.max_coast_time", 1.5)
        self.declare_parameter("tracking.position_gain", 0.4)
        self.declare_parameter("tracking.velocity_gain", 0.0)
        self.declare_parameter("tracking.max_speed", 1.0)

        # ---- outputs ----
        self.declare_parameter("boxes_topic", "cam_brick_boxes")
        self.declare_parameter("detections_topic", "brick_detections")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "mecanumbot/base_link")
        self.declare_parameter("debug_mode", False)
        self.declare_parameter("debug_topic", "cam_brick_detections/debug_image/compressed")
        self.declare_parameter("calibrate", False)

        self._read_parameters()

        self._frame_lock = threading.Lock()
        self._neck_lock = threading.Lock()
        self._frame = None
        self._frame_stamp = None
        self._frame_count = 0
        self._processed_count = -1
        self._warned_size = False
        self._last_calibration = 0.0

        self.boxes_pub = self.create_publisher(Detection2DArray, self.boxes_topic, 10)
        self.detections_pub = self.create_publisher(
            Detection3DArray, self.detections_topic, 10
        )
        self.debug_pub = None
        if self.debug_mode:
            self.debug_pub = self.create_publisher(CompressedImage, self.debug_topic, 2)
        self.frames_pub = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.neck_mount = None
        self.neck_history = None
        if bool(self.get_parameter("neck.enabled").value):

            def neck(name):
                return float(self.get_parameter(f"neck.{name}").value)

            self.neck_mount = ball_locating.NeckMount(
                pivot_x=neck("pivot_x"),
                pivot_z=neck("pivot_z"),
                lever_x=neck("lever_x"),
                lever_z=neck("lever_z"),
                level_ticks=neck("level_ticks"),
                rad_per_tick=neck("rad_per_tick"),
                pitch_at_level=math.radians(neck("pitch_at_level_deg")),
            )
            self.neck_history = ball_locating.NeckHistory(stale_s=neck("stale_s"))
            self.create_subscription(
                OpenCRState,
                str(self.get_parameter("neck.topic").value),
                self._neck_callback,
                10,
            )
            self.get_logger().info(self.neck_mount.describe())

        self.trackers = {
            colour.name: ball_locating.BallTracker(self._tracker_config())
            for colour in self.colours
        }

        self._running = True
        self._capture_thread = None
        if self.camera_source == SOURCE_DEVICE:
            if self.publish_frames:
                self.frames_pub = self.create_publisher(
                    CompressedImage, self.frames_topic, 2
                )
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
        elif self.camera_source == SOURCE_TOPIC:
            self.create_subscription(
                CompressedImage, self.camera_topic, self._frame_callback, 2
            )
        else:
            raise ValueError(
                f"camera_source '{self.camera_source}', expected "
                f"'{SOURCE_DEVICE}' or '{SOURCE_TOPIC}'"
            )

        self.create_timer(
            1.0 / max(self.process_rate, 0.5),
            self._process,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info(
            f"Brick detector: {', '.join(c.name for c in self.colours)} bricks "
            f"{self.geometry.diameter * 100:.1f} cm, frame "
            f"{self.camera.width:.0f}x{self.camera.height:.0f} at "
            f"{math.degrees(self.camera.hfov):.1f} deg, from "
            + (
                f"{self.webcam_device}"
                + (f", republished on {self.frames_topic}" if self.publish_frames else "")
                if self.camera_source == SOURCE_DEVICE
                else f"{self.camera_topic}"
            )
            + f", searched every {1.0 / max(self.process_rate, 0.5):.2f} s at "
            f"1/{self.process_scale:g} size, range from '{self.range_source}'"
            + (", image MIRRORED" if self.mirror else "")
            + "."
        )

    # --- configuration ---------------------------------------------------------

    def _read_parameters(self):
        gp = self.get_parameter
        self.camera_source = str(gp("camera_source").value)
        self.webcam_device = str(gp("webcam_device").value)
        self.camera_fps = float(gp("camera_fps").value)
        self.camera_topic = str(gp("camera_topic").value)
        self.publish_frames = bool(gp("publish_frames").value)
        self.frames_topic = str(gp("frames_topic").value)
        self.frames_frame_id = str(gp("frames_frame_id").value)
        self.jpeg_quality = int(gp("jpeg_quality").value)
        self.camera = ball_locating.CameraModel(
            width=float(gp("camera_params.camera_width").value),
            height=float(gp("camera_params.camera_height").value),
            hfov=float(gp("camera_params.camera_fov").value),
            vfov=float(gp("camera_params.camera_vfov").value),
        )
        self.mirror = bool(gp("mirror_image_x").value)
        self.process_rate = float(gp("process_rate").value)
        self.process_scale = max(1.0, float(gp("process_scale").value))
        self.open_kernel = int(gp("open_kernel").value)
        self.geometry = ball_locating.BallGeometry(
            diameter=float(gp("brick.size").value),
            camera_x=float(gp("brick.camera_x").value),
            camera_z=float(gp("brick.camera_z").value),
            camera_pitch=math.radians(float(gp("brick.camera_pitch_deg").value)),
            floor_z=float(gp("brick.floor_z").value),
            min_range=float(gp("brick.min_range").value),
            max_range=float(gp("brick.max_range").value),
        )
        self.range_source = str(gp("brick.range_source").value)
        self.gate = bd.BrickGate(
            **{
                field: float(gp(f"gate.{field}").value)
                for field in (
                    "min_area",
                    "min_aspect",
                    "max_aspect",
                    "min_fill",
                    "min_size_ratio",
                    "max_size_ratio",
                )
            }
        )
        self.boxes_topic = str(gp("boxes_topic").value)
        self.detections_topic = str(gp("detections_topic").value)
        self.map_frame = str(gp("map_frame").value)
        self.base_frame = str(gp("base_frame").value)
        self.debug_mode = bool(gp("debug_mode").value)
        self.debug_topic = str(gp("debug_topic").value)
        self.calibrate = bool(gp("calibrate").value)

    def _tracker_config(self):
        def track(name):
            return self.get_parameter(f"tracking.{name}").value

        return ball_locating.BallTrackerConfig(
            max_association_distance=float(track("max_association_distance")),
            min_hits=int(track("min_hits")),
            max_coast_time=float(track("max_coast_time")),
            position_gain=float(track("position_gain")),
            velocity_gain=float(track("velocity_gain")),
            max_speed=float(track("max_speed")),
        )

    # --- frames ----------------------------------------------------------------

    def _open_camera(self):
        capture = cv2.VideoCapture(self.webcam_device, cv2.CAP_V4L2)
        if not capture.isOpened():
            return None
        # MJPG, or the USB link cannot carry 1280x720 at any useful rate.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.camera.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.camera.height))
        capture.set(cv2.CAP_PROP_FPS, self.camera_fps)
        # Keep the newest frame, not a queue of old ones.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _capture_loop(self):
        capture = None
        while self._running and rclpy.ok():
            if capture is None:
                capture = self._open_camera()
                if capture is None:
                    self.get_logger().error(
                        f"Cannot open {self.webcam_device}. Is another node holding "
                        "it -- a pose detector with camera_source:=direct, the "
                        "Deep3R client? Retrying.",
                        throttle_duration_sec=5.0,
                    )
                    time.sleep(1.0)
                    continue
            ok, frame = capture.read()
            if not ok or frame is None:
                self.get_logger().warn(
                    f"{self.webcam_device} returned no frame; reopening",
                    throttle_duration_sec=5.0,
                )
                capture.release()
                capture = None
                time.sleep(0.5)
                continue
            stamp = self.get_clock().now()
            self._store_frame(frame, stamp)
            if self.frames_pub is not None:
                self._republish(frame, stamp)
        if capture is not None:
            capture.release()

    def _frame_callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if stamp.nanoseconds == 0:
            stamp = self.get_clock().now()
        self._store_frame(frame, stamp)

    def _store_frame(self, frame, stamp):
        height, width = frame.shape[:2]
        if not self._warned_size and (
            width != int(self.camera.width) or height != int(self.camera.height)
        ):
            self._warned_size = True
            self.get_logger().warn(
                f"The camera delivers {width}x{height}, not the "
                f"{self.camera.width:.0f}x{self.camera.height:.0f} every camera node "
                "is configured for. Bearings and ranges here are scaled to the "
                "configured frame, and the pose detector's appsrc expects it too: "
                "set camera_width/camera_height to a mode the camera has."
            )
        with self._frame_lock:
            self._frame = frame
            self._frame_stamp = stamp
            self._frame_count += 1

    def _republish(self, frame, stamp):
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.frames_frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.frames_pub.publish(msg)

    def _neck_callback(self, msg):
        stamp = msg.header.stamp
        seconds = (
            stamp.sec + stamp.nanosec * 1e-9
            if stamp.sec or stamp.nanosec
            else self.get_clock().now().nanoseconds * 1e-9
        )
        with self._neck_lock:
            self.neck_history.submit(seconds, int(msg.pos_n))

    # --- the search ------------------------------------------------------------

    def _geometry_at(self, stamp_seconds):
        """Return the camera geometry a frame taken at this stamp was seen with."""
        if self.neck_mount is None:
            return self.geometry
        with self._neck_lock:
            ticks = self.neck_history.at(stamp_seconds)
        if ticks is not None:
            placed = self.neck_mount.geometry(self.geometry, ticks)
            if placed is not None:
                return placed
        self.get_logger().warn(
            "Placing bricks with the fixed brick.camera_pitch_deg "
            f"({math.degrees(self.geometry.camera_pitch):+.1f} deg): no usable neck "
            "reading for the frame. Heights and floor checks are only as right as "
            "that pitch.",
            throttle_duration_sec=5.0,
        )
        return self.geometry

    def _process(self):
        with self._frame_lock:
            if self._frame is None or self._frame_count == self._processed_count:
                return
            frame = self._frame
            stamp = self._frame_stamp
            self._processed_count = self._frame_count

        scale = self.process_scale
        if scale > 1.0:
            small = cv2.resize(
                frame,
                (int(frame.shape[1] / scale), int(frame.shape[0] / scale)),
                interpolation=cv2.INTER_AREA,
            )
            scale = frame.shape[1] / float(small.shape[1])
        else:
            small = frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        stamp_seconds = stamp.nanoseconds * 1e-9
        geometry = self._geometry_at(stamp_seconds)
        accepted, rejected = bd.detect_bricks(
            hsv,
            self.colours,
            self.gate,
            self.camera,
            geometry,
            scale=scale,
            mirror=self.mirror,
            open_kernel=self.open_kernel,
            prefer=self.range_source,
        )

        self._publish_boxes(accepted, stamp)
        self._publish_located(accepted)
        if self.debug_pub is not None:
            self._publish_debug(small, scale, accepted, rejected, stamp)
        if self.calibrate:
            self._log_calibration(hsv, scale)

    def _publish_boxes(self, accepted, stamp):
        msg = Detection2DArray()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.frames_frame_id
        for candidate in accepted:
            blob = candidate.blob
            detection = Detection2D()
            detection.header = msg.header
            detection.id = blob.colour
            detection.bbox.center.position.x = blob.u
            detection.bbox.center.position.y = blob.v
            detection.bbox.size_x = blob.width
            detection.bbox.size_y = blob.height
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = blob.colour
            hypothesis.hypothesis.score = float(candidate.score)
            detection.results.append(hypothesis)
            msg.detections.append(detection)
        # Published even when empty: an empty frame is news to a tree that is
        # centring on a brick -- it has gone under the lens.
        self.boxes_pub.publish(msg)

    def _map_transform(self):
        try:
            return self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception as error:
            self.get_logger().warn(
                f"{self.map_frame} <- {self.base_frame} unavailable: {error}",
                throttle_duration_sec=2.0,
            )
            return None

    def _publish_located(self, accepted):
        transform = self._map_transform()
        if transform is None:
            return
        by_colour = {}
        for candidate in accepted:
            observation = candidate.observation
            local = Pose(position=Point(x=observation.x, y=observation.y, z=observation.z))
            mapped = do_transform_pose(local, transform)
            by_colour[candidate.blob.colour] = ball_locating.BallMeasurement(
                x=mapped.position.x,
                y=mapped.position.y,
                z=mapped.position.z,
                score=candidate.score,
            )

        now = self.get_clock().now()
        msg = Detection3DArray()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self.map_frame
        for name, tracker in self.trackers.items():
            measurement = by_colour.get(name)
            tracks = tracker.step(
                [] if measurement is None else [measurement], now.nanoseconds * 1e-9
            )
            if not tracks:
                continue
            # One brick per colour: the most-seen track, should a bad range ever
            # have started a second.
            track = max(tracks, key=lambda t: t.hits)
            x, y, z = track.position
            detection = Detection3D()
            detection.header = msg.header
            detection.id = name
            detection.bbox.center.position.x = x
            detection.bbox.center.position.y = y
            detection.bbox.center.position.z = z
            detection.bbox.center.orientation.w = 1.0
            size = self.geometry.diameter
            detection.bbox.size.x = size
            detection.bbox.size.y = size
            detection.bbox.size.z = size
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = name
            hypothesis.hypothesis.score = float(track.score)
            hypothesis.pose.pose.position.x = x
            hypothesis.pose.pose.position.y = y
            hypothesis.pose.pose.position.z = z
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            msg.detections.append(detection)
        if msg.detections:
            self.detections_pub.publish(msg)

    # --- seeing what it saw ----------------------------------------------------

    def _publish_debug(self, small, scale, accepted, rejected, stamp):
        image = small.copy()

        def corners(blob):
            u = blob.u
            if self.mirror:
                # Drawn on the unmirrored frame, so reflect back.
                u = self.camera.width - u
            left = int((u - blob.width / 2.0) / scale)
            top = int((blob.v - blob.height / 2.0) / scale)
            right = int((u + blob.width / 2.0) / scale)
            bottom = int((blob.v + blob.height / 2.0) / scale)
            return (left, top), (right, bottom)

        for blob, reason in rejected:
            first, second = corners(blob)
            cv2.rectangle(image, first, second, REJECT_DRAW, 1)
            cv2.putText(
                image, f"{blob.colour}: {reason}", (first[0], max(10, first[1] - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, REJECT_DRAW, 1,
            )
        for candidate in accepted:
            blob = candidate.blob
            colour = DRAW_COLOURS.get(blob.colour, (255, 255, 255))
            first, second = corners(blob)
            cv2.rectangle(image, first, second, colour, 2)
            cv2.putText(
                image,
                f"{blob.colour} {candidate.observation.range:.2f} m "
                f"z={candidate.observation.z * 100:.0f} cm",
                (first[0], max(12, first[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
            )
        if self.calibrate:
            box = self._calibration_box(scale)
            first = (int((box[0] - box[2] / 2) / scale), int((box[1] - box[3] / 2) / scale))
            second = (int((box[0] + box[2] / 2) / scale), int((box[1] + box[3] / 2) / scale))
            cv2.rectangle(image, first, second, (255, 255, 255), 1)
        cv2.putText(
            image, f"{len(accepted)} brick(s)", (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.frames_frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.debug_pub.publish(msg)

    def _calibration_box(self, scale):
        side = 0.05 * self.camera.width
        return (self.camera.width / 2.0, self.camera.height / 2.0, side, side)

    def _log_calibration(self, hsv, scale):
        now = time.monotonic()
        if now - self._last_calibration < 2.0:
            return
        self._last_calibration = now
        box = self._calibration_box(scale)
        small_box = tuple(value / scale for value in box)
        median = bd.mean_hsv(hsv, small_box)
        if median is None:
            return
        self.get_logger().info(
            f"calibrate: median HSV in the centre box is h={median[0]} "
            f"s={median[1]} v={median[2]} (OpenCV scale: h 0..179)"
        )

    def destroy_node(self):
        self._running = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BrickDetectNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
