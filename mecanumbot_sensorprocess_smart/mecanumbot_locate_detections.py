"""
Placing what the cameras see, in the map frame.

Two things are located here, and they are located differently because the
sensors can say different things about them.

**People** are a bearing plus a range from two sensors: the camera says which
direction, the LiDAR says how far, and a Kalman estimate per person carries the
result across the frames where one of the two is missing. That is the original
job of this node and `person_tracking.py` explains it.

The LiDAR's half of that is done on **clusters** -- the surfaces the scan saw,
`scan_clusters.py` -- rather than on the raw beams inside a bearing wedge. A
camera bearing is ranged by the cluster it points at; the clusters the size of
a person that the map does not explain keep a person tracked where the camera
cannot follow them; and those same clusters go out on `lidar_candidates`, which
is what the trees point the head at. The fusion runs on every scan, so it no
longer stops when the DR-SPAAM node does.

**Balls** are the fetch game's target, and the LiDAR cannot help at all: it scans
one horizontal plane about 0.12 m up and a tennis ball on the floor is never in
it, so a range looked up inside the ball's bearing would be the range of the
wall behind it. Both the bearing and the range therefore come from the camera --
from the ball's apparent size, or from where its ray meets the floor. That
geometry is `ball_locating.py`; what is here is the ROS end of it.

The two paths share this node rather than getting one each because they share
the transform, the clock and the camera geometry, and because a ball and a
person are the same picture. They do not share a tracker: a crowd of people and
a rolling ball are different association problems.
"""

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, ReliabilityPolicy
from mecanumbot_msgs.msg import CamPersonDetectionArray, OpenCRState
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
from vision_msgs.msg import (
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import OccupancyGrid
from tf2_ros import TransformListener, Buffer
from transforms3d.euler import euler2quat, quat2euler
from rclpy.duration import Duration
import math
import threading
import numpy as np
import copy

from mecanumbot_sensorprocess_smart import ball_locating
from mecanumbot_sensorprocess_smart import scan_clusters
from mecanumbot_sensorprocess_smart.scan_wedge import wedge_ranges
from mecanumbot_sensorprocess_smart.person_gating import TYPE_CLOSE_RANGE
from mecanumbot_sensorprocess_smart.person_tracking import (
    CameraCoverage,
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
        # The same question of the LiDAR people detector's last message. The
        # fusion used to run *on* that message, so it was never stale; it now
        # runs on the scan, and a DR-SPAAM node that has died must not leave
        # its last detections standing in the room.
        self.declare_parameter("dets_timeout", 0.5)
        # ---- where the camera points, in the LiDAR's terms ----
        # MEASURED 2026-10-02 from the bags of 2026-09-23 .. 09-30: with the
        # robot standing still, the legs the LiDAR sees sit 7.5 deg clockwise
        # of the bearing the camera gives the same person (266 detections, five
        # bags, per-bag medians -5.8 .. -10.2). Added to every camera bearing
        # before it is looked up in the scan. Which of the two sensors is the
        # one mounted askew does not matter here -- positions are taken in the
        # LiDAR's frame, which is the one AMCL localises -- but it is worth
        # finding out on the robot, and re-measuring if either is remounted.
        self.declare_parameter("camera_bearing_offset_deg", -7.5)
        # Degrees a bearing wedge is widened by on each side when it is matched
        # against the scan. The offset above is a median with a spread of about
        # 2.5 deg, and the camera frame is a few hundred ms older than the scan,
        # which on a robot turning at 0.3 rad/s is several degrees more.
        self.declare_parameter("wedge_margin_deg", 4.0)
        # A camera detection labelled close_range (its box runs off the top of
        # the frame) is only used if a LiDAR candidate within `max_range` lies
        # in its wedge; see scan_clusters.corroborates_close_range.
        self.declare_parameter("close_range.require_lidar", True)
        self.declare_parameter("close_range.max_range", 3.0)
        # ---- scan clusters (see scan_clusters.py) ----
        self.declare_parameter("clusters.enabled", True)
        # Metres two neighbouring returns may differ by and still be one surface.
        self.declare_parameter("clusters.jump", 0.2)
        self.declare_parameter("clusters.max_gap", 1)
        # What makes a cluster a candidate: person-sized, in range, and not
        # already explained by the static map grown by `map_inflation`.
        self.declare_parameter("clusters.min_width", 0.10)
        self.declare_parameter("clusters.max_width", 1.2)
        self.declare_parameter("clusters.max_range", 4.5)
        self.declare_parameter("clusters.map_inflation", 0.2)
        self.declare_parameter("clusters.max_explained", 0.5)
        # Whether candidates may keep an existing track alive, and how near its
        # prediction one has to be. They never start one.
        self.declare_parameter("clusters.sustain_tracks", True)
        self.declare_parameter("clusters.sustain_gate", 0.45)
        self.declare_parameter("clusters.candidates_topic", "lidar_candidates")
        # ---- a detection ranged onto mapped structure ----
        # `obstacle_buffer_x` above is how far behind an occluder the person is
        # put. These two decide whether there *is* a behind: see
        # scan_clusters.place_on_structure.
        self.declare_parameter("occlusion.front_standoff", 0.3)
        self.declare_parameter("occlusion.max_depth", 0.6)
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
        self.declare_parameter("tracking.coast_velocity_time", 0.4)
        # --- the camera's blind zone ---
        # Corroboration is only meaningful where the camera could have supplied
        # it. Close to the robot the vertical field of view has run out and off
        # to the side there is no picture at all, so in both places the camera's
        # silence is absence of evidence, not evidence of absence, and a track
        # there holds its corroboration clock instead of spending it.
        self.declare_parameter("tracking.camera_frame", "mecanumbot/head_link")
        self.declare_parameter("tracking.exempt_close_range", True)
        self.declare_parameter("tracking.exempt_outside_fov", True)
        self.declare_parameter("tracking.camera_blind_range", 0.9)
        self.declare_parameter("tracking.camera_half_fov_deg", 30.0)
        self.declare_parameter("tracking.max_blind_zone_time", 30.0)
        self.declare_parameter("tracking.blind_zone_min_hits", 5)
        self.declare_parameter("tracking.blind_zone_creates_tracks", True)

        # ---- the camera, as this node has to model it ----
        # The pose detector publishes bearings, having done the image -> angle
        # conversion itself; the fetch detector publishes boxes in pixels and
        # leaves the conversion here, which is where the range has to be worked
        # out anyway. So this node needs the lens. These three MUST match the
        # detector's own `camera_params`: they describe the same camera, and
        # nothing checks that at startup because the two nodes never talk. A
        # box arriving outside the declared frame is warned about once, which is
        # the cheapest symptom of their having drifted apart.
        self.declare_parameter("camera_params.camera_width", 1280)
        self.declare_parameter("camera_params.camera_height", 720)
        self.declare_parameter("camera_params.camera_fov", math.radians(51.0))
        # 0.0 derives the vertical field of view from the frame shape, assuming
        # square pixels.
        self.declare_parameter("camera_params.camera_vfov", 0.0)

        # ---- the fetch game's ball (see ball_locating.py) ----
        self.declare_parameter("ball.enabled", True)
        self.declare_parameter("ball.boxes_topic", "cam_ball_boxes")
        # The label the located ball is published under. It has to be the
        # detector's own name for the class, because that is what a consumer
        # matches on -- the numeric class id belongs to whichever model was
        # loaded and means nothing once the detection has left the camera.
        self.declare_parameter("ball.class_id", "sports ball")
        self.declare_parameter("ball.detection_timeout", 0.6)
        self.declare_parameter("ball.publish_rate", 10.0)
        # A regulation tennis ball. Change it for a different ball and every
        # apparent-size range changes with it, proportionally.
        self.declare_parameter("ball.diameter", ball_locating.TENNIS_BALL_DIAMETER)
        # Which estimator supplies the published range: `size` (works with no
        # mounting measurements at all) or `ground_plane` (better close in, once
        # camera_z and camera_pitch_deg have actually been measured).
        self.declare_parameter("ball.range_source", ball_locating.SOURCE_SIZE)
        # Where the camera sits in the robot's base frame [m], and where the
        # floor is in it. Parameters, not the URDF: its camera translations
        # were never measured. Measure them on the robot.
        self.declare_parameter("ball.camera_x", 0.128)
        self.declare_parameter("ball.camera_z", 0.2245)
        self.declare_parameter("ball.camera_pitch_deg", 0.0)
        self.declare_parameter("ball.floor_z", -0.01)
        self.declare_parameter("ball.min_range", 0.15)
        self.declare_parameter("ball.max_range", 6.0)
        # Metres the two range estimators may differ by before it is worth
        # saying so. A persistent disagreement is what a wrong camera height or
        # tilt looks like, and it is otherwise invisible: each estimator on its
        # own produces a perfectly plausible number.
        self.declare_parameter("ball.disagreement_warn", 0.6)
        # ---- which way the neck has the camera pointing (see NeckMount) ----
        # The fetch tree sweeps the head, so one fixed camera_pitch_deg is wrong
        # for almost every frame, and so is the ball height worked out from it:
        # a ball on the floor seen with the head down and placed as if the
        # camera looked level comes out as high as the camera. With this on,
        # each frame is placed with the neck's position at the frame's stamp,
        # and camera_x / camera_z / camera_pitch_deg are only the fallback.
        self.declare_parameter("ball.neck.enabled", True)
        self.declare_parameter("ball.neck.topic", "opencr_state")
        self.declare_parameter("ball.neck.stale_s", 0.5)
        # The same model and numbers as mecanumbot_deep3r's `camera.*`. They
        # describe the same servo, so change both or neither.
        self.declare_parameter("ball.neck.pivot_x", 0.1063)
        self.declare_parameter("ball.neck.pivot_z", 0.170)
        self.declare_parameter("ball.neck.lever_x", 0.022)
        self.declare_parameter("ball.neck.lever_z", 0.0545)
        self.declare_parameter("ball.neck.level_ticks", 600.0)
        self.declare_parameter("ball.neck.rad_per_tick", 0.005061)
        # The lens tilt at level_ticks, positive up: level at ~505 ticks
        # (measured 2026-10-21), so 27.5 deg at 600. See the config file.
        self.declare_parameter("ball.neck.pitch_at_level_deg", 27.5)
        self.declare_parameter("ball.tracking.max_association_distance", 0.6)
        self.declare_parameter("ball.tracking.min_hits", 2)
        self.declare_parameter("ball.tracking.max_coast_time", 1.0)
        self.declare_parameter("ball.tracking.position_gain", 0.5)
        self.declare_parameter("ball.tracking.velocity_gain", 0.12)
        self.declare_parameter("ball.tracking.max_speed", 3.0)

        # ---- person boxes from the fetch detector ----
        # The fetch detector replaces the pose detector rather than joining it
        # (two networks on one Orin Nano is most of the GPU), so its person
        # boxes have to be able to feed the same fusion. They arrive as pixels
        # and become the same bearing wedge a CamPersonDetection carries, which
        # is all this node ever used of one. Harmless with both detectors
        # running: the two wedges land on the same person and the tracker's
        # association absorbs the duplicate.
        self.declare_parameter("person_boxes.enabled", True)
        self.declare_parameter("person_boxes.topic", "cam_people_boxes")

        # Publishers
        self.people_pub = self.create_publisher(PoseArray, "people_fusion", 10)
        self.debug_mode = self.get_parameter("debug_mode").value
        # What the LiDAR sees that might be a person: map-frame points, one per
        # scan, EMPTY when there are none -- so a consumer can tell "nothing in
        # the room" from "nobody is publishing", which `people_fusion` cannot.
        self.candidates_pub = self.create_publisher(
            PoseArray, str(self.get_parameter("clusters.candidates_topic").value), 10
        )

        # The ball goes out twice, for two kinds of consumer. `ball_fusion`
        # mirrors `people_fusion` -- a PoseArray in the map frame, which is what
        # rviz draws and what a behaviour that only wants a place to drive to
        # needs. `ball_detections` is the labelled form, the same
        # vision_msgs/Detection3DArray the Deep3R seeking system passes object
        # hypotheses around in, carrying the score and the ball's diameter as
        # the bbox size. The fetch tree reads the second, because it wants a
        # confidence to gate on.
        self.ball_enabled = bool(self.get_parameter("ball.enabled").value)
        self.ball_pub = self.create_publisher(PoseArray, "ball_fusion", 10)
        self.ball_detections_pub = self.create_publisher(
            Detection3DArray, "ball_detections", 10
        )

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
        if self.ball_enabled:
            self.ball_boxes_sub = self.create_subscription(
                Detection2DArray,
                str(self.get_parameter("ball.boxes_topic").value),
                self.ball_boxes_callback,
                10,
                callback_group=self.camera_cb_group,
            )
        if bool(self.get_parameter("person_boxes.enabled").value):
            self.person_boxes_sub = self.create_subscription(
                Detection2DArray,
                str(self.get_parameter("person_boxes.topic").value),
                self.person_boxes_callback,
                10,
                callback_group=self.camera_cb_group,
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
        self.laser_received_time = None
        self.cam_detections = []
        self.scan_data = None
        self.map_data = None
        self.map_array = None
        self.static_map = None

        self.dets_timeout = float(self.get_parameter("dets_timeout").value)
        self.camera_bearing_offset = math.radians(
            float(self.get_parameter("camera_bearing_offset_deg").value)
        )
        self.wedge_margin = math.radians(
            float(self.get_parameter("wedge_margin_deg").value)
        )
        self.close_range_require_lidar = bool(
            self.get_parameter("close_range.require_lidar").value
        )
        self.close_range_max_range = float(
            self.get_parameter("close_range.max_range").value
        )
        self.clusters_enabled = bool(self.get_parameter("clusters.enabled").value)
        self.cluster_jump = float(self.get_parameter("clusters.jump").value)
        self.cluster_max_gap = int(self.get_parameter("clusters.max_gap").value)
        self.map_inflation = float(self.get_parameter("clusters.map_inflation").value)
        self.candidate_config = scan_clusters.CandidateConfig(
            min_width=float(self.get_parameter("clusters.min_width").value),
            max_width=float(self.get_parameter("clusters.max_width").value),
            max_explained=float(self.get_parameter("clusters.max_explained").value),
            max_range=float(self.get_parameter("clusters.max_range").value),
        )
        self.clusters_sustain = bool(
            self.get_parameter("clusters.sustain_tracks").value
        )
        self.cluster_gate = float(self.get_parameter("clusters.sustain_gate").value)
        self.occlusion_config = scan_clusters.OcclusionConfig(
            behind_buffer=float(self.get_parameter("obstacle_buffer_x").value),
            front_standoff=float(self.get_parameter("occlusion.front_standoff").value),
            max_occluder_depth=float(self.get_parameter("occlusion.max_depth").value),
        )
        self.get_logger().info(
            "People are ranged on scan clusters: camera bearings shifted "
            f"{math.degrees(self.camera_bearing_offset):+.1f} deg into the LiDAR's "
            f"frame, wedges widened by {math.degrees(self.wedge_margin):.1f} deg; "
            f"candidates are {self.candidate_config.min_width:.2f}.."
            f"{self.candidate_config.max_width:.2f} m wide within "
            f"{self.candidate_config.max_range:.1f} m and "
            f"{'sustain' if self.clusters_sustain else 'do not sustain'} tracks."
        )
        self.amcl_pose = None
        self.last_published_time = None
        self.cam_stamp = None
        self.cam_received_time = None
        self.fused_poses = PoseArray()
        # merge_detections() runs on the scan callback and publish_tracks() on
        # a timer; under the multi-threaded executor those are different
        # threads, so the handover between them is the one place that locks.
        self._measurement_lock = threading.Lock()
        self._pending_measurements = None

        # --- the ball path ---
        self.ball_boxes = []
        self.ball_received_time = None
        self.person_boxes = []
        self.person_boxes_time = None
        self._ball_lock = threading.Lock()
        self._warned_box_bounds = False
        self._warned_range_disagreement = False
        self.camera = ball_locating.CameraModel(
            width=float(self.get_parameter("camera_params.camera_width").value),
            height=float(self.get_parameter("camera_params.camera_height").value),
            hfov=float(self.get_parameter("camera_params.camera_fov").value),
            vfov=float(self.get_parameter("camera_params.camera_vfov").value),
        )
        self.ball_geometry = ball_locating.BallGeometry(
            diameter=float(self.get_parameter("ball.diameter").value),
            camera_x=float(self.get_parameter("ball.camera_x").value),
            camera_z=float(self.get_parameter("ball.camera_z").value),
            camera_pitch=math.radians(
                float(self.get_parameter("ball.camera_pitch_deg").value)
            ),
            floor_z=float(self.get_parameter("ball.floor_z").value),
            min_range=float(self.get_parameter("ball.min_range").value),
            max_range=float(self.get_parameter("ball.max_range").value),
        )
        self.ball_range_source = str(self.get_parameter("ball.range_source").value)
        self.ball_class_id = str(self.get_parameter("ball.class_id").value)
        self.ball_detection_timeout = float(
            self.get_parameter("ball.detection_timeout").value
        )
        self.ball_disagreement_warn = float(
            self.get_parameter("ball.disagreement_warn").value
        )
        self.ball_tracker = ball_locating.BallTracker(self._build_ball_config())
        self.ball_boxes_stamp = None

        # Where the neck has the camera pointing. Without this every frame is
        # placed with the one fixed pitch above, which the fetch tree's head
        # sweep makes wrong for almost every frame.
        self.neck_mount = None
        self.neck_history = None
        if self.ball_enabled and bool(self.get_parameter("ball.neck.enabled").value):

            def neck(name):
                return float(self.get_parameter(f"ball.neck.{name}").value)

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
            self.neck_sub = self.create_subscription(
                OpenCRState,
                str(self.get_parameter("ball.neck.topic").value),
                self.neck_callback,
                10,
            )
            # Logged, as the fixed mounting is: a run should say what it assumed.
            self.get_logger().info(self.neck_mount.describe())

        self.tracking_enabled = bool(self.get_parameter("tracking.enabled").value)
        self.camera_frame = str(self.get_parameter("tracking.camera_frame").value)
        self.exempt_close_range = bool(
            self.get_parameter("tracking.exempt_close_range").value
        )
        self.exempt_outside_fov = bool(
            self.get_parameter("tracking.exempt_outside_fov").value
        )
        self.camera_blind_range = float(
            self.get_parameter("tracking.camera_blind_range").value
        )
        self.camera_half_fov = math.radians(
            float(self.get_parameter("tracking.camera_half_fov_deg").value)
        )
        # With neither exemption there is no blind zone to look up a transform
        # for, and the tracker behaves exactly as it did before.
        self._blind_zone_enabled = self.exempt_close_range or self.exempt_outside_fov
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

        if self.ball_enabled:
            # Published on a timer for the same reason `people_fusion` is: a
            # steady rate through a run of frames with no detection, so a
            # behaviour tree can tell "no ball in the last second" from "the
            # detector has not spoken since the run started".
            rate = float(self.get_parameter("ball.publish_rate").value)
            self.ball_timer = self.create_timer(
                1.0 / max(rate, 1.0), self.publish_balls
            )
            # With the neck tracked, the fixed mounting is what a frame with no
            # neck reading falls back on, and the log says which it is.
            mounting = "fallback camera" if self.neck_mount is not None else "camera"
            self.get_logger().info(
                f"Ball locating on: {self.camera.width:.0f}x{self.camera.height:.0f} "
                f"at {math.degrees(self.camera.hfov):.1f} deg "
                f"(focal {self.camera.focal_x:.0f} px), ball "
                f"{self.ball_geometry.diameter * 100:.1f} cm, {mounting} at x="
                f"{self.ball_geometry.camera_x:.2f} z={self.ball_geometry.camera_z:.2f} "
                f"pitched {math.degrees(self.ball_geometry.camera_pitch):+.1f} deg, "
                f"range from '{self.ball_range_source}'."
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
        """Keep the scan and run one round of fusion on it.

        The fusion used to run on the DR-SPAAM node's `dets` message instead,
        which made that node a single point of failure for a pipeline that
        mostly does not use its output: with it down, a camera detection was
        never ranged and `people_fusion` was silent.
        """
        self.scan_data = msg
        self.merge_detections()

    def _lidar_people_are_fresh(self):
        """Say whether the DR-SPAAM detections are recent enough to use."""
        if self.laser_received_time is None:
            return False
        age = (self.get_clock().now() - self.laser_received_time).nanoseconds / 1e9
        return age <= self.dets_timeout

    def amcl_callback(self, msg):
        self.amcl_pose = msg.pose.pose

    def map_callback(self, msg):
        self.map_data = msg
        # Convert map 1D array to 2D numpy array for fast spatial lookups
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.static_map = scan_clusters.StaticMap(
            self.map_array,
            msg.info.resolution,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
            inflation=self.map_inflation,
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

        self.laser_received_time = self.get_clock().now()

    def arrange_with_scan_dets(self, bound_angle_min, bound_angle_max):
        """
        Pick the DR-SPAAM person inside a camera bearing wedge, or None.

        Takes the two bounds rather than a message, because the wedge now comes
        from two places: a `CamPersonDetection` from the pose detector, which
        computed the angles itself, and a bounding box from the fetch detector,
        whose angles are computed here. The bounds are in the LiDAR's frame:
        the caller has already added `camera_bearing_offset`.
        """
        if not self._lidar_people_are_fresh():
            return None
        # Ensure correct min/max bounds even if wrapped
        ang_min = min(bound_angle_min, bound_angle_max) - self.wedge_margin
        ang_max = max(bound_angle_min, bound_angle_max) + self.wedge_margin

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

    def extrap_from_raw_scan(self, bound_angle_min, bound_angle_max):
        """
        Take a range from the raw scan inside a bearing wedge, or None.

        The last resort, for a wedge with no DR-SPAAM person and no candidate
        cluster in it: whatever the camera is looking at is on, against or
        behind something the map already knows, and `handle_map_occlusion`
        decides which. The range is an actual return near the front of the
        wedge -- see `scan_clusters.structure_range` for what the percentile it
        replaces used to make up.
        """
        if self.scan_data is None:
            return None

        # Order the person bounding angles correctly
        p_min = min(bound_angle_min, bound_angle_max)
        p_max = max(bound_angle_min, bound_angle_max)

        # The bearings are signed and the LD08 scan runs 0..2pi, so the beams
        # are matched modulo a full turn -- see scan_wedge.py for what indexing
        # them directly used to throw away.
        slice_ranges = wedge_ranges(
            self.scan_data.ranges,
            self.scan_data.angle_min,
            self.scan_data.angle_increment,
            p_min,
            p_max,
        )
        if slice_ranges.size == 0:
            return None

        # Filter out inf, nan, and out-of-range limits
        valid_mask = (
            (slice_ranges > self.scan_data.range_min)
            & (slice_ranges < self.scan_data.range_max)
            & ~np.isinf(slice_ranges)
            & ~np.isnan(slice_ranges)
        )
        distance = scan_clusters.structure_range(slice_ranges[valid_mask])
        if distance is None:
            return None

        center_angle = p_min + (p_max - p_min) / 2.0
        x = distance * math.cos(center_angle)
        y = distance * math.sin(center_angle)
        return Pose(position=Point(x=x, y=y, z=0.0))

    def handle_map_occlusion(self, local_pose, transform):
        """
        Place a base-frame point that may have ranged onto mapped structure.

        Returns the map-frame ``(x, y)``. A point on free floor is simply
        transformed; one on a wall or a partition is put behind it or in front
        of it, whichever the map says is possible -- see
        `scan_clusters.place_on_structure`, which replaced a walk along the ray
        that did not stop at the edge of the mapped room.
        """
        mapped = do_transform_pose(local_pose, transform)
        hit = (mapped.position.x, mapped.position.y)
        robot = (
            transform.transform.translation.x,
            transform.transform.translation.y,
        )
        return scan_clusters.place_on_structure(
            self.static_map, robot, hit, self.occlusion_config
        )

    def _build_tracker_config(self):
        """Assemble the map-frame tracker settings from the ROS parameters."""

        def track(name):
            return self.get_parameter(f"tracking.{name}").value

        return TrackerConfig(
            max_association_distance=float(track("max_association_distance")),
            min_hits=int(track("min_hits")),
            max_coast_time=float(track("max_coast_time")),
            max_uncorroborated_time=float(track("max_uncorroborated_time")),
            max_blind_zone_time=float(track("max_blind_zone_time")),
            blind_zone_min_hits=int(track("blind_zone_min_hits")),
            blind_zone_creates_tracks=bool(track("blind_zone_creates_tracks")),
            measurement_noise=float(track("measurement_noise")),
            process_noise=float(track("process_noise")),
            max_reported_speed=float(track("max_reported_speed")),
            coast_velocity_time=float(track("coast_velocity_time")),
        )

    def _camera_coverage(self):
        """Return where the camera is looking, as a :class:`CameraCoverage`.

        The pose comes from the *head*, not the base: the head turns, so a
        person the robot's body has its back to may still be in shot. With no
        transform there is no way to tell what the camera could see, and None
        makes the tracker fall back on demanding corroboration everywhere --
        the stricter of the two behaviours, which is the right way to fail.
        """
        if not self._blind_zone_enabled:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", self.camera_frame, rclpy.time.Time()
            )
        except Exception as error:
            self.get_logger().warn(
                f"camera transform unavailable: {error}", throttle_duration_sec=5.0
            )
            return None

        translation = transform.transform.translation
        q = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        # The camera's axis in the frame the tracks are placed in, which is the
        # LiDAR's: the same offset the bearings get.
        return CameraCoverage(
            x=translation.x,
            y=translation.y,
            yaw=yaw + self.camera_bearing_offset,
            blind_range=self.camera_blind_range,
            half_fov=self.camera_half_fov,
            exempt_close_range=self.exempt_close_range,
            exempt_outside_fov=self.exempt_outside_fov,
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

    @staticmethod
    def _planar_pose(transform):
        """Return a transform as the planar pose ``(x, y, yaw)``."""
        q = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
            yaw,
        )

    def _scan_candidates(self, pose):
        """
        Return the scan's person-sized clusters the static map does not explain.

        In the scan's frame, which this node has always treated as the base
        frame: `base_scan` sits straight above `base_link`, unrotated. `pose`
        is that frame in the map, and is only used to ask the map about them.
        """
        scan = self.scan_data
        if not self.clusters_enabled or scan is None:
            return []
        clusters = scan_clusters.segment_scan(
            scan.ranges,
            scan.angle_min,
            scan.angle_increment,
            range_min=scan.range_min,
            range_max=scan.range_max,
            max_range=self.candidate_config.max_range,
            jump=self.cluster_jump,
            max_gap=self.cluster_max_gap,
        )
        return scan_clusters.candidates(
            clusters, pose, self.static_map, self.candidate_config
        )

    def _publish_candidates(self, candidates, pose, stamp):
        """Publish where the LiDAR sees something that might be a person."""
        message = PoseArray()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        if candidates:
            xs, ys = scan_clusters.to_map([(c.x, c.y) for c in candidates], pose)
            for x, y in zip(xs.tolist(), ys.tolist()):
                message.poses.append(Pose(position=Point(x=x, y=y, z=0.0)))
        self.candidates_pub.publish(message)
        return [(p.position.x, p.position.y) for p in message.poses]

    def _locate_from_bearing(
        self, ang_min, ang_max, candidates, transform, close_range=False
    ):
        """
        Place a camera bearing wedge in the map, or return None.

        The camera gives a bearing and no range, so the range is looked up in
        whatever the LiDAR has in that direction, best evidence first: a
        DR-SPAAM person, then the candidate cluster the wedge points at, and
        only then the bare scan -- which by that point can only be the room
        itself, so the result is placed against or behind it.

        The bounds are the camera's own; `camera_bearing_offset` brings them
        into the LiDAR's frame here, once, for all three.

        A `close_range` detection is dropped, None, unless a candidate in the
        wedge is near enough to be the body that fills the frame.
        """
        low = min(ang_min, ang_max) + self.camera_bearing_offset
        high = max(ang_min, ang_max) + self.camera_bearing_offset

        if (
            close_range
            and self.close_range_require_lidar
            and self.clusters_enabled
            and not scan_clusters.corroborates_close_range(
                candidates, low, high, self.wedge_margin, self.close_range_max_range
            )
        ):
            self.get_logger().info(
                "Dropped a close_range camera detection: no LiDAR candidate "
                f"within {self.close_range_max_range:.1f} m in its wedge",
                throttle_duration_sec=2.0,
            )
            return None

        person_pose = self.arrange_with_scan_dets(low, high)
        if person_pose is not None:
            return self.handle_map_occlusion(person_pose, transform)

        cluster = scan_clusters.choose_for_wedge(
            candidates, low, high, self.wedge_margin
        )
        if cluster is not None:
            mapped = do_transform_pose(
                Pose(position=Point(x=cluster.x, y=cluster.y, z=0.0)), transform
            )
            return mapped.position.x, mapped.position.y

        person_pose = self.extrap_from_raw_scan(low, high)
        if person_pose is None:
            return None
        return self.handle_map_occlusion(person_pose, transform)

    def _camera_wedges(self):
        """
        Return every fresh camera bearing wedge, from either detector.

        A wedge is `(min, max, close_range)`, the bounds in radians in the
        robot's frame, positive to the left, which is what the LiDAR lookup
        wants. Only the pose detector labels a detection close_range. The pose detector supplies
        them ready-made on `cam_people_detections`; the fetch detector supplies
        boxes and they are computed here from the lens. Both are staleness
        checked against the same timeout: a bearing from a frame that is seconds
        old keeps placing a person wherever the scan happens to hit inside a
        wedge that no longer means anything, and that is true whichever detector
        drew the box.
        """
        wedges = []
        if self._camera_is_fresh():
            for person in self.cam_detections:
                wedges.append(
                    (
                        person.bound_angle_min.data,
                        person.bound_angle_max.data,
                        person.type.data == TYPE_CLOSE_RANGE,
                    )
                )
        if self._boxes_are_fresh(self.person_boxes_time):
            for box in self.person_boxes:
                centre_u, _, width, _, _ = box
                wedges.append(
                    (
                        self.camera.bearing(centre_u + width / 2.0),
                        self.camera.bearing(centre_u - width / 2.0),
                        False,
                    )
                )
        return wedges

    def merge_detections(self):
        """Turn this round's detections into map-frame measurements.

        Called from the scan callback, which is the faster of the two inputs.
        `people_fusion` is not published here: with tracking on, the
        measurements are left for :meth:`publish_tracks` to fold in on its
        timer, so it runs at a steady rate whether or not a detection arrived.
        Publishing on arrival is what used to make a rejected camera frame look
        exactly like an empty room. `lidar_candidates` *is* published here,
        once per scan.
        """
        transform = self._map_transform()
        if transform is None:
            return
        self.trans = transform
        pose = self._planar_pose(transform)

        candidates = self._scan_candidates(pose)
        cluster_points = self._publish_candidates(
            candidates, pose, self.get_clock().now().to_msg()
        )

        camera_points = []
        for ang_min, ang_max, close_range in self._camera_wedges():
            placed = self._locate_from_bearing(
                ang_min, ang_max, candidates, transform, close_range
            )
            if placed is not None:
                camera_points.append(placed)
            if self.debug_mode:
                self.fill_bound_angle(ang_min, ang_max)

        if not self.tracking_enabled:
            self._publish_untracked(camera_points)
            self._publish_debug_fov()
            return

        # LiDAR people the camera did not vouch for. These may sustain a track
        # the camera has lost - the whole point, for a person the pose network
        # cannot skeletonise - but never start one, because a leg-sized return
        # on its own is not evidence of a person.
        lidar_points = []
        if self._lidar_people_are_fresh():
            for laser_pose in self.laser_detections:
                mapped = do_transform_pose(laser_pose, transform)
                lidar_points.append((mapped.position.x, mapped.position.y))

        measurements = combine_measurements(
            camera_points,
            lidar_points,
            self.tracker.config.max_association_distance,
            cluster_points=cluster_points if self.clusters_sustain else (),
            cluster_gate=self.cluster_gate,
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
        tracks = self.tracker.step(
            measurements or [], now.nanoseconds * 1e-9, self._camera_coverage()
        )
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

    # --- the ball ------------------------------------------------------------

    def _build_ball_config(self):
        """Assemble the ball tracker settings from the ROS parameters."""

        def track(name):
            return self.get_parameter(f"ball.tracking.{name}").value

        return ball_locating.BallTrackerConfig(
            max_association_distance=float(track("max_association_distance")),
            min_hits=int(track("min_hits")),
            max_coast_time=float(track("max_coast_time")),
            position_gain=float(track("position_gain")),
            velocity_gain=float(track("velocity_gain")),
            max_speed=float(track("max_speed")),
        )

    def ball_boxes_callback(self, msg):
        """Keep this frame's ball boxes; the placing happens on the timer."""
        with self._ball_lock:
            self.ball_boxes = self._unpack(msg)
            self.ball_received_time = self.get_clock().now()
            # The frame's own stamp, which is what the neck is looked up at.
            self.ball_boxes_stamp = self._stamp_seconds(msg.header.stamp)

    def neck_callback(self, msg):
        """Record where the neck was told to be, against the board's stamp."""
        stamp = self._stamp_seconds(msg.header.stamp)
        with self._ball_lock:
            self.neck_history.submit(stamp, int(msg.pos_n))

    def _stamp_seconds(self, stamp):
        """Return a header stamp in seconds, or now when it was never set."""
        if stamp.sec or stamp.nanosec:
            return stamp.sec + stamp.nanosec * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9

    def person_boxes_callback(self, msg):
        """Keep this frame's person boxes, for the bearing wedges."""
        with self._ball_lock:
            self.person_boxes = self._unpack(msg)
            self.person_boxes_time = self.get_clock().now()

    def _unpack(self, msg):
        """Reduce a Detection2DArray to `(u, v, width, height, score)` tuples."""
        boxes = []
        for detection in msg.detections:
            score = 0.0
            for result in detection.results:
                score = max(score, float(result.hypothesis.score))
            boxes.append(
                (
                    float(detection.bbox.center.position.x),
                    float(detection.bbox.center.position.y),
                    float(detection.bbox.size_x),
                    float(detection.bbox.size_y),
                    score,
                )
            )
        self._check_box_bounds(boxes)
        return boxes

    def _check_box_bounds(self, boxes):
        """
        Warn once when a box does not fit the frame this node was told about.

        The detector and this node hold two copies of the camera geometry and
        never compare them, because they never talk. A box centred outside the
        declared frame is the cheapest evidence that the copies have drifted --
        and until they agree, every bearing and every apparent-size range
        computed here is wrong by the ratio between them.
        """
        if self._warned_box_bounds:
            return
        for centre_u, centre_v, _, _, _ in boxes:
            if 0.0 <= centre_u <= self.camera.width and 0.0 <= centre_v <= self.camera.height:
                continue
            self._warned_box_bounds = True
            self.get_logger().warn(
                f"A detection box is centred at ({centre_u:.0f}, {centre_v:.0f}), "
                f"outside the {self.camera.width:.0f}x{self.camera.height:.0f} frame "
                "this node is configured for. The detector's camera_params and "
                "this node's have drifted apart; every bearing and range here is "
                "wrong until they match. Reported once."
            )
            return

    def _boxes_are_fresh(self, received):
        """Say whether a box message arrived recently enough to act on."""
        if received is None:
            return False
        age = (self.get_clock().now() - received).nanoseconds / 1e9
        return age <= self.ball_detection_timeout

    def _ball_measurements(self, transform):
        """Place this frame's ball boxes in the map frame."""
        with self._ball_lock:
            fresh = self._boxes_are_fresh(self.ball_received_time)
            boxes = list(self.ball_boxes) if fresh else []
            stamp = self.ball_boxes_stamp
        if not boxes:
            return []

        # One tilt for the whole message: every box in it is from one frame.
        geometry = self._ball_geometry_at(stamp)
        measurements = []
        for centre_u, centre_v, width, height, score in boxes:
            box = (centre_u, centre_v, width, height)
            observation = ball_locating.locate(
                self.camera,
                box,
                geometry,
                score=score,
                prefer=self.ball_range_source,
            )
            if observation is None:
                continue
            self._check_range_agreement(observation, box, geometry)
            local = Pose(
                position=Point(
                    x=observation.x, y=observation.y, z=observation.z
                )
            )
            mapped = do_transform_pose(local, transform)
            measurements.append(
                ball_locating.BallMeasurement(
                    x=mapped.position.x,
                    y=mapped.position.y,
                    z=mapped.position.z,
                    score=observation.score,
                )
            )
        return measurements

    def _ball_geometry_at(self, stamp):
        """
        Return the camera geometry a frame taken at `stamp` was seen with.

        The neck's position at that instant when there is one. Otherwise the
        fixed `ball.camera_*` numbers, and said out loud: a height worked out
        from a tilt the head did not have is exactly what made the fetch tree
        call a ball on the floor out of reach.
        """
        if self.neck_mount is None:
            return self.ball_geometry
        with self._ball_lock:
            ticks = self.neck_history.at(stamp)
        if ticks is None:
            reason = "there is no neck reading within ball.neck.stale_s of the frame"
        else:
            placed = self.neck_mount.geometry(self.ball_geometry, ticks)
            if placed is not None:
                return placed
            reason = f"the neck reads {ticks} ticks, which is not a head position"
        self.get_logger().warn(
            "Placing a ball with the fixed camera pitch of "
            f"{math.degrees(self.ball_geometry.camera_pitch):+.1f} deg because "
            f"{reason}; its height is only as right as that pitch.",
            throttle_duration_sec=5.0,
        )
        return self.ball_geometry

    def _check_range_agreement(self, observation, box, geometry):
        """
        Say once when the two range estimators disagree badly.

        They measure the same distance by unrelated routes -- the ball's size
        and the floor's geometry -- so agreement is evidence that the camera's
        height and tilt are right, and a persistent gap is evidence that they
        are not. Neither estimator can notice this alone: each produces a
        perfectly plausible number from wrong inputs. `geometry` is the one
        the frame was actually placed with, which with the neck tracked is not
        the fixed `ball.camera_*`.
        """
        if self._warned_range_disagreement:
            return
        if observation.size_range <= 0.0 or observation.ground_range <= 0.0:
            return
        gap = abs(observation.size_range - observation.ground_range)
        if gap <= self.ball_disagreement_warn:
            return
        self._warned_range_disagreement = True
        if self.neck_mount is not None:
            suspect = (
                "ball.neck.pitch_at_level_deg, or the head having moved on from "
                "the frame"
            )
        else:
            suspect = "ball.camera_z or ball.camera_pitch_deg"
        implied = ball_locating.floor_pitch(self.camera, box, geometry)
        implied_text = (
            ""
            if implied is None
            else f" If the ball is on the floor, the camera was pitched "
            f"{math.degrees(implied):+.1f} deg."
        )
        self.get_logger().warn(
            f"The ball's apparent size says {observation.size_range:.2f} m and the "
            f"floor plane says {observation.ground_range:.2f} m, a gap of "
            f"{gap:.2f} m. The two are independent, so this is most likely the "
            f"camera's tilt ({math.degrees(geometry.camera_pitch):+.1f} deg) or "
            f"height ({geometry.camera_z:.2f} m) not matching the robot -- check "
            f"{suspect} -- or the ball not being ball.diameter across, or not "
            f"on the floor.{implied_text} Published range is from "
            f"'{self.ball_range_source}'. Reported once."
        )

    def publish_balls(self):
        """
        Advance the ball tracker and publish where every ball is.

        Both forms go out together and describe the same tracks, so a consumer
        never has to reconcile them. Nothing is published when there is no
        confirmed ball: an empty PoseArray and no message mean the same thing
        to a consumer that ages its last message, and this way the topic's
        traffic is a sign that something is being seen.
        """
        transform = self._map_transform()
        if transform is None:
            return

        now = self.get_clock().now()
        tracks = self.ball_tracker.step(
            self._ball_measurements(transform), now.nanoseconds * 1e-9
        )
        if not tracks:
            return

        stamp = now.to_msg()
        poses = PoseArray()
        poses.header.stamp = stamp
        poses.header.frame_id = "map"
        detections = Detection3DArray()
        detections.header = poses.header

        for track in tracks:
            x, y, z = track.position
            poses.poses.append(Pose(position=Point(x=x, y=y, z=z)))

            detection = Detection3D()
            detection.header = poses.header
            detection.id = self.ball_class_id
            detection.bbox.center.position.x = x
            detection.bbox.center.position.y = y
            detection.bbox.center.position.z = z
            detection.bbox.center.orientation.w = 1.0
            detection.bbox.size.x = self.ball_geometry.diameter
            detection.bbox.size.y = self.ball_geometry.diameter
            detection.bbox.size.z = self.ball_geometry.diameter
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = self.ball_class_id
            hypothesis.hypothesis.score = float(track.score)
            hypothesis.pose.pose.position.x = x
            hypothesis.pose.pose.position.y = y
            hypothesis.pose.pose.position.z = z
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            detections.detections.append(detection)

        self.ball_pub.publish(poses)
        self.ball_detections_pub.publish(detections)

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
