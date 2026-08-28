import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from mecanumbot_msgs.msg import CamPersonDetectionArray, CamPersonDetection
from std_msgs.msg import Float32 as Float
from std_msgs.msg import Header, String
from geometry_msgs.msg import Pose, PoseArray, Point
import os
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp, GLib
import pyds
import ctypes
import numpy as np
import cv2
import math
import transforms3d as t3d
from ament_index_python.packages import get_package_share_directory

from mecanumbot_sensorprocess_smart.ros4hri_bridge import (
    BodyIdTracker,
    Ros4HriBodyBroadcaster,
    normalized_roi,
    skeleton2d_from_coco17,
)
from mecanumbot_sensorprocess_smart.person_gating import (
    DetectionConfirmer,
    GateConfig,
    evaluate_evidence,
    is_close_range,
)

# `CamPersonDetection.type`: how much of the body the detection is based on.
# Close to the robot the camera, mounted at shin height, frames only the legs,
# so a consumer that needs arms or a head -- the ostensive tree reading
# gestures, for instance -- has to be able to tell the two apart.
TYPE_FULL_BODY = "full_body"
TYPE_CLOSE_RANGE = "close_range"

# PersonKeypoints declares its 17 Pose fields in COCO-17 order, which is the
# order the pose parser emits them in, so the two are filled by zipping.
KEYPOINT_FIELDS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Standard YOLO pose skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Head/Face
    (5, 6),  # Shoulders
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # Arms
    (11, 12),
    (5, 11),
    (6, 12),  # Torso/Hips
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # Legs
]

# A skeleton has to land inside its own bounding box. These decide when the
# node is entitled to say that it did not: how many joints have to be visible
# before the frame is worth judging, and how far outside the box a joint may
# still sit (as a fraction of the box) before it counts as outside it.
KEYPOINT_ALIGNMENT_MIN_VISIBLE = 4
KEYPOINT_ALIGNMENT_MARGIN = 0.25


class DeepStreamPersonDetectNode(Node):
    def __init__(self, namespace=""):
        super().__init__("mecanumbot_cam_detect_people_ds")
        self.declare_parameters(
            namespace=namespace,
            parameters=[
                ("camera_params.camera_width", 1280),
                ("camera_params.camera_height", 720),
                ("camera_params.camera_fov", math.radians(60.0)),
                ("from_topic", False),
                ("camera_topic", "camera/image_raw/compressed"),
                ("webcam_device", "/dev/video0"),
                ("debug_mode", False),
                ("keypoint_scaling", "auto"),
                ("ros4hri.enabled", True),
                ("ros4hri.prefix", "/humans"),
                ("ros4hri.publish_rate", 30.0),
                ("ros4hri.body_timeout", 0.5),
                ("ros4hri.publish_roi", True),
                ("ros4hri.iou_threshold", 0.3),
                ("ros4hri.frame_id", ""),
                # ---- detection gate (see person_gating.py) ----
                ("detection_gate.box_conf_acquire", 0.6),
                ("detection_gate.box_conf_retain", 0.35),
                ("detection_gate.keypoint_conf", 0.3),
                ("detection_gate.best_keypoint_conf_acquire", 0.7),
                ("detection_gate.best_keypoint_conf_retain", 0.5),
                ("detection_gate.min_valid_keypoints_acquire", 6),
                ("detection_gate.min_valid_keypoints_retain", 3),
                ("detection_gate.min_torso_keypoints_acquire", 2),
                ("detection_gate.min_torso_keypoints_retain", 1),
                ("detection_gate.min_box_height", 40.0),
                ("detection_gate.max_box_aspect_ratio", 1.6),
                # ---- close range: only the legs of a nearby person are in
                # frame, so the torso requirement is unmeetable ----
                ("detection_gate.proximity_enabled", True),
                ("detection_gate.proximity_min_height_fraction", 0.6),
                ("detection_gate.proximity_top_margin", 8.0),
                ("detection_gate.proximity_box_conf_acquire", 0.5),
                ("detection_gate.proximity_box_conf_retain", 0.3),
                ("detection_gate.proximity_min_valid_keypoints_acquire", 2),
                ("detection_gate.proximity_min_valid_keypoints_retain", 1),
                ("detection_gate.proximity_min_lower_body_acquire", 2),
                ("detection_gate.proximity_min_lower_body_retain", 1),
                ("detection_gate.proximity_max_box_aspect_ratio", 2.5),
                ("detection_gate.min_hits", 2),
                ("detection_gate.max_missed_time", 0.5),
                ("detection_gate.iou_threshold", 0.3),
                ("detection_gate.log_rejections", False),
            ],
        )

        self.camera_width = self.get_parameter("camera_params.camera_width").value
        self.camera_height = self.get_parameter("camera_params.camera_height").value
        self.camera_fov = self.get_parameter("camera_params.camera_fov").value
        self.from_topic = self.get_parameter("from_topic").value
        self.webcam_device = self.get_parameter("webcam_device").value
        self.Y_padding = 0  # (self.camera_width - self.camera_height) / 2.0
        self.debug_mode = self.get_parameter("debug_mode").value
        self.keypoint_scaling = str(self.get_parameter("keypoint_scaling").value)

        # A detection is accepted on keypoint evidence and temporal
        # consistency, not on the box score alone -- see person_gating.py for
        # why one confidence threshold cannot both keep bean bags out and let
        # partly occluded people in.
        self.gate_config = self._build_gate_config()
        self.confirmer = DetectionConfirmer(self.gate_config)
        self.log_rejections = bool(
            self.get_parameter("detection_gate.log_rejections").value
        )
        # Joint visibility, i.e. whether a keypoint is usable. Much lower than
        # the box gates on purpose: it decides which joints are drawn and fed
        # into the angular bounds, never whether the object is a person.
        self.min_conf_threshold = self.gate_config.keypoint_conf

        # Initialize GStreamer
        Gst.init(None)
        self.pipeline = Gst.Pipeline()

        # Build Pipeline Elements
        self.get_logger().info("source: " + str(self.from_topic))
        if self.from_topic:
            self.source = Gst.ElementFactory.make("appsrc", "ros-image-source")
            self.source.set_property("is-live", True)
            caps = Gst.Caps.from_string(
                f"video/x-raw, format=BGR, width={self.camera_width}, height={self.camera_height}, framerate=15/1"
            )
            self.source.set_property("caps", caps)

            sensor_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.get_parameter("camera_topic").value,
                self.image_callback,
                sensor_qos,
            )
        else:
            self.source = Gst.ElementFactory.make("v4l2src", "webcam-source")
            self.source.set_property("device", self.webcam_device)

            # Force hardware webcam to physically capture at 1280x720 (16:9)
            self.webcam_caps = Gst.ElementFactory.make("capsfilter", "webcam_caps")
            caps = Gst.Caps.from_string(
                f"video/x-raw, width={self.camera_width}, height={self.camera_height}"
            )
            self.webcam_caps.set_property("caps", caps)

        self.vidconv_src = Gst.ElementFactory.make("nvvideoconvert", "convertor_src")
        self.mux = Gst.ElementFactory.make("nvstreammux", "muxer")
        self.mux.set_property("width", self.camera_width)
        self.mux.set_property("height", self.camera_height)
        self.mux.set_property("batch-size", 1)
        self.mux.set_property("batched-push-timeout", 40000)

        self.nvinfer = Gst.ElementFactory.make("nvinfer", "primary-inference")
        path = get_package_share_directory("mecanumbot_sensorprocess_smart")
        nvinfer_config = os.path.join(
            path, "deepstream_config", "config_infer_yolo26_pose.txt"
        )
        self.nvinfer.set_property("config-file-path", nvinfer_config)
        self._resolve_keypoint_scaling(nvinfer_config)
        self.network_input_size = self._read_infer_dims(nvinfer_config)
        self._announced_frame_size = False
        self._announced_network_size = False
        self._warned_network_size = False
        self._checked_alignment = False
        self._dumped_raw_metadata = False
        self._frame_shape = None
        self._warned_surface_size = False
        # Metadata pixels -> surface pixels. Unity unless nvstreammux hands the
        # probe a frame of a different size than it was configured for.
        self._overlay_scale = (1.0, 1.0)

        # --- NEW ELEMENTS FOR IMAGE EXTRACTION ---
        # Converts infer output format to RGBA so Python can read it
        self.vidconv_out = Gst.ElementFactory.make("nvvideoconvert", "convertor_out")
        self.capsfilter_out = Gst.ElementFactory.make("capsfilter", "capsfilter_rgba")
        caps_rgba = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        self.capsfilter_out.set_property("caps", caps_rgba)

        self.sink = Gst.ElementFactory.make("fakesink", "fakesink")

        # Add all elements to pipeline
        # Add all elements to pipeline
        elements_to_add = [
            self.source,
            self.vidconv_src,
            self.mux,
            self.nvinfer,
            self.vidconv_out,
            self.capsfilter_out,
            self.sink,
        ]
        if not self.from_topic:
            elements_to_add.insert(1, self.webcam_caps)

        for elem in elements_to_add:
            self.pipeline.add(elem)

        # Link elements appropriately based on source type
        if self.from_topic:
            self.source.link(self.vidconv_src)
        else:
            self.source.link(self.webcam_caps)
            self.webcam_caps.link(self.vidconv_src)

        vidconv_src_pad = self.vidconv_src.get_static_pad("src")
        mux_sink_pad = self.mux.get_request_pad("sink_0")
        vidconv_src_pad.link(mux_sink_pad)

        self.mux.link(self.nvinfer)
        self.nvinfer.link(self.vidconv_out)
        self.vidconv_out.link(self.capsfilter_out)
        self.capsfilter_out.link(self.sink)

        # Attach Probe to the end of the capsfilter so RGBA format is guaranteed
        probe_pad = self.capsfilter_out.get_static_pad("src")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, self.metadata_probe, 0)

        self.camera_right_yaw = -self.camera_fov / 2
        self.camera_left_yaw = self.camera_fov / 2

        self.people_left_FOV = PoseArray()
        self.people_left_FOV.header.frame_id = (
            f'{self.get_namespace().strip("/")}/head_link'
            if self.get_namespace().strip("/")
            else "head_link"
        )
        self.people_right_FOV = PoseArray()
        self.people_right_FOV.header.frame_id = (
            f'{self.get_namespace().strip("/")}/head_link'
            if self.get_namespace().strip("/")
            else "head_link"
        )
        # Publishers
        self.people_pub = self.create_publisher(
            CamPersonDetectionArray, "cam_people_detections", 10
        )

        if self.debug_mode:
            self.debug_image_pub = self.create_publisher(
                CompressedImage, "cam_people_detections/debug_image/compressed", 10
            )

        self.people_msg = CamPersonDetectionArray()
        ros_namespace = self.get_namespace().strip("/")
        self.people_msg.header.frame_id = (
            f"{ros_namespace}/head_link" if ros_namespace else "head_link"
        )

        # --- ROS4HRI (REP-155) ---
        # DeepStream and ROS4HRI are orthogonal: nvinfer decides *how* bodies are
        # found, ROS4HRI decides *how they are published*. The metadata probe
        # below therefore feeds both the mecanumbot-native messages and the
        # /humans/bodies tree from the same NvDsObjectMeta.
        self.ros4hri_enabled = bool(self.get_parameter("ros4hri.enabled").value)
        self.hri_frame_id = (
            str(self.get_parameter("ros4hri.frame_id").value)
            or self.people_msg.header.frame_id
        )
        self.hri_publish_roi = bool(self.get_parameter("ros4hri.publish_roi").value)
        self.hri_broadcaster = None
        self.body_id_tracker = None
        if self.ros4hri_enabled:
            body_timeout = float(self.get_parameter("ros4hri.body_timeout").value)
            self.hri_broadcaster = Ros4HriBodyBroadcaster(
                self,
                prefix=str(self.get_parameter("ros4hri.prefix").value),
                body_timeout=body_timeout,
                publish_roi=self.hri_publish_roi,
            )
            self.body_id_tracker = BodyIdTracker(
                iou_threshold=float(self.get_parameter("ros4hri.iou_threshold").value),
                max_missed_time=body_timeout,
            )
            # Entity creation/teardown must not happen on the GStreamer streaming
            # thread that runs the probe, so the broadcaster is drained here.
            publish_rate = max(
                1.0, float(self.get_parameter("ros4hri.publish_rate").value)
            )
            self.hri_timer = self.create_timer(
                1.0 / publish_rate, self.hri_broadcaster.flush
            )
            self.get_logger().info(
                f"ROS4HRI publishing enabled under "
                f"{str(self.get_parameter('ros4hri.prefix').value).rstrip('/')}/bodies/"
            )

        self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info("DeepStream Pipeline Running!")
        self._announce_input_geometry()

    def _build_gate_config(self):
        """Assemble the detection gate thresholds from the ROS parameters."""

        def gate(name):
            return self.get_parameter(f"detection_gate.{name}").value

        config = GateConfig(
            box_conf_acquire=float(gate("box_conf_acquire")),
            box_conf_retain=float(gate("box_conf_retain")),
            keypoint_conf=float(gate("keypoint_conf")),
            best_keypoint_conf_acquire=float(gate("best_keypoint_conf_acquire")),
            best_keypoint_conf_retain=float(gate("best_keypoint_conf_retain")),
            min_valid_keypoints_acquire=int(gate("min_valid_keypoints_acquire")),
            min_valid_keypoints_retain=int(gate("min_valid_keypoints_retain")),
            min_torso_keypoints_acquire=int(gate("min_torso_keypoints_acquire")),
            min_torso_keypoints_retain=int(gate("min_torso_keypoints_retain")),
            min_box_height=float(gate("min_box_height")),
            max_box_aspect_ratio=float(gate("max_box_aspect_ratio")),
            proximity_enabled=bool(gate("proximity_enabled")),
            proximity_min_height_fraction=float(gate("proximity_min_height_fraction")),
            proximity_top_margin=float(gate("proximity_top_margin")),
            proximity_box_conf_acquire=float(gate("proximity_box_conf_acquire")),
            proximity_box_conf_retain=float(gate("proximity_box_conf_retain")),
            proximity_min_valid_keypoints_acquire=int(
                gate("proximity_min_valid_keypoints_acquire")
            ),
            proximity_min_valid_keypoints_retain=int(
                gate("proximity_min_valid_keypoints_retain")
            ),
            proximity_min_lower_body_acquire=int(
                gate("proximity_min_lower_body_acquire")
            ),
            proximity_min_lower_body_retain=int(
                gate("proximity_min_lower_body_retain")
            ),
            proximity_max_box_aspect_ratio=float(
                gate("proximity_max_box_aspect_ratio")
            ),
            min_hits=int(gate("min_hits")),
            max_missed_time=float(gate("max_missed_time")),
            iou_threshold=float(gate("iou_threshold")),
        )

        # Hysteresis only exists if retain is looser than acquire; misordered
        # thresholds would silently make the retain gate the stricter of the
        # two, which is the opposite of the intent.
        if config.box_conf_retain > config.box_conf_acquire:
            self.get_logger().warn(
                f"detection_gate.box_conf_retain ({config.box_conf_retain}) is above "
                f"box_conf_acquire ({config.box_conf_acquire}); hysteresis is inverted."
            )
        if config.keypoint_conf >= config.box_conf_acquire:
            self.get_logger().warn(
                f"detection_gate.keypoint_conf ({config.keypoint_conf}) is at or above "
                "box_conf_acquire; joints on partly occluded people will be discarded."
            )
        if config.proximity_box_conf_retain > config.proximity_box_conf_acquire:
            self.get_logger().warn(
                f"detection_gate.proximity_box_conf_retain "
                f"({config.proximity_box_conf_retain}) is above "
                f"proximity_box_conf_acquire ({config.proximity_box_conf_acquire}); "
                "hysteresis is inverted for close-range detections."
            )

        self.get_logger().info(
            "Detection gate: acquire on box>"
            f"{config.box_conf_acquire:.2f} with >={config.min_valid_keypoints_acquire} "
            f"keypoints (>={config.min_torso_keypoints_acquire} torso) over "
            f"{config.keypoint_conf:.2f}, confirmed after {config.min_hits} frames; "
            f"retain on box>{config.box_conf_retain:.2f} for up to "
            f"{config.max_missed_time:.2f}s."
        )
        if config.proximity_enabled:
            self.get_logger().info(
                "Close-range branch: boxes starting within "
                f"{config.proximity_top_margin:.0f}px of the top edge and covering "
                f">={config.proximity_min_height_fraction:.0%} of the frame height "
                f"acquire on box>{config.proximity_box_conf_acquire:.2f} with "
                f">={config.proximity_min_lower_body_acquire} hip/knee/ankle joints "
                "and no torso requirement."
            )
        else:
            self.get_logger().info(
                "Close-range branch disabled; a person nearer than about 2m has no "
                "torso in frame and will not be acquired."
            )
        return config

    def _resolve_keypoint_scaling(self, nvinfer_config):
        """Pick the inverse of the pre-processing letterbox/stretch nvinfer applies.

        Keypoints come out of the parser in network-input coordinates, so mapping
        them back to image pixels has to undo exactly the resize nvinfer did.
        That is controlled by `maintain-aspect-ratio` in the nvinfer config:
        1 means letterboxed (uniform scale plus padding), 0 means the frame was
        stretched independently in x and y.
        """
        if self.keypoint_scaling not in ("auto", "letterbox", "stretch"):
            self.get_logger().warn(
                f"Unknown keypoint_scaling '{self.keypoint_scaling}', using 'letterbox'."
            )
            self.keypoint_scaling = "letterbox"

        if self.keypoint_scaling != "auto":
            self.get_logger().info(f"Keypoint scaling: {self.keypoint_scaling}")
            return

        maintain_aspect_ratio = None
        try:
            with open(nvinfer_config, "r") as config_file:
                for line in config_file:
                    line = line.split("#", 1)[0].strip()
                    if line.startswith("maintain-aspect-ratio"):
                        maintain_aspect_ratio = int(line.split("=", 1)[1].strip())
        except (OSError, ValueError) as exc:
            self.get_logger().warn(f"Could not read {nvinfer_config}: {exc}")

        if maintain_aspect_ratio is None:
            self.get_logger().warn(
                "maintain-aspect-ratio not found in the nvinfer config, "
                "assuming 'letterbox'."
            )
            self.keypoint_scaling = "letterbox"
        else:
            self.keypoint_scaling = "letterbox" if maintain_aspect_ratio else "stretch"
            self.get_logger().info(
                f"Keypoint scaling: {self.keypoint_scaling} "
                f"(maintain-aspect-ratio={maintain_aspect_ratio})"
            )

    def _read_infer_dims(self, nvinfer_config):
        """
        Return the configured network input as (width, height), or None.

        `infer-dims` is optional: without it nvinfer takes the input shape from
        the model, which is only knowable once inference has run. The real
        value is announced from the first inferred frame instead.
        """
        try:
            with open(nvinfer_config, "r") as config_file:
                for line in config_file:
                    line = line.split("#", 1)[0].strip()
                    if line.startswith("infer-dims"):
                        # DeepStream spells this channels;height;width.
                        dims = line.split("=", 1)[1].strip().split(";")
                        return (int(dims[2]), int(dims[1]))
        except (OSError, ValueError, IndexError) as exc:
            self.get_logger().warn(
                f"Could not read infer-dims from {nvinfer_config}: {exc}"
            )
        return None

    def _announce_input_geometry(self):
        """
        Log the frame and network geometry the angular mapping depends on.

        Every bearing this node reports is derived from a keypoint's x within
        `camera_width`, so a source that does not deliver that size is worth
        seeing in the log rather than inferring from wrong angles.
        """
        if self.from_topic:
            source = f"topic '{self.get_parameter('camera_topic').value}'"
        else:
            source = f"webcam {self.webcam_device}"
        self.get_logger().info(
            f"Input image: {self.camera_width}x{self.camera_height} requested from "
            f"{source}, streammux {self.camera_width}x{self.camera_height}, "
            f"HFOV {math.degrees(self.camera_fov):.1f} deg."
        )
        if self.network_input_size is None:
            self.get_logger().info(
                "Network input: taken from the model (no infer-dims in the nvinfer "
                "config); announced from the first inferred frame."
            )
        else:
            net_width, net_height = self.network_input_size
            self.get_logger().info(
                f"Network input: {net_width}x{net_height} (infer-dims), undone with "
                f"the {self.keypoint_scaling} mapping."
            )

    def _announce_frame_size(self, frame_meta):
        """
        Log the size the source really delivered, once, and flag a mismatch.

        nvstreammux rescales whatever arrives to `camera_width`x`camera_height`,
        so a differing aspect ratio is silently distorted into that frame and
        every angle derived from it is off.
        """
        self._announced_frame_size = True
        # Older pyds builds do not carry the source frame size on the metadata.
        width = int(getattr(frame_meta, "source_frame_width", 0) or 0)
        height = int(getattr(frame_meta, "source_frame_height", 0) or 0)
        if not width or not height:
            return
        self.get_logger().info(f"First frame arrived at {width}x{height}.")
        if (width, height) == (self.camera_width, self.camera_height):
            return
        source_aspect = width / height
        configured_aspect = self.camera_width / self.camera_height
        message = (
            f"Source delivers {width}x{height} but camera_params say "
            f"{self.camera_width}x{self.camera_height}; nvstreammux is rescaling."
        )
        if abs(source_aspect - configured_aspect) > 0.01:
            self.get_logger().warn(
                f"{message} The aspect ratios differ ({source_aspect:.3f} vs "
                f"{configured_aspect:.3f}), so the frame is distorted and every "
                "reported angle is wrong until the two match."
            )
        else:
            self.get_logger().info(message)

    def _resolve_network_size(self, mask_params):
        """
        Return the network input size the keypoints are expressed in.

        The parser stamps the network dimensions onto every mask it emits, so
        the metadata is the only source that reflects the engine nvinfer
        actually loaded. `infer-dims` says what the config asks for, which is
        not the same thing: the engine filename carries no input size, so an
        engine built from an earlier export keeps being loaded and keeps
        running at its own size however the ONNX changes. The declared size is
        therefore used to cross-check the metadata, never to override it.
        """
        net_width = float(getattr(mask_params, "width", 0.0) or 0.0)
        net_height = float(getattr(mask_params, "height", 0.0) or 0.0)
        declared = self.network_input_size

        if net_width > 0.0 and net_height > 0.0:
            if (
                declared is not None
                and (int(net_width), int(net_height)) != tuple(declared)
                and not self._warned_network_size
            ):
                self._warned_network_size = True
                self.get_logger().warn(
                    f"The metadata reports a {int(net_width)}x{int(net_height)} "
                    f"network, but infer-dims declares {declared[0]}x{declared[1]}. "
                    "That is what a stale model-engine-file looks like: nvinfer "
                    "does not rebuild an engine when the ONNX behind it is "
                    "re-exported at a different imgsz. Delete the engine to force a "
                    "rebuild. The metadata size is the one used here, being the one "
                    "that actually ran."
                )
            return net_width, net_height

        if declared is not None:
            if not self._warned_network_size:
                self._warned_network_size = True
                self.get_logger().warn(
                    "The mask metadata carries no network size; falling back to the "
                    f"{declared[0]}x{declared[1]} infer-dims from the nvinfer config."
                )
            return float(declared[0]), float(declared[1])

        if not self._warned_network_size:
            self._warned_network_size = True
            self.get_logger().error(
                "The mask metadata carries no network size and the nvinfer config "
                "sets no infer-dims, so there is nothing to undo the input resize "
                "with. The keypoints are left in network coordinates, which is only "
                "right if the network input happens to match the frame."
            )
        return float(self.camera_width), float(self.camera_height)

    def _network_to_pixel_transform(self, mask_params):
        """Return (gain_x, gain_y, pad_x, pad_y) undoing the nvinfer input resize."""
        net_width, net_height = self._resolve_network_size(mask_params)

        if not self._announced_network_size:
            self._announced_network_size = True
            self.get_logger().info(
                f"Network input: {int(net_width)}x{int(net_height)}, undone with the "
                f"{self.keypoint_scaling} mapping onto "
                f"{self.camera_width}x{self.camera_height} pixels."
            )

        if self.keypoint_scaling == "stretch":
            return (
                net_width / self.camera_width,
                net_height / self.camera_height,
                0.0,
                0.0,
            )

        gain = min(net_width / self.camera_width, net_height / self.camera_height)
        pad_x = (net_width - self.camera_width * gain) / 2.0
        pad_y = (net_height - self.camera_height * gain) / 2.0
        return (gain, gain, pad_x, pad_y)

    def image_callback(self, msg):
        self.get_logger().debug("Received image from ROS topic.")
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            data = cv_image.tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            self.source.emit("push-buffer", buffer)
        except Exception as e:
            self.get_logger().error(f"Image decode failed: {e}")

    def XYN_to_Pose(self, x, y, conf):
        msg = Pose()
        if conf > self.min_conf_threshold:
            msg.position.x = float(x)
            msg.position.y = float(y)
            msg.position.z = 0.0
        else:
            msg.position.x = float("nan")
            msg.position.y = float("nan")
            msg.position.z = 0.0
        return msg

    def cam_to_angle(self, X):
        X_inv = (
            1 - X
        )  # Invert X to match the robot's coordinate system rather than the camera's coordinate system
        angle = (
            1 - X_inv
        ) * self.camera_right_yaw + X_inv * self.camera_left_yaw  # direction: right to left increase
        # self.get_logger().info(f"####### Calculated angle: {angle} from X: {X} with camera FOV: {math.degrees(self.camera_fov)} degrees")
        return angle

    def _submit_ros4hri(self, candidates):
        """Convert one frame of DeepStream detections into the /humans/bodies tree.

        Runs on the GStreamer streaming thread, so it only builds messages and
        hands them to the broadcaster; the actual publishing happens on the ROS
        executor thread.
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.hri_frame_id

        boxes = [self._to_corners(c["rect"]) for c in candidates]
        now = self.get_clock().now().nanoseconds * 1e-9
        body_ids = self.body_id_tracker.update(boxes, now)

        bodies = []
        for body_id, candidate in zip(body_ids, candidates):
            skeleton_msg = skeleton2d_from_coco17(
                candidate["keypoints"],
                self.camera_width,
                self.camera_height,
                self.min_conf_threshold,
                header=header,
            )
            roi_msg = None
            if self.hri_publish_roi:
                left, top, width, height = candidate["rect"]
                roi_msg = normalized_roi(
                    left,
                    top,
                    width,
                    height,
                    self.camera_width,
                    self.camera_height,
                    confidence=candidate["confidence"],
                    header=header,
                )
            bodies.append((body_id, skeleton_msg, roi_msg))

        self.hri_broadcaster.submit(bodies)

    @staticmethod
    def _to_corners(rect):
        """(left, top, width, height) -> (xmin, ymin, xmax, ymax), as IoU wants."""
        left, top, width, height = rect
        return (left, top, left + width, top + height)

    def _pixel_keypoints(self, keypoints, mask_params):
        """Map the parser's network-space keypoints into camera pixel space.

        Removes the input padding and divides by the gain, i.e. undoes exactly
        the resize nvinfer performed on the way in.
        """
        gain_x, gain_y, pad_x, pad_y = self._network_to_pixel_transform(mask_params)
        pixel_kpts = []
        for i in range(17):
            conf = float(keypoints[i][0])
            px = (float(keypoints[i][1]) - pad_x) / gain_x
            py = (float(keypoints[i][2]) - pad_y) / gain_y
            pixel_kpts.append((conf, px, py))
        return pixel_kpts

    def _update_overlay_scale(self, frame_shape):
        """
        Keep the overlay in the surface's pixels, not the muxer's.

        Every coordinate in the metadata -- `rect_params` from nvinfer and the
        keypoints mapped here alike -- is in the frame nvinfer believes it
        inferred on, which is the size `nvstreammux` was configured for. That
        is not always the size of the buffer the probe receives: a source whose
        frames are passed through rather than rescaled leaves the surface at
        its own resolution, and the overlay then lands on a differently shaped
        image. Drawing at the muxer's scale would shrink the whole annotation
        towards the top-left corner, box and skeleton together, in exact
        proportion to the height the muxer thinks it has.

        Only the drawing is corrected. The published bearings and the detection
        gate are normalised by `camera_params`, which is the same space the
        metadata is in, so they stay self-consistent -- but the absolute
        thresholds among them (`min_box_height`, `max_box_aspect_ratio`) are
        measured on a frame of the wrong shape, which is why the mismatch is
        worth a warning rather than a silent correction.
        """
        height, width = float(frame_shape[0]), float(frame_shape[1])
        if width <= 0.0 or height <= 0.0:
            return
        self._overlay_scale = (width / self.camera_width, height / self.camera_height)
        if self._warned_surface_size:
            return
        self._warned_surface_size = True
        if (int(width), int(height)) == (self.camera_width, self.camera_height):
            return
        self.get_logger().warn(
            f"The probe receives {int(width)}x{int(height)} frames, but "
            f"camera_params say {self.camera_width}x{self.camera_height}, which is "
            "also what nvstreammux was configured for and therefore the space every "
            "box and keypoint arrives in. The overlay is rescaled by "
            f"({self._overlay_scale[0]:.3f}, {self._overlay_scale[1]:.3f}) to land on "
            "the image, but the detection gate still measures min_box_height and "
            "max_box_aspect_ratio on a frame of the wrong shape. Set "
            f"camera_params.camera_width/camera_height to {int(width)}/{int(height)} "
            "so the two agree."
        )

    def _dump_raw_metadata(self, keypoints, mask_params, rect, pixel_kpts):
        """
        Print every number the keypoint mapping depends on, once.

        The mapping is a chain of assumptions -- what the parser writes into
        the mask, what coordinate space it is in, what the network input was,
        what resize nvinfer applied -- and a wrong skeleton looks the same
        whichever link broke. Dumping the raw triplets next to the mapped ones,
        with the box nvinfer mapped itself for comparison, settles which one it
        is from a single frame instead of from the shape of the overlay.
        """
        if self._dumped_raw_metadata:
            return
        self._dumped_raw_metadata = True
        raw = [
            "({:.2f}, {:.2f}, {:.2f})".format(*(float(v) for v in keypoints[i]))
            for i in range(min(4, len(keypoints)))
        ]
        mapped = [
            "(conf {:.2f} -> {:.0f}, {:.0f})".format(c, x, y)
            for c, x, y in pixel_kpts[:4]
        ]
        self.get_logger().info(
            "Raw metadata dump (once, debug_mode only):\n"
            f"  surface (h, w, c):   {self._frame_shape}, overlay rescaled by "
            f"{self._overlay_scale[0]:.3f}, {self._overlay_scale[1]:.3f}\n"
            f"  mask width x height: {mask_params.width}x{mask_params.height}, "
            f"size {mask_params.size} bytes ({mask_params.size // 4} floats)\n"
            f"  camera_params:       {self.camera_width}x{self.camera_height}\n"
            f"  keypoint_scaling:    {self.keypoint_scaling}\n"
            f"  box from nvinfer:    left {rect[0]:.0f} top {rect[1]:.0f} "
            f"{rect[2]:.0f}x{rect[3]:.0f}\n"
            f"  first raw triplets:  {', '.join(raw)}\n"
            f"  same four mapped:    {', '.join(mapped)}\n"
            "The raw triplets are (conf, x, y) if the first number of each is in "
            "[0, 1]; if instead the last one is, the parser writes (x, y, conf) and "
            "this node is reading the triplet backwards. The mapped joints have to "
            "fall inside the box above, which nvinfer mapped with the resize it "
            "really applied."
        )

    def _check_keypoint_alignment(self, pixel_kpts, rect):
        """
        Report once when the mapped keypoints miss their own bounding box.

        `rect_params` arrives already in frame pixels -- nvinfer undoes its own
        input resize for the box -- so the box is an independent witness to the
        mapping applied to the keypoints here. A skeleton drawn outside its own
        box means the two disagree, and nothing else says so: the detection is
        still published, with bearings read off joints that are in the wrong
        place. Judged on the first frame that shows enough of a body, because
        a mapping error is systematic and one frame settles it.
        """
        if self._checked_alignment:
            return
        visible = [
            (px, py) for conf, px, py in pixel_kpts if conf > self.min_conf_threshold
        ]
        if len(visible) < KEYPOINT_ALIGNMENT_MIN_VISIBLE:
            return
        self._checked_alignment = True

        left, top, width, height = rect
        margin_x = width * KEYPOINT_ALIGNMENT_MARGIN
        margin_y = height * KEYPOINT_ALIGNMENT_MARGIN
        inside = sum(
            1
            for px, py in visible
            if left - margin_x <= px <= left + width + margin_x
            and top - margin_y <= py <= top + height + margin_y
        )
        if inside * 2 >= len(visible):
            return

        xs = [px for px, _ in visible]
        ys = [py for _, py in visible]
        self.get_logger().warn(
            f"Only {inside} of {len(visible)} mapped keypoints land inside their own "
            f"box: the joints span x {min(xs):.0f}..{max(xs):.0f}, y "
            f"{min(ys):.0f}..{max(ys):.0f}, while the box is ({left:.0f},{top:.0f}) "
            f"{width:.0f}x{height:.0f}. The box is mapped into pixels by nvinfer and "
            "the keypoints by this node, so the two mappings disagree. Check "
            f"keypoint_scaling (now '{self.keypoint_scaling}') against "
            "maintain-aspect-ratio in the nvinfer config, and the network input size "
            "logged above against the model that was meant to be loaded. Every "
            "bearing this node reports is wrong until they agree. Reported once."
        )

    def _build_person_msg(self, pixel_kpts, rect, close_range=False):
        """Build one CamPersonDetection from pixel-space keypoints and its box.

        `close_range` only labels the detection: the bearing is computed from
        whichever joints were found, so legs alone give a bearing exactly the
        way a whole body does. What the label carries is that the upper body is
        outside the frame rather than merely undetected.
        """
        person_msg = CamPersonDetection()
        person_msg.type = String(
            data=TYPE_CLOSE_RANGE if close_range else TYPE_FULL_BODY
        )

        # Normalized [0, 1] coordinates; joints below the visibility threshold
        # become NaN inside XYN_to_Pose. KEYPOINT_FIELDS is in COCO-17 order,
        # which is the order the parser emits.
        for field, (conf, px, py) in zip(KEYPOINT_FIELDS, pixel_kpts):
            setattr(
                person_msg.keypoints,
                field,
                self.XYN_to_Pose(
                    px / self.camera_width, py / self.camera_height, conf
                ),
            )

        angles = [
            self.cam_to_angle(k[1] / self.camera_width)
            for k in pixel_kpts
            if k[0] > self.min_conf_threshold
        ]
        if not angles:
            # No joint cleared the visibility threshold, which happens on
            # distant or heavily occluded people. The bounding box still
            # brackets them, and falling back to it is what stops the bearing
            # collapsing to the default 0.0 -- i.e. dead ahead, which the
            # fusion layer would otherwise take at face value.
            left, _, width, _ = rect
            angles = [
                self.cam_to_angle(left / self.camera_width),
                self.cam_to_angle((left + width) / self.camera_width),
            ]

        person_msg.bound_angle_min = Float(data=float(min(angles)))
        person_msg.bound_angle_max = Float(data=float(max(angles)))
        return person_msg

    def _collect_candidates(self, frame_meta):
        """Read every object in the frame into plain Python, before gating.

        The gate associates detections across frames, so it has to see the
        whole frame at once; this pass exists to give it that.
        """
        candidates = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            mask_params = obj_meta.mask_params
            if mask_params.size > 0:
                raw_data = mask_params.get_mask_array()
                keypoints = np.array(raw_data).flatten()[:51].reshape((17, 3))
                rect = (
                    float(obj_meta.rect_params.left),
                    float(obj_meta.rect_params.top),
                    float(obj_meta.rect_params.width),
                    float(obj_meta.rect_params.height),
                )
                pixel_kpts = self._pixel_keypoints(keypoints, mask_params)
                if self.debug_mode:
                    self._dump_raw_metadata(keypoints, mask_params, rect, pixel_kpts)
                self._check_keypoint_alignment(pixel_kpts, rect)
                candidates.append(
                    {
                        "rect": rect,
                        "keypoints": pixel_kpts,
                        "confidence": float(obj_meta.confidence),
                        "evidence": evaluate_evidence(
                            pixel_kpts,
                            obj_meta.confidence,
                            rect[2],
                            rect[3],
                            self.gate_config,
                            box_top=rect[1],
                            image_height=self.camera_height,
                        ),
                    }
                )
            l_obj = l_obj.next
        return candidates

    def _draw_detection(
        self, debug_img, candidate, accepted, reason, close_range=False
    ):
        """Annotate one candidate, accepted or not.

        Rejected boxes are drawn too, in red and labelled with the check they
        failed -- tuning the gate is impossible without seeing what it threw
        away and why. Accepted close-range boxes are drawn green rather than
        blue, so which of the two gates let a box through is visible without
        reading the label.
        """
        scale_x, scale_y = self._overlay_scale
        rect = [
            candidate["rect"][0] * scale_x,
            candidate["rect"][1] * scale_y,
            candidate["rect"][2] * scale_x,
            candidate["rect"][3] * scale_y,
        ]
        pixel_kpts = [
            (conf, px * scale_x, py * scale_y)
            for conf, px, py in candidate["keypoints"]
        ]
        if not accepted:
            colour = (0, 0, 255)
        elif close_range:
            colour = (0, 200, 0)
        else:
            colour = (255, 0, 0)

        x1, y1 = int(rect[0]), int(rect[1])
        w, h = int(rect[2]), int(rect[3])
        cv2.rectangle(debug_img, (x1, y1), (x1 + w, y1 + h), colour, 2)

        box_label = f"{candidate['confidence']:.2f}"
        if close_range:
            box_label += " near"
        if not accepted:
            box_label += f" {reason}"
        (text_w, text_h), baseline = cv2.getTextSize(
            box_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_x = x1
        label_y = max(y1, text_h + baseline + 4)
        cv2.rectangle(
            debug_img,
            (label_x, label_y - text_h - baseline - 4),
            (label_x + text_w + 8, label_y + 2),
            colour,
            -1,
        )
        cv2.putText(
            debug_img,
            box_label,
            (label_x + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Skeleton lines, using TRUE integer pixel coordinates
        for p1, p2 in SKELETON_CONNECTIONS:
            conf_p1, x_p1, y_p1 = pixel_kpts[p1]
            conf_p2, x_p2, y_p2 = pixel_kpts[p2]
            if (
                conf_p1 > self.min_conf_threshold
                and conf_p2 > self.min_conf_threshold
            ):
                cv2.line(
                    debug_img,
                    (int(x_p1), int(y_p1)),
                    (int(x_p2), int(y_p2)),
                    (0, 255, 255),
                    2,
                )

        # Keypoint dots
        for i in range(17):
            kconf, kx, ky = pixel_kpts[i]
            if kconf > self.min_conf_threshold:
                px = int(kx)
                py = int(ky)
                cv2.circle(debug_img, (px, py), 4, (0, 255, 0), -1)
                label = f"{kconf:.2f}"
                text_pos = (px + 6, py - 6)
                cv2.putText(
                    debug_img,
                    label,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    debug_img,
                    label,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    def metadata_probe(self, pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        detected_people = []

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if not self._announced_frame_size:
                self._announce_frame_size(frame_meta)

            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_copy = np.array(n_frame, copy=True, order="C")
            self._frame_shape = frame_copy.shape
            self._update_overlay_scale(frame_copy.shape)
            debug_img = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)

            candidates = self._collect_candidates(frame_meta)

            # Keypoint evidence plus hysteresis and temporal confirmation. The
            # box score alone cannot tell a bean bag from a person who happens
            # to be half behind one.
            now = self.get_clock().now().nanoseconds * 1e-9
            verdicts = self.confirmer.update(
                [(self._to_corners(c["rect"]), c["evidence"]) for c in candidates],
                now,
            )

            hri_candidates = []
            for candidate, (accepted, reason) in zip(candidates, verdicts):
                close_range = is_close_range(candidate["evidence"], self.gate_config)
                if accepted:
                    detected_people.append(
                        self._build_person_msg(
                            candidate["keypoints"],
                            candidate["rect"],
                            close_range=close_range,
                        )
                    )
                    if self.ros4hri_enabled:
                        hri_candidates.append(candidate)
                elif self.log_rejections:
                    self.get_logger().info(
                        f"rejected box@{candidate['confidence']:.2f}: {reason}"
                    )

                if self.debug_mode:
                    self._draw_detection(
                        debug_img, candidate, accepted, reason, close_range
                    )

            if self.ros4hri_enabled:
                self._submit_ros4hri(hri_candidates)

            if self.debug_mode:
                debug_msg = CompressedImage()
                debug_msg.header.stamp = self.get_clock().now().to_msg()
                debug_msg.format = "jpeg"
                _, encoded_img = cv2.imencode(".jpg", debug_img)
                debug_msg.data = encoded_img.tobytes()
                self.debug_image_pub.publish(debug_msg)

            # Very important to prevent memory leaks on Jetson hardware!
            try:
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except AttributeError:
                # Fails gracefully if you are on an older DeepStream version that doesn't mandate unmapping
                pass

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        # Publish the Metadata back to ROS
        if detected_people:
            # self.get_logger().info(f"Detected {len(detected_people)} people in the frame.")
            self.people_msg.header.stamp = self.get_clock().now().to_msg()
            self.people_msg.people = detected_people
            self.people_pub.publish(self.people_msg)

        return Gst.PadProbeReturn.OK


def main(args=None):
    rclpy.init(args=args)
    node = DeepStreamPersonDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.set_state(Gst.State.NULL)
        if node.hri_broadcaster is not None:
            node.hri_broadcaster.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
