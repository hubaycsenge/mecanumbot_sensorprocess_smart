"""
The fetch game's detector: one DeepStream pass that finds people and balls.

`mecanumbot_onboard_cam_detect_people` runs a **pose** model and reports
skeletons, which is what the leading and ostensive experiments need. The fetch
game needs something that model cannot give: a tennis ball. A pose network has
exactly one class, so a second class is not a threshold away -- it is a
different network.

So this is that network. A plain YOLO detector (`yolo26m` by default, the COCO
80-class model), run through the same DeepStream pipeline, filtered down to the
two classes the game is about and published as bounding boxes on **two topics**:

    cam_people_boxes   vision_msgs/Detection2DArray   COCO class 0
    cam_ball_boxes     vision_msgs/Detection2DArray   COCO class 32, sports ball

Two topics rather than one array with labels because the consumers are
different: the ball boxes become a ball position in the map frame, the person
boxes become a bearing wedge the LiDAR ranges, and a subscriber that only wants
one of them should not have to filter the other out at 15 Hz.

**Pixels, not angles.** The pose node publishes bearings, having done the
image -> angle conversion itself; this one publishes the boxes as they came out
of the network. That is the standard type's shape (`Detection2D` is a box in an
image) and it puts the camera model in one place -- `mecanumbot_locate_detections`
-- which is where the range has to be worked out anyway. It also means the two
nodes cannot disagree about which way round the frame is, because only one of
them has an opinion.

Nothing here is pose-specific, so nothing here reads a mask: the parser is
DeepStream-Yolo's ordinary `NvDsInferParseYolo` and the metadata is boxes and
class ids. What survives from the pose node is the pipeline (appsrc from a ROS
topic or v4l2src from a webcam, nvstreammux, nvinfer, a probe on an RGBA
capsfilter) and the rendering of the nvinfer config from `model_params`, which
is shared code in `nvinfer_config.py`.

## Running it alongside the pose detector

Both may run at once -- they are separate nvinfer instances with separate
`gie-unique-id`s and separate topics -- but on an Orin Nano two networks on one
camera stream is most of the GPU. The intended arrangement is that the fetch
game runs this one *instead* (`use_fetch_detector:=true use_pose_detector:=false`
on the base launch), and `mecanumbot_locate_detections` accepts person evidence
from either source, so `people_fusion` keeps flowing whichever is up.
"""

import math
import os
import tempfile

import gi
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst  # noqa: E402  (gi requires the version call first)

import cv2  # noqa: E402
import pyds  # noqa: E402

from mecanumbot_sensorprocess_smart import nvinfer_config  # noqa: E402
from mecanumbot_sensorprocess_smart.object_gating import (  # noqa: E402
    BoxConfirmer,
    ClassGate,
    evaluate_shape,
)

# COCO class ids, for the default model. They are parameters rather than
# constants because a re-trained model renumbers them, but these are the two the
# shipped `yolo26m` uses.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# Colours for the debug overlay: people blue, balls yellow, rejects red.
COLOUR_PERSON = (255, 0, 0)
COLOUR_BALL = (0, 220, 220)
COLOUR_REJECTED = (0, 0, 255)


class DeepStreamObjectDetectNode(Node):
    """Detect people and balls in one DeepStream pass, publish both as boxes."""

    def __init__(self, namespace=""):
        super().__init__("mecanumbot_cam_detect_objects_ds")
        self.declare_parameters(
            namespace=namespace,
            parameters=[
                ("camera_params.camera_width", 1280),
                ("camera_params.camera_height", 720),
                ("camera_params.camera_fov", math.radians(60.0)),
                # ---- the detection model (see nvinfer_config.py) ----
                ("model_params.imgsz", 640),
                ("model_params.model_name", "yolo26m"),
                ("model_params.precision", "fp16"),
                ("model_params.models_dir", ""),
                ("model_params.custom_lib_path", ""),
                ("model_params.nvinfer_config", ""),
                ("from_topic", False),
                ("camera_topic", "camera/image_raw/compressed"),
                ("webcam_device", "/dev/video0"),
                ("debug_mode", False),
                ("log_rejections", False),
                # ---- what to keep, and what to call it ----
                ("classes.person_id", COCO_PERSON),
                ("classes.person_label", "person"),
                ("classes.person_topic", "cam_people_boxes"),
                ("classes.ball_id", COCO_SPORTS_BALL),
                ("classes.ball_label", "sports ball"),
                ("classes.ball_topic", "cam_ball_boxes"),
                # ---- the person gate ----
                # Wide shape bounds: this detector has no skeleton to judge on,
                # and the fusion layer still needs the LiDAR to agree there is
                # something at that bearing before a person is published.
                ("person_gate.conf_acquire", 0.5),
                ("person_gate.conf_retain", 0.3),
                ("person_gate.min_size", 20.0),
                ("person_gate.min_aspect", 0.05),
                ("person_gate.max_aspect", 2.5),
                ("person_gate.min_hits", 2),
                ("person_gate.max_missed_time", 0.5),
                ("person_gate.iou_threshold", 0.3),
                # ---- the ball gate ----
                # A tennis ball is round, so the aspect bounds are tight and are
                # the main thing keeping a yellow floor marking out. They are
                # not 1.0 +- nothing because a ball in flight smears and a ball
                # against a chair leg is clipped.
                ("ball_gate.conf_acquire", 0.4),
                ("ball_gate.conf_retain", 0.25),
                ("ball_gate.min_size", 6.0),
                ("ball_gate.min_aspect", 0.55),
                ("ball_gate.max_aspect", 1.8),
                ("ball_gate.min_hits", 2),
                ("ball_gate.max_missed_time", 0.4),
                ("ball_gate.iou_threshold", 0.2),
            ],
        )

        self.camera_width = int(self.get_parameter("camera_params.camera_width").value)
        self.camera_height = int(
            self.get_parameter("camera_params.camera_height").value
        )
        self.camera_fov = float(self.get_parameter("camera_params.camera_fov").value)
        self.from_topic = bool(self.get_parameter("from_topic").value)
        self.webcam_device = str(self.get_parameter("webcam_device").value)
        self.debug_mode = bool(self.get_parameter("debug_mode").value)
        self.log_rejections = bool(self.get_parameter("log_rejections").value)

        self.gates = self._build_gates()
        self.confirmers = {
            gate.class_id: BoxConfirmer(gate) for gate in self.gates.values()
        }

        self._frame_id = self._namespaced("head_link")
        self.publishers_by_class = {
            gate.class_id: self.create_publisher(
                Detection2DArray, gate.topic, 10
            )
            for gate in self.gates.values()
        }

        self._build_pipeline()

        if self.debug_mode:
            self.debug_image_pub = self.create_publisher(
                CompressedImage, "cam_object_detections/debug_image/compressed", 10
            )

        self._announced_frame_size = False
        self._warned_surface_size = False
        self._overlay_scale = (1.0, 1.0)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info("DeepStream fetch detector running.")
        self._announce_input_geometry()

    # --- configuration --------------------------------------------------------

    def _namespaced(self, link):
        namespace = self.get_namespace().strip("/")
        return f"{namespace}/{link}" if namespace else link

    def _build_gates(self):
        """Assemble one `ClassGate` per class this node publishes."""

        def gate(prefix, name):
            return self.get_parameter(f"{prefix}.{name}").value

        def build(prefix, id_param, label_param, topic_param):
            return ClassGate(
                class_id=int(self.get_parameter(id_param).value),
                label=str(self.get_parameter(label_param).value),
                topic=str(self.get_parameter(topic_param).value),
                conf_acquire=float(gate(prefix, "conf_acquire")),
                conf_retain=float(gate(prefix, "conf_retain")),
                min_size=float(gate(prefix, "min_size")),
                min_aspect=float(gate(prefix, "min_aspect")),
                max_aspect=float(gate(prefix, "max_aspect")),
                min_hits=int(gate(prefix, "min_hits")),
                max_missed_time=float(gate(prefix, "max_missed_time")),
                iou_threshold=float(gate(prefix, "iou_threshold")),
            )

        person = build(
            "person_gate",
            "classes.person_id",
            "classes.person_label",
            "classes.person_topic",
        )
        ball = build(
            "ball_gate",
            "classes.ball_id",
            "classes.ball_label",
            "classes.ball_topic",
        )
        if person.class_id == ball.class_id:
            raise ValueError(
                "classes.person_id and classes.ball_id are both "
                f"{person.class_id}; the two classes have to be different ids "
                "in whichever model is loaded"
            )
        for entry in (person, ball):
            if entry.conf_retain > entry.conf_acquire:
                self.get_logger().warn(
                    f"{entry.label}: conf_retain ({entry.conf_retain}) is above "
                    f"conf_acquire ({entry.conf_acquire}); hysteresis is inverted."
                )
        self.get_logger().info(
            f"Publishing class {person.class_id} ('{person.label}') on "
            f"{person.topic} and class {ball.class_id} ('{ball.label}') on "
            f"{ball.topic}."
        )
        return {"person": person, "ball": ball}

    def _model_path(self):
        """Return `(onnx, engine)` for the configured model, or `(None, None)`."""
        models_dir = str(self.get_parameter("model_params.models_dir").value or "")
        if not models_dir:
            models_dir = os.path.join(
                get_package_share_directory("mecanumbot_sensorprocess_smart"), "models"
            )
        imgsz = int(self.get_parameter("model_params.imgsz").value)
        model_name = str(self.get_parameter("model_params.model_name").value)
        precision = str(self.get_parameter("model_params.precision").value)

        onnx, engine = nvinfer_config.model_paths(
            models_dir, imgsz, model_name, precision
        )
        if not os.path.isfile(onnx):
            self.get_logger().error(
                f"No ONNX at {onnx}. model_params.imgsz={imgsz} selects "
                f"models/imgsz_{imgsz}/, so either export the model at that size "
                f"(models/conv_to_onnx.py {model_name} --imgsz {imgsz}) or set "
                "imgsz to a size that is present."
            )
            return None, None
        return onnx, engine

    def _render_nvinfer_config(self):
        """
        Write the nvinfer config for the selected detector and return its path.

        The packaged `config_infer_yolo26_det.txt` is a template; the model, its
        engine and the input size it was exported at are rewritten here from
        `model_params`. `model_params.nvinfer_config` bypasses all of it and
        hands nvinfer the named file untouched.
        """
        share = get_package_share_directory("mecanumbot_sensorprocess_smart")
        template = os.path.join(
            share, "deepstream_config", "config_infer_yolo26_det.txt"
        )

        override = str(self.get_parameter("model_params.nvinfer_config").value or "")
        if override:
            self.get_logger().info(f"nvinfer config: {override} (used as-is).")
            return override

        onnx, engine = self._model_path()
        if onnx is None:
            self.get_logger().warn(
                f"Falling back to the packaged {template} unchanged; whatever model "
                "it names is the one that will run."
            )
            return template

        imgsz = int(self.get_parameter("model_params.imgsz").value)
        custom_lib = str(self.get_parameter("model_params.custom_lib_path").value or "")
        # Relative paths in an nvinfer config resolve against the config's own
        # directory, so they have to be absolutized before the copy moves.
        substitutions = {
            "onnx-file": onnx,
            "model-engine-file": engine,
            "infer-dims": "3;{};{}".format(imgsz, imgsz),
            "labelfile-path": os.path.join(share, "deepstream_config", "labels_coco.txt"),
        }
        if custom_lib:
            substitutions["custom-lib-path"] = custom_lib

        rendered = os.path.join(
            tempfile.gettempdir(),
            "mecanumbot_nvinfer_{}_imgsz{}.txt".format(
                str(self.get_parameter("model_params.model_name").value), imgsz
            ),
        )
        try:
            nvinfer_config.render_config(template, substitutions, rendered)
        except OSError as exc:
            self.get_logger().error(
                f"Could not render {template} to {rendered}: {exc}; using the "
                "template unchanged."
            )
            return template

        self.get_logger().info(
            f"nvinfer config: {rendered} (rendered from {template}) -> "
            f"{os.path.basename(onnx)} at {imgsz}x{imgsz}."
        )
        if not os.path.isfile(engine):
            self.get_logger().info(
                f"No engine at {engine} yet; nvinfer will build one on startup "
                "(minutes). models/build_engine.py builds it ahead of time."
            )
        return rendered

    # --- pipeline -------------------------------------------------------------

    def _build_pipeline(self):
        """Assemble and link the GStreamer pipeline, and attach the probe."""
        Gst.init(None)
        self.pipeline = Gst.Pipeline()

        if self.from_topic:
            self.source = Gst.ElementFactory.make("appsrc", "ros-image-source")
            self.source.set_property("is-live", True)
            self.source.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, format=BGR, width={self.camera_width}, "
                    f"height={self.camera_height}, framerate=15/1"
                ),
            )
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
            self.webcam_caps = Gst.ElementFactory.make("capsfilter", "webcam_caps")
            self.webcam_caps.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, width={self.camera_width}, "
                    f"height={self.camera_height}"
                ),
            )

        self.vidconv_src = Gst.ElementFactory.make("nvvideoconvert", "convertor_src")
        self.mux = Gst.ElementFactory.make("nvstreammux", "muxer")
        self.mux.set_property("width", self.camera_width)
        self.mux.set_property("height", self.camera_height)
        self.mux.set_property("batch-size", 1)
        self.mux.set_property("batched-push-timeout", 40000)

        self.nvinfer = Gst.ElementFactory.make("nvinfer", "object-inference")
        rendered = self._render_nvinfer_config()
        self.nvinfer.set_property("config-file-path", rendered)
        self.network_input_size = self._read_infer_dims(rendered)

        self.vidconv_out = Gst.ElementFactory.make("nvvideoconvert", "convertor_out")
        self.capsfilter_out = Gst.ElementFactory.make("capsfilter", "capsfilter_rgba")
        self.capsfilter_out.set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        )
        self.sink = Gst.ElementFactory.make("fakesink", "fakesink")

        elements = [
            self.source,
            self.vidconv_src,
            self.mux,
            self.nvinfer,
            self.vidconv_out,
            self.capsfilter_out,
            self.sink,
        ]
        if not self.from_topic:
            elements.insert(1, self.webcam_caps)
        for element in elements:
            self.pipeline.add(element)

        if self.from_topic:
            self.source.link(self.vidconv_src)
        else:
            self.source.link(self.webcam_caps)
            self.webcam_caps.link(self.vidconv_src)

        self.vidconv_src.get_static_pad("src").link(self.mux.get_request_pad("sink_0"))
        self.mux.link(self.nvinfer)
        self.nvinfer.link(self.vidconv_out)
        self.vidconv_out.link(self.capsfilter_out)
        self.capsfilter_out.link(self.sink)

        self.capsfilter_out.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, self.metadata_probe, 0
        )

    def _read_infer_dims(self, config_path):
        try:
            return nvinfer_config.read_infer_dims(config_path)
        except (OSError, ValueError, IndexError) as exc:
            self.get_logger().warn(
                f"Could not read infer-dims from {config_path}: {exc}"
            )
            return None

    def _announce_input_geometry(self):
        """
        Log the frame geometry every downstream bearing depends on.

        This node publishes pixels, so the geometry matters one step later --
        `mecanumbot_locate_detections` turns a box into a bearing and a range
        using its own `camera_params`. Two copies of the same three numbers is a
        drift risk, so both are logged where a run can be checked against them.
        """
        source = (
            f"topic '{self.get_parameter('camera_topic').value}'"
            if self.from_topic
            else f"webcam {self.webcam_device}"
        )
        self.get_logger().info(
            f"Input image: {self.camera_width}x{self.camera_height} requested from "
            f"{source}, HFOV {math.degrees(self.camera_fov):.1f} deg. The boxes are "
            "published in these pixels; mecanumbot_locate_detections must be "
            "configured with the same frame size."
        )
        if self.network_input_size is not None:
            width, height = self.network_input_size
            self.get_logger().info(f"Network input: {width}x{height} (infer-dims).")

    # --- ROS input ------------------------------------------------------------

    def image_callback(self, msg):
        """Push one compressed ROS frame into the appsrc."""
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            data = frame.tobytes()
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            self.source.emit("push-buffer", buffer)
        except Exception as error:
            self.get_logger().error(f"Image decode failed: {error}")

    # --- detection ------------------------------------------------------------

    def _collect_candidates(self, frame_meta):
        """Read one frame's objects into plain Python, grouped by class."""
        candidates = {gate.class_id: [] for gate in self.gates.values()}
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            class_id = int(obj_meta.class_id)
            if class_id in candidates:
                rect = (
                    float(obj_meta.rect_params.left),
                    float(obj_meta.rect_params.top),
                    float(obj_meta.rect_params.width),
                    float(obj_meta.rect_params.height),
                )
                candidates[class_id].append((rect, float(obj_meta.confidence)))
            l_obj = l_obj.next
        return candidates

    def _judge(self, gate, entries, now):
        """
        Run one class's candidates through the shape check and the confirmer.

        Shape first and separately: a box of the wrong shape is not a marginal
        detection of the right thing, and letting it into the confirmer would
        let it hold a track alive.
        """
        shaped, verdicts = [], []
        for rect, score in entries:
            ok, reason = evaluate_shape(rect[2], rect[3], gate)
            verdicts.append([False, reason])
            if ok:
                shaped.append((len(verdicts) - 1, _corners(rect), score))

        confirmed = self.confirmers[gate.class_id].update(
            [(box, score) for _, box, score in shaped], now
        )
        for (index, _, _), (accepted, reason) in zip(shaped, confirmed):
            verdicts[index] = [accepted, reason]
        return [tuple(verdict) for verdict in verdicts]

    def _detection_msg(self, gate, rect, score, header):
        """One `Detection2D`: the box in image pixels, with its class and score."""
        detection = Detection2D()
        detection.header = header
        detection.id = gate.label
        bbox = BoundingBox2D()
        bbox.center.position.x = float(rect[0] + rect[2] / 2.0)
        bbox.center.position.y = float(rect[1] + rect[3] / 2.0)
        bbox.center.theta = 0.0
        bbox.size_x = float(rect[2])
        bbox.size_y = float(rect[3])
        detection.bbox = bbox
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = gate.label
        hypothesis.hypothesis.score = float(score)
        detection.results.append(hypothesis)
        return detection

    def metadata_probe(self, pad, info, u_data):
        """Turn one batch of DeepStream metadata into the two ROS topics."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        now = self.get_clock().now()
        stamp = now.to_msg()
        accepted = {gate.class_id: [] for gate in self.gates.values()}

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if not self._announced_frame_size:
                self._announce_frame_size(frame_meta)

            debug_img = None
            if self.debug_mode:
                surface = pyds.get_nvds_buf_surface(
                    hash(gst_buffer), frame_meta.batch_id
                )
                frame_copy = np.array(surface, copy=True, order="C")
                self._update_overlay_scale(frame_copy.shape)
                debug_img = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)

            candidates = self._collect_candidates(frame_meta)
            seconds = now.nanoseconds * 1e-9
            for gate in self.gates.values():
                entries = candidates[gate.class_id]
                for (rect, score), (ok, reason) in zip(
                    entries, self._judge(gate, entries, seconds)
                ):
                    if ok:
                        accepted[gate.class_id].append((rect, score))
                    elif self.log_rejections:
                        self.get_logger().info(
                            f"rejected {gate.label}@{score:.2f}: {reason}"
                        )
                    if debug_img is not None:
                        self._draw(debug_img, gate, rect, score, ok, reason)

            if debug_img is not None:
                self._publish_debug(debug_img, stamp)

            # Very important to prevent memory leaks on Jetson hardware.
            try:
                pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            except AttributeError:
                # Older DeepStream versions do not mandate unmapping.
                pass

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        for gate in self.gates.values():
            entries = accepted[gate.class_id]
            if not entries:
                continue
            msg = Detection2DArray()
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame_id
            for rect, score in entries:
                msg.detections.append(
                    self._detection_msg(gate, rect, score, msg.header)
                )
            self.publishers_by_class[gate.class_id].publish(msg)

        return Gst.PadProbeReturn.OK

    # --- diagnostics ----------------------------------------------------------

    def _announce_frame_size(self, frame_meta):
        """Log the size the source really delivered, once, and flag a mismatch."""
        self._announced_frame_size = True
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
                f"{message} The aspect ratios differ, so the frame is distorted "
                "and every range computed from a box width downstream is wrong "
                "until the two match."
            )
        else:
            self.get_logger().info(message)

    def _update_overlay_scale(self, frame_shape):
        """
        Keep the debug overlay in the surface's pixels, not the muxer's.

        Every coordinate in the metadata is in the frame nvinfer believes it
        inferred on, which is the size nvstreammux was configured for. A source
        whose frames are passed through rather than rescaled leaves the surface
        at its own resolution, and the overlay would then land on a differently
        shaped image. Only the drawing is corrected: the published boxes stay in
        the metadata's own space, which is the space the fusion layer's
        `camera_params` describe.
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
            f"The probe receives {int(width)}x{int(height)} frames but "
            f"camera_params say {self.camera_width}x{self.camera_height}. The "
            "overlay is rescaled to land on the image; the published boxes are "
            "left in the metadata's space. Set camera_params to "
            f"{int(width)}/{int(height)} so the two agree."
        )

    def _draw(self, debug_img, gate, rect, score, accepted, reason):
        """Annotate one box, accepted or not; rejects carry the check they failed."""
        scale_x, scale_y = self._overlay_scale
        x1 = int(rect[0] * scale_x)
        y1 = int(rect[1] * scale_y)
        x2 = int((rect[0] + rect[2]) * scale_x)
        y2 = int((rect[1] + rect[3]) * scale_y)
        if not accepted:
            colour = COLOUR_REJECTED
        elif gate.class_id == self.gates["ball"].class_id:
            colour = COLOUR_BALL
        else:
            colour = COLOUR_PERSON
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), colour, 2)

        label = f"{gate.label} {score:.2f}"
        if not accepted:
            label += f" {reason}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        label_y = max(y1, text_h + baseline + 4)
        cv2.rectangle(
            debug_img,
            (x1, label_y - text_h - baseline - 4),
            (x1 + text_w + 8, label_y + 2),
            colour,
            -1,
        )
        cv2.putText(
            debug_img,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _publish_debug(self, debug_img, stamp):
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.format = "jpeg"
        _, encoded = cv2.imencode(".jpg", debug_img)
        msg.data = encoded.tobytes()
        self.debug_image_pub.publish(msg)


def _corners(rect):
    """`(left, top, width, height)` -> `(xmin, ymin, xmax, ymax)`, as IoU wants."""
    left, top, width, height = rect
    return (left, top, left + width, top + height)


def main(args=None):
    """Run the fetch detector."""
    rclpy.init(args=args)
    node = DeepStreamObjectDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pipeline.set_state(Gst.State.NULL)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
