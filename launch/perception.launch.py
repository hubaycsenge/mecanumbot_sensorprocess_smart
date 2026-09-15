"""
The perception pipeline, started by whatever behaviour actually needs it.

This used to be part of `mecanumbot_bringup`'s `launch_mecanumbot_base.launch.py`,
so every run of the robot -- a teleop session, a mapping run, T1 exploration --
carried a LiDAR person detector, a DeepStream network and a fusion node whether
or not anything subscribed to them. On an Orin Nano that is not free: the
network is most of the GPU, and it holds the camera open so nothing else can
have it.

So it lives here, in the package that owns the nodes, and each behaviour launch
includes it with the detector that behaviour needs. The base launch starts
drivers, nav2 and the GUI, and nothing that looks at people.

It could not live in `mecanumbot_bringup`, which is otherwise where launch
orchestration goes: `mecanumbot_bringup` depends on `mecanumbot_web`, which
depends on `mecanumbot_leading_behaviour` (it starts trees from the browser), so
a behaviour package that depended on `mecanumbot_bringup` would close a
dependency cycle. Depending on the perception package instead does not.

## The three things it starts

* `mecanumbot_lidar_detect_people` -- DR-SPAAM on the scan, publishing
  `dr_spaam/dets` and `subject_pose`.
* one camera detector, chosen by `detector`:
  - `pose` -- `mecanumbot_onboard_cam_detect_people`, skeletons on
    `cam_people_detections`. What the leading and ostensive experiments need.
  - `fetch` -- `mecanumbot_onboard_cam_detect_objects`, people **and tennis
    balls** as boxes. The only detector that can see a ball; it has no
    keypoints, so the ostensive gestures are unavailable while it runs.
  - `none` -- LiDAR only, for a bench run with no camera.
  Never both: a pose network has one class, so the ball is a different network
  rather than a different threshold, and two networks on one camera stream is
  most of the GPU.
* `mecanumbot_locate_detections` -- the fusion, publishing `people_fusion` and,
  with the fetch detector, `ball_fusion` / `ball_detections`.

## `camera_source`: how frames reach the network

**`direct` is the default, and the intended way to run on the robot.** The
camera can only be opened once, so this is a choice between two paths:

* **direct** -- the DeepStream detector opens the USB webcam itself, inside its
  own GStreamer pipeline (`v4l2src` on `/dev/video0` -> nvvideoconvert ->
  nvinfer). **No ROS 2 middleware carries a frame**: no camera node, no JPEG
  encode, no DDS transport, no decode. It is the cheapest path there is, and
  why it is the default. The cost is that nothing else can open the camera, so
  there is no `/camera/image_raw/compressed`; the only picture on the ROS graph
  is the detector's own `debug_image` (below).
* **topic** -- the detector subscribes to `camera_topic` and pushes each decoded
  frame into an `appsrc`. That costs a whole camera node, a JPEG encode, the
  transport and a decode, and exists only for when something else needs the raw
  stream too (a clean recording, the Deep3R client in T2).
  **It does not start the camera**: nothing in this file or any behaviour launch
  publishes `camera_topic` (the include was removed in 2f7aade), so start it
  first or the detector never gets a frame and stays silent:

      ros2 launch mecanumbot_camera_stream camera_compressed.launch.py width:=1280 height:=720

  `camera_fps` and `jpeg_quality` are still declared but went only to that
  include.

Any other value stops the launch, and so does the old `use_camera` argument
this replaced (2026-09-15): `use_camera:=false` read as "no camera" when it
meant "open the camera directly", and a stray `use_camera:=true` would
otherwise be silently ignored.

## One frame size, three nodes

`camera_width` / `camera_height` go to the camera publisher, to the detector and
to the fusion node together. All three had their own copy before, and nothing
compared them -- but every bearing and every apparent-size range is computed
from the frame size, so a disagreement is not a warning, it is silently wrong
numbers. The default is 1280x720, which is what the DeepStream detectors were
already configured for; the camera publisher's own config file says 640x480 and
is overridden here.

## `debug_image`: what the detector saw

Sets `debug_mode` on whichever camera detector runs, which publishes the frame
the network was given with every box drawn on it -- accepted ones in colour,
refused ones in red with the check they failed:

* `pose`  -> `/<namespace>/cam_people_detections/debug_image/compressed`
* `fetch` -> `/<namespace>/cam_object_detections/debug_image/compressed`

It is the only way to see the camera with `camera_source:=direct`, and it is
**on by default**, so that what the robot saw during a behaviour can be
surveyed afterwards from a recording. It costs a copy of the frame out of GPU
memory and a JPEG encode per frame; `debug_image:=false` saves both.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Every detector and the fusion node take their parameters from this one file,
# each under the node name it registers as. Renaming a node in a launch file
# silently drops every parameter in it.
PERCEPTION_YAML = os.path.join(
    get_package_share_directory("mecanumbot_sensorprocess_smart"),
    "config",
    "lidar_peopledetect_config.yaml",
)

'''
CAMERA_LAUNCH = os.path.join(
    get_package_share_directory("mecanumbot_camera_stream"),
    "launch",
    "camera_compressed.launch.py",
)
'''

# `direct`: the detector's GStreamer pipeline opens the webcam (v4l2src), no ROS
# in the frame path -- the default. `topic`: frames come over ROS from a camera
# node somebody else started.
CAMERA_SOURCES = ("direct", "topic")


def _check_camera_source(context):
    """Refuse an unknown `camera_source`, or the `use_camera` it replaced, and say which path runs."""
    if "use_camera" in context.launch_configurations:
        raise RuntimeError(
            "use_camera was replaced by camera_source: use_camera:=false is now "
            "camera_source:=direct (the default), use_camera:=true is "
            "camera_source:=topic."
        )
    source = context.launch_configurations.get("camera_source", "")
    if source not in CAMERA_SOURCES:
        raise RuntimeError(
            f"camera_source:={source!r} is not one of {', '.join(CAMERA_SOURCES)}."
        )
    if source == "direct":
        message = (
            "[perception] camera_source=direct: the detector opens the webcam "
            "itself (v4l2src), no ROS image topic in the frame path"
        )
    else:
        message = (
            "[perception] camera_source=topic: the detector reads "
            f"{context.launch_configurations.get('camera_topic')} -- this launch "
            "does NOT start the camera; run camera_compressed.launch.py or the "
            "detector gets no frames"
        )
    return [LogInfo(msg=message)]


def generate_launch_description():
    """Build the launch description for the perception pipeline."""
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    detector = LaunchConfiguration("detector")
    camera_source = LaunchConfiguration("camera_source")
    camera_topic = LaunchConfiguration("camera_topic")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")

    run_pose = IfCondition(PythonExpression(["'", detector, "' == 'pose'"]))
    run_fetch = IfCondition(PythonExpression(["'", detector, "' == 'fetch'"]))

    # The node's `from_topic` is true only for `camera_source:=topic`; with the
    # default `direct` it builds its pipeline on v4l2src instead of an appsrc.
    from_topic = ParameterValue(
        PythonExpression(["'", camera_source, "' == 'topic'"]), value_type=bool
    )
    debug_mode = ParameterValue(LaunchConfiguration("debug_image"), value_type=bool)
    width = ParameterValue(camera_width, value_type=int)
    height = ParameterValue(camera_height, value_type=int)

    camera_params = {
        "camera_params.camera_width": width,
        "camera_params.camera_height": height,
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="mecanumbot",
                description="Namespace the perception nodes run in",
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="false", description="Use the sim clock"
            ),
            DeclareLaunchArgument(
                "detector",
                default_value="pose",
                description=(
                    "Which camera detector to run: pose (skeletons, for leading "
                    "and ostensive) | fetch (people and balls, for the fetch "
                    "game) | none (LiDAR only)"
                ),
            ),
            DeclareLaunchArgument(
                "use_lidar_people",
                default_value="true",
                description="Run DR-SPAAM on the scan",
            ),
            DeclareLaunchArgument(
                "camera_source",
                default_value="direct",
                choices=list(CAMERA_SOURCES),
                description=(
                    "direct (default): the detector opens the webcam itself "
                    "with v4l2src, no ROS 2 middleware in the frame path, "
                    "cheapest. topic: read camera_topic instead -- does NOT "
                    "start the camera, run camera_compressed.launch.py first"
                ),
            ),
            DeclareLaunchArgument(
                "camera_topic",
                default_value="/camera/image_raw/compressed",
                description=(
                    "Only with camera_source:=topic: where the compressed "
                    "frames are published and read. "
                    "Absolute on purpose: the publisher is not namespaced and "
                    "the detectors are, so a relative name would be looked for "
                    "under /<namespace>/ and never found"
                ),
            ),
            DeclareLaunchArgument(
                "debug_image",
                default_value="true",
                description=(
                    "Publish the camera detector's annotated frame on "
                    "<detector topic>/debug_image/compressed. Costs a frame copy "
                    "and a JPEG encode per frame"
                ),
            ),
            DeclareLaunchArgument(
                "camera_width",
                default_value="1280",
                description="Frame width, for the camera AND everything that "
                "turns a pixel into an angle. Must be a mode the camera has",
            ),
            DeclareLaunchArgument(
                "camera_height", default_value="720", description="Frame height"
            ),
            DeclareLaunchArgument(
                "camera_fps", default_value="15.0", description="Frames per second"
            ),
            DeclareLaunchArgument(
                "jpeg_quality",
                default_value="80",
                description="JPEG quality of the published stream",
            ),
            DeclareLaunchArgument(
                "yolo_imgsz",
                default_value="1280",
                description=(
                    "Input size the pose model expects. Selects "
                    "mecanumbot_sensorprocess_smart models/imgsz_<n>/, so it has "
                    "to be a size the model was exported at (640 and 1280 ship)"
                ),
            ),
            DeclareLaunchArgument(
                "yolo_model",
                default_value="yolo26m-pose",
                description="Pose model stem inside models/imgsz_<yolo_imgsz>/",
            ),
            DeclareLaunchArgument(
                "fetch_imgsz",
                default_value="640",
                description="Input size the fetch detector expects",
            ),
            DeclareLaunchArgument(
                "fetch_model",
                default_value="yolo26m",
                description="Detection model stem inside models/imgsz_<fetch_imgsz>/",
            ),
            LogInfo(
                msg=[
                    "[perception] detector=",
                    detector,
                    "  camera_source=",
                    camera_source,
                    "  frame=",
                    camera_width,
                    "x",
                    camera_height,
                ]
            ),
            OpaqueFunction(function=_check_camera_source),

            Node(
                namespace=namespace,
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_lidar_detect_people",
                name="mecanumbot_lidar_detect_people",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_lidar_people")),
                parameters=[PERCEPTION_YAML, {"use_sim_time": use_sim_time}],
                remappings=[("map", "/map")],
            ),
            Node(
                namespace=namespace,
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_onboard_cam_detect_people",
                # Must match the YAML top-level key, which is the name the node
                # registers for itself -- naming it after the executable instead
                # silently dropped every parameter in the file.
                name="mecanumbot_cam_detect_people_ds",
                output="screen",
                condition=run_pose,
                parameters=[
                    PERCEPTION_YAML,
                    camera_params,
                    {
                        "use_sim_time": use_sim_time,
                        "from_topic": from_topic,
                        "camera_topic": camera_topic,
                        "debug_mode": debug_mode,
                        # The ONNX exports live one folder per input size; this
                        # picks the folder, and the node rewrites infer-dims and
                        # the model/engine paths in the nvinfer config to match.
                        "model_params.imgsz": ParameterValue(
                            LaunchConfiguration("yolo_imgsz"), value_type=int
                        ),
                        "model_params.model_name": LaunchConfiguration("yolo_model"),
                    },
                ],
            ),
            Node(
                namespace=namespace,
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_onboard_cam_detect_objects",
                name="mecanumbot_cam_detect_objects_ds",
                output="screen",
                condition=run_fetch,
                parameters=[
                    PERCEPTION_YAML,
                    camera_params,
                    {
                        "use_sim_time": use_sim_time,
                        "from_topic": from_topic,
                        "camera_topic": camera_topic,
                        "debug_mode": debug_mode,
                        "model_params.imgsz": ParameterValue(
                            LaunchConfiguration("fetch_imgsz"), value_type=int
                        ),
                        "model_params.model_name": LaunchConfiguration("fetch_model"),
                    },
                ],
            ),
            # The fusion needs the same frame size as the detector: with the
            # fetch detector it works the ball's range out from the apparent
            # diameter of its box, and with either it turns a person's box or
            # bearing into an angle.
            Node(
                namespace=namespace,
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_locate_detections",
                name="mecanumbot_locate_detections",
                output="screen",
                parameters=[
                    PERCEPTION_YAML,
                    camera_params,
                    {"use_sim_time": use_sim_time},
                ],
            ),
        ]
    )
