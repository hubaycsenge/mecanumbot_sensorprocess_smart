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

## `use_camera`: who owns the camera

The camera can only be opened once, so this is a choice and not a flag.

* **false** (the default) -- the DeepStream detector opens the camera itself
  through `nvarguscamerasrc`. Cheapest path: no JPEG encode, no decode, no
  topic. But nothing else can have the camera, so **there is no
  `/camera/image_raw/compressed`** for a recording, the web GUI or an operator
  to look at.
* **true** -- `mecanumbot_camera_stream`'s compressed publisher owns the camera
  and the detector subscribes to its topic. That costs a JPEG encode on the
  publisher and a decode in the detector, and it is what the leading experiment
  runs with, because a trial that is not recorded from the robot's own point of
  view is a trial that cannot be scored afterwards.

## One frame size, three nodes

`camera_width` / `camera_height` go to the camera publisher, to the detector and
to the fusion node together. All three had their own copy before, and nothing
compared them -- but every bearing and every apparent-size range is computed
from the frame size, so a disagreement is not a warning, it is silently wrong
numbers. The default is 1280x720, which is what the DeepStream detectors were
already configured for; the camera publisher's own config file says 640x480 and
is overridden here.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
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

CAMERA_LAUNCH = os.path.join(
    get_package_share_directory("mecanumbot_camera_stream"),
    "launch",
    "camera_compressed.launch.py",
)


def generate_launch_description():
    """Build the launch description for the perception pipeline."""
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    detector = LaunchConfiguration("detector")
    use_camera = LaunchConfiguration("use_camera")
    camera_topic = LaunchConfiguration("camera_topic")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")

    run_pose = IfCondition(PythonExpression(["'", detector, "' == 'pose'"]))
    run_fetch = IfCondition(PythonExpression(["'", detector, "' == 'fetch'"]))

    # The detectors read the frame off a topic only when something else owns the
    # camera, which is exactly when `use_camera` is true.
    from_topic = ParameterValue(use_camera, value_type=bool)
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
                "use_camera",
                default_value="false",
                description=(
                    "Publish /camera/image_raw/compressed and feed the detector "
                    "from it, instead of letting the detector open the camera "
                    "directly. The camera can only be opened once, so this is "
                    "the choice between having the stream and not paying for it"
                ),
            ),
            DeclareLaunchArgument(
                "camera_topic",
                default_value="/camera/image_raw/compressed",
                description=(
                    "Where the compressed frames are published and read. "
                    "Absolute on purpose: the publisher is not namespaced and "
                    "the detectors are, so a relative name would be looked for "
                    "under /<namespace>/ and never found"
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
                    "  use_camera=",
                    use_camera,
                    "  frame=",
                    camera_width,
                    "x",
                    camera_height,
                ]
            ),
            # The camera, when something other than the detector is to own it.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(CAMERA_LAUNCH),
                condition=IfCondition(use_camera),
                launch_arguments={
                    "topic_name": camera_topic,
                    "width": camera_width,
                    "height": camera_height,
                    "fps": LaunchConfiguration("camera_fps"),
                    "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                }.items(),
            ),
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
