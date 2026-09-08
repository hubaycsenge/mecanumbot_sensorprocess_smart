"""
The perception pipeline on its own: LiDAR people, camera people, and the fusion.

`detector` picks which camera detector runs, and the choice is a real one rather
than a preference:

* ``pose`` -- the Ultralytics pose node (`mecanumbot_cam_detect_people`),
  reporting skeletons. What the leading and ostensive experiments need.
* ``fetch`` -- the DeepStream detector (`mecanumbot_onboard_cam_detect_objects`),
  reporting people *and* tennis balls as boxes. A pose network has one class, so
  the ball is not a threshold away from the pose model -- it is a different
  network, and running both at once is most of an Orin Nano's GPU.
* ``both`` -- for a bench comparison. `mecanumbot_locate_detections` accepts
  person evidence from either, so the two wedges land on the same person and the
  tracker's association absorbs the duplicate.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

yaml_file = os.path.join(
    get_package_share_directory("mecanumbot_sensorprocess_smart"),
    "config",
    "lidar_peopledetect_config.yaml",
)


def generate_launch_description():
    """Bring up the people (and, with the fetch detector, ball) pipeline."""
    detector = LaunchConfiguration("detector")
    run_pose = PythonExpression(["'", detector, "' in ('pose', 'both')"])
    run_fetch = PythonExpression(["'", detector, "' in ('fetch', 'both')"])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "detector",
                default_value="pose",
                description="Which camera detector to run: pose | fetch | both",
            ),
            Node(
                namespace="mecanumbot",
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_lidar_detect_people",
                name="mecanumbot_lidar_detect_people",  # must match YAML top-level key
                output="screen",
                parameters=[yaml_file],
            ),
            Node(
                namespace="mecanumbot",
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_cam_detect_people",
                name="mecanumbot_cam_detect_people",  # must match YAML top-level key
                output="screen",
                condition=IfCondition(run_pose),
                parameters=[yaml_file, {"from_topic": True}],
            ),
            Node(
                namespace="mecanumbot",
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_onboard_cam_detect_objects",
                name="mecanumbot_cam_detect_objects_ds",  # must match YAML key
                output="screen",
                condition=IfCondition(run_fetch),
                parameters=[yaml_file, {"from_topic": True}],
            ),
            Node(
                namespace="mecanumbot",
                package="mecanumbot_sensorprocess_smart",
                executable="mecanumbot_locate_detections",
                name="mecanumbot_locate_detections",  # must match YAML top-level key
                output="screen",
                parameters=[yaml_file],
            ),
        ]
    )
