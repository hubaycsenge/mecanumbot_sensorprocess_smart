"""
The perception pipeline on its own, for a run with no behaviour tree.

A thin wrapper over `perception.launch.py`, which is where all of it actually
lives and which the behaviour launch files include. This name is kept because it
is the one in every README and in a good deal of muscle memory, and because
"start the people detection" is a reasonable thing to want on its own -- a bench
check, an rviz session, a recording with no tree running.

Every argument is passed straight through; see `perception.launch.py` for what
they mean and, in particular, for what `use_camera` is choosing between.

    ros2 launch mecanumbot_sensorprocess_smart mecanumbot_peopledetect.launch.py
    ros2 launch mecanumbot_sensorprocess_smart mecanumbot_peopledetect.launch.py \
        detector:=fetch use_camera:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

PERCEPTION_LAUNCH = os.path.join(
    get_package_share_directory("mecanumbot_sensorprocess_smart"),
    "launch",
    "perception.launch.py",
)


def generate_launch_description():
    """Include the perception pipeline, passing every argument through."""
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PERCEPTION_LAUNCH),
            ),
        ]
    )
