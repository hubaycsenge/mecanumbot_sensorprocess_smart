import os
from setuptools import find_packages, setup
from glob import glob

package_name = "mecanumbot_sensorprocess_smart"


def share_files(pattern):
    """Return only the regular files matching pattern (skips __pycache__ etc.)."""
    return [path for path in glob(pattern) if os.path.isfile(path)]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", share_files("config/*")),
        ("share/" + package_name + "/models", share_files("models/*")),
        # One folder per exported input size; mecanumbot_onboard_cam_detect_people
        # picks between them with model_params.imgsz.
        *[
            ("share/" + package_name + "/models/" + os.path.basename(directory),
             share_files(os.path.join(directory, "*")))
            for directory in sorted(glob("models/imgsz_*"))
            if os.path.isdir(directory)
        ],
        ("share/" + package_name + "/launch", share_files("launch/*")),
        (
            "share/" + package_name + "/deepstream_config",
            share_files("deepstream_config/*"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Csenge Hubay",
    maintainer_email="csengehubay@gmail.com",
    description="Functions for extracting and processing data from mecanumbot sensors",
    license="Apache License 2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "mecanumbot_lidar_detect_people = mecanumbot_sensorprocess_smart.mecanumbot_lidar_detect_people:main",
            "mecanumbot_cam_detect_people = mecanumbot_sensorprocess_smart.mecanumbot_cam_detect_people:main",
            "mecanumbot_onboard_cam_detect_people = mecanumbot_sensorprocess_smart.mecanumbot_onboard_cam_detect_people:main",
            "mecanumbot_locate_detections = mecanumbot_sensorprocess_smart.mecanumbot_locate_detections:main",
            "mecanumbot_detect_tennis = mecanumbot_sensorprocess_smart.mecanumbot_detect_tennis:main",
        ],
    },
)
