# Smart sensor processer nodes for the mecanumbot package

This package provides ROS 2 nodes that extract information from mecanumbot's on-board sensors.

## Available nodes

| Node | Purpose | File |
| --- | --- | --- |
| `mecanumbot_lidar_detect_people` | Runs DR-SPAAM on LiDAR scans to detect and track people. | `mecanumbot_sensorprocess_smart/mecanumbot_lidar_detect_people.py` |
| `mecanumbot_cam_detect_people` | Runs YOLO pose inference on the main camera or a compressed image topic to detect people. | `mecanumbot_sensorprocess_smart/mecanumbot_cam_detect_people.py` |
| `mecanumbot_onboard_cam_detect_people` | Runs the DeepStream-based camera people detector on NVIDIA hardware. | `mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_people.py` |
| `mecanumbot_locate_detections` | Fuses camera and LiDAR detections and projects them into map space. | `mecanumbot_sensorprocess_smart/mecanumbot_locate_detections.py` |
| `mecanumbot_detect_tennis` | Detects tennis balls from the camera stream and publishes their presence state. | `mecanumbot_sensorprocess_smart/mecanumbot_detect_tennis.py` |

## Launch file

`launch/mecanumbot_peopledetect.launch.py` starts the LiDAR people detector, the camera people detector, and the detection-localization node in the `mecanumbot` namespace.

## Node: mecanumbot_lidar_detect_people

### Publishers

| Topic                                    | Data type                     | Function                                                                                           |
| ---------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------- |
| dets (configurable by detections_topic)  | geometry_msgs/msg/PoseArray   | Publishes filtered and tracked people detections as 2D poses.                                      |
| dets_marker (configurable by rviz_topic) | visualization_msgs/msg/Marker | Publishes RViz LINE_LIST circles around tracked detections.                                        |
| subject_pose                             | geometry_msgs/msg/PoseStamped | Publishes selected leading subject pose transformed into map frame (when leading mode is enabled). |

### Subscribers

| Topic | Data type                 | Processing                                                                                                                 |
| ----- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| scan  | sensor_msgs/msg/LaserScan | Preprocesses scan, runs DR-SPAAM inference, confidence filtering, multi-object tracking, then publishes detection outputs. |

### Behavior

- Loads DR-SPAAM weights from package share models folder.
- Detects CPU/GPU capability via torch and maps model loading accordingly.
- Uses nearest-neighbor interpolation and median filtering for robust scan preprocessing.
- Uses a Kalman filter + Hungarian assignment tracker to suppress one-frame noise and keep stable tracked targets.
- Uses TF lookup and pose transform from mecanumbot/base_scan to map for subject_pose output.

### External dependency notes

- DR-SPAAM package: https://github.com/VisualComputingInstitute/DR-SPAAM-Detector
- Dataset reference: https://robotics.upo.es/datasets/frog/laser2d_people/

## File functions

| File or folder                                                   | Function                                                      |
| ---------------------------------------------------------------- | ------------------------------------------------------------- |
| mecanumbot_sensorprocess_smart/mecanumbot_lidar_detect_people.py | Main detection node and tracking pipeline.                    |
| mecanumbot_sensorprocess_smart/mecanumbot_cam_detect_people.py   | Main camera people detection node.                            |
| mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_people.py | DeepStream-based camera people detection node.         |
| mecanumbot_sensorprocess_smart/mecanumbot_locate_detections.py   | Detection fusion and localization node.                        |
| mecanumbot_sensorprocess_smart/mecanumbot_detect_tennis.py       | Tennis ball detection node.                                    |
| launch/mecanumbot_peopledetect.launch.py                         | Launches the people-detection pipeline with shared parameters. |
| param/lidar_peopledetect_config.yaml                             | Runtime ROS parameters for node topics and thresholds.        |
| models/dr_spaam_5_on_frog.pth                                    | DR-SPAAM pretrained weights file used for inference.          |
| config/lidar_peopledetect_config.yaml                            | Additional packaged config copy for deployment compatibility. |
