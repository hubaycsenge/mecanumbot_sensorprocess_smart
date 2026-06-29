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

## Node: mecanumbot_cam_detect_people

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `cam_people_detections` | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Publishes detected people with pose keypoints and angular bounds. |

### Subscribers

| Topic | Data type | Processing |
| --- | --- | --- |
| `camera/image_raw/compressed` or webcam device | `sensor_msgs/msg/CompressedImage` or OpenCV video capture | Runs YOLO pose inference on each frame and converts detections into mecanumbot message format. |
| `/amcl_pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | Updates robot orientation so camera-side angle bounds are expressed relative to the robot pose. |

### Behavior

- Supports either a compressed ROS image topic or a local webcam.
- Loads the pose model from the package share `models/` directory.
- Converts 17-keypoint YOLO pose output into `CamPersonDetection` messages.
- Publishes the result in the camera frame with left/right angular bounds when AMCL pose is available.

## Node: mecanumbot_onboard_cam_detect_people

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `cam_people_detections` | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Publishes pose-based people detections from the DeepStream pipeline. |
| `cam_people_detections/debug_image/compressed` | `sensor_msgs/msg/CompressedImage` | Publishes an annotated debug image when `debug_mode` is enabled. |

### Subscribers

| Topic | Data type | Processing |
| --- | --- | --- |
| `camera/image_raw/compressed` or webcam device | `sensor_msgs/msg/CompressedImage` or V4L2 camera input | Feeds frames into the DeepStream pipeline for pose inference. |

### Behavior

- Uses GStreamer and NVIDIA DeepStream instead of the pure PyTorch/OpenCV path.
- Supports either a ROS image topic or direct webcam input.
- Extracts pose keypoints from NVIDIA metadata and maps them into `CamPersonDetection` messages.
- Can publish a debug overlay image for inspection when `debug_mode` is set.

## Node: mecanumbot_locate_detections

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `people_fusion` | `geometry_msgs/msg/PoseArray` | Publishes fused detections in map space. |
| `cam_people_detections/left_FOV` | `geometry_msgs/msg/PoseArray` | Publishes the camera detections that fall into the left field of view. |
| `cam_people_detections/right_FOV` | `geometry_msgs/msg/PoseArray` | Publishes the camera detections that fall into the right field of view. |

### Subscribers

| Topic | Data type | Processing |
| --- | --- | --- |
| `cam_people_detections` | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Receives camera detections for fusion. |
| `dets` | `geometry_msgs/msg/PoseArray` | Receives LiDAR people detections. |
| `scan` | `sensor_msgs/msg/LaserScan` | Stores the current scan for spatial reasoning. |
| `/map` | `nav_msgs/msg/OccupancyGrid` | Loads the static map grid. |
| `/amcl_pose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | Tracks the robot pose in map coordinates. |

### Behavior

- Fuses camera and LiDAR detections into map coordinates.
- Uses TF, AMCL pose, and the occupancy grid to reason about relative position.
- Publishes left/right camera field-of-view splits for downstream consumers.
- Keeps an obstacle buffer parameter for map-based filtering.

## Node: mecanumbot_detect_tennis

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `tennis_ball_info` | `std_msgs/msg/Int32` | Publishes the number of seconds since the last tennis ball was seen. |

### Subscribers

| Topic | Data type | Processing |
| --- | --- | --- |
| `camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Feeds frames into the YOLO-based tennis ball detector. |

### Behavior

- Runs YOLO inference on camera frames and filters detections for the sports ball class.
- Uses a single-worker thread pool so image processing does not build a backlog.
- Tracks the time since the last positive tennis-ball detection and publishes it as an integer.

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
