# Smart sensor processer nodes for the mecanumbot package

This package provides ROS 2 nodes that extract information from mecanumbot's on-board sensors.

## Available nodes

| Node                                   | Purpose                                                                                   | File                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `mecanumbot_lidar_detect_people`       | Runs DR-SPAAM on LiDAR scans to detect and track people.                                  | `mecanumbot_sensorprocess_smart/mecanumbot_lidar_detect_people.py`       |
| `mecanumbot_cam_detect_people`         | Runs YOLO pose inference on the main camera or a compressed image topic to detect people. | `mecanumbot_sensorprocess_smart/mecanumbot_cam_detect_people.py`         |
| `mecanumbot_onboard_cam_detect_people` | Runs the DeepStream-based camera people detector on NVIDIA hardware.                      | `mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_people.py` |
| `mecanumbot_locate_detections`         | Fuses camera and LiDAR detections and projects them into map space.                       | `mecanumbot_sensorprocess_smart/mecanumbot_locate_detections.py`         |
| `mecanumbot_detect_tennis`             | Detects tennis balls from the camera stream and publishes their presence state.           | `mecanumbot_sensorprocess_smart/mecanumbot_detect_tennis.py`             |

The camera detector comes in two variants: `mecanumbot_cam_detect_people` (PyTorch /
Ultralytics, portable) and `mecanumbot_onboard_cam_detect_people` (DeepStream, Jetson
only). Run one or the other — both publish `cam_people_detections`.

## Pipeline

```text
scan ──► mecanumbot_lidar_detect_people ──► dr_spaam/dets ─┐
                                                           ├─► mecanumbot_locate_detections ──► people_fusion
camera ─► mecanumbot_cam_detect_people ──► cam_people_detections ┘
```

`people_fusion` (`geometry_msgs/PoseArray`, `map` frame) is what the behaviour trees
in `mecanumbot_behaviours` consume. The LiDAR node additionally publishes
`subject_pose` directly for the leading behaviours.

## Launch file

`launch/mecanumbot_peopledetect.launch.py` starts the LiDAR people detector, the
camera people detector (with `from_topic` forced to `true`), and the
detection-localization node in the `mecanumbot` namespace, all with
`param/lidar_peopledetect_config.yaml` applied. Node names must match the YAML's
top-level keys, so do not rename them in the launch file. The DeepStream and tennis
ball nodes are not started by this launch file.

## Node: mecanumbot_lidar_detect_people

### Publishers

| Topic                                    | Data type                     | Function                                                                                                                                           |
| ---------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| dets (configurable by detections_topic)  | geometry_msgs/msg/PoseArray   | Publishes filtered and tracked people detections as 2D poses.                                                                                      |
| subject_pose                             | geometry_msgs/msg/PoseStamped | Publishes selected leading subject pose transformed into map frame (only when `leading_mode` is enabled).                                          |
| dets_marker (configurable by rviz_topic) | visualization_msgs/msg/Marker | RViz LINE_LIST circles around tracked detections. Currently disabled — the publisher is commented out and `publish_rviz` is hard-coded to `False`. |

### Subscribers

| Topic                             | Data type                  | Processing                                                                                                                                |
| --------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| scan (configurable by scan_topic) | sensor_msgs/msg/LaserScan  | Preprocesses scan, runs DR-SPAAM inference, confidence filtering, map filtering, multi-object tracking, then publishes detection outputs. |
| keepout_filter_mask               | nav_msgs/msg/OccupancyGrid | Builds an inflated obstacle mask (TRANSIENT_LOCAL QoS) used to reject detections that fall inside static obstacles.                       |

### Parameters

| Parameter                   | Default                  | Function                                                                              |
| --------------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| `weight_file`               | `dr_spaam_5_on_frog.pth` | Checkpoint name, resolved inside the package share `models/` folder.                  |
| `conf_thresh`               | `0.45`                   | Minimum DR-SPAAM class score for a detection to be kept.                              |
| `stride`                    | `1`                      | Detector stride passed to DR-SPAAM.                                                   |
| `scan_topic`                | `/mecanumbot/scan`       | Input laser scan topic.                                                               |
| `detections_topic`          | `dets`                   | Output detection topic (`dr_spaam/dets` in the shipped YAML).                         |
| `rviz_topic`                | `dets_marker`            | Marker topic name (unused while marker publishing is disabled).                       |
| `leading_mode`              | `true`                   | Enables the `subject_pose` publisher.                                                 |
| `obstacle_exclusion_radius` | `0.2`                    | Inflation radius in metres applied to the keepout mask.                               |
| `detection_frame`           | `base_scan`              | Accepts `base_scan` or `map`; anything else falls back to `base_scan` with a warning. |

### Behavior

- Loads DR-SPAAM weights from the package share `models/` folder and fails fast with
  `FileNotFoundError` if the checkpoint is missing.
- Detects CPU/GPU capability via torch and monkey-patches `torch.load` to map the
  checkpoint onto the available device.
- Preprocesses each scan: invalid/inf/NaN ranges are replaced by the max range, a
  size-3 median filter is applied, then the scan is resampled to 240 points by nearest
  index.
- Rejects detections landing inside the inflated keepout mask, using a single cached
  TF lookup per frame.
- Tracks the survivors with a Kalman filter plus Hungarian assignment; a track is only
  published once it has at least 2 hits **and** has exceeded 0.1 m/s at some point, so
  stationary false positives are suppressed.
- Uses TF from the scan frame to `map` for the `subject_pose` output, and keeps
  republishing the last known subject pose when no new one can be computed.
- Spins on a `MultiThreadedExecutor`.

### External dependency notes

- DR-SPAAM package: https://github.com/VisualComputingInstitute/DR-SPAAM-Detector
- Dataset reference: https://robotics.upo.es/datasets/frog/laser2d_people/
- Also needs `torch`, `scipy`, `filterpy` and `tf2_geometry_msgs`.

## Node: mecanumbot_cam_detect_people

### Publishers

| Topic                   | Data type                                     | Function                                                                                                        |
| ----------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `cam_people_detections` | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Publishes detected people with pose keypoints and angular bounds, stamped in the `<namespace>/head_link` frame. |

### Subscribers

| Topic                                          | Data type                                                 | Processing                                                                                      |
| ---------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `camera/image_raw/compressed` or webcam device | `sensor_msgs/msg/CompressedImage` or OpenCV video capture | Runs YOLO pose inference on each frame and converts detections into mecanumbot message format.  |
| `/amcl_pose`                                   | `geometry_msgs/msg/PoseWithCovarianceStamped`             | Updates robot orientation so camera-side angle bounds are expressed relative to the robot pose. |

### Parameters

| Parameter                        | Default                       | Function                                                                     |
| -------------------------------- | ----------------------------- | ---------------------------------------------------------------------------- |
| `camera_params.camera_width`     | `640.0`                       | Frame width requested from the webcam.                                       |
| `camera_params.camera_height`    | `480.0`                       | Frame height requested from the webcam.                                      |
| `camera_params.camera_fov`       | `60°` (in radians)            | Horizontal field of view used to convert normalized x into a yaw angle.      |
| `from_topic`                     | `false`                       | `true` subscribes to `camera_topic`, `false` opens `webcam_device` directly. |
| `camera_topic`                   | `camera/image_raw/compressed` | Compressed image input topic.                                                |
| `webcam_device`                  | `/dev/video0`                 | V4L2 device used in webcam mode.                                             |
| `img_process_params.weight_file` | `yolo26n-pose.pt`             | Pose model, resolved inside the package share `models/` folder.              |

### Behavior

- Supports either a compressed ROS image topic or a local webcam (webcam mode reads
  frames from a 15 Hz timer).
- Loads the pose model from the package share `models/` directory and moves it to CUDA
  when available.
- Converts 17-keypoint YOLO pose output into `CamPersonDetection` messages; detections
  that do not yield exactly 17 keypoints are skipped.
- Left/right angular bounds are computed from the minimum and maximum keypoint x, and
  are only filled once an AMCL pose has been received. In topic mode no frame is
  processed at all until `/amcl_pose` arrives.

## Node: mecanumbot_onboard_cam_detect_people

ROS node name: `mecanumbot_cam_detect_people_ds`.

### Publishers

| Topic                                          | Data type                                     | Function                                                                                              |
| ---------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `cam_people_detections`                        | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Publishes pose-based people detections from the DeepStream pipeline.                                  |
| `cam_people_detections/debug_image/compressed` | `sensor_msgs/msg/CompressedImage`             | Annotated debug image (boxes, skeleton, per-keypoint confidences), only when `debug_mode` is enabled. |

### Subscribers

| Topic                                          | Data type                                              | Processing                                                    |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------- |
| `camera/image_raw/compressed` or webcam device | `sensor_msgs/msg/CompressedImage` or V4L2 camera input | Feeds frames into the DeepStream pipeline for pose inference. |

### Parameters

| Parameter                     | Default                       | Function                                                                              |
| ----------------------------- | ----------------------------- | ------------------------------------------------------------------------------------- |
| `camera_params.camera_width`  | `1280`                        | Pipeline and streammux width.                                                         |
| `camera_params.camera_height` | `720`                         | Pipeline and streammux height.                                                        |
| `camera_params.camera_fov`    | `60°` (in radians)            | Horizontal field of view used for the angular bounds.                                 |
| `from_topic`                  | `false`                       | `true` pushes ROS frames into an `appsrc`, `false` uses `v4l2src` on `webcam_device`. |
| `camera_topic`                | `camera/image_raw/compressed` | Compressed image input topic.                                                         |
| `webcam_device`               | `/dev/video0`                 | V4L2 device used in webcam mode.                                                      |
| `debug_mode`                  | `false`                       | Enables the annotated debug image publisher.                                          |

Confidence handling is hard-coded rather than parameterized: an object is published
when its overall detection confidence exceeds `0.3`, and individual keypoints below
`min_conf_threshold` (`0.15`) are emitted as `NaN` positions and excluded from the
angular bounds.

### Behavior

- Uses GStreamer and NVIDIA DeepStream instead of the pure PyTorch/OpenCV path:
  `source → nvvideoconvert → nvstreammux → nvinfer → nvvideoconvert → capsfilter(RGBA) → fakesink`,
  with a buffer probe on the capsfilter reading the inference metadata.
- Supports either a ROS image topic or direct webcam input.
- Extracts pose keypoints from NVIDIA metadata, removes the letterbox padding, and
  normalizes them to `[0, 1]` before mapping into `CamPersonDetection` messages.
- Unmaps the NvDs buffer surface after every frame to avoid leaking memory on Jetson.

### DeepStream configuration

`deepstream_config/config_infer_yolo26_pose.txt` points `nvinfer` at
`models/yolo26n-pose.onnx` and its FP16 engine, and at the `DeepStream-Yolo-Pose`
custom parser library. These are **absolute paths under `/home/ubuntu/`** — they have
to be edited to match the deployment machine before this node will start.

## Node: mecanumbot_locate_detections

### Publishers

| Topic                             | Data type                     | Function                                                                                    |
| --------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| `people_fusion`                   | `geometry_msgs/msg/PoseArray` | Publishes fused detections in map space.                                                    |
| `cam_people_detections/left_FOV`  | `geometry_msgs/msg/PoseArray` | Left field-of-view bound of each camera detection. Only created when `debug_mode` is true.  |
| `cam_people_detections/right_FOV` | `geometry_msgs/msg/PoseArray` | Right field-of-view bound of each camera detection. Only created when `debug_mode` is true. |

### Subscribers

| Topic                   | Data type                                     | Processing                                                             |
| ----------------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| `cam_people_detections` | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Receives camera detections for fusion (own callback group).            |
| `dets`                  | `geometry_msgs/msg/PoseArray`                 | Receives LiDAR people detections; each message triggers a fusion pass. |
| `scan`                  | `sensor_msgs/msg/LaserScan`                   | Stores the current scan for range extrapolation.                       |
| `/map`                  | `nav_msgs/msg/OccupancyGrid`                  | Loads the static map grid (TRANSIENT_LOCAL QoS).                       |
| `/amcl_pose`            | `geometry_msgs/msg/PoseWithCovarianceStamped` | Tracks the robot pose in map coordinates.                              |

### Parameters

| Parameter           | Default | Function                                                                         |
| ------------------- | ------- | -------------------------------------------------------------------------------- |
| `obstacle_buffer_x` | `0.5`   | Metres added behind a wall when a detection has to be pushed out of an obstacle. |
| `debug_mode`        | `false` | Enables the left/right FOV publishers.                                           |

### Behavior

- For every camera detection, resolves a range in three steps: first look for a LiDAR
  detection whose bearing falls inside the person's angular bounds; if there is none,
  extrapolate from the raw scan using the 20th percentile of the valid ranges inside
  the bounds (so background hits do not dominate); if that also fails, drop the
  detection.
- Validates the result against the static map: a pose landing on an occupied cell is
  ray-traced outward until free space is found (up to 4 m of wall thickness) and then
  offset by `obstacle_buffer_x`.
- Transforms the accepted poses from `mecanumbot/base_link` into `map` and publishes
  them as a single `PoseArray`.
- Re-publishes the previous fused array when the camera timestamp has not changed, so
  downstream consumers keep seeing the last known people.
- Runs on a 4-thread `MultiThreadedExecutor`.

## Node: mecanumbot_detect_tennis

ROS node name: `mecanumbot_cam_detect_tennis`.

### Publishers

| Topic              | Data type            | Function                                                             |
| ------------------ | -------------------- | -------------------------------------------------------------------- |
| `tennis_ball_info` | `std_msgs/msg/Int32` | Publishes the number of seconds since the last tennis ball was seen. |

### Subscribers

| Topic                         | Data type                         | Processing                                             |
| ----------------------------- | --------------------------------- | ------------------------------------------------------ |
| `camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Feeds frames into the YOLO-based tennis ball detector. |

### Behavior

- Runs YOLO inference at 320 px on camera frames and keeps detections of COCO class 32
  (`sports ball`) above 0.5 confidence.
- Uses a single-worker thread pool and a busy flag so image processing cannot build up
  a backlog; frames arriving while one is in flight are dropped.
- Tracks the time since the last positive tennis-ball detection and publishes it as an
  integer.
- Configuration is hard-coded — this node declares no ROS parameters. It also loads
  `yolov8n.pt`, which is **not** among the shipped model files; drop that checkpoint
  into `models/` before running it.

## File functions

| File or folder                                                         | Function                                                             |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| mecanumbot_sensorprocess_smart/mecanumbot_lidar_detect_people.py       | Main detection node and tracking pipeline.                           |
| mecanumbot_sensorprocess_smart/mecanumbot_cam_detect_people.py         | Main camera people detection node.                                   |
| mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_people.py | DeepStream-based camera people detection node.                       |
| mecanumbot_sensorprocess_smart/mecanumbot_locate_detections.py         | Detection fusion and localization node.                              |
| mecanumbot_sensorprocess_smart/mecanumbot_detect_tennis.py             | Tennis ball detection node.                                          |
| launch/mecanumbot_peopledetect.launch.py                               | Launches the people-detection pipeline with shared parameters.       |
| param/lidar_peopledetect_config.yaml                                   | Runtime ROS parameters for node topics and thresholds.               |
| config/lidar_peopledetect_config.yaml                                  | Identical copy of the param file, kept for deployment compatibility. |
| models/dr_spaam_5_on_frog.pth                                          | DR-SPAAM pretrained weights used by the LiDAR detector.              |
| models/dr_spaam.onnx                                                   | ONNX export of the DR-SPAAM model.                                   |
| models/yolo26n-pose.pt                                                 | YOLO pose weights used by the Ultralytics camera detector.           |
| models/yolo26n-pose.onnx, models/yolo26n-pose.onnx_b1_gpu0_fp16.engine | ONNX and TensorRT FP16 build used by the DeepStream detector.        |
| deepstream_config/config_infer_yolo26_pose.txt                         | `nvinfer` configuration for the DeepStream pose model.               |
| deepstream_config/labels.txt                                           | Class label file referenced by the `nvinfer` config.                 |

Only the LiDAR node's parameters live in the YAML file; the camera, fusion and tennis
nodes rely on their in-code defaults unless overridden on the command line or in the
launch file.

## Build and run

```bash
colcon build --symlink-install --packages-select mecanumbot_sensorprocess_smart
source install/setup.bash

# whole people-detection pipeline
ros2 launch mecanumbot_sensorprocess_smart mecanumbot_peopledetect.launch.py

# individual nodes
ros2 run mecanumbot_sensorprocess_smart mecanumbot_onboard_cam_detect_people
ros2 run mecanumbot_sensorprocess_smart mecanumbot_detect_tennis
```

Python dependencies (`torch`, `ultralytics`, `opencv-python`, `scipy`, `filterpy`,
`transforms3d`, `dr_spaam`, and `pyds`/`gi` for the DeepStream node) are not declared
in `package.xml`, so `rosdep` will not install them for you.
