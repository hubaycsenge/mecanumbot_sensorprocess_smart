# Smart sensor processer nodes for the mecanumbot package

This package provides ROS 2 nodes that extract information from mecanumbot's on-board sensors.

## Available nodes

| Node                                   | Purpose                                                                                   | File                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `mecanumbot_lidar_detect_people`       | Runs DR-SPAAM on LiDAR scans to detect and track people.                                  | `mecanumbot_sensorprocess_smart/mecanumbot_lidar_detect_people.py`       |
| `mecanumbot_cam_detect_people`         | Runs YOLO pose inference on the main camera or a compressed image topic to detect people. | `mecanumbot_sensorprocess_smart/mecanumbot_cam_detect_people.py`         |
| `mecanumbot_onboard_cam_detect_people` | Runs the DeepStream-based camera people detector on NVIDIA hardware.                      | `mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_people.py` |
| `mecanumbot_onboard_cam_detect_objects`| Runs a plain DeepStream YOLO detector that finds **people and tennis balls**, for the fetch game. | `mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_objects.py` |
| `mecanumbot_locate_detections`         | Fuses camera and LiDAR detections and projects them into map space; also places the ball. | `mecanumbot_sensorprocess_smart/mecanumbot_locate_detections.py`         |
| `mecanumbot_detect_tennis`             | Detects tennis balls from the camera stream and publishes their presence state.           | `mecanumbot_sensorprocess_smart/mecanumbot_detect_tennis.py`             |

The camera detector comes in **three** variants, and which one is running changes what
the robot can do:

| Variant | Model | Publishes | Use it for |
| --- | --- | --- | --- |
| `mecanumbot_cam_detect_people` | YOLO pose, PyTorch/Ultralytics, portable | `cam_people_detections` (skeletons) | a dev machine, or a run without DeepStream |
| `mecanumbot_onboard_cam_detect_people` | YOLO pose, DeepStream, Jetson only | `cam_people_detections` + ROS4HRI | leading and ostensive experiments — anything needing keypoints |
| `mecanumbot_onboard_cam_detect_objects` | plain YOLO (COCO), DeepStream, Jetson only | `cam_people_boxes`, `cam_ball_boxes` (boxes, no keypoints) | the fetch game — the only variant that can see a ball |

Run **one** of them, which `perception.launch.py`'s `detector` argument enforces. On an
Orin Nano two networks on one camera stream is most of the GPU, and the fetch detector
is a straight trade rather than an upgrade: a pose network
has exactly one class, so there is no threshold at which it starts finding tennis
balls, and a plain detector has no skeletons, so the ostensive gestures are unavailable
while it is the one in use. `mecanumbot_locate_detections` accepts person evidence from
either kind, so `people_fusion` keeps flowing whichever is up.

The two pose variants additionally publish the ROS4HRI (REP-155) `/humans/bodies` tree;
see [ROS4HRI (REP-155) output](#ros4hri-rep-155-output).

## Pipeline

```text
scan ──► mecanumbot_lidar_detect_people ──► dr_spaam/dets ─┐
                                                           ├─► mecanumbot_locate_detections ──► people_fusion
camera ─► mecanumbot_cam_detect_people ──► cam_people_detections ┘
                                       └─► /humans/bodies/…  (ROS4HRI, DeepStream variant)

                       ... or, for the fetch game, instead of the pose detector:

camera ─► mecanumbot_onboard_cam_detect_objects ─┬─► cam_people_boxes ─► (same fusion) ─► people_fusion
                                                 └─► cam_ball_boxes ──► mecanumbot_locate_detections
                                                                          ├─► ball_fusion      (PoseArray, map)
                                                                          └─► ball_detections  (Detection3DArray, map)
```

`people_fusion` (`geometry_msgs/PoseArray`, `map` frame) is what the behaviour trees
in `mecanumbot_behaviours` consume. `ball_detections`
(`vision_msgs/Detection3DArray`, `map` frame) is the same interface for the ball, and
is what `mecanumbot_fetch_behaviour` reads. The LiDAR node additionally publishes
`subject_pose` directly for the leading behaviours. The ROS4HRI topics are a parallel,
standards-compliant output for external HRI tooling; nothing inside this repository
consumes them yet.

## Launch files

`launch/perception.launch.py` is the whole pipeline, and it is what every behaviour
launch file includes. `launch/mecanumbot_peopledetect.launch.py` is a thin wrapper
over it under the older name, for a run with no behaviour tree.

**The base launch no longer starts any of this.** It used to, so every run of the
robot — a teleop session, a mapping run, T1 exploration — carried a DR-SPAAM
detector, a DeepStream network and the fusion node whether or not anything subscribed
to them. On an Orin Nano that is not free: the network is most of the GPU, and it
holds the camera open so nothing else can have it. Each behaviour now starts the
detector it needs:

| Launch | detector | `use_camera` |
| --- | --- | --- |
| `mecanumbot_leading_behaviour` | `pose` | **true** — a leading trial is scored afterwards from what the robot could see |
| `mecanumbot_ostensive_behaviour` | `pose` | false |
| `mecanumbot_seek` | `pose` (for the alert's audience) | false |
| `mecanumbot_fetch_behaviour` | `fetch` | false |

`mecanumbot_bringup`'s `launch_external.launch.py` used to start `mecanumbot_lidar_detect_people` unconditionally on the operator PC, under the same node name and namespace as the one here — so it either duplicated the robot's detector on `dr_spaam/dets` and `subject_pose` or ran with nothing subscribed. It is now behind `use_people_detection`, default false. Set it true only to run DR-SPAAM off the robot, and then keep the robot's off (`use_lidar_people:=false`, or the tree's `use_perception:=false`).

### Arguments

| Argument | Default | Function |
| --- | --- | --- |
| `namespace` | `mecanumbot` | Namespace the perception nodes run in. |
| `detector` | `pose` | `pose` (DeepStream skeletons) \| `fetch` (DeepStream people **and balls**) \| `none` (LiDAR only). |
| `use_lidar_people` | `true` | Run DR-SPAAM on the scan. |
| `use_camera` | `false` | See below — this is a choice, not a flag. |
| `camera_topic` | `/camera/image_raw/compressed` | Where the frames are published and read. Absolute on purpose. |
| `camera_width` / `camera_height` | `1280` / `720` | The frame size, for **all three** of camera, detector and fusion. |
| `camera_fps`, `jpeg_quality` | `15.0`, `80` | Only used when `use_camera` is true. |
| `yolo_imgsz` / `yolo_model` | `1280` / `yolo26m-pose` | The pose model. |
| `fetch_imgsz` / `fetch_model` | `640` / `yolo26m` | The fetch model. |

### `use_camera`: who owns the camera

The camera can only be opened once, so this is a choice between two things you might
want and cannot both have for free:

* **false** — the DeepStream detector opens the camera itself through
  `nvarguscamerasrc`. Cheapest path: no JPEG encode, no decode, no topic. But nothing
  else can have the camera, so **there is no `/camera/image_raw/compressed`** for a
  recording, the web GUI, or an operator to look at.
* **true** — `mecanumbot_camera_stream`'s compressed publisher owns the camera and the
  detector subscribes to its topic. That costs a JPEG encode on the publisher and a
  decode in the detector. It is what the leading experiment runs with.

### One frame size, three nodes

`camera_width` / `camera_height` go to the camera publisher, to the detector and to the
fusion node together. All three had their own copy before and nothing compared them —
but every bearing and every apparent-size range is computed from the frame size, so a
disagreement is not a warning, it is silently wrong numbers. (`camera_compressed.launch.py`
declared `width`/`height` arguments and then dropped them on the floor; that is fixed,
so passing them now does something.)

Node names must match the YAML's top-level keys, so do not rename them in the launch
file. The legacy tennis-ball node and the portable Ultralytics detector are not started
by either launch file.

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
| `stride`                    | `2`                      | Detector stride passed to DR-SPAAM. See *GPU load control* below.                     |
| `scan_topic`                | `/mecanumbot/scan`       | Input laser scan topic.                                                               |
| `detections_topic`          | `dets`                   | Output detection topic (`dr_spaam/dets` in the shipped YAML).                         |
| `rviz_topic`                | `dets_marker`            | Marker topic name (unused while marker publishing is disabled).                       |
| `leading_mode`              | `true`                   | Enables the `subject_pose` publisher.                                                 |
| `obstacle_exclusion_radius` | `0.2`                    | Inflation radius in metres applied to the keepout mask.                               |
| `detection_frame`           | `base_scan`              | Accepts `base_scan` or `map`; anything else falls back to `base_scan` with a warning. |
| `track_require_motion`      | `true`                   | Only publish a track that has been seen moving. See *Motion, and re-seeding it* below. |
| `track_reseed_memory`       | `1.5`                    | Seconds a dropped track's motion evidence is kept for its replacement to inherit.     |
| `track_reseed_distance`     | `0.5`                    | Metres within which a new track counts as the replacement of a dropped one.           |

#### GPU load control

| Parameter                   | Default | Function                                                                                                         |
| --------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `max_inference_rate`        | `5.0`   | Upper bound in Hz on how often DR-SPAAM runs. `0.0` removes the cap (one inference per scan, the old behaviour). |
| `publish_on_skipped_scans`  | `true`  | On scans where inference was skipped, extrapolate the tracks and publish anyway, so outputs keep LiDAR rate.     |
| `use_amp`                   | `true`  | FP16 autocast for the convolutions, with an automatic FP32 fallback if it misbehaves at start-up.                |
| `cudnn_benchmark`           | `true`  | Lets cuDNN autotune its 1D convolution kernels once; the input shape never changes.                              |
| `torch_threads`             | `2`     | `torch.set_num_threads` value. `0` leaves the torch default alone.                                                |
| `idle_skip_range`           | `0.0`   | Skip inference when no return is closer than this many metres. `0.0` disables the check.                          |
| `expected_points`           | `240`   | Number of points the scan is resampled to before cutout extraction.                                              |
| `angle_increment`           | `0.026` | Angular increment in radians declared to DR-SPAAM's laser spec.                                                  |
| `perf_log_period`           | `0.0`   | Seconds between throughput reports (inference rate, mean latency, duty cycle). `0.0` disables them.               |
| `track_max_distance`        | `0.5`   | Maximum association distance in metres between a track and a detection.                                          |
| `track_max_missed_time`     | `0.4`   | Seconds a track survives without a measurement.                                                                  |
| `track_min_hits`            | `2`     | Measurements required before a track is published.                                                               |

### Behavior

- Loads DR-SPAAM weights from the package share `models/` folder and fails fast with
  `FileNotFoundError` if the checkpoint is missing.
- Detects CPU/GPU capability via torch and monkey-patches `torch.load` to map the
  checkpoint onto the available device. The detector is built with `gpu=` matching that
  detection, so a CPU-only host no longer tries to move the model to CUDA.
- Runs a warm-up inference pair at start-up. Two passes are needed because the first
  only seeds DR-SPAAM's auto-regressive feature template and the second is the one that
  exercises the spatial-attention gate. This pays the CUDA context and cuDNN autotuning
  cost before the first real scan, and is where half precision is validated.
- Preprocesses each scan: invalid/inf/NaN ranges are replaced by the max range, a
  size-3 median filter is applied, then the scan is resampled to `expected_points`
  points by nearest index.
- Rejects detections landing inside the inflated keepout mask, using a single cached
  TF lookup per frame.
- Tracks the survivors with a Kalman filter plus Hungarian assignment; a track is only
  published once it has at least `track_min_hits` hits **and** has exceeded 0.1 m/s at
  some point, so stationary false positives are suppressed.
- Uses TF from the scan frame to `map` for the `subject_pose` output, and keeps
  republishing the last known subject pose when no new one can be computed.
- Spins on a `MultiThreadedExecutor`.

### Motion, and re-seeding it

A track is only published once it has been seen exceeding 0.1 m/s at least once
(`track_require_motion`). This is the guard that keeps `dets` free of furniture: a 2D
scan at ankle height is full of table legs, bin corners and door frames, DR-SPAAM will
call some of them people, and almost none of them move. The flag is sticky, so a person
who walks in and then stands still stays a person.

Sticky **per track** was the problem. Close to the robot a person's two legs subtend a
wide angle and the detector resolves them as one blob, then two, then one again; each
flicker longer than `track_max_missed_time` drops the track, and the replacement starts
over with no motion of its own. A person standing half a metre away — exactly where the
camera is blind and the LiDAR is the only sensor left — could therefore stop being
reported altogether, having no way to prove again something they had already proved.

So an expiring track leaves a **ghost**: its last position and whether it had been seen
moving, kept for `track_reseed_memory` seconds. A new track starting within
`track_reseed_distance` of a ghost inherits its motion evidence. It does **not** inherit
`hits`, so it still has to be seen `track_min_hits` times before it is published — the
re-seed restores what was already established about this person and proves the rest
again. A ghost of something that never moved lends nothing, so the furniture guard is
untouched.

`track_require_motion: false` publishes stationary detections outright. That also
publishes the furniture, which is why it is not the default.

The tracker itself lives in `lidar_tracking.py`, apart from the node, so it can be
unit-tested on a machine with neither `torch` nor `dr_spaam` installed
(`test/test_lidar_tracking.py`).

### Keeping the GPU load down on a Jetson Orin Nano

Three independent levers, all of which leave the published interface unchanged:

1. **Inference rate cap** (`max_inference_rate`, biggest win). DR-SPAAM no longer runs
   once per scan. Between inferences the Kalman tracker is advanced with
   `predict_only()`, so `dets` and `subject_pose` still update at full LiDAR rate --
   they are extrapolated rather than frozen. At the default 5 Hz against a 10 Hz LiDAR
   this halves GPU time; 2 Hz cuts it to about a fifth.
2. **`stride`** (default raised from 1 to 2). This is DR-SPAAM's own speed knob: it
   subsamples the scan points that become cutout centres, so 2 halves the number of
   cutouts the network sees and roughly halves both GPU and CPU time. It does reduce the
   candidate sampling density, so set it back to `1` if recall on distant people
   matters more than load.
3. **`use_amp`** (FP16 autocast). Note that `model.half()` does **not** work here:
   `_SpatialAttention` creates its neighbour mask lazily at runtime instead of
   registering it as a buffer, so `.half()` never reaches it and it stays FP32. That
   FP32 mask promotes the masked softmax back to FP32 and the following weighted-average
   matmul then fails with `expected m1 and m2 to have the same dtype, but got: float !=
   c10::Half` (and the `1e10` masking constant is `inf` in FP16 anyway). Autocast leaves
   that arithmetic in FP32 and only casts the convolutions, which is where the time
   goes.

Because the tracker now ages in **seconds** rather than in frames, lowering
`max_inference_rate` does not silently shorten how long a track survives an occlusion.

Set `perf_log_period: 10.0` to have the node report its actual inference rate, mean
latency and duty cycle, which is the quickest way to confirm the saving on hardware.

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
| `/humans/bodies/tracked`                       | `hri_msgs/msg/IdsList`                        | ROS4HRI: IDs of the bodies currently being tracked. Republished whenever the set changes.             |
| `/humans/bodies/<id>/skeleton2d`               | `hri_msgs/msg/Skeleton2D`                     | ROS4HRI: 18-joint normalized skeleton, created and destroyed with the body.                           |
| `/humans/bodies/<id>/roi`                      | `hri_msgs/msg/NormalizedRegionOfInterest2D`   | ROS4HRI: normalized bounding box, only when `ros4hri.publish_roi` is enabled.                         |

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
| `keypoint_scaling`            | `auto`                        | How to invert the `nvinfer` input resize: `letterbox`, `stretch`, or `auto`.           |
| `model_params.imgsz`          | `1280`                        | Input size the pose model expects; selects `models/imgsz_<n>/`.                       |
| `model_params.model_name`     | `yolo26m-pose`                | Model stem inside that folder.                                                        |
| `model_params.precision`      | `fp16`                        | Engine precision; part of the engine filename, must match `network-mode`.             |
| `model_params.models_dir`     | `''`                          | Where the `imgsz_<n>` folders live; empty means the package share `models/`.           |
| `model_params.custom_lib_path`| `''`                          | The pose parser library, `~`/`$USER` expanded. Empty searches the config's path, then `~/deepstream_source` and `~/Documents/installed_external`. |
| `model_params.nvinfer_config` | `''`                          | A complete nvinfer config to use untouched, disabling the substitution below.          |
| `ros4hri.enabled`             | `true`                        | Publishes the `/humans/bodies` tree in addition to the native messages.               |
| `ros4hri.prefix`              | `/humans`                     | Root of the ROS4HRI topic tree. Absolute, so the node namespace does not shift it.    |
| `ros4hri.publish_rate`        | `30.0`                        | Hz at which queued bodies are published and per-body publishers reconciled.           |
| `ros4hri.body_timeout`        | `0.5`                         | Seconds an unseen body keeps its ID and its publishers.                               |
| `ros4hri.publish_roi`         | `true`                        | Also publishes `<id>/roi` next to `<id>/skeleton2d`.                                  |
| `ros4hri.iou_threshold`       | `0.3`                         | Minimum box IoU for a detection to inherit an existing body ID.                       |
| `ros4hri.frame_id`            | `''`                          | Header frame for the ROS4HRI messages; empty reuses `<namespace>/head_link`.          |

Detection gate parameters, all under `detection_gate.` — see the section below for what
they do:

| Parameter                             | Default | Function                                                                       |
| ------------------------------------- | ------- | ------------------------------------------------------------------------------ |
| `box_conf_acquire`                    | `0.6`   | Box confidence needed to start treating a blob as a person.                    |
| `box_conf_retain`                     | `0.35`  | Box confidence needed to keep an already-confirmed person.                     |
| `keypoint_conf`                       | `0.3`   | Joint visibility threshold: below it a keypoint is `NaN` and is not drawn.     |
| `best_keypoint_conf_acquire`          | `0.7`   | The best single joint must reach this to acquire.                              |
| `best_keypoint_conf_retain`           | `0.5`   | ... and this to retain.                                                        |
| `min_valid_keypoints_acquire`         | `6`     | Joints over `keypoint_conf` needed to acquire.                                 |
| `min_valid_keypoints_retain`          | `3`     | ... and to retain.                                                             |
| `min_torso_keypoints_acquire`         | `2`     | Of the four shoulder/hip joints, how many are needed to acquire.               |
| `min_torso_keypoints_retain`          | `1`     | ... and to retain.                                                             |
| `min_box_height`                      | `40.0`  | Minimum box height in pixels, in both modes.                                   |
| `max_box_aspect_ratio`                | `1.6`   | Maximum width/height, in both modes. People are taller than wide.              |
| `proximity_enabled`                   | `true`  | Enables the close-range branch below; `false` restores torso-only behaviour.   |
| `proximity_min_height_fraction`       | `0.6`   | Share of the frame height a box must fill to count as close.                   |
| `proximity_top_margin`                | `8.0`   | Pixels from the top edge within which a box counts as clipped by it; a floor for small frames. |
| `proximity_top_fraction`              | `0.08`  | The same tolerance as a share of the frame height; the larger of the two wins (58 px at 720p). |
| `proximity_dominant_height_fraction`  | `0.85`  | A box filling this much of the frame is close whatever its top edge does.       |
| `proximity_box_conf_acquire`          | `0.4`   | Box confidence to acquire a close-range (cropped, therefore lower-scoring) body. |
| `proximity_box_conf_retain`           | `0.3`   | ... and to retain it.                                                          |
| `proximity_best_keypoint_conf_acquire` | `0.5`  | Confidence the best joint needs to acquire close up; knees and ankles score below faces. |
| `proximity_best_keypoint_conf_retain` | `0.35`  | ... and to retain.                                                             |
| `proximity_min_valid_keypoints_acquire` | `2`   | Joints over `keypoint_conf` needed to acquire at close range.                  |
| `proximity_min_valid_keypoints_retain`  | `1`   | ... and to retain.                                                             |
| `proximity_min_lower_body_acquire`    | `2`     | Of the six hip/knee/ankle joints, how many are needed to acquire close up.     |
| `proximity_min_lower_body_retain`     | `1`     | ... and to retain.                                                             |
| `proximity_max_box_aspect_ratio`      | `2.5`   | Maximum width/height for a close-range box.                                    |
| `min_hits`                            | `2`     | Frames a blob must pass the acquire gate before anything is published.         |
| `max_missed_time`                     | `0.5`   | Seconds a confirmed track survives without a detection.                        |
| `iou_threshold`                       | `0.3`   | Minimum box IoU to associate a detection with an existing track.               |
| `log_rejections`                      | `false` | Logs every rejected box with the check it failed. Verbose; for tuning.         |

### Detection gate

The pose network reports one box confidence plus 17 keypoint confidences per object.
Deciding on the box confidence alone — which is what this node used to do — has no
threshold that works: a bean bag or a wall-panel occluder can score a plausible box
while producing no coherent skeleton, and a partly occluded person scores *lower* than
such a prop. Raising the threshold to exclude the props therefore drops real people, and
lowering it to keep them lets the props back in. The two failure modes are not ordered
along the same axis, so they are separated along three:

1. **Keypoint evidence.** How much of a body was actually found: how many joints cleared
   `keypoint_conf`, how confident the best one is, and whether a torso (at least
   `min_torso_keypoints_*` of the four shoulder/hip joints) is present. This is the check
   props fail — the network has no body parts to place on them. A cheap box-geometry
   check (`min_box_height`, `max_box_aspect_ratio`) rejects squat floor props outright.
2. **Hysteresis.** A blob must clear the strict `*_acquire` thresholds to be taken
   seriously; once confirmed it is kept on the looser `*_retain` ones. This is what stops
   a person who turns away, is partly occluded, or walks into poor light from dropping
   out. Retention is still evidence-based, so a prop that happens to overlap a confirmed
   person does not inherit its track.
3. **Temporal confirmation.** A candidate must pass the acquire gate on `min_hits`
   frames before it is published, and a confirmed track tolerates `max_missed_time` of
   dropout before it must be re-acquired. Single-frame flickers in either direction never
   reach the fusion layer. At `min_hits: 2` this costs one frame (~66 ms at 15 fps) of
   latency on a newly appearing person.

An unconvincing frame updates no state at all: it neither refreshes a track nor starts
one, so props never accumulate hits and a person who genuinely leaves expires on
`max_missed_time`.

#### Close range: the person standing next to the robot

The camera is on the head, about **0.22 m** off the floor (`head_joint` at z = 0.168 in
`mecanumbot.urdf`, plus the camera offset), with a vertical field of view of roughly
**36°** at 1280×720. That geometry decides what a person looks like as they approach:

| Distance | What is in frame                     | Torso keypoints available |
| -------- | ------------------------------------ | ------------------------- |
| 0.6 m    | floor to ~0.4 m — calves and knees    | none                      |
| 1.5 m    | floor to ~0.7 m — up to mid-thigh     | none                      |
| 2.3 m    | floor to ~0.95 m — hips just arrive   | hips                      |
| 3.7 m    | floor to ~1.4 m — shoulders arrive    | hips and shoulders        |

So a person closer than about 2.3 m has **no torso in the image at all**, and stage 1
above — which requires `min_torso_keypoints_acquire` of the four shoulder/hip joints —
could not acquire them however good the detection was. It is not a tuning problem: the
evidence the gate asks for is outside the field of view. Close range is also where a
person matters most, so the gate has a second branch for it.

Which branch applies is decided from the **box geometry, not from the keypoints** —
inferring "this is a close body" from the very joints the branch then stops requiring
would make the relaxation self-justifying. A detection is close-range when **either**:

- its box starts within `proximity_top_margin` px **or** `proximity_top_fraction` of the
  frame height of the **top** edge, whichever is larger (the body carries on above the
  field of view), **and** spans at least `proximity_min_height_fraction` of the frame; or
- its box spans `proximity_dominant_height_fraction` of the frame, wherever its top
  edge happens to sit — nothing at a distance is that big.

For this camera those defaults amount to "nearer than roughly 3.4 m", which overlaps the
2.3 m at which hips appear — the two branches cover the whole approach with no distance
at which a person falls between them.

**Why "starts at the top edge" has to be forgiving.** The box is drawn round the body
the *network* found, not round the person. Loose clothing is not recognised as body, so
a detection that is unmistakably close — legs filling the picture — can still have its
top edge tens of pixels inside the frame. A long skirt is the case this was written for;
a coat, or a chair under a floor-length tablecloth, behaves the same way. The original
absolute `proximity_top_margin: 8.0` could not be met by any of them: the detection fell
back on the full-body gate and was rejected as `keypoints 3<6` — for a torso that was
never in shot. Hence the fractional margin (0.08, i.e. 58 px at 720p) and the
dominant-height escape hatch.

On that branch the torso requirement is replaced by a **lower-body** one
(`proximity_min_lower_body_*`, counted over hips, knees and ankles), the keypoint counts
drop to `proximity_min_valid_keypoints_*`, the box-confidence gates drop to
`proximity_box_conf_*` because a cropped body scores lower than a whole one, and the
shape check loosens to `proximity_max_box_aspect_ratio` — legs seen from half a metre
can be wider than the slice of them that fits in the frame is tall.

What keeps the relaxation honest is that the geometry and the keypoints have to agree. A
wall panel or a bean bag pushed up against the camera has exactly the same box geometry,
but it still cannot produce a leg: `proximity_min_lower_body_acquire` of the six
hip/knee/ankle joints is the check that does the work, and the joints it counts have to
clear `proximity_best_keypoint_conf_acquire`. That last one is lower than the whole-body
`best_keypoint_conf_acquire` — knees and ankles score below faces and shoulders, so the
whole-body value was unmeetable on a legs-only view — but it is still far above
`keypoint_conf`, so one joint has to be placed convincingly. Small floor props are
unaffected either way: they are not clipped by the top of the frame and do not fill it,
so they are still judged on the ordinary gate.

Because the geometry test is now the looser of the two, the lower-body count is the only
thing standing between a draped chair and a person. `test_person_gating.py` pins that
down (`TestLooseClothing`), including the tablecloth case.

Detections accepted this way are published with `type` set to `close_range` rather than
`full_body`. The bearing itself is computed exactly as usual, from whichever joints were
found — legs give a bearing the way a whole body does — but a consumer that needs arms or
a head (the ostensive tree reading gestures, for instance) can now tell that the upper
body is *outside the frame* rather than merely undetected. On the debug image close-range
boxes are green rather than blue and labelled `near`; rejections on this branch have
their reason prefixed `near-`.

Set `proximity_enabled: false` to go back to the torso-only behaviour.

Note that `keypoint_conf` is a *visibility* threshold — it decides which joints are
usable, published as coordinates rather than `NaN`, and fed into the angular bounds. It
is deliberately far below the box gates. Setting the two equal (as this node previously
did, at `0.6`) is what made distant and partly occluded people lose every keypoint; the
node now warns at start-up if `keypoint_conf` is raised to the acquire threshold.

When no joint at all clears `keypoint_conf`, the angular bounds fall back to the edges of
the bounding box. Previously they were left unset in that case, i.e. `bound_angle_min ==
bound_angle_max == 0.0`, which `mecanumbot_locate_detections` reads as a person straight
ahead.

The logic lives in `person_gating.py`, kept free of ROS, DeepStream and NumPy so it can
be unit-tested off the Jetson — `test/test_person_gating.py` is the only real test in
this package, and it runs without a ROS graph:

```bash
python3 -m pytest src/mecanumbot_sensorprocess_smart/test/test_person_gating.py -v
```

Tuning: run with `debug_mode` enabled and watch
`cam_people_detections/debug_image/compressed`. Accepted boxes are blue, rejected ones
red and labelled with the check they failed, so the parameter to change is named on the
image. `log_rejections` puts the same reasons in the log.

### ROS4HRI (REP-155) output

DeepStream and ROS4HRI are orthogonal and compose without friction: DeepStream decides
*how* bodies are found, ROS4HRI only fixes *how they are published* (`hri_msgs` on a
prescribed topic layout). No extra process, bridge or model change is involved -- the
existing `nvinfer` buffer probe feeds both output paths from the same
`NvDsObjectMeta`, and the native `cam_people_detections` output is untouched.

Three things do need handling, and are what `ros4hri_bridge.py` exists for:

- **Keypoint convention.** YOLO-pose emits COCO-17. `hri_msgs/Skeleton2D` follows the
  OpenPose COCO-18 order, which is indexed differently *and* contains a `NECK` joint
  COCO-17 does not have; it is synthesised as the shoulder midpoint and only published
  when both shoulders were found.
- **Coordinates.** ROS4HRI requires x and y in `[0, 1]`, so unavailable joints cannot be
  signalled with `NaN` the way `CamPersonDetection` does. They are emitted with `c = 0.0`
  instead, and all coordinates are clamped into range.
- **Body IDs.** ROS4HRI identifies bodies by a string ID that has to persist across
  frames, which a per-frame detector does not provide. `BodyIdTracker` assigns short
  random IDs (REP-155 asks for IDs carrying no personal information) and holds them
  through greedy IoU association. Per-body publishers are created and destroyed as
  bodies appear and disappear, driven from a timer -- DeepStream calls its probes on a
  GStreamer streaming thread, and rclpy entity management does not belong there.

Only the `bodies` part of the tree is published. `/humans/faces/*` and
`/humans/persons/*` would need face detection and identity recognition respectively,
which this pipeline does not do, so no `candidate_matches` are advertised either.
`<id>/position` is likewise absent: it needs depth, which the fusion done by
`mecanumbot_locate_detections` provides in map space rather than per body.

Consumers can use the standard `hri`/`pyhri` client libraries directly; the publishers
use default (reliable) QoS to stay compatible with them.

### Keypoint scaling

Keypoints leave the parser in **network-input** coordinates, so mapping them back to
image pixels has to undo exactly the resize `nvinfer` performed. Which resize that was
is decided by `maintain-aspect-ratio` in the nvinfer config:

- `letterbox` (`maintain-aspect-ratio=1`): uniform scale plus centred padding, undone
  with a single gain and an x/y pad.
- `stretch` (`maintain-aspect-ratio=0`): x and y scaled independently, no padding.
- `auto`: read `maintain-aspect-ratio` out of the config file and pick accordingly.

The shipped `config_infer_yolo26_pose.txt` sets `maintain-aspect-ratio=0`, so the
default is `auto` and resolves to `stretch`. Choosing `letterbox` against that config
invents padding that `nvinfer` never added: at 1280x720 it puts every joint at
`1.78*y - 280`, i.e. a skeleton stretched by nearly a factor of two and pushed off the
bottom of its own box, while leaving x correct. The error is **independent of the
network size** — 640x640 and 1280x1280 give the same wrong `1.78*y - 280` — so changing
the exported `imgsz` neither causes nor cures it.

The network size itself is taken from the mask metadata, which the parser stamps with
the dimensions of the engine that actually ran. `infer-dims` in the nvinfer config
records the size the ONNX was exported at and is used only to cross-check it, because
the two can genuinely differ: the engine filename encodes the batch, the GPU and the
precision but **not the input size**, so after re-exporting the ONNX at a new `imgsz`
`nvinfer` loads the old engine and keeps running at the old size. A mismatch is a
warning at the first detection; the fix is to delete the `model-engine-file` and let it
rebuild.

With `debug_mode` on, the node dumps every number the mapping depends on once, from
the first detection: the surface it draws the debug image on, `mask_params` (dimensions
and float count), the `camera_params`, the resolved mapping, the box as `nvinfer`
mapped it, and the first four raw triplets next to what they mapped to. A misplaced
skeleton looks the same whichever link in the chain broke, and that one line separates
them — in particular, the raw triplets are `(conf, x, y)` if the first number of each is
in `[0, 1]`, and `(x, y, conf)` if the last one is. Upstream `DeepStream-Yolo-Pose`
writes `(x, y, conf)`; this node reads `(conf, x, y)`, so a checkout that follows
upstream needs the triplet reversed here.

### Metadata space vs. surface size

Everything in the metadata — `rect_params` from `nvinfer` and the keypoints mapped here
alike — is in the frame `nvinfer` believes it inferred on, which is the size
`nvstreammux` was configured for, i.e. `camera_params`. The buffer the probe receives is
not always that size: a source whose frames are passed through rather than rescaled
leaves the surface at its own resolution. Drawing the annotation at the muxer's scale
then shrinks it towards the top-left corner — box and skeleton together, in exact
proportion to the height the muxer thinks it has. A 4:3 source treated as 1280x720 puts
every joint at `0.75*y`, so the knees land on the thighs and the feet on the shins,
while x is untouched because the widths agree.

The node rescales the overlay onto the surface it is drawn on and says so once. It does
**not** rescale the detections: the bearings and the gate are normalised by
`camera_params`, the same space the metadata is in, so they stay self-consistent. What
does not survive the mismatch are the gate's absolute thresholds — `min_box_height` and
`max_box_aspect_ratio` are then measured on a frame of the wrong shape — so the right
fix is to set `camera_params` to the size the warning reports.

Whichever mapping is in force, the node checks it once against the bounding box:
`rect_params` reaches the probe already in frame pixels, mapped by `nvinfer` itself, so
a skeleton that lands outside its own box means the node's mapping and `nvinfer`'s
disagree. That warning names both suspects — `keypoint_scaling` and the network size —
because the failure is otherwise silent: detections still publish, with bearings read
off joints that are in the wrong place, and the fusion in
`mecanumbot_locate_detections` then fails to confirm the people the LiDAR found.

### Behavior

- Announces its input geometry at startup: the requested frame size and source, the
  `nvstreammux` size and the horizontal FOV, plus the network input size — read from
  `infer-dims` if the nvinfer config sets it (the rendered config always does, from
  `model_params.imgsz`), otherwise reported from the first inferred frame, since without
  `infer-dims` nvinfer takes the shape from the model itself. The
  size the source *actually* delivered is logged from the first frame as well, and a
  mismatch against `camera_params` is a warning when the aspect ratios differ: every
  bearing is derived from a keypoint's x within `camera_width`, so a frame that
  `nvstreammux` had to distort into that shape makes all of them wrong.
- Uses GStreamer and NVIDIA DeepStream instead of the pure PyTorch/OpenCV path:
  `source → nvvideoconvert → nvstreammux → nvinfer → nvvideoconvert → capsfilter(RGBA) → fakesink`,
  with a buffer probe on the capsfilter reading the inference metadata.
- Supports either a ROS image topic or direct webcam input.
- Extracts pose keypoints from NVIDIA metadata, undoes the `nvinfer` input resize
  (see *Keypoint scaling*), and normalizes them to `[0, 1]` before mapping into
  `CamPersonDetection` messages.
- Unmaps the NvDs buffer surface after every frame to avoid leaking memory on Jetson.

### DeepStream configuration

An ONNX export is fixed to the `imgsz` it was exported at, so the exports are stored
**one folder per size** — `models/imgsz_640/`, `models/imgsz_1280/` — with the
size-independent `.pt` checkpoints left at the top of `models/`. `model_params.imgsz`
picks the folder and `model_params.model_name` the file in it; `perception.launch.py`
exposes both as the `yolo_imgsz` and `yolo_model` arguments (and `fetch_imgsz` /
`fetch_model` for the other detector), and every behaviour launcher forwards them:

```bash
ros2 launch mecanumbot_sensorprocess_smart perception.launch.py yolo_imgsz:=640
ros2 launch mecanumbot_leading_behaviour launch_wifi_condition_sequence.launch.py \
    yolo_imgsz:=640
```

`deepstream_config/config_infer_yolo26_pose.txt` is the **template** for that choice,
not the file `nvinfer` is given. At startup the node renders a copy of it into
`/tmp/mecanumbot_nvinfer_<model>_imgsz<n>.txt` with four lines rewritten — `onnx-file`,
`model-engine-file`, `infer-dims` and `labelfile-path` (absolutized, since relative
paths in an nvinfer config resolve against the config's own directory) — and points
`nvinfer` at the copy. The rendered file is logged and left on disk to be read. The
values in the template itself only apply when the file is fed to `nvinfer` directly
(`deepstream-app`, `gst-launch`) or via `model_params.nvinfer_config`.

That is what keeps `infer-dims` and the model in step: they are no longer two lines that
have to be edited together. The engine follows too, since it is named after its own ONNX
(`<onnx>_b1_gpu0_fp16.engine`) inside the size folder, so a 640 engine can no longer be
loaded for a 1280 export — the engine filename does not encode the input size, and that
mismatch used to be silent. Re-exporting at the *same* size still requires deleting the
engine by hand; `models/conv_to_onnx.py` names any it finds.

`custom-lib-path` is rewritten too, because where `DeepStream-Yolo-Pose` was built
differs per machine. The template names `~/deepstream_source/DeepStream-Yolo-Pose/…`
(the robot's layout); the node expands `~` and `$VARS` in it and, if nothing is there,
looks for the same library under `~/Documents/installed_external/` (a laptop's), writing
whichever exists into the copy and logging it. Setting `model_params.custom_lib_path`
replaces the search with that one path, also expanded (`$HOME/…`, `/home/$USER/…`).
If nothing is found the error lists every path tried. `nvinfer` expands nothing itself,
so a template fed to it directly needs an absolute path.

The rendered copy is written even when the selected ONNX is missing — with only the
paths absolutized and the library resolved — so the fallback runs the model the
template names rather than failing on the library first.

```bash
# export at a size and build its engine (on the Jetson; engines do not transfer)
python3 models/conv_to_onnx.py yolo26m-pose --imgsz 640
python3 models/build_engine.py models/imgsz_640/yolo26m-pose.onnx
```

Engines are gitignored build artifacts. `nvinfer` builds a missing one on the first
launch, but it writes it next to the ONNX it loaded — under `install/` in a
`--symlink-install` workspace, where the next `colcon build` may not preserve it. Point
`model_params.models_dir` at the source tree, or build engines ahead of time with
`build_engine.py`, to keep them.

## Node: mecanumbot_onboard_cam_detect_objects

The fetch game's detector. One DeepStream pass over a plain YOLO detector (`yolo26m`,
the COCO 80-class model), filtered down to the two classes the game is about and
published as bounding boxes on two topics. Registers as
**`mecanumbot_cam_detect_objects_ds`**, which is the YAML key its parameters must sit
under.

Why a second node rather than a parameter on the pose one: a pose network has exactly
one class. Finding a tennis ball is not a threshold away from finding a person with
a pose model — it is a different network, a different parser
(`NvDsInferParseYolo`, from **DeepStream-Yolo**, not the `…_Yolo_pose` library the
pose node uses) and a different nvinfer config. The pipeline around it is the same,
and `nvinfer_config.py` is the code the two share.

### Publishers

| Topic | Data type | Function |
| --- | --- | --- |
| `cam_people_boxes` | `vision_msgs/msg/Detection2DArray` | Accepted person boxes, in image pixels. |
| `cam_ball_boxes` | `vision_msgs/msg/Detection2DArray` | Accepted ball boxes, in image pixels. |
| `cam_object_detections/debug_image/compressed` | `sensor_msgs/msg/CompressedImage` | Every box, accepted or refused, with the check it failed. Only when `debug_mode` is true. |

Two topics rather than one labelled array because the consumers are different: the
ball boxes become a ball position, the person boxes become a bearing wedge the LiDAR
ranges, and a subscriber that wants one should not filter the other out at 15 Hz.

**Pixels, not angles.** The pose node publishes bearings, having done the image → angle
conversion itself; this one publishes boxes as they came out of the network. That is
the standard type's shape, it puts the camera model in one place
(`mecanumbot_locate_detections`, which has to have it anyway to work out the range),
and it means only one of the two nodes has an opinion about which way round the frame
is.

### Subscribers

| Topic | Data type | Processing |
| --- | --- | --- |
| `camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Decoded and pushed into the pipeline's appsrc. Only when `from_topic` is true; otherwise the source is `webcam_device`. |

### Parameters

| Parameter | Default | Function |
| --- | --- | --- |
| `camera_params.camera_width` / `_height` | `1280` / `720` | Frame the boxes are expressed in. Must match `mecanumbot_locate_detections`. |
| `camera_params.camera_fov` | `60°` | Horizontal field of view, logged for cross-checking against the fusion node's copy. |
| `model_params.imgsz` | `640` | Selects `models/imgsz_<n>/`. |
| `model_params.model_name` | `yolo26m` | Stem of the ONNX in that folder. |
| `model_params.precision` | `fp16` | Must match `network-mode` in the nvinfer config. |
| `model_params.models_dir` | `''` | Empty means the package share `models/`. |
| `model_params.custom_lib_path` | `''` | Where **DeepStream-Yolo** was built, `~`/`$USER` expanded. Empty searches the config's path, then `~/deepstream_source` and `~/Documents/installed_external`. |
| `model_params.nvinfer_config` | `''` | A complete config to hand nvinfer untouched; disables all substitution. |
| `classes.person_id` / `classes.ball_id` | `0` / `32` | COCO numbering for the shipped model. |
| `classes.person_label` / `classes.ball_label` | `person` / `sports ball` | What travels downstream — a numeric id means nothing once the detection has left the camera. |
| `classes.person_topic` / `classes.ball_topic` | `cam_people_boxes` / `cam_ball_boxes` | Where each class is published. |
| `person_gate.*`, `ball_gate.*` | see below | Per-class shape, confidence and confirmation. |
| `debug_mode` | `false` | Publishes the annotated frame. Costs a JPEG encode per frame. |
| `log_rejections` | `false` | Logs every refused box with the check it failed. |

### The gate, without keypoints

`person_gating.py` decides whether a box is a person by looking at the skeleton inside
it, and that whole argument depends on the model emitting keypoints. This one does not,
so `object_gating.py` keeps the two stages that survive — hysteresis (`conf_acquire`
to be taken seriously, the looser `conf_retain` to be kept) and temporal confirmation
(`min_hits` frames to be published, `max_missed_time` of dropout tolerated) — plus a
shape check.

Shape is doing real work for the ball and almost none for the person. **A tennis ball
is round**, so its box is square at every range and from every direction; no other
object in this system has that property, and `min_aspect` / `max_aspect` around 1 are
the cheapest thing separating a ball from a yellow floor marking, a reflection or a
chair leg. They are not `1.0 ± nothing` because a ball in flight smears and a ball
against an obstacle is clipped. A person is only loosely "taller than wide", and close
to the camera not even that, so their bounds are wide on purpose: the work of not
believing a coat rack is a person is done downstream, where the LiDAR has to agree
there is something at that bearing.

Shape is checked *before* the confirmer and separately from it, so a box of the wrong
shape cannot hold a track alive.

### The nvinfer config

`deepstream_config/config_infer_yolo26_det.txt` is a template; `onnx-file`,
`model-engine-file`, `infer-dims` and `labelfile-path` are rewritten at startup from
`model_params` into a copy in `/tmp`, exactly as the pose node does it, and the shared
mechanics live in `nvinfer_config.py`.

Two settings in it are not free choices:

- **`maintain-aspect-ratio=1`.** Letterboxed, not stretched. A tennis ball is
  recognised by being round, and stretching a 16:9 frame into a square network input
  turns every ball into an ellipse of aspect ~1.78 — which breaks the shape gate above
  *and* the apparent-size range downstream, since that reads the ball's diameter off
  the box. The pose config stretches because a person's shape is not what identifies
  them.
- **`gie-unique-id=2`.** Different from the pose node's, so the two can run at once.

The `custom-lib-path` points at **DeepStream-Yolo**'s
`libnvdsinfer_custom_impl_Yolo.so` — a different library from the pose node's
`libnvdsinfer_custom_impl_Yolo_pose.so`, with a different parser function. It is found
the same way as the pose node's: `~/deepstream_source/DeepStream-Yolo/…` as written,
then `~/Documents/installed_external/DeepStream-Yolo/…`, or `model_params.custom_lib_path`
(with `~`/`$USER` expanded). The library has to be **built** on each machine — a
checkout alone has only the sources:

```bash
cd ~/deepstream_source/DeepStream-Yolo   # or ~/Documents/installed_external/DeepStream-Yolo
CUDA_VER=12.6 make -C nvdsinfer_custom_impl_Yolo   # CUDA_VER = the installed CUDA
```

**`yolo26m.pt` is not among the shipped model files** -- only the pose checkpoints and
`yolo26n.pt` are. Fetch it (ultralytics downloads it on first use) and export it before
this node can run:

```bash
# in the package's models/ folder, on a machine with ultralytics
python3 conv_to_onnx.py yolo26m --imgsz 640
# the engine is built by nvinfer on the first launch (minutes), or ahead of time
# on the Jetson -- engines do not transfer between machines
python3 build_engine.py imgsz_640/yolo26m.onnx
```

## Node: mecanumbot_locate_detections

### Publishers

| Topic                             | Data type                     | Function                                                                                    |
| --------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------- |
| `people_fusion`                   | `geometry_msgs/msg/PoseArray` | Publishes fused detections in map space.                                                    |
| `cam_people_detections/left_FOV`  | `geometry_msgs/msg/PoseArray` | Left field-of-view bound of each camera detection. Only created when `debug_mode` is true.  |
| `cam_people_detections/right_FOV` | `geometry_msgs/msg/PoseArray` | Right field-of-view bound of each camera detection. Only created when `debug_mode` is true. |
| `ball_fusion`                     | `geometry_msgs/msg/PoseArray` | Tracked balls in map space. Mirrors `people_fusion`; what rviz draws. |
| `ball_detections`                 | `vision_msgs/msg/Detection3DArray` | The same balls, labelled and with a score and a diameter. What `mecanumbot_fetch_behaviour` reads. |

### Subscribers

| Topic                   | Data type                                     | Processing                                                             |
| ----------------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| `cam_people_detections` | `mecanumbot_msgs/msg/CamPersonDetectionArray` | Receives camera detections for fusion (own callback group).            |
| `dets`                  | `geometry_msgs/msg/PoseArray`                 | Receives LiDAR people detections; each message triggers a fusion pass. |
| `scan`                  | `sensor_msgs/msg/LaserScan`                   | Stores the current scan for range extrapolation.                       |
| `/map`                  | `nav_msgs/msg/OccupancyGrid`                  | Loads the static map grid (TRANSIENT_LOCAL QoS).                       |
| `/amcl_pose`            | `geometry_msgs/msg/PoseWithCovarianceStamped` | Tracks the robot pose in map coordinates.                              |
| `cam_ball_boxes`        | `vision_msgs/msg/Detection2DArray`            | Ball boxes from the fetch detector, in image pixels. Placed on the ball timer, not on arrival. |
| `cam_people_boxes`      | `vision_msgs/msg/Detection2DArray`            | Person boxes from the fetch detector; converted to the same bearing wedge a `CamPersonDetection` carries. |

### Parameters

| Parameter                           | Default | Function                                                                         |
| ----------------------------------- | ------- | -------------------------------------------------------------------------------- |
| `obstacle_buffer_x`                 | `0.5`   | Metres added behind a wall when a detection has to be pushed out of an obstacle. |
| `debug_mode`                        | `false` | Enables the left/right FOV publishers.                                           |
| `cam_detection_timeout`             | `0.6`   | Seconds the last `CamPersonDetectionArray` stays usable. Past it the bearings are dropped rather than re-used against fresh scans. |
| `tracking.enabled`                  | `true`  | Map-frame Kalman tracking. `false` republishes the raw per-frame fusion.          |
| `tracking.publish_rate`             | `10.0`  | Hz at which `people_fusion` is published, independent of when detections arrive.  |
| `tracking.max_association_distance` | `0.9`   | Metres a measurement may sit from a track's prediction and still be the same person. |
| `tracking.min_hits`                 | `2`     | Camera-corroborated measurements before a new track is published.                |
| `tracking.max_coast_time`           | `1.2`   | Seconds a track survives on prediction alone, with no measurement at all.        |
| `tracking.max_uncorroborated_time`  | `4.0`   | Seconds a track survives on LiDAR-only measurements after the camera last agreed. |
| `tracking.measurement_noise`        | `0.25`  | Metres. The scale of the range jump when the bearing wedge slides off a leg.     |
| `tracking.process_noise`            | `0.08`  | How readily the estimate follows a person changing direction.                    |
| `tracking.max_reported_speed`       | `2.5`   | m/s ceiling on the reported velocity; above this is an association error.        |
| `tracking.camera_frame`             | `mecanumbot/head_link` | Frame the blind zone is measured from. The head, not the base: the head turns. |
| `tracking.exempt_close_range`       | `true`  | Treat a track nearer than `camera_blind_range` as one the camera cannot check.   |
| `tracking.exempt_outside_fov`       | `true`  | Treat a track outside `camera_half_fov_deg` as one the camera cannot check.      |
| `tracking.camera_blind_range`       | `0.9`   | Metres inside which there is not enough of a person in frame to detect at all.   |
| `tracking.camera_half_fov_deg`      | `30.0`  | Half the camera's horizontal field of view.                                      |
| `tracking.max_blind_zone_time`      | `30.0`  | Seconds a track is reported on the exemption before the camera has to agree again. |
| `tracking.blind_zone_min_hits`      | `5`     | Consecutive LiDAR-only measurements inside the blind zone that confirm a track.  |
| `tracking.blind_zone_creates_tracks`| `true`  | Whether a LiDAR-only measurement in the blind zone may start a track, not just sustain one. |
| `camera_params.camera_width` / `_height` | `1280` / `720` | The frame the incoming boxes are in. Must match the detector's. |
| `camera_params.camera_fov`          | `60°`   | Horizontal field of view. The bearing and the apparent-size range both come from it. |
| `camera_params.camera_vfov`         | `0.0`   | `0.0` derives it from the frame shape, assuming square pixels. |
| `ball.enabled`                      | `true`  | The ball path as a whole. |
| `ball.boxes_topic`                  | `cam_ball_boxes` | Where the ball boxes come from. |
| `ball.class_id`                     | `sports ball` | The label the located ball is published under. |
| `ball.detection_timeout`            | `0.6`   | Seconds a ball box stays usable. |
| `ball.publish_rate`                 | `10.0`  | Hz at which `ball_fusion` / `ball_detections` are published. |
| `ball.diameter`                     | `0.067` | A regulation tennis ball [m]. Every apparent-size range scales with it. |
| `ball.range_source`                 | `size`  | `size` or `ground_plane` — which estimator supplies the published range. |
| `ball.camera_x` / `ball.camera_z`   | `0.13` / `0.21` | Where the camera is in the base frame [m]. **Measure these**, see below. |
| `ball.camera_pitch_deg`             | `0.0`   | Positive looks up. |
| `ball.floor_z`                      | `-0.01` | The floor in the base frame; `base_link` sits 0.01 m above `base_footprint`. |
| `ball.min_range` / `ball.max_range` | `0.15` / `6.0` | Outside this band a range is not believed and nothing is published. |
| `ball.disagreement_warn`            | `0.6`   | Metres the two estimators may differ by before it is worth a one-off warning. |
| `ball.tracking.*`                   | — | Alpha-beta smoothing in the map frame; see `ball_locating.py`. |
| `person_boxes.enabled`              | `true`  | Accept person evidence from the fetch detector as well as the pose one. |
| `person_boxes.topic`                | `cam_people_boxes` | Where those boxes come from. |

### Behavior

- For every **fresh** camera detection (no older than `cam_detection_timeout`), resolves
  a range in three steps: first look for a LiDAR detection whose bearing falls inside the
  person's angular bounds — of those, the one nearest the middle of the wedge, which is
  the one the camera is actually looking at; if there is none, extrapolate from the raw
  scan using the 20th percentile of the valid ranges inside the bounds (so background
  hits do not dominate); if that also fails, drop the detection.
- Validates the result against the static map: a pose landing on an occupied cell is
  ray-traced outward until free space is found (up to 4 m of wall thickness) and then
  offset by `obstacle_buffer_x`.
- Transforms the accepted poses from `mecanumbot/base_link` into `map`, and hands them
  to the map-frame tracker along with the LiDAR-only detections (see below).
- Publishes the tracker's confirmed people as a `PoseArray` on a timer at
  `tracking.publish_rate`, with a current stamp.
- Runs on a 4-thread `MultiThreadedExecutor`. The tracker is stepped only from the
  publish timer, so it is touched by one thread; the measurement handover from the
  LiDAR callback is the one place that locks.

#### Map-frame tracking

Without it this node was a pure function of the current frame: a camera bearing arrived,
a LiDAR range was looked up inside it, a point was published. Anything that stopped the
camera producing a detection for one frame — a gate rejection, a missed box, a person
walking out of the tilted-down field of view — stopped `people_fusion` too, and the
behaviour layer, which judges "is somebody there" by the **age of the last message**,
read that as the person having gone. Each person is now a constant-velocity Kalman
estimate of their map-frame position (`person_tracking.py`), which buys three things:

- **Coasting.** A missed frame is predicted through rather than lost, for
  `tracking.max_coast_time`.
- **Smoothing.** The range is the noisy half of a fused position — a percentile over a
  bearing wedge, which jumps by tens of centimetres as the wedge slides across a leg, a
  coat and the floor behind. The filter averages that instead of handing the jump to
  Nav2 as a new goal.
- **Velocity**, which is what tells a leading behaviour whether the person is following
  or has stopped.

**Corroboration.** A track is only ever *created* by a camera-corroborated measurement,
because `dets` alone cannot tell a person from any other leg-sized thing the LiDAR sees.
Once created it can be *updated* by a LiDAR-only one, which is what carries a person
through a run of camera rejections — but only for `tracking.max_uncorroborated_time`,
after which the camera has to agree again or the track is dropped. Coasting is memory,
not belief: it must not turn a person who left into a permanent phantom.

**The blind zone.** That rule has one exception, and it is the close-range case the
leading experiment runs into. Demanding corroboration only makes sense where the camera
could have supplied it. The camera sits on the head at ~0.22 m with a ~36° vertical field
of view, so a person nearer than about a metre has nothing in frame the pose network can
call a body — the `proximity_*` gate above buys back the last stretch of that, but not
all of it — and a person outside the horizontal field of view is not in the picture at
all, which is most of a leading trial, since the human being led walks behind the robot.
In neither case is the camera's silence evidence that nobody is there. It is evidence of
nothing.

A track the camera *could not* have seen therefore holds `tracking.max_uncorroborated_time`
instead of spending it, and a consistent run of LiDAR-only measurements there may also
**create** one — the only path by which the LiDAR alone asserts a person, which is why it
costs `tracking.blind_zone_min_hits` rather than `tracking.min_hits`. What keeps it
honest is that the exemption is bounded on three sides:

- **Geometry.** It applies only inside `camera_blind_range` of the head or outside
  `camera_half_fov_deg` of where the head is pointing, computed from the live
  `map → head_link` transform. With no transform there is no exemption at all — the
  stricter behaviour, which is the right way to fail.
- **`tracking.max_coast_time`, unchanged.** The LiDAR has to keep measuring. A track
  nothing is measuring still dies in about a second, blind zone or not, so a person who
  walks away still ends.
- **`tracking.max_blind_zone_time`.** Past it the track is *silenced* rather than
  dropped: it goes on absorbing the LiDAR returns, so the next one cannot simply spawn a
  replacement and start the clock over, and one corroborated frame brings the person
  straight back. Expiring it instead would be no bound at all.

Set both `exempt_*` false to demand corroboration everywhere, which is the behaviour
before this existed; `tracking.blind_zone_creates_tracks: false` keeps the clock hold for
tracks that already exist while restoring "only the camera introduces a person".

Note this is deliberately **not** a filter on the camera's bounding boxes. An
image-space filter would smooth the bearing, which is the half the camera measures well,
and leave the fusion with no bearing at all on the frames the gate rejects — so nothing
would be published either way. `mecanumbot_lidar_detect_people` tracks its own detections
with the same model; `person_tracking.py` is the equivalent one frame further down, where
both sensors have been combined.

#### Locating the ball

The LiDAR cannot help with a ball at all, and that is the whole reason this half of the
node exists. The LDS-02 scans one horizontal plane about 0.12 m up; a tennis ball is
0.067 m across and sits on the floor, so it is **never** in that plane. Looking a range
up inside the ball's bearing wedge — which is exactly what the person path does — would
return the range of whatever is *behind* the ball, and the robot would drive
confidently past the thing it was sent for.

So both the bearing and the range come from the camera, by the two routes a single
camera has. The geometry is `ball_locating.py`; the node is the ROS end of it.

**Apparent size** (`ball.range_source: size`, the default). A tennis ball is a known
0.067 m across and is round, so its box width is its diameter whatever direction it is
seen from — no other object in this system has that property. With a pinhole model,
`range = f · D / d_px`. It degrades gracefully (a ball at 4 m is about 18 px across at
720p through a 60° lens, still measurable), it needs nothing but the lens, and **it does
not care where the camera is pointing** — which matters, because the fetch tree sweeps
the neck while it searches and nothing tells this node what the tilt currently is.

**Ground plane** (`ball.range_source: ground_plane`). The ball rests on the floor, so
its ray meets a known plane — solved for one radius above the floor, since that is where
the ball's centre is. More accurate close in, worthless near the horizon where a small
elevation error is a large range error, and it depends on `ball.camera_z` and
`ball.camera_pitch_deg` being right. Switch to it once those have actually been
measured.

Both are computed whenever they can be, one is published, and a persistent disagreement
is logged **once**, because that is what a wrong camera mounting looks like and there is
otherwise no way to notice it: each estimator on its own produces a perfectly plausible
number from wrong inputs. The other estimator also stands in when the preferred one has
nothing to say — a ball above the horizon has no ground solution but still has a size.
With neither in `[ball.min_range, ball.max_range]`, **nothing is published**; a box of
three pixels is not a marginal measurement of a distant ball, it is not a measurement.

**The mounting numbers are parameters, not TF lookups**, and that is deliberate. The
URDF's `head_link` is rotated 90° about x and `camera_link` another 90° about y, so
neither is an x-forward, z-up frame and neither gives the camera's height above the
floor without unpicking two rotations that were written for the meshes. Two measured
numbers are more honest than a derivation nobody can check by looking at the robot —
and they are logged at startup, so a run says what it assumed.

**The published `z` is the point.** The grabbers are a horizontal pincer whose shafts
sit at about z = 0.034 with a 0.116 m clear gap, and there is no lift, so a ball on a
table or in somebody's hand is one the robot cannot have however close it drives. From
a bearing alone, "the ball is on the table" and "the ball is not here" are the same
fact. This is the same argument `mecanumbot_seek` makes about the Deep3R point cloud's
height, one sensor cheaper.

Tracking is an alpha-beta filter per ball in the map frame — deliberately lighter than
the person tracker's Kalman, because a ball is one cleanly-shaped unambiguous target
with none of a crowd's association problems. What the filter is actually for is the
range noise: an apparent-size range goes as 1/d_px, so one pixel of box jitter at 4 m is
about 20 cm. `ball.tracking.max_coast_time` is short (1 s) on purpose — a ball that has
been picked up must stop being reported quickly, or the robot keeps driving at the floor
where it used to be. Height is smoothed but never given a velocity: somebody picking the
ball up is a step, not a trend.

#### Person boxes from the fetch detector

The fetch detector replaces the pose detector rather than joining it, so its person
boxes have to be able to feed the same fusion. They arrive as pixels and become the same
bearing wedge a `CamPersonDetection` carries, which is all this node ever used of one;
everything downstream — the LiDAR range lookup, the map occlusion check, the tracker —
is unchanged. Running both detectors is harmless: the two wedges land on the same person
and the tracker's association absorbs the duplicate.

The one thing to watch is that the detector and this node hold **two copies of the
camera geometry** and never compare them, because they never talk. A box centred outside
the declared frame is warned about once, which is the cheapest symptom of their having
drifted apart — and until they agree, every bearing and every apparent-size range
computed here is wrong by the ratio between them.

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
| mecanumbot_sensorprocess_smart/mecanumbot_lidar_detect_people.py       | Main LiDAR detection node; the tracker itself is in `lidar_tracking.py`. |
| mecanumbot_sensorprocess_smart/mecanumbot_cam_detect_people.py         | Main camera people detection node.                                   |
| mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_people.py | DeepStream-based camera people detection node.                       |
| mecanumbot_sensorprocess_smart/mecanumbot_onboard_cam_detect_objects.py | DeepStream people-and-balls detection node, for the fetch game.      |
| mecanumbot_sensorprocess_smart/mecanumbot_locate_detections.py         | Detection fusion and localization node; also places the ball.        |
| mecanumbot_sensorprocess_smart/nvinfer_config.py                       | Rendering an nvinfer config for a selected model; shared by both DeepStream nodes. |
| mecanumbot_sensorprocess_smart/object_gating.py                        | Shape, hysteresis and temporal gate for the keypoint-free detector.  |
| mecanumbot_sensorprocess_smart/ball_locating.py                        | Pinhole camera model, the two ball range estimators, and the map-frame ball tracker. |
| mecanumbot_sensorprocess_smart/mecanumbot_detect_tennis.py             | Tennis ball detection node.                                          |
| mecanumbot_sensorprocess_smart/ros4hri_bridge.py                       | ROS4HRI conversion, body ID tracking and `/humans/bodies` publishing. |
| mecanumbot_sensorprocess_smart/person_gating.py                        | Keypoint-evidence, hysteresis and temporal gate for camera detections. |
| mecanumbot_sensorprocess_smart/person_tracking.py                      | Map-frame constant-velocity tracking of the fused detections, and the camera blind zone. |
| mecanumbot_sensorprocess_smart/lidar_tracking.py                       | Scan-frame tracking of the DR-SPAAM detections, the motion gate and its re-seeding. |
| test/test_person_gating.py                                             | Unit tests for the detection gate; run without a ROS graph.          |
| test/test_object_gating.py                                             | Unit tests for the fetch detector's gate; run without a ROS graph.   |
| test/test_ball_locating.py                                             | Unit tests for the ball geometry and its tracker; run without a ROS graph. |
| test/test_person_tracking.py                                           | Unit tests for the map-frame tracker; run without a ROS graph.       |
| test/test_lidar_tracking.py                                            | Unit tests for the DR-SPAAM tracker; run without ROS, torch or `dr_spaam`. |
| launch/perception.launch.py                                            | The pipeline: DR-SPAAM, one camera detector, the fusion, and optionally the camera itself. Included by every behaviour launch file. |
| launch/mecanumbot_peopledetect.launch.py                               | Thin wrapper over `perception.launch.py` under its older name, for a run with no tree. |
| config/lidar_peopledetect_config.yaml                                  | Runtime ROS parameters for node topics and thresholds.               |
| models/dr_spaam_5_on_frog.pth                                          | DR-SPAAM pretrained weights used by the LiDAR detector.              |
| models/dr_spaam.onnx                                                   | ONNX export of the DR-SPAAM model.                                   |
| models/yolo26{n,s,m}-pose.pt                                           | YOLO pose checkpoints; size-independent, used by the Ultralytics camera detector. |
| models/imgsz_640/, models/imgsz_1280/                                  | ONNX exports and their TensorRT engines, one folder per input size; `model_params.imgsz` selects one. |
| models/conv_to_onnx.py                                                 | Exports a checkpoint to ONNX at a given `imgsz`, into that size's folder. |
| models/build_engine.py                                                 | Builds the TensorRT engine for an ONNX under the name `nvinfer` looks for. |
| deepstream_config/config_infer_yolo26_pose.txt                         | Template `nvinfer` configuration; the node renders a copy per model and size. |
| deepstream_config/labels.txt                                           | Class label file referenced by the `nvinfer` config.                 |

The YAML file carries the LiDAR node's parameters, the gate and ROS4HRI blocks for the
DeepStream camera node (under its ROS node name, `mecanumbot_cam_detect_people_ds`) and
the fusion node's tracking block (under `mecanumbot_locate_detections`); the Ultralytics
camera and tennis nodes rely on their in-code defaults unless overridden on the command
line or in the launch file.

## Build and run

```bash
colcon build --symlink-install --packages-select mecanumbot_sensorprocess_smart
source install/setup.bash

# the whole pipeline, on its own (the base launch does not start it)
ros2 launch mecanumbot_sensorprocess_smart perception.launch.py

# ... with the fetch detector instead of the pose one, so balls are found too
ros2 launch mecanumbot_sensorprocess_smart perception.launch.py detector:=fetch

# ... and with the camera published for a recording, which the detector then reads
ros2 launch mecanumbot_sensorprocess_smart perception.launch.py use_camera:=true

# the older name still works; it is a wrapper over the same file
ros2 launch mecanumbot_sensorprocess_smart mecanumbot_peopledetect.launch.py

# individual nodes
ros2 run mecanumbot_sensorprocess_smart mecanumbot_onboard_cam_detect_people
ros2 run mecanumbot_sensorprocess_smart mecanumbot_onboard_cam_detect_objects
ros2 run mecanumbot_sensorprocess_smart mecanumbot_detect_tennis
```

Checking the fetch detector and the located ball:

```bash
ros2 topic echo /mecanumbot/cam_ball_boxes
ros2 topic echo /mecanumbot/ball_detections     # what the fetch tree reads
ros2 topic echo /mecanumbot/ball_fusion         # the same, as poses, for rviz
```

Checking the ROS4HRI output of the DeepStream node:

```bash
ros2 topic echo /humans/bodies/tracked
ros2 topic echo /humans/bodies/<id>/skeleton2d
```

ROS packages are declared in `package.xml` (including `hri_msgs` and `vision_msgs`), so
`rosdep` covers them. The plain-Python and NVIDIA dependencies (`torch`, `ultralytics`, `opencv-python`,
`scipy`, `filterpy`, `transforms3d`, `dr_spaam`, and `pyds`/`gi` for the DeepStream node)
are not, so those still have to be installed by hand.
