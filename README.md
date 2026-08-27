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
only). Run one or the other — both publish `cam_people_detections`. The DeepStream
variant additionally publishes the ROS4HRI (REP-155) `/humans/bodies` tree; see
[ROS4HRI (REP-155) output](#ros4hri-rep-155-output).

## Pipeline

```text
scan ──► mecanumbot_lidar_detect_people ──► dr_spaam/dets ─┐
                                                           ├─► mecanumbot_locate_detections ──► people_fusion
camera ─► mecanumbot_cam_detect_people ──► cam_people_detections ┘
                                       └─► /humans/bodies/…  (ROS4HRI, DeepStream variant)
```

`people_fusion` (`geometry_msgs/PoseArray`, `map` frame) is what the behaviour trees
in `mecanumbot_behaviours` consume. The LiDAR node additionally publishes
`subject_pose` directly for the leading behaviours. The ROS4HRI topics are a parallel,
standards-compliant output for external HRI tooling; nothing inside this repository
consumes them yet.

## Launch file

`launch/mecanumbot_peopledetect.launch.py` starts the LiDAR people detector, the
camera people detector (with `from_topic` forced to `true`), and the
detection-localization node in the `mecanumbot` namespace, all with
`config/lidar_peopledetect_config.yaml` applied. Node names must match the YAML's
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
| `stride`                    | `2`                      | Detector stride passed to DR-SPAAM. See *GPU load control* below.                     |
| `scan_topic`                | `/mecanumbot/scan`       | Input laser scan topic.                                                               |
| `detections_topic`          | `dets`                   | Output detection topic (`dr_spaam/dets` in the shipped YAML).                         |
| `rviz_topic`                | `dets_marker`            | Marker topic name (unused while marker publishing is disabled).                       |
| `leading_mode`              | `true`                   | Enables the `subject_pose` publisher.                                                 |
| `obstacle_exclusion_radius` | `0.2`                    | Inflation radius in metres applied to the keepout mask.                               |
| `detection_frame`           | `base_scan`              | Accepts `base_scan` or `map`; anything else falls back to `base_scan` with a warning. |

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
| `keypoint_scaling`            | `letterbox`                   | How to invert the `nvinfer` input resize: `letterbox`, `stretch`, or `auto`.           |
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
| `proximity_top_margin`                | `8.0`   | Pixels from the top edge within which a box counts as clipped by it.           |
| `proximity_box_conf_acquire`          | `0.5`   | Box confidence to acquire a close-range (cropped, therefore lower-scoring) body. |
| `proximity_box_conf_retain`           | `0.3`   | ... and to retain it.                                                          |
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
would make the relaxation self-justifying. A detection is close-range when its box
starts within `proximity_top_margin` of the **top** edge (the body carries on above the
field of view) and spans at least `proximity_min_height_fraction` of the frame height.
For this camera those defaults amount to "nearer than roughly 3.4 m", which overlaps the
2.3 m at which hips appear — the two branches cover the whole approach with no distance
at which a person falls between them.

On that branch the torso requirement is replaced by a **lower-body** one
(`proximity_min_lower_body_*`, counted over hips, knees and ankles), the keypoint counts
drop to `proximity_min_valid_keypoints_*`, the box-confidence gates drop to
`proximity_box_conf_*` because a cropped body scores lower than a whole one, and the
shape check loosens to `proximity_max_box_aspect_ratio` — legs seen from half a metre
can be wider than the slice of them that fits in the frame is tall.

What keeps the relaxation honest is that the geometry and the keypoints have to agree. A
wall panel or a bean bag pushed up against the camera has exactly the same box geometry,
but it still cannot produce a leg, and `best_keypoint_conf_*` is *not* relaxed on this
branch. Small floor props are unaffected either way: they are not clipped by the top of
the frame, so they are still judged on the ordinary gate.

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

Note that the shipped `config_infer_yolo26_pose.txt` sets `maintain-aspect-ratio=0`
(stretch) while the default `letterbox` mapping assumes padding. With a 1280x720 source
and a 640x640 network the two disagree in y by 280 px (`2*ky - 280` versus `1.125*ky`),
while agreeing in x. The default is left at `letterbox` so behaviour does not change
silently; switch to `auto` or `stretch` to make the mapping consistent with the config.

### Behavior

- Announces its input geometry at startup: the requested frame size and source, the
  `nvstreammux` size and the horizontal FOV, plus the network input size — read from
  `infer-dims` if the nvinfer config sets it, otherwise reported from the first inferred
  frame, since without `infer-dims` nvinfer takes the shape from the model itself. The
  size the source *actually* delivered is logged from the first frame as well, and a
  mismatch against `camera_params` is a warning when the aspect ratios differ: every
  bearing is derived from a keypoint's x within `camera_width`, so a frame that
  `nvstreammux` had to distort into that shape makes all of them wrong.
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
| mecanumbot_sensorprocess_smart/ros4hri_bridge.py                       | ROS4HRI conversion, body ID tracking and `/humans/bodies` publishing. |
| mecanumbot_sensorprocess_smart/person_gating.py                        | Keypoint-evidence, hysteresis and temporal gate for camera detections. |
| test/test_person_gating.py                                             | Unit tests for the detection gate; run without a ROS graph.          |
| launch/mecanumbot_peopledetect.launch.py                               | Launches the people-detection pipeline with shared parameters.       |
| config/lidar_peopledetect_config.yaml                                  | Runtime ROS parameters for node topics and thresholds.               |
| models/dr_spaam_5_on_frog.pth                                          | DR-SPAAM pretrained weights used by the LiDAR detector.              |
| models/dr_spaam.onnx                                                   | ONNX export of the DR-SPAAM model.                                   |
| models/yolo26n-pose.pt                                                 | YOLO pose weights used by the Ultralytics camera detector.           |
| models/yolo26n-pose.onnx, models/yolo26n-pose.onnx_b1_gpu0_fp16.engine | ONNX and TensorRT FP16 build used by the DeepStream detector.        |
| deepstream_config/config_infer_yolo26_pose.txt                         | `nvinfer` configuration for the DeepStream pose model.               |
| deepstream_config/labels.txt                                           | Class label file referenced by the `nvinfer` config.                 |

The YAML file carries the LiDAR node's parameters plus the ROS4HRI block for the
DeepStream camera node (under its ROS node name, `mecanumbot_cam_detect_people_ds`); the
Ultralytics camera, fusion and tennis nodes rely on their in-code defaults unless
overridden on the command line or in the launch file.

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

Checking the ROS4HRI output of the DeepStream node:

```bash
ros2 topic echo /humans/bodies/tracked
ros2 topic echo /humans/bodies/<id>/skeleton2d
```

ROS packages are declared in `package.xml` (including `hri_msgs`), so `rosdep` covers
them. The plain-Python and NVIDIA dependencies (`torch`, `ultralytics`, `opencv-python`,
`scipy`, `filterpy`, `transforms3d`, `dr_spaam`, and `pyds`/`gi` for the DeepStream node)
are not, so those still have to be installed by hand.
