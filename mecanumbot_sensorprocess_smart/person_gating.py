#!/usr/bin/env python3
"""Confidence-based gating for the camera person detector.

The pose network reports one box confidence plus 17 keypoint confidences per
object. Judging a detection on the box confidence alone is what lets padded
props through: a bean bag or a wall-panel occluder can score a plausible box
while producing no joint the network is sure of. Raising that single threshold
to keep the props out then drops partly visible people, who score *lower* than
a confidently-misclassified prop. One threshold cannot separate the two, because
the two failure modes are not ordered along the same axis.

The decision is therefore split in three:

1. **Confidence** - the box score, and how sure the network is of the joints
   it placed: how many cleared the visibility threshold and how confident the
   best one is. *Which* joints they are is deliberately not asked. Props fail
   here whatever their box score, because the network has nothing to place.
2. **Hysteresis** - a detection must clear the strict *acquire* gate to be
   taken seriously, but once confirmed it is kept on the looser *retain* gate.
   This is what stops a person who turns away, is partly occluded or walks
   into poor light from dropping out.
3. **Temporal confirmation** - a candidate must survive ``min_hits`` frames
   before it is published, and a confirmed track tolerates ``max_missed_time``
   of dropout before it has to be re-acquired. Single-frame flickers, in
   either direction, never reach the fusion layer.

Until 2026-10-19 stage 1 also asked for a particular part of the body: a torso
far away, legs close up (a second branch, picked by the box running off the top
of the frame), and briefly a face at the bottom edge. Each of those was an
assumption about which part of a person a 30-degree camera at shin height
frames, and each held only for one tilt of the head. The trees tilt it, so in
the 2026-10-16 leading pilot the gate threw out people it was sure of: chest
and arms at 0.73-0.88 "with no legs", head and shoulders cut off by the bottom
edge at 0.32-0.54 under a whole-body box threshold. The head is now aimed at
the middle of the body the LiDAR expects (``body_gaze`` in
``mecanumbot_movement_behaviours``), the part of a person in frame is whatever
that shows, and the gate takes confidence wherever it is found.

What is left of the close-range geometry is a label: :func:`is_close_range`
still says when a box runs off the top of the frame and fills it, and the node
publishes such a detection as ``close_range``, so a consumer can tell that the
rest of the body is outside the frame rather than undetected.

Everything here is pure Python: no ROS, no DeepStream, no NumPy. That is
deliberate - it keeps the part of the detector that encodes the actual
research decision unit-testable on a development machine, where neither
``pyds`` nor the Jetson camera exists.
"""

import math
from dataclasses import dataclass

# COCO-17 keypoint indices, as emitted by YOLO-pose / DeepStream-Yolo-Pose.
# Mirrors the constants in ros4hri_bridge, duplicated rather than imported so
# that this module stays free of hri_msgs and can be tested without ROS.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

NUM_KEYPOINTS = 17


@dataclass(frozen=True)
class GateConfig:
    """
    Thresholds for the three-stage gate.

    The ``*_acquire`` values decide whether an unknown blob is a person; the
    ``*_retain`` values decide whether an already-confirmed person is still
    there. Retain must be looser than acquire or the hysteresis does nothing.
    """

    # Stage 1/2: box confidence. Low enough for a body the frame cuts off,
    # which the network scores lower than a whole one; the joint confidences
    # below are what keep a prop out at this level.
    box_conf_acquire: float = 0.4
    box_conf_retain: float = 0.3

    # Stage 1: joint confidence. `keypoint_conf` is the visibility threshold
    # for a single joint and is deliberately much lower than the box gates -
    # it decides whether a joint is drawn and used, not whether the detection
    # is a person. Any joint counts, wherever it is on the body.
    keypoint_conf: float = 0.3
    best_keypoint_conf_acquire: float = 0.5
    best_keypoint_conf_retain: float = 0.35
    min_valid_keypoints_acquire: int = 2
    min_valid_keypoints_retain: int = 1

    # Stage 1: box sanity. A sliver of a box is not a detection, and a box far
    # wider than tall is a floor prop more often than it is a body; the limit
    # is loose because a cropped body -- head and shoulders, a pair of legs up
    # close -- is legitimately wider than a standing one.
    min_box_height: float = 40.0
    max_box_aspect_ratio: float = 3.5

    # The close-range label (not a gate any more): a box that starts within
    # `proximity_top_margin` px or `proximity_top_fraction` of the image height
    # of the top edge, whichever is larger, and fills
    # `proximity_min_height_fraction` of the frame, or fills
    # `proximity_dominant_height_fraction` of it wherever its top sits, is a
    # body that carries on above the field of view. Set `proximity_enabled`
    # false to publish every detection as full_body.
    proximity_enabled: bool = True
    proximity_min_height_fraction: float = 0.6
    proximity_top_margin: float = 8.0
    proximity_top_fraction: float = 0.08
    proximity_dominant_height_fraction: float = 0.85

    # Stage 3: temporal confirmation.
    min_hits: int = 2
    max_missed_time: float = 0.5
    iou_threshold: float = 0.3


@dataclass(frozen=True)
class PoseEvidence:
    """
    What was actually observed for one detection, before any thresholding.

    ``box_top`` and ``image_height`` are only needed for the close-range label
    and default to zero, which reads as "frame geometry unknown".
    """

    box_confidence: float
    valid_keypoints: int
    best_keypoint_conf: float
    box_width: float
    box_height: float
    box_top: float = 0.0
    image_height: float = 0.0

    @property
    def aspect_ratio(self):
        """Width over height; > 1 means wider than tall."""
        if self.box_height <= 0.0:
            return float("inf")
        return self.box_width / self.box_height

    @property
    def height_fraction(self):
        """Return the share of the image height the box spans, 0.0 if unknown."""
        if self.image_height <= 0.0:
            return 0.0
        return self.box_height / self.image_height


def evaluate_evidence(
    keypoints,
    box_confidence,
    box_width,
    box_height,
    cfg,
    box_top=0.0,
    image_height=0.0,
):
    """Summarise one detection's confidence and geometry.

    Args:
        keypoints: sequence of 17 ``(confidence, x, y)`` tuples.
        box_confidence: the network's overall confidence for the object.
        box_width: bounding box width in pixels.
        box_height: bounding box height in pixels.
        cfg: the :class:`GateConfig` supplying ``keypoint_conf``.
        box_top: y of the top of the box in pixels; with ``image_height`` it is
            what tells the close-range label that the body leaves the frame.
        image_height: frame height in pixels. Zero leaves the label off.

    Returns:
        A :class:`PoseEvidence`.
    """
    valid = 0
    best = 0.0
    for index in range(min(NUM_KEYPOINTS, len(keypoints))):
        confidence = float(keypoints[index][0])
        if not math.isfinite(confidence):
            continue
        if confidence > best:
            best = confidence
        if confidence > cfg.keypoint_conf:
            valid += 1

    return PoseEvidence(
        box_confidence=float(box_confidence),
        valid_keypoints=valid,
        best_keypoint_conf=best,
        box_width=float(box_width),
        box_height=float(box_height),
        box_top=float(box_top),
        image_height=float(image_height),
    )


FACE_KEYPOINTS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)


def head_at_bottom(
    keypoints, image_height, keypoint_conf=0.5, bottom_fraction=0.2, min_face=2
):
    """
    Return the image x of a head sitting in the bottom band of the frame, or None.

    This is not a detection and the gate does not see it. It is a cue to the
    behaviour layer that somebody low down -- sitting, on a bean bag -- is at
    the edge of a camera that is tilted up to search: their face is in shot and
    the rest of them is below it, so the gate has little to go on until the
    head tilts down (2026-09-30, the 08:46 run: the face keypoints were at
    0.85-0.99 while the box was turned down at 0.55).

    At least `min_face` of the nose, eyes and ears have to be found with
    `keypoint_conf`, and their mean y has to lie in the bottom `bottom_fraction`
    of the frame. Joints the model did not find are NaN (DeepStream) or (0, 0)
    (Ultralytics); neither counts.

    Args:
        keypoints: sequence of 17 ``(confidence, x, y)`` tuples, in pixels.
        image_height: frame height in pixels.
        keypoint_conf: how sure the model has to be of a face joint.
        bottom_fraction: the band, as a fraction of the frame height.
        min_face: how many face joints have to be in it.

    Returns:
        The mean x of the face joints in pixels, or None.
    """
    if image_height <= 0.0:
        return None
    xs, ys = [], []
    for index in FACE_KEYPOINTS:
        if index >= len(keypoints):
            continue
        confidence, x, y = (float(v) for v in keypoints[index])
        if confidence < keypoint_conf or not (math.isfinite(x) and math.isfinite(y)):
            continue
        if x == 0.0 and y == 0.0:
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < min_face:
        return None
    if sum(ys) / len(ys) < (1.0 - bottom_fraction) * image_height:
        return None
    return sum(xs) / len(xs)


def is_close_range(evidence, cfg):
    """
    Say whether the box geometry means the body runs out of the top of the frame.

    A label, not a gate: the node publishes such a detection as
    ``close_range``. It is geometric on purpose -- it describes the frame, not
    the joints. A box that starts at the top edge and fills most of the image
    height continues above the field of view. "Starts at the top edge" is
    forgiving, because the box is drawn round the body the network found, and
    on loose clothing -- a long skirt, a coat -- its top lands tens of pixels
    inside the frame; so either of these is enough:

    * the top of the box is within ``proximity_top_margin`` px or
      ``proximity_top_fraction`` of the image height of the top edge,
      whichever is larger, **and** the box fills
      ``proximity_min_height_fraction`` of the frame; or
    * the box fills ``proximity_dominant_height_fraction`` of the frame,
      wherever its top happens to sit.
    """
    if not cfg.proximity_enabled or evidence.image_height <= 0.0:
        return False
    if evidence.height_fraction >= cfg.proximity_dominant_height_fraction:
        return True
    top_margin = max(
        cfg.proximity_top_margin,
        cfg.proximity_top_fraction * evidence.image_height,
    )
    if evidence.box_top > top_margin:
        return False
    return evidence.height_fraction >= cfg.proximity_min_height_fraction


def check_evidence(evidence, cfg, strict):
    """Decide whether the evidence looks like a person.

    Args:
        evidence: a :class:`PoseEvidence`.
        cfg: the :class:`GateConfig` to apply.
        strict: ``True`` applies the acquire thresholds (used for detections
            that are not yet a confirmed track), ``False`` the retain ones.

    Returns:
        ``(passed, reason)``. ``reason`` names the first failed check, or
        ``'ok'``, and is what the debug overlay and the logs report.
    """
    if strict:
        box_conf = cfg.box_conf_acquire
        best_conf = cfg.best_keypoint_conf_acquire
        min_valid = cfg.min_valid_keypoints_acquire
    else:
        box_conf = cfg.box_conf_retain
        best_conf = cfg.best_keypoint_conf_retain
        min_valid = cfg.min_valid_keypoints_retain

    if evidence.box_confidence <= box_conf:
        return False, f"box-conf {evidence.box_confidence:.2f}<={box_conf:.2f}"
    if evidence.box_height < cfg.min_box_height:
        return False, f"box-height {evidence.box_height:.0f}px"
    if evidence.aspect_ratio > cfg.max_box_aspect_ratio:
        return False, f"box-shape {evidence.aspect_ratio:.2f}"
    if evidence.valid_keypoints < min_valid:
        return False, f"keypoints {evidence.valid_keypoints}<{min_valid}"
    if evidence.best_keypoint_conf < best_conf:
        return (
            False,
            f"weak-pose {evidence.best_keypoint_conf:.2f}<{best_conf:.2f}",
        )
    return True, "ok"


def iou(a, b):
    """Intersection over union of two ``(xmin, ymin, xmax, ymax)`` boxes."""
    inter_xmin = max(a[0], b[0])
    inter_ymin = max(a[1], b[1])
    inter_xmax = min(a[2], b[2])
    inter_ymax = min(a[3], b[3])
    inter_w = inter_xmax - inter_xmin
    inter_h = inter_ymax - inter_ymin
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0
    intersection = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


class _Track:
    """One blob being followed across frames."""

    __slots__ = ("box", "hits", "last_seen", "confirmed")

    def __init__(self, box, now):
        self.box = box
        self.hits = 1
        self.last_seen = now
        self.confirmed = False


class DetectionConfirmer:
    """Turns per-frame candidates into confirmed detections.

    Holds the hysteresis and the temporal confirmation: a blob has to pass the
    strict gate for ``min_hits`` frames before anything is published, after
    which it is kept on the loose gate and survives ``max_missed_time`` of
    dropout. Association between frames is greedy IoU, which is sufficient
    because the detector runs at camera rate and the number of people in the
    experiment is small.

    Not thread-safe: it is driven from the DeepStream buffer probe, which is a
    single streaming thread.
    """

    def __init__(self, cfg):
        """Store the config and start with no tracks."""
        self._cfg = cfg
        self._tracks = []

    @property
    def track_count(self):
        """Return how many blobs are currently being followed, confirmed or not."""
        return len(self._tracks)

    def _associate(self, boxes):
        """Greedily match this frame's boxes to existing tracks.

        Returns a list, parallel to ``boxes``, of the matched ``_Track`` or
        ``None``.
        """
        pairs = []
        for det_index, box in enumerate(boxes):
            for track_index, track in enumerate(self._tracks):
                overlap = iou(box, track.box)
                if overlap >= self._cfg.iou_threshold:
                    pairs.append((overlap, det_index, track_index))
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        matched = [None] * len(boxes)
        taken = set()
        for _, det_index, track_index in pairs:
            if matched[det_index] is None and track_index not in taken:
                matched[det_index] = self._tracks[track_index]
                taken.add(track_index)
        return matched

    def update(self, candidates, now):
        """Advance the tracks by one frame and say which candidates to publish.

        Args:
            candidates: list of ``(box, evidence)`` pairs for this frame, where
                ``box`` is ``(xmin, ymin, xmax, ymax)`` in pixels.
            now: current time in seconds; only differences matter.

        Returns:
            List of ``(publish, reason)``, parallel to ``candidates``.
        """
        self._tracks = [
            track
            for track in self._tracks
            if now - track.last_seen <= self._cfg.max_missed_time
        ]

        boxes = [box for box, _ in candidates]
        matched = self._associate(boxes)

        results = []
        for (box, evidence), track in zip(candidates, matched):
            # A confirmed track is re-tested on the loose gate; anything else
            # has to clear the strict one. Checking strict first means a blob
            # that is convincing on its own merits is never held back by a
            # track it happens to overlap.
            passed, reason = check_evidence(evidence, self._cfg, strict=True)
            if not passed and track is not None and track.confirmed:
                passed, reason = check_evidence(evidence, self._cfg, strict=False)

            if not passed:
                # Deliberately no state update: an unconvincing frame neither
                # refreshes a track nor starts one, so props never accumulate
                # hits and a fading person expires on `max_missed_time`.
                results.append((False, reason))
                continue

            if track is None:
                track = _Track(box, now)
                self._tracks.append(track)
            else:
                track.box = box
                track.last_seen = now
                track.hits += 1

            if track.hits >= self._cfg.min_hits:
                track.confirmed = True

            if track.confirmed:
                results.append((True, reason))
            else:
                results.append(
                    (False, f"unconfirmed {track.hits}/{self._cfg.min_hits}")
                )

        return results
