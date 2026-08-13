#!/usr/bin/env python3
"""Evidence-based gating for the camera person detector.

The pose network reports one box confidence plus 17 keypoint confidences per
object. Judging a detection on the box confidence alone is what lets padded
props through: a bean bag or a wall-panel occluder can score a plausible box
while producing no coherent skeleton at all. Raising that single threshold to
keep the props out then drops partly occluded people, who score *lower* than a
confidently-misclassified prop. One threshold cannot separate the two, because
the two failure modes are not ordered along the same axis.

The decision is therefore split in three:

1. **Keypoint evidence** - how much of a body was actually found: how many
   keypoints cleared the visibility threshold, how confident the best one is,
   and whether a torso (shoulders/hips) is present. Props fail here whatever
   their box score, because the network has no body parts to put anywhere.
2. **Hysteresis** - a detection must clear the strict *acquire* gate to be
   taken seriously, but once confirmed it is kept on the looser *retain* gate.
   This is what stops a person who turns away, is partly occluded or walks
   into poor light from dropping out.
3. **Temporal confirmation** - a candidate must survive ``min_hits`` frames
   before it is published, and a confirmed track tolerates ``max_missed_time``
   of dropout before it has to be re-acquired. Single-frame flickers, in
   either direction, never reach the fusion layer.

Stage 1 has a second branch for people standing **close to the robot**. The
onboard camera is mounted on the head at roughly 0.2 m, i.e. shin height, with
a vertical field of view of about 36 degrees, so the band of the world it sees
at 0.6 m spans only about 0.0-0.4 m: a person that close shows knees and
calves and nothing else. Demanding a torso of them is demanding something the
optics cannot deliver, so a close-range body would be rejected however good the
detection was - and close range is exactly where a person matters most.

Close range is recognised geometrically rather than assumed: the box has to run
off the *top* of the frame (the body continues above the field of view) and
fill most of the frame's height. A detection that does gets the torso
requirement replaced by a **lower-body** one - hips, knees and ankles - and
looser keypoint counts. The compensating guard is that the geometry and the
keypoints have to agree: a wall panel or a bean bag pushed up against the
camera also fills the frame, but it still has no leg to place, and
``best_keypoint_conf_*`` is unchanged in this branch.

Everything here is pure Python: no ROS, no DeepStream, no NumPy. That is
deliberate - it keeps the part of the detector that encodes the actual
research decision unit-testable on a development machine, where neither
``pyds`` nor the Jetson camera exists.
"""

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

# The joints that make a blob a body. A bean bag or a panel may well produce a
# scattering of confident-looking limb keypoints, but the network has to place
# a shoulder/hip rectangle somewhere consistent to claim a torso.
TORSO_KEYPOINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# What is left of a person once they are too close for the torso to be in
# frame. Hips are in both sets on purpose: at the near edge of the close-range
# regime they are the last torso joint still visible, and at the far edge the
# first to come back.
LOWER_BODY_KEYPOINTS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the three-stage gate.

    The ``*_acquire`` values decide whether an unknown blob is a person; the
    ``*_retain`` values decide whether an already-confirmed person is still
    there. Retain must be looser than acquire or the hysteresis does nothing.
    """

    # Stage 1/2: box confidence.
    box_conf_acquire: float = 0.6
    box_conf_retain: float = 0.35

    # Stage 1: keypoint evidence. `keypoint_conf` is the visibility threshold
    # for a single joint and is deliberately much lower than the box gates -
    # it decides whether a joint is drawn and used, not whether the detection
    # is a person.
    keypoint_conf: float = 0.3
    best_keypoint_conf_acquire: float = 0.7
    best_keypoint_conf_retain: float = 0.5
    min_valid_keypoints_acquire: int = 4
    min_valid_keypoints_retain: int = 2
    min_torso_keypoints_acquire: int = 2
    min_torso_keypoints_retain: int = 1

    # Stage 1: box geometry. People are taller than they are wide; floor props
    # are squat. Applied in both modes - a bean bag stays a bean bag.
    min_box_height: float = 40.0
    max_box_aspect_ratio: float = 1.6

    # Stage 1b: close range. A person nearer than roughly two metres has no
    # torso in the frame of a shin-height camera, so for boxes whose geometry
    # says the body leaves the top of the image these thresholds replace the
    # torso and keypoint-count ones above. `proximity_min_height_fraction` is
    # a fraction of the image height, `proximity_top_margin` a pixel tolerance
    # on "touches the top edge". Set `proximity_enabled` false to go back to
    # the torso-only behaviour.
    proximity_enabled: bool = True
    proximity_min_height_fraction: float = 0.6
    proximity_top_margin: float = 8.0
    # A cropped body is a partial body, and the network scores it lower than a
    # whole one, so the box gates are relaxed here too.
    proximity_box_conf_acquire: float = 0.5
    proximity_box_conf_retain: float = 0.3
    proximity_min_valid_keypoints_acquire: int = 2
    proximity_min_valid_keypoints_retain: int = 1
    # Of the six hip/knee/ankle joints. This is the check that keeps the
    # relaxation honest: a frame-filling prop still cannot produce a leg.
    proximity_min_lower_body_acquire: int = 2
    proximity_min_lower_body_retain: int = 1
    # Legs seen from 0.5 m can be wider than the slice of them that fits in
    # the frame is tall, so the standing-person shape check is loosened.
    proximity_max_box_aspect_ratio: float = 2.5

    # Stage 3: temporal confirmation.
    min_hits: int = 2
    max_missed_time: float = 0.5
    iou_threshold: float = 0.3


@dataclass(frozen=True)
class PoseEvidence:
    """What was actually observed for one detection, before any thresholding.

    ``box_top`` and ``image_height`` are only needed by the close-range branch
    and default to zero, which reads as "frame geometry unknown" and leaves
    that branch switched off.
    """

    box_confidence: float
    valid_keypoints: int
    torso_keypoints: int
    best_keypoint_conf: float
    box_width: float
    box_height: float
    lower_body_keypoints: int = 0
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
    """Summarise one detection's keypoint and geometry evidence.

    Args:
        keypoints: sequence of 17 ``(confidence, x, y)`` tuples.
        box_confidence: the network's overall confidence for the object.
        box_width: bounding box width in pixels.
        box_height: bounding box height in pixels.
        cfg: the :class:`GateConfig` supplying ``keypoint_conf``.
        box_top: y of the top of the box in pixels; with ``image_height`` it is
            what tells the close-range branch that the body leaves the frame.
        image_height: frame height in pixels. Zero disables the close-range
            branch for this detection.

    Returns:
        A :class:`PoseEvidence`.
    """
    valid = 0
    torso = 0
    lower_body = 0
    best = 0.0
    for index in range(min(NUM_KEYPOINTS, len(keypoints))):
        confidence = float(keypoints[index][0])
        if confidence > best:
            best = confidence
        if confidence > cfg.keypoint_conf:
            valid += 1
            if index in TORSO_KEYPOINTS:
                torso += 1
            if index in LOWER_BODY_KEYPOINTS:
                lower_body += 1

    return PoseEvidence(
        box_confidence=float(box_confidence),
        valid_keypoints=valid,
        torso_keypoints=torso,
        best_keypoint_conf=best,
        box_width=float(box_width),
        box_height=float(box_height),
        lower_body_keypoints=lower_body,
        box_top=float(box_top),
        image_height=float(image_height),
    )


def is_close_range(evidence, cfg):
    """Say whether the box geometry means the body runs out of the frame.

    The test is deliberately geometric and not keypoint-based: it decides
    *which* gate a detection is judged by, so inferring it from the very
    keypoints the close-range gate then relaxes would make the relaxation
    self-justifying. A body whose box starts at the top edge of the image and
    fills most of its height continues above the field of view, which for a
    camera at shin height means the person is standing close.
    """
    if not cfg.proximity_enabled or evidence.image_height <= 0.0:
        return False
    if evidence.box_top > cfg.proximity_top_margin:
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
        ``'ok'``, and is what the debug overlay and the logs report. Failures
        on the close-range branch are prefixed ``near-`` so it is visible on
        the overlay which of the two gates a box was judged by.
    """
    close_range = is_close_range(evidence, cfg)

    if close_range:
        max_aspect = cfg.proximity_max_box_aspect_ratio
        if strict:
            box_conf = cfg.proximity_box_conf_acquire
            min_valid = cfg.proximity_min_valid_keypoints_acquire
            min_body = cfg.proximity_min_lower_body_acquire
            best_conf = cfg.best_keypoint_conf_acquire
        else:
            box_conf = cfg.proximity_box_conf_retain
            min_valid = cfg.proximity_min_valid_keypoints_retain
            min_body = cfg.proximity_min_lower_body_retain
            best_conf = cfg.best_keypoint_conf_retain
    else:
        max_aspect = cfg.max_box_aspect_ratio
        if strict:
            box_conf = cfg.box_conf_acquire
            best_conf = cfg.best_keypoint_conf_acquire
            min_valid = cfg.min_valid_keypoints_acquire
            min_body = cfg.min_torso_keypoints_acquire
        else:
            box_conf = cfg.box_conf_retain
            best_conf = cfg.best_keypoint_conf_retain
            min_valid = cfg.min_valid_keypoints_retain
            min_body = cfg.min_torso_keypoints_retain

    prefix = "near-" if close_range else ""

    if evidence.box_confidence <= box_conf:
        return (
            False,
            f"{prefix}box-conf {evidence.box_confidence:.2f}<={box_conf:.2f}",
        )
    if evidence.box_height < cfg.min_box_height:
        return False, f"{prefix}box-height {evidence.box_height:.0f}px"
    if evidence.aspect_ratio > max_aspect:
        return False, f"{prefix}box-shape {evidence.aspect_ratio:.2f}"
    if evidence.valid_keypoints < min_valid:
        return (
            False,
            f"{prefix}keypoints {evidence.valid_keypoints}<{min_valid}",
        )
    # The same slot in the two branches: what part of a body has to be there.
    # Far away that is a torso, close up it is a leg, because close up a torso
    # is above the top of the image.
    found_body = (
        evidence.lower_body_keypoints if close_range else evidence.torso_keypoints
    )
    if found_body < min_body:
        label = "legs" if close_range else "torso"
        return False, f"{prefix}{label} {found_body}<{min_body}"
    if evidence.best_keypoint_conf < best_conf:
        return (
            False,
            f"{prefix}weak-pose {evidence.best_keypoint_conf:.2f}<{best_conf:.2f}",
        )
    return True, "near-ok" if close_range else "ok"


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
                # `reason` rather than a literal 'ok': it carries which of the
                # two gates accepted the box, which the overlay reports.
                results.append((True, reason))
            else:
                results.append(
                    (False, f"unconfirmed {track.hits}/{self._cfg.min_hits}")
                )

        return results
