#!/usr/bin/env python3
"""
Gating a plain bounding-box detector, where there are no keypoints to judge on.

`person_gating.py` decides whether a box is a person by looking at the skeleton
inside it, because one confidence threshold cannot separate a bean bag from a
half-occluded human. That whole argument depends on the model emitting
keypoints, and the fetch detector does not: it is an ordinary YOLO detector
whose job is to find a person *and* a tennis ball in the same frame, and a
tennis ball has no pose.

So the evidence available here is the box itself -- its score, its size and its
shape -- plus how the box behaves over time. What is left of the three-stage
gate is therefore stages 2 and 3:

* **hysteresis** -- a box has to clear the strict `conf_acquire` to be taken
  seriously and is then kept on the looser `conf_retain`, so a ball rolling
  through a shadow or a person turning away does not drop out and back in;
* **temporal confirmation** -- a candidate has to survive `min_hits` frames
  before it is published, and a confirmed track tolerates `max_missed_time` of
  dropout. A single-frame flicker never reaches the fusion layer.

Shape is doing real work for the ball and almost none for the person. A tennis
ball is round, so its box is square whatever its range -- the aspect bounds
below are the cheapest thing that separates a ball from a yellow stripe on a
floor, a chair leg or a reflection. A person is only loosely "taller than wide",
and close to the camera not even that, so their bounds are wide on purpose: the
work of not believing a coat rack is a person is done downstream, where the
LiDAR has to agree there is something at that bearing.

Pure Python: no ROS, no DeepStream, no numpy, so the part of the detector that
decides what is real stays testable on a machine with no Jetson attached.
"""

from dataclasses import dataclass

# Why a box was refused. Short strings rather than an enum because they are
# drawn on the debug image next to the box that failed.
REASON_ACCEPTED = ""
REASON_SCORE = "score"
REASON_SIZE = "size"
REASON_SHAPE = "shape"
REASON_UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class ClassGate:
    """
    What one class has to look like before the node will publish it.

    One of these per class the detector reports, because a ball and a person are
    different shapes and are worth believing on different evidence. `label` is
    the string that travels out in the message: it is the detector's name for
    the class, and downstream code matches on it rather than on the numeric id,
    which is a property of whichever model was loaded.
    """

    class_id: int
    label: str
    topic: str
    conf_acquire: float = 0.5
    conf_retain: float = 0.3
    # Pixels, on the shorter side of the box. A ball a handful of pixels across
    # is a JPEG artefact with a class label.
    min_size: float = 8.0
    # Bounds on width / height. A ball is square (bounded both ways); a person
    # is only loosely upright, so the lower bound is what does the work there.
    min_aspect: float = 0.0
    max_aspect: float = 1.0e6
    # Frames a box must pass `conf_acquire` before it is published, and how long
    # a confirmed box survives without one.
    min_hits: int = 2
    max_missed_time: float = 0.5
    iou_threshold: float = 0.3


def box_iou(first, second):
    """Intersection over union of two `(xmin, ymin, xmax, ymax)` boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def evaluate_shape(width, height, gate):
    """
    Say whether a box could be this class at all, before any score is looked at.

    Returns `(ok, reason)`. Geometry is checked before confidence because it is
    the cheaper answer and the more honest one: a box of the wrong shape is not
    a marginal detection of the right thing, it is a different thing.
    """
    if width <= 0.0 or height <= 0.0:
        return False, REASON_SIZE
    if min(width, height) < gate.min_size:
        return False, REASON_SIZE
    aspect = width / height
    if aspect < gate.min_aspect or aspect > gate.max_aspect:
        return False, REASON_SHAPE
    return True, REASON_ACCEPTED


class _Track:
    """One box being followed across frames, with its hit count and its age."""

    __slots__ = ("box", "hits", "last_seen", "confirmed")

    def __init__(self, box, now):
        self.box = box
        self.hits = 1
        self.last_seen = now
        self.confirmed = False


class BoxConfirmer:
    """
    Temporal confirmation and hysteresis for one class.

    Greedy IoU association, the same model the person gate uses and for the same
    reason: it is the association a person would do by eye, and a detector that
    needs more than that is a detector whose boxes are not stable enough to act
    on anyway.

    `update()` takes this frame's candidates and returns one `(accepted,
    reason)` per candidate, in the order they were given. A track that is not
    matched this frame is kept until `max_missed_time` has passed -- that is
    what carries a ball through the frames where it is behind a chair leg --
    but a track nothing matches is never itself published: only boxes that
    arrived this frame are.
    """

    def __init__(self, gate):
        self.gate = gate
        self._tracks = []

    def update(self, candidates, now):
        """Judge this frame's `(box, score)` candidates; `box` is a corner tuple."""
        self._expire(now)
        verdicts = []
        claimed = set()

        for box, score in candidates:
            track, index = self._match(box, claimed)
            threshold = (
                self.gate.conf_retain
                if track is not None and track.confirmed
                else self.gate.conf_acquire
            )
            if score < threshold:
                verdicts.append((False, REASON_SCORE))
                continue

            if track is None:
                track = _Track(box, now)
                self._tracks.append(track)
                index = len(self._tracks) - 1
            else:
                track.box = box
                track.hits += 1
                track.last_seen = now
            claimed.add(index)

            if track.hits >= self.gate.min_hits:
                track.confirmed = True
            verdicts.append(
                (True, REASON_ACCEPTED)
                if track.confirmed
                else (False, REASON_UNCONFIRMED)
            )
        return verdicts

    # --- internals -----------------------------------------------------------

    def _expire(self, now):
        self._tracks = [
            track
            for track in self._tracks
            if now - track.last_seen <= self.gate.max_missed_time
        ]

    def _match(self, box, claimed):
        best, best_index, best_iou = None, None, self.gate.iou_threshold
        for index, track in enumerate(self._tracks):
            if index in claimed:
                continue
            overlap = box_iou(box, track.box)
            if overlap >= best_iou:
                best, best_index, best_iou = track, index, overlap
        return best, best_index
