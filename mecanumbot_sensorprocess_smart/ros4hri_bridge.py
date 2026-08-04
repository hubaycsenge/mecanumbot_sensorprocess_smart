#!/usr/bin/env python3
"""ROS4HRI (REP-155) publishing helpers.

The detection backend is irrelevant to ROS4HRI: the standard only fixes the
message types (``hri_msgs``) and the topic layout, so any pipeline able to
produce 2D bodies can feed it -- including NVIDIA DeepStream, whose ``nvinfer``
metadata is converted here into ``hri_msgs`` inside the buffer probe.

This module owns the parts that are pure ROS4HRI bookkeeping and therefore have
nothing to do with the detector:

* remapping COCO-17 pose keypoints (what YOLO-pose emits) onto the OpenPose
  COCO-18 order that ``hri_msgs/Skeleton2D`` mandates, synthesising the ``NECK``
  joint that COCO-17 does not have;
* assigning stable, anonymous body IDs across frames;
* creating and tearing down the ``/humans/bodies/<id>/...`` publishers as bodies
  appear and disappear, and keeping ``/humans/bodies/tracked`` in sync.
"""

import math
import random
import string
import threading

from hri_msgs.msg import (
    IdsList,
    NormalizedPointOfInterest2D,
    NormalizedRegionOfInterest2D,
    Skeleton2D,
)

# COCO-17 keypoint order, as emitted by YOLO-pose / DeepStream-Yolo-Pose.
COCO17_NOSE = 0
COCO17_LEFT_EYE = 1
COCO17_RIGHT_EYE = 2
COCO17_LEFT_EAR = 3
COCO17_RIGHT_EAR = 4
COCO17_LEFT_SHOULDER = 5
COCO17_RIGHT_SHOULDER = 6
COCO17_LEFT_ELBOW = 7
COCO17_RIGHT_ELBOW = 8
COCO17_LEFT_WRIST = 9
COCO17_RIGHT_WRIST = 10
COCO17_LEFT_HIP = 11
COCO17_RIGHT_HIP = 12
COCO17_LEFT_KNEE = 13
COCO17_RIGHT_KNEE = 14
COCO17_LEFT_ANKLE = 15
COCO17_RIGHT_ANKLE = 16

# hri_msgs/Skeleton2D follows the OpenPose COCO-18 convention, which orders the
# joints differently from COCO-17 and adds NECK. Every Skeleton2D index except
# NECK maps to exactly one COCO-17 index; NECK is synthesised from the shoulders.
SKELETON2D_FROM_COCO17 = {
    Skeleton2D.NOSE: COCO17_NOSE,
    Skeleton2D.RIGHT_SHOULDER: COCO17_RIGHT_SHOULDER,
    Skeleton2D.RIGHT_ELBOW: COCO17_RIGHT_ELBOW,
    Skeleton2D.RIGHT_WRIST: COCO17_RIGHT_WRIST,
    Skeleton2D.LEFT_SHOULDER: COCO17_LEFT_SHOULDER,
    Skeleton2D.LEFT_ELBOW: COCO17_LEFT_ELBOW,
    Skeleton2D.LEFT_WRIST: COCO17_LEFT_WRIST,
    Skeleton2D.RIGHT_HIP: COCO17_RIGHT_HIP,
    Skeleton2D.RIGHT_KNEE: COCO17_RIGHT_KNEE,
    Skeleton2D.RIGHT_ANKLE: COCO17_RIGHT_ANKLE,
    Skeleton2D.LEFT_HIP: COCO17_LEFT_HIP,
    Skeleton2D.LEFT_KNEE: COCO17_LEFT_KNEE,
    Skeleton2D.LEFT_ANKLE: COCO17_LEFT_ANKLE,
    Skeleton2D.LEFT_EYE: COCO17_LEFT_EYE,
    Skeleton2D.RIGHT_EYE: COCO17_RIGHT_EYE,
    Skeleton2D.LEFT_EAR: COCO17_LEFT_EAR,
    Skeleton2D.RIGHT_EAR: COCO17_RIGHT_EAR,
}

SKELETON2D_NUM_JOINTS = 18

# A body ID becomes a topic name token in /humans/bodies/<id>/..., and a ROS
# topic token must not start with a digit, so the first character is drawn from
# letters only.
_ID_FIRST_ALPHABET = string.ascii_lowercase
_ID_ALPHABET = string.ascii_lowercase + string.digits


def _clamp01(value):
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _invalid_point():
    """Build a point ROS4HRI consumers must ignore: in-range coordinates, zero confidence."""
    return NormalizedPointOfInterest2D(x=0.0, y=0.0, c=0.0)


def skeleton2d_from_coco17(
    keypoints, frame_width, frame_height, min_confidence, header=None
):
    """Build a ``hri_msgs/Skeleton2D`` from COCO-17 keypoints.

    Args:
        keypoints: sequence of 17 ``(confidence, x_pixel, y_pixel)`` tuples, in
            the coordinate frame of the *undistorted source image* (i.e. after
            any letterbox padding has already been removed).
        frame_width: source image width in pixels, used to normalise x.
        frame_height: source image height in pixels, used to normalise y.
        min_confidence: keypoints at or below this confidence are emitted as
            ``c = 0.0`` rather than as coordinates. ROS4HRI requires x and y to
            stay inside [0, 1], so unavailable joints cannot be signalled with
            NaN the way the mecanumbot-native messages do.
        header: optional ``std_msgs/Header`` to stamp the message with.

    Returns:
        A fully populated ``Skeleton2D`` with all 18 joints present.
    """
    joints = [_invalid_point() for _ in range(SKELETON2D_NUM_JOINTS)]

    for skeleton_idx, coco_idx in SKELETON2D_FROM_COCO17.items():
        confidence, x_px, y_px = keypoints[coco_idx]
        if confidence > min_confidence:
            joints[skeleton_idx] = NormalizedPointOfInterest2D(
                x=_clamp01(x_px / frame_width),
                y=_clamp01(y_px / frame_height),
                c=float(confidence),
            )

    # NECK is not a COCO-17 joint; OpenPose defines it as the midpoint between
    # the shoulders, so it is only meaningful when both shoulders were found.
    left_conf, left_x, left_y = keypoints[COCO17_LEFT_SHOULDER]
    right_conf, right_x, right_y = keypoints[COCO17_RIGHT_SHOULDER]
    if left_conf > min_confidence and right_conf > min_confidence:
        joints[Skeleton2D.NECK] = NormalizedPointOfInterest2D(
            x=_clamp01(0.5 * (left_x + right_x) / frame_width),
            y=_clamp01(0.5 * (left_y + right_y) / frame_height),
            c=float(min(left_conf, right_conf)),
        )

    msg = Skeleton2D()
    if header is not None:
        msg.header = header
    msg.skeleton = joints
    return msg


def normalized_roi(
    left, top, width, height, frame_width, frame_height, confidence=0.0, header=None
):
    """Build a ``hri_msgs/NormalizedRegionOfInterest2D`` from a pixel bounding box."""
    msg = NormalizedRegionOfInterest2D()
    if header is not None:
        msg.header = header
    msg.xmin = _clamp01(left / frame_width)
    msg.ymin = _clamp01(top / frame_height)
    msg.xmax = _clamp01((left + width) / frame_width)
    msg.ymax = _clamp01((top + height) / frame_height)
    msg.c = float(confidence)
    return msg


class BodyIdTracker:
    """Assigns stable ROS4HRI body IDs to bounding boxes across frames.

    ROS4HRI identifies bodies by a string ID that has to persist for as long as
    the same body is being observed, which a stateless per-frame detector cannot
    provide. Greedy IoU association is enough here: the detector runs at camera
    rate, so frame-to-frame overlap is large, and the number of simultaneous
    bodies is small. IDs are short random strings, as REP-155 asks for IDs that
    carry no personal information, and always start with a letter so that they
    are valid ROS topic name tokens.
    """

    def __init__(self, iou_threshold=0.3, max_missed_time=0.5, id_length=5):
        """Set the association threshold, how long an unseen ID is held, and its length."""
        self._iou_threshold = iou_threshold
        self._max_missed_time = max_missed_time
        self._id_length = id_length
        self._tracks = (
            {}
        )  # body_id -> {'box': (xmin, ymin, xmax, ymax), 'last_seen': float}

    def _new_id(self):
        while True:
            body_id = random.choice(_ID_FIRST_ALPHABET) + "".join(
                random.choice(_ID_ALPHABET) for _ in range(self._id_length - 1)
            )
            if body_id not in self._tracks:
                return body_id

    @staticmethod
    def _iou(a, b):
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

    def update(self, boxes, now):
        """Match ``boxes`` against the active tracks and return one ID per box.

        Args:
            boxes: list of ``(xmin, ymin, xmax, ymax)`` boxes, any consistent unit.
            now: current time in seconds, used to expire unseen tracks.

        Returns:
            List of body ID strings, parallel to ``boxes``.
        """
        for body_id in [
            i
            for i, t in self._tracks.items()
            if now - t["last_seen"] > self._max_missed_time
        ]:
            del self._tracks[body_id]

        candidates = []
        for det_idx, box in enumerate(boxes):
            for body_id, track in self._tracks.items():
                iou = self._iou(box, track["box"])
                if iou >= self._iou_threshold:
                    candidates.append((iou, det_idx, body_id))
        candidates.sort(key=lambda c: c[0], reverse=True)

        assigned = [None] * len(boxes)
        taken_ids = set()
        for _, det_idx, body_id in candidates:
            if assigned[det_idx] is None and body_id not in taken_ids:
                assigned[det_idx] = body_id
                taken_ids.add(body_id)

        for det_idx, box in enumerate(boxes):
            if assigned[det_idx] is None:
                assigned[det_idx] = self._new_id()
            self._tracks[assigned[det_idx]] = {"box": box, "last_seen": now}

        return assigned


class Ros4HriBodyBroadcaster:
    """Publishes the ``/humans/bodies`` half of the ROS4HRI topic tree.

    Per-body topics have to be created and destroyed at runtime, which must not
    happen on a detector thread -- DeepStream calls its buffer probes on a
    GStreamer streaming thread, not on the ROS executor. Detections are therefore
    handed over through :meth:`submit` and all rclpy entity management happens in
    :meth:`flush`, which the owning node drives from a timer.
    """

    def __init__(
        self, node, prefix="/humans", body_timeout=0.5, publish_roi=True, queue_depth=1
    ):
        """Advertise ``<prefix>/bodies/tracked``; per-body topics follow on demand."""
        self._node = node
        self._prefix = prefix.rstrip("/")
        self._body_timeout = body_timeout
        self._publish_roi = publish_roi
        self._queue_depth = queue_depth

        self._lock = threading.Lock()
        self._pending = None  # most recent frame's bodies, or None
        self._publishers = {}  # body_id -> {'skeleton': pub, 'roi': pub}
        self._last_seen = {}  # body_id -> seconds
        self._last_tracked_ids = None

        self._tracked_pub = node.create_publisher(
            IdsList, f"{self._prefix}/bodies/tracked", queue_depth
        )

    def submit(self, bodies):
        """Hand a frame's worth of bodies over to the publishing thread.

        Safe to call from any thread. Only the most recent frame is kept: a
        skeleton stream is state, not a log, so dropping a superseded frame is
        preferable to letting a backlog build up.

        Args:
            bodies: list of ``(body_id, skeleton_msg, roi_msg_or_None)`` tuples.
        """
        with self._lock:
            self._pending = list(bodies)

    def flush(self):
        """Publish the last submitted frame. Must run on the ROS executor thread."""
        with self._lock:
            bodies = self._pending
            self._pending = None

        now = self._node.get_clock().now().nanoseconds * 1e-9

        if bodies is not None:
            for body_id, skeleton_msg, roi_msg in bodies:
                pubs = self._publishers.get(body_id)
                if pubs is None:
                    pubs = self._create_body_publishers(body_id)
                self._last_seen[body_id] = now
                pubs["skeleton"].publish(skeleton_msg)
                if roi_msg is not None and pubs["roi"] is not None:
                    pubs["roi"].publish(roi_msg)

        for body_id in [
            i for i, seen in self._last_seen.items() if now - seen > self._body_timeout
        ]:
            self._destroy_body_publishers(body_id)

        # /humans/bodies/tracked is the discovery mechanism for the whole tree,
        # so it has to be republished whenever the set changes -- including when
        # it becomes empty, which is how consumers learn a body is gone.
        tracked_ids = sorted(self._publishers.keys())
        if tracked_ids != self._last_tracked_ids:
            msg = IdsList()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.ids = tracked_ids
            self._tracked_pub.publish(msg)
            self._last_tracked_ids = tracked_ids

    def _create_body_publishers(self, body_id):
        base = f"{self._prefix}/bodies/{body_id}"
        pubs = {
            "skeleton": self._node.create_publisher(
                Skeleton2D, f"{base}/skeleton2d", self._queue_depth
            ),
            "roi": (
                self._node.create_publisher(
                    NormalizedRegionOfInterest2D, f"{base}/roi", self._queue_depth
                )
                if self._publish_roi
                else None
            ),
        }
        self._publishers[body_id] = pubs
        self._node.get_logger().debug(f"ROS4HRI: body {body_id} appeared")
        return pubs

    def _destroy_body_publishers(self, body_id):
        pubs = self._publishers.pop(body_id, None)
        self._last_seen.pop(body_id, None)
        if pubs is None:
            return
        for pub in pubs.values():
            if pub is not None:
                self._node.destroy_publisher(pub)
        self._node.get_logger().debug(f"ROS4HRI: body {body_id} disappeared")

    def shutdown(self):
        """Tear down every per-body publisher. Call before destroying the node."""
        for body_id in list(self._publishers.keys()):
            self._destroy_body_publishers(body_id)
