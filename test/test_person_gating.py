#!/usr/bin/env python3
"""Unit tests for the camera person-detection gate.

The detector node itself cannot run off the Jetson (it needs `pyds`, GStreamer
and a CSI camera), so the decision logic was deliberately kept in a pure-Python
module. These tests are therefore the only feedback loop for it on a
development machine, and they encode the two failure modes the gate exists to
fix: props being published as people, and real people dropping out.
"""

import pytest

from mecanumbot_sensorprocess_smart.person_gating import (
    DetectionConfirmer,
    GateConfig,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_KNEE,
    LEFT_SHOULDER,
    NOSE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_SHOULDER,
    check_evidence,
    evaluate_evidence,
    head_at_bottom,
    iou,
    is_close_range,
)

CONFIG = GateConfig()

# A box that is plausibly a standing person: taller than wide, well clear of
# the minimum height.
PERSON_BOX = (100.0, 50.0, 200.0, 450.0)  # xmin, ymin, xmax, ymax

# A 720p frame, and in it the box of somebody standing close enough that the
# body carries on above the top of the frame.
IMAGE_HEIGHT = 720.0
NEAR_BOX = (400.0, 0.0, 700.0, 690.0)


def keypoints(confidences):
    """Build 17 (conf, x, y) keypoints from a confidence per joint."""
    return [(c, 150.0, 100.0 + 10.0 * i) for i, c in enumerate(confidences)]


def joints_keypoints(found):
    """Build keypoints with the given ``{joint: confidence}`` found, the rest not."""
    confidences = [0.05] * NUM_KEYPOINTS
    for index, conf in found.items():
        confidences[index] = conf
    return keypoints(confidences)


def person_keypoints(conf=0.9, count=12):
    """Build a body: `count` confident joints."""
    return keypoints([conf] * count + [0.05] * (NUM_KEYPOINTS - count))


def prop_keypoints():
    """Build a bean bag: nothing is placed anywhere with any confidence."""
    return keypoints([0.12] * NUM_KEYPOINTS)


def evidence_for(kpts, box_conf, box=PERSON_BOX, image_height=0.0):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return evaluate_evidence(
        kpts, box_conf, width, height, CONFIG, box_top=box[1], image_height=image_height
    )


def frame_evidence(kpts, box_conf, box):
    """Evidence for a box in a 720p frame, so the close-range label can apply."""
    return evidence_for(kpts, box_conf, box, IMAGE_HEIGHT)


class TestEvidence:
    def test_counts_only_keypoints_above_the_visibility_threshold(self):
        ev = evidence_for(person_keypoints(conf=0.9, count=12), 0.8)
        assert ev.valid_keypoints == 12
        assert ev.best_keypoint_conf == pytest.approx(0.9)

    def test_a_nan_confidence_is_not_a_joint(self):
        kpts = joints_keypoints({NOSE: 0.9})
        kpts[LEFT_EYE] = (float("nan"), 0.0, 0.0)
        ev = evidence_for(kpts, 0.8)
        assert ev.valid_keypoints == 1
        assert ev.best_keypoint_conf == pytest.approx(0.9)

    def test_aspect_ratio_of_a_degenerate_box_is_not_a_division_error(self):
        ev = evaluate_evidence(person_keypoints(), 0.9, 100.0, 0.0, CONFIG)
        assert ev.aspect_ratio == float("inf")


class TestAcquireGate:
    def test_confident_person_passes(self):
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.9), CONFIG, strict=True
        )
        assert (passed, reason) == (True, "ok")

    def test_prop_with_a_high_box_score_is_rejected_on_keypoints(self):
        # This is the bean bag case: the detector is *sure* it is a person, but
        # cannot place a single joint. Box confidence alone would publish it.
        passed, reason = check_evidence(
            evidence_for(prop_keypoints(), 0.95), CONFIG, strict=True
        )
        assert not passed
        assert "keypoints" in reason

    def test_joints_placed_unsurely_are_not_enough(self):
        # Joints over the visibility threshold, none of them convincing.
        passed, reason = check_evidence(
            evidence_for(person_keypoints(conf=0.4, count=8), 0.9), CONFIG, strict=True
        )
        assert not passed
        assert "weak-pose" in reason

    def test_one_convincing_joint_is_not_enough_on_its_own(self):
        one = joints_keypoints({LEFT_KNEE: 0.99})
        passed, reason = check_evidence(evidence_for(one, 0.9), CONFIG, strict=True)
        assert not passed
        assert reason == "keypoints 1<2"

    def test_which_joints_they_are_does_not_matter(self):
        # A torso, a pair of legs, a face, an arm: the same verdict for each.
        for found in (
            {LEFT_SHOULDER: 0.9, RIGHT_SHOULDER: 0.9},
            {LEFT_KNEE: 0.9, RIGHT_ANKLE: 0.9},
            {NOSE: 0.9, LEFT_EYE: 0.9},
            {LEFT_SHOULDER: 0.9, LEFT_ELBOW: 0.9},
        ):
            ev = evidence_for(joints_keypoints(found), 0.6)
            assert check_evidence(ev, CONFIG, strict=True) == (True, "ok"), found

    def test_a_box_far_wider_than_tall_is_rejected(self):
        wide_box = (100.0, 300.0, 700.0, 400.0)  # 600x100
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.9, box=wide_box), CONFIG, strict=True
        )
        assert not passed
        assert "box-shape" in reason

    def test_a_sliver_is_rejected(self):
        sliver = (100.0, 50.0, 120.0, 80.0)
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.9, box=sliver), CONFIG, strict=True
        )
        assert not passed
        assert "box-height" in reason

    def test_low_box_confidence_is_rejected(self):
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.35), CONFIG, strict=True
        )
        assert not passed
        assert "box-conf" in reason


class TestRetainGate:
    def test_partly_occluded_person_fails_acquire_but_passes_retain(self):
        # The reason hysteresis exists: someone who turns away or steps behind
        # an obstacle scores below the acquire gate, and should not be dropped.
        ev = evidence_for(person_keypoints(conf=0.45, count=3), 0.35)
        assert not check_evidence(ev, CONFIG, strict=True)[0]
        assert check_evidence(ev, CONFIG, strict=False)[0]

    def test_prop_fails_retain_as_well(self):
        ev = evidence_for(prop_keypoints(), 0.95)
        assert not check_evidence(ev, CONFIG, strict=False)[0]


class TestThePilot:
    """
    The 2026-10-16 leading pilot: people lost with the head tilted up.

    Every case is read off a debug frame the old gate rejected
    (``bag_debug/rosbag2_2026_10_16-*``), boxes in 1280x720 pixels. The old
    gate asked for a torso far away and legs close up; the camera, tilted up,
    showed chest and arms close up and head and shoulders further off.
    """

    def test_close_person_framed_by_chest_and_arms(self):
        # Box 0.84 filling the frame's height, a shoulder at 0.82 and an elbow
        # at 0.98: rejected as "near-legs 0<2".
        kpts = joints_keypoints({LEFT_SHOULDER: 0.82, LEFT_ELBOW: 0.98})
        ev = frame_evidence(kpts, 0.84, (0.0, 0.0, 220.0, 720.0))
        assert check_evidence(ev, CONFIG, strict=True) == (True, "ok")
        assert is_close_range(ev, CONFIG)

    def test_head_and_shoulders_cut_off_by_the_bottom_edge(self):
        # Box 0.47, the face at 0.89-0.95, a shoulder at 0.35, the box 3.3
        # times wider than tall: rejected as "box-conf 0.47<=0.60".
        kpts = joints_keypoints(
            {NOSE: 0.94, LEFT_EYE: 0.89, RIGHT_EYE: 0.95, LEFT_SHOULDER: 0.35}
        )
        ev = frame_evidence(kpts, 0.47, (760.0, 586.0, 1200.0, 720.0))
        assert check_evidence(ev, CONFIG, strict=True) == (True, "ok")
        assert not is_close_range(ev, CONFIG)

    def test_a_squarer_head_and_shoulders_box(self):
        # Box 0.54, face at 0.77-1.00 and a shoulder at 1.00.
        kpts = joints_keypoints(
            {NOSE: 0.99, LEFT_EYE: 1.0, RIGHT_EYE: 1.0, LEFT_EAR: 0.77,
             RIGHT_SHOULDER: 1.0}
        )
        ev = frame_evidence(kpts, 0.54, (552.0, 440.0, 944.0, 720.0))
        assert check_evidence(ev, CONFIG, strict=True) == (True, "ok")

    def test_a_weak_box_is_kept_but_not_acquired(self):
        # Box 0.32 with the face at 0.39-0.86: under the acquire threshold, but
        # a person already being followed is not dropped on it.
        kpts = joints_keypoints(
            {NOSE: 0.55, LEFT_EYE: 0.85, RIGHT_EYE: 0.86, RIGHT_EAR: 0.39}
        )
        ev = frame_evidence(kpts, 0.32, (974.0, 600.0, 1100.0, 720.0))
        assert not check_evidence(ev, CONFIG, strict=True)[0]
        assert check_evidence(ev, CONFIG, strict=False)[0]

    def test_legs_in_a_long_skirt(self):
        # 2026-09: box 0.84, a knee and both ankles at ~0.99, rejected as
        # "keypoints 3<6" by the whole-body count.
        kpts = joints_keypoints({LEFT_KNEE: 0.99, LEFT_ANKLE: 0.99, RIGHT_ANKLE: 0.99})
        ev = frame_evidence(kpts, 0.84, (140.0, 48.0, 770.0, 693.0))
        assert check_evidence(ev, CONFIG, strict=True) == (True, "ok")

    def test_bare_legs_at_a_low_box_score(self):
        # Box 0.44, a knee at 0.55 and an ankle at 0.33.
        kpts = joints_keypoints({LEFT_KNEE: 0.55, LEFT_ANKLE: 0.33})
        ev = frame_evidence(kpts, 0.44, NEAR_BOX)
        assert check_evidence(ev, CONFIG, strict=True)[0]

    def test_props_with_the_same_boxes_are_still_rejected(self):
        # A wall panel against the camera, a tablecloth over a chair, a bean
        # bag at the bottom of the frame: the geometry of each case above, and
        # no joint to show for it.
        for box in (
            (0.0, 0.0, 220.0, 720.0),
            (760.0, 586.0, 1200.0, 720.0),
            (140.0, 48.0, 770.0, 693.0),
            NEAR_BOX,
        ):
            ev = frame_evidence(prop_keypoints(), 0.95, box)
            assert not check_evidence(ev, CONFIG, strict=True)[0], box
            assert not check_evidence(ev, CONFIG, strict=False)[0], box


class TestCloseRangeLabel:
    """The geometry that still labels a detection ``close_range``."""

    def test_a_box_from_the_top_edge_filling_the_frame_is_close(self):
        assert is_close_range(frame_evidence(person_keypoints(), 0.7, NEAR_BOX), CONFIG)

    def test_a_box_in_the_middle_of_the_frame_is_not(self):
        mid_box = (400.0, 120.0, 700.0, 600.0)
        assert not is_close_range(
            frame_evidence(person_keypoints(), 0.7, mid_box), CONFIG
        )

    def test_a_short_box_at_the_top_edge_is_not(self):
        short_box = (400.0, 0.0, 700.0, 300.0)
        assert not is_close_range(
            frame_evidence(person_keypoints(), 0.7, short_box), CONFIG
        )

    def test_a_box_that_misses_the_top_edge_by_a_skirt_is_close(self):
        assert is_close_range(
            frame_evidence(person_keypoints(), 0.8, (140.0, 48.0, 770.0, 693.0)),
            CONFIG,
        )

    def test_a_box_filling_the_frame_is_close_wherever_its_top_sits(self):
        deep = (140.0, 150.0, 770.0, 715.0)  # 0.785 of the frame, top 150 px
        assert not is_close_range(frame_evidence(person_keypoints(), 0.8, deep), CONFIG)
        tall = (140.0, 80.0, 770.0, 715.0)  # 0.88 of the frame, top 80 px
        assert is_close_range(frame_evidence(person_keypoints(), 0.8, tall), CONFIG)

    def test_frame_height_is_needed(self):
        assert not is_close_range(evidence_for(person_keypoints(), 0.7, NEAR_BOX), CONFIG)

    def test_the_label_can_be_switched_off(self):
        cfg = GateConfig(proximity_enabled=False)
        assert not is_close_range(frame_evidence(person_keypoints(), 0.7, NEAR_BOX), cfg)

    def test_the_label_does_not_change_the_verdict(self):
        kpts = joints_keypoints({LEFT_KNEE: 0.9, RIGHT_ANKLE: 0.9})
        near = frame_evidence(kpts, 0.6, NEAR_BOX)
        far = evidence_for(kpts, 0.6)
        assert check_evidence(near, CONFIG, True) == check_evidence(far, CONFIG, True)


class TestIou:
    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_identical_boxes(self):
        assert iou(PERSON_BOX, PERSON_BOX) == pytest.approx(1.0)

    def test_partial_overlap(self):
        # 10x10 and 10x10 sharing a 5x10 strip: 50 / (100 + 100 - 50).
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)


class TestConfirmer:
    def person(self, box=PERSON_BOX, box_conf=0.9, conf=0.9, count=12):
        return (box, evidence_for(person_keypoints(conf, count), box_conf, box))

    def prop(self, box=PERSON_BOX):
        return (box, evidence_for(prop_keypoints(), 0.95, box))

    def test_person_is_published_only_after_min_hits(self):
        confirmer = DetectionConfirmer(CONFIG)
        first = confirmer.update([self.person()], 0.0)
        assert first[0][0] is False
        assert "unconfirmed" in first[0][1]

        second = confirmer.update([self.person()], 0.1)
        assert second[0][0] is True

    def test_single_frame_prop_never_confirms(self):
        confirmer = DetectionConfirmer(CONFIG)
        for i in range(10):
            result = confirmer.update([self.prop()], 0.1 * i)
            assert result[0][0] is False
        assert confirmer.track_count == 0

    def test_confirmed_person_survives_a_weak_frame(self):
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.person()], 0.0)
        assert confirmer.update([self.person()], 0.1)[0][0] is True

        # Same person, now half occluded: below acquire, above retain.
        weak = (PERSON_BOX, evidence_for(person_keypoints(0.45, 3), 0.35))
        assert confirmer.update([weak], 0.2)[0][0] is True

    def test_confirmation_is_lost_after_a_long_dropout(self):
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.person()], 0.0)
        assert confirmer.update([self.person()], 0.1)[0][0] is True

        # Nothing for longer than max_missed_time, then a weak return: the
        # track has expired, so retain no longer applies and it must be
        # re-acquired on the strict gate.
        confirmer.update([], 0.2)
        weak = (PERSON_BOX, evidence_for(person_keypoints(0.45, 3), 0.35))
        assert confirmer.update([weak], 1.0)[0][0] is False

    def test_track_is_kept_across_a_short_dropout(self):
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.person()], 0.0)
        confirmer.update([self.person()], 0.1)
        confirmer.update([], 0.2)  # one frame with nothing detected

        weak = (PERSON_BOX, evidence_for(person_keypoints(0.45, 3), 0.35))
        assert confirmer.update([weak], 0.3)[0][0] is True

    def test_moving_person_stays_associated(self):
        confirmer = DetectionConfirmer(CONFIG)
        box = PERSON_BOX
        confirmer.update([self.person(box)], 0.0)
        # Shift by a quarter of the width; IoU stays well above threshold.
        shifted = (box[0] + 25, box[1], box[2] + 25, box[3])
        assert confirmer.update([self.person(shifted)], 0.1)[0][0] is True
        assert confirmer.track_count == 1

    def test_two_people_are_tracked_independently(self):
        confirmer = DetectionConfirmer(CONFIG)
        other = (500.0, 50.0, 600.0, 450.0)
        confirmer.update([self.person(), self.person(other)], 0.0)
        result = confirmer.update([self.person(), self.person(other)], 0.1)
        assert [r[0] for r in result] == [True, True]
        assert confirmer.track_count == 2

    def test_prop_next_to_a_confirmed_person_does_not_inherit_the_track(self):
        # A prop that overlaps a confirmed person would pass the retain gate
        # if retention were purely positional, so retention is still evidence
        # based -- the prop has to look like a body too.
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.person()], 0.0)
        confirmer.update([self.person()], 0.1)
        assert confirmer.update([self.prop()], 0.2)[0][0] is False

    def near(self, box_conf, conf=0.9, count=8):
        return (NEAR_BOX, frame_evidence(person_keypoints(conf, count), box_conf, NEAR_BOX))

    def test_a_weak_frame_filling_box_is_not_held_on_retain(self):
        # The 2026-10-21 run: legs close up confirmed, then a shelf of plants
        # blurred by a turn filled the frame at 0.34 and, overlapping the legs'
        # box like any two frame-filling boxes do, was kept as the same person.
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.near(0.6)], 0.0)
        assert confirmer.update([self.near(0.6)], 0.1)[0][0] is True
        weak = self.near(0.34, conf=0.45, count=3)
        assert is_close_range(weak[1], CONFIG)
        assert confirmer.update([weak], 0.2)[0][0] is False

    def test_a_frame_filling_box_on_the_acquire_gate_is_still_kept(self):
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.near(0.6)], 0.0)
        confirmer.update([self.near(0.6)], 0.1)
        assert confirmer.update([self.near(0.45)], 0.2)[0][0] is True

    def test_close_range_retain_brings_the_hysteresis_back(self):
        cfg = GateConfig(close_range_retain=True)
        confirmer = DetectionConfirmer(cfg)
        near = (NEAR_BOX, evidence_for(person_keypoints(0.9, 8), 0.6, NEAR_BOX, IMAGE_HEIGHT))
        confirmer.update([near], 0.0)
        confirmer.update([near], 0.1)
        weak = (
            NEAR_BOX,
            evidence_for(person_keypoints(0.45, 3), 0.34, NEAR_BOX, IMAGE_HEIGHT),
        )
        assert confirmer.update([weak], 0.2)[0][0] is True


# --- the look-down cue -----------------------------------------------------

H = 720.0


def _face(y, conf=0.95, x=640.0, n=5):
    """Return 17 keypoints with `n` face joints at height `y`, the rest missing."""
    missing = (0.0, float("nan"), float("nan"))
    return [(conf, x + 10.0 * i, y) if i < n else missing for i in range(17)]


def test_face_at_the_bottom_edge_is_a_cue():
    """The bean bag of 2026-09-30: a face at the bottom, nothing below it."""
    assert head_at_bottom(_face(690.0), H) == pytest.approx(660.0)


def test_face_in_the_middle_is_not():
    assert head_at_bottom(_face(300.0), H) is None


def test_one_face_joint_is_not_enough():
    assert head_at_bottom(_face(690.0, n=1), H) is None


def test_unsure_face_joints_do_not_count():
    assert head_at_bottom(_face(690.0, conf=0.3), H) is None


def test_missing_joints_as_zeros_do_not_count():
    keypoints = [(0.9, 0.0, 0.0)] * 17
    assert head_at_bottom(keypoints, H) is None


def test_no_frame_height_no_cue():
    assert head_at_bottom(_face(690.0), 0.0) is None
