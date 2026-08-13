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
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    check_evidence,
    evaluate_evidence,
    iou,
    is_close_range,
)

CONFIG = GateConfig()

# A box that is plausibly a standing person: taller than wide, well clear of
# the minimum height.
PERSON_BOX = (100.0, 50.0, 200.0, 450.0)  # xmin, ymin, xmax, ymax
PERSON_W = PERSON_BOX[2] - PERSON_BOX[0]
PERSON_H = PERSON_BOX[3] - PERSON_BOX[1]

# A 720p frame, and in it the box of someone standing close enough that the
# camera -- mounted at shin height -- frames their legs only: it starts at the
# top edge because the body carries on above the field of view, and runs most
# of the way down the image.
IMAGE_HEIGHT = 720.0
NEAR_BOX = (400.0, 0.0, 700.0, 690.0)


def keypoints(confidences):
    """Build 17 (conf, x, y) keypoints from a confidence per joint."""
    return [(c, 150.0, 100.0 + 10.0 * i) for i, c in enumerate(confidences)]


def person_keypoints(conf=0.9, count=12):
    """Build a body: `count` confident joints, including a full torso."""
    confidences = [0.05] * NUM_KEYPOINTS
    torso = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    for index in torso:
        confidences[index] = conf
    remaining = [i for i in range(NUM_KEYPOINTS) if i not in torso]
    for index in remaining[: max(0, count - len(torso))]:
        confidences[index] = conf
    return keypoints(confidences)


def prop_keypoints():
    """Build a bean bag: nothing is placed anywhere with any confidence."""
    return keypoints([0.12] * NUM_KEYPOINTS)


def legs_keypoints(conf=0.9, joints=(LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)):
    """Build what is left of a person standing too close for their torso."""
    confidences = [0.05] * NUM_KEYPOINTS
    for index in joints:
        confidences[index] = conf
    return keypoints(confidences)


def evidence_for(kpts, box_conf, box=PERSON_BOX, image_height=0.0):
    width = box[2] - box[0]
    height = box[3] - box[1]
    return evaluate_evidence(
        kpts, box_conf, width, height, CONFIG, box_top=box[1], image_height=image_height
    )


def near_evidence(kpts, box_conf=0.7, box=NEAR_BOX):
    """Evidence for a detection the close-range branch will pick up."""
    return evidence_for(kpts, box_conf, box=box, image_height=IMAGE_HEIGHT)


class TestEvidence:
    def test_counts_only_keypoints_above_the_visibility_threshold(self):
        ev = evidence_for(person_keypoints(conf=0.9, count=12), 0.8)
        assert ev.valid_keypoints == 12
        assert ev.torso_keypoints == 4
        assert ev.best_keypoint_conf == pytest.approx(0.9)

    def test_aspect_ratio_of_a_degenerate_box_is_not_a_division_error(self):
        ev = evaluate_evidence(person_keypoints(), 0.9, 100.0, 0.0, CONFIG)
        assert ev.aspect_ratio == float("inf")


class TestAcquireGate:
    def test_confident_person_passes(self):
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.9), CONFIG, strict=True
        )
        assert passed, reason

    def test_prop_with_a_high_box_score_is_rejected_on_keypoints(self):
        # This is the bean bag case: the detector is *sure* it is a person, but
        # cannot place a single joint. Box confidence alone would publish it.
        passed, reason = check_evidence(
            evidence_for(prop_keypoints(), 0.95), CONFIG, strict=True
        )
        assert not passed
        assert "keypoints" in reason

    def test_torso_is_required_even_when_enough_joints_are_found(self):
        # An occluder can pick up scattered limb-like keypoints; what it does
        # not produce is a consistent shoulder/hip rectangle.
        confidences = [0.9] * NUM_KEYPOINTS
        for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            confidences[index] = 0.05
        passed, reason = check_evidence(
            evidence_for(keypoints(confidences), 0.9), CONFIG, strict=True
        )
        assert not passed
        assert "torso" in reason

    def test_squat_box_is_rejected(self):
        wide_box = (100.0, 50.0, 600.0, 250.0)  # 500x200, wider than tall
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.9, box=wide_box), CONFIG, strict=True
        )
        assert not passed
        assert "box-shape" in reason

    def test_low_box_confidence_is_rejected(self):
        passed, reason = check_evidence(
            evidence_for(person_keypoints(), 0.4), CONFIG, strict=True
        )
        assert not passed
        assert "box-conf" in reason


class TestRetainGate:
    def test_partly_occluded_person_fails_acquire_but_passes_retain(self):
        # The reason hysteresis exists: someone who turns away or steps behind
        # an obstacle scores below the acquire gate, and should not be dropped.
        ev = evidence_for(person_keypoints(conf=0.55, count=4), 0.45)
        assert not check_evidence(ev, CONFIG, strict=True)[0]
        assert check_evidence(ev, CONFIG, strict=False)[0]

    def test_prop_fails_retain_as_well(self):
        ev = evidence_for(prop_keypoints(), 0.95)
        assert not check_evidence(ev, CONFIG, strict=False)[0]


class TestCloseRange:
    """The person-standing-next-to-the-robot case.

    The camera is on the head at roughly 0.2 m, so from about three metres in
    the torso is above the top of the frame and only the legs are left. These
    tests pin down that such a person is still detected, and that relaxing the
    torso requirement to get them does not also admit a prop.
    """

    def test_geometry_decides_what_counts_as_close(self):
        assert is_close_range(near_evidence(legs_keypoints()), CONFIG)
        # Same detection, but the box sits in the middle of the frame: the body
        # ends inside the field of view, so it is not a cropped near body.
        mid_box = (400.0, 120.0, 700.0, 600.0)
        assert not is_close_range(
            evidence_for(legs_keypoints(), 0.7, mid_box, IMAGE_HEIGHT), CONFIG
        )
        # Top-clipped but small: something in the distance, not up close.
        short_box = (400.0, 0.0, 700.0, 300.0)
        assert not is_close_range(
            evidence_for(legs_keypoints(), 0.7, short_box, IMAGE_HEIGHT), CONFIG
        )

    def test_frame_height_is_needed_to_judge_proximity(self):
        # Without it there is no way to tell a cropped body from a whole one,
        # so the detection is judged on the ordinary gate.
        assert not is_close_range(evidence_for(legs_keypoints(), 0.7), CONFIG)

    def test_legs_only_person_is_rejected_by_the_ordinary_gate(self):
        # This is the bug the close-range branch exists for: nothing is wrong
        # with the detection, the torso is simply not in the picture.
        passed, reason = check_evidence(
            evidence_for(legs_keypoints(), 0.7), CONFIG, strict=True
        )
        assert not passed
        assert "torso" in reason

    def test_legs_only_person_is_acquired_when_the_box_says_close(self):
        passed, reason = check_evidence(
            near_evidence(legs_keypoints()), CONFIG, strict=True
        )
        assert passed, reason
        assert reason == "near-ok"

    def test_one_leg_is_enough_to_retain_but_not_to_acquire(self):
        # Half a stride, or one leg behind the other: two joints on one leg.
        one_leg = near_evidence(legs_keypoints(joints=(LEFT_KNEE,)), box_conf=0.4)
        assert not check_evidence(one_leg, CONFIG, strict=True)[0]
        assert check_evidence(one_leg, CONFIG, strict=False)[0]

    def test_frame_filling_prop_is_still_rejected(self):
        # A wall panel pushed up against the camera has exactly the geometry
        # the close-range branch looks for -- and no leg to show for it.
        passed, reason = check_evidence(
            near_evidence(prop_keypoints(), box_conf=0.95), CONFIG, strict=True
        )
        assert not passed
        assert reason.startswith("near-")
        assert "keypoints" in reason

    def test_hips_alone_satisfy_the_lower_body_requirement(self):
        # At the far edge of the close-range band the hips are the lowest part
        # of the torso still in frame, and they count as lower body.
        hips = near_evidence(legs_keypoints(joints=(LEFT_HIP, RIGHT_HIP)))
        assert check_evidence(hips, CONFIG, strict=True)[0]

    def test_wide_leg_box_is_allowed_only_at_close_range(self):
        # Legs a metre from the lens are wide relative to the slice of them
        # that fits in the frame; the standing-person shape check would reject
        # the same box at a distance.
        wide = (100.0, 0.0, 1500.0, 700.0)  # 1400x700, ratio 2.0
        assert check_evidence(
            evidence_for(legs_keypoints(), 0.7, wide, IMAGE_HEIGHT), CONFIG, strict=True
        )[0]
        assert not check_evidence(
            evidence_for(person_keypoints(), 0.9, wide), CONFIG, strict=True
        )[0]

    def test_branch_can_be_switched_off(self):
        cfg = GateConfig(proximity_enabled=False)
        assert not is_close_range(near_evidence(legs_keypoints()), cfg)
        assert not check_evidence(near_evidence(legs_keypoints()), cfg, strict=True)[0]

    def test_close_person_is_confirmed_and_published(self):
        confirmer = DetectionConfirmer(CONFIG)
        candidate = (NEAR_BOX, near_evidence(legs_keypoints()))
        assert confirmer.update([candidate], 0.0)[0][0] is False  # unconfirmed
        assert confirmer.update([candidate], 0.1)[0] == (True, "near-ok")

    def test_person_walking_towards_the_robot_keeps_its_track(self):
        # Whole body at a distance, then legs only as they arrive: the branch
        # changes underneath the track, which must not restart it.
        confirmer = DetectionConfirmer(CONFIG)
        far = (NEAR_BOX, evidence_for(person_keypoints(), 0.9, NEAR_BOX))
        confirmer.update([far], 0.0)
        assert confirmer.update([far], 0.1)[0][0] is True

        near = (NEAR_BOX, near_evidence(legs_keypoints()))
        assert confirmer.update([near], 0.2)[0][0] is True
        assert confirmer.track_count == 1


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
        weak = (PERSON_BOX, evidence_for(person_keypoints(0.55, 4), 0.45))
        assert confirmer.update([weak], 0.2)[0][0] is True

    def test_confirmation_is_lost_after_a_long_dropout(self):
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.person()], 0.0)
        assert confirmer.update([self.person()], 0.1)[0][0] is True

        # Nothing for longer than max_missed_time, then a weak return: the
        # track has expired, so retain no longer applies and it must be
        # re-acquired on the strict gate.
        confirmer.update([], 0.2)
        weak = (PERSON_BOX, evidence_for(person_keypoints(0.55, 4), 0.45))
        assert confirmer.update([weak], 1.0)[0][0] is False

    def test_track_is_kept_across_a_short_dropout(self):
        confirmer = DetectionConfirmer(CONFIG)
        confirmer.update([self.person()], 0.0)
        confirmer.update([self.person()], 0.1)
        confirmer.update([], 0.2)  # one frame with nothing detected

        weak = (PERSON_BOX, evidence_for(person_keypoints(0.55, 4), 0.45))
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
