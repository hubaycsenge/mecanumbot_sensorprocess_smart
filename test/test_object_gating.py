#!/usr/bin/env python3
"""
Tests for the fetch detector's gate, which has no keypoints to judge on.

`person_gating.py` decides on the skeleton inside the box. This detector emits
no skeleton, so what is left is shape, hysteresis and temporal confirmation --
and the claims below are what each of those three is actually for.
"""

import pytest

from mecanumbot_sensorprocess_smart.object_gating import (
    REASON_SCORE,
    REASON_SHAPE,
    REASON_SIZE,
    REASON_UNCONFIRMED,
    BoxConfirmer,
    ClassGate,
    box_iou,
    evaluate_shape,
)


def ball_gate(**overrides):
    """Return the shipped ball gate: round, small, believed on a lowish score."""
    settings = dict(
        class_id=32,
        label="sports ball",
        topic="cam_ball_boxes",
        conf_acquire=0.4,
        conf_retain=0.25,
        min_size=6.0,
        min_aspect=0.55,
        max_aspect=1.8,
        min_hits=2,
        max_missed_time=0.4,
        iou_threshold=0.2,
    )
    settings.update(overrides)
    return ClassGate(**settings)


def box(left, top, width, height):
    """`(xmin, ymin, xmax, ymax)`, as the confirmer wants it."""
    return (left, top, left + width, top + height)


# --- shape -------------------------------------------------------------------


def test_a_round_box_is_the_right_shape_for_a_ball():
    assert evaluate_shape(20.0, 20.0, ball_gate())[0]


def test_a_long_smear_is_not_a_ball():
    """The cheapest thing separating a ball from a yellow stripe on the floor."""
    ok, reason = evaluate_shape(60.0, 20.0, ball_gate())
    assert not ok
    assert reason == REASON_SHAPE


def test_a_tall_sliver_is_not_a_ball_either():
    ok, reason = evaluate_shape(10.0, 40.0, ball_gate())
    assert not ok
    assert reason == REASON_SHAPE


def test_a_ball_of_three_pixels_is_a_jpeg_artefact():
    ok, reason = evaluate_shape(3.0, 3.0, ball_gate())
    assert not ok
    assert reason == REASON_SIZE


def test_a_degenerate_box_is_refused_on_size_not_by_dividing_by_zero():
    ok, reason = evaluate_shape(0.0, 20.0, ball_gate())
    assert not ok
    assert reason == REASON_SIZE


def test_the_person_gate_tolerates_someone_close_to_the_camera():
    """
    Wide bounds on purpose: a person at half a metre is wider than they are tall.

    Keeping people out on shape here would undo the whole point of the close
    range branch the pose detector has -- and there is no skeleton to fall back
    on.
    """
    person = ball_gate(class_id=0, label="person", min_aspect=0.05, max_aspect=2.5)
    assert evaluate_shape(300.0, 200.0, person)[0]
    assert evaluate_shape(60.0, 400.0, person)[0]


# --- confirmation ------------------------------------------------------------


def test_one_frame_never_reaches_the_fusion_layer():
    confirmer = BoxConfirmer(ball_gate(min_hits=2))
    accepted, reason = confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.0)[0]
    assert not accepted
    assert reason == REASON_UNCONFIRMED


def test_a_box_that_persists_is_published():
    confirmer = BoxConfirmer(ball_gate(min_hits=2))
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.0)
    assert confirmer.update([(box(1, 1, 20, 20), 0.9)], 0.05)[0][0]


def test_a_score_under_the_acquire_gate_is_refused():
    confirmer = BoxConfirmer(ball_gate(conf_acquire=0.4))
    accepted, reason = confirmer.update([(box(0, 0, 20, 20), 0.3)], 0.0)[0]
    assert not accepted
    assert reason == REASON_SCORE


def test_a_confirmed_box_is_kept_on_the_looser_retain_gate():
    """
    The hysteresis: a ball rolling through a shadow should not blink out.

    Without it the same detection oscillates across one threshold and the tree
    downstream sees the ball appear and vanish several times a second.
    """
    confirmer = BoxConfirmer(ball_gate(min_hits=2, conf_acquire=0.4, conf_retain=0.25))
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.0)
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.05)
    assert confirmer.update([(box(0, 0, 20, 20), 0.3)], 0.10)[0][0]


def test_an_unconfirmed_box_still_has_to_clear_the_acquire_gate():
    confirmer = BoxConfirmer(ball_gate(min_hits=3, conf_acquire=0.4, conf_retain=0.25))
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.0)
    accepted, reason = confirmer.update([(box(0, 0, 20, 20), 0.3)], 0.05)[0]
    assert not accepted
    assert reason == REASON_SCORE


def test_a_track_expires_after_the_dropout_it_is_allowed():
    confirmer = BoxConfirmer(ball_gate(min_hits=2, max_missed_time=0.4))
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.0)
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.05)
    # Long enough away that the ball has to be re-acquired from scratch.
    accepted, reason = confirmer.update([(box(0, 0, 20, 20), 0.3)], 5.0)[0]
    assert not accepted
    assert reason == REASON_SCORE


def test_two_balls_in_one_frame_are_two_tracks():
    confirmer = BoxConfirmer(ball_gate(min_hits=1))
    verdicts = confirmer.update(
        [(box(0, 0, 20, 20), 0.9), (box(400, 300, 20, 20), 0.9)], 0.0
    )
    assert [accepted for accepted, _ in verdicts] == [True, True]


def test_one_track_is_not_claimed_by_two_boxes():
    """
    Two overlapping boxes are two things, however much they overlap.

    Letting the second claim the first's track would confirm it instantly on
    evidence that belongs to a different detection.
    """
    confirmer = BoxConfirmer(ball_gate(min_hits=2))
    confirmer.update([(box(0, 0, 20, 20), 0.9)], 0.0)
    verdicts = confirmer.update(
        [(box(0, 0, 20, 20), 0.9), (box(2, 2, 20, 20), 0.9)], 0.05
    )
    assert verdicts[0] == (True, "")
    assert verdicts[1] == (False, REASON_UNCONFIRMED)


# --- association -------------------------------------------------------------


def test_iou_of_a_box_with_itself_is_one():
    assert box_iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert box_iou(box(0, 0, 10, 10), box(50, 50, 10, 10)) == 0.0


def test_iou_of_touching_boxes_is_zero_not_a_division_error():
    assert box_iou(box(0, 0, 10, 10), box(10, 0, 10, 10)) == 0.0
