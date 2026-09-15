#!/usr/bin/env python3
"""
Tests for the drawing the Ultralytics detectors put on their debug images.

What matters is that a missing joint is never drawn as if it were found -- a
bone to the image corner reads as a gesture that did not happen -- and that
the image still encodes when a box runs off the frame.
"""

import math

import cv2
import numpy as np

from mecanumbot_sensorprocess_smart.debug_overlay import (
    COLOUR_BALL,
    SKELETON_CONNECTIONS,
    draw_box,
    draw_skeleton,
    encode_jpeg,
    visible_joints,
)


def blank(width=320, height=240):
    """Return a black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_skeleton_indexes_the_17_coco_joints():
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON_CONNECTIONS)


def test_ultralytics_missing_joint_is_not_visible_whatever_its_confidence():
    assert visible_joints([(0.0, 0.0, 0.9)], 0.3) == [False]


def test_deepstream_nan_joint_is_not_visible():
    assert visible_joints([(math.nan, 10.0, 0.9), (10.0, 10.0, math.nan)], 0.3) == [
        False,
        False,
    ]


def test_confidence_threshold_is_strict():
    assert visible_joints([(5.0, 5.0, 0.3), (5.0, 5.0, 0.31)], 0.3) == [False, True]


def test_joint_without_a_confidence_counts_as_found():
    assert visible_joints([(5.0, 5.0, None)], 0.3) == [True]


def test_skeleton_draws_nothing_for_an_unfound_person():
    image = blank()
    draw_skeleton(image, [(0.0, 0.0, 0.9)] * 17, 0.3)
    assert not image.any()


def test_bone_is_drawn_only_between_two_found_joints():
    keypoints = [(0.0, 0.0, 0.0)] * 17
    keypoints[5] = (100.0, 100.0, None)  # left shoulder
    keypoints[6] = (200.0, 100.0, None)  # right shoulder
    image = blank()
    draw_skeleton(image, keypoints, 0.3)
    assert image[100, 150].any()
    assert not image[0:20, 0:20].any()


def test_box_is_drawn_in_its_colour():
    image = blank()
    draw_box(image, (50.0, 60.0, 150.0, 160.0), COLOUR_BALL, "sports ball 0.80")
    assert tuple(image[160, 100]) == COLOUR_BALL


def test_box_off_the_frame_still_encodes():
    image = blank()
    draw_box(image, (-40.0, -40.0, 400.0, 300.0), COLOUR_BALL, "sports ball 0.80")
    data = encode_jpeg(image)
    assert data is not None
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
