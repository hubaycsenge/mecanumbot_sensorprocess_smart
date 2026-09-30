#!/usr/bin/env python3
"""
Unit tests for the camera-wedge lookup into the LiDAR scan.

The case that matters is the robot's own: an LD08 scan that starts at 0 and runs
a full turn, looked up with signed camera bearings. Until 2026-09-30 a person
to the right of the camera axis found no beams at all. No ROS.
"""

import math

import numpy as np

from mecanumbot_sensorprocess_smart.scan_wedge import wedge_ranges

N = 360
INC = 2.0 * math.pi / N


def ld08_scan():
    """Return a 0..2pi scan whose range at each beam is its angle in degrees."""
    return np.arange(N, dtype=float)


def signed_scan():
    """Return a -pi..pi scan with the same labelling, degrees 0..359."""
    return np.mod(np.arange(N, dtype=float) - 180.0, 360.0)


def test_right_of_axis_finds_beams_on_a_0_to_2pi_scan():
    got = wedge_ranges(ld08_scan(), 0.0, INC, math.radians(-25.5), math.radians(-18.5))
    assert sorted(got) == list(range(335, 342))


def test_wedge_straddling_the_axis_keeps_both_halves():
    got = wedge_ranges(ld08_scan(), 0.0, INC, math.radians(-5.5), math.radians(5.5))
    assert sorted(got) == [0, 1, 2, 3, 4, 5, 355, 356, 357, 358, 359]


def test_left_of_axis_unchanged():
    got = wedge_ranges(ld08_scan(), 0.0, INC, math.radians(9.5), math.radians(12.5))
    assert sorted(got) == [10, 11, 12]


def test_same_wedge_on_a_signed_scan():
    got = wedge_ranges(signed_scan(), -math.pi, INC, math.radians(-25.5), math.radians(-18.5))
    assert sorted(got) == list(range(335, 342))


def test_bearing_order_does_not_matter():
    a = wedge_ranges(ld08_scan(), 0.0, INC, math.radians(-18.5), math.radians(-25.5))
    b = wedge_ranges(ld08_scan(), 0.0, INC, math.radians(-25.5), math.radians(-18.5))
    assert sorted(a) == sorted(b)


def test_empty_scan():
    assert wedge_ranges([], 0.0, INC, -0.1, 0.1).size == 0
