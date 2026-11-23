#!/usr/bin/env python3
"""
Finding the object-choice task's coloured bricks, on synthetic HSV frames.

Each brick is drawn where the camera model says a brick on the floor at a given
range would appear, so the gates are tested against the geometry they enforce
rather than against hand-picked pixel numbers. No ROS.
"""

import math

import numpy as np
import pytest

from mecanumbot_sensorprocess_smart import ball_locating
from mecanumbot_sensorprocess_smart import brick_detection as bd

CAMERA = ball_locating.CameraModel(width=1280, height=720, hfov=math.radians(51.0))
# Head tilted down 15 deg, the survey tilt: the floor from ~0.5 m out is in view.
GEOMETRY = ball_locating.BallGeometry(
    diameter=bd.BRICK_SIZE, camera_x=0.128, camera_z=0.2245, camera_pitch=math.radians(-15.0)
)
COLOURS = bd.colours_from_table(bd.DEFAULT_COLOURS)
GATE = bd.BrickGate()

# Representative HSV of each brick colour, and of a grey floor.
HSV = {"red": (2, 200, 180), "yellow": (27, 200, 200), "green": (60, 180, 150),
       "blue": (112, 200, 150)}
FLOOR = (15, 30, 120)


def floor_pixel(distance, bearing=0.0, geometry=GEOMETRY):
    """Return the image (u, v) of a brick centre on the floor `distance` m away."""
    drop = geometry.camera_z - (geometry.floor_z + geometry.diameter / 2.0)
    # Ray elevation in the robot frame that meets the brick-centre plane.
    elevation_world = -math.asin(drop / distance)
    elevation_image = elevation_world - geometry.camera_pitch
    u = CAMERA.width / 2.0 - CAMERA.focal_x * math.tan(bearing)
    v = CAMERA.height / 2.0 - CAMERA.focal_y * math.tan(elevation_image)
    return u, v


def blank():
    frame = np.zeros((CAMERA.height, CAMERA.width, 3), np.uint8)
    frame[...] = FLOOR
    return frame


def draw(frame, colour, u, v, width, height):
    top, bottom = int(v - height / 2), int(v + height / 2)
    left, right = int(u - width / 2), int(u + width / 2)
    frame[top:bottom, left:right] = HSV[colour]


def draw_brick(frame, colour, distance, bearing=0.0, widen=1.0):
    u, v = floor_pixel(distance, bearing)
    width = bd.expected_width_px(CAMERA, bd.BRICK_SIZE, distance) * widen
    draw(frame, colour, u, v, width, width * 0.95)
    return u, v


# --- colour ranges ----------------------------------------------------------


def test_red_wraps_through_zero_hue():
    red = bd.ColourRange.from_values("red", bd.DEFAULT_COLOURS["red"])
    assert red.wraps
    hsv = np.array([[[2, 200, 180], [175, 200, 180], [90, 200, 180]]], np.uint8)
    assert list(red.mask(hsv)[0]) == [255, 255, 0]


def test_grey_floor_is_no_colour():
    frame = blank()
    assert all(colour.mask(frame).max() == 0 for colour in COLOURS)


@pytest.mark.parametrize(
    "values",
    [[0, 10, 0, 255, 0], [0, 200, 0, 255, 0, 255], [0, 10, 200, 100, 0, 255]],
)
def test_bad_ranges_are_refused(values):
    with pytest.raises(ValueError):
        bd.ColourRange.from_values("bad", values)


# --- finding and accepting bricks -------------------------------------------


@pytest.mark.parametrize("distance", [0.8, 1.5, 2.5])
def test_a_brick_on_the_floor_is_found_and_ranged(distance):
    frame = blank()
    draw_brick(frame, "blue", distance)
    accepted, _ = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert [c.blob.colour for c in accepted] == ["blue"]
    observation = accepted[0].observation
    assert observation.source == ball_locating.SOURCE_GROUND
    assert observation.range == pytest.approx(distance, rel=0.05)
    assert observation.z == pytest.approx(bd.BRICK_SIZE / 2.0 + GEOMETRY.floor_z, abs=0.02)


def test_one_brick_per_colour_and_all_colours():
    frame = blank()
    draw_brick(frame, "red", 1.8, bearing=math.radians(15))
    draw_brick(frame, "green", 1.8, bearing=math.radians(-15))
    draw_brick(frame, "yellow", 1.2)
    accepted, _ = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    found = {c.blob.colour: c.observation for c in accepted}
    assert set(found) == {"red", "green", "yellow"}
    # Left is positive y in the robot frame.
    assert found["red"].y > 0.2 > -0.2 > found["green"].y


def test_a_second_blob_of_one_colour_is_not_a_second_brick():
    frame = blank()
    draw_brick(frame, "red", 1.0, bearing=math.radians(10))
    draw_brick(frame, "red", 2.0, bearing=math.radians(-10))
    accepted, rejected = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert len(accepted) == 1
    # The larger, nearer one is kept.
    assert accepted[0].observation.range == pytest.approx(1.0, rel=0.05)
    assert any(reason == "one red too many" for _, reason in rejected)


# --- identical bricks -------------------------------------------------------

# Unpainted light wood under room light: orange, pale, bright.
HSV["wood"] = (16, 90, 190)


def test_a_wooden_brick_is_wood_and_no_painted_colour():
    frame = blank()
    draw_brick(frame, "wood", 1.5)
    accepted, _ = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert [c.blob.colour for c in accepted] == ["wood"]


def test_identical_bricks_are_all_found_when_allowed():
    frame = blank()
    bearings = [20, 7, -7, -20]
    for bearing in bearings:
        draw_brick(frame, "wood", 1.6, bearing=math.radians(bearing))
    accepted, rejected = bd.detect_bricks(
        frame, COLOURS, GATE, CAMERA, GEOMETRY, max_per_colour=6
    )
    assert [c.blob.colour for c in accepted] == ["wood"] * 4
    assert not rejected
    # Four distinct places, left to right in the robot frame.
    ys = sorted((c.observation.y for c in accepted), reverse=True)
    assert all(a - b > 0.2 for a, b in zip(ys, ys[1:]))


def test_the_per_colour_cap_keeps_the_largest():
    frame = blank()
    for distance, bearing in ((0.9, 15), (1.5, 0), (2.4, -15)):
        draw_brick(frame, "wood", distance, bearing=math.radians(bearing))
    accepted, rejected = bd.detect_bricks(
        frame, COLOURS, GATE, CAMERA, GEOMETRY, max_per_colour=2
    )
    ranges = sorted(c.observation.range for c in accepted)
    assert ranges == [pytest.approx(0.9, rel=0.05), pytest.approx(1.5, rel=0.05)]
    assert [reason for _, reason in rejected] == ["one wood too many"]


def test_a_big_patch_is_too_big_for_a_brick_where_it_meets_the_floor():
    frame = blank()
    draw_brick(frame, "green", 1.5, widen=4.0)
    accepted, rejected = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert accepted == []
    assert [reason for _, reason in rejected] == [bd.REJECT_SIZE]


def test_a_long_strip_is_the_wrong_shape():
    frame = blank()
    u, v = floor_pixel(1.2)
    draw(frame, "yellow", u, v, 300, 30)
    accepted, rejected = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert accepted == []
    assert rejected[0][1] == bd.REJECT_ASPECT


def test_a_blob_above_the_horizon_is_not_on_the_floor():
    frame = blank()
    # Level camera, blob in the top half: its ray never meets the floor.
    level = ball_locating.BallGeometry(diameter=bd.BRICK_SIZE, camera_pitch=0.0)
    draw(frame, "blue", 640, 200, 40, 40)
    accepted, rejected = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, level)
    assert accepted == []
    assert rejected[0][1] == bd.REJECT_HORIZON


def test_speckle_is_dropped():
    frame = blank()
    frame[400, 640] = HSV["red"]
    frame[410, 300:302] = HSV["red"]
    accepted, rejected = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert accepted == [] and rejected == []


def test_a_downscaled_frame_reports_full_frame_boxes():
    frame = blank()
    u, v = draw_brick(frame, "blue", 1.5)
    small = frame[::2, ::2]
    accepted, _ = bd.detect_bricks(small, COLOURS, GATE, CAMERA, GEOMETRY, scale=2.0)
    assert len(accepted) == 1
    assert accepted[0].blob.u == pytest.approx(u, abs=3)
    assert accepted[0].blob.v == pytest.approx(v, abs=3)
    assert accepted[0].observation.range == pytest.approx(1.5, rel=0.06)


def test_mirroring_reflects_the_side():
    frame = blank()
    draw_brick(frame, "red", 1.5, bearing=math.radians(15))
    straight, _ = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    mirrored, _ = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY, mirror=True)
    assert straight[0].observation.y == pytest.approx(-mirrored[0].observation.y, abs=1e-6)


def test_score_is_the_fill():
    frame = blank()
    draw_brick(frame, "blue", 1.0)
    accepted, _ = bd.detect_bricks(frame, COLOURS, GATE, CAMERA, GEOMETRY)
    assert 0.9 <= accepted[0].score <= 1.0


def test_median_hsv_reads_a_brick_colour_for_calibration():
    frame = blank()
    draw(frame, "green", 640, 500, 60, 60)
    assert bd.mean_hsv(frame, (640, 500, 40, 40)) == HSV["green"]
    assert bd.mean_hsv(frame, (5000, 500, 40, 40)) is None
