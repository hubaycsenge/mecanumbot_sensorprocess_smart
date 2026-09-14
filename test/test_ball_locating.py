#!/usr/bin/env python3
"""
Tests for the geometry that decides where the robot drives to get the ball.

The claims worth pinning down are the ones that would otherwise fail silently:
a bearing that runs the wrong way puts the robot on the wrong side of the ball
and nothing complains, and a range estimator that quietly returns something
plausible from wrong inputs is exactly the failure `_check_range_agreement`
exists to catch.

No ROS: `ball_locating` is plain Python so this runs anywhere.
"""

import math

import pytest

from mecanumbot_sensorprocess_smart.ball_locating import (
    SOURCE_GROUND,
    SOURCE_SIZE,
    TENNIS_BALL_DIAMETER,
    BallGeometry,
    BallMeasurement,
    BallTracker,
    BallTrackerConfig,
    CameraModel,
    NeckHistory,
    NeckMount,
    direction,
    floor_height,
    floor_pitch,
    graspable,
    locate,
    range_from_ground,
    range_from_size,
)

HD = CameraModel(width=1280.0, height=720.0, hfov=math.radians(60.0))


def geometry(**overrides):
    """Return a camera 0.21 m up, looking level, unless a test says otherwise."""
    return BallGeometry(**overrides)


# --- the lens ---------------------------------------------------------------


def test_focal_length_matches_the_field_of_view():
    # Half the frame subtends half the FOV, by construction.
    assert HD.bearing(0.0) == pytest.approx(math.radians(30.0), abs=1e-9)
    assert HD.bearing(1280.0) == pytest.approx(math.radians(-30.0), abs=1e-9)


def test_bearing_is_positive_to_the_left():
    """The left of the image is the robot's left, which is positive yaw in ROS."""
    assert HD.bearing(100.0) > 0.0
    assert HD.bearing(1180.0) < 0.0
    assert HD.bearing(640.0) == pytest.approx(0.0, abs=1e-12)


def test_elevation_is_positive_upwards_and_rows_count_downwards():
    assert HD.elevation(0.0) > 0.0
    assert HD.elevation(719.0) < 0.0


def test_vertical_fov_is_derived_from_square_pixels():
    # 720/1280 of the tangent, not of the angle.
    expected = 2.0 * math.atan((720.0 / 2.0) / HD.focal_x)
    assert HD.vertical_fov == pytest.approx(expected)


def test_given_vertical_fov_overrides_the_derivation():
    camera = CameraModel(1280.0, 720.0, math.radians(60.0), vfov=math.radians(50.0))
    assert camera.vertical_fov == pytest.approx(math.radians(50.0))
    assert camera.focal_y != pytest.approx(camera.focal_x)


# --- range from apparent size ------------------------------------------------


def test_range_from_size_is_inverse_in_the_box_width():
    near = range_from_size(HD, 50.0, 50.0, TENNIS_BALL_DIAMETER)
    far = range_from_size(HD, 25.0, 25.0, TENNIS_BALL_DIAMETER)
    assert far == pytest.approx(2.0 * near)


def test_range_from_size_takes_the_larger_apparent_diameter():
    """
    A clipped box is always too small, so the less truncated view wins.

    A ball half behind a chair leg gives a narrow box and a tall one; believing
    the narrow one would place the ball twice as far away as it is.
    """
    clipped = range_from_size(HD, 12.0, 24.0, TENNIS_BALL_DIAMETER)
    whole = range_from_size(HD, 24.0, 24.0, TENNIS_BALL_DIAMETER)
    assert clipped == pytest.approx(whole)


def test_range_from_size_is_none_for_a_degenerate_box():
    assert range_from_size(HD, 0.0, 0.0, TENNIS_BALL_DIAMETER) is None


# --- range from the ground plane ---------------------------------------------


def test_ground_range_grows_as_the_ray_flattens():
    steep, _, _ = direction(HD, 640.0, 700.0, 0.0)
    shallow, _, _ = direction(HD, 640.0, 380.0, 0.0)
    assert range_from_ground(steep, geometry()) < range_from_ground(
        shallow, geometry()
    )


def test_ground_range_declines_to_answer_above_the_horizon():
    """A ball detected level with or above the camera is not on the floor."""
    level, _, _ = direction(HD, 640.0, 360.0, 0.0)
    assert range_from_ground(level, geometry()) is None
    above, _, _ = direction(HD, 640.0, 100.0, 0.0)
    assert range_from_ground(above, geometry()) is None


def test_ground_range_solves_for_the_ball_centre_not_the_floor():
    """
    The ball's centre sits a radius up, so the plane is a radius above the floor.

    Solving for the floor itself would put every ball slightly too far away, by
    an amount that grows as the camera looks further out.
    """
    unit, _, _ = direction(HD, 640.0, 650.0, 0.0)
    place = geometry()
    solved = range_from_ground(unit, place)
    height = place.camera_z + solved * unit[2]
    assert height == pytest.approx(place.floor_z + place.diameter / 2.0)


# --- placing a ball ----------------------------------------------------------


def test_a_ball_dead_ahead_has_no_lateral_offset():
    observation = locate(HD, (640.0, 600.0, 25.0, 25.0), geometry(), score=0.9)
    assert observation is not None
    assert observation.y == pytest.approx(0.0, abs=1e-9)
    assert observation.x > 0.0


def test_a_ball_left_of_centre_is_placed_to_the_robots_left():
    observation = locate(HD, (200.0, 600.0, 25.0, 25.0), geometry())
    assert observation.y > 0.0


def test_the_size_estimator_ignores_where_the_box_sits_in_the_frame():
    """
    Apparent size gives a range that does not care where the camera points.

    Only the range: which ray the ball is along is still the tilt's business,
    and that is what the neck tests below are about. But a wrong tilt costs
    this estimator a direction, and the ground estimator a direction and a
    distance, which is why it is the default.
    """
    high = locate(HD, (640.0, 400.0, 25.0, 25.0), geometry(), prefer=SOURCE_SIZE)
    low = locate(HD, (640.0, 700.0, 25.0, 25.0), geometry(), prefer=SOURCE_SIZE)
    assert high.range == pytest.approx(low.range)


def test_both_estimates_are_reported_whichever_is_published():
    observation = locate(HD, (640.0, 650.0, 25.0, 25.0), geometry())
    assert observation.source == SOURCE_SIZE
    assert observation.size_range > 0.0
    assert observation.ground_range > 0.0


def test_the_preferred_estimator_is_the_one_published():
    observation = locate(
        HD, (640.0, 650.0, 25.0, 25.0), geometry(), prefer=SOURCE_GROUND
    )
    assert observation.source == SOURCE_GROUND
    assert observation.range == pytest.approx(observation.ground_range)


def test_the_other_estimator_stands_in_when_the_preferred_one_has_nothing():
    """A ball above the horizon has no ground solution, but still has a size."""
    observation = locate(
        HD, (640.0, 200.0, 25.0, 25.0), geometry(), prefer=SOURCE_GROUND
    )
    assert observation is not None
    assert observation.source == SOURCE_SIZE


def test_a_range_outside_the_band_is_not_a_position():
    """
    Nothing is published rather than a guess.

    A box of two pixels above the horizon is not a marginal measurement of a
    distant ball; it is not a measurement. The size estimator puts it tens of
    metres away and the ground estimator has nothing to say, so there is no
    answer -- which is what lets the fusion node stay silent instead of
    publishing a place for the robot to drive to.
    """
    assert locate(HD, (640.0, 200.0, 2.0, 2.0), geometry(max_range=6.0)) is None


def test_the_ground_estimate_is_used_when_size_is_out_of_band():
    place = geometry(max_range=6.0)
    observation = locate(HD, (640.0, 700.0, 3.0, 3.0), place, prefer=SOURCE_SIZE)
    assert observation is not None
    assert observation.source == SOURCE_GROUND


# --- height ------------------------------------------------------------------


def test_a_ball_on_the_floor_reads_one_radius_high():
    unit_place = geometry()
    observation = locate(
        HD, (640.0, 650.0, 25.0, 25.0), unit_place, prefer=SOURCE_GROUND
    )
    assert floor_height(observation, unit_place) == pytest.approx(
        unit_place.diameter / 2.0
    )


def test_graspable_is_a_height_band_and_not_a_distance():
    """
    The pincer has no lift, so height is what decides -- as in mecanumbot_seek.

    A ball in somebody's hand is a metre up and unobtainable however close the
    robot drives; the tree needs that to be a different answer from "not here".
    """
    place = geometry()
    on_floor = locate(HD, (640.0, 650.0, 25.0, 25.0), place, prefer=SOURCE_GROUND)
    assert graspable(on_floor, place, 0.0, 0.15)

    in_hand = locate(HD, (640.0, 300.0, 25.0, 25.0), place, prefer=SOURCE_SIZE)
    assert not graspable(in_hand, place, 0.0, 0.15)


def test_camera_pitch_tilts_the_whole_ray():
    """Looking down puts a ball at the same image row lower and nearer."""
    level = locate(HD, (640.0, 600.0, 25.0, 25.0), geometry(camera_pitch=0.0))
    tilted = locate(
        HD, (640.0, 600.0, 25.0, 25.0), geometry(camera_pitch=math.radians(-15.0))
    )
    assert tilted.z < level.z
    assert tilted.x < level.x


# --- the neck ----------------------------------------------------------------


def floor_ball_box(camera, place, forward):
    """
    Return the box a ball on the floor `forward` metres ahead would make.

    Seen through `place`, dead ahead so the column is the centre one. Written
    as the forward projection rather than by running `locate` backwards, so the
    tests below check the geometry and not just that it is self-consistent.
    """
    ahead = forward - place.camera_x
    below = (place.floor_z + place.diameter / 2.0) - place.camera_z
    distance = math.hypot(ahead, below)
    in_frame = math.atan2(below, ahead) - place.camera_pitch
    return (
        camera.width / 2.0,
        camera.height / 2.0 - camera.focal_y * math.tan(in_frame),
        camera.focal_x * place.diameter / distance,
        camera.focal_y * place.diameter / distance,
    )


def test_the_neck_at_level_puts_the_lens_where_the_fixed_numbers_do():
    """The neck model and `BallGeometry`'s defaults describe the same camera."""
    place = NeckMount().geometry(geometry(), 600)
    assert place.camera_pitch == pytest.approx(0.0)
    assert place.camera_x == pytest.approx(0.13, abs=0.005)
    assert place.camera_z == pytest.approx(0.21, abs=0.005)


def test_a_larger_neck_position_looks_further_up():
    """The trees' convention: larger neck positions look further up."""
    neck = NeckMount()
    assert neck.pitch(600) == pytest.approx(0.0)
    assert neck.pitch(300) < neck.pitch(600) < neck.pitch(700)


def test_a_neck_reading_the_head_cannot_have_is_refused():
    """The board reports 0 before its first command; that is not a head pose."""
    assert NeckMount().geometry(geometry(), 0) is None


def test_a_ball_on_the_floor_seen_with_the_head_down_is_on_the_floor():
    """
    The fetch tree's failure: a floor ball called out of reach mid-sweep.

    Seen with the head tilted down and placed as if the camera looked level,
    the ray runs level instead of down, and the ball comes out a hand's width
    above the grabbers. Placed with the tilt the neck actually had, it is one
    radius up.
    """
    place = NeckMount().geometry(geometry(), 530)
    box = floor_ball_box(HD, place, forward=1.0)

    seen = locate(HD, box, place, prefer=SOURCE_SIZE)
    assert seen.x == pytest.approx(1.0)
    assert floor_height(seen, place) == pytest.approx(place.diameter / 2.0)
    assert graspable(seen, place, 0.0, 0.12)

    as_if_level = locate(HD, box, geometry(), prefer=SOURCE_SIZE)
    assert not graspable(as_if_level, geometry(), 0.0, 0.12)


def test_floor_pitch_recovers_the_tilt_a_floor_ball_was_seen_with():
    """
    A ball known to be on the floor measures the camera's tilt.

    That is the calibration `NeckMount.pitch_at_level` is waiting for: read
    this at the level neck position and it is the number to set.
    """
    place = NeckMount().geometry(geometry(), 530)
    box = floor_ball_box(HD, place, forward=1.0)
    assert floor_pitch(HD, box, place) == pytest.approx(place.camera_pitch)


def test_the_neck_history_answers_for_the_frame_not_for_now():
    """A box arrives after the head has moved on; its frame had the older tilt."""
    history = NeckHistory(stale_s=0.5)
    history.submit(0.0, 300)
    history.submit(0.3, 400)
    history.submit(0.6, 500)
    assert history.at(0.45) == 400
    assert history.at(0.6) == 500


def test_the_neck_history_refuses_frames_it_cannot_speak_for():
    """
    Before the first reading, or long after the last, there is no answer.

    A board that stops publishing leaves a plausible last value behind, and
    that is exactly the one not to believe.
    """
    history = NeckHistory(stale_s=0.5)
    assert history.at(1.0) is None
    history.submit(1.0, 600)
    assert history.at(0.9) is None
    assert history.at(1.4) == 600
    assert history.at(1.6) is None


# --- tracking ----------------------------------------------------------------


def config(**overrides):
    """Return a tracker confirming on two hits, unless a test says otherwise."""
    return BallTrackerConfig(**overrides)


def test_a_single_sighting_is_not_published():
    tracker = BallTracker(config(min_hits=2))
    assert tracker.step([BallMeasurement(1.0, 0.0, 0.03)], 0.0) == []


def test_a_second_sighting_confirms_the_ball():
    tracker = BallTracker(config(min_hits=2))
    tracker.step([BallMeasurement(1.0, 0.0, 0.03)], 0.0)
    tracks = tracker.step([BallMeasurement(1.02, 0.0, 0.03)], 0.1)
    assert len(tracks) == 1
    assert tracks[0].position[0] == pytest.approx(1.01, abs=0.01)


def test_the_estimate_is_smoothed_rather_than_snapped():
    """The range noise is the point: 1/d_px means a pixel is centimetres."""
    tracker = BallTracker(config(min_hits=1, position_gain=0.5))
    tracker.step([BallMeasurement(1.0, 0.0, 0.03)], 0.0)
    tracks = tracker.step([BallMeasurement(1.4, 0.0, 0.03)], 0.1)
    assert 1.0 < tracks[0].position[0] < 1.4


def test_a_ball_that_stops_being_seen_expires():
    """
    A picked-up ball has to stop being reported quickly.

    Coasting a ball the way a person is coasted would leave the robot driving at
    the floor where the ball used to be.
    """
    tracker = BallTracker(config(min_hits=1, max_coast_time=1.0))
    tracker.step([BallMeasurement(1.0, 0.0, 0.03)], 0.0)
    assert tracker.step([], 0.5) != []
    assert tracker.step([], 2.0) == []


def test_two_balls_far_apart_are_two_tracks():
    tracker = BallTracker(config(min_hits=1, max_association_distance=0.6))
    tracks = tracker.step(
        [BallMeasurement(1.0, 0.0, 0.03), BallMeasurement(3.0, 0.0, 0.03)], 0.0
    )
    assert len(tracks) == 2


def test_a_jump_beyond_the_association_distance_starts_a_new_track():
    tracker = BallTracker(config(min_hits=1, max_association_distance=0.5))
    tracker.step([BallMeasurement(1.0, 0.0, 0.03)], 0.0)
    tracks = tracker.step([BallMeasurement(5.0, 0.0, 0.03)], 0.1)
    assert len(tracks) == 2


def test_reported_speed_is_clamped_rather_than_believed():
    tracker = BallTracker(
        config(min_hits=1, max_association_distance=10.0, velocity_gain=1.0, max_speed=3.0)
    )
    tracker.step([BallMeasurement(0.0, 0.0, 0.03)], 0.0)
    tracker.step([BallMeasurement(5.0, 0.0, 0.03)], 0.01)
    assert tracker.tracks[0].speed <= 3.0 + 1e-9


def test_height_is_smoothed_but_never_extrapolated():
    """Somebody picking the ball up is a step, not a trend."""
    tracker = BallTracker(config(min_hits=1, position_gain=0.5))
    tracker.step([BallMeasurement(1.0, 0.0, 0.03)], 0.0)
    tracker.step([BallMeasurement(1.0, 0.0, 1.03)], 0.1)
    coasted = tracker.step([], 0.2)
    assert coasted[0].position[2] == pytest.approx(0.53, abs=1e-6)
