#!/usr/bin/env python3
"""
Unit tests for the scan clusters and what the fusion asks of them.

Each of these is a way a person the camera had found was not tracked in the
bags of 2026-09-23 .. 2026-09-30: ranged onto the wall behind their legs, ranged
onto nothing at all between the two, missed because the camera's bearings are
7.5 degrees off the LiDAR's, or published beyond the wall they were sitting
against. No ROS.
"""

import math

import numpy as np
import pytest

from mecanumbot_sensorprocess_smart.scan_clusters import (
    CandidateConfig,
    OcclusionConfig,
    StaticMap,
    candidates,
    choose_for_wedge,
    corroborates_close_range,
    place_on_structure,
    segment_scan,
    structure_range,
    to_map,
)

# The LD08 as the bags have it: a full turn from 0, about 1.7 degrees a beam.
N = 210
INC = 2.0 * math.pi / N


def scan(background=4.25, **objects):
    """
    Return a 0..2pi scan of a wall at `background` with objects in front.

    Each keyword is ``name=(bearing_deg, beams, range)``: `beams` neighbouring
    returns at `range`, centred on the bearing.
    """
    ranges = np.full(N, float(background))
    for bearing_deg, beams, distance in objects.values():
        centre = int(round(math.radians(bearing_deg) / INC)) % N
        for offset in range(beams):
            ranges[(centre - beams // 2 + offset) % N] = distance
    return ranges


def clusters_of(ranges, **kwargs):
    return segment_scan(ranges, 0.0, INC, **kwargs)


def room(size=6.0, resolution=0.05, origin=-1.0):
    """Return a square room: walls two cells thick, unknown outside them."""
    cells = int(round(size / resolution))
    pad = int(round(1.0 / resolution))
    grid = np.full((cells + 2 * pad, cells + 2 * pad), -1, dtype=np.int8)
    grid[pad : pad + cells, pad : pad + cells] = 0
    grid[pad : pad + 2, pad : pad + cells] = 100
    grid[pad + cells - 2 : pad + cells, pad : pad + cells] = 100
    grid[pad : pad + cells, pad : pad + 2] = 100
    grid[pad : pad + cells, pad + cells - 2 : pad + cells] = 100
    return grid, resolution, origin - 1.0


class TestSegmentation:
    def test_legs_in_front_of_a_wall_are_their_own_cluster(self):
        found = clusters_of(scan(legs=(20.0, 2, 1.98)))
        legs = [c for c in found if c.range < 3.0]
        assert len(legs) == 1
        assert legs[0].beams == 2
        assert legs[0].range == pytest.approx(1.98)
        assert math.degrees(legs[0].bearing) == pytest.approx(20.0, abs=1.5)

    def test_a_cluster_position_is_a_real_return_not_a_blend(self):
        # Two beams of leg at 1.98 m and five of wall at 4.25 m is the wedge of
        # the 09:02 bag of 2026-09-30. The percentile lookup made 2.47 m of it.
        found = clusters_of(scan(legs=(20.0, 2, 1.98)))
        assert not any(2.2 < c.range < 4.0 for c in found)

    def test_right_of_the_axis_is_a_negative_bearing(self):
        found = clusters_of(scan(legs=(-21.0, 5, 2.0)))
        legs = [c for c in found if c.range < 3.0][0]
        assert math.degrees(legs.bearing) == pytest.approx(-21.0, abs=1.5)
        assert legs.bearing_min < legs.bearing < legs.bearing_max

    def test_a_cluster_across_the_seam_is_one_cluster(self):
        found = clusters_of(scan(legs=(0.0, 6, 1.5)))
        legs = [c for c in found if c.range < 3.0]
        assert len(legs) == 1
        assert legs[0].beams == 6
        assert abs(math.degrees(legs[0].bearing)) < 2.0

    def test_one_dropped_return_does_not_split_a_person(self):
        ranges = scan(legs=(40.0, 5, 1.2))
        ranges[int(round(math.radians(40.0) / INC))] = float("inf")
        legs = [c for c in clusters_of(ranges) if c.range < 3.0]
        assert len(legs) == 1

    def test_width_is_the_size_of_the_thing(self):
        legs = [c for c in clusters_of(scan(legs=(30.0, 5, 2.0))) if c.range < 3.0][0]
        # Five beams at 2 m, 1.7 degrees apart, is about 0.3 m of surface.
        assert legs.width == pytest.approx(0.30, abs=0.06)

    def test_returns_past_max_range_are_not_clustered(self):
        assert clusters_of(scan(background=8.0), max_range=6.0) == []

    def test_an_empty_scan_is_no_clusters(self):
        assert segment_scan([], 0.0, INC) == []


class TestCandidates:
    def test_a_person_sized_cluster_off_the_map_is_a_candidate(self):
        grid, resolution, origin = room()
        static = StaticMap(grid, resolution, origin, origin)
        pose = (1.0, 1.0, 0.0)
        # Wall of the room is 3 m ahead; legs at 1.5 m are not on the map.
        ranges = scan(background=3.0, legs=(0.0, 4, 1.5))
        found = candidates(clusters_of(ranges), pose, static)
        assert len(found) == 1
        assert found[0].range == pytest.approx(1.5)

    def test_the_wall_is_not_a_candidate(self):
        found = candidates(clusters_of(scan(background=3.0)), None, None)
        assert found == []  # far too wide to be anybody

    def test_something_the_map_explains_is_not_a_candidate(self):
        grid, resolution, origin = room()
        # A mapped pillar 1.5 m in front of the robot.
        column = int(round((2.5 - origin) / resolution))
        row = int(round((1.0 - origin) / resolution))
        grid[row - 2 : row + 3, column - 2 : column + 3] = 100
        static = StaticMap(grid, resolution, origin, origin)
        ranges = scan(background=6.0, pillar=(0.0, 4, 1.45))
        clusters = clusters_of(ranges, max_range=5.0)
        assert candidates(clusters, (1.0, 1.0, 0.0), static) == []
        # ... and with no map to explain it, it is one.
        assert len(candidates(clusters, (1.0, 1.0, 0.0), None)) == 1

    def test_a_table_leg_is_too_thin(self):
        found = candidates(clusters_of(scan(leg=(10.0, 1, 1.0))), None, None)
        assert found == []

    def test_beyond_reach_is_not_a_candidate(self):
        cfg = CandidateConfig(max_range=4.5)
        found = candidates(
            clusters_of(scan(background=9.0, legs=(0.0, 3, 5.5)), max_range=6.0),
            None,
            None,
            cfg,
        )
        assert found == []

    def test_unknown_space_counts_as_explained(self):
        grid, resolution, origin = room()
        static = StaticMap(grid, resolution, origin, origin)
        assert bool(static.explained([-50.0], [-50.0])[0])
        assert static.state(-50.0, -50.0) == StaticMap.UNKNOWN


class TestWedge:
    def test_the_wedge_picks_the_cluster_it_points_at(self):
        found = candidates(
            clusters_of(scan(person=(-21.0, 5, 2.0), other=(15.0, 4, 1.0))), None, None
        )
        chosen = choose_for_wedge(found, math.radians(-26.0), math.radians(-14.0))
        assert chosen.range == pytest.approx(2.0)

    def test_the_camera_offset_is_what_finds_the_legs(self):
        # The 09:02 bag: the camera says -19..-7 degrees, the legs are at -21.
        # Unshifted and with no margin the wedge's middle is 8 degrees off them.
        found = candidates(clusters_of(scan(person=(-25.0, 3, 2.0))), None, None)
        low, high = math.radians(-19.0), math.radians(-7.0)
        assert choose_for_wedge(found, low, high) is None
        offset = math.radians(-7.5)
        chosen = choose_for_wedge(found, low + offset, high + offset)
        assert chosen is not None and chosen.range == pytest.approx(2.0)

    def test_of_two_in_the_wedge_the_one_nearest_its_middle(self):
        found = candidates(
            clusters_of(scan(middle=(0.0, 4, 3.0), edge=(11.0, 3, 1.0))), None, None
        )
        chosen = choose_for_wedge(found, math.radians(-10.0), math.radians(10.0))
        assert chosen.range == pytest.approx(3.0)

    def test_nothing_in_the_wedge_is_none(self):
        found = candidates(clusters_of(scan(person=(90.0, 4, 2.0))), None, None)
        assert choose_for_wedge(found, math.radians(-10.0), math.radians(10.0)) is None

    def test_a_near_body_backs_up_a_close_range_box(self):
        found = candidates(clusters_of(scan(legs=(0.0, 8, 0.8))), None, None)
        low, high = math.radians(-25.0), math.radians(25.0)
        assert corroborates_close_range(found, low, high, max_range=3.0)

    def test_a_frame_filling_box_on_an_empty_wedge_is_not_backed_up(self):
        # A blurred plant shelf: only the room in that direction.
        found = candidates(clusters_of(scan(person=(90.0, 4, 2.0))), None, None)
        low, high = math.radians(-25.0), math.radians(25.0)
        assert not corroborates_close_range(found, low, high, max_range=3.0)

    def test_a_far_candidate_does_not_back_up_a_close_range_box(self):
        found = candidates(clusters_of(scan(person=(0.0, 3, 3.8))), None, None)
        low, high = math.radians(-25.0), math.radians(25.0)
        assert not corroborates_close_range(found, low, high, max_range=3.0)

    def test_the_fallback_range_is_an_actual_return(self):
        wedge = [1.98, 1.99, 4.25, 4.25, 4.25, 4.25, 4.25]
        assert structure_range(wedge) in wedge
        # ... where the interpolating percentile is not.
        assert float(np.percentile(wedge, 20)) not in wedge

    def test_the_fallback_has_nothing_to_say_about_no_returns(self):
        assert structure_range([]) is None


class TestStructure:
    def setup_method(self):
        grid, resolution, origin = room()
        self.grid = grid
        self.resolution = resolution
        self.origin = origin
        # Room interior runs from -1 to 5 in both axes, walls at its edges.
        self.robot = (2.0, 2.0)

    def static(self):
        return StaticMap(self.grid, self.resolution, self.origin, self.origin, 0.1)

    def test_free_floor_is_left_where_it_is(self):
        assert place_on_structure(self.static(), self.robot, (3.0, 2.0)) == (3.0, 2.0)

    def test_against_the_outer_wall_the_person_is_in_front_of_it(self):
        # The chair by the wall, 2026-09-30: the old walk went on through the
        # wall and published the person off the map.
        cfg = OcclusionConfig(front_standoff=0.3)
        x, y = place_on_structure(self.static(), self.robot, (4.95, 2.0), cfg)
        assert x == pytest.approx(4.65, abs=0.01)
        assert y == pytest.approx(2.0)
        assert self.static().state(x, y) == StaticMap.FREE

    def test_behind_a_partition_with_floor_on_the_far_side(self):
        column = int(round((3.5 - self.origin) / self.resolution))
        self.grid[:, column : column + 2] = np.where(
            self.grid[:, column : column + 2] == 0, 100, self.grid[:, column : column + 2]
        )
        cfg = OcclusionConfig(behind_buffer=0.5)
        x, y = place_on_structure(self.static(), self.robot, (3.5, 2.0), cfg)
        assert x > 3.9  # past the partition
        assert self.static().state(x, y) == StaticMap.FREE

    def test_a_partition_with_no_room_behind_it_is_not_looked_behind(self):
        # A panel 0.2 m in front of the outer wall: free on the far side, but
        # not for a person half a metre further on.
        column = int(round((4.6 - self.origin) / self.resolution))
        self.grid[:, column : column + 2] = np.where(
            self.grid[:, column : column + 2] == 0, 100, self.grid[:, column : column + 2]
        )
        x, _ = place_on_structure(self.static(), self.robot, (4.6, 2.0))
        assert x < 4.6

    def test_with_no_map_the_hit_is_kept(self):
        assert place_on_structure(None, self.robot, (4.95, 2.0)) == (4.95, 2.0)

    def test_scan_points_go_through_the_pose(self):
        xs, ys = to_map([(1.0, 0.0)], (2.0, 3.0, math.pi / 2.0))
        assert xs[0] == pytest.approx(2.0)
        assert ys[0] == pytest.approx(4.0)
