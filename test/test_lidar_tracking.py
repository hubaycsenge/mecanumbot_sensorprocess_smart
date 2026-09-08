#!/usr/bin/env python3
"""Unit tests for the DR-SPAAM detection tracker.

The claims here are about what the LiDAR alone is allowed to call a person.
Two of them are in tension and both matter: a 2D scan at ankle height must not
report the furniture, and it must not lose a person who is standing still half
a metre from the robot -- which is exactly where the camera cannot help and the
LiDAR is the only sensor left.

Pure numpy: no ROS, no torch, no `dr_spaam`, so this runs on a development
machine.
"""

import numpy as np
import pytest

from mecanumbot_sensorprocess_smart.lidar_tracking import MultiObjectTracker

DT = 0.2  # one cycle at the 5 Hz inference cap the Orin runs at.


def detections(*points):
    """Build the detection array for one inference."""
    return np.array(points, dtype=float).reshape(-1, 2)


def walk(tracker, start, steps=6, speed=0.5, dt=DT):
    """Walk a person along +x fast enough to trip the motion threshold."""
    position = np.array(start, dtype=float)
    for _ in range(steps):
        position = position + np.array([speed * dt, 0.0])
        tracker.update(detections(position), dt)
    return position


class TestConfirmation:
    def test_a_single_detection_is_not_a_person(self):
        tracker = MultiObjectTracker()
        assert len(tracker.update(detections((2.0, 0.0)), DT)) == 0

    def test_furniture_is_never_published(self):
        # A table leg is seen every cycle for a minute and never moves. This is
        # what `require_motion` is for, and it is the guard the re-seed below
        # must not break.
        tracker = MultiObjectTracker()
        for _ in range(300):
            tracker.update(detections((1.5, 0.5)), DT)
        assert len(tracker._confirmed_positions()) == 0

    def test_a_walking_person_is_published(self):
        tracker = MultiObjectTracker()
        walk(tracker, (2.0, 0.0))
        assert len(tracker._confirmed_positions()) == 1

    def test_require_motion_can_be_switched_off(self):
        # Publishes stationary detections outright -- and the furniture with
        # them, which is why it is not the default.
        tracker = MultiObjectTracker(require_motion=False)
        for _ in range(4):
            tracker.update(detections((1.5, 0.5)), DT)
        assert len(tracker._confirmed_positions()) == 1


class TestReseeding:
    """A dropped track must not cost a standing person their motion evidence."""

    def _person_who_stopped(self, **kwargs):
        """Walk someone in, then have them stand still at (3.0, 0.0)."""
        tracker = MultiObjectTracker(**kwargs)
        walk(tracker, (2.0, 0.0), steps=6, speed=0.9)
        for _ in range(5):
            tracker.update(detections((3.0, 0.0)), DT)
        return tracker

    def test_standing_still_does_not_lose_a_person_who_walked_in(self):
        tracker = self._person_who_stopped()
        assert len(tracker._confirmed_positions()) == 1

    def test_a_flicker_that_drops_the_track_does_not_lose_them_either(self):
        # The close-range failure: two legs resolve as one blob and then two,
        # the track misses `max_missed_time` of updates and is replaced. The
        # replacement inherits the motion this person has no way to show again.
        tracker = self._person_who_stopped()
        for _ in range(4):
            tracker.update(detections(), DT)
        assert tracker.tracks == []

        for _ in range(2):
            tracker.update(detections((3.02, 0.0)), DT)
        assert len(tracker._confirmed_positions()) == 1

    def test_the_replacement_still_has_to_be_seen_more_than_once(self):
        # Re-seeding restores what was established about this person; it does
        # not excuse the new track from proving it is there at all.
        tracker = self._person_who_stopped()
        for _ in range(4):
            tracker.update(detections(), DT)
        assert len(tracker.update(detections((3.02, 0.0)), DT)) == 0

    def test_the_memory_is_not_a_licence_for_the_furniture(self):
        # A ghost only lends its motion to something that turns up where it
        # was. A chair on the other side of the robot inherits nothing.
        tracker = self._person_who_stopped()
        for _ in range(4):
            tracker.update(detections(), DT)
        for _ in range(6):
            tracker.update(detections((-2.0, 1.0)), DT)
        assert len(tracker._confirmed_positions()) == 0

    def test_the_memory_expires(self):
        # Long after the person left, a fresh return in the same spot is a
        # fresh object and has to earn its own motion evidence.
        tracker = self._person_who_stopped()
        elapsed = 0.0
        while elapsed <= tracker.reseed_memory + 1.0:
            tracker.update(detections(), DT)
            elapsed += DT
        for _ in range(6):
            tracker.update(detections((3.02, 0.0)), DT)
        assert len(tracker._confirmed_positions()) == 0

    def test_a_ghost_of_something_that_never_moved_lends_nothing(self):
        tracker = MultiObjectTracker()
        for _ in range(5):
            tracker.update(detections((1.5, 0.5)), DT)
        for _ in range(4):
            tracker.update(detections(), DT)
        for _ in range(6):
            tracker.update(detections((1.5, 0.5)), DT)
        assert len(tracker._confirmed_positions()) == 0

    def test_reseeding_is_off_when_the_memory_is_zero(self):
        tracker = self._person_who_stopped(reseed_memory=0.0)
        for _ in range(4):
            tracker.update(detections(), DT)
        for _ in range(6):
            tracker.update(detections((3.02, 0.0)), DT)
        assert len(tracker._confirmed_positions()) == 0


class TestSkippedInferences:
    """`predict_only` runs on the scans where the GPU was spared."""

    def test_tracks_keep_moving_between_inferences(self):
        tracker = MultiObjectTracker()
        walk(tracker, (2.0, 0.0), speed=1.0)
        before = tracker._confirmed_positions()[0][0]
        tracker.predict_only(0.1)
        after = tracker._confirmed_positions()[0][0]
        assert after > before

    def test_a_track_still_expires_while_the_detector_is_skipped(self):
        tracker = MultiObjectTracker()
        walk(tracker, (2.0, 0.0))
        for _ in range(10):
            tracker.predict_only(0.1)
        assert tracker.tracks == []

    def test_a_skipped_run_ages_the_reseed_memory_too(self):
        tracker = MultiObjectTracker()
        position = walk(tracker, (2.0, 0.0))
        elapsed = 0.0
        while elapsed <= tracker.reseed_memory + 1.0:
            tracker.predict_only(0.1)
            elapsed += 0.1
        assert tracker.ghosts == []
        for _ in range(6):
            tracker.update(detections(position), DT)
        assert len(tracker._confirmed_positions()) == 0


class TestAssociation:
    def test_two_people_keep_their_own_tracks(self):
        tracker = MultiObjectTracker()
        for step in range(6):
            offset = 0.1 * step
            tracker.update(detections((2.0 + offset, 0.0), (2.0 + offset, 1.5)), DT)
        positions = tracker._confirmed_positions()
        assert len(positions) == 2
        assert abs(positions[0][1] - positions[1][1]) == pytest.approx(1.5, abs=0.3)

    def test_a_detection_beyond_the_gate_starts_a_new_track(self):
        tracker = MultiObjectTracker()
        walk(tracker, (2.0, 0.0))
        assert tracker.next_id == 1
        tracker.update(detections((2.0, 4.0)), DT)
        assert tracker.next_id == 2
