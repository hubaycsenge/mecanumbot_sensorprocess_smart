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

from mecanumbot_sensorprocess_smart.lidar_tracking import (
    MultiObjectTracker,
    Track,
    stamp_gap_seconds,
    from_frame,
    nearest_index,
    to_frame,
)

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


class TestTrackingFrame:
    """
    The tracker runs in odom, and the robot is what moves.

    Each case drives the robot and feeds the tracker what the node would: the
    scan-frame detection carried into odom through the robot's pose.
    """

    @staticmethod
    def drive(tracker, world_point_at, steps=15, speed=0.26):
        """Drive the robot along +x; return the scan-frame view of the last point."""
        for i in range(steps):
            pose = (speed * DT * (i + 1), 0.0, 0.0)
            in_scan = from_frame([world_point_at(i)], *pose)
            tracker.update(to_frame(in_scan, *pose), DT)
        return in_scan

    def test_a_person_following_the_robot_is_published(self):
        # A metre behind a robot at full speed: still in the scan frame, which
        # is why the scan-frame tracker never published a follower.
        tracker = MultiObjectTracker()
        in_scan = self.drive(tracker, lambda i: (0.26 * DT * (i + 1) - 1.0, 0.0))
        assert np.allclose(in_scan, [[-1.0, 0.0]])
        assert len(tracker._confirmed_positions()) == 1

    def test_furniture_driven_past_is_not_published(self):
        # Sweeps through the scan frame at the robot's speed; stands still in odom.
        tracker = MultiObjectTracker()
        self.drive(tracker, lambda i: (1.5, 0.5))
        assert len(tracker._confirmed_positions()) == 0

    def test_turning_on_the_spot_keeps_a_track(self):
        # A standing person two metres off, while the robot turns 90 degrees in
        # a second: in the scan frame they jump ~0.6 m per cycle, beyond the gate.
        tracker = MultiObjectTracker(require_motion=False)
        person = np.array([[2.0, 0.0]])
        for i in range(5):
            pose = (0.0, 0.0, (i + 1) * np.pi / 10)
            tracker.update(to_frame(from_frame(person, *pose), *pose), DT)
        assert len(tracker.tracks) == 1


class TestFrames:
    def test_round_trip(self):
        pose = (1.2, -0.4, 2.1)
        points = np.array([[0.5, 0.3], [-2.0, 1.0]])
        assert np.allclose(from_frame(to_frame(points, *pose), *pose), points)

    def test_a_quarter_turn(self):
        assert np.allclose(to_frame([[1.0, 0.0]], 1.0, 0.0, np.pi / 2), [[1.0, 1.0]])

    def test_nearest_index(self):
        assert nearest_index([[3.0, 0.0], [0.5, -0.5], [-1.0, 0.0]]) == 1
        assert nearest_index(np.empty((0, 2))) is None


class TestRotationLoophole:
    """What the robot's own turning may and may not be taken to prove.

    Tracking in odom stops the furniture a driving robot passes from looking
    like people. It does not cover turning on the spot: a detection that keeps
    a fixed bearing while the robot spins sweeps an arc in odom at `omega * r`,
    which at the ranges and spin rates of the 2026-09-22/23 bags is walking
    pace. These are the claims that close that, and the ones that make sure it
    is closed no further than that -- a leading tree spins exactly when it most
    needs to see a person.
    """

    SPIN = 0.39  # rad/s, the in-place turn rate in the leading trees.
    RANGE = 0.87  # m, the range every artefact detection in those bags sat at.

    def spin_in_place(self, tracker, steps=12, yaw_rate=SPIN, radius=RANGE, dt=DT):
        """Sweep a fixed-bearing return around the robot, as a spin does.

        The robot stands still and turns; a detection the robot carries with it
        traces a circle of `radius` in the tracking frame. Nothing has moved in
        the room, so nothing here is a person.
        """
        angle = 0.0
        for _ in range(steps):
            angle += yaw_rate * dt
            point = (radius * np.cos(angle), radius * np.sin(angle))
            tracker.update(detections(point), dt, yaw_rate)
        return tracker

    def test_a_spin_does_not_manufacture_a_person(self):
        tracker = MultiObjectTracker()
        self.spin_in_place(tracker)
        assert len(tracker._confirmed_positions()) == 0

    def test_without_the_limit_the_spin_does_manufacture_one(self):
        # The behaviour every bag before this was recorded with. Kept as a test
        # so the loophole cannot be reopened without a red light.
        tracker = MultiObjectTracker(motion_yaw_rate_limit=0.0)
        self.spin_in_place(tracker)
        assert len(tracker._confirmed_positions()) == 1

    def test_the_arc_really_is_fast_enough_to_have_fooled_the_gate(self):
        # Guards the premise rather than the fix: if the arc were slower than
        # the motion threshold there would have been no loophole, and the two
        # tests above would pass for the wrong reason.
        speed_thresh = Track(np.array([0.0, 0.0]), 0).speed_thresh
        assert self.SPIN * self.RANGE > speed_thresh

    def test_a_person_who_walked_before_the_spin_survives_it(self):
        # Motion evidence already earned was earned honestly. Losing it here
        # would mean the robot forgets the human every time it turns to check
        # on them, which is the manoeuvre this whole gate exists to serve.
        tracker = MultiObjectTracker()
        walk(tracker, (2.0, 0.0), steps=6, speed=0.9)
        assert len(tracker._confirmed_positions()) == 1
        for _ in range(6):
            tracker.update(detections((2.9, 0.0)), DT, self.SPIN)
        assert len(tracker._confirmed_positions()) == 1

    def test_a_person_walking_after_the_spin_is_still_found(self):
        # The gate withholds a conclusion; it does not poison the track. Once
        # the robot stops turning the same person earns their evidence.
        tracker = MultiObjectTracker()
        self.spin_in_place(tracker)
        assert len(tracker._confirmed_positions()) == 0
        walk(tracker, (0.87, 0.0), steps=8, speed=0.9)
        assert len(tracker._confirmed_positions()) == 1

    def test_slow_turning_still_counts_as_evidence(self):
        # Ordinary path following yaws gently and must not be swept up: the
        # median over the 2026-09-23 bag was 4.6 deg/s, well under the limit.
        tracker = MultiObjectTracker()
        walk_rate = np.radians(4.6)
        position = np.array([2.0, 0.0])
        for _ in range(8):
            position = position + np.array([0.9 * DT, 0.0])
            tracker.update(detections(position), DT, walk_rate)
        assert len(tracker._confirmed_positions()) == 1

    def test_a_spin_cannot_launder_evidence_through_a_reseed(self):
        # The other way the same rotation could get in: a spin drops and
        # replaces tracks constantly, and a replacement inherits the ghost's
        # motion. That inheritance is refused while the robot is turning.
        tracker = MultiObjectTracker(motion_yaw_rate_limit=0.0)
        self.spin_in_place(tracker)  # a ghost with has_moved, made by rotation
        assert len(tracker._confirmed_positions()) == 1
        tracker.motion_yaw_rate_limit = 0.15
        for _ in range(4):
            tracker.update(detections(), DT, TestRotationLoophole.SPIN)
        assert tracker.tracks == []
        for _ in range(6):
            tracker.update(detections((0.87, 0.02)), DT, TestRotationLoophole.SPIN)
        assert len(tracker._confirmed_positions()) == 0

    def test_a_reseed_while_standing_still_is_untouched(self):
        # The close-range flicker case must keep working: it is the reason the
        # re-seed exists, and the robot is not turning while it happens. The
        # person has to have stopped before the flicker, or the estimate coasts
        # away at walking speed while the track is missing and the ghost is
        # left too far from the return that replaces it to lend it anything --
        # which is the tracker behaving correctly, not the gate.
        tracker = MultiObjectTracker()
        walk(tracker, (2.0, 0.0), steps=6, speed=0.9)
        for _ in range(5):
            tracker.update(detections((3.08, 0.0)), DT)
        for _ in range(4):
            tracker.update(detections(), DT)
        assert tracker.tracks == []
        for _ in range(2):
            tracker.update(detections((3.10, 0.0)), DT)
        assert len(tracker._confirmed_positions()) == 1

    def test_turning_is_a_question_about_the_limit(self):
        tracker = MultiObjectTracker(motion_yaw_rate_limit=0.15)
        assert tracker.turning(0.39)
        assert tracker.turning(-0.39)
        assert not tracker.turning(0.05)
        assert not MultiObjectTracker(motion_yaw_rate_limit=0.0).turning(10.0)


class TestStampGap:
    """
    Comparing a scan's stamp with a transform's.

    The gap decides whether falling back to the newest transform is routine or
    worth a warning: `odom` at 50 Hz means the newest one is up to 20 ms old,
    while the scan is stamped now, so asking TF for the scan's own time is
    asking for the future and is refused. That fallback costs 0.2 deg of yaw at
    a spin's rate -- 4 mm at 1 m -- so it is not the error the scan-time lookup
    exists to prevent, and it must not fill the log.
    """

    class Stamp:
        """The `.sec` / `.nanosec` of a `builtin_interfaces/Time`."""

        def __init__(self, sec, nanosec):
            self.sec = sec
            self.nanosec = nanosec

    def test_a_scan_ahead_of_the_transform_is_positive(self):
        gap = stamp_gap_seconds(self.Stamp(10, 500_000_000), self.Stamp(10, 490_000_000))
        assert gap == pytest.approx(0.01)

    def test_a_scan_behind_the_transform_is_negative(self):
        gap = stamp_gap_seconds(self.Stamp(10, 490_000_000), self.Stamp(10, 500_000_000))
        assert gap == pytest.approx(-0.01)

    def test_it_carries_across_a_second_boundary(self):
        gap = stamp_gap_seconds(self.Stamp(11, 5_000_000), self.Stamp(10, 995_000_000))
        assert gap == pytest.approx(0.01)

    def test_identical_stamps_are_no_gap(self):
        assert stamp_gap_seconds(self.Stamp(7, 123), self.Stamp(7, 123)) == 0.0

    def test_the_gap_seen_at_the_lab_is_inside_the_default_tolerance(self):
        # The real numbers from the warning: requested 1790161555.820934,
        # newest 1790161555.811217. One odom period, and the reason the default
        # tolerance is 0.02 s rather than something tighter.
        gap = stamp_gap_seconds(
            self.Stamp(1790161555, 820_934_000), self.Stamp(1790161555, 811_217_000)
        )
        assert gap == pytest.approx(0.0097, abs=1e-4)
        assert 0.0 <= gap <= 0.02

    def test_a_real_stall_is_outside_it(self):
        gap = stamp_gap_seconds(self.Stamp(12, 0), self.Stamp(10, 0))
        assert gap > 0.02
