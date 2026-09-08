#!/usr/bin/env python3
"""Unit tests for the map-frame person tracker.

These pin down the two things the tracker was added for - that a person is not
lost when the camera fails to see them for a moment, and that coasting does not
become a phantom - plus the noise smoothing that makes the reported position
worth handing to Nav2. No ROS: the tracker takes and returns plain floats.
"""

import math

import pytest

from mecanumbot_sensorprocess_smart.person_tracking import (
    CameraCoverage,
    Measurement,
    PersonTracker,
    TrackerConfig,
    combine_measurements,
)

CONFIG = TrackerConfig()

#: A camera at the origin looking down +x, with the defaults the node ships.
CAMERA = CameraCoverage(x=0.0, y=0.0, yaw=0.0)


def confirmed(tracker, measurements, now):
    """Step the tracker and return the confirmed positions."""
    return [track.position for track in tracker.step(measurements, now)]


def seen_at(x, y):
    """Build a camera-corroborated measurement."""
    return [Measurement(x, y, True)]


def lidar_at(x, y):
    """Build a LiDAR-only measurement: enough to sustain a track, not start one."""
    return [Measurement(x, y, False)]


class TestAcquisition:
    def test_a_single_sighting_is_not_yet_a_person(self):
        tracker = PersonTracker(CONFIG)
        assert confirmed(tracker, seen_at(2.0, 0.0), 0.0) == []

    def test_two_sightings_confirm(self):
        tracker = PersonTracker(CONFIG)
        tracker.step(seen_at(2.0, 0.0), 0.0)
        positions = confirmed(tracker, seen_at(2.0, 0.0), 0.1)
        assert len(positions) == 1
        assert positions[0][0] == pytest.approx(2.0, abs=0.3)

    def test_lidar_alone_never_creates_a_person(self):
        # A leg-sized return with no camera behind it is a chair leg as easily
        # as a person; only the camera may assert that something is one.
        tracker = PersonTracker(CONFIG)
        for step in range(10):
            assert confirmed(tracker, lidar_at(2.0, 0.0), step * 0.1) == []
        assert tracker.track_count == 0


class TestCoasting:
    """The failure the tracker exists for: the camera drops a real person."""

    def _walking_person(self, tracker, steps=4, dt=0.1, speed=1.0):
        """Walk someone along +x, corroborated, and return the time reached."""
        for step in range(steps):
            now = step * dt
            tracker.step(seen_at(2.0 + speed * now, 0.0), now)
        return (steps - 1) * dt

    def test_a_missed_frame_does_not_lose_the_person(self):
        tracker = PersonTracker(CONFIG)
        now = self._walking_person(tracker)
        # The camera rejects the next frame -- the long-skirt case. Nothing at
        # all arrives, and the person is still reported.
        positions = confirmed(tracker, [], now + 0.1)
        assert len(positions) == 1

    def test_coasting_keeps_the_person_moving_not_frozen(self):
        tracker = PersonTracker(CONFIG)
        now = self._walking_person(tracker, steps=8, speed=1.0)
        last = tracker.confirmed_tracks()[0].position[0]
        coasted = confirmed(tracker, [], now + 0.3)[0][0]
        # A constant-velocity estimate carries on; a frozen one would not.
        assert coasted > last

    def test_lidar_sustains_a_track_the_camera_lost(self):
        tracker = PersonTracker(CONFIG)
        now = self._walking_person(tracker)
        # Well past max_coast_time, but the LiDAR keeps seeing them.
        for step in range(1, 15):
            now += 0.1
            tracker.step(lidar_at(2.4 + 0.02 * step, 0.0), now)
        assert len(tracker.confirmed_tracks()) == 1

    def test_coasting_expires_when_nothing_is_seen_at_all(self):
        tracker = PersonTracker(CONFIG)
        now = self._walking_person(tracker)
        tracker.step([], now + CONFIG.max_coast_time + 0.1)
        assert tracker.confirmed_tracks() == []

    def test_lidar_only_support_expires_without_the_camera_agreeing_again(self):
        # The other half of the bargain: memory, not belief. If the camera
        # never corroborates again the track dies, however long the LiDAR
        # keeps finding something in that spot.
        tracker = PersonTracker(CONFIG)
        now = self._walking_person(tracker)
        while now < CONFIG.max_uncorroborated_time + 1.0:
            now += 0.1
            tracker.step(lidar_at(2.4, 0.0), now)
        assert tracker.confirmed_tracks() == []

    def test_the_camera_coming_back_renews_the_budget(self):
        tracker = PersonTracker(CONFIG)
        now = self._walking_person(tracker)
        for _ in range(20):
            now += 0.1
            tracker.step(lidar_at(2.4, 0.0), now)
        now += 0.1
        tracker.step(seen_at(2.4, 0.0), now)
        while now < CONFIG.max_uncorroborated_time:
            now += 0.1
            tracker.step(lidar_at(2.4, 0.0), now)
        assert len(tracker.confirmed_tracks()) == 1


class TestBlindZone:
    """The close-range case: the camera is not disagreeing, it is absent.

    Every test here is a claim about the *sensor*. The exemption is only ever
    justified by the camera being unable to see the point in question, so each
    one pairs the relaxed behaviour inside the blind zone with the unchanged
    behaviour just outside it.
    """

    def near(self, tracker, now, x=0.5, y=0.0):
        """One LiDAR-only measurement inside the blind range."""
        return tracker.step(lidar_at(x, y), now, CAMERA)

    def test_the_camera_cannot_see_someone_standing_on_top_of_it(self):
        assert CAMERA.cannot_see(0.5, 0.0)
        assert not CAMERA.cannot_see(3.0, 0.0)

    def test_the_camera_cannot_see_behind_itself(self):
        # The person being led walks behind the robot for most of a trial.
        assert CAMERA.cannot_see(-3.0, 0.0)
        assert CAMERA.cannot_see(3.0, 3.0)

    def test_each_exemption_can_be_switched_off_alone(self):
        range_only = CameraCoverage(0.0, 0.0, 0.0, exempt_outside_fov=False)
        assert range_only.cannot_see(0.5, 0.0)
        assert not range_only.cannot_see(-3.0, 0.0)

        fov_only = CameraCoverage(0.0, 0.0, 0.0, exempt_close_range=False)
        assert not fov_only.cannot_see(0.5, 0.0)
        assert fov_only.cannot_see(-3.0, 0.0)

    def test_consistent_lidar_alone_is_a_person_in_the_blind_zone(self):
        # The repair the leading experiment needs: nobody is ever going to
        # corroborate a detection half a metre from a shin-height camera.
        tracker = PersonTracker(CONFIG)
        for step in range(CONFIG.blind_zone_min_hits + 1):
            self.near(tracker, step * 0.1)
        assert len(tracker.confirmed_tracks()) == 1

    def test_one_or_two_returns_are_not_yet_a_person(self):
        # It is paid for in consistency: `blind_zone_min_hits` is deliberately
        # higher than the corroborated `min_hits`.
        tracker = PersonTracker(CONFIG)
        for step in range(CONFIG.min_hits + 1):
            self.near(tracker, step * 0.1)
        assert tracker.confirmed_tracks() == []

    def test_lidar_alone_still_never_creates_a_person_where_the_camera_looks(self):
        # The original rule, unchanged, everywhere the camera had a chance.
        tracker = PersonTracker(CONFIG)
        for step in range(20):
            tracker.step(lidar_at(3.0, 0.0), step * 0.1, CAMERA)
        assert tracker.confirmed_tracks() == []

    def test_creation_can_be_switched_off_without_losing_the_clock_hold(self):
        cfg = TrackerConfig(blind_zone_creates_tracks=False)
        tracker = PersonTracker(cfg)
        for step in range(20):
            tracker.step(lidar_at(0.5, 0.0), step * 0.1, CAMERA)
        assert tracker.track_count == 0

    def test_a_person_close_by_outlives_the_uncorroborated_budget(self):
        # The actual failure: the subject walks up to the robot and stays
        # there. Before the exemption they vanished after
        # `max_uncorroborated_time` and could not come back, because at that
        # range the camera never agrees again.
        tracker = PersonTracker(CONFIG)
        tracker.step(seen_at(2.0, 0.0), 0.0, CAMERA)
        tracker.step(seen_at(2.0, 0.0), 0.1, CAMERA)
        now = 0.1
        while now < CONFIG.max_uncorroborated_time * 3:
            now += 0.1
            tracker.step(lidar_at(0.5, 0.0), now, CAMERA)
        assert len(tracker.confirmed_tracks()) == 1

    def test_the_same_person_is_lost_where_the_camera_could_have_seen_them(self):
        # The control for the test above: identical, except the person stands
        # where the camera is looking, so its silence is real disagreement.
        tracker = PersonTracker(CONFIG)
        tracker.step(seen_at(2.0, 0.0), 0.0, CAMERA)
        tracker.step(seen_at(2.0, 0.0), 0.1, CAMERA)
        now = 0.1
        while now < CONFIG.max_uncorroborated_time * 3:
            now += 0.1
            tracker.step(lidar_at(2.0, 0.0), now, CAMERA)
        assert tracker.confirmed_tracks() == []

    def test_walking_away_still_ends_the_track(self):
        # Coasting is memory, not belief, and the blind zone does not change
        # that: `max_coast_time` is untouched, so a track nothing is measuring
        # dies on schedule however blind the camera is.
        tracker = PersonTracker(CONFIG)
        for step in range(CONFIG.blind_zone_min_hits + 1):
            self.near(tracker, step * 0.1)
        now = CONFIG.blind_zone_min_hits * 0.1
        tracker.step([], now + CONFIG.max_coast_time + 0.1, CAMERA)
        assert tracker.confirmed_tracks() == []

    def test_the_exemption_itself_runs_out(self):
        # The outer bound. A leg-sized return beside the robot that the camera
        # never vouches for must not become a permanent person.
        tracker = PersonTracker(CONFIG)
        now = 0.0
        while now < CONFIG.max_blind_zone_time + 1.0:
            self.near(tracker, now)
            now += 0.1
        assert tracker.confirmed_tracks() == []

    def test_a_silenced_track_comes_back_the_moment_the_camera_can_check(self):
        # Silencing is not deletion: the track goes on absorbing the LiDAR
        # returns, so one corroborated frame restores the person rather than
        # starting the acquisition from scratch.
        tracker = PersonTracker(CONFIG)
        now = 0.0
        while now < CONFIG.max_blind_zone_time + 1.0:
            self.near(tracker, now)
            now += 0.1
        assert tracker.confirmed_tracks() == []
        assert tracker.track_count == 1

        now += 0.1
        tracker.step(seen_at(0.5, 0.0), now, CAMERA)
        assert len(tracker.confirmed_tracks()) == 1

    def test_the_camera_agreeing_resets_the_blind_budget(self):
        tracker = PersonTracker(CONFIG)
        now = 0.0
        while now < CONFIG.max_blind_zone_time - 1.0:
            self.near(tracker, now)
            now += 0.1
        # They step back into view for one frame, then return to the blind zone.
        now += 0.1
        tracker.step(seen_at(0.5, 0.0), now, CAMERA)
        for _ in range(20):
            now += 0.1
            self.near(tracker, now)
        assert len(tracker.confirmed_tracks()) == 1

    def test_without_a_camera_pose_nothing_is_exempt(self):
        # A missing head transform must fail towards the stricter behaviour,
        # not towards trusting the LiDAR everywhere.
        tracker = PersonTracker(CONFIG)
        for step in range(20):
            tracker.step(lidar_at(0.5, 0.0), step * 0.1, None)
        assert tracker.confirmed_tracks() == []

    def test_the_zone_follows_the_head_not_the_map_origin(self):
        camera = CameraCoverage(x=4.0, y=1.0, yaw=math.pi)
        assert camera.cannot_see(4.3, 1.0)
        assert camera.cannot_see(6.0, 1.0)
        assert not camera.cannot_see(2.0, 1.0)


class TestSmoothing:
    def test_range_noise_is_averaged_away(self):
        # The wedge lookup jumps by tens of centimetres as it slides off a leg.
        # The filtered position should sit near the truth, not near the jumps.
        tracker = PersonTracker(CONFIG)
        noise = [0.0, 0.3, -0.25, 0.28, -0.3, 0.22, -0.26, 0.3]
        for step, offset in enumerate(noise):
            tracker.step(seen_at(2.0 + offset, 0.0), step * 0.1)
        x, _ = tracker.confirmed_tracks()[0].position
        assert abs(x - 2.0) < 0.2

    def test_velocity_is_estimated_for_a_walking_person(self):
        tracker = PersonTracker(CONFIG)
        for step in range(20):
            now = step * 0.1
            tracker.step(seen_at(1.0 * now, 0.0), now)
        vx, vy = tracker.confirmed_tracks()[0].velocity
        assert vx == pytest.approx(1.0, abs=0.35)
        assert abs(vy) < 0.3

    def test_reported_speed_is_clamped_to_something_human(self):
        # Two people swapping tracks for a frame must not be reported as one
        # person sprinting.
        cfg = TrackerConfig(max_reported_speed=2.5, max_association_distance=10.0)
        tracker = PersonTracker(cfg)
        for step in range(6):
            now = step * 0.1
            tracker.step(seen_at(5.0 * now, 0.0), now)
        vx, vy = tracker.confirmed_tracks()[0].velocity
        assert math.hypot(vx, vy) <= 2.5 + 1e-6


class TestAssociation:
    def test_two_people_keep_separate_tracks(self):
        tracker = PersonTracker(CONFIG)
        for step in range(4):
            now = step * 0.1
            tracker.step(
                [Measurement(2.0, -1.0, True), Measurement(2.0, 1.5, True)], now
            )
        positions = sorted(t.position[1] for t in tracker.confirmed_tracks())
        assert len(positions) == 2
        assert positions[0] == pytest.approx(-1.0, abs=0.3)
        assert positions[1] == pytest.approx(1.5, abs=0.3)

    def test_a_measurement_beyond_the_gate_starts_a_new_track(self):
        tracker = PersonTracker(CONFIG)
        tracker.step(seen_at(2.0, 0.0), 0.0)
        tracker.step(seen_at(2.0, 0.0), 0.1)
        tracker.step(seen_at(2.0, 6.0), 0.2)
        assert tracker.track_count == 2


class TestCombineMeasurements:
    def test_a_lidar_point_on_a_camera_point_is_the_same_person(self):
        merged = combine_measurements([(2.0, 0.0)], [(2.1, 0.05)], 0.9)
        assert len(merged) == 1
        assert merged[0].corroborated

    def test_a_lidar_point_on_its_own_survives_uncorroborated(self):
        merged = combine_measurements([(2.0, 0.0)], [(2.0, 3.0)], 0.9)
        assert len(merged) == 2
        assert [m.corroborated for m in merged] == [True, False]

    def test_lidar_only_round_is_all_uncorroborated(self):
        merged = combine_measurements([], [(2.0, 0.0), (3.0, 1.0)], 0.9)
        assert len(merged) == 2
        assert not any(m.corroborated for m in merged)


class TestReset:
    def test_reset_forgets_everything(self):
        tracker = PersonTracker(CONFIG)
        tracker.step(seen_at(2.0, 0.0), 0.0)
        tracker.step(seen_at(2.0, 0.0), 0.1)
        tracker.reset()
        assert tracker.track_count == 0
        # And the clock restarts, so the first step after a reset has dt 0.
        assert confirmed(tracker, seen_at(2.0, 0.0), 99.0) == []
