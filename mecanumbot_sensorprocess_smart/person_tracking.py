#!/usr/bin/env python3
"""Map-frame tracking of fused person detections.

`mecanumbot_locate_detections` used to be a pure function of the current frame:
a camera bearing arrived, a LiDAR range was looked up inside it, and the
resulting point was published. Anything that stopped the camera producing a
detection for one frame - a gate rejection, a missed box, a person walking out
of the tilted-down field of view - stopped `people_fusion` too, and the
behaviour layer, which judges "is somebody there" by the age of the last
message, read that as the person having gone.

This module is the memory that was missing. Each person is a constant-velocity
Kalman filter over their map-frame position, so:

* a detection that skips a frame is **coasted** rather than lost - the estimate
  keeps moving at the velocity that was being observed, and only expires after
  `max_coast_time`;
* the range noise is **smoothed**. A LiDAR range taken through a camera bearing
  wedge jumps around by tens of centimetres as the wedge slides across a leg,
  a coat and the floor behind; the filter averages that over time instead of
  handing the raw jump to Nav2 as a new goal;
* a **velocity** falls out of it, which is what tells a leading behaviour
  whether the person it is walking is following or has stopped.

Why here and not on the camera's bounding boxes
-----------------------------------------------

A Kalman filter on the image-space box would smooth the *bearing* and keep a
box alive across a dropped frame. That is the wrong quantity and the wrong
place for two reasons. The bearing is not what the behaviour layer consumes -
it consumes a map-frame position, and the range, which the camera never
measures, is by far the noisier half of that. And the camera's own failure
mode is not jitter but rejection: an image-space filter would still leave
`merge_detections` with no bearing to look a range up inside, so nothing would
be published at all.

Filtering in the map frame instead makes the two sensors complementary rather
than serial. The camera says *that* a person is there and roughly where to
look; the LiDAR says *how far*; the filter carries the estimate over whichever
of the two is missing this frame. `mecanumbot_lidar_detect_people` already
tracks this way on its own detections, and this is deliberately the same
model - constant velocity, greedy nearest-neighbour association, ages measured
in seconds so the behaviour does not change with the detector's rate.

Corroboration
-------------

A track is only ever *created* by a camera-corroborated measurement, because
`dets` alone cannot tell a person from any other leg-sized thing the LiDAR
sees. Once created, it can be *updated* by an uncorroborated one, which is what
carries a person through a run of camera rejections - but only for
`max_uncorroborated_time`, after which the camera has to agree again or the
track is dropped. Coasting is memory, not belief: it must not turn a person who
left into a permanent phantom.

The blind zone
--------------

That rule has one exception, and it is the close-range case the leading
experiment runs into. Demanding corroboration only makes sense where the camera
could have supplied it. The camera sits on the head at about 0.22 m with a
~36 degree vertical field of view, so a person nearer than roughly a metre has
nothing in frame the pose network can call a body, and a person outside the
horizontal field of view is not in the picture at all. In neither case is the
camera's silence evidence that nobody is there - it is evidence of nothing.

So a track the camera *cannot* be expected to see does not accrue
uncorroborated time, and a consistent run of LiDAR-only measurements there may
also *create* one, which the ordinary rule forbids. What keeps that honest is
that the exemption is bounded on three sides: it applies only inside a
geometrically defined blind zone (:class:`CameraCoverage`), the LiDAR has to
keep measuring - ``max_coast_time`` is unchanged, so a track with no
measurement at all still dies in about a second - and it stops being reported
after ``max_blind_zone_time`` whatever the geometry says. A person who walks
away stops being detected and expires; a person standing half a metre from the
robot does not.

No ROS, no message types: everything here is plain floats and numpy, so it is
unit-testable on a development machine.
"""

import math
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Measurement:
    """One map-frame observation of a person.

    `corroborated` is whether the camera agreed a person is at this point.
    Uncorroborated measurements (a bare LiDAR detection) may update a track
    that already exists but may never start one.
    """

    x: float
    y: float
    corroborated: bool = True


@dataclass(frozen=True)
class CameraCoverage:
    """Where the camera is, and where it could see a person if one were there.

    Pose is the camera frame's position and yaw in the map, i.e. the head, not
    the base: the head turns, and a person behind the robot's body may be
    squarely in front of its camera.

    The two exemptions are separate because they fail for different reasons and
    an experiment may want only one of them:

    * ``exempt_close_range`` - nearer than ``blind_range`` the *vertical* field
      of view has run out. At 0.6 m a camera at 0.22 m looking through ~36
      degrees sees the world from the floor to about 0.4 m, which is a pair of
      calves; below that there is not enough of a person in frame for any gate
      to accept, however it is tuned.
    * ``exempt_outside_fov`` - beyond ``half_fov`` off the camera axis the
      person is not in the picture at all. This is the one that matters while
      the robot is leading, because the human it is leading spends most of the
      run behind it.

    Both are statements about the *sensor*, not about the person, which is what
    makes them safe to act on: they say the camera had nothing to contribute
    here, so its silence must not be counted as disagreement.
    """

    x: float
    y: float
    yaw: float
    blind_range: float = 0.9
    half_fov: float = 0.52
    exempt_close_range: bool = True
    exempt_outside_fov: bool = True

    def cannot_see(self, x, y):
        """Say whether a person at ``(x, y)`` would be invisible to the camera."""
        dx = float(x) - self.x
        dy = float(y) - self.y
        if self.exempt_close_range and math.hypot(dx, dy) <= self.blind_range:
            return True
        if self.exempt_outside_fov:
            bearing = math.atan2(dy, dx) - self.yaw
            bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
            if abs(bearing) > self.half_fov:
                return True
        return False


@dataclass(frozen=True)
class TrackerConfig:
    """Association, noise and expiry settings for :class:`PersonTracker`."""

    # How far a measurement may sit from a track's prediction and still be
    # taken as the same person, in metres. A person walks at ~1.4 m/s and the
    # pipeline runs at ~10 Hz, so the prediction should be well within this;
    # what it really has to absorb is the range noise of the wedge lookup.
    max_association_distance: float = 0.9

    # Corroborated measurements a new track needs before it is published. The
    # camera gate upstream already applies its own temporal confirmation, so
    # this is a second, cheap guard against a single bad range.
    min_hits: int = 2

    # How long a track survives on prediction alone, with no measurement of
    # any kind. Long enough to cross a run of dropped frames, short enough
    # that a person who walks out of the room stops being reported.
    max_coast_time: float = 1.2

    # How long a track survives on uncorroborated (LiDAR-only) measurements
    # after the camera last agreed. This is the budget for the failure the
    # screenshots show: the person is plainly there and the LiDAR sees them,
    # but the pose network cannot make a skeleton out of a long skirt.
    max_uncorroborated_time: float = 4.0

    # How long a track survives inside the camera's blind zone, where
    # `max_uncorroborated_time` is held rather than spent. This is the outer
    # bound on the exemption: past it the camera has to agree again whatever
    # the geometry says, so a mis-detected chair leg beside the robot cannot
    # become a permanent person. Generous, because the case it is sized for is
    # a human walking at the robot's shoulder for a whole leading trial.
    max_blind_zone_time: float = 30.0

    # Consecutive LiDAR-only measurements, taken where the camera could not
    # have seen anything, that confirm a track the camera never vouched for.
    # Deliberately higher than `min_hits`: this is the one path by which the
    # LiDAR alone may assert a person, so it has to be paid for in
    # consistency. At the 10 Hz publish rate five hits is half a second.
    blind_zone_min_hits: int = 5

    # Whether an uncorroborated measurement in the blind zone may *create* a
    # track, as opposed to only sustaining one. False keeps the original rule
    # -- only the camera introduces a person -- while still holding the
    # corroboration clock for tracks that already exist.
    blind_zone_creates_tracks: bool = True

    # Measurement noise, in metres. The range comes from a percentile over a
    # scan wedge, so it is biased and coarse rather than gaussian; 0.25 m is
    # the scale of the jump seen when the wedge slides off a leg.
    measurement_noise: float = 0.25

    # Process noise: how much the constant-velocity assumption is allowed to
    # be wrong. A person changing direction is the case this covers.
    process_noise: float = 0.08

    # Initial uncertainty on the first measurement, in metres. High, so the
    # second measurement moves the estimate most of the way.
    initial_position_variance: float = 1.0
    initial_velocity_variance: float = 4.0

    # Speeds above this are not a person walking, they are an association
    # error; the reported velocity is clamped so a consumer cannot be told the
    # subject is sprinting because two people swapped tracks for a frame.
    max_reported_speed: float = 2.5


class PersonTrack:
    """One person, as a constant-velocity estimate of where they are."""

    __slots__ = (
        "track_id",
        "kf",
        "hits",
        "corroborated_hits",
        "blind_hits",
        "time_since_update",
        "time_since_corroboration",
        "time_in_blind_zone",
        "_cfg",
    )

    def __init__(self, measurement, track_id, cfg):
        """Start a track at `measurement`, with no velocity assumed yet."""
        self._cfg = cfg
        self.track_id = track_id

        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([measurement.x, measurement.y, 0.0, 0.0]).reshape(4, 1)
        # F is rebuilt on every predict() from the measured time step: the
        # camera and the LiDAR do not tick together, so dt is not a constant.
        kf.F = np.eye(4)
        kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        kf.P = np.diag(
            [
                cfg.initial_position_variance,
                cfg.initial_position_variance,
                cfg.initial_velocity_variance,
                cfg.initial_velocity_variance,
            ]
        )
        kf.R = np.eye(2) * (cfg.measurement_noise**2)
        kf.Q = np.eye(4) * cfg.process_noise
        self.kf = kf

        self.hits = 1
        self.corroborated_hits = 1 if measurement.corroborated else 0
        # Uncorroborated hits taken where the camera could not have seen the
        # person. Reset by any measurement the camera *could* have vouched for
        # and did not, so it counts consistency in the blind zone rather than
        # accumulating over a whole run.
        self.blind_hits = 0 if measurement.corroborated else 1
        self.time_since_update = 0.0
        self.time_since_corroboration = 0.0
        self.time_in_blind_zone = 0.0

    @property
    def position(self):
        """Return the filtered ``(x, y)`` in the map frame."""
        return float(self.kf.x[0, 0]), float(self.kf.x[1, 0])

    @property
    def velocity(self):
        """Return the estimated ``(vx, vy)`` in m/s, clamped to a human speed."""
        vx = float(self.kf.x[2, 0])
        vy = float(self.kf.x[3, 0])
        speed = float(np.hypot(vx, vy))
        limit = self._cfg.max_reported_speed
        if speed > limit and speed > 0.0:
            scale = limit / speed
            return vx * scale, vy * scale
        return vx, vy

    @property
    def blind_exhausted(self):
        """Say whether this track has spent too long unable to be checked.

        The outer bound on the exemption. Note that it *silences* a track
        rather than deleting it: an expired one would be re-created by the very
        next LiDAR return in the same spot and confirmed all over again, which
        is no bound at all. Silenced, it goes on absorbing those returns - so
        no replacement is spawned - and ends when the LiDAR stops seeing
        anything there, or comes back the moment the camera can vouch for it.
        """
        return self.time_in_blind_zone > self._cfg.max_blind_zone_time

    @property
    def confirmed(self):
        """Say whether this track has enough evidence to be published.

        Either the camera vouched for it `min_hits` times, or the LiDAR did so
        `blind_zone_min_hits` times somewhere the camera could not have looked
        - and in neither case has the blind-zone budget run out.
        """
        if self.blind_exhausted:
            return False
        if self.corroborated_hits >= self._cfg.min_hits:
            return True
        return self.blind_hits >= self._cfg.blind_zone_min_hits

    def predict(self, dt, camera_blind=False):
        """Advance the motion model by `dt` seconds and return the prediction.

        `camera_blind` says the camera could not have seen this person over the
        interval just elapsed. The corroboration clock is then held instead of
        advanced - the camera is not disagreeing, it is absent - and the time
        spent that way is charged to `max_blind_zone_time` instead.
        """
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt
        self.kf.predict()
        self.time_since_update += dt
        if camera_blind:
            self.time_in_blind_zone += dt
        else:
            self.time_since_corroboration += dt
        return self.position

    def update(self, measurement, camera_blind=False):
        """Fold one measurement into the estimate.

        `camera_blind` is what separates the two kinds of uncorroborated
        measurement. One taken where the camera was looking is the camera
        declining to agree, and it clears the blind-zone evidence; one taken
        where the camera is blind is the only evidence available, and it counts.
        """
        self.kf.update(np.array([measurement.x, measurement.y]).reshape(2, 1))
        self.time_since_update = 0.0
        self.hits += 1
        if measurement.corroborated:
            self.corroborated_hits += 1
            self.time_since_corroboration = 0.0
            self.time_in_blind_zone = 0.0
        elif camera_blind:
            self.blind_hits += 1
        else:
            self.blind_hits = 0

    def expired(self, cfg):
        """Say whether this track has run out of coasting or of corroboration.

        `max_coast_time` is what stops the blind-zone exemption becoming a
        phantom: whatever the camera geometry says, a track that nothing is
        measuring at all still dies in about a second. The third budget,
        `max_blind_zone_time`, is not an expiry - see
        :attr:`blind_exhausted` for why it silences instead.
        """
        if self.time_since_update > cfg.max_coast_time:
            return True
        return self.time_since_corroboration > cfg.max_uncorroborated_time


class PersonTracker:
    """Every person currently being followed in the map frame.

    Driven by :meth:`step`, which is called both when detections arrive and on
    a timer, so the estimate keeps moving between detections instead of
    freezing at the last point that happened to be measured.
    """

    def __init__(self, cfg=None):
        """Start with no tracks; `cfg` defaults to a :class:`TrackerConfig`."""
        self._cfg = cfg or TrackerConfig()
        self._tracks = []
        self._next_id = 0
        self._last_step = None

    @property
    def config(self):
        """Return the :class:`TrackerConfig` in force."""
        return self._cfg

    @property
    def track_count(self):
        """Return how many tracks exist, confirmed or not."""
        return len(self._tracks)

    def _associate(self, measurements, predictions):
        """Match measurements to tracks by nearest neighbour, within the gate.

        Returns ``(matches, unmatched_measurements)`` where `matches` is a list
        of ``(track_index, measurement_index)``.
        """
        if not measurements or not self._tracks:
            return [], list(range(len(measurements)))

        points = np.array([[m.x, m.y] for m in measurements], dtype=float)
        cost = np.linalg.norm(predictions[:, None, :] - points[None, :, :], axis=2)
        track_indices, measurement_indices = linear_sum_assignment(cost)

        matches = []
        matched_measurements = set()
        for track_index, measurement_index in zip(track_indices, measurement_indices):
            if cost[track_index, measurement_index] <= self._cfg.max_association_distance:
                matches.append((track_index, measurement_index))
                matched_measurements.add(measurement_index)
        unmatched = [
            index
            for index in range(len(measurements))
            if index not in matched_measurements
        ]
        return matches, unmatched

    def step(self, measurements, now, camera=None):
        """Advance every track to `now` and fold in this round's measurements.

        Args:
            measurements: :class:`Measurement` list; may be empty, which is the
                coasting case and the whole reason this class exists.
            now: seconds on a monotonic clock. Only differences matter, but
                they must be differences on the *same* clock.
            camera: a :class:`CameraCoverage` for where the camera is pointing
                now, or None. None means "assume the camera could have seen
                everything", which is the original behaviour: every
                uncorroborated measurement then counts against the track.

        Returns:
            The confirmed tracks, as a list of :class:`PersonTrack`.
        """
        dt = 0.0 if self._last_step is None else max(0.0, now - self._last_step)
        self._last_step = now

        # Blind status is read off each track's *previous* estimate, before the
        # motion model moves it. The two differ by one step of a walking pace,
        # which is far inside the blind radius, and taking it beforehand keeps
        # the decision independent of the prediction it is about to justify.
        blind = [self._is_blind(camera, track.position) for track in self._tracks]

        predictions = np.array(
            [
                track.predict(dt, camera_blind)
                for track, camera_blind in zip(self._tracks, blind)
            ],
            dtype=float,
        ).reshape(-1, 2)

        matches, unmatched = self._associate(list(measurements), predictions)
        for track_index, measurement_index in matches:
            self._tracks[track_index].update(
                measurements[measurement_index], blind[track_index]
            )

        for index in unmatched:
            measurement = measurements[index]
            # An uncorroborated measurement that matched nothing is usually just
            # a leg-sized LiDAR return, and only the camera may assert that
            # something previously unseen is a person. The exception is a return
            # from inside the blind zone: there the camera was never going to
            # corroborate anything, so refusing to start a track means refusing
            # to see anyone standing close. Such a track still has to earn
            # `blind_zone_min_hits` before it is published.
            if not measurement.corroborated and not (
                self._cfg.blind_zone_creates_tracks
                and self._is_blind(camera, (measurement.x, measurement.y))
            ):
                continue
            self._tracks.append(PersonTrack(measurement, self._next_id, self._cfg))
            self._next_id += 1

        self._tracks = [
            track for track in self._tracks if not track.expired(self._cfg)
        ]
        return self.confirmed_tracks()

    @staticmethod
    def _is_blind(camera, position):
        """Say whether the camera could have seen a person at `position`."""
        if camera is None:
            return False
        return camera.cannot_see(position[0], position[1])

    def confirmed_tracks(self):
        """Return the tracks with enough camera evidence to be published."""
        return [track for track in self._tracks if track.confirmed]

    def reset(self):
        """Forget every track; used when localisation jumps and the map moves."""
        self._tracks = []
        self._last_step = None


def combine_measurements(camera_points, lidar_points, min_separation):
    """Merge camera-corroborated and LiDAR-only points into one measurement set.

    Both sensors see the same people, so a person standing in front of the
    robot usually produces one of each. Handing both to :meth:`PersonTracker.step`
    would have the assignment match one of them to the track and treat the
    other as a new person, so the LiDAR-only points that fall on top of a
    camera-corroborated one are dropped here: they are the same observation,
    and the corroborated one is the better evidence.

    What survives is the LiDAR detections that stand on their own - a person
    the camera has lost. Those are marked uncorroborated, which lets them keep
    an existing track alive without being able to start one.

    Args:
        camera_points: ``(x, y)`` map-frame points the camera vouches for.
        lidar_points: ``(x, y)`` map-frame points from the LiDAR alone.
        min_separation: metres below which a LiDAR point is taken to be the
            same person as a camera one. The tracker's association distance is
            the right value: closer than that and the tracker could not have
            told them apart anyway.

    Returns:
        A list of :class:`Measurement`.
    """
    measurements = [Measurement(float(x), float(y), True) for x, y in camera_points]
    for x, y in lidar_points:
        x = float(x)
        y = float(y)
        if any(
            np.hypot(x - m.x, y - m.y) <= min_separation
            for m in measurements
            if m.corroborated
        ):
            continue
        measurements.append(Measurement(x, y, False))
    return measurements
