#!/usr/bin/env python3
"""
Turning a tennis ball's bounding box into a place the robot can drive to.

A person is located by two sensors between them: the camera says *which
direction*, the LiDAR says *how far*. A ball on the floor cannot be located that
way. The LDS-02 scans one horizontal plane about 0.12 m up; a tennis ball is
0.067 m across and sits on the floor, so it is never in that plane. Looking a
range up inside the ball's bearing wedge would return the range of whatever is
*behind* the ball -- a wall, a chair, a person's shins -- and the robot would
drive confidently past the thing it was sent for.

So the range comes from the camera too, by the two routes a single camera has:

**Apparent size.** A tennis ball is a known 0.067 m across and is round, so its
box width is its diameter whatever direction it is seen from -- no other object
in this system has that property. With a pinhole model, ``range = f * D / d_px``.
This is the primary estimator. It degrades gracefully (a ball at 4 m is about 18
px across at 720p with a 60 degree lens, still measurable), it needs nothing but
the lens, and the *range* does not care where the camera is pointing. The
height does: it is that range along a ray the tilt points -- see `NeckMount`.

**Ground plane.** The ball is resting on the floor, so the ray through the
bottom of its box meets a known plane. This is the more accurate of the two
close in, and it is worthless near the horizon, where a small error in elevation
is a large error in range. It also depends on the camera's height and tilt being
right, which apparent size does not.

Both are computed whenever they can be, one is published, and a persistent
disagreement between them is logged once -- because that is what a wrong camera
mounting looks like, and there is otherwise no way to notice it: each estimator
on its own produces a plausible number.

## What comes out

A point in the robot's own frame with a real ``z``. The height is not decoration:
it is the difference between a ball on the floor (drive at it and close the
grabbers) and a ball in somebody's hand or on a table (the grabbers are a
horizontal pincer at z ~ 0.034 with no lift, so that ball is not obtainable).
The same argument `mecanumbot_seek` makes about the point cloud's z applies
here, one sensor cheaper.

## Tracking

An alpha-beta filter per ball, in the map frame. Deliberately lighter than
`person_tracking.py`'s Kalman: a ball is a small, cleanly-shaped, unambiguous
target with none of the association problems a crowd of people has, and it is
either where it was or being carried. What the filter is actually for is the
range noise -- an apparent-size range goes as 1/d_px, so one pixel of box jitter
at 4 m is about 20 cm -- and for carrying the estimate across the frames where
the ball is behind a chair leg.

Pure Python, no ROS and no numpy, so the geometry that decides where the robot
drives is testable on a machine with no robot attached.
"""

import math
from collections import deque
from dataclasses import dataclass, replace

# Diameter of a regulation tennis ball [m]. ITF says 6.54-6.86 cm.
TENNIS_BALL_DIAMETER = 0.067

# Which estimator produced a range. Carried on the observation so a log or a
# recording can say how a given position was arrived at.
SOURCE_SIZE = "size"
SOURCE_GROUND = "ground_plane"


@dataclass(frozen=True)
class CameraModel:
    """
    A pinhole camera, described by the two numbers a launch file knows.

    The vertical field of view is derived from the horizontal one and the frame
    shape on the assumption of square pixels, which is what every camera on this
    robot has; pass `vfov` to override that for one that does not.

    Note this is a **pinhole** model -- bearing is `atan`, not a linear ramp
    across the frame. `mecanumbot_onboard_cam_detect_people` uses the linear
    approximation for its person bearings, which is fine there because the
    bearing only has to pick a wedge for the LiDAR to search inside. Here the
    bearing is half of a position the robot drives to, and at the edge of a 60
    degree frame the two differ by about 1.5 degrees, which is 8 cm at a metre.
    """

    width: float
    height: float
    hfov: float
    vfov: float = 0.0

    @property
    def focal_x(self):
        """Focal length in pixels, horizontally."""
        return (self.width / 2.0) / math.tan(self.hfov / 2.0)

    @property
    def focal_y(self):
        """Focal length in pixels, vertically -- equal to `focal_x` if derived."""
        if self.vfov > 0.0:
            return (self.height / 2.0) / math.tan(self.vfov / 2.0)
        return self.focal_x

    @property
    def vertical_fov(self):
        """Return the vertical field of view, given or derived from the frame."""
        if self.vfov > 0.0:
            return self.vfov
        return 2.0 * math.atan((self.height / 2.0) / self.focal_x)

    def bearing(self, u):
        """Yaw of an image column, positive to the robot's **left**."""
        return math.atan2(self.width / 2.0 - float(u), self.focal_x)

    def elevation(self, v):
        """Pitch of an image row, positive **up**; image rows count downwards."""
        return math.atan2(self.height / 2.0 - float(v), self.focal_y)


@dataclass(frozen=True)
class BallGeometry:
    """
    Where the camera is and how big the ball is: everything the range needs.

    The mounting numbers are parameters rather than TF lookups on purpose. The
    URDF's `head_link` is rotated 90 degrees about x and `camera_link` another
    90 about y, so neither is an x-forward, z-up frame and neither gives the
    camera's height above the floor without unpicking two rotations that were
    written for the mesh rather than for optics. Two measured numbers are more
    honest than a derivation nobody can check by looking at the robot -- and
    they are logged at startup so a run says what it assumed.

    `camera_pitch` is positive when the camera looks **up**, and it is the tilt
    of *one frame*. Neither estimator escapes it: apparent size gives how far
    along the ray the ball is, but the ray's direction is the tilt, and so is
    the ball's height. The neck moves, so the node rebuilds this for every frame
    with `NeckMount.geometry`; the numbers here are the fallback for a frame
    with no neck reading.
    """

    diameter: float = TENNIS_BALL_DIAMETER
    # Camera position in the robot's base frame [m].
    camera_x: float = 0.13
    camera_z: float = 0.21
    camera_pitch: float = 0.0
    # Where the floor is in that same base frame. base_link sits 0.01 m above
    # base_footprint on this robot, so the floor is a centimetre below it.
    floor_z: float = -0.01
    # Ranges outside this band are not believed: nearer than the first the ball
    # is under the camera's chin, further than the second the box is a handful
    # of pixels and the 1/d_px error swamps it.
    min_range: float = 0.15
    max_range: float = 6.0


@dataclass(frozen=True)
class NeckMount:
    """
    Where the neck puts the camera, from the neck servo's position.

    The fetch tree sweeps the head while it searches, so one fixed pitch is
    wrong for nearly every frame -- and not by a little. A ball on the floor
    seen by a camera tilted down, and modelled as level, is placed along a ray
    that runs level: it comes out about as high as the camera, and the fetch
    tree decides it is not on the floor.

    The model is `mecanumbot_deep3r`'s `camera_pose.NeckCamera`, with the same
    defaults, and the two have to agree because they describe the same servo:
    a pivot fixed on the base, a lever to the lens that turns with the head,
    and a tilt linear in the servo's ticks::

        pitch = pitch_at_level + (ticks - level_ticks) * rad_per_tick
        lens  = pivot + R(pitch) @ lever

    Pivot and lever are the URDF's translations -- the part of it written for
    the robot rather than the meshes -- and at level they put the lens at
    (0.128, 0.206) m, the 0.13 / 0.21 `BallGeometry` defaults to.
    `level_ticks` is the trees' `neck_level_pos` (6.0 board units, scaled by
    100 on the way to the servo), `rad_per_tick` is the constant
    `mecanumbot_sensorproc_node` uses for the same servo, and
    **`pitch_at_level` is unmeasured**: nothing establishes that the neutral
    driving gaze is optically level. `floor_pitch` is how to measure it.

    `ticks` is the neck's *goal*. The firmware echoes the last command back as
    `OpenCRState.pos_n` and never reads the AX-12A's present position, so while
    the head is moving this is where it is going, not where it is.
    """

    pivot_x: float = 0.1063
    pivot_z: float = 0.1679
    lever_x: float = 0.022
    lever_z: float = 0.038
    level_ticks: float = 600.0
    rad_per_tick: float = 0.005061
    pitch_at_level: float = 0.0
    # Beyond this a reading is not a head position. The board reports 0 before
    # its first command, which is -174 degrees here, and that is the case this
    # is for; the neck's own 200..860 stays inside it.
    max_abs_pitch: float = math.radians(120.0)

    def pitch(self, ticks):
        """Return the camera's pitch for a neck at `ticks`, positive up."""
        return self.pitch_at_level + (float(ticks) - self.level_ticks) * self.rad_per_tick

    def geometry(self, base, ticks):
        """
        Return `base` with the camera moved to where the neck puts it, or None.

        None when the reading implies a pitch the head cannot have, so the
        caller can say so rather than place a ball with a camera that looks
        backwards.
        """
        pitch = self.pitch(ticks)
        if not math.isfinite(pitch) or abs(pitch) > self.max_abs_pitch:
            return None
        c, s = math.cos(pitch), math.sin(pitch)
        return replace(
            base,
            camera_x=self.pivot_x + c * self.lever_x - s * self.lever_z,
            camera_z=self.pivot_z + s * self.lever_x + c * self.lever_z,
            camera_pitch=pitch,
        )

    def describe(self):
        """Return one log line saying what this model assumes."""
        level = self.geometry(BallGeometry(), self.level_ticks)
        return (
            f"camera tilt from the neck: lens at ({level.camera_x:.3f}, "
            f"{level.camera_z:.3f}) m at {self.level_ticks:.0f} ticks, pitch "
            f"{math.degrees(self.pitch_at_level):+.1f} deg there, "
            f"{math.degrees(self.rad_per_tick):.3f} deg per tick "
            f"({math.degrees(self.pitch(200)):+.0f}..{math.degrees(self.pitch(860)):+.0f} "
            "deg over the neck's 200..860)"
        )


@dataclass(frozen=True)
class BallObservation:
    """One ball, placed in the robot's base frame, with its working shown."""

    x: float
    y: float
    z: float
    range: float
    bearing: float
    elevation: float
    score: float
    source: str
    size_range: float = 0.0
    ground_range: float = 0.0


def range_from_size(camera, width_px, height_px, diameter):
    """
    Range implied by how big the ball looks, or None if the box is degenerate.

    Both dimensions are used and the **smaller** of the two ranges wins, which
    is the same as saying the larger apparent diameter wins. A ball is round, so
    the two should agree; where they do not, it is because the box is clipped --
    by the frame edge, by a chair leg, by a hand -- and a clipped box is always
    too small, so it always over-estimates the range. Taking the larger
    dimension takes the less truncated view of the same ball.
    """
    ranges = []
    if width_px > 0.0:
        ranges.append(camera.focal_x * diameter / float(width_px))
    if height_px > 0.0:
        ranges.append(camera.focal_y * diameter / float(height_px))
    return min(ranges) if ranges else None


def direction(camera, u, v, pitch):
    """
    Return the unit vector towards an image point, in the robot's base frame.

    Forward, left, up -- the ROS convention -- with the camera's own tilt
    already applied, so the caller only has to add the mounting offset.
    """
    bearing = camera.bearing(u)
    elevation = camera.elevation(v)
    forward = math.cos(elevation) * math.cos(bearing)
    left = math.cos(elevation) * math.sin(bearing)
    up = math.sin(elevation)
    # Rotate about the left axis by the camera's tilt; positive pitch looks up.
    return (
        forward * math.cos(pitch) - up * math.sin(pitch),
        left,
        forward * math.sin(pitch) + up * math.cos(pitch),
    ), bearing, elevation


def range_from_ground(unit, geometry):
    """
    Range at which a ray meets the floor plane, ball-centre high, or None.

    The ball's centre sits one radius above the floor, so the plane solved for
    is `floor_z + radius` rather than the floor itself. Only a ray heading
    downwards from above that plane has a solution, which is the honest failure:
    a ball detected at or above the horizon is one this estimator has nothing to
    say about.
    """
    plane = geometry.floor_z + geometry.diameter / 2.0
    drop = geometry.camera_z - plane
    if drop <= 0.0 or unit[2] >= -1e-6:
        return None
    return drop / -unit[2]


def floor_pitch(camera, box, geometry):
    """
    Return the camera pitch that would put this ball on the floor, or None.

    The calibration `NeckMount.pitch_at_level` is waiting for. With a ball
    known to be on the floor, its apparent size says how far along the ray it
    is and the floor says how far below the lens, which together fix the ray's
    angle below horizontal; take away where the box sits in the frame and what
    is left is the camera's tilt. Read with the head at the level neck
    position, it is the number to set.

    The lens height is `geometry`'s as it stands. A few degrees of tilt moves
    the lens by millimetres, well inside what one pixel of box jitter does to
    the range.
    """
    centre_u, centre_v, width_px, height_px = box
    distance = range_from_size(camera, width_px, height_px, geometry.diameter)
    if distance is None:
        return None
    drop = (geometry.floor_z + geometry.diameter / 2.0) - geometry.camera_z
    # `direction` makes the ray's vertical part a*sin(pitch) + b*cos(pitch),
    # which is one arcsine once a and b are folded into a single angle.
    bearing = camera.bearing(centre_u)
    elevation = camera.elevation(centre_v)
    a = math.cos(elevation) * math.cos(bearing)
    b = math.sin(elevation)
    ratio = drop / (distance * math.hypot(a, b))
    if abs(ratio) > 1.0:
        return None
    return math.asin(ratio) - math.atan2(b, a)


def locate(camera, box, geometry, score=0.0, prefer=SOURCE_SIZE):
    """
    Place one ball box in the robot's base frame, or return None.

    `box` is `(centre_u, centre_v, width_px, height_px)` in image pixels, which
    is what `vision_msgs/BoundingBox2D` carries. `prefer` picks which estimator
    supplies the published range; the other is still computed and reported on
    the observation, so a caller can compare them.

    Returns None when neither estimator produced a range inside the geometry's
    band. That is a real answer -- a box too small to measure and too high to
    project is not evidence about where anything is -- and it is why the fusion
    node can publish nothing rather than publishing a guess.
    """
    centre_u, centre_v, width_px, height_px = box
    unit, bearing, elevation = direction(camera, centre_u, centre_v, geometry.camera_pitch)

    size_range = range_from_size(camera, width_px, height_px, geometry.diameter)
    ground_range = range_from_ground(unit, geometry)

    ordered = (
        (size_range, ground_range, SOURCE_SIZE, SOURCE_GROUND)
        if prefer == SOURCE_SIZE
        else (ground_range, size_range, SOURCE_GROUND, SOURCE_SIZE)
    )
    chosen, fallback, chosen_name, fallback_name = ordered

    source = chosen_name
    distance = chosen
    if not _in_band(distance, geometry):
        distance, source = fallback, fallback_name
    if not _in_band(distance, geometry):
        return None

    return BallObservation(
        x=geometry.camera_x + distance * unit[0],
        y=distance * unit[1],
        z=geometry.camera_z + distance * unit[2],
        range=distance,
        bearing=bearing,
        elevation=elevation,
        score=float(score),
        source=source,
        size_range=0.0 if size_range is None else size_range,
        ground_range=0.0 if ground_range is None else ground_range,
    )


def floor_height(observation, geometry):
    """How far the ball's centre is above the floor [m]."""
    return observation.z - geometry.floor_z


def graspable(observation, geometry, height_min, height_max):
    """
    Say whether the ball is in the height band the grabbers can close on.

    The pincer has no lift, so this is hardware and not taste: a ball on a
    table or in a hand is one the robot cannot have however close it gets. The
    fetch tree turns this into the difference between closing the grabbers and
    asking for the ball.
    """
    height = floor_height(observation, geometry)
    return float(height_min) <= height <= float(height_max)


def _in_band(distance, geometry):
    return (
        distance is not None
        and geometry.min_range <= distance <= geometry.max_range
    )


class NeckHistory:
    """
    The neck's recent positions, so a frame is placed with the tilt it had.

    The fetch tree moves the head every few tenths of a second while it
    searches, and a ball box arrives after the network has run on its frame,
    so the neck's *latest* position is often not the one the frame was taken
    with. This answers for a stamp instead: the newest reading at or before it.

    It refuses rather than guesses in two cases, the rule `mecanumbot_deep3r`'s
    `NeckTracker` follows. A stamp older than everything kept belongs to a
    frame whose reading is gone. And a newest reading more than `stale_s`
    behind the stamp is a board that has stopped publishing, which leaves a
    plausible last value behind -- the one not to believe.
    """

    def __init__(self, stale_s=0.5, depth=256):
        self.stale_s = float(stale_s)
        self._readings = deque(maxlen=int(depth))

    def submit(self, stamp, ticks):
        """Record that the neck was at `ticks` from `stamp` [s] on."""
        if self._readings and stamp < self._readings[-1][0]:
            # The clock stepped back -- a restarted sim, the Jetson correcting
            # its clock -- and nothing kept is comparable with what follows.
            self._readings.clear()
        self._readings.append((float(stamp), int(ticks)))

    def at(self, stamp):
        """Return the neck's ticks at `stamp` [s], or None."""
        for reading_stamp, ticks in reversed(self._readings):
            if reading_stamp <= stamp:
                if stamp - reading_stamp > self.stale_s:
                    return None
                return ticks
        return None


# --- tracking ---------------------------------------------------------------


@dataclass(frozen=True)
class BallTrackerConfig:
    """How the map-frame ball estimate is smoothed and how long it survives."""

    # Metres a measurement may sit from a track's prediction and still be the
    # same ball. Tighter than the person tracker's: balls do not stand in
    # crowds, and a jump this large is a second ball or a bad range.
    max_association_distance: float = 0.6
    # Measurements before a new ball is published, and how long one survives
    # with none. Short: a ball that has been picked up should stop being
    # reported quickly, or the robot will keep driving at the floor.
    min_hits: int = 2
    max_coast_time: float = 1.0
    # Alpha-beta gains. Alpha is how much of each measurement is believed;
    # beta how much of the disagreement is charged to velocity.
    position_gain: float = 0.5
    velocity_gain: float = 0.12
    # m/s. Above this is an association error, not a ball, so the velocity is
    # clamped rather than believed.
    max_speed: float = 3.0


@dataclass
class BallMeasurement:
    """One map-frame observation of a ball."""

    x: float
    y: float
    z: float
    score: float = 0.0


@dataclass
class BallTrack:
    """One ball being followed, in the map frame."""

    x: float
    y: float
    z: float
    score: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    last_update: float = 0.0
    confirmed: bool = False

    @property
    def position(self):
        """Return the estimate as `(x, y, z)`."""
        return (self.x, self.y, self.z)

    @property
    def speed(self):
        """How fast the ball is estimated to be moving [m/s]."""
        return math.hypot(self.vx, self.vy)


class BallTracker:
    """
    Alpha-beta tracking of balls in the map frame.

    Greedy nearest-neighbour association, one track per ball, ages measured in
    seconds so the behaviour does not change with the detector's frame rate --
    the same model `person_tracking.py` and `lidar_tracking.py` use, minus the
    corroboration machinery, which has nothing to corroborate here.
    """

    def __init__(self, config=None):
        self.config = config or BallTrackerConfig()
        self.tracks = []
        self._last_step = None

    def step(self, measurements, now):
        """Advance to `now`, fold in the measurements, return confirmed tracks."""
        dt = 0.0 if self._last_step is None else max(0.0, now - self._last_step)
        self._last_step = now

        for track in self.tracks:
            track.x += track.vx * dt
            track.y += track.vy * dt

        unmatched = list(measurements)
        for track in self.tracks:
            match = self._closest(track, unmatched)
            if match is None:
                continue
            unmatched.remove(match)
            self._correct(track, match, dt, now)

        for measurement in unmatched:
            track = BallTrack(
                x=measurement.x,
                y=measurement.y,
                z=measurement.z,
                score=measurement.score,
                last_update=now,
            )
            # A first sighting already counts as a hit, so `min_hits: 1` has to
            # mean "publish it now" rather than "publish it next frame".
            track.confirmed = track.hits >= self.config.min_hits
            self.tracks.append(track)

        self.tracks = [
            track
            for track in self.tracks
            if now - track.last_update <= self.config.max_coast_time
        ]
        return [track for track in self.tracks if track.confirmed]

    # --- internals -----------------------------------------------------------

    def _closest(self, track, measurements):
        best, best_distance = None, self.config.max_association_distance
        for measurement in measurements:
            distance = math.hypot(measurement.x - track.x, measurement.y - track.y)
            if distance <= best_distance:
                best, best_distance = measurement, distance
        return best

    def _correct(self, track, measurement, dt, now):
        alpha = self.config.position_gain
        beta = self.config.velocity_gain
        residual_x = measurement.x - track.x
        residual_y = measurement.y - track.y

        track.x += alpha * residual_x
        track.y += alpha * residual_y
        # Height is smoothed but never given a velocity: a ball's height changes
        # when somebody picks it up, which is a step and not a trend.
        track.z += alpha * (measurement.z - track.z)
        track.score = measurement.score

        if dt > 1e-6:
            track.vx += beta * residual_x / dt
            track.vy += beta * residual_y / dt
            speed = track.speed
            if speed > self.config.max_speed:
                scale = self.config.max_speed / speed
                track.vx *= scale
                track.vy *= scale

        track.hits += 1
        track.last_update = now
        if track.hits >= self.config.min_hits:
            track.confirmed = True
