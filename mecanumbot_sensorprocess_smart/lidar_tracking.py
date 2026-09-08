#!/usr/bin/env python3
"""Scan-frame tracking of DR-SPAAM person detections.

The detector reports a set of points per inference and nothing else: no
identity, no history, and no way to tell a person from the two seconds of noise
that look like one. This module is the layer that turns those points into
people - a constant-velocity Kalman estimate each, associated between frames by
nearest neighbour, aged in seconds rather than in frames so that capping the
inference rate to save the Orin's GPU does not also change what counts as a
detection.

Two guards decide what is published:

* ``min_hits`` - a point has to be seen more than once. This is what keeps
  single-frame noise out.
* ``require_motion`` - a track has to have been observed moving at least once.
  A 2D LiDAR at ankle height sees table legs, bin corners and door frames, and
  DR-SPAAM will happily call some of them people; almost none of them move.
  The flag is sticky, so a person who walks in and then stands still stays a
  person.

Re-seeding
----------

Motion being *sticky per track* was the problem this module was split out to
fix. Close to the robot a person's two legs subtend a wide angle, and the
detector resolves them as one blob, then two, then one again; each flicker that
outlasts ``max_missed_time`` drops the track, and the replacement starts over
with ``has_moved`` False. A person standing half a metre away - which is where
the camera is blind and the LiDAR is the only sensor left - could therefore
stop being reported entirely, because the fresh track had no motion of its own
to show and the person was no longer providing any.

So an expiring track leaves a **ghost**: its last position and whether it had
been seen moving, kept for ``reseed_memory`` seconds. A new track starting
within ``reseed_distance`` of a ghost inherits its motion evidence. It does not
inherit ``hits``, so it still has to be seen ``min_hits`` times before it is
published - the re-seed restores what was already established about this
person, and proves the rest again.

No ROS and no torch: this is plain numpy, filterpy and scipy, so it can be
tested on a development machine where neither ``dr_spaam`` nor CUDA exists.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Track:
    """Represents a single tracked person."""

    def __init__(self, detection, track_id, has_moved=False):
        """Start a track at `detection`, optionally inheriting motion evidence."""
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([detection[0], detection[1], 0.0, 0.0]).reshape(4, 1)

        # F is rebuilt on every predict() from the measured time step, because the
        # network no longer runs once per scan: the interval between two tracker
        # updates depends on the inference rate cap, not on the LiDAR rate.
        self.kf.F = np.eye(4)

        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.kf.P *= 10.0
        self.kf.R *= 0.5
        self.kf.Q *= 0.01

        self.time_since_update = 0.0
        self.hits = 1
        # Inherited from a ghost when this track replaces one that was dropped
        # a moment ago a few centimetres away; see the module docstring.
        self.has_moved = bool(has_moved)
        self.speed_thresh = 0.1

    @property
    def position(self):
        """Return the estimated ``(x, y)`` as a length-2 array."""
        return self.kf.x[:2].reshape(-1)

    def predict(self, dt):
        """Advance the motion model by `dt` seconds and return the prediction."""
        self.kf.F[0, 2] = dt
        self.kf.F[1, 3] = dt
        self.kf.predict()
        self.time_since_update += dt
        return self.position

    def update(self, detection):
        """Fold one detection into the estimate."""
        self.kf.update(detection.reshape(2, 1))
        self.time_since_update = 0.0
        self.hits += 1

        vx = self.kf.x[2, 0]
        vy = self.kf.x[3, 0]
        speed = np.hypot(vx, vy)

        if speed > self.speed_thresh:
            self.has_moved = True


class Ghost:
    """What a dropped track leaves behind for its replacement to inherit."""

    __slots__ = ("x", "y", "has_moved", "age")

    def __init__(self, position, has_moved):
        """Record where a track was and whether it had been seen moving."""
        self.x = float(position[0])
        self.y = float(position[1])
        self.has_moved = bool(has_moved)
        self.age = 0.0


class MultiObjectTracker:
    """Manages all active tracks and matches new detections.

    Ages tracks in seconds rather than in frames so that the behaviour does not
    change when the detector runs at a lower rate than the LiDAR.
    """

    def __init__(
        self,
        max_distance=0.5,
        max_missed_time=0.4,
        min_hits=2,
        require_motion=True,
        reseed_memory=1.5,
        reseed_distance=None,
    ):
        """Configure association, confirmation and the re-seed memory.

        `reseed_distance` defaults to `max_distance`: a replacement further off
        than the association gate is a different object, not the same one seen
        again.
        """
        self.max_distance = max_distance
        self.max_missed_time = max_missed_time
        self.min_hits = min_hits
        self.require_motion = require_motion
        self.reseed_memory = reseed_memory
        self.reseed_distance = (
            max_distance if reseed_distance is None else reseed_distance
        )
        self.tracks = []
        self.ghosts = []
        self.next_id = 0

    def _confirmed_positions(self):
        valid_positions = [
            t.position
            for t in self.tracks
            if t.hits >= self.min_hits and (t.has_moved or not self.require_motion)
        ]
        return (
            np.array(valid_positions) if len(valid_positions) > 0 else np.empty((0, 2))
        )

    def _inherited_motion(self, detection):
        """Say whether a track starting here replaces one that had moved."""
        if not self.ghosts:
            return False
        for ghost in self.ghosts:
            if not ghost.has_moved:
                continue
            if np.hypot(detection[0] - ghost.x, detection[1] - ghost.y) <= (
                self.reseed_distance
            ):
                return True
        return False

    def _spawn(self, detection):
        """Add a track for an unmatched detection, re-seeding it if it can be."""
        self.tracks.append(
            Track(detection, self.next_id, self._inherited_motion(detection))
        )
        self.next_id += 1

    def _expire(self, dt):
        """Drop the tracks that ran out of time, leaving each one a ghost."""
        alive = []
        for track in self.tracks:
            if track.time_since_update <= self.max_missed_time:
                alive.append(track)
            else:
                self.ghosts.append(Ghost(track.position, track.has_moved))
        self.tracks = alive

        for ghost in self.ghosts:
            ghost.age += dt
        self.ghosts = [g for g in self.ghosts if g.age <= self.reseed_memory]

    def predict_only(self, dt):
        """Advance the motion model without a measurement.

        Used on scans where the detector was skipped, so that the published
        detections keep moving at LiDAR rate instead of freezing between
        inferences.
        """
        for track in self.tracks:
            track.predict(dt)
        self._expire(dt)
        return self._confirmed_positions()

    def update(self, detections, dt):
        """Fold one inference's detections in and return the confirmed people."""
        if len(self.tracks) == 0:
            predicted_positions = np.empty((0, 2))
        else:
            predicted_positions = np.array([track.predict(dt) for track in self.tracks])

        matched_indices = []
        unmatched_detections = list(range(len(detections)))

        if len(self.tracks) > 0 and len(detections) > 0:
            cost_matrix = np.linalg.norm(
                predicted_positions[:, None, :] - detections[None, :, :], axis=2
            )
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(track_indices, det_indices):
                if cost_matrix[t_idx, d_idx] < self.max_distance:
                    matched_indices.append((t_idx, d_idx))
                    unmatched_detections.remove(d_idx)

        for t_idx, d_idx in matched_indices:
            self.tracks[t_idx].update(detections[d_idx])

        for d_idx in unmatched_detections:
            self._spawn(detections[d_idx])

        self._expire(dt)

        return self._confirmed_positions()
