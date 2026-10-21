#!/usr/bin/env python3
"""
The things the LiDAR sees, and which of them might be a person.

The scan was used in exactly one way until 2026-10-02: a camera detection gave a
bearing wedge, and the 20th percentile of the ranges inside it was the person's
range. The bags of 2026-09-23 .. 2026-09-30 show three ways that goes wrong, and
between them they are why a person the camera had plainly found was so often
never tracked:

* **the percentile is not a return.** The scan has about 210 beams, 1.7 degrees
  apart, so a person's legs at 2 m are two of the seven beams in their wedge and
  the other five are the wall behind them. The 20th percentile of that is the
  wall, or -- interpolated between the two -- a point in mid-air between the
  legs and the wall that nothing occupies. Either way the fused position jumped
  by two metres from one frame to the next and the tracker, which associates
  within 0.9 m, never saw the same person twice;
* **the wedge is in the wrong place.** The camera's bearings sit about 7.5
  degrees anticlockwise of where the LiDAR sees the same legs (266 detections
  with the robot standing still, over five bags and three days; the per-bag
  medians run from -5.8 to -10.2 degrees). At 2 m that is 26 cm, which is the
  whole width of a pair of legs: half the time they were outside the wedge that
  was meant to find them. `camera_bearing_offset` in the fusion node is that
  number;
* **a LiDAR that only confirms cannot find anybody.** Nothing the scan saw could
  say "something is standing there" unless DR-SPAAM said it first, and on this
  sensor it almost never does (45 non-empty `dets` messages out of 5502 in the
  long run of 2026-09-30).

So the scan is first cut into **clusters** -- runs of neighbouring beams at
about the same range, i.e. one surface each -- and everything else is asked of
those. A cluster is an actual object with an actual position, so a camera wedge
is ranged by choosing the cluster it points at instead of by averaging beams
across several. And a cluster that is the size of a person and that the static
map does not explain is a **candidate**: not a person, but a place worth
pointing the camera at, which is what the trees' head now does with them.

No ROS here, only numpy and scipy: the node passes the scan's fields in.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

FULL_TURN = 2.0 * math.pi

# nav_msgs/OccupancyGrid: above this a cell is an obstacle, below zero it was
# never observed.
OCCUPIED_THRESHOLD = 50


def wrap(angle):
    """Bring an angle into ``-pi .. pi``."""
    return (angle + math.pi) % FULL_TURN - math.pi


@dataclass(frozen=True)
class ScanCluster:
    """
    One surface the scan saw: a run of neighbouring beams at similar range.

    Everything is in the scan's own frame. ``bearing_min`` and ``bearing_max``
    are the outer edges of its first and last beam, unwrapped around
    ``bearing`` so that ``bearing_min <= bearing <= bearing_max`` holds even for
    a cluster lying across the scan's seam. ``width`` is the extent of its
    returns plus one beam's footprint, so a single return still has the width
    of the beam that made it.
    """

    x: float
    y: float
    range: float
    bearing: float
    bearing_min: float
    bearing_max: float
    width: float
    beams: int
    points: tuple = ()

    def overlaps(self, low, high, margin=0.0):
        """Say whether this cluster lies inside a bearing wedge, at all."""
        centre = (low + high) / 2.0
        half = abs(high - low) / 2.0 + margin
        offset = wrap(self.bearing - centre)
        extent = (self.bearing_max - self.bearing_min) / 2.0
        return abs(offset) <= half + extent

    def offset_from(self, bearing):
        """Return how far this cluster's centre is from a bearing, in radians."""
        return abs(wrap(self.bearing - bearing))


def segment_scan(
    ranges,
    angle_min,
    angle_increment,
    range_min=0.0,
    range_max=float("inf"),
    max_range=6.0,
    jump=0.2,
    max_gap=1,
):
    """
    Cut one scan into the surfaces it saw.

    Two neighbouring valid returns belong to the same cluster when their ranges
    differ by less than ``jump`` metres; up to ``max_gap`` invalid beams between
    them are bridged, because the LD08 drops single returns off dark trousers.
    Returns beyond ``max_range`` are ignored outright: at 1.7 degrees a beam a
    person's leg is narrower than the gap between beams from about 4 m.

    The scan is a full turn, so a cluster may straddle the seam between the
    last beam and the first; those two ends are joined.
    """
    ranges = np.asarray(ranges, dtype=float)
    count = ranges.size
    if count == 0 or angle_increment == 0.0:
        return []

    limit = min(float(range_max), float(max_range))
    valid = np.isfinite(ranges) & (ranges > max(float(range_min), 0.0)) & (ranges < limit)

    runs = []
    current = []
    for index in range(count):
        if not valid[index]:
            continue
        if (
            current
            and index - current[-1] <= max_gap + 1
            and abs(ranges[index] - ranges[current[-1]]) < jump
        ):
            current.append(index)
        else:
            if current:
                runs.append(current)
            current = [index]
    if current:
        runs.append(current)

    covers_full_turn = abs(count * angle_increment) >= FULL_TURN - 2.0 * abs(
        angle_increment
    )
    if (
        covers_full_turn
        and len(runs) > 1
        and runs[0][0] + (count - 1 - runs[-1][-1]) <= max_gap
        and abs(ranges[runs[0][0]] - ranges[runs[-1][-1]]) < jump
    ):
        runs[0] = runs.pop() + runs[0]

    return [_cluster(ranges, angle_min, angle_increment, run) for run in runs]


def _cluster(ranges, angle_min, angle_increment, run):
    """Build the :class:`ScanCluster` for one run of beam indices."""
    indices = np.asarray(run)
    reach = ranges[indices]
    angles = np.unwrap(angle_min + angle_increment * indices)
    xs = reach * np.cos(angles)
    ys = reach * np.sin(angles)
    x = float(xs.mean())
    y = float(ys.mean())
    distance = float(np.median(reach))
    bearing = math.atan2(y, x)

    half_beam = abs(angle_increment) / 2.0
    # The beam angles, expressed around the centroid's bearing so the extent is
    # a plain interval whatever side of the seam the cluster is on.
    relative = (angles - bearing + math.pi) % FULL_TURN - math.pi
    # The diagonal of the returns' bounding box rather than the chord between
    # the first and the last: a wall that runs round a corner, or all the way
    # round the robot, ends close to where it began and is not thereby small.
    extent = math.hypot(float(np.ptp(xs)), float(np.ptp(ys)))
    return ScanCluster(
        x=x,
        y=y,
        range=distance,
        bearing=bearing,
        bearing_min=bearing + float(relative.min()) - half_beam,
        bearing_max=bearing + float(relative.max()) + half_beam,
        width=extent + distance * abs(angle_increment),
        beams=int(indices.size),
        points=tuple(zip(xs.tolist(), ys.tolist())),
    )


class StaticMap:
    """
    The occupancy grid, asked the two things the fusion needs of it.

    *Is this return part of the room?* -- ``explained`` -- with the obstacles
    grown by ``inflation`` metres, because AMCL is good to a few centimetres and
    a wall return that lands one cell off the wall is still the wall.

    *What is this cell?* -- ``state`` -- on the grid as it was mapped.

    Unknown and out-of-map space counts as explained. A person can stand in a
    doorway the map never saw the far side of, but so does every beam that
    leaves the room through it, and those are not people.
    """

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"

    def __init__(self, data, resolution, origin_x, origin_y, inflation=0.2):
        """Take the grid as a ``(height, width)`` array of occupancy values."""
        self.data = np.asarray(data)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.height, self.width = self.data.shape
        occupied = self.data > OCCUPIED_THRESHOLD
        cells = int(round(float(inflation) / self.resolution)) if inflation > 0 else 0
        if cells > 0:
            reach = np.arange(-cells, cells + 1)
            disc = reach[:, None] ** 2 + reach[None, :] ** 2 <= cells**2
            occupied = binary_dilation(occupied, structure=disc)
        # Explained: a grown obstacle, or somewhere the map knows nothing about.
        self._explained = occupied | (self.data < 0)

    def _cells(self, xs, ys):
        columns = np.floor((np.asarray(xs, dtype=float) - self.origin_x) / self.resolution)
        rows = np.floor((np.asarray(ys, dtype=float) - self.origin_y) / self.resolution)
        columns = columns.astype(int)
        rows = rows.astype(int)
        inside = (
            (columns >= 0) & (columns < self.width) & (rows >= 0) & (rows < self.height)
        )
        return rows, columns, inside

    def state(self, x, y):
        """Return ``FREE``, ``OCCUPIED`` or ``UNKNOWN`` for one map point."""
        rows, columns, inside = self._cells([x], [y])
        if not inside[0]:
            return self.UNKNOWN
        value = int(self.data[rows[0], columns[0]])
        if value < 0:
            return self.UNKNOWN
        return self.OCCUPIED if value > OCCUPIED_THRESHOLD else self.FREE

    def explained(self, xs, ys):
        """Return, per map point, whether the static map accounts for it."""
        rows, columns, inside = self._cells(xs, ys)
        result = np.ones(inside.shape, dtype=bool)
        result[inside] = self._explained[rows[inside], columns[inside]]
        return result

    def explained_fraction(self, xs, ys):
        """Return the share of a set of map points the static map accounts for."""
        flags = self.explained(xs, ys)
        return float(flags.mean()) if flags.size else 1.0


def to_map(points, pose):
    """Scan-frame ``(x, y)`` points through the planar pose ``(x, y, yaw)``."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    tx, ty, yaw = pose
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return (
        tx + cos_yaw * points[:, 0] - sin_yaw * points[:, 1],
        ty + sin_yaw * points[:, 0] + cos_yaw * points[:, 1],
    )


@dataclass(frozen=True)
class CandidateConfig:
    """What makes a cluster worth pointing the camera at."""

    # Metres. A leg is about 0.12 m across and a person on a bean bag about a
    # metre; a table leg is under the first and a wall is over the second.
    min_width: float = 0.10
    max_width: float = 1.2
    # A cluster with more than this share of its returns on the static map is
    # the room, not something standing in it.
    max_explained: float = 0.5
    # Beyond this a leg is one beam or none.
    max_range: float = 4.5


def is_person_sized(cluster, cfg):
    """Say whether a cluster could be a person, by its size and range alone."""
    return cfg.min_width <= cluster.width <= cfg.max_width and (
        cluster.range <= cfg.max_range
    )


def is_explained(cluster, pose, static_map, cfg):
    """Say whether the static map accounts for a cluster (False with no map)."""
    if static_map is None or pose is None:
        return False
    xs, ys = to_map(cluster.points, pose)
    return static_map.explained_fraction(xs, ys) > cfg.max_explained


def candidates(clusters, pose, static_map, cfg=None):
    """
    Return the clusters that are person-sized and not part of the room.

    ``pose`` is the scan frame's ``(x, y, yaw)`` in the map. With no map or no
    pose nothing can be explained, and every person-sized cluster is returned:
    more to look at, never fewer.
    """
    cfg = cfg or CandidateConfig()
    return [
        cluster
        for cluster in clusters
        if is_person_sized(cluster, cfg)
        and not is_explained(cluster, pose, static_map, cfg)
    ]


def choose_for_wedge(candidate_clusters, low, high, margin=0.0):
    """
    Pick the candidate a camera bearing wedge is pointing at, or None.

    Of the candidates the wedge touches, the one whose centre is nearest the
    middle of it: a wedge is drawn round one person, and the table leg that
    happens to clip its edge is not who the camera was looking at.
    """
    centre = (low + high) / 2.0
    touching = [c for c in candidate_clusters if c.overlaps(low, high, margin)]
    if not touching:
        return None
    return min(touching, key=lambda c: (c.offset_from(centre), c.range))


def corroborates_close_range(candidate_clusters, low, high, margin=0.0, max_range=3.0):
    """
    Say whether the LiDAR backs up a camera detection that fills the frame.

    A ``close_range`` box claims a body so near that it runs off the top of the
    picture, and that is a claim the scan can check: somewhere in the wedge
    there has to be a candidate -- person-sized, not the mapped room -- within
    ``max_range``. A shelf of plants or a chair back that the pose network
    turned into a frame-filling skeleton has none (the 2026-10-21 leading run),
    and a person close enough to fill the frame always does, legs and all.
    The candidate checked is the one :func:`choose_for_wedge` would range the
    detection by, so a near table leg at the wedge's edge does not vouch for
    whatever is in the middle of it.
    """
    cluster = choose_for_wedge(candidate_clusters, low, high, margin)
    return cluster is not None and cluster.range <= max_range


def structure_range(ranges, quantile=0.2):
    """
    Return a near return among ``ranges`` -- an actual one, never a blend.

    The fallback for a wedge no candidate lies in: whatever the camera is
    looking at is against, on, or behind something the map already knows. The
    old lookup took ``np.percentile(ranges, 20)``, which interpolates, and with
    a handful of beams split between a near surface and a far one that is a
    range at which there is nothing. This takes the return at that rank instead.
    """
    ranges = np.sort(np.asarray(ranges, dtype=float))
    if ranges.size == 0:
        return None
    return float(ranges[int(math.floor(quantile * (ranges.size - 1)))])


@dataclass(frozen=True)
class OcclusionConfig:
    """How a camera detection ranged onto mapped structure is placed."""

    # Metres behind an occluder the person is put: the camera looks over the
    # blue panels, the LiDAR does not see through them.
    behind_buffer: float = 0.5
    # Metres in front of structure with nowhere to stand behind it.
    front_standoff: float = 0.3
    # The thickest thing that still counts as an occluder with a far side.
    max_occluder_depth: float = 0.6


def place_on_structure(static_map, robot_xy, hit_xy, cfg=None):
    """
    Place a person whose bearing ranged onto something the map knows.

    The camera saw a person and the scan, in that direction, saw the room. Two
    things can be true. The person is **behind** it -- the camera sits high
    enough to look over a partition the LiDAR cannot see through -- or the
    person is **against** it: sitting on a chair by the wall, where their legs
    and the wall are one return.

    The old rule assumed the first, walked along the ray until the map was free
    again and put the person half a metre further on. Against a room's outer
    wall there is no free space on the far side, only unmapped space, which the
    walk did not stop at: it ran on until the ray left the map, and somebody
    sitting by the wall on 2026-09-30 was published at (3.2, 3.0), nearly two
    metres beyond that wall and off the map altogether. Here the far side has
    to exist: free, mapped floor within
    ``max_occluder_depth`` of the hit, and still free where the person would be
    put. Otherwise they are in front, ``front_standoff`` short of the hit.

    Returns the ``(x, y)`` to use, in the map frame. A hit that is not on mapped
    structure at all is returned unchanged.
    """
    cfg = cfg or OcclusionConfig()
    hit_x, hit_y = float(hit_xy[0]), float(hit_xy[1])
    if static_map is None or not bool(static_map.explained([hit_x], [hit_y])[0]):
        return hit_x, hit_y

    dx = hit_x - float(robot_xy[0])
    dy = hit_y - float(robot_xy[1])
    reach = math.hypot(dx, dy)
    if reach < 1e-6:
        return hit_x, hit_y
    ux, uy = dx / reach, dy / reach

    step = static_map.resolution / 2.0
    travelled = 0.0
    entered = False
    while travelled <= cfg.max_occluder_depth:
        state = static_map.state(hit_x + ux * travelled, hit_y + uy * travelled)
        if state == StaticMap.OCCUPIED:
            entered = True
        elif state == StaticMap.FREE and entered:
            beyond = travelled + cfg.behind_buffer
            spot = (hit_x + ux * beyond, hit_y + uy * beyond)
            if static_map.state(*spot) == StaticMap.FREE:
                return spot
            break
        elif state == StaticMap.UNKNOWN and entered:
            break
        travelled += step

    back = min(cfg.front_standoff, reach)
    return hit_x - ux * back, hit_y - uy * back
