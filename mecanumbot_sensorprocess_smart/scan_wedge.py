"""
The LiDAR returns inside a camera bearing wedge.

A camera detection is a bearing and no range, so the range is read off the scan
between the detection's two bearings. The bearings are signed, positive to the
left of the camera axis. The scan has its own angle convention, and on this
robot it is not the signed one: the LD08 driver publishes `angle_min` ~ 0 and
sweeps counterclockwise through a full turn, so everything to the right of the
robot sits at 3pi/2 .. 2pi, not at -pi/2 .. 0.

Until 2026-09-30 the lookup turned the two bearings into indices with
`(bearing - angle_min) / angle_increment` and clamped them into the array. A
bearing to the right of the axis came out negative and clamped to index 0 --
both ends of the wedge did, so the wedge was empty and the detection was
dropped, and a wedge straddling the axis kept only its left half. In the bags
of 2026-09-30 that was 228 of 456 camera detections of a person thrown away
outright and 72 more ranged on half a wedge; the person sitting on the bean bag
was one of them. The lookup here compares angles modulo a full turn, so it does
not care which convention the driver uses.

No ROS here: the node passes the scan's fields in.
"""

import math

import numpy as np

FULL_TURN = 2.0 * math.pi


def wedge_ranges(ranges, angle_min, angle_increment, bearing_a, bearing_b):
    """
    Return the scan ranges whose beams lie between two bearings.

    The wedge runs counterclockwise from the smaller bearing to the larger, and
    is compared modulo a full turn, so it can straddle the scan's seam and a
    signed bearing finds its beam whatever the scan's own `angle_min` is. The
    returned values are raw: filtering out inf, NaN and out-of-range returns is
    left to the caller, who knows the scan's limits.
    """
    ranges = np.asarray(ranges, dtype=float)
    if ranges.size == 0 or angle_increment == 0.0:
        return ranges[:0]
    low, high = min(bearing_a, bearing_b), max(bearing_a, bearing_b)
    width = high - low
    if width >= FULL_TURN:
        return ranges
    beams = angle_min + angle_increment * np.arange(ranges.size)
    offset = np.mod(beams - low, FULL_TURN)
    return ranges[offset <= width]
