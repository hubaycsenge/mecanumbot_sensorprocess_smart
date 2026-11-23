#!/usr/bin/env python3
"""
Finding coloured bricks on the floor by their colour, and deciding which are real.

The object-choice task puts a few small cubes on the floor, and the person
points at one. They may each be a different colour, or all be the same plain
wood -- the classic paradigm's identical targets. Either way they are found by
colour; what makes one brick *that* brick is the colour when it is unique, and
where it lies when it is not (`max_per_colour`, and the object-choice tree's
`brick_identity.py`). No network on this robot can name
them -- the fetch detector's COCO classes have no "brick", and the pose
detector has one class -- so they are found the way a coloured object on a
plain floor has always been found: threshold the frame in HSV, take the
connected blobs, and keep the ones shaped like a brick.

Colour alone believes too much. A red shirt, a green chair cushion and a blue
bin are all bigger, better-lit blobs than a 6.5 cm cube. Two gates throw them
out, both cheap:

**Shape.** A cube seen from a camera 0.2 m off the floor is a squat rectangle:
its width is one face or two (s to s*sqrt(2)), its height one face plus a sliver
of the top. Its box is near square, and most of the box is the colour -- unlike
a strap, a chair leg or the outline of a poster. `min_aspect`/`max_aspect` and
`min_fill` say so.

**Being on the floor at the size it is.** This is the gate that does the real
work. The brick rests on the floor, so the ray through its box meets the floor
plane at a range (`ball_locating.range_from_ground`, solved for half a brick
up), and a brick of known size at that range has a known apparent width. A blob
far bigger than that is something large and near, or something above the floor
-- and a blob with no floor solution at all is above the horizon, which no brick
on the floor ever is. That one comparison removes nearly everything a colour
threshold lets through.

Both gates need the camera's tilt, which is the neck's, which is why this module
takes a `BallGeometry` per frame rather than owning one.

Pure numpy plus OpenCV's connected components -- no ROS -- so the thresholds and
the gates are testable on a synthetic frame.
"""

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from mecanumbot_sensorprocess_smart import ball_locating

# Edge of the bricks the object-choice task uses [m]: about a tennis ball's
# 0.067, so the grabbers that close on a ball close on one of these too.
BRICK_SIZE = 0.065

# OpenCV hue runs 0..179 (degrees halved); saturation and value 0..255. A range
# whose low hue is above its high hue wraps through 0 -- which red needs, since
# it sits on both ends of the hue circle. These are a starting point for a
# room's lighting, not a calibration of it: the debug image shows each mask, and
# the thresholds are parameters.
DEFAULT_COLOURS = {
    "red": (170, 8, 120, 255, 70, 255),
    "yellow": (20, 34, 110, 255, 90, 255),
    "green": (40, 85, 90, 255, 50, 255),
    "blue": (95, 130, 110, 255, 50, 255),
    # Unpainted light wood (beech, pine): an orange hue at LOW saturation, which
    # is what separates it from the four above -- and also what it shares with
    # skin, cardboard and a wooden floor. The shape and floor gates carry more
    # of the weight here than for a painted brick, and the range needs
    # calibrating in the room more than any other.
    "wood": (8, 28, 35, 170, 90, 255),
}

# Why a blob was refused. Carried on the rejection so the debug image and the
# log can say which check failed rather than only that one did.
REJECT_SMALL = "too small"
REJECT_ASPECT = "wrong shape"
REJECT_FILL = "too hollow"
REJECT_HORIZON = "above the horizon"
REJECT_SIZE = "wrong size for the floor"
REJECT_RANGE = "out of range"


@dataclass(frozen=True)
class ColourRange:
    """One brick colour: a box in HSV, with the hue allowed to wrap through 0."""

    name: str
    h_lo: int
    h_hi: int
    s_lo: int
    s_hi: int
    v_lo: int
    v_hi: int

    @classmethod
    def from_values(cls, name, values):
        """Build one from `[h_lo, h_hi, s_lo, s_hi, v_lo, v_hi]`, checking the bounds."""
        values = [int(value) for value in values]
        if len(values) != 6:
            raise ValueError(
                f"colour '{name}' needs six numbers (h_lo, h_hi, s_lo, s_hi, "
                f"v_lo, v_hi), got {len(values)}"
            )
        h_lo, h_hi, s_lo, s_hi, v_lo, v_hi = values
        for label, value, top in (
            ("h_lo", h_lo, 179),
            ("h_hi", h_hi, 179),
            ("s_lo", s_lo, 255),
            ("s_hi", s_hi, 255),
            ("v_lo", v_lo, 255),
            ("v_hi", v_hi, 255),
        ):
            if not 0 <= value <= top:
                raise ValueError(
                    f"colour '{name}': {label}={value} is outside 0..{top} "
                    "(OpenCV hue is 0..179, saturation and value 0..255)"
                )
        if s_lo > s_hi or v_lo > v_hi:
            raise ValueError(
                f"colour '{name}': a saturation or value range runs backwards; "
                "only the hue may wrap"
            )
        return cls(name, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi)

    @property
    def wraps(self):
        """Say whether the hue range runs through 0, as red's does."""
        return self.h_lo > self.h_hi

    def mask(self, hsv):
        """Return the pixels of this colour as a uint8 mask of 0 and 255."""
        hue = hsv[..., 0]
        saturation = hsv[..., 1]
        value = hsv[..., 2]
        if self.wraps:
            in_hue = (hue >= self.h_lo) | (hue <= self.h_hi)
        else:
            in_hue = (hue >= self.h_lo) & (hue <= self.h_hi)
        selected = (
            in_hue
            & (saturation >= self.s_lo)
            & (saturation <= self.s_hi)
            & (value >= self.v_lo)
            & (value <= self.v_hi)
        )
        return selected.astype(np.uint8) * 255


def colours_from_table(table):
    """Turn `{name: [six numbers]}` into `ColourRange`s, in the table's order."""
    return [ColourRange.from_values(name, values) for name, values in table.items()]


@dataclass(frozen=True)
class Blob:
    """One connected patch of one colour, as a box in image pixels."""

    colour: str
    u: float
    v: float
    width: float
    height: float
    area: float

    @property
    def box(self):
        """Return `(centre_u, centre_v, width, height)`, BoundingBox2D's shape."""
        return (self.u, self.v, self.width, self.height)

    @property
    def fill(self):
        """Fraction of the box the colour covers."""
        box_area = self.width * self.height
        return 0.0 if box_area <= 0.0 else float(self.area) / box_area

    @property
    def aspect(self):
        """Width over height."""
        return 0.0 if self.height <= 0.0 else float(self.width) / self.height

    def scaled(self, factor):
        """Return the same blob in a frame `factor` times larger."""
        factor = float(factor)
        return replace(
            self,
            u=self.u * factor,
            v=self.v * factor,
            width=self.width * factor,
            height=self.height * factor,
            area=self.area * factor * factor,
        )

    def mirrored(self, frame_width):
        """Return the same blob in the left-right reflection of the frame."""
        return replace(self, u=float(frame_width) - self.u)


def find_blobs(mask, colour, min_area=1.0, open_kernel=3):
    """
    Return the connected blobs of a mask, largest first.

    A small morphological opening first removes the speckle a colour threshold
    always leaves on a textured floor -- single pixels and hairlines that would
    otherwise each become a blob. `min_area` is in the mask's own pixels.
    """
    if open_kernel and open_kernel > 1:
        kernel = np.ones((int(open_kernel), int(open_kernel)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < min_area:
            continue
        blobs.append(
            Blob(
                colour=colour,
                u=x + width / 2.0,
                v=y + height / 2.0,
                width=float(width),
                height=float(height),
                area=float(area),
            )
        )
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    return blobs


@dataclass(frozen=True)
class BrickGate:
    """What a blob has to be to count as a brick on the floor."""

    # Full-frame pixels. A 6.5 cm brick at 4 m through the 51 deg lens at
    # 1280 px is about 21 px wide, so ~250 px of area; this lets a distant one
    # through and drops speckle.
    min_area: float = 80.0
    # Width over height. A cube is one or two faces wide and a face plus a
    # sliver of its top high, so near square either way.
    min_aspect: float = 0.5
    max_aspect: float = 2.2
    # How much of the box the colour fills. A cube's silhouette fills most of
    # its box; an outline, a strap or a lattice does not.
    min_fill: float = 0.45
    # Observed width over the width a brick would have at the range where its
    # ray meets the floor. 1.0 face-on, 1.41 corner-on; the margins absorb a
    # few degrees of tilt error and a mask that bleeds or erodes at the edges.
    min_size_ratio: float = 0.6
    max_size_ratio: float = 2.0


def shape_rejection(blob, gate):
    """Return why a blob's shape is not a brick's, or None if it could be one."""
    if blob.area < gate.min_area:
        return REJECT_SMALL
    if not gate.min_aspect <= blob.aspect <= gate.max_aspect:
        return REJECT_ASPECT
    if blob.fill < gate.min_fill:
        return REJECT_FILL
    return None


def floor_size_ratio(camera, box, geometry):
    """
    Return observed width over a floor brick's expected width, or None.

    None when the ray through the box never meets the floor -- the blob is at
    or above the horizon, and no brick on the floor is. `geometry.diameter` is
    the brick's edge, and the plane solved for is half of it up, the brick's
    centre.
    """
    centre_u, centre_v, width_px, _ = box
    unit, _, _ = ball_locating.direction(camera, centre_u, centre_v, geometry.camera_pitch)
    ground = ball_locating.range_from_ground(unit, geometry)
    if ground is None or ground <= 0.0:
        return None
    expected = camera.focal_x * geometry.diameter / ground
    if expected <= 0.0:
        return None
    return float(width_px) / expected


def floor_rejection(camera, blob, geometry, gate):
    """Return why a blob cannot be a brick on the floor, or None if it can."""
    ratio = floor_size_ratio(camera, blob.box, geometry)
    if ratio is None:
        return REJECT_HORIZON
    if not gate.min_size_ratio <= ratio <= gate.max_size_ratio:
        return REJECT_SIZE
    return None


@dataclass(frozen=True)
class BrickCandidate:
    """A blob that passed every gate, placed in the robot's base frame."""

    blob: Blob
    observation: object  # ball_locating.BallObservation

    @property
    def score(self):
        """How brick-like the blob is, 0..1: its fill, which the gate floors."""
        return max(0.0, min(1.0, self.blob.fill))


def detect_bricks(
    hsv,
    colours,
    gate,
    camera,
    geometry,
    scale=1.0,
    mirror=False,
    open_kernel=3,
    prefer=ball_locating.SOURCE_GROUND,
    max_per_colour=1,
):
    """
    Find at most `max_per_colour` bricks of each colour in an HSV frame.

    `hsv` may be a downscaled copy of the camera frame; `scale` is how many
    full-frame pixels one of its pixels is, and every box is reported in the
    full frame, which is what `camera` describes. With `mirror`, boxes are
    reflected left-right before anything is measured off them, for a camera
    that hands over a mirror image of the room.

    Returns `(accepted, rejected)`: `BrickCandidate`s, at most `max_per_colour`
    of each colour, largest first, and `(blob, reason)` pairs for the debug
    image. With the default of one, the colour is the brick's identity, so two
    red blobs that both pass are one brick and a doubt, and the larger is kept.
    With identical bricks, every one of them is wanted, and the cap only bounds
    what a badly set range can flood the tracker with.
    """
    limit = max(1, int(max_per_colour))
    accepted = []
    rejected = []
    min_area_scaled = gate.min_area / (float(scale) ** 2)
    for colour in colours:
        kept = 0
        blobs = find_blobs(
            colour.mask(hsv),
            colour.name,
            min_area=max(1.0, 0.25 * min_area_scaled),
            open_kernel=open_kernel,
        )
        for small in blobs:
            blob = small.scaled(scale)
            if mirror:
                blob = blob.mirrored(camera.width)
            reason = shape_rejection(blob, gate)
            if reason is None:
                reason = floor_rejection(camera, blob, geometry, gate)
            observation = None
            if reason is None:
                observation = ball_locating.locate(
                    camera, blob.box, geometry, score=blob.fill, prefer=prefer
                )
                if observation is None:
                    reason = REJECT_RANGE
            if reason is not None:
                rejected.append((blob, reason))
                continue
            # Largest first, so the ones that pass first are the ones kept.
            if kept < limit:
                accepted.append(BrickCandidate(blob, observation))
                kept += 1
            else:
                rejected.append((blob, f"one {colour.name} too many"))
    return accepted, rejected


def mean_hsv(hsv, box):
    """
    Return the median `(h, s, v)` inside a box, for setting a colour's range.

    The calibration helper: hold a brick in the middle of the frame and read
    what the camera thinks its colour is under this room's light. Median rather
    than mean so the box's edge, which is floor, does not drag it.
    """
    centre_u, centre_v, width, height = box
    rows, cols = hsv.shape[:2]
    top = max(0, int(centre_v - height / 2.0))
    bottom = min(rows, int(centre_v + height / 2.0))
    left = max(0, int(centre_u - width / 2.0))
    right = min(cols, int(centre_u + width / 2.0))
    if bottom <= top or right <= left:
        return None
    patch = hsv[top:bottom, left:right].reshape(-1, 3)
    return tuple(int(value) for value in np.median(patch, axis=0))


def expected_width_px(camera, size, distance):
    """Width in pixels of a brick `size` across at `distance` [m], face-on."""
    if distance <= 0.0:
        return 0.0
    return camera.focal_x * float(size) / float(distance)


def horizon_row(camera, pitch):
    """Return the image row of the horizon for a camera pitched `pitch` (up +)."""
    return camera.height / 2.0 + camera.focal_y * math.tan(float(pitch))
