"""
Webcam exposure, set per application rather than left to whoever ran last.

A UVC webcam (the C270) keeps its exposure controls until it loses power, and
neither OpenCV nor GStreamer's v4l2src resets them on open. So a node that sets
a short manual exposure changes the camera for every node opened after it,
until the camera is replugged or the robot rebooted. Every node that opens the
camera directly therefore says what it wants, through three parameters:

    exposure_mode      "auto"   -- the camera's auto exposure (the default for
                                   the detectors: dim rooms stay visible)
                       "manual" -- a fixed exposure_ms (sharp during turns, but
                                   dark anywhere dimmer than a lit office)
                       "leave"  -- touch nothing; keep whatever the camera has
    exposure_ms        the manual exposure [ms]; ignored unless "manual"
    exposure_keep_fps  True: auto exposure may not lengthen a frame past the
                       frame period (exposure_auto_priority = 0), so the frame
                       rate holds in dim light at the price of darker frames.
                       False: it may, which is the camera's own default.

No ROS here: the node passes its parameter values in and logs the result.
"""

import fcntl
import os
import struct

MODES = ("auto", "manual", "leave")

# V4L2 control ids (linux/v4l2-controls.h).
V4L2_EXPOSURE_AUTO = 0x009A0901           # menu: 1 = manual, 3 = aperture priority (auto)
V4L2_EXPOSURE_ABSOLUTE = 0x009A0902       # units of 100 us
V4L2_EXPOSURE_AUTO_PRIORITY = 0x009A0903  # 0 = keep the frame rate, 1 = may drop it
VIDIOC_S_CTRL = 0xC008561C

EXPOSURE_MANUAL = 1
EXPOSURE_AUTO = 3


def device_path(device):
    """Return the /dev path for a device given as a path or an index."""
    text = str(device)
    return text if text.startswith("/dev/") else f"/dev/video{int(text)}"


def plan(mode, exposure_ms=10.0, keep_fps=False):
    """Return the (control id, value) pairs a mode sets, in the order to set them."""
    if mode not in MODES:
        raise ValueError(f"exposure_mode must be one of {MODES}, not {mode!r}")
    if mode == "leave":
        return []
    priority = (V4L2_EXPOSURE_AUTO_PRIORITY, 0 if keep_fps else 1)
    if mode == "auto":
        return [(V4L2_EXPOSURE_AUTO, EXPOSURE_AUTO), priority]
    units = max(1, int(round(float(exposure_ms) * 10.0)))
    # Priority first, then manual mode, then the time: a manual exposure is
    # only accepted once auto exposure is off.
    return [
        (V4L2_EXPOSURE_AUTO_PRIORITY, 0),
        (V4L2_EXPOSURE_AUTO, EXPOSURE_MANUAL),
        (V4L2_EXPOSURE_ABSOLUTE, units),
    ]


def describe(mode, exposure_ms=10.0, keep_fps=False):
    """Return a one-line description of what a mode asks the camera for."""
    if mode == "leave":
        return "exposure left as the camera has it"
    if mode == "manual":
        return f"manual exposure {float(exposure_ms):g} ms, frame rate kept"
    return "auto exposure" + (", frame rate kept" if keep_fps else "")


def apply(device, mode, exposure_ms=10.0, keep_fps=False):
    """Set the exposure controls for a mode; return the controls that failed."""
    controls = plan(mode, exposure_ms, keep_fps)
    if not controls:
        return []
    failed = []
    try:
        fd = os.open(device_path(device), os.O_RDWR)
    except OSError:
        return [cid for cid, _ in controls]
    try:
        for cid, value in controls:
            try:
                fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", cid, int(value))))
            except OSError:
                failed.append(cid)
    finally:
        os.close(fd)
    return failed


def apply_and_log(logger, device, mode, exposure_ms=10.0, keep_fps=False):
    """Apply a mode and log what was set, warning on any control that failed."""
    failed = apply(device, mode, exposure_ms, keep_fps)
    text = describe(mode, exposure_ms, keep_fps)
    if failed:
        logger.warn(
            f"camera {device}: could not set {text} "
            f"(controls {', '.join(hex(c) for c in failed)} refused)"
        )
    else:
        logger.info(f"camera {device}: {text}")
