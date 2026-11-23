"""Which V4L2 controls each exposure mode sets, in which order."""

import pytest

from mecanumbot_sensorprocess_smart import camera_exposure as ce


def test_auto_turns_auto_exposure_on_and_lets_it_lengthen_frames():
    assert ce.plan("auto") == [
        (ce.V4L2_EXPOSURE_AUTO, ce.EXPOSURE_AUTO),
        (ce.V4L2_EXPOSURE_AUTO_PRIORITY, 1),
    ]


def test_auto_keep_fps_holds_the_frame_rate():
    assert (ce.V4L2_EXPOSURE_AUTO_PRIORITY, 0) in ce.plan("auto", keep_fps=True)


def test_manual_sets_mode_before_time_in_100us_units():
    assert ce.plan("manual", exposure_ms=10.0) == [
        (ce.V4L2_EXPOSURE_AUTO_PRIORITY, 0),
        (ce.V4L2_EXPOSURE_AUTO, ce.EXPOSURE_MANUAL),
        (ce.V4L2_EXPOSURE_ABSOLUTE, 100),
    ]


def test_leave_touches_nothing():
    assert ce.plan("leave") == []
    assert ce.apply("/dev/does-not-exist", "leave") == []


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        ce.plan("bright")


def test_missing_device_reports_every_control_failed():
    assert len(ce.apply("/dev/does-not-exist", "auto")) == 2


def test_device_index_becomes_a_path():
    assert ce.device_path(0) == "/dev/video0"
    assert ce.device_path("/dev/video2") == "/dev/video2"
