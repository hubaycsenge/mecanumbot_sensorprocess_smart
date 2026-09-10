#!/usr/bin/env python3
"""
Tests for where an nvinfer config's parser library and relative paths resolve.

The same templates are launched on the robot (user `ubuntu`, libraries built in
~/deepstream_source) and on a laptop (another user, libraries vendored under
~/Documents/installed_external). The claims below are what lets one template
serve both without an edit.
"""

import os

import pytest

from mecanumbot_sensorprocess_smart import nvinfer_config

LIBRARY = "DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Return a fresh home directory that `~` and `$HOME` expand to."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "someone")
    return tmp_path


def build(root, library=LIBRARY):
    """Create an empty parser library under `root` and return its path."""
    path = root / library
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return str(path)


def test_template_path_written_against_home_expands(home):
    built = build(home / "deepstream_source")
    found, _ = nvinfer_config.find_custom_lib(
        "", "~/deepstream_source/" + LIBRARY, LIBRARY
    )
    assert found == built


def test_robot_layout_found_without_configuration(home):
    built = build(home / "deepstream_source")
    found, _ = nvinfer_config.find_custom_lib("", "/home/ubuntu/nowhere.so", LIBRARY)
    assert found == built


def test_laptop_layout_found_when_the_template_names_the_robot(home):
    built = build(home / "Documents" / "installed_external")
    found, tried = nvinfer_config.find_custom_lib(
        "", "~/deepstream_source/" + LIBRARY, LIBRARY
    )
    assert found == built
    # the robot's location was looked at first, and is not repeated
    assert tried[0] == str(home / "deepstream_source" / LIBRARY)
    assert len(tried) == len(set(tried))


def test_configured_path_expands_variables(home):
    built = build(home / "elsewhere")
    found, _ = nvinfer_config.find_custom_lib(
        "$HOME/elsewhere/" + LIBRARY, "", LIBRARY
    )
    assert found == built


def test_configured_path_is_the_only_candidate(home):
    build(home / "deepstream_source")
    found, tried = nvinfer_config.find_custom_lib(
        "/opt/${USER}/missing.so", "", LIBRARY
    )
    assert found is None
    assert tried == ["/opt/someone/missing.so"]


def test_nothing_built_reports_every_place_looked(home):
    found, tried = nvinfer_config.find_custom_lib("", None, LIBRARY)
    assert found is None
    assert tried == [
        str(home / "deepstream_source" / LIBRARY),
        str(home / "Documents" / "installed_external" / LIBRARY),
    ]


def test_relative_paths_resolve_against_the_template(tmp_path):
    config_dir = tmp_path / "share" / "deepstream_config"
    config_dir.mkdir(parents=True)
    template = config_dir / "config.txt"
    template.write_text(
        "[property]\n"
        "onnx-file=../models/imgsz_640/yolo26m.onnx\n"
        "model-engine-file=/abs/yolo26m.engine\n"
        "labelfile-path=labels_coco.txt\n"
        "custom-lib-path=~/lib.so\n"
    )
    resolved = nvinfer_config.absolute_paths(str(template))
    assert resolved == {
        "onnx-file": str(tmp_path / "share" / "models" / "imgsz_640" / "yolo26m.onnx"),
        "labelfile-path": str(config_dir / "labels_coco.txt"),
    }


def test_rendered_copy_carries_the_found_library(home, tmp_path):
    built = build(home / "Documents" / "installed_external")
    template = tmp_path / "config.txt"
    template.write_text(
        "[property]\ncustom-lib-path=~/deepstream_source/" + LIBRARY + "\n"
    )
    found, _ = nvinfer_config.find_custom_lib(
        "",
        nvinfer_config.read_setting(str(template), "custom-lib-path"),
        LIBRARY,
    )
    rendered = tmp_path / "rendered.txt"
    nvinfer_config.render_config(
        str(template), {"custom-lib-path": found}, str(rendered)
    )
    assert nvinfer_config.read_setting(str(rendered), "custom-lib-path") == built
    assert os.path.isabs(built)
