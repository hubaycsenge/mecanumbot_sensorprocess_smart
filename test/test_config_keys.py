#!/usr/bin/env python3
"""
The perception parameter file has to reach the nodes it is written for.

It did not, for as long as it existed: its top-level keys were bare node names,
which match a node in the root namespace only, and every node here runs in
`mecanumbot`. Each one started on the defaults in its own source. These tests
hold the two things that go wrong silently -- a key no namespaced node matches,
and a topic name that would cut the fusion off from the LiDAR detector the
moment the file did load. No ROS.
"""

import os

import yaml

CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "config", "lidar_peopledetect_config.yaml"
)

NODES = (
    "mecanumbot_lidar_detect_people",
    "mecanumbot_cam_detect_people_ds",
    "mecanumbot_cam_detect_objects_ds",
    "mecanumbot_locate_detections",
)


def load():
    with open(CONFIG, "r") as handle:
        return yaml.safe_load(handle)


def test_every_key_matches_its_node_in_any_namespace():
    for key in load():
        assert key.startswith("/**/"), (
            f"'{key}' only matches a node in the root namespace; the nodes run "
            "in 'mecanumbot', so nothing under it would be loaded"
        )


def test_every_node_has_a_section():
    assert sorted(load()) == sorted(f"/**/{node}" for node in NODES)


def test_every_section_is_a_parameter_block():
    for key, section in load().items():
        assert list(section) == ["ros__parameters"], key


def test_the_lidar_detector_publishes_where_the_fusion_listens():
    # mecanumbot_locate_detections subscribes to `dets`, and the trees wait for
    # /mecanumbot/dets before their first tick.
    parameters = load()["/**/mecanumbot_lidar_detect_people"]["ros__parameters"]
    assert parameters["detections_topic"] == "dets"
