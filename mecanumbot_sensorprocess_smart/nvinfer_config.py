#!/usr/bin/env python3
"""
Building an nvinfer config for the model a DeepStream node was asked to run.

Two nodes in this package render their own nvinfer configuration from a packaged
template: `mecanumbot_onboard_cam_detect_people`, which runs a pose model, and
`mecanumbot_onboard_cam_detect_objects`, which runs a plain detector for the
fetch game. What they share is the mechanics and only the mechanics.

An ONNX export is fixed to the input size it was exported at, so the exports are
stored one folder per size -- `models/imgsz_<n>/` -- the engine sits beside its
own ONNX under the name nvinfer derives from the config, and three lines of that
config (`onnx-file`, `model-engine-file`, `infer-dims`) have to agree with the
folder that was picked. Keeping those three in step by hand is how a stale
engine gets loaded silently: the engine filename carries no input size, so an
engine built from an earlier export keeps being used and keeps running at its
own size however the ONNX is re-exported.

What each node *puts* in its config -- the parser, the class count, the network
type -- is that node's business and stays in its own template. Only the
substitution is here.

Plain file handling: no ROS, no DeepStream, no numpy. That keeps it testable on
a development machine and keeps the two nodes from drifting apart on the one
thing they genuinely do the same way.
"""

import os


def model_paths(models_dir, imgsz, model_name, precision, batch=1, gpu=0):
    """
    Return `(onnx, engine)` for one model at one input size.

    The engine name is the one nvinfer derives from the config itself
    (`<onnx>_b<batch>_gpu<gpu>_<precision>.engine`), so naming it here is how a
    node can tell whether the engine already exists without asking DeepStream.
    Neither path is checked for existence -- the caller decides what a missing
    ONNX means, and a missing engine is normal on a first launch.
    """
    onnx = os.path.join(
        models_dir, "imgsz_{}".format(int(imgsz)), "{}.onnx".format(model_name)
    )
    engine = "{}_b{}_gpu{}_{}.engine".format(onnx, int(batch), int(gpu), precision)
    return onnx, engine


def render_config(template, substitutions, output_path):
    """
    Write `template` to `output_path` with the named keys replaced.

    A key the template does not carry at all is appended rather than dropped:
    nvinfer needs it either way, and a template written before the key existed
    should not silently lose it. Everything else -- comments, ordering, the
    settings the node has no opinion about -- is passed through untouched, so
    the rendered copy is readable next to the original when a run has to be
    explained.

    Raises `OSError` if the template cannot be read or the copy cannot be
    written; the caller decides whether to fall back to the template.
    """
    remaining = dict(substitutions)
    with open(template, "r") as handle:
        lines = handle.readlines()

    rendered = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in remaining:
            rendered.append("{}={}\n".format(key, remaining.pop(key)))
        else:
            rendered.append(line)

    if remaining:
        rendered.append("# added by mecanumbot_sensorprocess_smart\n")
        rendered.extend(
            "{}={}\n".format(key, value) for key, value in remaining.items()
        )

    with open(output_path, "w") as handle:
        handle.writelines(rendered)
    return output_path


def read_setting(config_path, key):
    """
    Return one setting's raw string value from an nvinfer config, or None.

    Comments are stripped first, so a key that appears only in a commented
    explanation is not mistaken for the setting. The *last* assignment wins,
    which is what nvinfer itself does -- these files repeat keys more than once.
    """
    value = None
    with open(config_path, "r") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            name, _, raw = line.partition("=")
            if name.strip() == key:
                value = raw.strip()
    return value


def read_infer_dims(config_path):
    """
    Return the declared network input as `(width, height)`, or None.

    `infer-dims` is optional -- without it nvinfer takes the input shape from
    the model, which is only knowable once inference has run -- and DeepStream
    spells it `channels;height;width`, which is the opposite order to the one
    everything downstream wants.
    """
    raw = read_setting(config_path, "infer-dims")
    if not raw:
        return None
    dims = raw.split(";")
    if len(dims) < 3:
        return None
    return (int(dims[2]), int(dims[1]))
