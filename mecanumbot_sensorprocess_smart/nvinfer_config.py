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

# Where the DeepStream-Yolo checkouts are built, searched in this order when
# the configured parser library is not there: the robot keeps them in
# ~/deepstream_source, a development machine in the workspace's vendored
# installed_external/. Written against `~` so that neither is tied to one user.
BUILD_ROOTS = ("~/deepstream_source", "~/Documents/installed_external")

# The settings in an nvinfer config that name a file. A relative one resolves
# against the config's own directory, so it stops resolving once a rendered
# copy is written anywhere else.
PATH_KEYS = ("onnx-file", "model-engine-file", "labelfile-path")


def expand_path(path):
    """Return `path` with `~` and environment variables (`$HOME`, `$USER`) expanded."""
    return os.path.expandvars(os.path.expanduser(path)) if path else path


def find_custom_lib(configured, template_value, library, roots=BUILD_ROOTS):
    """
    Return `(found, tried)` for the parser library nvinfer has to load.

    `configured` is `model_params.custom_lib_path`. When it is set it is the
    only candidate: an explicit path that does not exist is a mistake to
    report, not one to paper over by loading some other build. When it is
    empty, the path written in the template is tried first and then `library`
    (relative, e.g. `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/lib....so`)
    under each of `roots`. Every candidate has `~` and `$VARS` expanded, which
    nvinfer itself does not do.

    `found` is the first candidate that exists, or None; `tried` is every
    expanded path looked at, in order, so a failure can say where it looked.
    """
    if configured:
        candidates = [configured]
    else:
        candidates = [template_value] + [os.path.join(root, library) for root in roots]

    tried = []
    for candidate in candidates:
        if not candidate:
            continue
        path = expand_path(candidate)
        if path in tried:
            continue
        tried.append(path)
        if os.path.isfile(path):
            return path, tried
    return None, tried


def absolute_paths(config_path, keys=PATH_KEYS):
    """
    Return the named file settings of a config, resolved against its directory.

    Only relative values are returned; an absolute one, and one written against
    `~` or a variable, is already independent of where the config sits.
    """
    base = os.path.dirname(os.path.abspath(config_path))
    resolved = {}
    for key in keys:
        value = read_setting(config_path, key)
        if value and not os.path.isabs(value) and not value.startswith(("~", "$")):
            resolved[key] = os.path.normpath(os.path.join(base, value))
    return resolved


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
