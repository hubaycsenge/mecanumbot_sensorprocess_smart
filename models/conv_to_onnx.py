#!/usr/bin/env python3
"""
Export a YOLO pose checkpoint to ONNX at a fixed input size.

An ONNX export is fixed to the `imgsz` it was exported at, so the exports are
kept one folder per size -- `models/imgsz_<n>/` -- next to the size-independent
`.pt` checkpoints. `mecanumbot_onboard_cam_detect_people` selects the folder
with its `model_params.imgsz` parameter (the `yolo_imgsz` launch argument), so
an export only has to land in the right folder to be selectable; nothing in the
nvinfer config has to be edited to match.

    python3 conv_to_onnx.py yolo26m-pose --imgsz 1280
    python3 conv_to_onnx.py yolo26n-pose --imgsz 640

The TensorRT engine is built beside the ONNX by build_engine.py, or by nvinfer
on the first launch. An engine is never shared between sizes, because it sits
in the folder of the ONNX it was built from.
"""

import argparse
import os
import shutil
import sys

from ultralytics import YOLO

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """Export the requested checkpoint into its imgsz folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", nargs="?", default="yolo26m-pose",
        help="checkpoint stem in models/ (default: %(default)s)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="input size to fix the export at (default: %(default)s)",
    )
    args = parser.parse_args()

    checkpoint = os.path.join(MODELS_DIR, "{}.pt".format(args.model))
    if not os.path.isfile(checkpoint):
        parser.error("no such checkpoint: {}".format(checkpoint))

    target_dir = os.path.join(MODELS_DIR, "imgsz_{}".format(args.imgsz))
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "{}.onnx".format(args.model))

    model = YOLO(checkpoint)
    exported = model.export(format="onnx", imgsz=args.imgsz)

    if os.path.abspath(exported) != target:
        shutil.move(exported, target)
    print("Wrote {}".format(target))

    # nvinfer does not rebuild an engine when the ONNX behind it changes, and
    # the engine filename carries no input size, so a stale engine beside a
    # fresh export keeps running at the old size.
    engine_prefix = target + "_b"
    stale = [
        name for name in os.listdir(target_dir)
        if os.path.join(target_dir, name).startswith(engine_prefix)
    ]
    for name in stale:
        print("Delete the stale engine {} before launching.".format(
            os.path.join(target_dir, name)
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
