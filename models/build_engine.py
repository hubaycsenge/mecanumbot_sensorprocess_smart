#!/usr/bin/env python3
"""
Build a TensorRT engine from a YOLO pose ONNX file.

Standalone alternative to letting DeepStream's nvinfer build the engine on the
first launch, which hides parser errors and restarts from scratch every time
the node is started. Uses the TensorRT Python API, so it also works on images
that ship without the `trtexec` binary (the `tensorrt` samples package).

The output filename defaults to the exact name nvinfer looks for,
`<onnx>_b<batch>_gpu<gpu>_<precision>.engine`, so the result drops straight
into `model-engine-file` in `deepstream_config/config_infer_yolo26_pose.txt`.

Run it on the Jetson: engines are tied to the GPU architecture and the
TensorRT version and do not transfer from a development machine.

    python3 build_engine.py yolo26n-pose.onnx
    python3 build_engine.py yolo26m-pose.onnx --workspace 1024
"""

import argparse
import os
import sys
import time

import tensorrt as trt


def engine_path_for(onnx_path, batch, gpu, precision):
    """Return the engine filename nvinfer derives from an nvinfer config."""
    return "{}_b{}_gpu{}_{}.engine".format(onnx_path, batch, gpu, precision)


def build(onnx_path, out_path, precision, workspace_mb, verbose):
    """Build and serialize an engine, returning True on success."""
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            print("ONNX parsing failed:", file=sys.stderr)
            for i in range(parser.num_errors):
                print("  [{}] {}".format(i, parser.get_error(i)), file=sys.stderr)
            return False

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print("  input  {}: {} {}".format(i, tensor.name, tensor.shape))
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print("  output {}: {} {}".format(i, tensor.name, tensor.shape))

    config = builder.create_builder_config()
    # max_workspace_size was removed in TensorRT 10.
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
        )
    else:  # pragma: no cover - TensorRT < 8.4
        config.max_workspace_size = workspace_mb * 1024 * 1024

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("warning: platform reports no fast FP16 support", file=sys.stderr)
        config.set_flag(trt.BuilderFlag.FP16)

    print("Building {} engine (this takes minutes, do not interrupt)...".format(precision))
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("Engine build failed; see the TensorRT log above.", file=sys.stderr)
        return False

    with open(out_path, "wb") as handle:
        handle.write(serialized)
    print(
        "Wrote {} ({:.1f} MB) in {:.0f} s".format(
            out_path, os.path.getsize(out_path) / 1e6, time.time() - started
        )
    )
    return True


def main():
    """Parse arguments and build the requested engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", help="path to the ONNX file")
    parser.add_argument("-o", "--output", help="engine path (default: the nvinfer name)")
    parser.add_argument(
        "--precision", choices=("fp16", "fp32"), default="fp16",
        help="must match network-mode in the nvinfer config (fp16 -> 2, fp32 -> 0)",
    )
    parser.add_argument("--batch", type=int, default=1, help="batch size in the engine name")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id in the engine name")
    parser.add_argument(
        "--workspace", type=int, default=2048,
        help="workspace pool limit in MB; lower it if the build is killed",
    )
    parser.add_argument("--verbose", action="store_true", help="verbose TensorRT log")
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    if not os.path.isfile(onnx_path):
        parser.error("no such ONNX file: {}".format(onnx_path))

    out_path = args.output or engine_path_for(
        onnx_path, args.batch, args.gpu, args.precision
    )
    print("TensorRT {}".format(trt.__version__))
    print("  onnx:   {}".format(onnx_path))
    print("  engine: {}".format(out_path))

    return 0 if build(onnx_path, out_path, args.precision, args.workspace, args.verbose) else 1


if __name__ == "__main__":
    sys.exit(main())
