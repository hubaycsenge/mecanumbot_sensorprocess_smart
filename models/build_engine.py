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
TensorRT version and do not transfer from a development machine. Free the RAM
first -- the builder needs several GB of host memory and is silently killed by
the OOM reaper when the ROS stack and the desktop session are still up.

    python3 build_engine.py yolo26n-pose.onnx
    python3 build_engine.py yolo26m-pose.onnx --workspace 1024
"""

import argparse
import os
import shutil
import sys
import threading
import time

import tensorrt as trt

# Below this much free host memory the builder is likely to be OOM-killed.
LOW_MEMORY_WARN_MB = 4096


def engine_path_for(onnx_path, batch, gpu, precision):
    """Return the engine filename nvinfer derives from an nvinfer config."""
    return "{}_b{}_gpu{}_{}.engine".format(onnx_path, batch, gpu, precision)


def available_memory_mb():
    """Return available host memory in MB, or None if it cannot be read."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def format_duration(seconds):
    """Return a compact m/s rendering of a duration."""
    minutes, seconds = divmod(int(seconds), 60)
    if minutes:
        return "{}m{:02d}s".format(minutes, seconds)
    return "{}s".format(seconds)


class ProgressReporter(object):
    """
    Render builder progress on a single terminal line.

    Falls back to periodic plain lines when stdout is not a terminal, so the
    output stays readable when the build is run under nohup or piped to a log.
    """

    def __init__(self, stream=None, min_interval=0.1):
        self.stream = stream or sys.stdout
        self.interval = min_interval
        self.tty = self.stream.isatty()
        self.started = time.time()
        self.last_draw = 0.0
        self.width = 0

    def draw(self, label, current, total, force=False):
        """Draw the bar for one phase, throttled to the redraw interval."""
        now = time.time()
        if not force and now - self.last_draw < self.interval:
            return
        self.last_draw = now
        elapsed = format_duration(now - self.started)

        if total > 0:
            fraction = min(1.0, current / float(total))
            counter = "{}/{}".format(current, total)
        else:
            fraction = 0.0
            counter = str(current)

        if not self.tty:
            self.stream.write(
                "  [{:3.0f}%] {} ({}) {}\n".format(fraction * 100, label, counter, elapsed)
            )
            self.stream.flush()
            return

        columns = shutil.get_terminal_size((80, 24)).columns
        suffix = " {:3.0f}% {} {}".format(fraction * 100, counter, elapsed)
        bar_width = max(10, min(30, columns - len(label) - len(suffix) - 6))
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)

        line = "  [{}]{} {}".format(bar, suffix, label)
        line = line[: columns - 1]
        self.width = max(self.width, len(line))
        self.stream.write("\r" + line.ljust(self.width))
        self.stream.flush()

    def clear(self):
        """Erase the current bar line."""
        if self.tty and self.width:
            self.stream.write("\r" + " " * self.width + "\r")
            self.stream.flush()
            self.width = 0


def make_progress_monitor(reporter):
    """
    Return an IProgressMonitor forwarding builder phases to the reporter.

    Returns None when the installed TensorRT predates the progress monitor API
    (added in TensorRT 8.6), so callers can fall back to a heartbeat.
    """
    if not hasattr(trt, "IProgressMonitor"):
        return None

    class _Monitor(trt.IProgressMonitor):
        """Track nested builder phases and render the innermost one."""

        def __init__(self):
            trt.IProgressMonitor.__init__(self)
            self.stack = []
            self.totals = {}

        def phase_start(self, phase_name, parent_phase, num_steps):
            """Record a phase the builder has entered."""
            self.stack.append(phase_name)
            self.totals[phase_name] = num_steps
            reporter.draw(phase_name, 0, num_steps, force=True)

        def step_complete(self, phase_name, step):
            """Report one completed step; returning False aborts the build."""
            reporter.draw(phase_name, step, self.totals.get(phase_name, 0))
            return True

        def phase_finish(self, phase_name):
            """Drop a finished phase and redraw its parent, if any."""
            if phase_name in self.stack:
                self.stack.remove(phase_name)
            self.totals.pop(phase_name, None)
            if self.stack:
                parent = self.stack[-1]
                reporter.draw(parent, 0, self.totals.get(parent, 0), force=True)
            else:
                reporter.clear()

    return _Monitor()


class Heartbeat(object):
    """Print elapsed time and free memory while a blocking build runs."""

    def __init__(self, reporter, period=5.0):
        self.reporter = reporter
        self.period = period
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.wait(self.period):
            free = available_memory_mb()
            label = "building" if free is None else "building ({} MB free)".format(free)
            self.reporter.draw(label, 0, 0, force=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.reporter.clear()
        return False


def build(onnx_path, out_path, precision, workspace_mb, verbose, progress):
    """Build and serialize an engine, returning True on success."""
    severity = trt.Logger.VERBOSE if verbose else trt.Logger.INFO
    logger = trt.Logger(severity)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    # EXPLICIT_BATCH is implied and deprecated from TensorRT 10 onwards.
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
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

    free = available_memory_mb()
    if free is not None:
        print("  host memory available: {} MB".format(free))
        if free < LOW_MEMORY_WARN_MB:
            print(
                "warning: under {} MB free; the builder is likely to be OOM-killed. "
                "Stop the ROS stack and the desktop session first.".format(
                    LOW_MEMORY_WARN_MB
                ),
                file=sys.stderr,
            )

    reporter = ProgressReporter()
    monitor = make_progress_monitor(reporter) if progress else None
    if monitor is not None:
        config.progress_monitor = monitor

    print("Building {} engine (this takes minutes, do not interrupt)...".format(precision))
    started = time.time()
    if monitor is not None:
        serialized = builder.build_serialized_network(network, config)
        reporter.clear()
    elif progress:
        with Heartbeat(reporter):
            serialized = builder.build_serialized_network(network, config)
    else:
        serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        print("Engine build failed; see the TensorRT log above.", file=sys.stderr)
        return False

    with open(out_path, "wb") as handle:
        handle.write(serialized)
    print(
        "Wrote {} ({:.1f} MB) in {}".format(
            out_path, os.path.getsize(out_path) / 1e6, format_duration(time.time() - started)
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
    parser.add_argument(
        "--no-progress", dest="progress", action="store_false",
        help="disable the progress bar",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="verbose TensorRT log; implies --no-progress, the log would break the bar",
    )
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

    progress = args.progress and not args.verbose
    ok = build(onnx_path, out_path, args.precision, args.workspace, args.verbose, progress)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
