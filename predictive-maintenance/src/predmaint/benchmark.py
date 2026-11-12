"""Measure serving latency on this machine, in process, for the served model.

    python -m predmaint.benchmark                 # FD001, writes docs/benchmark.json
    python -m predmaint.benchmark --repeats 50

Reports the median and 95th percentile of: one prediction without an explanation, one with
integrated gradients, one replay tick for the whole fleet, and one engine-detail refresh (the
attribution heatmap the dashboard shows for a watched engine). Inference runs on the CPU, as it
does in the Docker image. The results set how many watched engines the replay re-explains per
tick (see ``Settings.explain_per_tick``).

The predictor reuses an attribution map when the same window is explained again. The
attribution timings clear that cache before every timed call, so they measure the computation;
``engine_detail_repeated_state`` measures the reuse separately.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from predmaint.config import PROJECT_ROOT, Settings
from predmaint.inference import Predictor
from predmaint.replay import FleetReplay


def cpu_name() -> str:
    """The processor's marketing name where the OS gives it, else Python's processor string."""
    try:
        if sys.platform == "win32":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        pass
    return platform.processor() or platform.machine()


def _time(
    fn: Callable[[], object],
    repeats: int,
    warmup: int = 3,
    before: Callable[[], object] | None = None,
) -> dict:
    """Median and p95 of ``fn`` in milliseconds; ``before`` runs untimed ahead of every call."""
    for _ in range(warmup):
        if before:
            before()
        fn()
    samples = []
    for _ in range(repeats):
        if before:
            before()
        start = time.perf_counter()
        fn()
        samples.append(1000 * (time.perf_counter() - start))
    values = np.array(samples)
    return {
        "median_ms": round(float(np.median(values)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "runs": repeats,
    }


def run(model_dir: Path, repeats: int) -> dict:
    predictor = Predictor(model_dir)
    replay_rows = pd.read_csv(model_dir / "replay.csv")
    unit = int(replay_rows["unit"].iloc[0])
    engine = replay_rows[replay_rows["unit"] == unit].sort_values("cycle")
    history = engine.head(predictor.contract.window + 20)
    fleet = FleetReplay(predictor, replay_rows, n_engines=12)
    results = {
        "predict_without_explanation": _time(
            lambda: predictor.predict_readings(history, explain=False), repeats
        ),
        "predict_with_integrated_gradients": _time(
            lambda: predictor.predict_readings(history, explain=True),
            repeats,
            before=predictor.clear_explanations,
        ),
        "replay_tick_whole_fleet": _time(fleet.step, repeats),
        "engine_detail_with_attribution": _time(lambda: fleet.detail(fleet.units[0]), repeats),
    }
    return {
        "model": model_dir.name,
        "networks": len(predictor.model.members),
        "fleet_engines": len(fleet.units),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": {
            "platform": platform.platform(),
            "processor": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "latency": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m predmaint.benchmark", description=__doc__)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "benchmark.json")
    args = parser.parse_args(argv)
    report = run(Settings.from_env().artifacts_dir / args.subset, args.repeats)
    args.out.write_text(json.dumps(report, indent=2))
    for name, stats in report["latency"].items():
        print(f"{name:36s} median {stats['median_ms']:8.1f} ms   p95 {stats['p95_ms']:8.1f} ms")
    hardware = report["hardware"]
    print(f"Hardware: {hardware['processor']} ({hardware['logical_cpus']} logical CPUs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
