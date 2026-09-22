"""Benchmark CPU vs MPS for k-NN and voxel downsample.

k-NN is tiled, so memory is O(T*N) and inputs of a few hundred thousand
points fit on device; work is still O(N^2). Voxel downsample is linear
and scales to many millions.

Usage:
    python3 benchmarks/run.py
    python3 benchmarks/run.py --quick   # smaller sizes, faster
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path
from statistics import median

import torch

from ghcompute.operations.knn import knn
from ghcompute.operations.voxel import voxel_downsample

KNN_SIZES_FULL = [1_000, 10_000, 50_000, 100_000, 200_000]
KNN_SIZES_QUICK = [1_000, 10_000, 50_000]
VOXEL_SIZES_FULL = [10_000, 100_000, 1_000_000, 5_000_000]
VOXEL_SIZES_QUICK = [10_000, 100_000, 1_000_000]


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except subprocess.SubprocessError:
            pass
    return platform.processor() or "unknown"


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            for line in out.splitlines():
                if "Chipset Model:" in line:
                    return line.split(":", 1)[1].strip()
        except subprocess.SubprocessError:
            pass
        return "Apple GPU (MPS)"
    return "none"


def time_op(fn, *args, n_runs: int = 5, sync) -> float:
    sync()
    fn(*args)  # warmup
    sync()
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args)
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return median(samples)


def make_sync(device: torch.device):
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "mps":
        return lambda: torch.mps.synchronize()
    return lambda: None


def bench_knn(sizes: list[int]) -> list[dict]:
    rows = []
    for n in sizes:
        torch.manual_seed(0)
        pts_cpu = torch.randn(n, 3)
        row = {"n": n}
        for device_str in ("cpu", "mps"):
            if device_str == "mps" and not torch.backends.mps.is_available():
                row[device_str] = None
                continue
            device = torch.device(device_str)
            pts = pts_cpu.to(device)
            sync = make_sync(device)
            try:
                row[device_str] = time_op(knn, pts, 16, n_runs=3, sync=sync)
            except (RuntimeError, MemoryError) as e:
                print(f"  knn {device_str} N={n}: FAILED: {type(e).__name__}: {e}")
                row[device_str] = None
        print(f"  knn N={n:>8}  cpu={row['cpu']!s:>10}  mps={row['mps']!s:>10}")
        rows.append(row)
    return rows


def bench_voxel(sizes: list[int]) -> list[dict]:
    rows = []
    for n in sizes:
        torch.manual_seed(0)
        pts_cpu = torch.randn(n, 3) * 10  # spread points over a wider range
        row = {"n": n}
        for device_str in ("cpu", "mps"):
            if device_str == "mps" and not torch.backends.mps.is_available():
                row[device_str] = None
                continue
            device = torch.device(device_str)
            pts = pts_cpu.to(device)
            sync = make_sync(device)
            try:
                row[device_str] = time_op(voxel_downsample, pts, 0.5, n_runs=3, sync=sync)
            except (RuntimeError, MemoryError) as e:
                print(f"  voxel {device_str} N={n}: FAILED: {type(e).__name__}: {e}")
                row[device_str] = None
        print(f"  voxel N={n:>8}  cpu={row['cpu']!s:>10}  mps={row['mps']!s:>10}")
        rows.append(row)
    return rows


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "OOM/skip"
    if value < 10:
        return f"{value:.2f} ms"
    if value < 1000:
        return f"{value:.1f} ms"
    return f"{value/1000:.2f} s"


def speedup(cpu: float | None, mps: float | None) -> str:
    if cpu is None or mps is None or mps == 0:
        return "—"
    return f"{cpu/mps:.1f}×"


def write_markdown(knn_rows, voxel_rows, cpu, gpu) -> str:
    lines = [
        "# Benchmarks",
        "",
        f"Hardware: {cpu}, {gpu}",
        f"PyTorch: {torch.__version__}",
        "",
        "Methodology: median of 3 runs after one warmup. Tiled matmul pairwise distance "
        "plus `topk` for k-NN, `scatter_add` over hashed voxel keys for voxel downsample. "
        "CPU is single-threaded (PyTorch default). Random point clouds.",
        "",
        "## k-NN (k=16)",
        "",
        "Tiled pairwise distance: O(T*N) memory per tile, O(N^2) work. "
        "Sub-linear k-NN (KD-tree or hash grid) is future work.",
        "",
        "| Points | CPU | MPS | Speedup |",
        "|---|---|---|---|",
    ]
    for r in knn_rows:
        lines.append(
            f"| {r['n']:,} | {fmt_ms(r['cpu'])} | {fmt_ms(r['mps'])} | {speedup(r['cpu'], r['mps'])} |"
        )

    lines += [
        "",
        "## Voxel downsample (size=0.5)",
        "",
        "Linear in input size. Scales to many millions of points.",
        "",
        "| Points | CPU | MPS | Speedup |",
        "|---|---|---|---|",
    ]
    for r in voxel_rows:
        lines.append(
            f"| {r['n']:,} | {fmt_ms(r['cpu'])} | {fmt_ms(r['mps'])} | {speedup(r['cpu'], r['mps'])} |"
        )

    return "\n".join(lines) + "\n"


def make_plot(knn_rows, voxel_rows, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, rows, title in [
        (ax1, knn_rows, "k-NN, k=16 (brute force)"),
        (ax2, voxel_rows, "Voxel downsample, size=0.5"),
    ]:
        ns = [r["n"] for r in rows if r["cpu"] is not None and r["mps"] is not None]
        cpus = [r["cpu"] for r in rows if r["cpu"] is not None and r["mps"] is not None]
        mpss = [r["mps"] for r in rows if r["cpu"] is not None and r["mps"] is not None]
        x = range(len(ns))
        ax.bar([i - 0.2 for i in x], cpus, width=0.4, label="CPU", color="#888")
        ax.bar([i + 0.2 for i in x], mpss, width=0.4, label="MPS", color="#0a84ff")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{n:,}" for n in ns], rotation=30)
        ax.set_ylabel("median runtime (ms, log scale)")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller sizes, faster run")
    args = parser.parse_args()

    knn_sizes = KNN_SIZES_QUICK if args.quick else KNN_SIZES_FULL
    voxel_sizes = VOXEL_SIZES_QUICK if args.quick else VOXEL_SIZES_FULL

    cpu = cpu_name()
    gpu = gpu_name()
    print(f"hardware: {cpu}, {gpu}")
    print(f"torch: {torch.__version__}")
    print()

    print("k-NN benchmarks:")
    knn_rows = bench_knn(knn_sizes)
    print()
    print("Voxel benchmarks:")
    voxel_rows = bench_voxel(voxel_sizes)
    print()

    out_dir = Path(__file__).parent
    md = write_markdown(knn_rows, voxel_rows, cpu, gpu)
    (out_dir / "results.md").write_text(md)
    print(f"wrote {out_dir / 'results.md'}")
    make_plot(knn_rows, voxel_rows, out_dir / "results.png")


if __name__ == "__main__":
    main()
