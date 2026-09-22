"""Local CLI for smoke-testing operations independent of Grasshopper."""

from __future__ import annotations

import argparse
import sys
import time

from ghcompute.device import select_device
from ghcompute.io import load_ply, save_ply
from ghcompute.operations.knn import knn
from ghcompute.operations.voxel import voxel_downsample


def _cmd_knn(args: argparse.Namespace) -> None:
    device = select_device()
    points = load_ply(args.path).to(device)
    t0 = time.perf_counter()
    idx, dist = knn(points, args.k)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"loaded {points.size(0)} points on {device}")
    print(f"k-NN k={args.k} in {elapsed:.1f}ms")
    print(f"first row indices:   {idx[0].tolist()}")
    print(f"first row distances: {[round(d, 4) for d in dist[0].tolist()]}")


def _cmd_voxel(args: argparse.Namespace) -> None:
    device = select_device()
    points = load_ply(args.path).to(device)
    n_in = points.size(0)
    t0 = time.perf_counter()
    out = voxel_downsample(points, args.size)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"loaded {n_in} points on {device}")
    print(f"voxel downsample size={args.size} -> {out.size(0)} centroids in {elapsed:.1f}ms")
    if args.out:
        save_ply(args.out, out)
        print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ghcompute")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_knn = sub.add_parser("knn", help="run k-NN on a PLY file")
    p_knn.add_argument("path")
    p_knn.add_argument("--k", type=int, default=8)
    p_knn.set_defaults(fn=_cmd_knn)

    p_vox = sub.add_parser("voxel", help="run voxel downsample on a PLY file")
    p_vox.add_argument("path")
    p_vox.add_argument("--size", type=float, required=True)
    p_vox.add_argument("--out", help="optional output PLY path")
    p_vox.set_defaults(fn=_cmd_voxel)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
