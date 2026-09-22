from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def load_ply(path: str | Path) -> torch.Tensor:
    """Load XYZ vertices from a PLY file as an (N, 3) float32 tensor on CPU."""
    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]
    coords = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    return torch.from_numpy(coords)


def save_ply(path: str | Path, points: torch.Tensor) -> None:
    """Write an (N, 3) float tensor to a PLY file (binary little-endian)."""
    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError(f"expected (N, 3) points, got {tuple(points.shape)}")
    arr = points.detach().cpu().numpy().astype(np.float32)
    structured = np.empty(arr.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    structured["x"], structured["y"], structured["z"] = arr[:, 0], arr[:, 1], arr[:, 2]
    PlyData([PlyElement.describe(structured, "vertex")], text=False).write(str(path))
