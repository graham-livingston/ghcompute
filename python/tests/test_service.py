"""Tests for the wire protocol service.

These spawn `python -m ghcompute.service` as a subprocess for each
case. That mirrors how the C# Grasshopper layer will invoke it in v0.1.
"""

from __future__ import annotations

import io
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch

from ghcompute.service import (
    OP_KNN,
    OP_VOXEL,
    STATUS_INTERNAL_ERROR,
    STATUS_OK,
    STATUS_USER_ERROR,
    read_response_frame,
    write_request_frame,
)


def _run_service(opcode: int, args: dict, payload: bytes) -> tuple[int, dict, bytes]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "ghcompute.service"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    request = io.BytesIO()
    write_request_frame(request, opcode, args, payload)
    out, err = proc.communicate(request.getvalue(), timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"service exited {proc.returncode}: stderr={err.decode(errors='replace')}")
    return read_response_frame(io.BytesIO(out))


def _points_to_payload(pts: torch.Tensor) -> bytes:
    return pts.to(torch.float32).numpy().tobytes()


def test_frame_helpers_roundtrip():
    buf = io.BytesIO()
    payload = b"\x01\x02\x03"
    write_request_frame(buf, OP_KNN, {"k": 4}, payload)
    buf.seek(0)
    # The request reader is in the service module; reuse it via the same names.
    from ghcompute.service import read_request_frame
    op, args, p = read_request_frame(buf)
    assert op == OP_KNN
    assert args == {"k": 4}
    assert p == payload


def test_knn_roundtrip():
    torch.manual_seed(0)
    pts = torch.randn(100, 3)
    status, meta, payload = _run_service(OP_KNN, {"k": 4}, _points_to_payload(pts))
    assert status == STATUS_OK, meta
    assert meta["shape"] == [100, 4]
    assert meta["device"] in {"cpu", "mps", "cuda"}
    n_idx_bytes = 100 * 4 * 4  # int32
    n_dist_bytes = 100 * 4 * 4  # float32
    assert len(payload) == n_idx_bytes + n_dist_bytes
    indices = np.frombuffer(payload[:n_idx_bytes], dtype=np.int32).reshape(100, 4)
    distances = np.frombuffer(payload[n_idx_bytes:], dtype=np.float32).reshape(100, 4)
    # Self-match must be excluded
    for i in range(100):
        assert i not in indices[i].tolist()
    # Distances ascending
    assert (distances[:, 1:] >= distances[:, :-1] - 1e-5).all()


def test_voxel_roundtrip():
    coords = torch.arange(10, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    status, meta, payload = _run_service(OP_VOXEL, {"size": 2.0}, _points_to_payload(grid))
    assert status == STATUS_OK, meta
    assert meta["shape"][1] == 3
    assert meta["shape"][0] == 125  # 5x5x5
    centroids = np.frombuffer(payload, dtype=np.float32).reshape(-1, 3)
    assert centroids.shape == (125, 3)


def test_unknown_opcode_returns_user_error():
    status, meta, payload = _run_service(0x7F, {}, b"")
    assert status == STATUS_USER_ERROR
    assert "unknown opcode" in meta["message"]
    assert payload == b""


def test_invalid_args_returns_user_error():
    pts = torch.randn(10, 3)
    # k=10 invalid because k must be < N
    status, meta, payload = _run_service(OP_KNN, {"k": 10}, _points_to_payload(pts))
    assert status == STATUS_USER_ERROR
    assert payload == b""


def test_malformed_payload_returns_internal_error():
    # Payload length 7 is not a multiple of 12 (3 floats * 4 bytes)
    status, meta, _ = _run_service(OP_KNN, {"k": 1}, b"\x00" * 7)
    assert status in {STATUS_USER_ERROR, STATUS_INTERNAL_ERROR}
