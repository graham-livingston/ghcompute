"""Stdin/stdout binary service.

Reads one length-prefixed request frame, runs the requested operation,
writes one response frame, exits. Protocol spec in docs/protocol.md.

The frame helpers (`read_request_frame`, `write_response_frame`, and
their inverses) are exposed so tests can build/parse frames without
re-implementing the protocol.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from typing import BinaryIO

import torch

from ghcompute.device import select_device
from ghcompute.operations.knn import knn
from ghcompute.operations.voxel import voxel_downsample

OP_KNN = 0x01
OP_VOXEL = 0x02

STATUS_OK = 0x00
STATUS_USER_ERROR = 0x01
STATUS_INTERNAL_ERROR = 0xFF


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError(f"expected {n} bytes, got {len(buf)} before EOF")
        buf.extend(chunk)
    return bytes(buf)


def read_request_frame(stream: BinaryIO) -> tuple[int, dict, bytes]:
    opcode = _read_exact(stream, 1)[0]
    args_len = struct.unpack(">I", _read_exact(stream, 4))[0]
    args = json.loads(_read_exact(stream, args_len).decode("utf-8")) if args_len else {}
    payload_len = struct.unpack(">I", _read_exact(stream, 4))[0]
    payload = _read_exact(stream, payload_len)
    return opcode, args, payload


def write_request_frame(stream: BinaryIO, opcode: int, args: dict, payload: bytes) -> None:
    args_bytes = json.dumps(args).encode("utf-8")
    stream.write(bytes([opcode]))
    stream.write(struct.pack(">I", len(args_bytes)))
    stream.write(args_bytes)
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def read_response_frame(stream: BinaryIO) -> tuple[int, dict, bytes]:
    status = _read_exact(stream, 1)[0]
    meta_len = struct.unpack(">I", _read_exact(stream, 4))[0]
    meta = json.loads(_read_exact(stream, meta_len).decode("utf-8")) if meta_len else {}
    payload_len = struct.unpack(">I", _read_exact(stream, 4))[0]
    payload = _read_exact(stream, payload_len)
    return status, meta, payload


def write_response_frame(stream: BinaryIO, status: int, meta: dict, payload: bytes) -> None:
    meta_bytes = json.dumps(meta).encode("utf-8")
    stream.write(bytes([status]))
    stream.write(struct.pack(">I", len(meta_bytes)))
    stream.write(meta_bytes)
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def _run(opcode: int, args: dict, payload: bytes, device: torch.device) -> tuple[dict, bytes]:
    if opcode not in (OP_KNN, OP_VOXEL):
        raise ValueError(f"unknown opcode 0x{opcode:02x}")
    if len(payload) == 0 or len(payload) % 12 != 0:
        raise ValueError(f"payload length {len(payload)} is not a non-zero multiple of 12 (3 floats * 4 bytes)")

    points = torch.frombuffer(bytearray(payload), dtype=torch.float32).reshape(-1, 3).to(device)

    t0 = time.perf_counter()
    if opcode == OP_KNN:
        idx, dist = knn(points, int(args["k"]))
        out = idx.to(torch.int32).cpu().numpy().tobytes() + dist.to(torch.float32).cpu().numpy().tobytes()
        meta = {"shape": [int(idx.size(0)), int(idx.size(1))]}
    else:  # OP_VOXEL
        result = voxel_downsample(points, float(args["size"]))
        out = result.to(torch.float32).cpu().numpy().tobytes()
        meta = {"shape": [int(result.size(0)), 3]}

    meta["device"] = str(device)
    meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
    return meta, out


def main() -> None:
    try:
        opcode, args, payload = read_request_frame(sys.stdin.buffer)
    except Exception as e:
        write_response_frame(sys.stdout.buffer, STATUS_INTERNAL_ERROR, {"message": f"frame read failed: {e}"}, b"")
        return

    device = select_device()
    try:
        meta, payload_out = _run(opcode, args, payload, device)
        write_response_frame(sys.stdout.buffer, STATUS_OK, meta, payload_out)
    except (ValueError, KeyError) as e:
        write_response_frame(sys.stdout.buffer, STATUS_USER_ERROR, {"message": str(e)}, b"")
    except Exception as e:
        write_response_frame(sys.stdout.buffer, STATUS_INTERNAL_ERROR, {"message": f"{type(e).__name__}: {e}"}, b"")


if __name__ == "__main__":
    main()
