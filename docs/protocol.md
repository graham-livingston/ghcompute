# Wire protocol (v0.1)

C# spawns `python -m ghcompute.service`, writes one request frame, reads one response frame, the Python process exits.

All multi-byte integers are big-endian.

## Request frame

```
[1 byte ] opcode
           0x01 knn
           0x02 voxel_downsample
[4 bytes] args_len  (uint32)
[args_len bytes] args, JSON-encoded UTF-8
           knn:   {"k": int}
           voxel: {"size": float}
[4 bytes] payload_len (uint32)
[payload_len bytes] points, float32 little-endian, 3*N values
```

## Response frame

```
[1 byte ] status
           0x00 ok
           0x01 user error  (bad args, malformed payload)
           0xFF internal error
[4 bytes] meta_len (uint32)
[meta_len bytes] meta, JSON-encoded UTF-8
           ok:    {"device": str, "elapsed_ms": float, "shape": [int, int]}
           error: {"message": str}
[4 bytes] payload_len (uint32)
[payload_len bytes] result bytes
           knn:   int32 indices [N*k] then float32 distances [N*k]
           voxel: float32 [3*M]
```

## Error handling

On `0x01` or `0xFF`, payload is empty (`payload_len = 0`). `meta.message` describes the failure. C# surfaces this as a Grasshopper component runtime warning.
