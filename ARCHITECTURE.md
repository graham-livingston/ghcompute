# GHCompute architecture

Design rationale for choices that aren't obvious from reading the code. If you're picking this up cold, read this first.

---

## The core decision: two-layer sidecar over single-language plugin

An earlier framing of this project was a pure-C# plugin using ComputeSharp (DirectX 12 compute shaders authored in C#). That plan was rejected after confirming the developer is on macOS — ComputeSharp targets DirectX 12, which Apple does not support. Cross-platform options through Vulkan + MoltenVK exist but add weeks of toolchain work for a 3-week sprint.

The version that *is* shippable on Mac in 3 weeks: a Python sidecar that does the GPU work, bridged from a thin C# Grasshopper plugin. Python's GPU ecosystem (PyTorch + MPS / CUDA / ROCm) already solves cross-vendor portability; we just plug into it.

This is also a *better* fit for the ETH application context:

- The Software Engineer for XR JD requires "strong proficiency in Python and C#." A two-language project demonstrates both naturally.
- COMPAS (the ETH-developed framework most relevant to the role) is Python-core with C# bindings for Unity/Rhino. SLAM/SfM/Gaussian Splatting workflows mentioned in the JD are Python+CUDA in practice.
- The architecture mirrors how research labs actually combine ML/numerical pipelines with CAD tools.

An even earlier framing — "wrap a Grasshopper group, run it on GPU transparently" — was rejected outright. CIL → GPU translation is a multi-year research problem (ILGPU is the closest existing answer and supports only a strict subset of C#). The shippable version is a curated library of GPU implementations of specific operations behind a clean abstraction. That is GHCompute.

---

## Layer separation

```
+----------------------------------------+
|  Grasshopper canvas                    |
+----------------------------------------+
              |
              v
+----------------------------------------+
|  GHCompute (C#)                        |   thin layer
|  - GpuKnnComponent                     |   point marshalling,
|  - GpuVoxelDownsampleComponent         |   subprocess management,
|  - PythonBridge                        |   error handling
+----------------------------------------+
              | stdin / stdout
              | length-prefixed binary frames
              v
+----------------------------------------+
|  ghcompute (Python package)            |
|  - service.py     (protocol)           |
|  - operations/                         |
|    - knn.py       (tiled matmul+topk)  |
|    - voxel.py     (scatter_add)        |
|  - io.py          (PLY, NPY)           |
+----------------------------------------+
              | (PyTorch device dispatch)
              v
+----------------------------------------+
|  PyTorch backend (auto-selected)       |
|  - MPS  (Apple Silicon)                |
|  - CUDA (NVIDIA)                       |
|  - CPU  (fallback)                     |
+----------------------------------------+
```

**Why this split:** the C# layer is the Grasshopper integration and nothing else. All compute, all GPU concerns, all algorithm logic lives in Python. This gives us:

- Native macOS GPU support via MPS without writing platform-specific code.
- Free CUDA/ROCm support — anyone on those backends gets it.
- A test suite that runs in CI without Rhino installed.
- Clean separation between "what the user sees" and "what the GPU does."

**Consequence:** the C# code does no math beyond data marshalling. If you're tempted to add an algorithm in C#, stop. It belongs in Python.

---

## Protocol design

C# spawns Python as a subprocess and communicates over stdin/stdout. Why not HTTP via Hops? Hops requires Rhino.Compute, which is heavier setup for users; the subprocess pattern is self-contained.

### Frame format (v0.1.0)

**Request:**
```
[1B opcode][4B json_len][json args bytes][4B payload_len][float32 payload bytes]
```

**Response:**
```
[1B status][4B length][result bytes]
```

Status: `0x00` ok, `0x01` user error (e.g. invalid args), `0xFF` internal error.

### Per-request lifecycle (v0.1.0)

The C# layer spawns a fresh Python process per GH evaluation. Process reads one request frame, writes one response frame, exits.

This is **slow** (Python startup ~150–300ms, PyTorch import ~500–1000ms) but **simple**. Acceptable for v0.1 because point clouds do not re-evaluate at frame rate. v0.2 will keep a persistent process and pipe many requests through one connection.

### Why not stream the protocol in v0.1

A persistent service would require: subprocess lifecycle in Rhino's AppDomain, health checks, restart-on-crash logic, request multiplexing, clean shutdown on Rhino exit. All doable, none cheap. v0.1 trades latency for simplicity. The README is honest about this and points to v0.2.

---

## Backend selection

PyTorch handles backend selection on its own. The Python layer:

```python
def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
```

The selected device name is reported back to the C# layer in the response and surfaced on the GH component's warning panel — important for trust ("am I actually running on GPU?").

---

## Algorithm notes

### k-NN

```python
def knn(points: torch.Tensor, k: int, tile_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    points_t = points.t().contiguous()                          # (3, N)
    norms_sq = (points * points).sum(dim=1)                     # (N,)
    for start in range(0, n, tile_size):
        chunk = points[start:end]                               # (T, 3)
        d_sq = chunk_norms_sq + norms_sq - 2.0 * (chunk @ points_t)  # (T, N)
        d_sq[rows, rows + start] = inf                          # drop self-match
        td_sq, ti = d_sq.topk(k, dim=1, largest=False)          # (T, k)
        out_dist[start:end] = td_sq.sqrt()
```

Brute force in work, O(N²), but tiled so peak memory is O(T·N). The tile size is chosen so the (T, N) block stays near 2 GB. The cross term is a single matmul per tile and sqrt runs only on the (T, k) result. Sub-linear search (spatial hash or KD-tree) is the v0.2 change.

### Voxel downsample

```python
def voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    voxels = (points / voxel_size).floor().long()
    keys = voxels[:, 0] * P1 + voxels[:, 1] * P2 + voxels[:, 2] * P3   # hash
    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    centroids = torch.zeros((unique_keys.size(0), 3), device=points.device)
    counts = torch.zeros(unique_keys.size(0), device=points.device)
    centroids.scatter_add_(0, inverse.unsqueeze(1).expand_as(points), points)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float))
    return centroids / counts.unsqueeze(1)
```

Standard pattern. `torch.unique` is the bottleneck on MPS — verify it's supported on the target PyTorch version during D2.

---

## Why PyTorch over alternatives

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **PyTorch** | First-class MPS + CUDA + CPU, massive ecosystem, easy install | Heavy dep (~700MB), startup latency | ✅ chosen |
| **JAX** | Compiles to XLA, very fast | MPS support immature, less GH-relevant | Skip |
| **CuPy** | NumPy-like, fast on CUDA | NVIDIA-only — defeats the Mac point | Skip |
| **Taichi** | Pure-Python kernels, GPU-friendly | Smaller ecosystem, additional learning curve | Future option |
| **Native Metal via PyObjC** | Maximum performance, no PyTorch dep | Mac-only, fragile, far more code | Skip |
| **ComputeSharp** | Pure C#, no Python | Windows-only via DX12 | Rejected (see top) |
| **Silk.NET + Vulkan + MoltenVK** | Cross-platform pure C# | Toolchain weeks | Future option |

The deciding factor: PyTorch+MPS is the *only* path that gives a real GPU story on Mac inside a 3-week budget while keeping the door open for CUDA users on Windows.

---

## C# layer guarantees

The C# code is responsible for:

- Detecting the user's Python (`which python3` → `GHCOMPUTE_PYTHON` env override → fail with clear message).
- Marshalling `IEnumerable<Point3d>` → `byte[]` of float32.
- Spawning `Process.Start("python3", "-m", "ghcompute.service")`, writing the request frame, reading the response frame.
- Handling Python errors: stderr → GH warning panel.
- Marshalling the response back to GH types.

The C# layer does NOT:

- Do any geometric math.
- Cache results.
- Manage long-running state.
- Implement fallbacks for missing Python features.

---

## Things that are intentionally not abstracted

Resist the temptation to over-engineer. The following are NOT abstracted in v0.1:

- **Logging.** Use `RhinoApp.WriteLine` directly on the C# side, `print(file=sys.stderr)` on the Python side.
- **Configuration.** No config files. Voxel size etc. come from component sockets.
- **Error types.** Throw `InvalidOperationException` with clear messages. No custom exception hierarchy.
- **DI container.** Static `PythonBridge` class. No DI.
- **Persistent state.** None. Per-request lifecycle.

If you find yourself adding any of the above, stop and ask whether it's serving v0.1 or imagined v0.5.

---

## Testing strategy

- **Python: `pytest` runs in CI on macOS runner.** No GPU on CI; tests run on `cpu` device. The same test code runs locally on `mps` and asserts agreement (within float tolerance).
- **C#: `dotnet build` runs in CI**, but no automated GH tests in v0.1. Manual verification via example `.gh` files.
- **End-to-end: a `tests/integration/` script** that spawns the Python service from C# (or directly from a small test harness) and validates the round-trip. Runs locally only.

---

## File layout — full

```
ghcompute/
├── .github/workflows/ci.yml
├── benchmarks/
│   ├── run.py
│   ├── results.md
│   └── results.png
├── examples/
│   ├── data/
│   │   └── bunny.ply
│   ├── knn_demo.gh
│   ├── voxel_demo.gh
│   └── demo.gif
├── python/
│   ├── pyproject.toml
│   ├── ghcompute/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── service.py
│   │   ├── io.py
│   │   └── operations/
│   │       ├── __init__.py
│   │       ├── knn.py
│   │       └── voxel.py
│   └── tests/
│       ├── test_knn.py
│       └── test_voxel.py
├── src/
│   ├── GHCompute.sln
│   └── GHCompute/
│       ├── GHCompute.csproj
│       ├── GHComputeInfo.cs
│       ├── PythonBridge.cs
│       ├── GpuKnnComponent.cs
│       └── GpuVoxelDownsampleComponent.cs
├── docs/
│   └── protocol.md
├── ARCHITECTURE.md
├── LICENSE
└── README.md
```
