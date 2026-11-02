# GHCompute

GPU-accelerated point-cloud primitives for Grasshopper, delivered through a two-layer architecture: a thin Grasshopper plugin in C# bridged to a PyTorch compute service in Python.

The first release ships GPU implementations of two operations that bottleneck most point-cloud pipelines: **k-nearest-neighbor search** and **voxel downsampling**. GPU work runs through PyTorch's MPS backend on Apple Silicon and CUDA on NVIDIA hardware, with a CPU fallback when neither is available.

---

## Why two layers

Most computational designers who need GPU compute drop out of Grasshopper entirely. The existing GPU-accelerated GH plugins (tOpos, Legolizer, CityLBM, FlexHopper) are end-to-end simulation solvers tied to NVIDIA CUDA. They do not expose lower-level primitives, and they do not run on Apple Silicon.

GHCompute takes a different approach: keep the Grasshopper plugin small and idiomatic in C#, and put the GPU compute in a Python sidecar where the ecosystem (PyTorch, MPS, CUDA, ROCm) already solves cross-vendor portability.

## Components

| Component | Inputs | Outputs | Notes |
|---|---|---|---|
| `GPU k-NN` | Points (list), k | Indices (tree, k per point), Distances (tree), Device | Tiled matmul pairwise distance + `topk`, runs on MPS / CUDA / CPU |
| `GPU Voxel Downsample` | Points (list), voxel size (model units) | Points (voxel centroids), Device | Quantized coordinates + hashed keys + `scatter_add` |

## Architecture

```
+-------------------------------+
|  Grasshopper canvas           |
+-------------------------------+
            |
            v
+-------------------------------+      stdin/stdout, length-prefixed binary frames
|  GHCompute (C#)               | <----------------------------------------------->  +--------------------------+
|  - GpuKnnComponent            |                                                    |  ghcompute (Python)      |
|  - GpuVoxelDownsampleComponent|                                                    |  - PyTorch + MPS/CUDA    |
|  - PythonBridge               |                                                    |  - knn, voxel_downsample |
+-------------------------------+                                                    +--------------------------+
```

The C# layer handles Grasshopper integration, point-cloud marshalling between `Point3d` and `float32` buffers, and lifecycle management of the Python subprocess. The Python layer does the GPU work. v0.1 spawns one Python process per request; the wire format is documented in [`docs/protocol.md`](./docs/protocol.md).

## Prior art

GHCompute stands on prior work and isn't trying to displace any of it. Credit where it's due:

- **[Cockroach](https://github.com/9and3/Cockroach)** by Andrea Settimi (EPFL IBOIS) — point-cloud post-processing and meshing in Rhino/GH using Open3D, CGAL, and Cilantro. CPU-based but the most complete point-cloud-first plugin in the GH ecosystem.
- **[Volvox](https://www.grasshopper3d.com/group/volvox)** — point-cloud creation, editing, and analysis in Grasshopper. CPU.
- **[GHGpuComputingTest](https://github.com/hrntsm/GHGpuComputingTest)** by Hironori Tsuruta — exploratory CUDA-via-Alea integration in Grasshopper. The closest direct precedent to GHCompute's GPU-compute-from-GH framing; explored a different point in the design space (NVIDIA-only, in-process JIT).
- **[tOpos](https://www.food4rhino.com/app/topos)** — CUDA-accelerated topology optimization. End-to-end solver, NVIDIA-only.
- **[Legolizer](https://www.food4rhino.com/en/app/legolizer)** — CUDA-accelerated voxel/LEGO generation.
- **[CityLBM](https://www.food4rhino.com/en/app/citylbm)** — GPU lattice-Boltzmann for urban wind simulation.
- **[FlexHopper](https://www.food4rhino.com/en/app/flexhopper)** — GPU particle physics in Grasshopper.

GHCompute occupies the gap between these: GPU-accelerated *primitives* for point clouds (rather than a complete simulation solver), with a sidecar architecture that runs on Apple Silicon as well as NVIDIA.

## Installation

1. Install Rhino 8 (Mac or Windows).
2. Install Python 3.11+ and the compute package: `pip install -e python` from a clone of this repo (pulls in `torch`, `numpy`, `plyfile`). MPS is used automatically on Apple Silicon, CUDA on supported NVIDIA hardware, CPU otherwise.
3. Download the latest `.gha` from [Releases](https://github.com/graham-livingston/ghcompute/releases) and drop it into your Grasshopper Libraries folder:
   - **Mac:** `~/Library/Application Support/McNeel/Rhinoceros/8.0/Plug-ins/Grasshopper (b45a29b1-4343-4035-989e-044e8580d9cf)/Libraries/`. The GUID-suffixed folder is the one Rhino 8 scans; the plain `Plug-ins/Grasshopper/Libraries/` folder is not. Run `xattr -c GHCompute.gha` first to clear the download quarantine flag.
   - **Windows:** `%APPDATA%\Grasshopper\Libraries\` (untested in v0.1).
4. Quit Rhino fully and reopen.

The plugin looks for `python3` on `PATH`, then in the usual Homebrew and system locations. Set `GHCOMPUTE_PYTHON=/path/to/python` to override. Each request has a 60 s timeout.

## Quick start

In Grasshopper, find the two components under the **GHCompute** tab. Feed any list of points (Populate 3D works well) into `GPU k-NN` with a `k` slider, or into `GPU Voxel Downsample` with a voxel-size slider. The component message bar shows the round-trip time. An example definition is coming in `examples/`.

The Python layer also runs standalone, without Rhino, on any PLY file:

```
python3 -m ghcompute.cli knn path/to/cloud.ply --k 8
python3 -m ghcompute.cli voxel path/to/cloud.ply --size 0.5 --out downsampled.ply
```

## Benchmarks

Measured on Apple M2 (8-core CPU, 10-core GPU), PyTorch 2.11, median of 3 runs after warmup. See `benchmarks/results.md` for the full table and methodology; reproduce via `python3 benchmarks/run.py`.

**Voxel downsample** — big MPS win at scale:

| Points | CPU | MPS | Speedup |
|---|---|---|---|
| 10,000 | 1.1 ms | 1.2 ms | 0.9× |
| 100,000 | 7.3 ms | 1.9 ms | **3.9×** |
| 1,000,000 | 68 ms | 13.5 ms | **5.0×** |

**k-NN, k=16** — tiled, matmul-based pairwise distance:

| Points | CPU | MPS |
|---|---|---|
| 1,000 | 3.5 ms | 6.3 ms |
| 10,000 | 200 ms | 356 ms |
| 50,000 | 5.4 s | 9.4 s |

### Honest read on these numbers

**Voxel downsample shows a clean 3.9–5× speedup on MPS at ≥100k points** — the algorithm parallelizes well over voxel cells, and `scatter_add` + `unique` are operations PyTorch's Metal backend handles efficiently.

**Brute-force k-NN is CPU-favored on Apple M2** at every size, even after rewriting the inner kernel as a matmul-based squared-distance computation (skipping the per-pair sqrt by deferring it to the (T, k) topk result). MKL's CPU paths for this pattern are very mature; MPS's `topk` on a (4096, N) tensor is the apparent bottleneck on Apple's GPU. On CUDA hardware the same code typically wins by 5–10× at this scale.

The takeaway: the sidecar architecture works, and **the gain is workload-dependent and device-dependent**. Voxel downsample demonstrates the win on Apple Silicon today; k-NN demonstrates that *brute-force* is the wrong shape of work for MPS regardless of how it's written. v0.2's spatial-hashed k-NN reduces the problem to O(N) average and should win on every backend.

## Future work

- **Spatial-hashed k-NN** — replace the O(N²) tiled brute-force kernel with a grid-based search that is O(N) on average. This is the single most impactful v0.2 change; benchmarks above are likely to flip in MPS's favor across all sizes.
- **Persistent Python service** — v0.1 spawns Python per request; v0.2 will keep a long-lived process and switch to a streaming binary protocol for sub-millisecond invocation overhead.
- **Pure-C# GPU backend on Windows** via [ComputeSharp](https://github.com/Sergio0694/ComputeSharp) — would let the plugin run without a Python dependency on Windows users' machines. Cross-platform Vulkan-via-MoltenVK is the longer-term path.
- **`GpuScriptComponent`** — let users author PyTorch operations directly inside Grasshopper, with typed input/output sockets.
- **Additional primitives** — normal estimation, radius search, mesh smoothing (Laplacian), ray-mesh intersection.
- **Gaussian Splat viewer component** — load `.splat` / `.ply` Gaussian Splatting files and render in a Rhino viewport. Speculative; depends on Rhino's display SDK.

## Building from source

```
git clone https://github.com/graham-livingston/ghcompute.git
cd ghcompute
# Python side
pip install -e "python[dev]"
pytest python/tests
# C# side (build the csproj directly; the .slnx needs .NET 10 tooling)
dotnet build src/GHCompute/GHCompute.csproj -c Release -f net7.0
# .gha lands at src/GHCompute/bin/Release/net7.0/GHCompute.gha
```

CI runs the Python tests on CPU and the C# build on every push. GPU tests are local-only.

## License

MIT. See [LICENSE](./LICENSE).

## Acknowledgements
