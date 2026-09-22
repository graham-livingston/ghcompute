# Benchmarks

Hardware: Apple M2, Apple M2
PyTorch: 2.11.0

Methodology: median of 3 runs after one warmup. Tiled matmul pairwise distance plus `topk` for k-NN, `scatter_add` over hashed voxel keys for voxel downsample. CPU is single-threaded (PyTorch default). Random point clouds.

## k-NN (k=16)

Tiled pairwise distance: O(T*N) memory per tile, O(N^2) work. Sub-linear k-NN (KD-tree or hash grid) is future work.

| Points | CPU | MPS | Speedup |
|---|---|---|---|
| 1,000 | 3.47 ms | 6.27 ms | 0.6× |
| 10,000 | 199.6 ms | 356.2 ms | 0.6× |
| 50,000 | 5.39 s | 9.35 s | 0.6× |

## Voxel downsample (size=0.5)

Linear in input size. Scales to many millions of points.

| Points | CPU | MPS | Speedup |
|---|---|---|---|
| 10,000 | 1.07 ms | 1.19 ms | 0.9× |
| 100,000 | 7.30 ms | 1.89 ms | 3.9× |
| 1,000,000 | 68.4 ms | 13.5 ms | 5.1× |
