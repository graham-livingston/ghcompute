import torch

_TARGET_TILE_BYTES = 2 * 1024**3  # ~2 GB working set per tile


def _auto_tile_size(n: int, dtype_bytes: int = 4) -> int:
    """Pick a tile size whose distance matrix fits ~2 GB.

    Returns at least 64 (small inputs are tile-bounded only by N) and at
    most 4096 (above this the kernel-launch amortization plateaus).
    """
    if n <= 0:
        return 64
    return max(64, min(4096, _TARGET_TILE_BYTES // (n * dtype_bytes)))


def knn(points: torch.Tensor, k: int, *, tile_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """k-nearest-neighbor search via tiled, matmul-based pairwise distance.

    Each tile of T queries builds a (T, N) squared-distance block from
    |x - y|^2 = |x|^2 + |y|^2 - 2 x.y, so the cross term is a single
    matmul. sqrt is applied to the (T, k) topk result only. Peak memory
    is O(T*N) instead of O(N^2); work is still O(N^2).

    Args:
        points: (N, 3) float tensor on any device.
        k: number of neighbors per query, must be in [1, N-1].
        tile_size: override the auto-chosen tile size.

    Returns:
        indices: (N, k) int64 tensor on the input device.
        distances: (N, k) float tensor on the input device, sorted ascending.
    """
    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError(f"expected (N, 3) points, got {tuple(points.shape)}")
    n = points.size(0)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k >= n:
        raise ValueError(f"k ({k}) must be < N ({n})")

    if tile_size is None:
        tile_size = _auto_tile_size(n)

    device = points.device
    dtype = points.dtype
    points_t = points.t().contiguous()  # (3, N) for matmul
    all_norms_sq = (points * points).sum(dim=1)  # (N,) computed once

    out_idx = torch.empty((n, k), dtype=torch.long, device=device)
    out_dist = torch.empty((n, k), dtype=dtype, device=device)

    for start in range(0, n, tile_size):
        end = min(start + tile_size, n)
        chunk = points[start:end]  # (T, 3)
        chunk_norms_sq = (chunk * chunk).sum(dim=1, keepdim=True)  # (T, 1)
        # |x - y|^2 = |x|^2 + |y|^2 - 2 x.y
        cross = chunk @ points_t  # (T, N) matmul
        d_sq = chunk_norms_sq + all_norms_sq - 2.0 * cross
        d_sq.clamp_(min=0.0)  # guard against fp negatives near 0

        rows = torch.arange(end - start, device=device)
        d_sq[rows, rows + start] = float("inf")  # exclude self-match

        td_sq, ti = d_sq.topk(k, dim=1, largest=False)
        out_idx[start:end] = ti
        out_dist[start:end] = td_sq.sqrt()  # sqrt only the (T, k) result

    return out_idx, out_dist
