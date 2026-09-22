import torch

_HASH_PRIMES = (73856093, 19349663, 83492791)


def voxel_downsample(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Voxel-grid downsample. Returns the centroid of each occupied voxel.

    Args:
        points: (N, 3) float tensor on any device.
        voxel_size: voxel edge length, in the same units as points.

    Returns:
        centroids: (M, 3) float tensor on the input device, M <= N.
    """
    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError(f"expected (N, 3) points, got {tuple(points.shape)}")
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    voxels = (points / voxel_size).floor().long()
    p1, p2, p3 = _HASH_PRIMES
    keys = (voxels[:, 0] * p1) ^ (voxels[:, 1] * p2) ^ (voxels[:, 2] * p3)

    unique_keys, inverse = torch.unique(keys, return_inverse=True)

    n_voxels = unique_keys.size(0)
    sums = torch.zeros((n_voxels, 3), device=points.device, dtype=points.dtype)
    counts = torch.zeros(n_voxels, device=points.device, dtype=points.dtype)
    sums.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), points)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=points.dtype))
    return sums / counts.unsqueeze(1)
