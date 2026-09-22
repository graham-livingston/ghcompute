import pytest
import torch

from ghcompute.operations.knn import knn


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        knn(torch.randn(10, 2), k=3)
    with pytest.raises(ValueError):
        knn(torch.randn(5), k=3)


def test_rejects_invalid_k():
    pts = torch.randn(10, 3)
    with pytest.raises(ValueError):
        knn(pts, k=0)
    with pytest.raises(ValueError):
        knn(pts, k=10)
    with pytest.raises(ValueError):
        knn(pts, k=11)


def test_collinear_neighbors_are_correct():
    # Four points along x-axis spaced 1 apart
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    idx, dist = knn(pts, k=1)
    # endpoints' nearest is their adjacent neighbor (1.0 away)
    assert idx[0, 0].item() == 1
    assert idx[3, 0].item() == 2
    assert torch.allclose(dist, torch.ones(4, 1))


def test_distances_are_sorted_ascending():
    torch.manual_seed(0)
    pts = torch.randn(100, 3)
    _, dist = knn(pts, k=8)
    assert (dist[:, 1:] >= dist[:, :-1] - 1e-6).all()


def test_self_match_is_excluded():
    torch.manual_seed(1)
    pts = torch.randn(50, 3)
    idx, _ = knn(pts, k=10)
    for i in range(50):
        assert i not in idx[i].tolist()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_matches_cpu():
    torch.manual_seed(2)
    pts_cpu = torch.randn(200, 3)
    pts_mps = pts_cpu.to("mps")
    idx_cpu, dist_cpu = knn(pts_cpu, k=8)
    idx_mps, dist_mps = knn(pts_mps, k=8)
    assert torch.equal(idx_cpu, idx_mps.cpu())
    assert torch.allclose(dist_cpu, dist_mps.cpu(), atol=1e-3)


def test_tiled_matches_untiled():
    torch.manual_seed(5)
    pts = torch.randn(500, 3)
    idx_big, dist_big = knn(pts, k=8, tile_size=4096)  # one tile (== brute force)
    idx_small, dist_small = knn(pts, k=8, tile_size=37)  # many small tiles
    assert torch.equal(idx_big, idx_small)
    assert torch.allclose(dist_big, dist_small)


def test_handles_input_larger_than_default_tile():
    # 5000 points with a tiny tile to verify tiling doesn't break the
    # output shape or self-exclusion at boundaries.
    torch.manual_seed(6)
    pts = torch.randn(5000, 3)
    idx, dist = knn(pts, k=8, tile_size=128)
    assert idx.shape == (5000, 8)
    assert dist.shape == (5000, 8)
    for i in range(0, 5000, 500):
        assert i not in idx[i].tolist()
    assert (dist[:, 1:] >= dist[:, :-1] - 1e-5).all()
