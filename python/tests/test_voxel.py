import pytest
import torch

from ghcompute.operations.voxel import voxel_downsample


def test_rejects_non_3d_input():
    with pytest.raises(ValueError):
        voxel_downsample(torch.randn(10, 2), voxel_size=1.0)


def test_rejects_invalid_voxel_size():
    pts = torch.randn(10, 3)
    with pytest.raises(ValueError):
        voxel_downsample(pts, voxel_size=0.0)
    with pytest.raises(ValueError):
        voxel_downsample(pts, voxel_size=-1.0)


def test_uniform_grid_downsamples_to_expected_count():
    # 10x10x10 grid of points spaced 1 apart in [0,9]
    coords = torch.arange(10, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    # voxel size 2 -> 5x5x5 = 125 voxels
    out = voxel_downsample(grid, voxel_size=2.0)
    assert out.size(0) == 125


def test_cluster_centroid_is_averaged():
    # Four points all in the voxel rooted at (0,0,0) for voxel_size=10
    pts = torch.tensor([[1.0, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    out = voxel_downsample(pts, voxel_size=10.0)
    assert out.size(0) == 1
    assert torch.allclose(out[0], torch.tensor([2.5, 2.5, 2.5]))


def test_output_count_never_exceeds_input():
    torch.manual_seed(3)
    pts = torch.randn(500, 3)
    out = voxel_downsample(pts, voxel_size=0.5)
    assert out.size(0) <= 500


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_matches_cpu():
    torch.manual_seed(4)
    pts_cpu = torch.randn(500, 3)
    out_cpu = voxel_downsample(pts_cpu, voxel_size=0.5)
    out_mps = voxel_downsample(pts_cpu.to("mps"), voxel_size=0.5).cpu()
    # Voxel order is unique-key dependent and may differ across devices;
    # compare as sorted point sets.
    cpu_sorted = out_cpu[torch.argsort(out_cpu.sum(dim=1))]
    mps_sorted = out_mps[torch.argsort(out_mps.sum(dim=1))]
    assert cpu_sorted.shape == mps_sorted.shape
    assert torch.allclose(cpu_sorted, mps_sorted, atol=1e-4)
