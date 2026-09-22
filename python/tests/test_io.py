import torch

from ghcompute.io import load_ply, save_ply


def test_ply_roundtrip(tmp_path):
    pts = torch.tensor(
        [[0.0, 0, 0], [1, 2, 3], [-1.5, 0.25, 100.0]],
        dtype=torch.float32,
    )
    out = tmp_path / "test.ply"
    save_ply(out, pts)
    loaded = load_ply(out)
    assert loaded.shape == pts.shape
    assert loaded.dtype == torch.float32
    assert torch.allclose(loaded, pts)


def test_save_rejects_non_3d():
    import pytest
    with pytest.raises(ValueError):
        save_ply("/dev/null", torch.randn(5, 2))
