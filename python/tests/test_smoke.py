import torch

import ghcompute
from ghcompute.device import select_device


def test_package_imports():
    assert ghcompute.__version__ == "0.1.0"


def test_device_selection_returns_a_device():
    device = select_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "mps", "cuda"}
