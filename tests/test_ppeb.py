import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "ChinesePaperCutting" / "ChinesePaperCutting_Transfer"
sys.path.insert(0, str(MODEL_DIR))

from model.framework import StripAttentionBlock
from model.ppeb import LightSymmetryGate, PaperCutPriorEnhancementBlock


def test_ppeb_preserves_shape():
    block = PaperCutPriorEnhancementBlock(dim=16)
    x = torch.randn(2, 16, 24, 24)
    y = block(x)
    assert y.shape == x.shape


def test_ppeb_backpropagates():
    block = PaperCutPriorEnhancementBlock(dim=8)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    y = block(x).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_ppeb_off_mode_is_identity():
    block = PaperCutPriorEnhancementBlock(dim=8)
    block.set_mode("off")
    x = torch.randn(1, 8, 12, 10)

    y = block(x)

    assert torch.equal(y, x)


@pytest.mark.parametrize("mode", ["edge", "freq", "full"])
def test_ppeb_ablation_modes_preserve_shape(mode):
    block = PaperCutPriorEnhancementBlock(dim=8)
    block.set_mode(mode)
    x = torch.randn(1, 8, 12, 10)

    y = block(x)

    assert y.shape == x.shape


def test_light_symmetry_gate_uses_scale_invariant_symmetry_hint():
    gate = LightSymmetryGate(dim=1)
    with torch.no_grad():
        for parameter in gate.gate.parameters():
            parameter.zero_()

    x = torch.tensor([[[[0.3, 0.1, 0.2, 0.4]]]], dtype=torch.float32)
    scaled_x = x * 10.0

    y = gate(x)
    scaled_y = gate(scaled_x)

    relative_boost = (y - x).abs().mean() / x.abs().mean()
    scaled_relative_boost = (scaled_y - scaled_x).abs().mean() / scaled_x.abs().mean()

    assert relative_boost > 0
    assert torch.isclose(relative_boost, scaled_relative_boost, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_ppeb_preserves_dtype_device_and_shape_for_non_square_inputs(device, dtype):
    block = PaperCutPriorEnhancementBlock(dim=8).to(device=device, dtype=dtype)
    x = torch.randn(2, 8, 5, 7, device=device, dtype=dtype)

    y = block(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_strip_attention_block_integration_handles_non_square_sequences():
    block = StripAttentionBlock(d_model=8, input_resolution=(3, 5), nhead=2, strip_width=2)
    x = torch.randn(1, 15, 8)

    y, arbitrary_input, output_shape = block((x, True, (3, 5)))

    assert arbitrary_input is True
    assert output_shape == (3, 5)
    assert y.shape == x.shape
