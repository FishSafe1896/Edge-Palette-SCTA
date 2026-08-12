from pathlib import Path
import sys

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "ChinesePaperCutting" / "ChinesePaperCutting_Transfer"
sys.path.insert(0, str(MODEL_DIR))

from model.configuration import TransModule_Config  # noqa: E402
from model.style_adapter import StyleConditionedTextureAdapter  # noqa: E402
from net import TransModule  # noqa: E402
from tools import load_network_weights, load_transmodule_state  # noqa: E402


def test_style_adapter_preserves_token_shape_and_dtype():
    adapter = StyleConditionedTextureAdapter(dim=8)
    content = torch.randn(2, 11, 8, dtype=torch.float32)
    style = torch.randn(2, 7, 8, dtype=torch.float32)

    out = adapter(content, style)

    assert out.shape == content.shape
    assert out.dtype == content.dtype


def test_style_adapter_is_identity_at_initialization():
    adapter = StyleConditionedTextureAdapter(dim=8)
    content = torch.randn(2, 11, 8)
    style = torch.randn(2, 7, 8)

    out = adapter(content, style)

    torch.testing.assert_close(out, content)


def test_style_adapter_uses_style_after_residual_projection_is_enabled():
    adapter = StyleConditionedTextureAdapter(dim=8)
    with torch.no_grad():
        adapter.residual_proj.weight.fill_(0.05)
        adapter.residual_proj.bias.zero_()

    content = torch.randn(2, 11, 8)
    style_a = torch.randn(2, 7, 8)
    style_b = style_a + 2.0

    out_a = adapter(content, style_a)
    out_b = adapter(content, style_b)

    assert not torch.allclose(out_a, out_b)


def test_transmodule_adapter_is_optional_and_preserves_shape():
    config = TransModule_Config(
        nlayer=1,
        d_model=8,
        nhead=2,
        mlp_ratio=2,
        qkv_bias=False,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    )
    without_adapter = TransModule(config)
    with_adapter = TransModule(config, use_style_adapter=True)
    content = torch.randn(2, 9, 8)
    style = torch.randn(2, 7, 8)

    assert without_adapter.style_adapter is None
    assert isinstance(with_adapter.style_adapter, StyleConditionedTextureAdapter)
    assert with_adapter(content, style).shape == content.shape


def test_load_transmodule_state_allows_warm_starting_adapter_from_old_state():
    config = TransModule_Config(
        nlayer=1,
        d_model=8,
        nhead=2,
        mlp_ratio=2,
        qkv_bias=False,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    )
    old_module = TransModule(config)
    new_module = TransModule(config, use_style_adapter=True)

    result = load_transmodule_state(
        new_module,
        old_module.state_dict(),
        allow_partial=True,
    )

    assert any(key.startswith("style_adapter.") for key in result.missing_keys)


def test_load_network_weights_warm_starts_model_without_optimizer_state():
    config = TransModule_Config(
        nlayer=1,
        d_model=8,
        nhead=2,
        mlp_ratio=2,
        qkv_bias=False,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True,
    )
    old_network = nn.Module()
    old_network.encoder = nn.Linear(8, 8)
    old_network.decoder = nn.Linear(8, 8)
    old_network.transModule = TransModule(config)
    new_network = nn.Module()
    new_network.encoder = nn.Linear(8, 8)
    new_network.decoder = nn.Linear(8, 8)
    new_network.transModule = TransModule(config, use_style_adapter=True)
    checkpoint = {
        "encoder": old_network.encoder.state_dict(),
        "decoder": old_network.decoder.state_dict(),
        "transModule": old_network.transModule.state_dict(),
        "optimizer": {"old": "state"},
    }

    result = load_network_weights(
        new_network,
        checkpoint,
        allow_partial_transmodule=True,
    )

    assert any(key.startswith("style_adapter.") for key in result.missing_keys)
