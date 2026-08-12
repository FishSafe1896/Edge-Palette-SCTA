import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src" / "ChinesePaperCutting" / "ChinesePaperCutting_Transfer"
sys.path.insert(0, str(MODEL_DIR))

from model.papercut_prior_loss import PaperCutPriorLoss


def test_pcp_loss_is_scalar_and_finite():
    loss_fn = PaperCutPriorLoss(edge_weight=1.0, freq_weight=1.0)
    content = torch.rand(2, 3, 32, 32)
    style = torch.rand(2, 3, 32, 32)
    output = torch.rand(2, 3, 32, 32, requires_grad=True)
    loss = loss_fn(content, style, output)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_pcp_loss_backpropagates_to_output():
    loss_fn = PaperCutPriorLoss(edge_weight=1.0, freq_weight=1.0)
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    output = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss_fn(content, style, output).backward()
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()


def test_pcp_loss_reports_weighted_components():
    loss_fn = PaperCutPriorLoss(
        edge_weight=1.0,
        freq_weight=0.5,
        palette_weight=0.25,
        smooth_weight=0.1,
        texture_weight=0.2,
    )
    content = torch.rand(1, 3, 32, 32)
    style = torch.rand(1, 3, 32, 32)
    output = torch.rand(1, 3, 32, 32, requires_grad=True)

    components = loss_fn.components(content, style, output)

    assert set(components) == {"edge", "freq", "palette", "smooth", "texture", "total"}
    for value in components.values():
        assert value.ndim == 0
        assert torch.isfinite(value)
    expected_total = (
        components["edge"]
        + components["freq"]
        + components["palette"]
        + components["smooth"]
        + components["texture"]
    )
    assert torch.isclose(components["total"], expected_total)


def test_palette_component_prefers_paper_cut_colors():
    loss_fn = PaperCutPriorLoss(
        edge_weight=0.0,
        freq_weight=0.0,
        palette_weight=1.0,
        smooth_weight=0.0,
    )
    content = torch.rand(1, 3, 16, 16)
    style = torch.rand(1, 3, 16, 16)
    palette_output = torch.zeros(1, 3, 16, 16)
    palette_output[:, 0, :, :] = 0.86
    random_output = torch.rand(1, 3, 16, 16)

    palette_loss = loss_fn.components(content, style, palette_output)["palette"]
    random_loss = loss_fn.components(content, style, random_output)["palette"]

    assert palette_loss < random_loss


def test_palette_component_penalizes_pale_pink():
    loss_fn = PaperCutPriorLoss(
        edge_weight=0.0,
        freq_weight=0.0,
        palette_weight=1.0,
        smooth_weight=0.0,
    )
    content = torch.rand(1, 3, 16, 16)
    style = torch.rand(1, 3, 16, 16)
    red_output = torch.zeros(1, 3, 16, 16)
    red_output[:, 0, :, :] = 0.86
    white_output = torch.ones(1, 3, 16, 16)
    pink_output = torch.ones(1, 3, 16, 16)
    pink_output[:, 1, :, :] = 0.82
    pink_output[:, 2, :, :] = 0.88

    red_loss = loss_fn.components(content, style, red_output)["palette"]
    white_loss = loss_fn.components(content, style, white_output)["palette"]
    pink_loss = loss_fn.components(content, style, pink_output)["palette"]

    assert pink_loss > red_loss
    assert pink_loss > white_loss


def test_clean_edge_target_suppresses_low_contrast_background_noise():
    loss_fn = PaperCutPriorLoss(edge_weight=1.0, freq_weight=0.0)
    content = torch.full((1, 3, 32, 32), 0.45)
    content[:, :, 8:24, 8:24] = 0.9
    content[:, :, :8, :8] = 0.50
    content[:, :, 1:8:2, :8] = 0.40

    raw_edge = loss_fn._normalized_edge(content)
    clean_edge = loss_fn._clean_edge_target(content)

    noisy_background = (slice(None), slice(None), slice(1, 7), slice(1, 7))
    assert clean_edge[noisy_background].mean() < raw_edge[noisy_background].mean()
    assert clean_edge[:, :, 8:24, 8:24].max() > 0


def test_output_edge_uses_soft_red_white_mask_to_ignore_pale_texture():
    loss_fn = PaperCutPriorLoss(edge_weight=1.0, freq_weight=0.0)
    pale_texture = torch.ones(1, 3, 32, 32)
    pale_texture[:, 1, :, ::2] = 0.82
    pale_texture[:, 2, :, 1::2] = 0.88
    red_block = torch.ones(1, 3, 32, 32)
    red_block[:, 0, 8:24, 8:24] = 0.86
    red_block[:, 1, 8:24, 8:24] = 0.02
    red_block[:, 2, 8:24, 8:24] = 0.02

    pale_edge = loss_fn._morphological_edge(loss_fn._paper_mask(pale_texture))
    red_edge = loss_fn._morphological_edge(loss_fn._paper_mask(red_block))

    assert pale_edge.mean() < 0.05
    assert red_edge.mean() > pale_edge.mean() + 0.1


def test_edge_component_prefers_matching_red_white_output_boundary():
    loss_fn = PaperCutPriorLoss(edge_weight=1.0, freq_weight=0.0)
    content = torch.zeros(1, 3, 32, 32)
    content[:, :, 8:24, 8:24] = 1.0
    style = torch.rand(1, 3, 32, 32)
    matching_output = torch.ones(1, 3, 32, 32)
    matching_output[:, 0, 8:24, 8:24] = 0.86
    matching_output[:, 1, 8:24, 8:24] = 0.02
    matching_output[:, 2, 8:24, 8:24] = 0.02
    pale_texture = torch.ones(1, 3, 32, 32)
    pale_texture[:, 1, :, ::2] = 0.82
    pale_texture[:, 2, :, 1::2] = 0.88

    matching_loss = loss_fn.components(content, style, matching_output)["edge"]
    pale_loss = loss_fn.components(content, style, pale_texture)["edge"]

    assert matching_loss < pale_loss


def test_texture_component_prefers_style_high_frequency_statistics():
    loss_fn = PaperCutPriorLoss(
        edge_weight=0.0,
        freq_weight=0.0,
        palette_weight=0.0,
        smooth_weight=0.0,
        texture_weight=1.0,
    )
    content = torch.rand(1, 3, 32, 32)
    style = torch.zeros(1, 3, 32, 32)
    style[:, :, ::2, ::2] = 1.0
    style[:, :, 1::2, 1::2] = 1.0
    style_like_output = style.clone()
    smooth_output = style.mean().expand_as(style).clone()

    style_like_loss = loss_fn.components(content, style, style_like_output)["texture"]
    smooth_loss = loss_fn.components(content, style, smooth_output)["texture"]

    assert style_like_loss < smooth_loss
