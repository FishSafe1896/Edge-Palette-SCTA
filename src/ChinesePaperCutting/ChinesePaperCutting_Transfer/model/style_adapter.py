import torch
import torch.nn as nn


class StyleConditionedTextureAdapter(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.25, alpha: float = 0.1):
        super().__init__()
        hidden_dim = max(16, int(dim * hidden_ratio))
        self.dim = dim
        self.alpha = alpha
        self.descriptor_proj = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim * 3),
        )
        self.residual_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)

    def _style_descriptor(self, style_feature: torch.Tensor) -> torch.Tensor:
        style_mean = style_feature.mean(dim=1)
        style_std = style_feature.var(dim=1, unbiased=False).add(1e-6).sqrt()
        if style_feature.size(1) > 1:
            texture = (style_feature[:, 1:, :] - style_feature[:, :-1, :]).abs().mean(dim=1)
        else:
            texture = torch.zeros_like(style_mean)
        return torch.cat([style_mean, style_std, texture], dim=1)

    def forward(self, content_feature: torch.Tensor, style_feature: torch.Tensor) -> torch.Tensor:
        content_mean = content_feature.mean(dim=1, keepdim=True)
        content_std = content_feature.var(dim=1, unbiased=False, keepdim=True).add(1e-6).sqrt()
        normalized = (content_feature - content_mean) / content_std

        style_descriptor = self._style_descriptor(style_feature)
        gamma, beta, gate = self.descriptor_proj(style_descriptor).chunk(3, dim=1)
        gamma = 1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)
        beta = 0.1 * torch.tanh(beta).unsqueeze(1)
        gate = torch.sigmoid(gate).unsqueeze(1)

        modulated = normalized * gamma + beta
        residual = self.residual_proj(modulated)
        return content_feature + self.alpha * gate * residual
