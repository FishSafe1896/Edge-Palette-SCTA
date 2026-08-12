import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePriorBranch(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect"),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return self.proj(x * (1.0 + edge))


class MultiScaleFrequencyBranch(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.local3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect")
        self.local5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode="reflect")
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffted = torch.fft.rfft2(x, norm="ortho")
        amp = torch.abs(ffted)
        phase = torch.angle(ffted)
        enhanced = torch.polar(torch.log1p(amp), phase)
        global_freq = torch.fft.irfft2(enhanced, s=x.shape[-2:], norm="ortho")
        return self.fuse(torch.cat([self.local3(x), self.local5(x), global_freq], dim=1))


class LightSymmetryGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = max(1, dim // 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flipped = torch.flip(x, dims=[-1])
        mismatch = torch.mean(torch.abs(x - flipped), dim=1, keepdim=True)
        energy = torch.mean(torch.abs(x), dim=1, keepdim=True)
        flipped_energy = torch.mean(torch.abs(flipped), dim=1, keepdim=True)
        eps = torch.finfo(x.dtype).eps
        symmetry_hint = 1.0 - mismatch / (energy + flipped_energy + eps)
        symmetry_hint = symmetry_hint.clamp(0.0, 1.0)
        return x * (1.0 + 0.05 * self.gate(x) * symmetry_hint)


class PaperCutPriorEnhancementBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.edge = EdgePriorBranch(dim)
        self.freq = MultiScaleFrequencyBranch(dim)
        self.symmetry_gate = LightSymmetryGate(dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.set_mode("full")

    def set_mode(self, mode: str) -> None:
        if mode not in {"off", "edge", "freq", "full"}:
            raise ValueError(f"Unsupported PPEB mode: {mode}")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x

        edge = self.edge(x) if self.mode in {"edge", "full"} else torch.zeros_like(x)
        freq = self.freq(x) if self.mode in {"freq", "full"} else torch.zeros_like(x)
        out = self.fuse(torch.cat([edge, freq], dim=1))
        if self.mode == "full":
            out = self.symmetry_gate(out)
        return x + out
