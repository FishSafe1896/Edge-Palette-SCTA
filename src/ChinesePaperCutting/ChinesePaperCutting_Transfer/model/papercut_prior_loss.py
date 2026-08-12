import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperCutPriorLoss(nn.Module):
    def __init__(
        self,
        edge_weight: float = 1.0,
        freq_weight: float = 0.1,
        palette_weight: float = 0.0,
        smooth_weight: float = 0.0,
        texture_weight: float = 0.0,
    ):
        super().__init__()
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight
        self.palette_weight = palette_weight
        self.smooth_weight = smooth_weight
        self.texture_weight = texture_weight

    def _gray(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1, keepdim=True)

    def _edge(self, x: torch.Tensor) -> torch.Tensor:
        gray = self._gray(x)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def _normalized_edge(self, x: torch.Tensor) -> torch.Tensor:
        edge = self._edge(x)
        scale = edge.mean(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-6)
        return edge / scale

    def _normalized_edge_map(self, edge: torch.Tensor) -> torch.Tensor:
        scale = edge.mean(dim=(-2, -1), keepdim=True).detach().clamp_min(1e-6)
        return edge / scale

    def _morphological_edge(self, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.clamp(0.0, 1.0)
        dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp_min(0.0)

    def _paper_mask(self, output: torch.Tensor) -> torch.Tensor:
        palette = output.new_tensor(
            [
                [0.86, 0.02, 0.02],
                [1.00, 1.00, 1.00],
            ]
        ).view(1, 2, 3, 1, 1)
        distance = (output.unsqueeze(1) - palette).pow(2).mean(dim=2)
        return torch.softmax(-distance / 0.05, dim=1)[:, :1]

    def _output_edge(self, output: torch.Tensor) -> torch.Tensor:
        mask = self._paper_mask(output)
        smoothed = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        return self._normalized_edge_map(self._morphological_edge(smoothed))

    def _clean_edge_target(self, content: torch.Tensor) -> torch.Tensor:
        gray = self._gray(content)
        flat = gray.flatten(2)
        low = torch.quantile(flat, 0.05, dim=2, keepdim=True).view(-1, 1, 1, 1)
        high = torch.quantile(flat, 0.95, dim=2, keepdim=True).view(-1, 1, 1, 1)
        stretched = ((gray - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
        threshold = stretched.mean(dim=(-2, -1), keepdim=True)
        binary = (stretched >= threshold).to(dtype=content.dtype)
        dilated = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
        return self._normalized_edge_map(self._morphological_edge(closed))

    def _frequency_profile(self, x: torch.Tensor) -> torch.Tensor:
        gray = self._gray(x)
        amp = torch.log1p(torch.abs(torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"))))
        height, width = gray.shape[-2:]
        y = torch.linspace(-1.0, 1.0, height, dtype=x.dtype, device=x.device)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=x.dtype, device=x.device)
        yy, xx = torch.meshgrid(y, x_axis, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy)
        bands = (
            radius <= 0.18,
            (radius > 0.18) & (radius <= 0.45),
            radius > 0.45,
        )
        values = []
        for mask in bands:
            band = amp[..., mask]
            values.append(band.mean(dim=-1))
        return torch.stack(values, dim=-1).squeeze(1)

    def _palette_loss(self, output: torch.Tensor) -> torch.Tensor:
        palette = output.new_tensor(
            [
                [0.86, 0.02, 0.02],
                [1.00, 1.00, 1.00],
            ]
        ).view(1, 2, 3, 1, 1)
        distance = (output.unsqueeze(1) - palette).pow(2).mean(dim=2)
        return distance.min(dim=1).values.mean()

    def _smoothness_loss(self, output: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]).mean()
        dy = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]).mean()
        return dx + dy

    def _texture_profile(self, x: torch.Tensor) -> torch.Tensor:
        gray = self._gray(x)
        profiles = []
        current = gray
        for kernel_size in (3, 5, 9):
            if min(current.shape[-2:]) < kernel_size:
                continue
            blurred = F.avg_pool2d(
                current,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                count_include_pad=False,
            )
            high = current - blurred
            profiles.extend([
                high.abs().mean(dim=(-2, -1)),
                high.pow(2).mean(dim=(-2, -1)),
            ])
            if min(current.shape[-2:]) >= 16:
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
        return torch.cat(profiles, dim=1)

    def components(
        self, content: torch.Tensor, style: torch.Tensor, output: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        edge_loss = self.edge_weight * F.l1_loss(
            self._output_edge(output),
            self._clean_edge_target(content).detach(),
        )
        freq_loss = self.freq_weight * F.l1_loss(
            self._frequency_profile(output),
            self._frequency_profile(style).detach(),
        )
        palette_loss = self.palette_weight * self._palette_loss(output)
        smooth_loss = self.smooth_weight * self._smoothness_loss(output)
        texture_loss = self.texture_weight * F.l1_loss(
            self._texture_profile(output),
            self._texture_profile(style).detach(),
        )
        total = edge_loss + freq_loss + palette_loss + smooth_loss + texture_loss
        return {
            "edge": edge_loss,
            "freq": freq_loss,
            "palette": palette_loss,
            "smooth": smooth_loss,
            "texture": texture_loss,
            "total": total,
        }

    def forward(self, content: torch.Tensor, style: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        return self.components(content, style, output)["total"]
