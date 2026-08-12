from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

from metrics import (
    _resolve_manifest_output_path,
    _resolve_manifest_reference_path,
    dominant_color_count,
    structural_similarity,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_lpips(device: torch.device):
    try:
        import lpips  # type: ignore
    except ImportError as exc:
        repo_lpips = Path(__file__).resolve().parents[1] / "external" / "PerceptualSimilarity"
        if not (repo_lpips / "lpips").exists():
            raise RuntimeError(
                "LPIPS package is not installed and external/PerceptualSimilarity "
                "was not found. Install `lpips` or clone richzhang/PerceptualSimilarity "
                "before running with --include_lpips."
            ) from exc
        sys.path.insert(0, str(repo_lpips))
        import lpips  # type: ignore
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def lpips_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def image_to_tensor(path: Path, transform: T.Compose, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB")).unsqueeze(0).to(device)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            transform_input=False,
            aux_logits=True,
        )
        self.features = nn.Sequential(
            model.Conv2d_1a_3x3,
            model.Conv2d_2a_3x3,
            model.Conv2d_2b_3x3,
            model.maxpool1,
            model.Conv2d_3b_1x1,
            model.Conv2d_4a_3x3,
            model.maxpool2,
            model.Mixed_5b,
            model.Mixed_5c,
            model.Mixed_5d,
            model.Mixed_6a,
            model.Mixed_6b,
            model.Mixed_6c,
            model.Mixed_6d,
            model.Mixed_6e,
            model.Mixed_7a,
            model.Mixed_7b,
            model.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


def fid_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(
    paths: list[Path],
    model: nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    transform = fid_transform(image_size)
    features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch = torch.cat(
            [image_to_tensor(path, transform, device) for path in batch_paths],
            dim=0,
        )
        features.append(model(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    diff = mu_real - mu_fake
    sqrt_result = linalg.sqrtm(sigma_real @ sigma_fake)
    covmean = sqrt_result[0] if isinstance(sqrt_result, tuple) else sqrt_result
    if not np.isfinite(covmean).all():
        eps = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + eps) @ (sigma_fake + eps))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


def summarize_method(
    manifest_csv: Path,
    output_dir: Path,
    output_csv: Path,
    method: str,
    device: torch.device,
    include_lpips: bool,
    image_size: int,
) -> None:
    rows = read_manifest(manifest_csv)
    lpips_model = load_lpips(device) if include_lpips else None
    lpips_tf = lpips_transform(image_size)

    out_rows = []
    for row in rows:
        pair_id = row["pair_id"]
        output_path = _resolve_manifest_output_path(output_dir, pair_id)
        content_path = _resolve_manifest_reference_path(manifest_csv, row["content_path"])
        style_path = _resolve_manifest_reference_path(manifest_csv, row["style_path"])
        with Image.open(output_path) as out_img, Image.open(content_path) as content_img:
            out_img = out_img.convert("RGB")
            content_img = content_img.convert("RGB")
            result = {
                "pair_id": pair_id,
                "method": method,
                "output_path": output_path.as_posix(),
                "content_path": content_path.as_posix(),
                "style_path": style_path.as_posix(),
                "content_ssim": structural_similarity(out_img, content_img),
                "dominant_color_count": dominant_color_count(out_img),
            }

        if lpips_model is not None:
            out_tensor = image_to_tensor(output_path, lpips_tf, device)
            content_tensor = image_to_tensor(content_path, lpips_tf, device)
            style_tensor = image_to_tensor(style_path, lpips_tf, device)
            result["lpips_content"] = float(lpips_model(out_tensor, content_tensor).item())
            result["lpips_style"] = float(lpips_model(out_tensor, style_tensor).item())
        else:
            result["lpips_content"] = ""
            result["lpips_style"] = ""
        out_rows.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair_id",
        "method",
        "output_path",
        "content_path",
        "style_path",
        "content_ssim",
        "lpips_content",
        "lpips_style",
        "dominant_color_count",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def summarize_fid(
    fake_dirs: dict[str, Path],
    real_style_dir: Path,
    output_csv: Path,
    device: torch.device,
    image_size: int,
    batch_size: int,
) -> None:
    real_paths = image_files(real_style_dir)
    if len(real_paths) < 2:
        raise ValueError("FID requires at least two real style images.")

    model = InceptionFeatureExtractor().to(device)
    model.eval()
    real_features = extract_features(real_paths, model, device, image_size, batch_size)

    rows = []
    for method, fake_dir in fake_dirs.items():
        fake_paths = image_files(fake_dir)
        if len(fake_paths) < 2:
            raise ValueError(f"FID requires at least two generated images for {method}.")
        fake_features = extract_features(fake_paths, model, device, image_size, batch_size)
        rows.append({
            "method": method,
            "n_fake": len(fake_paths),
            "n_real": len(real_paths),
            "fid": calculate_fid(real_features, fake_features),
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n_fake", "n_real", "fid"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    per_image = subparsers.add_parser("per-image")
    per_image.add_argument("--manifest_csv", type=Path, required=True)
    per_image.add_argument("--output_dir", type=Path, required=True)
    per_image.add_argument("--output_csv", type=Path, required=True)
    per_image.add_argument("--method", type=str, required=True)
    per_image.add_argument("--include_lpips", action="store_true")
    per_image.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    per_image.add_argument("--image_size", type=int, default=224)

    fid = subparsers.add_parser("fid")
    fid.add_argument("--real_style_dir", type=Path, required=True)
    fid.add_argument("--output_csv", type=Path, required=True)
    fid.add_argument("--method_dir", action="append", nargs=2, metavar=("METHOD", "DIR"), required=True)
    fid.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    fid.add_argument("--image_size", type=int, default=299)
    fid.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.mode == "per-image":
        summarize_method(
            manifest_csv=args.manifest_csv,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            method=args.method,
            device=device,
            include_lpips=args.include_lpips,
            image_size=args.image_size,
        )
    elif args.mode == "fid":
        summarize_fid(
            fake_dirs={method: Path(directory) for method, directory in args.method_dir},
            real_style_dir=args.real_style_dir,
            output_csv=args.output_csv,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )
    else:
        raise AssertionError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
