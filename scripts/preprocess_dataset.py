from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
STYLE_PREFIX = "style"
CONTENT_TRAIN_PREFIX = "content_train"
CONTENT_TEST_PREFIX = "content_test"
RED = np.array([220, 0, 0], dtype=np.uint8)
WHITE = np.array([255, 255, 255], dtype=np.uint8)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0, path.name.lower()


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def binarize_style_image(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = rgb.mean(axis=2)
    red_score = r - 0.5 * (g + b)
    red_mask = (red_score > 18.0) & (r > 80.0) & (brightness < 245.0)
    dark_mask = brightness < 128.0
    mask = red_mask | dark_mask
    output = np.empty_like(rgb, dtype=np.uint8)
    output[mask] = RED
    output[~mask] = WHITE
    return Image.fromarray(output, "RGB")


def normalize_style_image(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    saturation = (max_c - min_c) / np.maximum(max_c, 1.0)
    red_dominance = r - np.maximum(g, b)
    red_mask = (r > 80.0) & (red_dominance > 20.0) & (saturation > 0.18)

    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    output = np.empty_like(rgb)

    red_luma = np.clip(0.62 + 0.34 * luminance, 0.45, 0.96)
    output[..., 0] = np.where(red_mask, 220.0 * red_luma / 0.86, rgb[..., 0])
    output[..., 1] = np.where(red_mask, 10.0 * (1.0 - red_luma), rgb[..., 1])
    output[..., 2] = np.where(red_mask, 10.0 * (1.0 - red_luma), rgb[..., 2])

    non_red = ~red_mask
    bg_gray = np.clip(luminance * 255.0, 0.0, 255.0)
    bg_clean = np.where(bg_gray < 180.0, 235.0 + 20.0 * (bg_gray / 180.0), 248.0 + 7.0 * ((bg_gray - 180.0) / 75.0))
    bg_clean = np.clip(bg_clean, 235.0, 255.0)
    for channel in range(3):
        output[..., channel] = np.where(non_red, bg_clean, output[..., channel])

    output = np.clip(output, 0, 255).astype(np.uint8)
    return Image.fromarray(output, "RGB")


def enhance_content_image(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.28)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.45)
    edge = rgb.filter(ImageFilter.FIND_EDGES).convert("L")
    edge = ImageEnhance.Contrast(edge).enhance(1.5)
    edge_rgb = Image.merge("RGB", (edge, edge, edge))
    rgb = Image.blend(rgb, edge_rgb, 0.08)
    return rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=115, threshold=3)).convert("RGB")


def preprocess_split(
    source_dir: Path,
    target_dir: Path,
    prefix: str,
    processor: Callable[[Image.Image], Image.Image],
    mapping_csv: Path,
    output_extension: str = ".jpg",
) -> list[dict[str, str]]:
    source_paths = iter_images(source_dir)
    if not source_paths:
        raise FileNotFoundError(f"No supported images found in {source_dir}")

    temp_dir = target_dir.with_name(f"{target_dir.name}.__preprocess_tmp__")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    rows: list[dict[str, str]] = []
    try:
        for index, source_path in enumerate(source_paths, start=1):
            new_name = f"{prefix}_{index:04d}{output_extension}"
            target_path = temp_dir / new_name
            with Image.open(source_path) as image:
                processed = processor(image)
                if output_extension.lower() == ".png":
                    processed.save(target_path, format="PNG")
                else:
                    processed.save(target_path, format="JPEG", quality=95, subsampling=0)
                width, height = processed.size
            rows.append(
                {
                    "split": prefix,
                    "old_name": source_path.name,
                    "new_name": new_name,
                    "source_path": str(source_path),
                    "target_path": str(target_dir / new_name),
                    "width": str(width),
                    "height": str(height),
                    "processor": processor.__name__,
                }
            )

        if target_dir.exists():
            shutil.rmtree(target_dir)
        temp_dir.rename(target_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise

    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    write_mapping_rows(mapping_csv, rows, append=mapping_csv.exists())
    return rows


def write_mapping_rows(mapping_csv: Path, rows: list[dict[str, str]], append: bool) -> None:
    fieldnames = [
        "split",
        "old_name",
        "new_name",
        "source_path",
        "target_path",
        "width",
        "height",
        "processor",
    ]
    with mapping_csv.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        writer.writerows(rows)


def backup_data(data_dir: Path, backup_root: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root or data_dir / f"_backup_before_preprocess_{timestamp}"
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    for relative in (Path("ChinesePaperCutting"), Path("content") / "train", Path("content") / "test"):
        source = data_dir / relative
        if source.exists():
            shutil.copytree(source, backup_dir / relative)
    return backup_dir


def preprocess_dataset(data_dir: Path, make_backup: bool = True) -> Path:
    data_dir = data_dir.resolve()
    if make_backup:
        backup_data(data_dir)
    mapping_csv = data_dir / "preprocess_mapping.csv"
    if mapping_csv.exists():
        mapping_csv.unlink()
    preprocess_split(
        source_dir=data_dir / "ChinesePaperCutting",
        target_dir=data_dir / "ChinesePaperCutting",
        prefix=STYLE_PREFIX,
        processor=normalize_style_image,
        mapping_csv=mapping_csv,
        output_extension=".jpg",
    )
    preprocess_split(
        source_dir=data_dir / "content" / "train",
        target_dir=data_dir / "content" / "train",
        prefix=CONTENT_TRAIN_PREFIX,
        processor=enhance_content_image,
        mapping_csv=mapping_csv,
    )
    preprocess_split(
        source_dir=data_dir / "content" / "test",
        target_dir=data_dir / "content" / "test",
        prefix=CONTENT_TEST_PREFIX,
        processor=enhance_content_image,
        mapping_csv=mapping_csv,
    )
    return mapping_csv


def sync_data(source_data_dir: Path, target_data_dir: Path) -> None:
    for relative in (Path("ChinesePaperCutting"), Path("content") / "train", Path("content") / "test"):
        source = source_data_dir / relative
        target = target_data_dir / relative
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    for file_name in ("preprocess_mapping.csv",):
        source_file = source_data_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, target_data_dir / file_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--sync_to", type=Path, default=ROOT / "data")
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument("--no_sync", action="store_true")
    args = parser.parse_args()

    mapping_csv = preprocess_dataset(args.data_dir, make_backup=not args.no_backup)
    print(f"mapping_csv={mapping_csv}")
    if not args.no_sync:
        sync_data(args.data_dir.resolve(), args.sync_to.resolve())
        print(f"synced_to={args.sync_to.resolve()}")


if __name__ == "__main__":
    main()
