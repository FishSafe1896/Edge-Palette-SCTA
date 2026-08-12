from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _natural_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    number = int(digits) if digits else 0
    return number, path.name


def _image_paths(directory: Path) -> list[Path]:
    return [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def build_full_grid_manifest(content_dir: Path, style_dir: Path, output_csv: Path) -> int:
    content_paths = sorted(_image_paths(content_dir), key=_natural_key)
    style_paths = sorted(_image_paths(style_dir), key=_natural_key)

    if not content_paths:
        raise FileNotFoundError(f"No supported content images found in {content_dir}")
    if not style_paths:
        raise FileNotFoundError(f"No supported style images found in {style_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "content_id", "style_id", "content_path", "style_path"],
        )
        writer.writeheader()
        for content_path in content_paths:
            content_id = content_path.stem
            for style_path in style_paths:
                style_id = style_path.stem
                writer.writerow(
                    {
                        "pair_id": f"{content_id}__{style_id}",
                        "content_id": content_id,
                        "style_id": style_id,
                        "content_path": content_path.relative_to(ROOT).as_posix(),
                        "style_path": style_path.relative_to(ROOT).as_posix(),
                    }
                )
    return len(content_paths) * len(style_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=Path, default=ROOT / "data" / "content" / "test")
    parser.add_argument("--style_dir", type=Path, default=ROOT / "data" / "ChinesePaperCutting")
    parser.add_argument("--output_csv", type=Path, default=ROOT / "data" / "manifests" / "all_content_style_pairs.csv")
    args = parser.parse_args()

    pair_count = build_full_grid_manifest(args.content_dir, args.style_dir, args.output_csv)
    print(f"wrote={args.output_csv}")
    print(f"pairs={pair_count}")


if __name__ == "__main__":
    main()
