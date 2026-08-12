import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "data" / "content" / "test"
STYLE_DIR = ROOT / "data" / "ChinesePaperCutting"
OUT_DIR = ROOT / "data" / "manifests"
OUT_CSV = OUT_DIR / "test_pairs.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(directory: Path, pattern: str):
    return [
        path for path in directory.glob(pattern)
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def main() -> None:
    content = sorted(image_files(CONTENT_DIR, "content_test_*"))
    styles = sorted(image_files(STYLE_DIR, "style_*"), key=lambda p: (len(p.stem), p.name))
    selected_styles = styles[:10]

    assert len(content) > 0, "Expected at least one test image"
    assert len(selected_styles) == 10, f"Expected at least 10 style images, got {len(selected_styles)}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "content_path", "style_path"])
        writer.writeheader()
        pair_id = 0
        for c in content:
            for s in selected_styles:
                writer.writerow({
                    "pair_id": f"pair_{pair_id:04d}",
                    "content_path": c.relative_to(ROOT).as_posix(),
                    "style_path": s.relative_to(ROOT).as_posix(),
                })
                pair_id += 1
    print(f"wrote={OUT_CSV}")
    print(f"pairs={len(content) * len(selected_styles)}")


if __name__ == "__main__":
    main()
