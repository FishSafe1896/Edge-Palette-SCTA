from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_REPOS = [
    ROOT / "src" / "ChinesePaperCutting",
    ROOT / "src" / "baselines" / "StyTR-2",
    ROOT / "src" / "baselines" / "CAST_pytorch",
    ROOT / "src" / "baselines" / "S2WAT",
]


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    missing = [str(p) for p in EXPECTED_REPOS if not p.exists()]
    train_count = count_images(ROOT / "data" / "content" / "train")
    test_count = count_images(ROOT / "data" / "content" / "test")
    style_count = count_images(ROOT / "data" / "ChinesePaperCutting")

    print(f"content_train={train_count}")
    print(f"content_test={test_count}")
    print(f"style_images={style_count}")
    print(f"missing_repos={missing}")

    assert train_count > 0, f"Expected train images, got {train_count}"
    assert test_count > 0, f"Expected test images, got {test_count}"
    assert style_count >= 10, f"Expected at least 10 style images, got {style_count}"
    assert not missing, f"Missing repositories: {missing}"


if __name__ == "__main__":
    main()
