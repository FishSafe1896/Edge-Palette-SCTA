import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.metrics import (
    dominant_color_count,
    edge_density,
    high_low_frequency_ratio,
    structural_similarity,
    summarize_directory,
    summarize_manifest_outputs,
)


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_edge_density_blank_image_is_zero():
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    assert edge_density(img) == 0.0


def test_dominant_color_count_single_color_is_one():
    img = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8))
    assert dominant_color_count(img, bins=4) == 1


def test_frequency_ratio_is_finite_and_checkerboard_exceeds_blank():
    blank = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    checker = np.indices((32, 32)).sum(axis=0) % 2
    checker = Image.fromarray((checker * 255).astype(np.uint8)).convert("RGB")

    blank_ratio = high_low_frequency_ratio(blank)
    checker_ratio = high_low_frequency_ratio(checker)

    assert np.isfinite(blank_ratio)
    assert np.isfinite(checker_ratio)
    assert blank_ratio >= 0.0
    assert checker_ratio > blank_ratio


def test_structural_similarity_identical_image_is_one():
    img = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8))

    assert structural_similarity(img, img) == 1.0


def test_summarize_directory_writes_rows_and_headers(tmp_path):
    input_dir = tmp_path / "inputs"
    output_csv = tmp_path / "metrics" / "directory.csv"
    _write_image(
        input_dir / "a.jpg",
        np.zeros((16, 16, 3), dtype=np.uint8),
    )
    _write_image(
        input_dir / "b.jpg",
        np.full((16, 16, 3), 255, dtype=np.uint8),
    )

    summarize_directory(input_dir, output_csv)

    rows = _read_csv_rows(output_csv)
    assert output_csv.parent.exists()
    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "image",
        "edge_density",
        "dominant_color_count",
        "high_low_frequency_ratio",
    }


def test_summarize_manifest_outputs_and_cli_smoke(tmp_path):
    manifest_csv = tmp_path / "manifests" / "pairs.csv"
    content_dir = tmp_path / "content"
    output_dir = tmp_path / "outputs"
    output_csv = tmp_path / "metrics" / "manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_image(
        content_dir / "a.jpg",
        np.full((16, 16, 3), 64, dtype=np.uint8),
    )
    _write_image(
        content_dir / "b.jpg",
        np.full((16, 16, 3), 192, dtype=np.uint8),
    )
    _write_image(
        output_dir / "pair_0001.jpg",
        np.full((16, 16, 3), 64, dtype=np.uint8),
    )
    _write_image(
        output_dir / "pair_0002_output.jpg",
        np.full((16, 16, 3), 192, dtype=np.uint8),
    )

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "content_path", "style_path"])
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "pair_0001",
                "content_path": str(content_dir / "a.jpg"),
                "style_path": "style/a.jpg",
            }
        )
        writer.writerow(
            {
                "pair_id": "pair_0002",
                "content_path": str(content_dir / "b.jpg"),
                "style_path": "style/b.jpg",
            }
        )

    summarize_manifest_outputs(manifest_csv, output_dir, output_csv, method="ppeb")

    rows = _read_csv_rows(output_csv)
    assert output_csv.parent.exists()
    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        "pair_id",
        "method",
        "output_path",
        "content_ssim",
        "edge_density",
        "dominant_color_count",
        "high_low_frequency_ratio",
    }
    assert {row["pair_id"] for row in rows} == {"pair_0001", "pair_0002"}
    assert all(row["method"] == "ppeb" for row in rows)
    assert all(float(row["content_ssim"]) == 1.0 for row in rows)

    cli_output_csv = tmp_path / "metrics" / "cli_manifest.csv"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "metrics.py"),
            "--manifest_csv",
            str(manifest_csv),
            "--output_dir",
            str(output_dir),
            "--output_csv",
            str(cli_output_csv),
            "--method",
            "cli-smoke",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    cli_rows = _read_csv_rows(cli_output_csv)
    assert len(cli_rows) == 2
    assert all(row["method"] == "cli-smoke" for row in cli_rows)
