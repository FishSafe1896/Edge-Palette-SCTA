import csv
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.preprocess_dataset import (
    CONTENT_TEST_PREFIX,
    CONTENT_TRAIN_PREFIX,
    STYLE_PREFIX,
    binarize_style_image,
    enhance_content_image,
    normalize_style_image,
    preprocess_split,
)


def _save_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), "RGB").save(path)


def test_binarize_style_image_outputs_only_red_and_white():
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, :4] = [180, 20, 20]
    array[:, 4:] = [245, 245, 245]
    image = Image.fromarray(array, "RGB")

    output = binarize_style_image(image)

    unique = {tuple(pixel) for pixel in np.asarray(output).reshape(-1, 3)}
    assert unique == {(220, 0, 0), (255, 255, 255)}


def test_normalize_style_image_preserves_dark_non_red_holes():
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array[:, :3] = [180, 20, 20]
    array[:, 3:5] = [35, 35, 35]
    array[:, 5:] = [235, 235, 235]
    image = Image.fromarray(array, "RGB")

    output = normalize_style_image(image)
    out = np.asarray(output)

    assert out[:, :3, 0].mean() > out[:, :3, 1].mean() * 3
    assert out[:, 3:5].mean() > 220
    assert out[:, 5:].mean() > 245


def test_enhance_content_image_keeps_rgb_and_improves_contrast():
    gradient = np.tile(np.linspace(96, 160, 32, dtype=np.uint8), (32, 1))
    image = Image.fromarray(np.stack([gradient, gradient, gradient], axis=2), "RGB")

    output = enhance_content_image(image)

    assert output.mode == "RGB"
    assert np.asarray(output.convert("L")).std() > np.asarray(image.convert("L")).std()


def test_preprocess_split_renames_images_and_writes_mapping(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    mapping = tmp_path / "mapping.csv"
    _save_rgb(source / "b.jpg", np.full((6, 6, 3), 255, dtype=np.uint8))
    _save_rgb(source / "a.jpg", np.full((6, 6, 3), [180, 20, 20], dtype=np.uint8))

    rows = preprocess_split(
        source_dir=source,
        target_dir=target,
        prefix=STYLE_PREFIX,
        processor=binarize_style_image,
        mapping_csv=mapping,
    )

    assert [row["new_name"] for row in rows] == ["style_0001.jpg", "style_0002.jpg"]
    assert sorted(path.name for path in target.glob("*.jpg")) == ["style_0001.jpg", "style_0002.jpg"]
    with mapping.open("r", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert [row["split"] for row in csv_rows] == [STYLE_PREFIX, STYLE_PREFIX]
    assert {CONTENT_TRAIN_PREFIX, CONTENT_TEST_PREFIX}
