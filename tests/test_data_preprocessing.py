import numpy as np
import pytest

from src.data_preprocessing import class_names, load_and_preprocess_image


def test_image_is_resized_rgb_float_in_pixel_range(shapes):
    img = load_and_preprocess_image(shapes["train"] / "circle" / "000.png", (48, 40))
    assert img.shape == (48, 40, 3)
    assert img.dtype == np.float32
    assert 0 <= img.min() and img.max() <= 255  # scaling to [0, 1] happens inside the model


def test_grayscale_image_becomes_three_channels(tmp_path):
    from PIL import Image

    Image.new("L", (10, 10), 128).save(tmp_path / "g.png")
    assert load_and_preprocess_image(tmp_path / "g.png", (8, 8)).shape == (8, 8, 3)


def test_unreadable_or_missing_image_raises_value_error(tmp_path):
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot read image"):
        load_and_preprocess_image(tmp_path / "bad.jpg")
    with pytest.raises(ValueError):
        load_and_preprocess_image(tmp_path / "missing.jpg")


def test_class_names_are_sorted_folder_names(shapes):
    assert class_names(shapes["train"]) == ["circle", "square"]


def test_single_class_dataset_is_rejected(tmp_path):
    (tmp_path / "only").mkdir()
    with pytest.raises(ValueError, match="at least two"):
        class_names(tmp_path)
