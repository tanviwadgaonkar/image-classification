"""Dataset loading and single-image preprocessing.

Training data is a folder per class::

    data/my_dataset/
      cat/  001.jpg 002.png ...
      dog/  ...

Images are streamed from disk with ``tf.data`` (never all loaded into RAM),
labels are integer indices into the sorted class names, and pixel scaling and
augmentation happen *inside the model* (see model.py), so training and
inference cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def load_and_preprocess_image(image_path: str | Path, image_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Read one image as an RGB float32 array of shape (H, W, 3) in [0, 255].

    Raises ValueError for missing or unreadable files (cv2.imread used to
    return None here and crash later in resize).
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize((image_size[1], image_size[0]), Image.BILINEAR)
            return np.asarray(img, dtype=np.float32)
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"cannot read image {image_path}: {exc}") from exc


def class_names(data_dir: str | Path) -> list[str]:
    """Sorted class folder names: the label index of a class is its position here."""
    names = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    if len(names) < 2:
        raise ValueError(f"{data_dir} needs at least two class folders, found {names}")
    return names


def load_datasets(
    data_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Return ``(train_ds, val_ds, class_names)`` as batched ``tf.data`` datasets.

    The split is a seeded, stratification-free random split of files, the same
    on every run, so validation images are never trained on.
    """
    import tensorflow as tf

    names = class_names(data_dir)
    common = dict(
        labels="inferred", label_mode="int", class_names=names, image_size=image_size,
        batch_size=batch_size, seed=seed, validation_split=validation_split,
    )
    train = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE), names


def load_test_dataset(test_dir: str | Path, names: list[str], image_size=(224, 224), batch_size: int = 32):
    """A held-out folder dataset, labelled with the *training* class order."""
    import tensorflow as tf

    found = class_names(test_dir)
    if found != names:
        raise ValueError(f"test classes {found} do not match training classes {names}")
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int", class_names=names,
        image_size=image_size, batch_size=batch_size, shuffle=False,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
