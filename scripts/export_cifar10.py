"""Export CIFAR-10 as a folder-per-class dataset the pipeline can read.

    python scripts/export_cifar10.py --out data/cifar10 [--per-class 5000]

Writes data/cifar10/train/<class>/*.png (50,000 images) and
data/cifar10/test/<class>/*.png (10,000 images, the official test split,
never used for training or model selection).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def export(images: np.ndarray, labels: np.ndarray, out: Path, per_class: int | None) -> int:
    written = 0
    counts = dict.fromkeys(range(len(CLASSES)), 0)
    for name in CLASSES:
        (out / name).mkdir(parents=True, exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels.reshape(-1))):
        if per_class is not None and counts[int(label)] >= per_class:
            continue
        Image.fromarray(img).save(out / CLASSES[int(label)] / f"{i:05d}.png")
        counts[int(label)] += 1
        written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/cifar10")
    p.add_argument("--per-class", type=int, default=None, help="cap training images per class (quick runs)")
    a = p.parse_args()

    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    out = Path(a.out)
    n_train = export(x_train, y_train, out / "train", a.per_class)
    n_test = export(x_test, y_test, out / "test", None)
    print(f"wrote {n_train} train and {n_test} test images to {out}")


if __name__ == "__main__":
    main()
