"""Classify single images with a trained model.

    python -m src.predict --model-dir artifacts path/to/image.jpg [more.jpg ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_preprocessing import load_and_preprocess_image


class Classifier:
    def __init__(self, model_dir: str | Path) -> None:
        import tensorflow as tf

        model_dir = Path(model_dir)
        self.model = tf.keras.models.load_model(model_dir / "model.keras")
        self.class_names = json.loads((model_dir / "class_names.json").read_text())
        self.image_size = tuple(self.model.input_shape[1:3])

    def predict(self, image_path: str | Path, top_k: int = 3) -> list[dict]:
        image = load_and_preprocess_image(image_path, self.image_size)  # raw pixels; model rescales
        probs = self.model.predict(image[np.newaxis], verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [{"class": self.class_names[i], "probability": round(float(probs[i]), 4)} for i in order]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("images", nargs="+")
    a = p.parse_args(argv)
    clf = Classifier(a.model_dir)
    for path in a.images:
        print(path, json.dumps(clf.predict(path, a.top_k)))


if __name__ == "__main__":
    main()
