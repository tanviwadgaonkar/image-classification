"""Evaluate a trained model on a held-out folder dataset.

    python -m src.evaluate --model-dir artifacts --test-dir data/cifar10/test

Writes ``metrics.json`` (accuracy, macro-F1, per-class precision/recall/F1)
and ``confusion_matrix.png`` to the model directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_preprocessing import load_test_dataset


def evaluate_model(model_dir: str | Path, test_dir: str | Path, batch_size: int = 64) -> dict:
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    model_dir = Path(model_dir)
    model = tf.keras.models.load_model(model_dir / "model.keras")
    names = json.loads((model_dir / "class_names.json").read_text())
    image_size = tuple(model.input_shape[1:3])

    ds = load_test_dataset(test_dir, names, image_size, batch_size)
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = model.predict(ds, verbose=0).argmax(axis=1)

    report = classification_report(y_true, y_pred, labels=range(len(names)), target_names=names,
                                   output_dict=True, zero_division=0)
    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": round(float(report["accuracy"]), 4),
        "f1_macro": round(float(report["macro avg"]["f1-score"]), 4),
        "per_class": {n: {k: round(float(report[n][k]), 4) for k in ("precision", "recall", "f1-score")}
                      for n in names},
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _plot_confusion(confusion_matrix(y_true, y_pred, labels=range(len(names))), names,
                    model_dir / "confusion_matrix.png")
    return metrics


def _plot_confusion(cm: np.ndarray, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(names), 1 + 0.6 * len(names)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--test-dir", required=True)
    a = p.parse_args(argv)
    m = evaluate_model(a.model_dir, a.test_dir)
    print(json.dumps({k: m[k] for k in ("n_test", "accuracy", "f1_macro")}, indent=2))


if __name__ == "__main__":
    main()
