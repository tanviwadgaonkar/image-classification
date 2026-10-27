"""Train the CNN on a folder-per-class dataset.

    python -m src.train --data-dir data/cifar10/train --image-size 32 --epochs 30

Writes to ``--out`` (default ``artifacts/``):
    model.keras        the trained model (includes scaling + augmentation)
    class_names.json   label index -> class name
    history.json       per-epoch loss/accuracy for train and validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_preprocessing import load_datasets
from src.model import build_model


def train_model(
    data_dir: str | Path,
    out_dir: str | Path = "artifacts",
    image_size: int = 224,
    epochs: int = 30,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> dict:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, names = load_datasets(
        data_dir, (image_size, image_size), batch_size, validation_split, seed
    )
    model = build_model((image_size, image_size, 3), num_classes=len(names))

    callbacks = [
        # Stop when validation loss stops improving and keep the best weights,
        # instead of a fixed epoch count that over- or under-trains.
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    model.save(out / "model.keras")
    (out / "class_names.json").write_text(json.dumps(names, indent=2))
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    best = min(range(len(hist["val_loss"])), key=hist["val_loss"].__getitem__)
    return {"classes": names, "epochs_run": len(hist["loss"]), "best_epoch": best + 1,
            "val_accuracy": hist["val_accuracy"][best]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validation-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    summary = train_model(a.data_dir, a.out, a.image_size, a.epochs, a.batch_size, a.validation_split, a.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
