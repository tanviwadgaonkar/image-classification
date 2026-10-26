# Image Classification with a CNN

[![CI](https://github.com/tanviwadgaonkar/image-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/tanviwadgaonkar/image-classification/actions/workflows/ci.yml) [![Train on CIFAR-10](https://github.com/tanviwadgaonkar/image-classification/actions/workflows/train-cifar10.yml/badge.svg)](https://github.com/tanviwadgaonkar/image-classification/actions/workflows/train-cifar10.yml)

A compact convolutional neural network (TensorFlow / Keras) that trains on any **folder-per-class** image dataset, evaluates on a held-out test set, and classifies single images. Trained and evaluated end to end on **CIFAR-10** in CI.

```
data/<dataset>/train/<class>/*.png ──tf.data (streamed, seeded 80/20 split)──┐
                                                                             ▼
          model: augmentation ─► rescale ─► 3 × [conv-BN-ReLU ×2 ─► pool] ─► GAP ─► dropout ─► softmax
                                                                             │  EarlyStopping (best val_loss)
                                                                             ▼
                       artifacts/model.keras + class_names.json + history.json
                                                                             │
data/<dataset>/test/<class>/ ──────────► evaluate ─► metrics.json (accuracy, macro-F1, per class) + confusion_matrix.png
image.jpg ─────────────────────────────► predict  ─► top-k classes with probabilities
```

## Results on CIFAR-10

<!-- RESULTS -->

## Quick start

```bash
pip install -r requirements.txt

python scripts/export_cifar10.py --out data/cifar10          # 50k train / 10k test PNGs, folder per class
python -m src.train --data-dir data/cifar10/train --image-size 32 --batch-size 64 --epochs 40
python -m src.evaluate --model-dir artifacts --test-dir data/cifar10/test
python -m src.predict --model-dir artifacts some_photo.jpg
```

Any dataset laid out as `<root>/<class_name>/<image>` works the same way; use `--image-size 224` for photos.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q        # 13 tests
```

The suite generates a small synthetic dataset (red circles vs blue squares) and runs the real pipeline on it: image loading and error handling, class discovery, the train/validation split (disjoint, integer labels), model output shape and size, training to ≥ 90 % validation accuracy, held-out evaluation, and single-image prediction. CI fails if the TensorFlow tests are skipped.

## Design decisions

- **Preprocessing inside the model.** `Rescaling(1/255)` and the augmentation layers (flip, small rotation, zoom; active only during training) are part of the saved model, so `predict` feeds raw pixels exactly as training did — no separate preprocessing to keep in sync.
- **Global average pooling instead of Flatten.** At 224×224 the original `Flatten → Dense(64)` layer alone had ~12M parameters; the current model has under 400k in total at any input size, which overfits far less on small datasets.
- **Streaming input pipeline.** `image_dataset_from_directory` reads batches from disk with prefetching instead of loading every image into a NumPy array first, and produces the integer labels that `sparse_categorical_crossentropy` expects (the original code passed folder-name strings, which can't train).
- **Honest evaluation.** Early stopping and learning-rate decay watch the validation split carved from the training folder; the test folder is only used by `evaluate`, once.

## Layout

```
src/
  data_preprocessing.py  folder datasets (tf.data), single-image loading
  model.py               CNN with built-in rescaling and augmentation
  train.py               training CLI: early stopping, LR schedule, artifacts
  evaluate.py            test metrics + confusion matrix
  predict.py             top-k prediction for image files
scripts/export_cifar10.py
tests/                   pytest suite on a generated dataset
```
