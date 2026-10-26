"""Model, training, evaluation and prediction. Needs TensorFlow (skipped
where it can't be imported); CI runs these on Linux."""
import json

import pytest

tf = pytest.importorskip("tensorflow")

from src.data_preprocessing import load_datasets, load_test_dataset  # noqa: E402
from src.evaluate import evaluate_model  # noqa: E402
from src.model import build_model  # noqa: E402
from src.predict import Classifier  # noqa: E402
from src.train import train_model  # noqa: E402


def test_model_outputs_one_probability_per_class():
    model = build_model((32, 32, 3), num_classes=5)
    out = model(tf.zeros((2, 32, 32, 3)), training=False).numpy()
    assert out.shape == (2, 5)
    assert out.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_model_is_compact():
    # Global average pooling keeps the head small even at 224x224.
    assert build_model((224, 224, 3), num_classes=10).count_params() < 400_000


def test_model_rejects_fewer_than_two_classes():
    with pytest.raises(ValueError):
        build_model((32, 32, 3), num_classes=1)


def test_split_is_disjoint_and_complete(shapes):
    train, val, names = load_datasets(shapes["train"], (32, 32), batch_size=16, validation_split=0.25)
    n_train = sum(int(y.shape[0]) for _, y in train)
    n_val = sum(int(y.shape[0]) for _, y in val)
    assert names == ["circle", "square"]
    assert (n_train, n_val) == (90, 30)
    labels = {int(v) for _, y in train for v in y.numpy()}
    assert labels == {0, 1}  # integer labels, matching sparse_categorical_crossentropy


def test_test_set_with_different_classes_is_rejected(shapes, tmp_path):
    (tmp_path / "circle").mkdir()
    (tmp_path / "triangle").mkdir()
    with pytest.raises(ValueError, match="do not match"):
        load_test_dataset(tmp_path, ["circle", "square"], (32, 32))


@pytest.fixture(scope="module")
def trained(shapes, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    summary = train_model(shapes["train"], out, image_size=32, epochs=15, batch_size=16)
    return out, summary


def test_training_learns_and_writes_artifacts(trained):
    out, summary = trained
    assert {p.name for p in out.iterdir()} >= {"model.keras", "class_names.json", "history.json"}
    assert json.loads((out / "class_names.json").read_text()) == ["circle", "square"]
    assert summary["val_accuracy"] >= 0.9


def test_evaluation_on_held_out_images(trained, shapes):
    out, _ = trained
    metrics = evaluate_model(out, shapes["test"])
    assert metrics["n_test"] == 40
    assert metrics["accuracy"] >= 0.9
    assert set(metrics["per_class"]) == {"circle", "square"}
    assert (out / "metrics.json").exists() and (out / "confusion_matrix.png").exists()


def test_prediction_on_a_single_image(trained, shapes):
    out, _ = trained
    top = Classifier(out).predict(shapes["test"] / "square" / "000.png", top_k=2)
    assert top[0]["class"] == "square"
    assert top[0]["probability"] >= top[1]["probability"]
