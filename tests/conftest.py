"""A tiny synthetic dataset (red circles vs blue squares) generated per test
session, so the whole pipeline can be exercised end to end in seconds."""
import numpy as np
import pytest
from PIL import Image, ImageDraw


def _draw(kind: str, rng: np.random.Generator, size: int = 32) -> Image.Image:
    noise = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    d = ImageDraw.Draw(img)
    r = int(rng.integers(6, 12))
    cx, cy = (int(v) for v in rng.integers(r + 1, size - r - 1, 2))
    if kind == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 40, 40))
    else:
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=(40, 40, 220))
    return img


def make_dataset(root, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    for kind in ("circle", "square"):
        (root / kind).mkdir(parents=True)
        for i in range(per_class):
            _draw(kind, rng).save(root / kind / f"{i:03d}.png")
    return root


@pytest.fixture(scope="session")
def shapes(tmp_path_factory):
    base = tmp_path_factory.mktemp("shapes")
    return {
        "train": make_dataset(base / "train", per_class=60, seed=0),
        "test": make_dataset(base / "test", per_class=20, seed=1),
    }
