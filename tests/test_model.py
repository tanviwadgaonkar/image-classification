import pytest
from src.model import build_model

def test_build_model():
    model = build_model((224, 224, 3))
    assert model is not None
    assert len(model.layers) > 0
