from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def native() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "runtimes/paddleocr-worker/native_backend.py"
    spec = importlib.util.spec_from_file_location("native_paddle_backend", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_preprocessing_preserves_bgr_and_neutral_padding() -> None:
    image = np.zeros((48, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    prepared = native().prepare_image(image, (3, 48, 320))
    assert prepared.shape == (1, 3, 48, 320)
    assert prepared.dtype == np.float32
    assert prepared[0, 0, 0, 0] == 1
    assert prepared[0, 1, 0, 0] == -1
    assert np.all(prepared[:, :, :, 80:] == 0)


def test_native_ctc_keeps_blank_separated_repeated_letters_and_space() -> None:
    probabilities = np.zeros((1, 7, 4), dtype=np.float32)
    for position, token in enumerate((1, 1, 0, 1, 2, 2, 3)):
        probabilities[0, position, token] = 1
    text, confidence = native().decode_ctc(probabilities, ("", "A", "B", " "))
    assert text == "AAB "
    assert confidence == 1
    probabilities[:, :, :] = 0
    probabilities[:, :, 0] = 1
    assert native().decode_ctc(probabilities, ("", "A", "B", " ")) == ("", 0)


def test_native_ctc_rejects_invalid_vocabulary_shape_and_nonfinite_output() -> None:
    backend = native()
    with pytest.raises(ValueError, match="provider_output"):
        backend.decode_ctc(np.ones((1, 2, 4)), ("", "A"))
    with pytest.raises(ValueError, match="provider_output"):
        backend.decode_ctc(np.full((1, 2, 2), np.nan), ("", "A"))


def test_native_backend_rejects_unknown_artifact_before_loading_paddle(tmp_path: Path) -> None:
    (tmp_path / "inference.yml").write_text("Global: {model_name: unsupported}\n")
    with pytest.raises(ValueError, match="unsupported_native_model"):
        native().NativeCtcRecognizer(tmp_path, cpu_threads=1)
