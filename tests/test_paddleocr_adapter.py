from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from anpr_engine.domain import ArtifactIdentity, BBoxXYXY, CropReference, PixelCrop
from anpr_engine.model_manifest import (
    LicenseEvidence,
    LicenseReviewStatus,
    ModelArtifact,
    ModelProvider,
    ModelRole,
    artifact_path_sha256,
)
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.paddleocr_adapter import (
    PaddleOcrOutputError,
    PaddleOcrRecognizer,
    PaddleOcrRuntimeConfig,
    build_paddleocr_recognizer,
    parse_paddleocr_text_result,
)


class FakeBackend:
    def __init__(self, results: tuple[object, ...]) -> None:
        self.results = results
        self.input: np.ndarray[tuple[int, ...], np.dtype[np.uint8]] | None = None

    def predict(
        self,
        *,
        input: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
        batch_size: int,
    ) -> Iterable[object]:
        assert batch_size == 1
        self.input = input.copy()
        return self.results


class ResultObject:
    def __init__(self, payload: object) -> None:
        self.json = payload


def _crop() -> PlateCrop:
    rgb = np.zeros((8, 20, 3), dtype=np.uint8)
    rgb[0, 0] = (10, 20, 30)
    return PlateCrop(
        rgb=rgb,
        prepared_rgb=np.zeros((32, 160, 3), dtype=np.uint8),
        reference=CropReference(
            crop_id="crop-id",
            pixel_crop=PixelCrop(x1=0, y1=0, x2=20, y2=8),
            source_bbox=BBoxXYXY(x1=0, y1=0, x2=20, y2=8),
            preprocessing_signature="rectified-rgb-v1",
        ),
    )


def _identity() -> ArtifactIdentity:
    return ArtifactIdentity(
        name="latin_PP-OCRv5_mobile_rec",
        version="governed-version",
        sha256="a" * 64,
    )


def test_adapter_preserves_raw_text_confidence_and_converts_rgb_to_bgr() -> None:
    backend = FakeBackend((ResultObject({"res": {"rec_text": " 34 abc-123 ", "rec_score": 0.91}}),))
    result = PaddleOcrRecognizer(backend, identity=_identity()).recognize(_crop())
    # The established domain contract strips surrounding whitespace from all strings.
    assert result.raw_text == "34 abc-123"
    assert result.normalized_text is None
    assert result.recognition_confidence == 0.91
    assert result.preprocessing_signature == "rectified-rgb-v1"
    assert backend.input is not None
    assert tuple(backend.input[0, 0]) == (30, 20, 10)


def test_parser_accepts_direct_provider_mapping() -> None:
    assert parse_paddleocr_text_result({"rec_text": "34ABC123", "rec_score": 1}) == (
        "34ABC123",
        1.0,
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"res": {"rec_text": 123, "rec_score": 0.9}}, "rec_text"),
        ({"res": {"rec_text": "34ABC123", "rec_score": "0.9"}}, "rec_score"),
        ({"res": {"rec_text": "34ABC123", "rec_score": 1.2}}, "within"),
    ],
)
def test_parser_rejects_malformed_provider_output(payload: object, message: str) -> None:
    with pytest.raises(PaddleOcrOutputError, match=message):
        parse_paddleocr_text_result(ResultObject(payload))


def test_adapter_requires_exactly_one_result() -> None:
    backend = FakeBackend(())
    with pytest.raises(PaddleOcrOutputError, match="returned 0 results"):
        PaddleOcrRecognizer(backend, identity=_identity()).recognize(_crop())


def _approved_artifact(model_dir: Path) -> ModelArtifact:
    return ModelArtifact(
        role=ModelRole.RECOGNITION,
        provider=ModelProvider.PADDLEOCR,
        model_id="latin-ppocrv5-governed",
        architecture="latin_PP-OCRv5_mobile_rec",
        version="3.7.0-test",
        artifact_format="paddle-inference-directory",
        artifact_sha256=artifact_path_sha256(model_dir),
        source="governed-local-test-artifact",
        preprocessing_signature="paddleocr-native-v1",
        runtime="paddleocr",
        runtime_version="3.7.0",
        license=LicenseEvidence(
            license_id="Apache-2.0",
            review_status=LicenseReviewStatus.APPROVED,
            reference="test-license-evidence",
            reviewed_by="authorized-reviewer",
            reviewed_on="2026-10-13",
        ),
    )


def test_factory_requires_local_verified_artifact_and_passes_pinned_runtime_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "paddle-model"
    model_dir.mkdir()
    (model_dir / "inference.json").write_text("{}", encoding="utf-8")
    (model_dir / "inference.pdiparams").write_bytes(b"weights")
    observed: dict[str, object] = {}
    backend = FakeBackend(({"res": {"rec_text": "34ABC123", "rec_score": 0.95}},))
    module = ModuleType("paddleocr")

    def text_recognition(**kwargs: object) -> FakeBackend:
        observed.update(kwargs)
        return backend

    module.TextRecognition = text_recognition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    recognizer = build_paddleocr_recognizer(
        PaddleOcrRuntimeConfig(model_dir=model_dir, device="cpu", engine="paddle_static"),
        artifact=_approved_artifact(model_dir),
    )
    result = recognizer.recognize(_crop())
    assert result.raw_text == "34ABC123"
    assert result.recognizer.name == "latin-ppocrv5-governed"
    assert observed == {
        "model_name": "latin_PP-OCRv5_mobile_rec",
        "model_dir": str(model_dir),
        "device": "cpu",
        "engine": "paddle_static",
    }


def test_factory_rejects_changed_model_directory_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "paddle-model"
    model_dir.mkdir()
    weights = model_dir / "inference.pdiparams"
    weights.write_bytes(b"weights")
    artifact = _approved_artifact(model_dir)
    weights.write_bytes(b"changed")
    monkeypatch.delitem(sys.modules, "paddleocr", raising=False)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_paddleocr_recognizer(
            PaddleOcrRuntimeConfig(model_dir=model_dir),
            artifact=artifact,
        )
