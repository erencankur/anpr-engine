"""Optional PaddleOCR text-recognition adapter with no import-time dependency."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.domain import ArtifactIdentity, RecognitionResult
from anpr_engine.model_manifest import ModelArtifact, ModelProvider, ModelRole
from anpr_engine.recognition.crop import PlateCrop

PADDLEOCR_DECODER_VERSION = "paddleocr-native-decoder-v1"
PADDLEOCR_LATIN_CHARSET_VERSION = "paddleocr-latin-ppocrv5-v1"


class PaddleOcrOutputError(ValueError):
    """The provider returned an output that cannot be mapped without guessing."""


class PaddleOcrDependencyError(RuntimeError):
    """The optional PaddleOCR runtime is missing or does not expose the required API."""


class PaddleOcrRuntimeConfig(StrictConfigModel):
    model_name: Literal["latin_PP-OCRv5_mobile_rec"] = "latin_PP-OCRv5_mobile_rec"
    model_dir: Path
    device: Literal["cpu", "gpu"] = "cpu"
    engine: Literal["paddle_static", "onnxruntime"] = "paddle_static"


class PaddleTextRecognitionBackend(Protocol):
    def predict(
        self,
        *,
        input: NDArray[np.uint8],
        batch_size: int,
    ) -> Iterable[object]: ...


def _mapping_from_result(result: object) -> Mapping[str, object]:
    payload: object = result if isinstance(result, Mapping) else getattr(result, "json", None)
    if not isinstance(payload, Mapping):
        raise PaddleOcrOutputError("PaddleOCR result does not expose a JSON mapping")
    nested = payload.get("res", payload)
    if not isinstance(nested, Mapping):
        raise PaddleOcrOutputError("PaddleOCR result res field must be a mapping")
    return cast(Mapping[str, object], nested)


def parse_paddleocr_text_result(result: object) -> tuple[str, float]:
    payload = _mapping_from_result(result)
    text = payload.get("rec_text")
    score = payload.get("rec_score")
    if not isinstance(text, str):
        raise PaddleOcrOutputError("PaddleOCR rec_text must be a string")
    if not isinstance(score, int | float) or isinstance(score, bool):
        raise PaddleOcrOutputError("PaddleOCR rec_score must be numeric")
    confidence = float(score)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise PaddleOcrOutputError("PaddleOCR rec_score must be finite and within [0, 1]")
    return text, confidence


class PaddleOcrRecognizer:
    """Map recognition-only PaddleOCR output into the stable Recognizer contract."""

    def __init__(
        self,
        backend: PaddleTextRecognitionBackend,
        *,
        identity: ArtifactIdentity,
        charset_version: str = PADDLEOCR_LATIN_CHARSET_VERSION,
    ) -> None:
        if not charset_version:
            raise ValueError("PaddleOCR charset version must be non-empty")
        self._backend = backend
        self._identity = identity
        self._charset_version = charset_version

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        image = crop.rgb
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("PaddleOCR crop must be RGB uint8 HWC")
        bgr = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        results = tuple(self._backend.predict(input=bgr, batch_size=1))
        if len(results) != 1:
            raise PaddleOcrOutputError(
                f"PaddleOCR single-crop inference returned {len(results)} results"
            )
        raw_text, confidence = parse_paddleocr_text_result(results[0])
        return RecognitionResult(
            raw_text=raw_text,
            recognition_confidence=confidence,
            recognizer=self._identity,
            decoder_version=PADDLEOCR_DECODER_VERSION,
            charset_version=self._charset_version,
            preprocessing_signature=crop.reference.preprocessing_signature,
        )


def build_paddleocr_recognizer(
    config: PaddleOcrRuntimeConfig,
    *,
    artifact: ModelArtifact,
) -> PaddleOcrRecognizer:
    """Build from one local, approved and hash-verified model directory."""
    if artifact.role is not ModelRole.RECOGNITION:
        raise ValueError("PaddleOCR artifact must have recognition role")
    if artifact.provider is not ModelProvider.PADDLEOCR:
        raise ValueError("PaddleOCR adapter requires provider=paddleocr")
    if artifact.architecture != config.model_name:
        raise ValueError("PaddleOCR model name does not match artifact architecture")
    artifact.assert_runtime_eligible()
    artifact.verify_artifact(config.model_dir)

    try:
        module = import_module("paddleocr")
        text_recognition = cast(Any, module.TextRecognition)
    except (ImportError, AttributeError) as error:
        raise PaddleOcrDependencyError(
            "PaddleOCR runtime is unavailable; install the paddleocr optional dependency"
        ) from error
    backend = cast(
        PaddleTextRecognitionBackend,
        text_recognition(
            model_name=config.model_name,
            model_dir=str(config.model_dir),
            device=config.device,
            engine=config.engine,
        ),
    )
    identity = ArtifactIdentity(
        name=artifact.model_id,
        version=artifact.version,
        sha256=artifact.artifact_sha256,
        reference=artifact.source,
    )
    return PaddleOcrRecognizer(backend, identity=identity)
