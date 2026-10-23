from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never, cast

import numpy as np
import pytest
import torch

from anpr_engine.domain import (
    ArtifactIdentity,
    BBoxXYXY,
    CropReference,
    PixelCrop,
    RecognitionResult,
)
from anpr_engine.integration.plate_face_geometry import PlateFaceGeometry
from anpr_engine.model_manifest import artifact_path_sha256
from anpr_engine.recognition import web_runtime
from anpr_engine.recognition.crop import PlateCrop
from anpr_engine.recognition.web_runtime import WebOcrConfig, WebOcrRuntime


def scripted_reads(
    runtime: WebOcrRuntime,
    monkeypatch: pytest.MonkeyPatch,
    values: list[tuple[str, float] | None],
) -> list[PlateCrop]:
    seen: list[PlateCrop] = []

    def predict(value: PlateCrop) -> RecognitionResult:
        next_read = values[len(seen)]
        seen.append(value)
        if next_read is None:
            raise TimeoutError("private provider detail")
        text, score = next_read
        return RecognitionResult(
            raw_text=text,
            recognition_confidence=score,
            recognizer=runtime.identity,
            decoder_version="test",
            charset_version="test",
            preprocessing_signature="test",
        )

    monkeypatch.setattr(runtime._session, "recognize", predict)
    return seen


class Session:
    def __init__(
        self, command: tuple[str, ...], *, identity: ArtifactIdentity, timeout_seconds: float
    ) -> None:
        self.identity = identity
        self.closed = threading.Event()
        self.score = 0.999
        self.text = "34ABC123"
        self.fail = False

    def recognize(self, crop: PlateCrop) -> RecognitionResult:
        if self.fail:
            raise TimeoutError("private details")
        self.closed.clear()
        return RecognitionResult(
            raw_text=self.text,
            recognition_confidence=self.score,
            recognizer=self.identity,
            decoder_version="test",
            charset_version="test",
            preprocessing_signature="test",
        )

    def close(self) -> None:
        self.closed.set()


@pytest.fixture
def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[WebOcrRuntime, None, None]:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights").write_bytes(b"test")
    executable = tmp_path / "python"
    executable.touch()
    monkeypatch.setattr(web_runtime, "ManagedIsolatedPaddleOcrSession", Session)
    instance = WebOcrRuntime(
        WebOcrConfig(
            python=executable,
            worker=executable,
            model_dir=model,
            model_sha256=artifact_path_sha256(model),
            idle_seconds=1,
        )
    )
    yield instance
    instance.close()


def crop() -> PlateCrop:
    rgb = np.zeros((8, 20, 3), dtype=np.uint8)
    rgb[:] = [40, 130, 210]  # Distinct colour and gray views, not duplicate pixels.
    return PlateCrop(
        rgb,
        rgb,
        CropReference(
            crop_id="test",
            pixel_crop=PixelCrop(x1=0, y1=0, x2=20, y2=8),
            source_bbox=BBoxXYXY(x1=0, y1=0, x2=20, y2=8),
            preprocessing_signature="test",
        ),
    )


def test_idle_expiry_and_reuse(runtime: WebOcrRuntime) -> None:
    session = cast(Session, runtime._session)
    assert runtime.recognize(crop())["tr_text"] == "34ABC123"
    assert session.closed.wait(2)
    assert runtime.recognize(crop())["tr_text"] == "34ABC123"
    assert not session.closed.is_set()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.recognize(crop())


def test_stale_idle_callback_does_not_close_new_request(runtime: WebOcrRuntime) -> None:
    runtime.recognize(crop())
    old = runtime._generation
    runtime.recognize(crop())
    runtime._expire(old)
    assert not cast(Session, runtime._session).closed.is_set()


def test_uncertain_invalid_and_failed_ocr_do_not_expose_prediction(runtime: WebOcrRuntime) -> None:
    session = cast(Session, runtime._session)
    session.score = 0.8
    assert runtime.recognize(crop())["tr_text"] is None
    session.score = 1
    session.text = "NOT A PLATE"
    assert runtime.recognize(crop())["tr_text"] is None
    session.text = ""
    assert runtime.recognize(crop())["tr_text"] is None
    assert not session.closed.is_set()
    session.fail = True
    assert "private" not in str(runtime.recognize(crop()))
    assert session.closed.is_set()


def test_paddle_pipeline_does_not_build_project_recognizer(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anpr_engine.integration import anpr_v1

    class Detector:
        def eval(self) -> Detector:
            return self

    monkeypatch.setattr(anpr_v1.FrozenBundle, "verify", lambda _: None)
    monkeypatch.setattr(
        anpr_v1,
        "load_checkpoint",
        lambda *args: (Detector(), {"architecture": "yolox_multiscale_v1"}),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("legacy recognizer must not be loaded")

    monkeypatch.setattr(anpr_v1, "build_transformer_fixed_slot_recognizer", forbidden)
    pipeline = anpr_v1.AnprV1Pipeline(
        Path(__file__).resolve().parents[1],
        model_bundle=Path("unused"),
        device_name="cpu",
        web_ocr=runtime,
    )
    assert not hasattr(pipeline, "recognizer")
    assert pipeline.health()["recognizer"]["sha256"] == runtime.identity.sha256
    pipeline.close()


def test_format_invalid_confident_ocr_is_visible_but_not_validated(runtime: WebOcrRuntime) -> None:
    session = cast(Session, runtime._session)
    session.text = "82 TR 37"
    result = runtime.recognize(crop())
    assert result["display_text"] == "82TR37"
    assert result["tr_text"] is None
    assert result["structural_validation"]["valid"] is False
    session.score = 0.8
    uncertain = runtime.recognize(crop())
    assert uncertain["display_text"] is None
    assert uncertain["candidate_text"] == "82TR37"


def test_ocr_consensus_uses_votes_not_province_or_character_rewrites(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = scripted_reads(
        runtime, monkeypatch, [("31AKD783", 0.999), ("01AKD783", 0.98), ("01AKD783", 0.97)]
    )
    result = runtime.recognize(crop())
    assert len(seen) == 3
    assert len({value.reference.crop_id for value in seen}) == 3
    assert seen[1].reference.preprocessing_signature.endswith("ocr-view-v1:gray")
    assert result["candidate_text"] == result["raw_text"] == "01AKD783"
    assert result["display_text"] is None
    assert result["decoder_confidence"] == 0.97
    assert result["ocr_view"] == "gray" and result["ocr_consensus"] is True
    assert result["decoding_changed_raw"] is False


def test_three_disagreeing_reads_cannot_be_accepted(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_reads(
        runtime, monkeypatch, [("31AKD783", 0.999), ("01AKD783", 0.999), ("81AKD783", 0.999)]
    )
    result = runtime.recognize(crop())
    assert result["display_text"] is None and result["tr_text"] is None
    assert result["candidate_text"] == "31AKD783" and result["ocr_consensus"] is False


def test_agreement_does_not_inflate_confidence_or_rewrite_a_zero(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_reads(
        runtime, monkeypatch, [("31AKD783", 0.8), ("31AKD783", 0.999), ("31AKD783", 0.999)]
    )
    result = runtime.recognize(crop())
    assert result["candidate_text"] == "31AKD783"
    assert result["decoder_confidence"] == 0.8 and result["display_text"] is None


def test_confident_large_crop_keeps_the_single_pass_fast_path(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = scripted_reads(runtime, monkeypatch, [("34ABC123", 0.999)])
    value = crop()
    value = PlateCrop(np.zeros((48, 160, 3), np.uint8), value.prepared_rgb, value.reference)
    result = runtime.recognize(value)
    assert len(seen) == result["ocr_view_count"] == 1
    assert result["display_text"] == "34ABC123"


def test_verification_failure_keeps_an_unaccepted_candidate_and_stops(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = scripted_reads(runtime, monkeypatch, [("31AKD783", 0.999), None])
    result = runtime.recognize(crop())
    assert len(seen) == 2 and result["display_text"] is None
    assert result["candidate_text"] == "31AKD783"
    assert cast(Session, runtime._session).closed.is_set()
    assert "private provider detail" not in str(result)


def test_verification_views_preserve_all_original_pixels() -> None:
    from anpr_engine.recognition.web_runtime import recognition_view_rgb

    rgb = np.arange(12 * 60 * 3, dtype=np.uint8).reshape(12, 60, 3)
    bordered = recognition_view_rgb(rgb, "white_border")
    np.testing.assert_array_equal(bordered[2:-2, 2:-2], rgb)
    assert np.all(bordered[:2] == 255)
    gray = recognition_view_rgb(rgb, "gray")
    assert gray.shape == rgb.shape and np.all(gray[:, :, 0] == gray[:, :, 2])
    with pytest.raises(ValueError, match="view"):
        recognition_view_rgb(rgb, "unsupported")


@pytest.mark.parametrize(
    "span,expected",
    [(0, "gray"), (15, "gray"), (16, "contrast"), (80, "contrast"), (96, "gray"), (200, "gray")],
)
def test_verification_policy_is_bounded_and_pixel_driven(span: int, expected: str) -> None:
    pixels = np.full((16, 40, 3), 20, np.uint8)
    pixels[:, 20:] += span
    assert web_runtime.verification_views(pixels) == (expected, "white_border")


def test_contrast_expands_dim_glyphs_without_resizing_or_modifying_source() -> None:
    ramp = np.linspace(90, 140, 80).round().astype(np.uint8)
    pixels = np.repeat(np.tile(ramp, (16, 1))[:, :, None], 3, axis=2)
    saved = pixels.copy()
    enhanced = web_runtime.recognition_view_rgb(pixels, "contrast")
    assert enhanced.shape == pixels.shape and enhanced.dtype == pixels.dtype
    assert enhanced.flags.c_contiguous
    np.testing.assert_array_equal(pixels, saved)
    assert enhanced.min() == 0 and enhanced.max() == 255
    assert np.all(np.diff(enhanced[0, :, 0].astype(int)) >= 0)


@pytest.mark.parametrize("span", [0, 8, 160])
def test_contrast_does_not_stretch_flat_noise_or_high_contrast(span: int) -> None:
    pixels = np.full((16, 40, 3), 20, np.uint8)
    pixels[:, 20:] += span
    np.testing.assert_array_equal(web_runtime.recognition_view_rgb(pixels, "contrast"), pixels)


def test_low_contrast_disagreement_can_recover_by_two_real_reads(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = crop()
    pixels = np.full((8, 40, 3), 90, np.uint8)
    pixels[:, 20:] = 130
    value = PlateCrop(pixels, value.prepared_rgb, value.reference)
    seen = scripted_reads(
        runtime, monkeypatch, [("31AKD783", 0.95), ("01AKD783", 0.98), ("01AKD783", 0.97)]
    )
    result = runtime.recognize(value)
    assert len(seen) == 3
    assert result["ocr_view"] == "contrast" and result["candidate_text"] == "01AKD783"
    assert result["decoder_version"].endswith("/consensus-v2")
    assert result["decoder_confidence"] == 0.97 and result["display_text"] is None


def test_identical_gray_pixels_are_not_an_extra_vote_or_worker_call(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = crop()
    pixels = np.full((8, 20, 3), 100, np.uint8)
    seen = scripted_reads(runtime, monkeypatch, [("31AKD783", 0.999), ("01AKD783", 0.999)])
    result = runtime.recognize(PlateCrop(pixels, value.prepared_rgb, value.reference))
    assert len(seen) == result["ocr_view_count"] == 2
    assert result["ocr_consensus"] is False and result["display_text"] is None
    assert result["candidate_text"] == "31AKD783"


def test_disagreement_keeps_the_original_score_not_another_candidates_score(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripted_reads(runtime, monkeypatch, [("", 0), ("01AKD783", 0.98), ("81AKD783", 0.97)])
    result = runtime.recognize(crop())
    assert result["decoder_confidence"] == result["raw_confidence"] == 0
    assert result["candidate_text"] is None and result["ocr_consensus"] is False


def test_pipeline_passes_rectified_face_and_reports_same_geometry(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_plate_face_geometry import perspective_scene

    from anpr_engine.integration import anpr_v1

    class Detector:
        def eval(self) -> Detector:
            return self

        def __call__(self, tensor: Any) -> None:
            return None

    monkeypatch.setattr(anpr_v1.FrozenBundle, "verify", lambda _: None)
    monkeypatch.setattr(
        anpr_v1,
        "load_checkpoint",
        lambda *args: (Detector(), {"architecture": "yolox_multiscale_v1"}),
    )
    monkeypatch.setattr(
        anpr_v1,
        "letterbox",
        lambda *args: (
            np.zeros((384, 384, 3), np.uint8),
            SimpleNamespace(to_original_xyxy=lambda _: (100, 80, 580, 240)),
        ),
    )
    monkeypatch.setattr(
        anpr_v1,
        "decode_multiscale_outputs",
        lambda _: (torch.ones((1, 1)), torch.tensor([[[0.1, 0.1, 0.9, 0.9]]])),
    )
    observed: list[PlateCrop] = []
    session = cast(Session, runtime._session)
    original = session.recognize

    def recognize(value: PlateCrop) -> RecognitionResult:
        observed.append(value)
        return original(value)

    monkeypatch.setattr(session, "recognize", recognize)
    pipeline = anpr_v1.AnprV1Pipeline(
        Path(__file__).resolve().parents[1],
        model_bundle=Path("unused"),
        device_name="cpu",
        web_ocr=runtime,
    )
    image, corners = perspective_scene()
    try:
        result = pipeline.infer(
            image,
            batch_id="test",
            request_id="test",
            input_sha256="not-retained",
            filename="not-retained",
            output_directory=None,
        )
        geometry = result["plate_geometry"]
        assert geometry["state"] == "estimated" and geometry["rectification_applied"]
        recovered = np.array(geometry["quadrilateral_xy_normalized"]) * [700, 400]
        assert np.max(np.linalg.norm(recovered - corners, axis=1)) < 5
        assert observed and observed[0].reference.preprocessing_signature == "PLATE_FACE_CONTOUR_V1"
        assert observed[0].rgb.shape[1] / observed[0].rgb.shape[0] > 3
        assert result["recognition_crop_reference"]["method_id"] == "PLATE_FACE_CONTOUR_V1"
        assert result["plate_crops"]["rectified"]["media_type"] == "image/jpeg"
        assert result["plate_crops"]["recognition_input"]["metadata_removed"] is True
        monkeypatch.setattr(
            anpr_v1,
            "estimate_plate_face",
            lambda *args: PlateFaceGeometry(None, None, "NO_RELIABLE_FACE_CONTOUR"),
        )
        failed = pipeline.infer(
            image,
            batch_id="test",
            request_id="test",
            input_sha256="not-retained",
            filename="not-retained",
            output_directory=None,
        )
        assert failed["pipeline_status"] == "INVALID_CROP"
        assert failed["recognition"] is None
        assert failed["timings_ms"]["recognition_inference_ms"] == 0
        assert len(observed) == 1  # No unrectified OCR attempt or extra worker request.
        runtime.config = runtime.config.model_copy(update={"estimate_plate_geometry": False})
        rollback = pipeline.infer(
            image,
            batch_id="test",
            request_id="test",
            input_sha256="not-retained",
            filename="not-retained",
            output_directory=None,
        )
        assert rollback["plate_geometry"]["state"] == "unresolved"
        assert rollback["plate_geometry_reason"] == "GEOMETRY_DISABLED"
        assert "plate_crops" not in rollback
    finally:
        pipeline.close()


def test_learned_path_never_loads_legacy_detector_or_estimates_pixel_corners(
    runtime: WebOcrRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_plate_face_geometry import perspective_scene

    from anpr_engine.integration import anpr_v1

    image, corners = perspective_scene()

    class Learned:
        model_id = "hyperlpr-y5fu-640-corners-v1"
        model_sha = "0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c"

        def __init__(self, path: Path, size: int) -> None:
            assert size == 640

        def detect(self, value: Any) -> list[dict[str, Any]]:
            return [
                {"bbox_xyxy": [95, 80, 580, 290], "confidence": 0.9, "corners_xy": corners.tolist()}
            ]

    def forbidden(*args: Any, **kwargs: Any) -> Never:
        raise AssertionError("learned path must not use legacy models or pixel corner fitting")

    runtime.config = runtime.config.model_copy(
        update={"corner_model": Path("private.onnx"), "corner_model_size": 640}
    )
    monkeypatch.setattr(anpr_v1.FrozenBundle, "verify", lambda _: None)
    monkeypatch.setattr(anpr_v1, "HyperLprCornerDetector", Learned)
    monkeypatch.setattr(anpr_v1, "load_checkpoint", forbidden)
    monkeypatch.setattr(anpr_v1, "estimate_plate_face", forbidden)
    monkeypatch.setattr(anpr_v1, "refine_plate_boundary", forbidden)
    pipeline = anpr_v1.AnprV1Pipeline(
        Path(__file__).resolve().parents[1],
        model_bundle=Path("unused"),
        device_name="cpu",
        web_ocr=runtime,
    )
    try:
        result = pipeline.infer(
            image,
            batch_id="test",
            request_id="test",
            input_sha256="not-retained",
            filename="not-retained",
            output_directory=None,
        )
        assert not hasattr(pipeline, "detector") and not hasattr(pipeline, "recognizer")
        assert result["models"]["detector"]["version"] == Learned.model_id
        assert result["plate_geometry"]["method"] == "HYPERLPR_FOUR_CORNERS_V1"
        assert result["plate_crops"]["rectified"] is not None
        np.testing.assert_allclose(
            np.array(result["plate_geometry"]["quadrilateral_xy_normalized"]) * [700, 400], corners
        )
    finally:
        pipeline.close()
