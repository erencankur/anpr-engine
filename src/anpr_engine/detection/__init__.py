"""Project-owned YOLO-family plate detector boundary."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anpr_engine.detection.config import DetectorConfig, load_detector_config
    from anpr_engine.detection.data import DetectionDataset, DetectionManifestSample, load_samples
    from anpr_engine.detection.model import TinyYoloDetector, decode_predictions


def __getattr__(name: str) -> Any:
    if name in {"DetectorConfig", "load_detector_config"}:
        from anpr_engine.detection import config

        return getattr(config, name)
    if name in {"DetectionDataset", "DetectionManifestSample", "load_samples"}:
        from anpr_engine.detection import data

        return getattr(data, name)
    if name in {"TinyYoloDetector", "decode_predictions"}:
        from anpr_engine.detection import model

        return getattr(model, name)
    raise AttributeError(name)


__all__ = [
    "DetectionDataset",
    "DetectionManifestSample",
    "DetectorConfig",
    "TinyYoloDetector",
    "decode_predictions",
    "load_detector_config",
    "load_samples",
]
