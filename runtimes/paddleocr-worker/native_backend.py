"""Single-image CPU execution of the existing PaddleOCR CTC inference artifact.

Resize/normalization and CTC semantics follow PaddleX 3.7.2 text-recognition
processors. The network and vocabulary are loaded from the approved artifact;
this module does not define or train a new recognition model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def prepare_image(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("invalid_image")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("invalid_image")
    channels, target_height, base_width = shape
    target_width = min(3200, int(target_height * max(base_width / target_height, width / height)))
    resized_width = min(target_width, math.ceil(target_height * width / height))
    resized = cv2.resize(image, (resized_width, target_height)).astype(np.float32)
    normalized = resized.transpose(2, 0, 1) / 255.0
    normalized = (normalized - 0.5) / 0.5
    padded = np.zeros((channels, target_height, target_width), dtype=np.float32)
    padded[:, :, :resized_width] = normalized
    return np.ascontiguousarray(padded[None])


def decode_ctc(probabilities: np.ndarray, characters: tuple[str, ...]) -> tuple[str, float]:
    if (
        probabilities.ndim != 3
        or probabilities.shape[0] != 1
        or probabilities.shape[1] == 0
        or probabilities.shape[2] != len(characters)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("provider_output_error")
    indices = probabilities[0].argmax(axis=-1)
    scores = probabilities[0].max(axis=-1)
    selected = indices != 0
    selected[1:] &= indices[1:] != indices[:-1]
    text = "".join(characters[int(index)] for index in indices[selected])
    confidence = float(scores[selected].mean()) if selected.any() else 0.0
    return text, confidence


class NativeCtcRecognizer:
    def __init__(self, model_dir: Path, *, cpu_threads: int) -> None:
        if not 1 <= cpu_threads <= 2:
            raise ValueError("invalid_cpu_threads")
        metadata = yaml.safe_load((model_dir / "inference.yml").read_text())
        if (
            not isinstance(metadata, dict)
            or metadata.get("Global", {}).get("model_name") != "latin_PP-OCRv5_mobile_rec"
            or metadata.get("PostProcess", {}).get("name") != "CTCLabelDecode"
        ):
            raise ValueError("unsupported_native_model")
        operations = metadata["PreProcess"]["transform_ops"]
        shapes = [op["RecResizeImg"]["image_shape"] for op in operations if "RecResizeImg" in op]
        if shapes != [[3, 48, 320]] or not any(
            op.get("DecodeImage", {}).get("img_mode") == "BGR" for op in operations
        ):
            raise ValueError("unsupported_native_preprocessing")
        vocabulary = metadata["PostProcess"].get("character_dict")
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or not all(isinstance(char, str) and char for char in vocabulary)
        ):
            raise ValueError("invalid_native_vocabulary")
        self.characters = ("", *vocabulary, " ")
        self.shape = (3, 48, 320)
        from paddle import inference

        config = inference.Config(
            str(model_dir / "inference.json"), str(model_dir / "inference.pdiparams")
        )
        config.disable_gpu()
        config.disable_mkldnn()
        config.set_cpu_math_library_num_threads(cpu_threads)
        config.enable_new_ir(True)
        config.enable_new_executor()
        config.set_optimization_level(3)
        config.enable_memory_optim()
        config.disable_glog_info()
        self._predictor: Any = inference.create_predictor(config)
        if (
            len(self._predictor.get_input_names()) != 1
            or len(self._predictor.get_output_names()) != 1
        ):
            raise ValueError("unsupported_native_io")

    def predict(self, *, input: np.ndarray, batch_size: int) -> tuple[dict[str, object], ...]:
        if batch_size != 1 or self._predictor is None:
            raise ValueError("invalid_native_state")
        prepared = prepare_image(input, self.shape)
        handle = self._predictor.get_input_handle(self._predictor.get_input_names()[0])
        handle.reshape(prepared.shape)
        handle.copy_from_cpu(prepared)
        self._predictor.run()
        output = self._predictor.get_output_handle(self._predictor.get_output_names()[0])
        text, confidence = decode_ctc(output.copy_to_cpu(), self.characters)
        return ({"rec_text": text, "rec_score": confidence},)

    def close(self) -> None:
        self._predictor = None
