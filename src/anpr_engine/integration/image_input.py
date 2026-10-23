"""Torch-free decoding shared by legacy and lean HTTP inference."""

from __future__ import annotations

import io
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError


def decode_upload(content: bytes) -> tuple[NDArray[np.uint8], str]:
    if not content:
        raise ValueError("empty image")
    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format not in {"JPEG", "PNG"}:
                raise ValueError("unsupported image content")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("corrupt or unsupported image") from error
    return cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), opened.format or "UNKNOWN"
