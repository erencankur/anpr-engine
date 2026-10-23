# MMDetection development runtime

This directory records the exact package versions used for the RTMDet local-readiness smoke run.
It is not the production runtime and does not activate an external model in the ANPR pipeline.

The verified environment uses Python 3.11.16 on Apple Silicon, PyTorch 2.2.2, torchvision 0.17.2,
MMEngine 0.10.7, MMCV 2.1.0 with native CPU/MPS operations, and MMDetection 3.3.0 at source revision
`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`.

`mmcv-lite` is insufficient because RTMDet inference needs compiled operations such as NMS. On this
host MMCV was built with setuptools 69.5.1 and Ninja 1.13 using
`CFLAGS=-Wno-invalid-specialization` and `CXXFLAGS=-Wno-invalid-specialization` for the installed
Xcode Clang toolchain. A Linux/CUDA training environment needs its own pinned lock and capacity
probe; the macOS file must not be reused as proof that an unknown GPU configuration is valid.

Before any run, validate the plan against the no-copy COCO receipt:

```console
python scripts/validate_openmmlab_training_plan.py \
  --plan configs/public/openmmlab-detection-smoke-plan.yaml \
  --dataset-receipt /path/to/coco-bbox-export-receipt.json \
  --training-config /path/to/exact-mmdetection-config.py
```

After training and evaluation succeed, create the non-overwriting completion receipt before the
checkpoint can be considered for candidate registration:

```console
python scripts/create_openmmlab_training_receipt.py \
  --plan /path/to/frozen-plan.yaml \
  --dataset-receipt /path/to/coco-bbox-export-receipt.json \
  --training-config /path/to/exact-mmdetection-config.py \
  --trained-artifact /path/to/best.pth \
  --metrics /path/to/final-metrics.json \
  --output /path/to/new-training-receipt.json
```

See `docs/reports/rtmdet-detection-readiness.md` for the measured smoke and capacity results.
