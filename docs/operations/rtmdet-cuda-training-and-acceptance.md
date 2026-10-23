# RTMDet CUDA training and acceptance runbook

## Scope

This runbook starts only after a Linux/NVIDIA CUDA host is available. It does not authorize cloud
spend, external checkpoint download, final-test access, production activation, or model promotion.
The current project-owned detector remains the rollback route throughout.

## Stop conditions

Stop before training if any of the following is true:

- MMDetection source revision, code licence, training config, or dataset receipt does not match the
  frozen plan;
- the runtime imports `mmcv-lite` or native NMS is unavailable;
- the target host cannot complete the 640 px capacity probe without an out-of-memory error;
- train/validation grouping or annotation integrity differs from the governed COCO export;
- an external pretrained artifact lacks an exact URL, SHA-256, provenance, and approved licence;
- numeric acceptance limits or the final evaluation population are being chosen after looking at
  final-test results.

## Stage 1 — Target-host preflight

Record Python, PyTorch, CUDA runtime, GPU model, VRAM, MMEngine, MMCV, MMDetection, source revision,
and dependency-lock SHA-256 in private evidence. Verify native NMS with a minimal tensor smoke.
Exact target values are `TBD` until the host exists.

Do not reuse the macOS lock as Linux/CUDA evidence. Create a separate immutable environment lock.

## Stage 2 — Capacity probe

Run the frozen 128-record, 640 px capacity probe. Start with batch size one and increase only until
the host has a measured stable setting. Record:

| Field | Required value |
| --- | --- |
| GPU and VRAM | `TBD` from host |
| Batch size | `TBD` from measured probe |
| Workers | `TBD` from measured probe |
| Precision | `TBD`; must be parity-tested |
| Peak allocated/reserved memory | `TBD` from probe |
| Steady seconds per batch | `TBD` from probe |
| Estimated epoch and run duration | `TBD` from probe |

An out-of-memory retry may change only the declared capacity parameters. It must not silently alter
image size, augmentation, dataset, seed, architecture, or evaluation policy.

## Stage 3 — Freeze the executable plan

Create a new `openmmlab-training-plan-v1` document with the measured device, batch, and worker
values. Bind the exact MMDetection Python config SHA-256 and no-copy COCO receipt. Validate before
launch:

```console
python scripts/validate_openmmlab_training_plan.py \
  --plan /path/to/frozen-plan.yaml \
  --dataset-receipt /path/to/coco-bbox-export-receipt.json \
  --training-config /path/to/exact-mmdetection-config.py
```

The current random-initialization smoke plan is not the production training plan. Production
epochs, batch size, workers, device, and wall-time budget remain `TBD` until capacity evidence is
available.

## Stage 4 — Train and evaluate

Train only on the governed train split. Validation may be used for checkpoint selection and
development error analysis. Save the selected checkpoint and a machine-readable metrics report.
Do not call validation results independent final-test evidence.

The current recovered COCO release contains 7,524 train and 837 validation records but zero test
records. A final acceptance claim therefore requires a separately governed locked evaluation
population or explicitly authorized access to a preserved historical test release. That population
must remain unopened until the candidate identity and numeric gates are frozen.

## Stage 5 — Seal outputs

Create a completion receipt without overwriting prior evidence:

```console
python scripts/create_openmmlab_training_receipt.py \
  --plan /path/to/frozen-plan.yaml \
  --dataset-receipt /path/to/coco-bbox-export-receipt.json \
  --training-config /path/to/exact-mmdetection-config.py \
  --trained-artifact /path/to/best.pth \
  --metrics /path/to/final-metrics.json \
  --output /path/to/new-training-receipt.json
```

The receipt binds plan, config, dataset, framework revision, checkpoint, metrics, and completion
time. A checkpoint without this receipt is ineligible for candidate registration.

## Stage 6 — Acceptance and rollback

Before opening final evaluation data, freeze `TBD` numeric gates for localization accuracy,
critical misses, false positives on no-plate scenes, hard slices, latency, and memory. Compare the
candidate with the active detector on the same population. Promotion requires all gates, export or
serving parity, end-to-end shadow evidence, and explicit user approval.

Any gate failure keeps the active project-owned detector unchanged. Preserve the failed candidate
and its receipt for diagnosis; do not overwrite it or tune against the opened final population.
