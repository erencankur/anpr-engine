# Plate quadrilateral Detection Phase 0 baseline

Status: **engineering baseline recorded; private identities verified; corner-label gate remains
open**.

This record freezes public-repository facts and records non-identifying results from a read-only
private inventory. Private images, record identifiers, and absolute private paths are not copied
into the public tree.

## Source state

| Item | Frozen value |
| --- | --- |
| Git revision | `92c973f8067cc931abb5f67d79359119055b719d` |
| Branch at inspection | `main` |
| Detector architecture | `yolox_multiscale_v1` |
| Detector version | `detection-v3-20260827` |
| Detector SHA-256 | `0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea` |
| Detection input | 384 x 384 letterbox |
| Feature strides | 8, 16, 32 |
| Runtime confidence threshold | 0.15 in the frozen `AnprV1Pipeline` |
| Runtime NMS threshold | 0.30 in the frozen `AnprV1Pipeline` |
| Selected candidate | Highest-confidence surviving box, deterministic bbox tie-break |
| Crop policy | Zero-padding box crop followed by `BLUE_BAND_GLYPH_QUAD_V1` |
| Result schema | `anpr-v1-browser-result-v1` |

The standalone inference defaults differ from the frozen pipeline (`0.25` confidence and `0.50`
NMS in `DetectorInferenceConfig`). Quadrilateral v2 must use one explicit, versioned policy rather
than inherit either value silently.

## Published development-validation reference

These figures are copied from the existing public model registry. They are not reproduced by this
Phase 0 run because the private dataset and detector checkpoint are intentionally absent.

| Metric | Published value |
| --- | ---: |
| Precision | 66.45% |
| Recall | 95.58% |
| F1 | 78.39% |
| mAP@0.50 | 95.09% |
| mAP@0.50:0.95 | 56.47% |
| Top-1 correct | 94.86% |

These values are development-validation evidence only. They are not independent field or
production-accuracy claims.

## Reproduced public test baseline

The following focused V1 regression suite passed before implementation work began:

```text
tests/test_detector_foundation.py
tests/test_detector_training.py
tests/test_plate_boundary_refinement.py
tests/test_anpr_v1_browser.py
tests/test_phase2b_inference_service.py

71 passed
```

The run used CPython 3.11.16. No private model, dataset, image, annotation, training, or live
inference was used.

## Architecture decisions frozen before implementation

1. The first candidate architecture is `yolox_multiscale_quad_v2`: the existing backbone and
   FPN/PAN remain, while each dense head gains an eight-value corner branch.
2. The canonical corner order is top-left, top-right, bottom-right, bottom-left on the decoded,
   EXIF-orientation-corrected original image.
3. Boxes remain the first candidate-assignment and NMS geometry. The same candidate index owns its
   corner values.
4. The v1 architecture, checkpoint format, runtime path, API contract, and heuristic crop policy
   remain loadable and unchanged.
5. A v2 quadrilateral that fails validation returns an explicit unresolved/manual-review outcome.
   It must not silently claim a legacy box/heuristic crop as a model-predicted quadrilateral.
6. The Recognition checkpoint, decoder, and input contract stay frozen during paired evaluation.
7. External Detection, segmentation, or OCR weights are not introduced.

## Acceptance and rollback values still requiring private evidence

The public repository cannot determine the following values. They must be filled in the restricted
copy of `configs/public/detection-quadrilateral-acceptance-template.yaml` before the first official
candidate-selection run:

- approved dataset identity, split/group hashes, counts, and true-corner label coverage;
- annotation audit rate and maximum inter-annotator normalized corner error;
- corner-error normalization and PCK thresholds;
- valid-quad, polygon-IoU, box-parity, Recognition-regression, and abstention thresholds;
- target deployment hardware and latency/CPU/memory budgets;
- shadow duration, rollback triggers, and rollback-time objective.

## Phase 0 gate result

| Gate | Result | Required action |
| --- | --- | --- |
| Public V1 source/test baseline | PASS | Preserve with regression tests |
| Architecture and coordinate decisions | PASS | Encode in v2 types/tests |
| Private model/data availability | VERIFIED OUTSIDE PUBLIC TREE | Keep all payloads in approved private boundary |
| True outer-corner annotations | ABSENT; 8.361 canonical labels are bbox-only | Complete governed 256-record pilot; never substitute bbox corners |
| Numerical acceptance thresholds | TBD | Fill from approved baseline before model selection |
| Target-host resource budget | TBD | Measure and predeclare before release decision |

Engineering may proceed with schema, validators, geometry primitives, synthetic fixtures, and V1
compatibility tests. Training, official candidate selection, and rollout remain blocked until the
open private-evidence gates are resolved.
