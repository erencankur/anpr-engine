# RTMDet Detection local-readiness report

Date: 2026-10-14

## Outcome

The selected MMDetection RTMDet path now has a working, isolated Apple-arm64 development runtime.
The real no-copy COCO dataset was loaded, four annotations were visually checked after the
MMDetection transform, native MMCV ops were exercised, and a random-initialized RTMDet-tiny model
completed forward, loss, backward, optimizer, NMS, checkpoint, and COCO-evaluator paths.

This is a pipeline-readiness result, not a trained Detection candidate. The active project-owned
detector is unchanged.

## Pinned environment

| Component | Identity |
| --- | --- |
| MMDetection | 3.3.0, source revision `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` |
| MMDetection licence | Apache-2.0, LICENSE SHA-256 `874b8b2e6f12306a1ff7812c068a0dbf880418b8ac1f7190382bd6c6da9f23a1` |
| MMEngine | 0.10.7 |
| MMCV | 2.1.0 with locally compiled MPS/CPU ops |
| MMCV wheel | SHA-256 `842e287f268534727c18b1be76dbe8d9d397f33183965f88117cb2ac2b66d64e` |
| PyTorch / TorchVision | 2.2.2 / 0.17.2 |
| OpenCV | 4.11.0 |
| Environment freeze | SHA-256 `0c802d844ae7fcbc4dbd2b442c8a2c61bf462a101181db640345a765fe501023` |

The environment is outside the main ANPR service environment. Building MMCV required native ops;
`mmcv-lite` was explicitly rejected because it has no `_ext` NMS implementation.

The verified macOS package versions are recorded in
`runtimes/mmdetection/requirements-macos-arm64.txt`. This lock is development evidence, not a
substitute for a separate Linux/CUDA lock and target-GPU capacity probe.

MMDetection's official installation documentation supports macOS and CPU training, and identifies
full `mmcv` as the package containing native ops:
<https://github.com/open-mmlab/mmdetection/blob/main/docs/en/get_started.md>.

## Governed dataset input

- 8,361 images and 8,361 plate boxes.
- 7,524 train and 837 validation images.
- Source content SHA-256:
  `3cb623f75a8d7671d5ee77411e5ab7b8070b274d5602739a009ce09f6d142281`.
- No-copy COCO annotation SHA-256:
  `a21b73661bac56fa53c93b3a83a59168e15adbf1cfc5d617bb806939a115f21d`.
- Existing JPEGs are referenced in place; no PNG dataset copy was created.

## Empty-label review

The older Detection-B source contained 20 quarantined empty-label records. Visual review found:

| Disposition | Record count | Independent source groups | Decision |
| --- | ---: | ---: | --- |
| Plate visible / missing annotation | 13 | 5 | Exclude from negatives |
| Visually plate-free candidate | 6 | 2 | Do not admit yet |
| Uncertain due to background/kadraj edge | 1 | 1 | Exclude |

The six candidate negatives are augmentations of only two underlying scenes. Treating them as six
independent negatives would overstate diversity, so they were not added to training or validation.
They may be retained as a grouped diagnostic sample, but a professional no-plate acceptance track
still needs independently sourced and authorized scenes.

## Smoke and capacity results

| Run | Input | Batch | Result | Elapsed / steady state |
| --- | --- | ---: | --- | ---: |
| Pipeline smoke | 4 train + 4 validation at 320 px | 1 | Train/eval path passed; random model emitted no detections | 24 s process |
| Capacity probe | 128 train at 320 px | 1 | Passed | 56 s; 0.386 s/batch steady |
| Capacity probe | 128 train at 640 px | 4 | Passed | 57 s; 1.645 s/batch steady |

At 640 px and batch four, one full 7,524-image epoch is estimated at approximately 51.6 minutes.
The official RTMDet-tiny configuration is a 300-epoch recipe, so random initialization on this Mac
would require approximately 257.9 hours before validation/checkpoint overhead—about 10.7 days of
uninterrupted training. A long local run was therefore not started.

The official RTMDet model card documents the tiny configuration and its 300-epoch COCO recipe:
<https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md>.

## Pretrained-weight gate

Using an official ImageNet/COCO checkpoint would reduce training time and usually improve
convergence, but the code repository's Apache-2.0 licence does not by itself prove the licence of
every separately downloaded checkpoint. An OpenMMLab issue requesting model-zoo checkpoint licence
clarification remains unresolved:
<https://github.com/open-mmlab/mmdetection/issues/11484>.

Accordingly, no external RTMDet/CSPNeXt checkpoint was downloaded or admitted. Candidate training
must use either:

1. random/project-owned initialization on a suitable Linux CUDA host; or
2. an exact pretrained artifact with source URL, SHA-256, training-data provenance, and explicit
   licence approval.

## Decision

The runtime and real-data pipeline are ready. Production-scale training is deferred because the
current local random-init path is computationally inefficient and pretrained-weight rights remain
unresolved. The next valid execution point is a Linux CUDA training host or approved exact
pretrained artifact. The smoke checkpoints were hashed for the private receipt and then removed;
logs and configurations were retained, and the smoke checkpoints can be reproduced.

## Reproducible launch gate

The completed smoke run is represented by
`configs/public/openmmlab-detection-smoke-plan.yaml`. Before execution,
`scripts/validate_openmmlab_training_plan.py` checks that the framework licence review is approved
and that the dataset ID, exact annotation SHA-256, and exact MMDetection Python configuration hash
match the immutable inputs. The validated plan identity is
`4c8e80e359b1516a17dc3496b40cf412477600735338ab6ce8b2d0bbbf290487`; the bound smoke config
SHA-256 is `d62338ccdc7319457acb67112d1c805de017d17c4efb208e3cf54d05f59bbebc`.

The production-scale Linux/CUDA plan is intentionally not populated yet. Device model, VRAM,
batch size, worker count, wall-time budget, and resulting capacity measurement remain `TBD` until
a target host exists; inventing those values would make the plan appear runnable when it is not.

The service-side MMDetection bbox adapter is also ready. It maps `pred_instances` boxes, scores,
and labels without importing MMDetection into the main runtime; malformed shapes, non-finite
values, invalid confidence, and degenerate boxes fail closed. It deliberately preserves
out-of-frame geometry for downstream policy rather than concealing provider behavior by clipping.

Completed training runs must use `scripts/create_openmmlab_training_receipt.py` to bind the exact
plan, training config, dataset export, checkpoint, metrics report, source revision, and completion
time. The command refuses to overwrite an existing receipt.

The target-host sequence and stop conditions are defined in
[`rtmdet-cuda-training-and-acceptance.md`](../operations/rtmdet-cuda-training-and-acceptance.md).
The recovered COCO release has no test records, so validation can support development but cannot
serve as independent final acceptance evidence.

New final-test data must enter through the shared
[`evaluation-holdout-manifest-v1` process](../operations/evaluation-holdout-intake.md), which binds
authorization and the pre-frozen evaluation protocol and rejects exact training/validation image
or declared source-group leakage before any model inference.
