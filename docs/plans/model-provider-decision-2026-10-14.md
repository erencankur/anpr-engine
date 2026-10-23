# Professional model provider decision — 2026-10-14

## Decision

Ultralytics is not the primary ANPR provider. The professional migration will use:

- MMDetection RTMDet for bbox Detection;
- MMPose RTMPose for four-corner prediction after governed corner labels are ready; and
- PaddleOCR recognition-only architecture for OCR evaluation and fine-tuning.

The existing project-owned models remain the immutable baseline and rollback bundle.

## Licence rationale

Ultralytics Free/Pro use is published under AGPL-3.0, while non-AGPL commercial use requires an
Enterprise licence. This project does not have recorded Enterprise evidence, and the engineering
workflow cannot accept AGPL obligations on the owner's behalf. Therefore no Ultralytics package or
weight is admitted to the planned bundle. Existing Ultralytics-format exporters remain useful as
an interchange format and passive fallback tooling.

MMDetection and MMPose code are published under Apache-2.0. The code-path decision is approved for
the candidate architecture, subject to preserving required licence/NOTICE material. This approval
does **not** automatically approve arbitrary model-zoo checkpoints. Initial RTMDet/RTMPose plans use
`random_project_owned` initialization. Any future external checkpoint requires its own exact URL,
SHA-256, training-data provenance, and licence approval.

The MMDetection 3.3.0 source/runtime and RTMDet-tiny random-initialization path have now passed a
real-data Apple MPS smoke and capacity test. The environment is technically runnable, but the
projected 300-epoch local workload is approximately 257.9 hours before validation overhead. The
open OpenMMLab checkpoint-licence clarification request does not provide sufficient evidence to
approve model-zoo weights automatically. Production-scale RTMDet training therefore requires a
Linux CUDA host or a separately approved exact pretrained artifact.

PaddleOCR code is published under Apache-2.0 and the official model documentation lists
`latin_PP-OCRv5_mobile_rec` as a Latin-script recognition model supporting Turkish and digits.
The exact official inference artifact was downloaded and verified. Its packaged model card declares
`license: apache-2.0`; the model-card SHA-256 is
`897e932f55386124163c5d051ee42a7481fad52ea447c9e3b009d22e5424bada`, the inference-directory
SHA-256 is `ce3f608dd613dafc8329e3b781453e790deaa19a43445e36c682cc10e1422215`, and the
`inference.pdiparams` SHA-256 is
`53cdc8b481a7394bb108f96d0fb3432b0a8f392e22c7d18f06dbb2d42b8b25f9`. That exact artifact is
approved for controlled candidate evaluation. This decision does not approve differently named or
changed Paddle artifacts.

PaddleOCR runs behind the repository's isolated private worker boundary during evaluation. This
avoids the observed OpenCV namespace conflict with the main service. Separate pinned inference and
training environments were built outside the main service environment; neither changes production
routing. A one-epoch governed fine-tuning run produced an auditable candidate, but that candidate
is retained for development only until an independent final holdout and an owner-approved
confidence operating point exist.

Official references:

- <https://docs.ultralytics.com/platform/account/billing/>
- <https://github.com/open-mmlab/mmdetection/blob/main/LICENSE>
- <https://github.com/open-mmlab/mmdetection/issues/11484>
- <https://github.com/open-mmlab/mmpose/blob/main/LICENSE>
- <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE>
- <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.en.md>

## Recognition data decision

The 1,515-record `recognition-b-source-v1` release remains excluded because every record says
`training_authorization_state=NOT_GRANTED`. Those records are not modified or reclassified.

The authorized source is canonical `recognition-b/v3`, whose owner-controlled build evidence says
`owner_recognizer_training_authorization=GRANTED`. A no-copy inventory validated 1,311 records:
1,090 train and 221 validation. Its inventory content SHA-256 is
`0c11168aedb13c55fb95b073366f860fa4c57599f914354897bd3da96a125b1f`.

This removes the data-authorization blocker for controlled training. It does not turn the
validation split into an independent final test set; final acceptance still requires a separately
frozen, permitted holdout.

## Storage decision

Do not materialize the 3.2 GiB Detection dataset as PNG. PNG is lossless but photographs often
become substantially larger than JPEG, so a second self-contained copy could consume much of the
approximately 16 GiB currently free on the development volume. Training will use a verified
no-copy index or the existing JPEG layout. The PNG exporter remains tested but is not part of the
selected local workflow.
