# Professional ANPR model migration roadmap

Last updated: 2026-10-14

## 1. Decision and scope

The product goal is no longer to maximize architecture originality. The goal is to minimize
measured end-to-end ANPR error while preserving licensing clarity, deterministic deployment,
observability, and rollback. The current project-owned Detection and Recognition models remain
immutable baselines and rollback candidates until a replacement passes all acceptance gates.

No model can be assumed to be error-free. The production contract must distinguish an accepted
result from `no_plate`, invalid geometry, low-confidence Recognition, and `unresolved`. A system
that abstains on uncertain inputs is preferable to one that returns a plausible but incorrect
plate.

This roadmap supersedes the architecture preference in the earlier project-owned-model roadmap;
it does not overwrite or delete that document. Existing quadrilateral annotations, geometry
validation, perspective rectification, API v2 contracts, tests, and rollback runbooks remain
valuable and are reused.

## 2. Current system and known evidence

- Detection uses the project-owned `yolox_multiscale_v1` baseline. It predicts a single-class,
  axis-aligned plate box at three feature scales.
- The implemented project-owned quadrilateral candidate is `yolox_multiscale_quad_v2`. Its code
  and tests exist, but it has not been trained or accepted on governed corner ground truth.
- The canonical Detection v3 inventory contains 8,361 bbox-labelled images: 7,524 train and 837
  validation. These labels can train a bbox detector but cannot be treated as four-corner labels.
- The corner pilot contains 256 records. At the last verified checkpoint, 27/256 had human review
  decisions; the remaining 229 require later human review. Existing decisions must be retained.
- Recognition currently uses project-owned model families behind a common result and decoder
  boundary. These remain the baseline and rollback path.
- The private inference service still loads one fixed project-owned model pair on CPU. Provider
  manifests, exact external artifact identities, adapters, and isolated evaluation runtimes now
  exist, but no external candidate has been activated in the production route.
- The current full-suite baseline is recorded in the implementation checkpoint below and must stay
  green before any future promotion.
- Governed Recognition development evaluation now shows 51.58% exact accuracy for the frozen
  project recognizer, 74.21% for official zero-shot PaddleOCR, and 90.95% for the one-epoch
  fine-tuned PaddleOCR candidate on the same 221-record validation split. This is development
  evidence, not an independent final acceptance result.

## 3. Target architecture

```text
oriented input image
  -> plate detector
  -> bbox candidates + confidence
  -> four-corner/keypoint estimator
  -> quadrilateral geometry/quality gate
  -> perspective rectification
  -> recognition-only OCR
  -> Turkish plate profile validation + calibrated confidence
  -> accepted result | no_plate | unresolved
```

### 3.1 Detection decision

Preferred functional shape: a one-stage detector with four ordered plate keypoints. A YOLO Pose
model is the first candidate because it can predict a plate box and four explicit keypoints. A
rotated OBB alone is not the target: an OBB represents a rotated rectangle, whereas an oblique
plate may be a general perspective quadrilateral.

Two candidates enter the governed benchmark:

1. **MMDetection RTMDet + MMPose RTMPose — archived future option**: Apache-2.0 code path for a bbox detector
   followed by a four-keypoint estimator. Initial plans use random/project-owned initialization so
   no external checkpoint licence is assumed. Any future pretrained checkpoint requires separate
   exact provenance, SHA-256, training-data and licence evidence.
2. **Ultralytics YOLO Pose — inactive alternative**: not selected because Free/Pro use is AGPL-3.0
   and non-AGPL commercial use requires recorded Enterprise evidence. Existing format adapters and
   exporters remain tested interchange tooling, but no Ultralytics package or weight enters the
   candidate bundle under the current decision.

The first bbox-only benchmark may run on all governed v3 bbox labels. The corner candidate may
train only on human-approved four-corner annotations. Bbox corners must never be relabelled as
quadrilateral ground truth.

### 3.2 Recognition decision

The first external candidate is PaddleOCR `latin_PP-OCRv5_mobile_rec`, used as a
**recognition-only** model after plate rectification. The general document text detector is not
part of the ANPR path. Evaluation order:

1. immutable zero-shot checkpoint benchmark;
2. comparison with the current project-owned recognizer on the same frozen examples;
3. fine-tuning on governed plate crops and transcripts;
4. calibration and paired end-to-end evaluation;
5. promotion only if the predeclared acceptance gates pass.

The raw OCR output must remain observable. Turkish plate syntax may validate, rank, or reject a
prediction, but it must not silently replace uncertain characters and report the result as raw
model evidence.

### 3.3 Runtime and compatibility

- A provider-independent adapter boundary will isolate training frameworks from the public API.
- Every deployed artifact requires provider, architecture, model ID, exact SHA-256, source,
  licence decision, preprocessing signature, charset/keypoint schema, and export-runtime identity.
- ONNX Runtime is the default portable deployment candidate. TensorRT or OpenVINO may be selected
  only after target-host measurement. The serving backend is `TBD` until target hardware is fixed.
- The current development host is Apple arm64 with 8 CPU cores, 16 GiB memory, and an available
  PyTorch MPS backend; CUDA is unavailable. This is a development capability observation, not the
  production target-host decision. Nano/small experiments may run locally, while authoritative
  throughput claims require the still-`TBD` deployment host. CUDA/TensorRT work is outside the
  active no-GPU plan.
- PaddleOCR 3.7.0 and PaddlePaddle 3.3.1 import successfully on the current macOS arm64/Python
  3.11 development host, but PaddleOCR's dependency tree installs `opencv-contrib-python 4.10`
  beside the service's pinned `opencv-python-headless 4.12`. Because both distributions own the
  same `cv2` module, in-process co-installation is rejected. The Paddle native runtime must use an
  isolated worker/container, or the selected model must be exported to a dependency-compatible
  runtime after numerical-parity validation.
- Existing v1 output fields keep their meaning. Quadrilateral and model provenance remain additive
  v2 fields. The legacy bundle is kept deployable throughout shadow evaluation.

## 4. Data, annotation, and augmentation

### Detection data

- Preserve the existing train/validation groups and content hashes.
- Audit duplicates and near-duplicates across splits before model comparison.
- Add governed negative/no-plate examples and multi-plate examples; exact counts are `TBD` after
  private inventory.
- Complete the 256-record corner pilot, second-pass audit a stratified sample, and adjudicate
  disagreements before scaling annotation.
- Expand four-corner annotation with active learning: prioritize high disagreement, oblique,
  small, night, glare, blur, occlusion, and non-standard plate cases.

### Recognition data

- Inventory existing plate crops and transcripts without publishing plate identities.
- Normalize labels through one versioned policy; retain original labels for audit.
- Split by capture group/vehicle/session rather than individual crop to prevent leakage.
- Include detector-produced and rectified crops in evaluation; clean hand-cropped plates alone are
  insufficient.

### Augmentation

- Apply image and point transforms synchronously for corner training.
- Benchmark photometric changes, blur, compression, glare, low light, modest perspective, scale,
  and partial occlusion individually before enabling combinations.
- Reject augmentations that create impossible plates or change a transcript without updating its
  label. Synthetic plates may support training but may not replace the real-data holdout.

## 5. Training, inference, and postprocessing changes

### Training

- Pin framework, package, base-weight SHA-256, configuration, seed, dataset revision, and training
  environment for every run.
- Start with the smallest candidate likely to satisfy latency; compare at least nano/small classes
  on the same split rather than choosing the largest model by default.
- Separate bbox and corner evidence. If a single Pose model is used, report box and keypoint
  metrics separately.
- Fine-tune Recognition first with an unchanged pretrained vocabulary baseline, then test a
  plate-constrained charset/configuration. Do not claim improvement without paired evaluation.

### Inference

- Map provider-native results into internal bbox, ordered-keypoint, confidence, and provenance
  contracts.
- Keep unclipped coordinates until geometry validation; do not turn a large out-of-bounds failure
  into an apparently valid polygon through silent clipping.
- Rectify only a valid selected quadrilateral. Invalid or missing geometry becomes `unresolved`.
- Batch and precision changes require numerical-parity tests against the reference backend.

### Postprocessing

- Preserve raw Recognition text and confidence.
- Apply the existing Turkish plate profile as an explicit decoder/validator layer.
- Calibrate acceptance and rejection thresholds on a calibration split separate from final test.
- Never report a broad bbox fallback as a successful quadrilateral crop.

## 6. Tests, metrics, and acceptance criteria

Numerical thresholds are deliberately `TBD`. They will be filled before official candidate runs,
using the frozen baseline and product risk tolerance; they must not be selected after seeing the
final test result.

| Area | Required evidence | Acceptance threshold |
| --- | --- | --- |
| Bbox Detection | precision, recall, mAP50, mAP50-95, false positives/image | `TBD` |
| Corners | normalized corner error, PCK, polygon IoU, valid-quad rate | `TBD` |
| Rectification | usable crop rate, crop truncation/failure taxonomy | `TBD` |
| Recognition | exact normalized plate accuracy, CER, length accuracy | `TBD` |
| Confidence | calibration error, high-confidence wrong rate, reject rate | `TBD` |
| End to end | image-to-exact-plate accuracy including failures | `TBD` |
| Performance | cold start, p50/p95 latency, peak memory, artifact size | target host `TBD` |
| Robustness | day/night, size, angle, blur, glare, occlusion slices | no critical slice regression; numeric delta `TBD` |

Tests must include unit tests for adapters and coordinate conversion, malformed provider outputs,
checkpoint/manifest mismatch, geometry failures, preprocessing parity, deterministic evaluation,
and fail-closed missing dependencies. Integration tests must exercise project/project,
external/project, project/external, and external/external model pairs without changing public v1
semantics.

## 7. Risks, dependencies, and rollback

| Risk | Mitigation |
| --- | --- |
| Third-party licence conflict | Fail closed unless a versioned licence decision and exact artifact identity are present |
| Pretrained domain mismatch | Zero-shot benchmark, governed fine-tuning, slice metrics |
| Corner annotation scarcity | Finish pilot, audit agreement, active-learning expansion |
| Recognition improves clean crops but fails detector crops | Evaluate clean, detected, and rectified crop tracks separately |
| Framework/runtime drift | Pin dependencies and export artifacts; parity-test native and deployment runtimes |
| Confidence is over-trusted | Calibrate and measure high-confidence wrong predictions; retain unresolved state |
| Latency or memory regression | Benchmark on target hardware before promotion; retain smaller candidate |
| External supply-chain change | Store hashes, provenance, licence evidence, and an internally reproducible export receipt |

Rollback keeps the current project-owned detector, recognizer, configuration, hashes, and API
contract deployable as one immutable bundle. Model selection must be controlled by configuration,
not code deletion. Shadow comparison precedes any live switch. Live activation and rollback
rehearsal require explicit owner approval. No commit or push is made without the owner's explicit
permission; push requires a separate approval after the single final commit is approved.

## 8. Phased implementation

### Phase 0 — Governance and frozen benchmark

**Purpose:** make model selection auditable before adding heavy external runtimes.

**Work:** record provider/artifact/licence manifests; restore the full test baseline; freeze
evaluation split identities; inventory target hardware and existing Recognition data; define
provider-independent adapter protocols; mark thresholds and unknowns as `TBD`.

**Deliverables:** this roadmap, model manifest schema/configuration, licence gate, benchmark data
contract, baseline verification report.

**Validation:** schema/unit tests, missing-licence and hash-mismatch negative tests, deterministic
manifest serialization, full existing suite.

**Risks:** legal decision or target hardware remains unknown. These do not block adapter work but
block production candidate promotion.

**Estimated effort:** 2–4 engineering days; licence review and hardware decision excluded.

### Phase 1 — External Recognition baseline

**Purpose:** evaluate PP-OCRv5 without changing production routing.

**Work:** implement optional PaddleOCR adapter, pin a supported runtime, register exact model
provenance, reproduce preprocessing, add zero-shot benchmark and failure mapping.

**Deliverables:** recognition adapter, private benchmark command/report, dependency and export
receipt.

**Validation:** adapter contract tests, known-crop smoke tests, paired evaluation against the
project recognizer, native/export parity.

**Risks:** Paddle runtime/platform compatibility and generic-text domain mismatch.

**Estimated effort:** 3–5 engineering days plus benchmark runtime.

### Phase 2 — External bbox Detection baseline

**Purpose:** establish a strong plate localization baseline using all governed bbox data.

**Work:** use the selected RTMDet path; implement adapter and dataset conversion;
train nano/small candidates; evaluate and export; retain negative/multi-instance semantics.

**Deliverables:** bbox candidate checkpoints, immutable receipts, paired report, runtime adapter.

**Validation:** frozen validation metrics, error slices, export parity, target-host profile.

**Risks:** license block, domain shift, split leakage, or unacceptable target-host latency.

**Estimated effort:** 4–8 engineering days plus training compute.

### Phase 3 — Governed corner dataset

**Purpose:** obtain trustworthy supervision for perspective rectification.

**Work:** resume and finish the 256-record pilot; conduct second-pass audit and adjudication;
measure disagreement; define scale-up size through learning-curve evidence; annotate active-learning
selections.

**Deliverables:** versioned four-corner dataset, audit report, annotation guideline revision.

**Validation:** schema/geometric validation, inter-review agreement, visual audit, split lineage.

**Risks:** ambiguous physical boundaries and insufficient hard-case coverage.

**Estimated effort:** engineering 1–3 days; human annotation effort `TBD` after pilot timing.

### Phase 4 — Pose/corner candidate

**Purpose:** replace heuristic boundary estimation with model-predicted plate corners.

**Work:** train YOLO Pose or RTMPose with four ordered points; compare single-stage and optional
two-stage variants; calibrate keypoint confidence and geometry gate; export selected candidates.

**Deliverables:** corner checkpoints, training receipts, geometry policy, paired report.

**Validation:** corner/PCK/polygon metrics, rectification usability, hard-slice review, export
parity.

**Risks:** too little corner data or keypoint swaps under extreme perspective.

**Estimated effort:** 5–10 engineering days plus annotation and training compute.

### Phase 5 — Recognition fine-tuning

**Purpose:** adapt the external OCR candidate to real Turkish plates and detector crops.

**Work:** prepare governed transcripts; train/fine-tune candidate variants; calibrate confidence;
compare generic versus constrained vocabulary; preserve raw evidence.

**Deliverables:** fine-tuned Recognition checkpoints, dataset/run receipts, comparison report.

**Validation:** exact-match/CER, detector-crop evaluation, calibration and error-slice analysis.

**Risks:** noisy transcripts, leakage, synthetic-data overfit, decoder hiding raw errors.

**Estimated effort:** 5–10 engineering days plus training compute.

### Phase 6 — End-to-end integration and optimization

**Purpose:** provide one provider-independent ANPR pipeline with unchanged public compatibility.

**Work:** compose Detection, corner validation, rectification, Recognition, validation and status
mapping; add ONNX/selected backend; benchmark mixed model pairs; update model identity reporting.

**Deliverables:** integrated candidate bundle, API-compatible runtime, operational documentation.

**Validation:** full suite, end-to-end frozen set, malformed-input tests, latency/memory profile,
rollback smoke.

**Risks:** preprocessing mismatch or framework-specific numerical drift.

**Estimated effort:** 5–8 engineering days.

### Phase 7 — Shadow evaluation, acceptance, and controlled rollout

**Purpose:** prove the candidate before changing live behavior.

**Work:** run legacy and candidate bundles side by side; categorize disagreements; execute rollback
rehearsal; produce accept/revise/reject record; switch only with explicit approval.

**Deliverables:** shadow report, signed decision, release/rollback checklist.

**Validation:** all predeclared gates, target-host profile, critical-error review, rollback rehearsal.

**Risks:** production distribution differs from frozen data or high-confidence rare failures remain.

**Estimated effort:** 3–7 engineering days plus an observation window of `TBD`.

## 9. Summary and live status

| Faz | Amaç | Yapılacaklar | Teslimat | Başarı ölçütü | Efor | Durum |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Yönetişim ve benchmark | Manifest, lisans kapısı, frozen set, adaptör sınırı | Roadmap, model manifesti, baseline raporu | Kimliği/lisansı belirsiz model fail-closed; test baseline yeşil | 2–4 gün | Mühendislik kısmı tamamlandı; hedef host ve ürün kabul eşikleri `TBD` |
| 1 | External Recognition baseline | PP-OCRv5 adaptörü ve zero-shot kıyas | Adaptör ve karşılaştırma raporu | Exact-match/CER/latency ölçülebilir; eşik `TBD` | 3–5 gün | Tamamlandı: exact lisanslı artifact, izole runtime, 221 kayıtlık zero-shot ve paired kıyas hazır |
| 2 | External bbox Detection | RTMDet adaptörü, eğitim, export | Bbox checkpoint ve rapor | Frozen sette predeclared kapılar; eşik `TBD` | Aktif kapsam dışı | CUDA/RTMDet artık bekleyen iş değil. Hazırlık kanıtı arşivde korunuyor; aktif yol mevcut detector ile devam ediyor |
| 3 | Corner dataset | Pilot tamamlama, audit, active learning | Governed quad dataset | Geometri ve agreement kapıları; eşik `TBD` | 1–3 gün + manuel `TBD` | Kullanıcı kararıyla ertelendi; 27/256 review korunuyor |
| 4 | Pose/corner model | Dört keypoint eğitimi ve kalibrasyon | Pose checkpoint ve geometri politikası | Corner/rectification kapıları; eşik `TBD` | Aktif kapsam dışı | Manuel annotation ve GPU gerektiren learned-corner yolu opsiyonel geleceğe alındı; aktif teslimatı engellemiyor |
| 5 | Recognition fine-tuning | Gerçek crop/transcript eğitimi | Fine-tuned OCR ve rapor | Exact-match/CER/calibration kapıları; eşik `TBD` | 5–10 gün + compute | Geliştirme kanıtı tamamlandı: 1 epoch, 90.95% exact, 1.68% CER, güven eğrisi; bağımsız holdout ve kabul eşiği eksik |
| 6 | Entegrasyon | Provider-independent E2E ve export | Candidate bundle | E2E doğruluk, uyumluluk ve performans kapıları | 5–8 gün | No-GPU bundle ve varsayılan-kapalı fail-safe shadow sınırı doğrulandı. Aktif route değişmedi; bağımsız holdout ve hedef-host kanıtı eksik |
| 7 | Shadow/rollout | Shadow, kabul, rollback provası | Release kararı | Tüm kapılar ve açık kullanıcı onayı | 3–7 gün + pencere `TBD` | 10 görüntü/3 plaka development shadow tamamlandı: 5 kazanım, 0 regresyon. Yeni holdout, gerçek shadow penceresi ve rollout onayı eksik |

### No-GPU pivot — 2026-10-14

The owner has no suitable CUDA capacity. RTMDet and learned-corner training are therefore outside
the active scope, not failed acceptance gates or remaining delivery tasks. The active delivery path keeps the frozen project
detector and advances the isolated fine-tuned PaddleOCR recognizer. No multi-day Mac training,
cloud purchase, or unapproved pretrained checkpoint will be used. See the
[no-GPU delivery path](no-gpu-professional-anpr-path.md).

A GPU-free end-to-end development shadow was completed on 10 immutable field images representing
three physical plates from one session. Both arms shared the exact project Detection and
crop/refinement path. PaddleOCR achieved 10/10 exact versus 5/10 for the project recognizer, with
five wins and zero regressions. Steady-state mean CPU end-to-end latency was 73.03 ms versus
21.34 ms. This is small correlated development evidence, not a production claim. See the
[no-GPU shadow report](../reports/no-gpu-recognition-shadow-validation.md).

The persistent isolated worker then completed 663/663 measured CPU requests over three rounds of
the 221-crop governed development-validation population, with zero failures. Mean/p95/p99 latency
was 54.23/56.20/57.91 ms after one excluded warm-up. This closes the development-host repeatability
check; final target-host peak memory and independent acceptance evidence remain `TBD`.

The resource-safe worker now defaults to two native CPU threads, remains single-flight, and closes
the provider explicitly on shutdown. A deliberately short three-request smoke observed 865.95 MiB
maximum sampled worker RSS and 54.07 ms mean latency; it left no background worker running. This is
a development-host observation rather than a guaranteed peak or production-host acceptance result.

A disabled-by-default Recognition shadow controller now keeps the baseline authoritative under all
candidate outcomes. Development `SHADOW` mode lazily starts a managed worker, fails back on startup,
timeout, protocol, inference, or cleanup errors, opens a circuit to prevent restart loops, requires
an explicit reset, and recycles sessions after a bounded request count. No active-selection mode
or v1 service-route wiring was introduced.

The controller now provides privacy-safe aggregate lifecycle telemetry for disabled requests,
candidate successes/failures, circuit skips, session starts/closes, cleanup failures, and explicit
resets. Its snapshot carries no image/crop identifier, path, plate text, prediction, confidence, or
exception detail.

Development health reporting is side-effect free: it cannot start a candidate worker, reports only
`DISABLED`/`READY`/`CIRCUIT_OPEN`, binds the effective configuration by canonical SHA-256, and
explicitly declares the baseline authoritative and the candidate ineligible for promotion.

A deterministic synthetic control-plane rehearsal passed disabled/no-start, healthy shadow,
bounded recycling, failure fallback, circuit restart suppression, explicit reset/recovery, and
final close. It loaded no real model or private data, changed no production route, and produced an
immutable aggregate report with SHA-256
`8cd617e5086b5f88110b3e482fafedb849f8c8a95ae1801d1e54484e9f575232`.

A fail-closed Recognition promotion-readiness audit now binds the completed artifact, licence,
runtime, CPU stability, shadow, and control-rehearsal evidence. The current report records
`engineering_complete=true` but remains `BLOCKED` on ten final/external gates, with
`activation_performed=false` and `production_route_changed=false`. Its SHA-256 is
`9e3b11ea5debe12084a3968cfb80e54a718f29a3f3a40d948f24c0ee390a24aa`.

### Implementation checkpoint — 2026-10-14

- Added a strict provider manifest for project-owned, Ultralytics, OpenMMLab, and PaddleOCR
  artifacts. Runtime eligibility fails closed unless licence review is approved, and artifacts can
  be verified by exact SHA-256.
- Added an aggregate, provider-independent Recognition benchmark with exact-match/CER metrics,
  p50/p95 latency, high-confidence-wrong counts, slice metrics, deterministic private-dataset
  identity, and paired candidate win/regression comparison. Reports contain no plate text or
  private sample IDs.
- Added a dependency-optional PaddleOCR recognition adapter for the documented `rec_text` and
  `rec_score` boundary. It preserves the established public Recognition contract, converts RGB to
  provider-expected BGR, and rejects malformed or non-singleton provider results.
- Added deterministic file/directory artifact hashing. Model directories reject symbolic links,
  and the PaddleOCR factory requires matching provider, role, architecture, approved licence, and
  exact local directory hash before importing the optional runtime.
- Added a provider-independent bbox/four-keypoint output boundary with explicit Detection and
  per-corner confidence. It rejects malformed native arrays without clipping or reordering
  provider geometry and supports declared class/confidence candidate selection.
- Added an import-independent Ultralytics Pose adapter for native `boxes` and `keypoints` results.
  Four keypoints and their confidence values are mandatory; no Ultralytics package was added.
- Added a governed YOLO Pose dataset exporter and operation guide. It emits EXIF-oriented PNGs,
  `TL/TR/BR/BL` labels, split directories, `data.yaml`, and a path-free content receipt. Images
  containing any plate without approved four-corner supervision fail before output creation.
- Added a governed YOLO bbox exporter for the existing Detection releases. It preserves the legacy
  OpenCV pixel-orientation contract, supports explicit negative and multi-instance labels, rejects
  exact decoded-image duplicates, enforces optional atomic-group split isolation, and writes a
  path-free content receipt without modifying the source dataset.
- Added an external Detection training contract. A run cannot become eligible unless the exact
  base artifact, framework version, approved licence evidence, task role, and exported dataset
  receipt match the frozen plan. Completed weights and metrics are bound into an immutable receipt.
- Read-only inspection of the retained Detection v3 payload confirmed 7,524 train images/labels
  and 837 validation images/labels, no image/label stem mismatches, no repeated source stem across
  those splits, and no malformed class/field rows in the inspected labels. Its canonical manifest
  is absent from the retained evidence copy, so no replacement identity was invented and no real
  export was started.
- Added a fail-closed recovery inventory for the missing Detection v3 canonical manifest. The real
  frozen layout produced a new `detection-v3-recovered-v1` identity with 8,361 samples/instances,
  zero negative examples, 7,524 train and 837 validation records, and content SHA-256
  `3cb623f75a8d7671d5ee77411e5ab7b8070b274d5602739a009ce09f6d142281`. The manifest explicitly
  retains the missing-lineage state instead of impersonating the original release.
- Deferred the full PNG dataset materialization: the source is approximately 3.2 GiB while the
  development volume has approximately 16 GiB free. Export remains available but should run only
  when training is authorized and adequate headroom is reserved.
- Generalized the v2 quadrilateral response builder to accept governed external Detection and
  Recognition artifact identities while preserving the existing project-owned defaults and exact
  public response shape. No external model was activated in production routing.
- Added a paired end-to-end shadow evaluator. It reports baseline/candidate outcome transitions,
  exact-plate wins/regressions, high-confidence wrong predictions, latency, and day/night or other
  declared slices without serializing sample IDs or plate text. The first GPU-free development
  execution is complete; independent final shadow remains blocked on new holdout data.
- Recorded the provider decision: RTMDet/RTMPose is the selected Apache-2.0 code path and
  Ultralytics is inactive unless separate Enterprise or AGPL evidence is supplied. OpenMMLab
  training can start with random/project-owned initialization, avoiding unverified model-zoo
  checkpoint rights.
- Added and executed a no-copy COCO bbox export for all 8,361 Detection records. The annotation
  export is approximately 3.5 MiB, references hash-verified source JPEGs in place, and has SHA-256
  `a21b73661bac56fa53c93b3a83a59168e15adbf1cfc5d617bb806939a115f21d`.
- Built a pinned MMDetection 3.3.0 environment outside the main service environment. MMCV 2.1.0
  native Apple MPS/CPU ops were compiled and NMS-tested; `mmcv-lite` was rejected because it lacks
  the required native extension. The exact source revision, licence, wheel, and environment hashes
  are recorded in private evidence.
- Loaded the real no-copy COCO data through MMDetection, visually verified transformed boxes, and
  completed random-init RTMDet-tiny forward/backward/optimizer/NMS/evaluator smoke runs. A 640px,
  batch-four 128-sample capacity probe completed in 57 seconds with 1.645 seconds/batch steady
  state. Projecting the official 300-epoch random-init recipe gives approximately 257.9 hours on
  this host before validation overhead, so a multi-day local run was not started. See the
  [RTMDet local-readiness report](../reports/rtmdet-detection-readiness.md).
- Visually reviewed all 20 quarantined empty-label Detection-B records: 13 contain visible plates,
  six plate-free candidates represent only two underlying scenes, and one remains uncertain. None
  were admitted as negatives; treating two scenes as six independent negatives would inflate
  diversity and contaminate evaluation.
- Added a strict OpenMMLab YAML loader and command-line eligibility gate. The retained RTMDet smoke
  plan is bound to the approved MMDetection source revision, exact Python training config, and the
  no-copy COCO annotation hash; validation emits plan identity
  `4c8e80e359b1516a17dc3496b40cf412477600735338ab6ce8b2d0bbbf290487`. The verified macOS
  dependency set is now recorded separately from the future Linux/CUDA runtime. GPU model, VRAM,
  production batch/workers and wall-time budget remain `TBD` until a target host is selected.
- Added a dependency-free MMDetection bbox adapter at the main-runtime boundary. It accepts the
  provider's `pred_instances` arrays, preserves native XYXY geometry, separates confidence/class
  selection, accepts a legitimate empty prediction, and rejects malformed shapes, non-finite
  values, invalid scores, and degenerate boxes without importing MMDetection into the service.
- Added an immutable OpenMMLab completion receipt. It refuses missing artifacts and binds the
  eligible plan, exact Python config, dataset export, framework revision, trained checkpoint,
  metrics file, and UTC completion time by SHA-256; existing receipt files are never overwritten.
- Added the Linux/CUDA training and acceptance runbook with explicit stop conditions, measured
  capacity fields, immutable launch/receipt commands, final-data isolation, and rollback. The
  recovered COCO export has zero test records; its 837-record validation split is development
  evidence only, so final Detection acceptance population and numeric gates remain `TBD`.
- Added a privacy-safe shared final-holdout manifest and admission audit for Detection and
  Recognition. It binds authorization and the pre-frozen evaluation protocol, verifies the content
  identity of every forbidden training/validation inventory, rejects exact image and declared
  source-group leakage, carries no plate text/path, and keeps near-duplicate review explicitly
  `REQUIRED`. See [final-holdout intake](../operations/evaluation-holdout-intake.md).
- Added a strict `evaluation-protocol-v1` contract so the holdout cannot be bound to an arbitrary
  file. It requires exact baseline/candidate and hardware identities, a numeric operating threshold,
  declared slices, task-specific metric gates, and correct higher/lower-is-better comparisons.
  Product thresholds remain `TBD` until chosen before final-data access.
- Read-only Recognition inventory found 1,515 records across 805 groups. All 1,515 declare
  `training_authorization_state=NOT_GRANTED` and partially verified source provenance. They are not
  eligible for fine-tuning; benchmark use also requires an explicit permitted-use decision and a
  separately frozen evaluation split.
- Located the later canonical `recognition-b/v3` release with explicit owner training
  authorization. A no-copy PaddleOCR inventory validated 1,311 real crops: 1,090 train and 221
  validation. Its content SHA-256 is
  `0c11168aedb13c55fb95b073366f860fa4c57599f914354897bd3da96a125b1f`.
- Added an evaluation-only isolated PaddleOCR worker boundary. It passes bounded, hash-verified
  PNG bytes instead of host paths, validates strict JSON-lines requests/responses, verifies the
  exact model-directory identity before provider import, supports persistent single-flight
  subprocess use with timeout/fail-closed behavior, and does not alter the active route. Docker
  was installed but its daemon was not required.
- Built separate pinned PaddleOCR inference and training environments outside the main service
  environment. Downloaded and verified the exact official `latin_PP-OCRv5_mobile_rec` artifact;
  its packaged model card declares Apache-2.0, its directory SHA-256 is
  `ce3f608dd613dafc8329e3b781453e790deaa19a43445e36c682cc10e1422215`, and its parameter SHA-256
  is `53cdc8b481a7394bb108f96d0fb3432b0a8f392e22c7d18f06dbb2d42b8b25f9`.
- Benchmarked the frozen project recognizer and official PaddleOCR candidate on the same 221
  governed development-validation crops. Exact accuracy improved from 51.58% to 74.21% zero-shot;
  the paired comparison recorded 59 wins, 9 regressions, and 153 ties.
- Fine-tuned the official Latin PP-OCRv5 mobile recognizer for one CPU epoch using 1,090 governed
  training crops, batch size 16, learning rate 0.0001, seed 1024, and `RecAug` without plate
  concatenation. The exported inference artifact has directory SHA-256
  `5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964`.
- The fine-tuned candidate reached 90.95% exact normalized plate accuracy and 1.68% CER on the
  221-record development validation split. Against the project baseline it produced 87 wins,
  zero regressions, and 134 ties. A privacy-safe confidence curve is available; at threshold 0.90
  it covers 91.40% with 95.05% accepted accuracy, while threshold 0.99 covers 50.23% with 99.10%
  accepted accuracy. No production threshold was selected. See the
  [Recognition development-validation report](../reports/paddleocr-recognition-development-validation.md).
- Audited retained Recognition evidence for an unused final holdout. No frozen, authorized,
  previously untouched Recognition holdout release exists; retained reports explicitly describe
  that population as future work. No inference or split reassignment was performed, and the
  221-record development validation set remains development-only rather than being relabeled.
- Added and executed a no-copy end-to-end Recognition shadow runner over the frozen project
  detector and crop/refinement path. On 10 field-development images representing three physical
  plates from one session, the fine-tuned PaddleOCR arm reached 10/10 exact versus 5/10 baseline,
  with five wins, zero regressions, and no unresolved outcome. After one explicit warm-up, mean
  CPU end-to-end latency was 73.03 ms candidate versus 21.34 ms baseline. The privacy-safe report
  contains no path, sample ID, target, or prediction and is explicitly ineligible for promotion.
- Added and executed a privacy-safe persistent PaddleOCR worker soak. It verifies the immutable
  inventory, every selected crop hash, the candidate receipt, and all inference artifact files
  before execution. Three rounds over 221 crops completed 663/663 requests with zero failures;
  mean/p95/p99 latency was 54.23/56.20/57.91 ms. The report contains no sample path, label, or
  prediction and is explicitly ineligible for production promotion.
- Bounded the isolated worker to two CPU threads across PaddleX/OpenMP/BLAS/Accelerate/NumExpr,
  retained single-flight request handling, exposed read-only worker PID sampling, and explicitly
  closes the provider at shutdown. A three-request resource smoke observed 865.95 MiB maximum
  sampled RSS and left no worker process behind.
- Added a strict `DISABLED`/`SHADOW` Recognition control boundary and disabled public template.
  Disabled mode never calls the candidate factory; shadow mode always returns the project result
  as authoritative, closes and circuit-breaks a failed candidate, supports explicit reset, and
  bounds worker residency by request count. A managed lazy subprocess session connects the
  existing worker transport without changing the active v1 route.
- Added aggregate shadow health counters that make restart loops, failures, cleanup problems, and
  operator resets observable without serializing any plate, image, path, prediction, confidence,
  or exception content.
- Added a side-effect-free development health snapshot with deterministic configuration identity;
  reading health never starts PaddleOCR and cannot represent the candidate as production-active.
- Added and executed a synthetic Recognition shadow control rehearsal. All seven lifecycle and
  fallback checks passed without real model/private-data access or production-route changes; the
  aggregate output is non-overwriting and contains no inference content.
- Added and executed a typed, fail-closed Recognition promotion-readiness audit. Completed
  engineering evidence is hash-bound; inconsistent final states are rejected; missing protocol,
  holdout, near-duplicate review, final evaluation, target-host, live-shadow, rollback, and owner
  approval evidence remain explicit blockers. The auditor cannot activate or change a route.
- Removed redundant optimizer/latest/iteration checkpoint copies after export, reducing the
  retained fine-tune directory from approximately 766 MiB to 76 MiB. The best parameters, state,
  configuration, training log, and inference export remain. No Detection PNG copy was created.
- Verification: 373 tests pass and 1 expected private-bundle smoke test is skipped. The public
  candidate validator passes JSON/JSONL, YAML, schema/fixture, OpenAPI, documentation-link,
  privacy/secret/weight/dataset scan, contract, and CI checks across 352 public files. Static type
  checking passes for all 117 source and script files. Changed model-migration files pass lint;
  repository-wide
  lint remains blocked by pre-existing formatting findings in
  `deploy/docker/create_python_notices.py`, which was left untouched.
- No PaddleOCR/ONNX/Ultralytics/OpenMMLab package or model weight was installed in the retained main
  environment. The main environment remains on its pinned OpenCV/PyYAML stack; Paddle dependencies
  and weights live only behind the private isolated boundary. No route activation, commit, branch,
  push, or pull request occurred.
