# Project-owned quadrilateral plate Detection roadmap

Status: **planning only**. This document does not authorize training, a model replacement, deployment, or a public accuracy claim.

## Objective and current boundary

The desired output is the visible **four-corner boundary of the physical plate**, not an axis-aligned rectangle around it. For a fronto-parallel plate the four corners form a rectangle; for an oblique plate they form a perspective quadrilateral. The selected plate region will be warped to a fronto-parallel rectangle and only that rectified plate image will be prepared for the existing project-owned Recognition model.

The current project-owned Detection model predicts objectness and an axis-aligned `xyxy` box at three feature-map scales. The runtime selects the highest-ranked box, crops it, and then applies `BLUE_BAND_GLYPH_QUAD_V1`, an **image-processing heuristic** that may estimate a quadrilateral from glyphs and the blue band. That heuristic is not a detector-predicted plate boundary and can return the box crop unchanged. The web page currently draws the box on the source image. The owner's example illustrates the mismatch, but its image and plate text must not be copied into the public repository or treated as an independent evaluation sample.

Target data flow:

```text
One uploaded image
    → project-owned Detection: plate candidate + four ordered corners
    → quadrilateral validity/quality gate
    → perspective warp of the original, orientation-corrected image
    → aspect-preserving Recognition preprocessing
    → existing project-owned Recognition model
    → prediction and explicit human review
```

The current live release and its two immutable weight identities stay unchanged until a separately reviewed candidate passes the gates below. No external Detection, segmentation, or OCR weights are introduced.

## Scope and non-goals

- In scope: corner annotations on **already authorized** source images, a project-owned corner-prediction extension, geometric rectification, versioned API/UI contracts, controlled evaluation, resource checks, and rollback.
- Out of scope for this work: collecting more source images, changing the Recognition model or its labels, adding an external model, claiming independent field accuracy, changing the public daily limit or one-active-inference policy, and writing the deferred article.
- A four-corner polygon is sufficient for a planar plate. Pixel-level segmentation masks are not required unless evidence shows that corners cannot represent the useful plate boundary.
- The illustrative owner image is design context, not training or selection evidence. Public documentation will use synthetic or separately cleared examples.

## Phase 0 — Feasibility and frozen baseline

1. Inventory the **private** Detection images and annotations without publishing identities, filenames, plate transcriptions, record-level examples, or absolute private paths. Establish whether true outer-plate corner labels already exist; the current public training loader accepts a single class-0 YOLO box, which does not itself provide corners.
2. Freeze the current model, crop policy, Recognition model, image-orientation behavior, and development-validation procedure. Record immutable hashes privately. Reproduce baseline Detection localization, current crop success/failure, Recognition RAW exact/CER/length, latency, CPU, and memory on approved development evidence. Publish only approved percentages and non-identifying summaries.
3. Define the canonical coordinate system: top-left origin, coordinates on the **decoded, EXIF-orientation-corrected original image**, ordered `top-left → top-right → bottom-right → bottom-left`. Document image boundary conventions and letterbox inverse mapping.
4. Make an explicit go/no-go decision on labels. **Do not use a bounding box's four corners as a substitute for the true plate corners**: that would teach the current error. If true corner labels are absent, annotate a governed subset of existing permitted images and independently audit it. If re-annotation is not approved or sufficient geometry cannot be verified, supervised corner-model training stops; a separate geometry-only baseline may be studied, but must not be called detector-predicted corners.

Exit gate: an auditable, privacy-safe baseline and an approved source of four-corner ground truth, or a documented stop decision.

## Phase 1 — Corner labels and geometry contract

1. Define each corner at the visible outer boundary of the plate face, consistently excluding plate holders and surrounding vehicle body. Explicitly mark clipped, occluded, blurred, glared, or ambiguous boundaries as uncertain rather than inventing precise coordinates.
2. Keep existing source images and governed split membership intact. Audit repeated captures, physical-plate identity, near duplicates, and capture sessions so a source or identity does not leak across training and validation. An annotation-only revision must not silently become a new image sample.
3. Require ordered, in-bounds, non-self-intersecting, convex quadrilaterals with a minimum usable area. Audit a sample with a second pass and resolve disagreement according to a written corner-placement rule. Store private image/annotation provenance and revision hashes.
4. Implement synchronized transforms for images, boxes, and corners. Horizontal flip must also reorder corner identities; affine/perspective augmentation must transform every corner and reject invalid geometry. Keep transformation parameters realistic and avoid augmentations that create unreadable training targets without an explicit reason.
5. Derive a box from the annotated quadrilateral for the existing candidate assignment and backward comparison, while retaining the original four points as the authoritative crop target.

Exit gate: validated annotation schema, reproducible transforms, split-integrity checks, and a reviewed ambiguity policy.

## Phase 2 — Project-owned Detection extension

**Preferred first experiment:** extend each existing decoupled multi-scale head with eight corner values, while retaining the current objectness and box outputs. Reuse the project-owned backbone/FPN-PAN and initialize compatible parameters from the frozen Detection checkpoint; initialize the new corner branch separately. The old checkpoint itself remains immutable. A dense corner branch is the lowest-complexity starting point and preserves a single model pass.

1. Specify how corner offsets are encoded relative to the assigned plate candidate/grid cell, decoded back to image coordinates, and paired with each post-NMS candidate. Verify that letterbox padding and inverse transforms are identical for box and corners.
2. Train with a controlled corner loss plus the existing detection objectives. Mask unlabeled/uncertain corner targets; define loss weights, assignment policy, and selection metric **before** inspecting validation results. Track box and corner gradients independently.
3. Compare bounded alternatives only if the direct head lacks corner precision. A lightweight project-owned ROI corner refiner is a possible second experiment, but only after proving that its extra CPU and memory fit the shared-host budget. It must remain part of the project-owned Detection pipeline; do not substitute an external model.
4. Keep deterministic seeds, dataset membership hashes, checkpoint receipts, model/config identity, training logs, and one official candidate-selection rule. Development-validation data may select checkpoints; exposed examples are diagnostic only and may not tune the final gate.

Exit gate: reproducible corner predictions with acceptable box-detection parity and a candidate checkpoint that passes geometric and resource preflight. Loss reduction alone is insufficient.

## Phase 3 — Exact plate extraction and rectification

1. Validate the predicted quadrilateral before use: finite coordinates, expected corner order, convexity, no self-intersection, minimum edge lengths/area, bounded aspect and perspective distortion, and containment within the original image. Avoid silently clipping large out-of-bounds errors into apparently valid corners.
2. Map the valid four source points to a rectangle using a homography. Infer target width and height from opposing edge lengths, impose a bounded output pixel budget, and handle interpolation/border behavior explicitly. The warped image must represent the selected plate face, not a loose axis-aligned vehicle crop.
3. Feed the rectified RGB image through the current aspect-preserving Recognition resize/padding contract; keep Recognition weights and decoder frozen for this experiment. Preserve source-coordinate corners, warp dimensions, method/version, and safe status metadata without retaining uploaded images server-side.
4. If no valid quadrilateral exists, return an **unresolved/manual-review** outcome for the new route; do not quietly pass a broad box crop to Recognition while claiming polygon-only processing. The legacy box-plus-heuristic path remains available only as a separately identified rollback/shadow baseline until replacement is approved.
5. Explicitly test near-fronto-parallel plates, strongly oblique plates, tiny plates, partially visible plates, edge-of-frame plates, reflections, motion blur, and no-plate images. Check that rectification does not flip or mirror the text.

Exit gate: deterministic geometry tests and traceable proof that the new Recognition input is derived exclusively from the selected valid quadrilateral.

## Phase 4 — API, review UI, and privacy-safe integration

1. Introduce a versioned private/public result contract with `quadrilateral_xy_normalized` (four ordered points), boundary state, rectification state/method, and derived legacy `box_xyxy_normalized` for compatible consumers. Do not silently change the meaning of existing v1 fields; update schemas and generated clients together.
2. Draw the four connected edges and corner markers on the **original** image in the EN/TR interface. Label the preview as a Detection boundary, not a Recognition crop. Show the rectified crop only if explicitly supported without server retention or metadata leakage.
3. Ensure JSON export, annotated-image generation, screen-reader text, and owner review all distinguish a valid quadrilateral, unresolved geometry, no plate, and Recognition failure. Keep sensitive image/prediction content out of logs, metrics, traces, and persistent server storage.
4. Keep one active inference, zero unbounded queue, daily request policy, upload validation, private backend boundary, and model-hash readiness checks intact. No extra always-on worker is permitted solely for the corner head.

Exit gate: updated contract/schema tests, EN/TR UI checks, cancellation/timeout tests, no-retention tests, and hash-mismatch fail-closed behavior.

## Phase 5 — Controlled comparison and decision

Compare on the **same approved development-validation inputs**, with the same frozen Recognition model:

| Dimension | Required comparison |
| --- | --- |
| Localization | Box recall/false positives versus baseline; polygon IoU and corner-distance error for the new model |
| Geometry | Valid-quad rate, failed/uncertain rate, crop boundary coverage, rectification orientation, and warp artifacts |
| Recognition | RAW exact, character error, plate-length behavior, and regressions after the crop change |
| Operations | Per-stage latency, peak memory, CPU under one active request, cold start, timeout, and cancellation |
| Safety | Invalid-input and no-plate behavior, privacy leakage, model-hash integrity, and rollback |

Predeclare numerical acceptance thresholds after Phase 0 baseline inspection and **before** model selection. At minimum, require no material Detection recall regression, a demonstrated improvement in plate-boundary localization and/or frozen-Recognition outcomes, and resource use compatible with the shared Coolify host. Report failures and abstentions rather than excluding them from denominators. Do not convert development or previously exposed images into an independent real-world accuracy claim. Independent field evaluation is outside the owner's current scope, so any release claims must remain correspondingly narrow.

Decision: **accept, revise, or reject** the new candidate on documented evidence. A model that produces attractive polygon overlays but worsens OCR or shared-host stability is not accepted.

## Phase 6 — Shadow rollout, release, and rollback

1. Add the new model as a separately versioned private checkpoint and registry/model-card entry; verify its SHA-256 at startup. Do not overwrite or remove the current official Detection checkpoint.
2. First run an opt-in, bounded shadow comparison with no production routing change. Review polygon overlay, rectified crop, Recognition output, errors, and resource use using permitted development evidence.
3. Request separate owner approval before switching live routing. Build both images from the final reviewed source, inspect contents for private weights/data, mount private weights read-only, and preserve the one-worker resource limits. The prior release must remain immediately restorable.
4. After any approved switch, confirm live schema compatibility, HTTPS/proxy behavior, privacy controls, and resource limits. On bad geometry, OCR regression, timeout, memory pressure, or contract incompatibility, roll back to the frozen prior release and its original crop policy.

## Delivery and Git discipline

The requested deliverable **now** is this roadmap only. No training, new annotation, model/API/UI code, container build, DNS/Coolify change, deployment, history rewrite, or public claim is authorized by creating it.

For later authorized implementation, keep all related work scoped and reviewed, then land it as **one new commit on the public repository's `main`** through the existing pull-request/CI policy (for example, a squash merge). Do not create incremental public commits or push an unfinished candidate. Before that final commit, run Python/web tests, schema and generated-contract checks, type/lint/build checks, privacy/public-tree scans, model-identity verification, and a final diff review. Record rejected experiments privately when they contain sensitive evidence.

The deferred article and the skipped extended monitoring, load test, manual accessibility audit, and independent field evaluation do not become implied prerequisites of this development roadmap; their absence must remain visible in any release or accuracy statement.
