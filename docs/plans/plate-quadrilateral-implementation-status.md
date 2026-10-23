# Plate quadrilateral Detection implementation status

Last updated: 2026-09-26. No commit, push, branch, pull request, training run, model release,
deployment, shadow traffic, or live routing change has been made.

| Faz | Durum | Tamamlanan mühendislik çıktısı | Kalan kapı |
| --- | --- | --- | --- |
| 0 — Baseline | Koşullu tamamlandı | V1 source/test baseline, private dataset/model identity doğrulaması, 8.361 bbox-label envanteri, architecture decisions, acceptance template | Target-host budget ve numerical thresholds |
| 1 — Annotation | Devam ediyor | V2 annotation/manifest schemas, ordered-corner validator, deterministik 256-sample private pilot, 48 review-only öneri, 208 explicit unresolved kayıt, native review UI, append-only/undo karar günlüğü ve fail-closed finalizer | Kalan 229 kaydın insan review'u, second-annotator audit, adjudication, governed annotation release |
| 2 — Data | Tamamlandı | EXIF-oriented loader, identity checks, multi/negative targets, synchronized transforms, variable-instance collate | No code gate; distribution ranges still require real-data review |
| 3 — Model/training | Koşullu tamamlandı | 13-channel head, candidate-bound decode, masked multi-instance loss, V1 warm-start receipt, v2 checkpoint and training loop | Approved data, predeclared thresholds, training runs, selected checkpoint |
| 4 — Inference/geometry | Core tamamlandı | Candidate-preserving NMS, unclipped corner inverse mapping, geometry gate, strict selection, bounded homography | Candidate-model end-to-end run and calibrated policy thresholds |
| 5 — Contract/integration | Koşullu tamamlandı | Separate private/public v2 schemas, generated types, fail-closed response mapping | Signed model identity, private route activation, UI polygon/rectified-crop activation and browser validation |
| 6 — Evaluation | Koşullu tamamlandı | Polygon IoU, normalized corner error, PCK, failure-inclusive metrics, automatic acceptance gate | Paired V1/V2 data run, target-host profile, signed accept/revise/reject record |
| 7 — Rollout | Hazır, yürütülmedi | Shadow/rollback runbook and immutable identity checklist | Accepted candidate, explicit shadow approval, rollback rehearsal, separate live-switch approval |

## Current verification

- Tam Python suite: **251 passed, 1 skipped**. The skipped test requires the intentionally absent
  `ANPR_PRIVATE_MODEL_BUNDLE`.
- Yeni pilot/review focused suite: **11 passed**; Ruff ve strict mypy temiz.
- Gerçek private pilot dry run: **256 selected / 48 proposal / 208 unresolved**. Bütün kayıtlar
  `REVIEW_REQUIRED`; bbox corners hiçbir kayıtta quad ground truth yapılmadı.
- Local review smoke: HTML, current-record API ve private image response HTTP 200; karar yazmadan
  servis kapatıldı.
- Native review checkpoint: **27/256** karar private append-only event logunda; review güvenli biçimde
  duraklatıldı ve sonraki açılışta kaldığı yerden devam edecek.
- Changed Python files: Ruff clean and strict mypy clean.
- Generated Python/TypeScript contracts: deterministic check passed.
- JSON/YAML/schema/OpenAPI/link/contract/CI/public-tree privacy validation: passed.
- Web production build, ESLint, and TypeScript type-check: passed.
- Repository-wide Ruff remains blocked by pre-existing formatting findings in
  `deploy/docker/create_python_notices.py`; that unrelated file was not changed.

## Current blockers

The next meaningful execution step is the governed private evidence sequence:

1. review/correct the 256 private pilot records and perform a second-pass audit;
2. finalize approved decisions into governed v2 annotations while preserving original split/group
   lineage;
3. fill numerical acceptance thresholds and target-host budgets before candidate selection;
4. warm-start from the already verified immutable V1 checkpoint, then run training and paired
   evaluation on approved compute;
5. present the candidate/evidence for owner decision before any runtime activation.

Until these inputs exist, v1 remains the only active release and v2 makes no accuracy claim.
