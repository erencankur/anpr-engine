# No-GPU professional ANPR delivery path

Date: 2026-10-14

## Decision

External RTMDet and corner-model training are deferred because no suitable CUDA host is available.
The project will not start an approximately 10.7-day random-initialized Mac training run, purchase
cloud capacity, or download a checkpoint with unresolved rights.

The active delivery path keeps the frozen project Detection model and crop/refinement policy, and
advances the fine-tuned PaddleOCR Recognition candidate through development shadow, independent
holdout, and controlled rollout. This is a pragmatic scope decision, not a claim that the current
detector is optimal.

## Candidate bundle

| Component | No-GPU candidate |
| --- | --- |
| Detection | Frozen project `yolox_multiscale_v1` |
| Boundary/crop | Existing zero-padding crop plus image-only refinement |
| Recognition | Fine-tuned `latin_PP-OCRv5_mobile_rec` behind isolated worker |
| Validation | Existing structural validation; raw Recognition evidence retained |
| Rollback | Complete current project Detection + Recognition bundle |

No PaddleOCR package or weight enters the main service environment. The candidate remains behind
the bounded subprocess protocol and is not active in production routing.

## Phases

| Phase | Purpose | Work | Deliverable | Validation | Risk | Effort | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NG0 | Freeze pivot | Record no-CUDA constraint and unchanged detector | This decision document | Roadmap consistency | Treating deferral as model acceptance | <1 day | Complete |
| NG1 | Development E2E shadow | Run both recognizers behind identical detector/crop | Privacy-safe paired report | Wins/regressions, outcome transitions, latency | Small correlated sample | <1 day | Complete |
| NG2 | CPU runtime hardening | Warm worker, timeout/error checks, repeated-run profile | Runtime profile and failure report | Repeated-run stability and bounded failure | Paddle CPU latency | 1–2 days | Development-host automation complete; target-host peak RSS remains `TBD` |
| NG3 | Independent holdout | Collect and seal new authorized field data | Holdout manifest/audit receipt | Exact/group/near-duplicate gates | Manual collection quality | Manual effort `TBD` | Waiting for data |
| NG4 | Final evaluation | Freeze numeric protocol then run once | Final paired acceptance report | Predeclared gates and slices | Too little rare-case coverage | 1–2 days | Waiting for NG3 |
| NG5 | Controlled rollout | Shadow traffic, rollback rehearsal, explicit approval | Release decision | No critical regression; target-host limits | Runtime/process failure | 2–5 days + window | Waiting for NG4 |

## Current development evidence

The GPU-free end-to-end development shadow uses 10 immutable field images representing only three
physical plates from one capture session. The project detector, selected bbox, crop, and image-only
boundary refinement are shared by both arms; only Recognition differs.

| Metric | Project Recognition | Fine-tuned PaddleOCR |
| --- | ---: | ---: |
| Exact image-level accuracy | 5/10 (50%) | 10/10 (100%) |
| Paired wins / regressions | Baseline | 5 / 0 |
| Unresolved outcomes | 0 | 0 |
| Mean CPU end-to-end latency | 21.34 ms | 73.03 ms |
| High-confidence wrong at 0.90 | Not used for decision | 0 |

Latency is steady-state after one PaddleOCR warm-up. The result is development evidence only: the
images are correlated, the population is tiny, and it is not an independent final holdout.

## CPU worker soak

The isolated worker was exercised on all 221 governed development-validation crops for three
consecutive rounds in one persistent process. One warm-up request was excluded. All 663 measured
requests completed successfully: mean 54.23 ms, p50 53.82 ms, p95 56.20 ms, p99 57.91 ms, and
maximum 170.77 ms. Round means were 55.01, 53.81, and 53.88 ms, with zero protocol/runtime
failures. The aggregate private report SHA-256 is
`40190c3849f44063b541db937a9ad64aba8bc5dd203c45de7c4ed5721ff56c55`.

This closes the repeatability/error-rate portion of NG2 on the development Mac. A separate
three-request, two-thread smoke observed at most 865.95 MiB worker RSS after warm-up; the worker
exited immediately afterward. Because RSS was sampled rather than continuously instrumented, this
is not claimed as an absolute peak or as evidence for an unknown deployment host.

The candidate control boundary is now disabled by default and exposes no production-active mode.
When explicitly placed in development `SHADOW`, it keeps the project result authoritative, opens a
circuit after candidate failure, closes the failed session, requires manual reset before retry,
and recycles a healthy worker after a bounded request count. These paths are verified with fake
subprocesses and do not load PaddleOCR.

Privacy-safe shadow telemetry is also implemented. It reports only aggregate lifecycle counts
(success, failure, circuit skip, worker start/close, cleanup failure, and reset); it cannot emit
plate text, image identity, file path, confidence, or exception content.

A side-effect-free development health snapshot exposes only `DISABLED`, `READY`, or
`CIRCUIT_OPEN`, the canonical configuration SHA-256, and those aggregate counters. Reading health
cannot start PaddleOCR, and the snapshot explicitly marks the baseline as authoritative and the
candidate as ineligible for production promotion.

The synthetic rollback/control rehearsal passed all seven declared checks without loading a real
model or private data: disabled no-start, baseline authority, healthy shadow, bounded recycling,
failure fallback, circuit restart suppression, and explicit-reset recovery. The immutable aggregate
report SHA-256 is `8cd617e5086b5f88110b3e482fafedb849f8c8a95ae1801d1e54484e9f575232`.

## Acceptance gates

Final exact accuracy, error-rate, high-confidence-wrong, rejection, latency, memory, and critical
slice thresholds remain `TBD`. They must be written into `evaluation-protocol-v1` before opening a
new holdout. The existing 10-image set and 221 Recognition validation crops cannot be reused as
final acceptance evidence.

## Deferred work

- RTMDet/CUDA work is outside the active delivery scope. Existing readiness artifacts are retained
  only as archival evidence; CUDA access is not a prerequisite for this plan.
- The 229 remaining corner annotations and learned corner model remain optional future improvement.
- Full PNG export remains unnecessary; governed inputs continue to use no-copy references.

## Remaining active work

| Item | Owner | Can start now? | Completion condition |
| --- | --- | --- | --- |
| Freeze numeric acceptance gates | Product owner + engineering | Yes, but product risk tolerance is `TBD` | Thresholds are recorded before any final evaluation data is opened |
| Obtain a new independent authorized holdout | Data owner | No suitable unused set currently exists | Group-isolated, authorized manifest passes leakage and near-duplicate audit |
| Run final paired evaluation | Engineering | After holdout and gates | One immutable baseline/candidate report passes every predeclared gate |
| Measure final target host | Engineering/operations | After deployment host is named | Cold start, p50/p95/p99 latency and peak RSS meet frozen limits |
| Controlled live shadow and target-host rollback | Engineering/operations | After final acceptance | Control-plane rehearsal is complete; target-host exercise has no critical regression and rollback succeeds |
| Activate candidate | Product owner | Only after all prior gates | Explicit activation approval; current bundle remains rollback |

The independent holdout is the only missing evidence that cannot be manufactured from the existing
data without invalidating the evaluation. Reusing the 221 validation crops or 10 field images as a
"final" test would be data leakage, so the system remains a validated development candidate until
new evidence becomes available.

The fail-closed [production-readiness gate](../operations/recognition-production-readiness.md)
binds every completed engineering artifact and reports `engineering_complete=true`. Its current
status is `BLOCKED` with ten explicit external/final-acceptance blockers; it records
`activation_performed=false` and `production_route_changed=false`. Readiness report SHA-256:
`9e3b11ea5debe12084a3968cfb80e54a718f29a3f3a40d948f24c0ee390a24aa`.

## Rollback

The active project-only bundle stays unchanged until NG4 passes and the user explicitly approves
activation. Any worker startup, timeout, artifact mismatch, accuracy-gate failure, or target-host
latency failure keeps or restores the project recognizer without changing Detection behavior.
