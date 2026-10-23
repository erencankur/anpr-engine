# Project Detector + PaddleOCR development shadow report

Date: 2026-10-14

## Outcome

The no-GPU candidate path is technically viable. With the frozen project detector and identical
crop/refinement behavior, the fine-tuned PaddleOCR candidate was correct on all 10 development
images; the project recognizer was correct on five. The paired result is five wins, zero
regressions, and five ties.

This is not a production accuracy claim. The population contains only three physical plates from
one capture session and was already designated `DEVELOPMENT_REGRESSION`.

## Frozen identities

| Evidence | SHA-256 |
| --- | --- |
| Baseline bundle manifest | `85ed6fd0b64510cbb76c0219b61a29133f06a758b3344870a8e50227353ed326` |
| PaddleOCR candidate receipt | `6b8ae8c53a044cf632d272d855ded29078e31956c751ebfe88d3e525540ea18b` |
| PaddleOCR inference directory | `5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964` |
| Paired development population | `c48235e64d2b3a70eb6edde62d0c95e5f50d1acdade0934d3d20614f6f8b6a29` |
| Warm steady-state aggregate report | `e1499f3d292e9c5199782a2e4f04604707249b479a9abff4fb9cd03b8d55c6cf` |
| Persistent CPU worker soak report | `40190c3849f44063b541db937a9ad64aba8bc5dd203c45de7c4ed5721ff56c55` |
| Bounded CPU/RSS smoke report | `b9ca8498006efc9d507c8d192ac4027fac1e8678c24c83fce8db24b6381fb44e` |

Every input image was resolved through the governed Detection inventory and verified by exact file
SHA-256. No image was copied and no final holdout was accessed.

## Results

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Population | 10 | 10 |
| Exact accuracy | 50% | 100% |
| Prediction outcomes | 10 | 10 |
| No-plate / geometry / Recognition unresolved | 0 | 0 |
| Mean CPU end-to-end latency | 21.34 ms | 73.03 ms |
| Candidate high-confidence wrong at 0.90 | — | 0 |

The candidate adds approximately 51.69 ms mean steady-state latency on this development Mac CPU.
The measurement excludes candidate cold startup by performing one explicit warm-up request. A
preliminary cold-start-inclusive report was retained privately rather than overwritten.

## Privacy and interpretation

The retained aggregate report contains no source path, sample ID, target plate, or predicted plate.
The same frozen Detection result and crop/refinement policy feed both Recognition arms, so the
measured paired difference is attributable to Recognition behavior for this population.

The ten images are not independent observations: they represent three plates captured in one
session. The 100% result therefore supports continuing the candidate but cannot select a production
threshold, estimate rare-error risk, or authorize rollout.

## Persistent worker stability

The candidate worker processed the complete 221-crop governed development-validation index three
times in one persistent CPU process. After one excluded warm-up, 663/663 measured requests
succeeded with zero worker-protocol or provider-runtime failures.

| Metric | Result |
| --- | ---: |
| Mean / p50 | 54.23 / 53.82 ms |
| p95 / p99 | 56.20 / 57.91 ms |
| Maximum | 170.77 ms |
| Round means | 55.01 / 53.81 / 53.88 ms |

This is a runtime stability measurement, not another accuracy evaluation. A separate bounded
smoke used two CPU threads, one warm-up, and three measured requests. It observed 865.95 MiB
maximum sampled worker RSS and 54.07 ms mean latency, then closed the process. Sampling does not
establish an absolute peak; the final deployment-host profile remains `TBD`.

## Decision

Continue with the no-GPU bundle: frozen project Detection plus isolated fine-tuned PaddleOCR.
Remove RTMDet/CUDA work from the active scope. Before activation, collect a new authorized holdout,
freeze numeric gates,
run the paired evaluation once, and profile the actual deployment host. The current project-only
bundle remains the active route and rollback.
