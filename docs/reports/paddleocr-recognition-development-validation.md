# PaddleOCR Recognition development validation

Date: 2026-10-14

## Scope and decision

This report compares the frozen project recognizer, the official
`latin_PP-OCRv5_mobile_rec` artifact, and a one-epoch project fine-tune on the same governed
221-record Recognition validation split. It measures recognition-only behavior on prepared plate
crops; it is not an end-to-end ANPR score and it is not an independent final holdout.

The fine-tuned candidate is retained but is **not promoted**. It materially improves the measured
development split, while production promotion still requires a separately frozen authorized
holdout, a product-approved confidence operating point, target-host performance evidence, and an
explicit rollout decision.

## Reproducibility evidence

- Governed inventory: 1,311 crops, comprising 1,090 train and 221 validation records.
- Inventory content SHA-256:
  `0c11168aedb13c55fb95b073366f860fa4c57599f914354897bd3da96a125b1f`.
- PaddleOCR source revision:
  `cdc6d02f680f922b2f8c3605f716d3022a9c9a0f` (`release/3.7`).
- PaddleOCR source licence: Apache-2.0; LICENSE SHA-256:
  `3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e`.
- Official pretrained parameter SHA-256:
  `065aaafb07b9e9b44fe3a86ffb24f55d3c7e6821f2b72cb6c1deb05d5f60499c`.
- Fine-tuned inference-directory SHA-256:
  `5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964`.
- Training configuration SHA-256:
  `b8d743208757a275f8c954da54e5add6d5da2edf1b77d93a28d6825cee707cfd`.
- Training environment freeze SHA-256:
  `beb3ec1b648d758b2aecfc157e4773efeac36ae4fb73769d6add518138f99344`.

The aggregate reports contain no image path, sample ID, target plate, or predicted plate text.

## Independent-holdout inventory check

A read-only inventory of retained governed evidence found development validation material and
historical documents that explicitly reserve a future fresh field holdout, but no frozen,
authorized, previously untouched Recognition holdout release. Recognition-B v4 is documented as
development/training material and its reports state that neither TEST nor field holdout was
accessed. It is therefore not relabeled as independent evidence.

No records were copied, opened for inference, reassigned, or removed during this check. The final
holdout remains a genuinely new collection/governance task; the 221-record development validation
result must not be promoted into that role.

Future intake is now guarded by the shared
[`evaluation-holdout-manifest-v1` process](../operations/evaluation-holdout-intake.md). It rejects
exact training/validation image and declared source-group overlap without storing plate text in the
audit receipt; near-duplicate and lineage review remains a mandatory separate gate.

## Results

| Candidate | Exact plate accuracy | Character error rate | p95 CPU latency | Paired result vs project |
| --- | ---: | ---: | ---: | --- |
| Frozen project recognizer | 51.58% | 19.00% | 3.06 ms | Baseline |
| Official PaddleOCR, zero-shot | 74.21% | 9.56% | 52.98 ms | 59 wins, 9 regressions, 153 ties |
| PaddleOCR, one-epoch fine-tune | 90.95% | 1.68% | 59.80 ms | 87 wins, 0 regressions, 134 ties |

Latency was measured on the current macOS arm64 CPU development host and must not be presented as
target-production performance. The fine-tuned result remained unchanged across the independent
Paddle evaluation route and the repository worker route.

## Confidence observations

The confidence curve is descriptive and was generated on the development validation split. It
must not be used to choose a final threshold after observing the final holdout.

| Threshold | Coverage | Accuracy among accepted | Wrong accepted |
| ---: | ---: | ---: | ---: |
| 0.80 | 98.19% | 92.17% | 17 |
| 0.85 | 95.93% | 93.87% | 13 |
| 0.90 | 91.40% | 95.05% | 10 |
| 0.95 | 78.28% | 96.53% | 6 |
| 0.97 | 70.14% | 96.77% | 5 |
| 0.99 | 50.23% | 99.10% | 1 |

This shows why raw accuracy alone is insufficient: increasing the threshold reduces accepted
errors but also rejects more reads. The acceptable trade-off is a product-risk decision and
remains `TBD`.

## Training and storage outcome

The smoke run used batch size 16, learning rate 0.0001, seed 1024, `RecAug`, and no `RecConAug`.
The latter was intentionally disabled because concatenating unrelated plates can create invalid
training examples. The run completed one epoch on CPU and exported a native inference artifact.

Redundant optimizer, latest, iteration, and duplicate-best checkpoint files were removed after
hashing and export. The retained run directory decreased from approximately 766 MiB to 76 MiB.
The best parameters, state metadata, configuration, log, and inference export remain available.
No Detection PNG copy was created.

## Remaining acceptance gates

1. Freeze an authorized independent final holdout whose identities were not used for training,
   model selection, or threshold exploration.
2. Declare numeric acceptance limits for exact accuracy, high-confidence wrong rate, reject rate,
   and critical slices before opening that holdout.
3. Evaluate detector-produced and perspective-rectified crops, not only the prepared crop track.
4. Benchmark on the actual deployment hardware and validate native/export numerical parity for
   the selected serving backend.
5. Keep the current project recognizer as rollback and activate the candidate only through an
   explicit owner-approved shadow/rollout decision.
