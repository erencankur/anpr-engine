# Detection and Recognition final-holdout intake

## Purpose

Create a genuinely independent final-evaluation population without exposing plate text in public
receipts or reusing development data. This process is for data admission only; it does not run a
model, select a threshold, or authorize production promotion.

## Collection rules

- Collect from sessions, vehicles, cameras, and time windows not present in train or validation.
- Keep one record per declared source group in the statistical evaluation population. Additional
  correlated frames may be retained separately for robustness diagnostics.
- Detection collection must include independently sourced no-plate scenes and difficult slices;
  the two previously found negative source scenes are insufficient.
- Recognition holdout must use crops or end-to-end samples that were never used for training,
  checkpoint selection, confidence exploration, or manual error-driven development.
- Preserve original authorized data privately. Public receipts contain hashes and aggregate counts
  only—never plate text, source paths, sample IDs, or images.
- Determine sample size from the desired confidence interval, slice coverage, and acceptable rare
  error bound. Those product-risk inputs are currently `TBD`; no arbitrary count is declared here.

## Required frozen inputs

Before the data is opened for model inference, freeze:

1. candidate and baseline artifact identities;
2. normalization and inference configuration;
3. confidence operating point;
4. numeric acceptance gates and critical slices;
5. an evaluation protocol file and its SHA-256;
6. source authorization evidence and its SHA-256.

The holdout manifest references the evaluation-protocol hash. Changing thresholds or candidate
identity after results are visible requires a new development cycle and a fresh final population.

The protocol must validate as `evaluation-protocol-v1`. It records exact baseline/candidate
artifact hashes, target-hardware profile hash, operating confidence threshold, required slices,
and numeric metric gates. Detection and Recognition expose separate allowed metric sets; higher-is-
better metrics require `gte`, while error, rejection, latency, and memory limits require `lte`.
Probability-like gates must be within `[0, 1]`, and latency/memory limits must be positive.

Unknown product limits remain `TBD` outside a runnable protocol. The final protocol cannot contain
placeholder thresholds: numeric choices must be made before the holdout is opened.

## Privacy-safe manifest

Each `evaluation-holdout-manifest-v1` record contains only:

- a private-safe record identifier;
- SHA-256 of its source-group identity;
- SHA-256 of image bytes;
- SHA-256 of ground truth.

The manifest requires `authorization_state=GRANTED`, `intended_use=final_acceptance_only`, a UTC
lock timestamp, authorization-evidence SHA-256, and evaluation-protocol SHA-256. Record IDs, image
hashes, and source-group hashes must be unique.

## Leakage audit

Run the audit against every training and development-validation inventory before inference:

```console
python scripts/audit_evaluation_holdout.py \
  --manifest /path/to/private-holdout-manifest.json \
  --authorization-evidence /path/to/authorization-evidence.yaml \
  --evaluation-protocol /path/to/frozen-evaluation-protocol.json \
  --forbidden-inventory /path/to/dataset-training-inventory.json \
  --output /path/to/new-holdout-audit-receipt.json
```

Repeat `--forbidden-inventory` for every relevant release. The tool verifies the exact authorization
and structurally valid, task-matched evaluation-protocol files, verifies each inventory's own
content identity, rejects exact image overlap and declared source-group overlap, and refuses to
overwrite an existing receipt.

Exact SHA-256 comparison cannot detect crops, recompression, screenshots, or visually near-identical
frames. The emitted receipt therefore always records `near_duplicate_review_state=REQUIRED`.
Perceptual matching and a source-lineage review must be completed privately before the population
is considered sealed and eligible for final inference.

## Release decision

An audit receipt means only that the machine-checkable admission checks passed. It is not an
accuracy result and not a production approval. Final evaluation is one-time evidence after the
candidate and gates are frozen; failed gates retain the current project-owned production route.
