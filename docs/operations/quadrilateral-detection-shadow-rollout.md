# Quadrilateral Detection v2 shadow rollout and rollback

Status: **runbook prepared; rollout not authorized or executed**.

This runbook applies only after the candidate passes the predeclared gates in
`configs/public/detection-quadrilateral-acceptance-template.yaml`. It does not authorize access to
private data, a live routing change, or a public accuracy claim.

## Preconditions

- Phase 1 annotation audit has passed on the approved private dataset.
- The selected `yolox_multiscale_quad_v2` checkpoint has an immutable SHA-256, model card, training
  receipt, dataset/config identities, and a clean public/private boundary review.
- Paired V1/V2 evaluation passes every predeclared box, corner, polygon, frozen-Recognition, latency,
  CPU, and memory gate. Failures and geometry abstentions remain in the denominator.
- V1 image, V1 private model bundle, and `BLUE_BAND_GLYPH_QUAD_V1` configuration are verified and
  immediately available for rollback.
- The v2 private/public schemas and generated contract types pass; no v1 field changes meaning.
- The owner gives explicit authorization for a bounded shadow run. Live routing remains V1.

## Immutable candidate receipt

Record these values in the restricted deployment record, never in this public template:

| Item | Required value |
| --- | --- |
| V2 detector model ID | `TBD` |
| V2 detector SHA-256 | `TBD` |
| V2 config SHA-256 | `TBD` |
| Dataset/revision SHA-256 | `TBD` |
| Container image digest | `TBD` |
| Frozen Recognition SHA-256 | `25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7` |
| Geometry policy identity | `TBD` |
| Acceptance record | `TBD` |

Startup must fail closed if any configured identity differs from the mounted artifact.

## Shadow sequence

1. Deploy the v2 candidate as a separately named immutable image and read-only private bundle. Do
   not overwrite the V1 image, bundle, model registry entry, or volume.
2. Keep the existing one-active-inference, one model-bearing process, zero queue, timeout,
   cancellation, and no-retention controls.
3. Enable shadow only for the explicitly approved bounded input set. Do not duplicate arbitrary
   public uploads into a retained evaluation store.
4. Keep the user-facing response on V1. Collect only approved aggregate counters and stage resource
   metrics; never log images, crops, corners, plate text, predictions, filenames, private paths, or
   admission identifiers.
5. Compare candidate counts, selected boundary state, rejection reasons, rectification status,
   frozen-Recognition outcome, p50/p95 latency, CPU, and peak memory against the signed acceptance
   record.
6. Stop shadow immediately on any rollback trigger. Produce a privacy-safe decision record:
   `accept`, `revise`, or `reject`.

Shadow duration, input count, and alert thresholds remain `TBD` until the target host and approved
evidence are known.

## Live switch

A live switch requires a new, explicit owner approval after shadow acceptance. At switch time:

1. Verify the v2 image digest and both model SHA-256 values before readiness becomes healthy.
2. Activate the versioned v2 private route and matching public mapping together. Do not return a v2
   payload from a v1 schema route.
3. Run no-plate, valid-quad, geometry-unresolved, Recognition-unresolved, timeout, cancellation,
   hash-mismatch, and schema smoke cases.
4. Verify HTTPS/proxy behavior, one-active-inference, zero queue, no-retention cleanup, and bounded
   resource use.
5. Keep human review mandatory and release statements limited to the approved evidence.

## Rollback triggers

The numerical values come from the signed acceptance record. Roll back on any of the following:

- box recall or false positives exceed the approved regression bound;
- valid-quad rate, polygon IoU, corner error, or geometry abstention crosses its trigger;
- frozen-Recognition RAW exact or CER crosses its regression trigger;
- mirrored/flipped rectification, invalid quad entering Recognition, or contract mismatch occurs;
- p95 latency, CPU, peak memory, timeout, or cancellation behavior exceeds its budget;
- model/config hash mismatch, readiness inconsistency, privacy leakage, or retention failure occurs.

## Rollback procedure

1. Stop new admissions and allow the single active request to finish or reach its existing deadline.
2. Restore the immutable V1 image and original private bundle selection atomically.
3. Verify V1 detector and Recognition SHA-256 values, health/readiness, v1 schema, no-plate behavior,
   and `BLUE_BAND_GLYPH_QUAD_V1` crop policy.
4. Reopen admissions only after the V1 smoke checks pass.
5. Record rollback time, trigger, artifact identities, and safe aggregate impact. Do not include
   source images, crops, plate values, predictions, or private paths.
6. Leave the rejected v2 candidate disabled. A new rollout requires a new candidate identity,
   repeated gates, and explicit approval.

Rollback time objective: `TBD`. The rollback rehearsal must pass before any live switch.

## Git discipline

Implementation changes remain uncommitted until all authorized phases and tests are complete. The
final diff and evidence are presented to the owner first. A single commit requires explicit
permission; push requires separate explicit permission.
