# Quadrilateral annotation pilot operation

This workflow prepares and reviews true plate corners without modifying the canonical source
dataset. Queue files, decisions, images, and annotations remain in the private evidence boundary.

## Frozen pilot decision

- Source: governed `detection-v3` canonical images and YOLO box labels.
- Scope: at most 256 records selected deterministically across split, cohort, box-area quantile,
  and aspect-ratio quantile with seed `20260926`.
- Proposal method: pixel-only `BLUE_BAND_GLYPH_QUAD_V1`.
- Every record requires human review. A proposal is never ground truth by itself, and bbox corners
  are never substituted for physical plate corners.
- Review order is `TL -> TR -> BR -> BL` on the EXIF-oriented original image.

## Prepare the private queue

Run `scripts/prepare_quadrilateral_pilot.py` with the private canonical dataset root, an output path
inside private evidence, the governed dataset ID/content SHA-256, `--maximum-samples 256`, and
`--seed 20260926`.

The command is deterministic and refuses to replace a different existing queue. It records source
image/label hashes and only portable paths relative to the private dataset root.

## Review locally

The preferred UI is `scripts/review_quadrilateral_pilot_native.py`. Run it with the private queue,
canonical dataset root, and an events path inside private evidence. It reads pixels directly from
disk and does not require a browser or local HTTP server. The browser-based
`scripts/review_quadrilateral_pilot.py` remains a fallback.

For each sample:

1. Confirm that the yellow box points to the intended physical plate.
2. If the green proposal follows the visible outer plate face, approve it.
3. Otherwise click `TL`, `TR`, `BR`, `BL` and save the correction.
4. Mark the sample unusable when the true boundary cannot be supported by visible evidence.

Keyboard shortcuts: `A` approves an available proposal, `U` marks unusable, `Enter` saves four
clicked points, `R`/`Escape` resets points, and `Command-Z` undoes the latest saved decision.

Decisions are append-only and validated before writing. **Son kararı geri al** appends a `PENDING`
compensation event instead of deleting history. Closing the window is safe; reopening resumes from
the first pending record. Review events contain hashes and geometry, not image bytes or plate
transcription.

## Finalize after review

After all records have a decision, run `scripts/finalize_quadrilateral_pilot.py` with the queue,
event log, and a new output directory inside private evidence. It validates queue and source hashes,
requires a complete final state, writes one v2 annotation document per accepted/corrected sample,
and emits `reviewed-pilot-manifest.json`.

The finalized pilot deliberately has status `REVIEW_COMPLETE_NOT_TRAINING_RELEASE`. It remains
blocked from official training until second-pass audit, atomic-group lineage, and numerical
acceptance thresholds are recorded.

## Exit gate

Do not start official corner training until every pilot record has a decision, a second-pass audit
has measured disagreement, original atomic-group lineage has been restored in the governed v2
manifest, and numerical acceptance thresholds have been frozen. Pilot proposal rate or attractive
overlays alone are not an acceptance criterion.
