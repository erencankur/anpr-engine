# Ultralytics Pose dataset export

This operation converts a governed `detection-quadrilateral-manifest-v2` release into a
self-contained YOLO Pose dataset. It does not make heuristic proposals, infer missing corners, or
convert bbox corners into plate-boundary ground truth.

## Preconditions

- Every non-negative image must contain complete annotations for every visible plate instance.
- Every instance must have four human-approved, visible corners in `TL, TR, BR, BL` order and be
  marked usable for rectification.
- Train and validation splits must both be non-empty, atomic-group lineage must already be
  governed, and image SHA-256/dimensions must match each annotation.
- The output directory must not exist. An existing export is never overwritten.

## Command

Run `scripts/export_ultralytics_pose_dataset.py` with the governed manifest, its private dataset
root, and a new private output directory. The command prints only the aggregate export receipt; it
does not print source image paths or plate text.

The exporter writes EXIF-oriented RGB PNG images, YOLO Pose label files, `data.yaml`, and
`export-receipt.json`. The configuration uses four keypoints, three values per keypoint, and flip
mapping `[1, 0, 3, 2]` so horizontal augmentation preserves `TL/TR/BR/BL` identities.

## Failure behavior

The export stops before creating its output directory when it finds an identity mismatch,
dimension mismatch, missing train/validation split, or any plate instance without approved corner
supervision. This prevents an unlabelled plate from becoming a false negative during Pose
training.

The export receipt records dataset identity, aggregate sample/instance/split counts, and a content
hash. It deliberately omits absolute paths, image bytes, filenames from the source dataset, and
plate transcriptions.
