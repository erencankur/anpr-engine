# Ultralytics bbox dataset and training handoff

This workflow prepares the governed Detection v1/v2/v3 bbox data for a professional detector. It
does not install Ultralytics, download weights, start training, or authorize production use.

## Dataset export

Run `scripts/export_ultralytics_bbox_dataset.py` with the private Detection manifest, dataset root,
and a new output directory. The source dataset is read-only and an existing output directory is
never replaced.

Before creating output, the exporter requires:

- a supported dataset ID and lowercase content SHA-256;
- unique, filename-safe sample IDs;
- non-empty train and validation splits;
- paths that remain inside the declared dataset root;
- decodable color images and valid class-0 normalized YOLO bbox rows;
- no duplicate decoded image content; and
- when atomic group IDs are supplied, complete coverage with no group crossing splits.

Empty label files remain true negative samples. Multiple valid class-0 rows remain multi-plate
samples. The current released dataset may still contain only one plate per image; the exporter
does not invent missing negative or multi-plate examples.

Images are decoded and re-encoded using the same OpenCV pixel orientation used by the legacy
Detection loader. This avoids silently applying a new EXIF transform to bbox coordinates whose
annotation orientation contract predates the professional migration. The path-free receipt names
this policy as `opencv-imread-color-bgr-v1`.

### Missing canonical manifest recovery

If the frozen image/label layout remains available but its canonical manifest does not,
`scripts/recover_bbox_dataset_inventory.py` creates a **new** recovered dataset identity. It does
not claim that the lost lineage was reconstructed. The resulting manifest records the prior
dataset ID and declared prior hash separately, marks its lineage state as
`RECOVERED_FROM_FROZEN_LAYOUT_CANONICAL_MANIFEST_UNAVAILABLE`, hashes every image and label, rejects
unpaired files and duplicate decoded images, and computes its own content SHA-256.

The recovered manifest contains private relative file paths and must remain in governed private
evidence storage. The command prints only aggregate counts and identities. Existing output is
never overwritten. A recovered manifest may support a controlled candidate run, but its lineage
gap remains a release risk and must not be silently relabelled as the original canonical release.

The self-contained exporter converts images to PNG and therefore needs additional disk capacity.
It is not selected for the current local workflow. The selected RTMDet path uses
`scripts/export_coco_bbox_annotations.py`, which writes only COCO JSON and references the existing
hash-verified JPEG files in place. It creates no second image tree. The 8,361-image real export is
approximately 3.5 MiB instead of duplicating the 3.2 GiB source dataset.

## Training gate

A runnable `UltralyticsTrainingPlan` must freeze the exact framework version, base-model ID and
SHA-256, dataset-export SHA-256, random seed, image size, epochs, batch/workers, early stopping,
device, optimizer, and augmentation values. Placeholder values such as `TBD` are rejected.

Training eligibility is denied unless all of these conditions hold:

1. the base artifact is an Ultralytics artifact in the correct Detection or corner role;
2. its licence review is approved with reviewer and date;
3. its model ID, framework version, and SHA-256 match the frozen plan;
4. the local weight bytes match the recorded SHA-256; and
5. the dataset receipt schema, dataset ID, and exported-content SHA-256 match the plan.

After a real run, `ExternalTrainingReceipt` binds the plan identity, base weight, exported dataset,
trained weight, metrics file, and completion timestamp. Numerical acceptance thresholds remain
`TBD` until the frozen baseline and product risk tolerance are approved; they must be declared
before reading final-test results.
