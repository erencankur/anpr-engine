"""Append-only human decisions for a private quadrilateral pilot queue."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self, cast

from pydantic import Field, model_validator

from anpr_engine.configuration import StrictConfigModel
from anpr_engine.detection.annotations import NormalizedPoint
from anpr_engine.detection.quadrilateral import Quadrilateral, validate_normalized_quadrilateral
from anpr_engine.detection.quadrilateral_pilot import canonical_json_bytes, queue_content_sha256

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]


class PilotReviewDecision(StrEnum):
    APPROVED_PROPOSAL = "APPROVED_PROPOSAL"
    CORRECTED = "CORRECTED"
    UNUSABLE = "UNUSABLE"
    PENDING = "PENDING"


class PilotReviewEvent(StrictConfigModel):
    schema_version: str = "plate-quadrilateral-pilot-review-event-v1"
    sequence: Annotated[int, Field(gt=0)]
    sample_id: Identifier
    image_sha256: Sha256
    label_sha256: Sha256
    decision: PilotReviewDecision
    corners: tuple[NormalizedPoint, NormalizedPoint, NormalizedPoint, NormalizedPoint] | None
    reason: Annotated[str, Field(min_length=1, max_length=256)]
    recorded_at: datetime

    @model_validator(mode="after")
    def validate_decision_geometry(self) -> Self:
        requires_corners = self.decision in {
            PilotReviewDecision.APPROVED_PROPOSAL,
            PilotReviewDecision.CORRECTED,
        }
        if requires_corners != (self.corners is not None):
            raise ValueError(
                "approved/corrected decisions require corners; other decisions forbid them"
            )
        if self.corners is not None:
            quadrilateral = cast(Quadrilateral, tuple((point.x, point.y) for point in self.corners))
            validate_normalized_quadrilateral(quadrilateral)
        return self


def load_review_events(path: Path) -> tuple[PilotReviewEvent, ...]:
    if not path.exists():
        return ()
    events = tuple(
        PilotReviewEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if [event.sequence for event in events] != list(range(1, len(events) + 1)):
        raise ValueError("review event sequence is not contiguous")
    return events


def latest_review_state(events: tuple[PilotReviewEvent, ...]) -> dict[str, PilotReviewEvent]:
    state: dict[str, PilotReviewEvent] = {}
    for event in events:
        state[event.sample_id] = event
    return state


def append_review_event(
    path: Path,
    *,
    sample_id: str,
    image_sha256: str,
    label_sha256: str,
    decision: PilotReviewDecision,
    corners: list[dict[str, float]] | None,
    reason: str,
) -> PilotReviewEvent:
    events = load_review_events(path)
    event = PilotReviewEvent.model_validate(
        {
            "sequence": len(events) + 1,
            "sample_id": sample_id,
            "image_sha256": image_sha256,
            "label_sha256": label_sha256,
            "decision": decision,
            "corners": corners,
            "reason": reason,
            "recorded_at": datetime.now(UTC),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(existing.model_dump_json().encode() + b"\n" for existing in (*events, event))
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return event


def load_pilot_queue(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != (
        "plate-quadrilateral-pilot-queue-v1"
    ):
        raise ValueError("unsupported quadrilateral pilot queue")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("quadrilateral pilot queue has no records")
    identifiers = [record.get("sample_id") for record in records if isinstance(record, dict)]
    if len(identifiers) != len(records) or len(set(identifiers)) != len(records):
        raise ValueError("quadrilateral pilot queue sample IDs are invalid or duplicated")
    expected_records_sha256 = value.get("records_sha256")
    if expected_records_sha256 is not None and expected_records_sha256 != queue_content_sha256(
        records
    ):
        raise ValueError("quadrilateral pilot queue record identity changed")
    return cast(dict[str, Any], value)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_identical_or_new(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to replace different finalized review output: {path}")
        return
    _atomic_write(path, payload)


def finalize_pilot_review(queue_path: Path, events_path: Path, output_root: Path) -> dict[str, Any]:
    """Materialize reviewed annotations without claiming a training-ready release."""
    queue = load_pilot_queue(queue_path)
    events = load_review_events(events_path)
    state = latest_review_state(events)
    records = cast(list[dict[str, Any]], queue["records"])
    incomplete = [
        record["sample_id"]
        for record in records
        if record["sample_id"] not in state
        or state[record["sample_id"]].decision is PilotReviewDecision.PENDING
    ]
    if incomplete:
        raise ValueError(f"pilot review is incomplete: {len(incomplete)} records remain")

    reviewed: list[dict[str, Any]] = []
    accepted = 0
    unusable = 0
    for record in records:
        event = state[record["sample_id"]]
        if (
            event.image_sha256 != record["image_sha256"]
            or event.label_sha256 != record["label_sha256"]
        ):
            raise ValueError(f"review event source identity mismatch: {record['sample_id']}")
        annotation_path: str | None = None
        annotation_sha256: str | None = None
        if event.decision in {
            PilotReviewDecision.APPROVED_PROPOSAL,
            PilotReviewDecision.CORRECTED,
        }:
            if event.corners is None:
                raise ValueError("accepted review event is missing corners")
            annotation = {
                "schema_version": "plate-quadrilateral-annotation-v2",
                "image_sha256": event.image_sha256,
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "instances": [
                    {
                        "instance_id": "plate-1",
                        "class_id": "license_plate",
                        "corners": [point.model_dump(mode="json") for point in event.corners],
                        "corner_visibility": ["VISIBLE"] * 4,
                        "geometry_usable_for_rectification": True,
                        "geometry_reason": "FULLY_VISIBLE",
                    }
                ],
            }
            annotation_payload = canonical_json_bytes(annotation)
            annotation_path = f"annotations/{record['sample_id']}.json"
            annotation_sha256 = hashlib.sha256(annotation_payload).hexdigest()
            _write_identical_or_new(output_root / annotation_path, annotation_payload)
            accepted += 1
        else:
            unusable += 1
        reviewed.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "cohort": record["cohort"],
                "canonical_image_path": record["canonical_image_path"],
                "source_image_sha256": record["image_sha256"],
                "source_label_sha256": record["label_sha256"],
                "review_decision": event.decision,
                "review_event_sequence": event.sequence,
                "annotation_path": annotation_path,
                "annotation_sha256": annotation_sha256,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "plate-quadrilateral-reviewed-pilot-v1",
        "status": "REVIEW_COMPLETE_NOT_TRAINING_RELEASE",
        "dataset_id": queue["dataset_id"],
        "dataset_content_sha256": queue["dataset_content_sha256"],
        "queue_records_sha256": queue["records_sha256"],
        "review_event_count": len(events),
        "summary": {
            "selected": len(records),
            "accepted_geometry": accepted,
            "unusable_geometry": unusable,
        },
        "training_release_blockers": [
            "SECOND_PASS_AUDIT_REQUIRED",
            "ATOMIC_GROUP_LINEAGE_REQUIRED",
            "ACCEPTANCE_THRESHOLDS_REQUIRED",
        ],
        "records": reviewed,
    }
    manifest["content_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_identical_or_new(
        output_root / "reviewed-pilot-manifest.json", canonical_json_bytes(manifest)
    )
    return manifest
