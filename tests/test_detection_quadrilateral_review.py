from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from anpr_engine.detection.quadrilateral_pilot import canonical_json_bytes, queue_content_sha256
from anpr_engine.detection.quadrilateral_review import (
    PilotReviewDecision,
    PilotReviewEvent,
    append_review_event,
    finalize_pilot_review,
    latest_review_state,
    load_review_events,
)

SHA = "a" * 64
CORNERS = [
    {"x": 0.1, "y": 0.2},
    {"x": 0.8, "y": 0.2},
    {"x": 0.8, "y": 0.6},
    {"x": 0.1, "y": 0.6},
]


def test_append_only_review_log_and_latest_state(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = append_review_event(
        path,
        sample_id="sample-1",
        image_sha256=SHA,
        label_sha256=SHA,
        decision=PilotReviewDecision.CORRECTED,
        corners=CORNERS,
        reason="HUMAN_CORRECTED_CORNERS",
    )
    second = append_review_event(
        path,
        sample_id="sample-1",
        image_sha256=SHA,
        label_sha256=SHA,
        decision=PilotReviewDecision.PENDING,
        corners=None,
        reason="UNDO_LAST_DECISION",
    )

    events = load_review_events(path)
    assert [event.sequence for event in events] == [1, 2]
    assert first.recorded_at.tzinfo is not None
    assert latest_review_state(events)["sample-1"] == second


@pytest.mark.parametrize("decision", ["APPROVED_PROPOSAL", "CORRECTED"])
def test_approved_decisions_require_valid_corners(decision: str) -> None:
    with pytest.raises(ValidationError, match="require corners"):
        PilotReviewEvent.model_validate(
            {
                "sequence": 1,
                "sample_id": "sample-1",
                "image_sha256": SHA,
                "label_sha256": SHA,
                "decision": decision,
                "corners": None,
                "reason": "INVALID",
                "recorded_at": "2026-09-26T00:00:00Z",
            }
        )


def test_unusable_decision_forbids_fabricated_corners() -> None:
    with pytest.raises(ValidationError, match="forbid"):
        PilotReviewEvent.model_validate(
            {
                "sequence": 1,
                "sample_id": "sample-1",
                "image_sha256": SHA,
                "label_sha256": SHA,
                "decision": "UNUSABLE",
                "corners": CORNERS,
                "reason": "INVALID",
                "recorded_at": "2026-09-26T00:00:00Z",
            }
        )


def _queue(path: Path) -> None:
    records = [
        {
            "sample_id": "sample-1",
            "split": "train",
            "cohort": "A",
            "canonical_image_path": "train/A/images/sample-1.jpg",
            "canonical_label_path": "train/A/labels/sample-1.txt",
            "image_sha256": SHA,
            "label_sha256": "b" * 64,
            "image_width": 400,
            "image_height": 120,
            "box_xywh_normalized": [0.5, 0.5, 0.8, 0.5],
            "review_state": "REVIEW_REQUIRED",
            "proposal": None,
            "proposal_reason": "TEST",
        }
    ]
    payload = {
        "schema_version": "plate-quadrilateral-pilot-queue-v1",
        "dataset_id": "detection-v3",
        "dataset_content_sha256": "c" * 64,
        "selection": {},
        "policy": {},
        "summary": {},
        "records": records,
        "records_sha256": queue_content_sha256(records),
    }
    path.write_bytes(canonical_json_bytes(payload))


def test_finalize_requires_complete_review_and_writes_non_training_release(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    events = tmp_path / "events.jsonl"
    output = tmp_path / "release"
    _queue(queue)
    with pytest.raises(ValueError, match="1 records remain"):
        finalize_pilot_review(queue, events, output)

    append_review_event(
        events,
        sample_id="sample-1",
        image_sha256=SHA,
        label_sha256="b" * 64,
        decision=PilotReviewDecision.CORRECTED,
        corners=CORNERS,
        reason="HUMAN_CORRECTED_CORNERS",
    )
    manifest = finalize_pilot_review(queue, events, output)

    assert manifest["status"] == "REVIEW_COMPLETE_NOT_TRAINING_RELEASE"
    assert manifest["summary"] == {
        "selected": 1,
        "accepted_geometry": 1,
        "unusable_geometry": 0,
    }
    assert (output / "annotations/sample-1.json").is_file()
    assert manifest["training_release_blockers"] == [
        "SECOND_PASS_AUDIT_REQUIRED",
        "ATOMIC_GROUP_LINEAGE_REQUIRED",
        "ACCEPTANCE_THRESHOLDS_REQUIRED",
    ]
    assert finalize_pilot_review(queue, events, output) == manifest
