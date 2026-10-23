from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src/anpr_engine"
LAUNCHER = ROOT / "scripts/launch_anpr_v1_browser.py"
TRAINER = ROOT / "scripts/train_recognition.py"
PUBLIC_TOOLING = {
    "audit_recognition_promotion_readiness.py",
    "audit_evaluation_holdout.py",
    "export_coco_bbox_annotations.py",
    "export_ultralytics_bbox_dataset.py",
    "export_ultralytics_pose_dataset.py",
    "finalize_quadrilateral_pilot.py",
    "generate_api_contracts.py",
    "benchmark_project_recognizer.py",
    "benchmark_current_detector_paddleocr.py",
    "soak_paddleocr_worker.py",
    "create_openmmlab_training_receipt.py",
    "prepare_phase2d_context.py",
    "profile_web_ocr.py",
    "compare_paddle_backends.py",
    "launch_local_website.py",
    "deploy_learned_corners.py",
    "prepare_recognition_training_inventory.py",
    "prepare_quadrilateral_pilot.py",
    "recover_bbox_dataset_inventory.py",
    "rehearse_recognition_shadow_control.py",
    "review_quadrilateral_pilot.py",
    "review_quadrilateral_pilot_native.py",
    "validate_openmmlab_training_plan.py",
    "validate_public_candidate.py",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_runtime_imports_are_project_only() -> None:
    imported = {name for path in SOURCE_ROOT.rglob("*.py") for name in _imports(path)}
    allowed_dependencies = {
        "cv2",
        "numpy",
        "PIL",
        "pydantic",
        "starlette",
        "torch",
        "uvicorn",
        "yaml",
    }
    third_party_roots = {
        name.partition(".")[0]
        for name in imported
        if name.partition(".")[0] not in sys.stdlib_module_names
        and name.partition(".")[0] != "anpr_engine"
    }
    assert third_party_roots <= allowed_dependencies


def test_only_reviewed_project_entry_points_and_validation_tools_are_retained() -> None:
    expected = {
        LAUNCHER.name,
        TRAINER.name,
        "launch_private_inference_service.py",
        *PUBLIC_TOOLING,
    }
    assert {path.name for path in (ROOT / "scripts").glob("*.py")} == expected


def test_launcher_rejects_external_weight_and_worker_options() -> None:
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "weight" not in result.stdout.lower()
    assert "worker" not in result.stdout.lower()
    assert "secondary" not in result.stdout.lower()
    assert "local owner/development review tool" in result.stdout.lower()
    assert "not for public deployment" in result.stdout.lower()


def test_active_pipeline_is_the_approved_project_sequence() -> None:
    source = (SOURCE_ROOT / "integration/browser.py").read_text(encoding="utf-8")
    assert "PROJECT_DETECTION_BLUE_BAND_GLYPH_QUAD_V1_PROJECT_RECOGNITION_" in source
    assert "STRUCTURAL_VALIDATION_HUMAN_REVIEW" in source
