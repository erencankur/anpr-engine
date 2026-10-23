from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rehearse_recognition_shadow_control.py"
SPEC = importlib.util.spec_from_file_location("rehearse_recognition_shadow_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_control_rehearsal_passes_without_inference_content() -> None:
    report = MODULE.run_rehearsal()
    assert report["passed"] is True
    assert report["uses_real_model"] is False
    assert report["uses_private_data"] is False
    assert report["production_route_changed"] is False
    serialized = json.dumps(report, sort_keys=True)
    assert "SYNTHETIC-BASELINE" not in serialized
    assert "SYNTHETIC-CANDIDATE" not in serialized
    assert "raw_text" not in serialized
    assert "confidence" not in serialized


def test_control_rehearsal_output_is_non_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "rehearsal.json"
    output.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--output", str(output)])
    with pytest.raises(ValueError, match="already exists"):
        MODULE.main()
    assert output.read_text(encoding="utf-8") == "existing\n"
