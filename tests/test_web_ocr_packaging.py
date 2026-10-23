from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_paddle_build_context_contains_runtime_but_no_private_data(tmp_path: Path) -> None:
    context = tmp_path / "context"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_phase2d_context.py"),
            "inference-paddleocr-amd64",
            str(context),
        ],
        check=True,
        capture_output=True,
    )
    assert (context / "runtimes/paddleocr-worker/worker.py").is_file()
    assert (context / "runtimes/paddleocr-worker/native_backend.py").is_file()
    assert (context / "runtimes/lean-smoke/smoke.py").is_file()
    assert (context / "deploy/docker/buildkit.low-resource.toml").is_file()
    assert (context / "src/anpr_engine/recognition/web_runtime.py").is_file()
    assert (context / "src/anpr_engine/recognition/worker_protocol.py").is_file()
    assert (context / "deploy/docker/requirements.paddleocr-amd64.txt").is_file()
    assert (context / "deploy/docker/requirements.paddle-native-amd64.txt").is_file()
    config = json.loads((context / "configs/public/web-ocr-runtime-template.json").read_text())
    assert config["python"] == "/opt/paddle-venv/bin/python"
    assert config["worker"] == "/app/runtimes/paddleocr-worker/worker.py"
    assert config["backend"] == "paddle_inference"
    learned = json.loads(
        (context / "configs/public/web-ocr-learned-corners-template.json").read_text()
    )
    assert learned["corner_model_size"] == 640
    assert learned["cpu_threads"] == 1
    assert learned["idle_seconds"] == 15
    assert (context / "src/anpr_engine/detection/hyperlpr_corners.py").is_file()
    assert (context / "src/anpr_engine/detection/quadrilateral.py").is_file()
    dockerfile = (context / "deploy/docker/inference.paddleocr.amd64.Dockerfile").read_text()
    assert "configs/public/web-ocr-learned-corners-template.json" in dockerfile
    assert "/app/configs/web-ocr-learned-corners.json" in dockerfile
    assert not any(path.suffix in {".pt", ".onnx", ".pdiparams"} for path in context.rglob("*"))
    assert not (context / "tests").exists()
    assert not (context / "datasets").exists()

    # Test the actual exported context, not the full working tree. In editable
    # installs a missing module can otherwise be found through a site finder.
    code = r"""
import importlib.abc, importlib.machinery, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
class ContextOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'paddle', 'paddleocr', 'paddlex'}:
            raise ImportError('unexpected parent framework: ' + fullname)
        if fullname == 'anpr_engine' or fullname.startswith('anpr_engine.'):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if (spec is None or not spec.origin
                    or not pathlib.Path(spec.origin).resolve().is_relative_to(root)):
                raise ImportError('module missing from exported context: ' + fullname)
            return spec
sys.meta_path.insert(0, ContextOnly())
from anpr_engine.inference_service.app import create_app
from anpr_engine.integration.lean_corners import LeanCornerPipeline
assert 'torch' not in sys.modules and 'paddle' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(context / "src")],
        check=True,
        capture_output=True,
        text=True,
    )
    # Reproduce the first Linux build failure: the guard must not fill this
    # missing file from the developer's complete editable checkout.
    (context / "src/anpr_engine/detection/quadrilateral.py").unlink()
    broken = subprocess.run(
        [sys.executable, "-I", "-c", code, str(context / "src")],
        capture_output=True,
        text=True,
    )
    assert broken.returncode != 0
    assert (
        "module missing from exported context: anpr_engine.detection.quadrilateral" in broken.stderr
    )


def test_paddle_compose_selects_both_new_images_and_read_only_model_mount() -> None:
    override = yaml.safe_load((ROOT / "deploy/docker/compose.paddleocr-candidate.yaml").read_text())
    inference = override["services"]["inference"]
    web = override["services"]["web"]
    assert inference["image"] == "${ANPR_PADDLE_INFERENCE_IMAGE}"
    assert web["image"] == "${ANPR_PADDLE_WEB_IMAGE}"
    assert web["environment"]["ANPR_RECOGNITION_PROVIDER"] == "paddleocr"
    config = json.loads((ROOT / "configs/public/web-ocr-runtime-template.json").read_text())
    mount = inference["volumes"][0]
    assert mount["target"] == config["model_dir"]
    assert mount["read_only"] is True
    assert mount["bind"]["create_host_path"] is False
    assert "ports" not in inference


def test_learned_corner_overlay_pairs_provider_config_and_read_only_weight() -> None:
    overlay = yaml.safe_load(
        (ROOT / "deploy/docker/compose.learned-corners-candidate.yaml").read_text()
    )
    inference = overlay["services"]["inference"]
    web = overlay["services"]["web"]
    config = json.loads((ROOT / "configs/public/web-ocr-learned-corners-template.json").read_text())
    assert inference["environment"]["ANPR_WEB_OCR_CONFIG"] == (
        "/app/configs/web-ocr-learned-corners.json"
    )
    assert web["environment"]["ANPR_DETECTION_PROVIDER"] == "hyperlpr640"
    assert web["environment"]["ANPR_RECOGNITION_PROVIDER"] == "paddleocr"
    assert config["corner_model_size"] == 640
    assert config["backend"] == "paddle_inference"
    mount = inference["volumes"][0]
    assert mount["target"] == config["corner_model"]
    assert ":?" in mount["source"]
    assert mount["read_only"] is True
    assert mount["bind"]["create_host_path"] is False
    assert "ports" not in inference
    assert "image" not in inference  # Explicitly selected by the preceding Paddle overlay.


def test_low_resource_corner_defaults_bound_idle_and_logs_without_lowering_model_quality() -> None:
    config = json.loads((ROOT / "configs/public/web-ocr-learned-corners-template.json").read_text())
    assert config["idle_seconds"] == 15
    assert config["cpu_threads"] == 1
    assert config["corner_model_size"] == 640
    assert config["minimum_confidence"] == 0.99
    overlay = yaml.safe_load(
        (ROOT / "deploy/docker/compose.learned-corners-candidate.yaml").read_text()
    )
    for service in overlay["services"].values():
        assert service["logging"] == {
            "driver": "json-file",
            "options": {"max-size": "5m", "max-file": "3"},
        }
        # Container limits remain unchanged until measured in the actual host.
        assert "mem_limit" not in service
        assert "build" not in service


def test_lean_image_has_one_environment_without_torch_or_legacy_bundle() -> None:
    requirements = (ROOT / "deploy/docker/requirements.lean-corners-amd64.txt").read_text()
    names = {
        line.split("==")[0]
        for line in requirements.splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    }
    assert "paddlepaddle" in names and "starlette" in names
    assert not {"torch", "paddleocr", "paddlex"}.intersection(names)
    dockerfile = (ROOT / "deploy/docker/inference.lean-corners.amd64.Dockerfile").read_text()
    assert "ANPR_RUNTIME_PROFILE=lean_corners" in dockerfile
    assert "ANPR_PRIVATE_MODEL_BUNDLE=" not in dockerfile
    assert "ANPR_INFERENCE_BASE_IMAGE" not in dockerfile
    assert "COPY --from=builder" in dockerfile
    overlay = yaml.safe_load(
        (ROOT / "deploy/docker/compose.lean-corners-candidate.yaml").read_text()
    )
    inference = overlay["services"]["inference"]
    assert inference["environment"]["ANPR_RUNTIME_PROFILE"] == "lean_corners"
    assert {mount["target"] for mount in inference["volumes"]} == {
        "/models/corners/y5fu_640x_sim.onnx",
        "/models/paddleocr/inference",
    }
    assert "ANPR_PRIVATE_MODEL_BUNDLE" not in inference["environment"]
    assert "ports" not in inference


@pytest.mark.parametrize("layout", ["v2", "v1", "missing"])
def test_smoke_cgroup_snapshot_reports_current_and_peak(
    monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "lean_smoke", ROOT / "runtimes/lean-smoke/smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def read(path: Path, *args: object, **kwargs: object) -> str:
        name = path.as_posix()
        if layout == "missing" or (layout == "v1" and "/memory/" not in name):
            raise FileNotFoundError
        return str((256 if "peak" in name or "max_usage" in name else 128) * 1024**2)

    monkeypatch.setattr(Path, "read_text", read)
    values = module.memory_snapshot()
    assert values == (
        {"current_mib": None, "peak_mib": None}
        if layout == "missing"
        else {"current_mib": 128.0, "peak_mib": 256.0}
    )


def test_dedicated_builder_configuration_is_serial_without_gc() -> None:
    config = tomllib.loads((ROOT / "deploy/docker/buildkit.low-resource.toml").read_text())
    assert config["worker"]["oci"]["max-parallelism"] == 1
    assert config["worker"]["oci"]["gc"] is False
