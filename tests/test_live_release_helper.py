from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "live_release", ROOT / "scripts/deploy_learned_corners.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(
        (ROOT / "configs/public/live-corners-template.json").read_text()
    )
    value.update(public_origin="https://plates.test", internal_credential="a" * 48)
    return value


@pytest.mark.parametrize("available", [512, 4096, None])
def test_doctor_is_read_only_and_never_claims_production_ready(
    monkeypatch: pytest.MonkeyPatch, available: int | None
) -> None:
    module = helper()
    original_read = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.as_posix() == "/proc/meminfo":
            if available is None:
                raise OSError("unavailable")
            return f"MemTotal: 8388608 kB\nMemAvailable: {available * 1024} kB\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    report = module.doctor()
    assert report["host_memory"]["available_mib"] == available
    assert report["production_readiness"] == "NOT_VERIFIED"
    assert report["docker_cli_found"] is False
    assert not {"hostname", "ip", "env", "credential", "model_paths"}.intersection(report)
    assert any("RAM" in warning for warning in report["warnings"]) == (available != 4096)


@pytest.mark.parametrize("version", ["2.40.0", "UNEXPECTED_SENSITIVE_OUTPUT"])
def test_doctor_runs_only_bounded_compose_version_and_filters_output(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    module = helper()
    calls: list[list[str]] = []

    def fake(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert kwargs["timeout"] == 10
        return subprocess.CompletedProcess(args, 0, version, "")

    monkeypatch.setattr(module.shutil, "which", lambda _: "/docker")
    monkeypatch.setattr(module.subprocess, "run", fake)
    report = module.doctor()
    assert calls == [["docker", "compose", "version", "--short"]]
    assert report["compose_version"] == (version if version == "2.40.0" else None)
    assert "UNEXPECTED_SENSITIVE_OUTPUT" not in json.dumps(report)


def test_config_requires_private_permissions_and_real_settings(tmp_path: Path) -> None:
    module = helper()
    path = tmp_path / "live.json"
    path.write_text(json.dumps(config()))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="600"):
        module.load_config(path)
    path.chmod(0o600)
    assert module.load_config(path)["public_origin"] == "https://plates.test"
    value = config()
    value["public_origin"] = "http://plates.test"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="HTTPS"):
        module.load_config(path)


@pytest.mark.parametrize("action", ["check", "start", "stop"])
def test_actions_are_bounded_and_do_not_build_push_or_delete(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    module = helper()
    calls: list[list[str]] = []
    value = config()
    value["release_checks_confirmed"] = True

    def fake(args: list[str], env: dict[str, str] | None = None) -> str:
        calls.append(args)
        assert env is not None
        assert env["ANPR_INTERNAL_SERVICE_CREDENTIAL"] == "a" * 48
        assert "COMPOSE_FILE" not in env
        overlay = Path(args[args.index("-f") + 1])
        assert overlay.name == "compose.public-candidate.yaml"
        origin = json.loads(Path(args[args.index(str(overlay)) + 6]).read_text())
        assert origin["services"]["web"]["ports"] == ["127.0.0.1:39084:3000"]
        return ""

    monkeypatch.setattr(module, "run", fake)
    monkeypatch.setattr(module, "check_models", lambda _: None)
    monkeypatch.setattr(module, "image_id", lambda _: "sha256:" + "1" * 64)
    monkeypatch.setenv("COMPOSE_FILE", "/unrelated/compose.yaml")
    module.compose(value, action)
    assert len(calls) == (1 if action == "check" else 2)
    assert not any(
        word in {"build", "push", "down", "rm", "prune"} for call in calls for word in call
    )
    if action == "start":
        assert "--wait" in calls[-1] and "--no-build" in calls[-1]
    if action == "stop":
        assert "stop" in calls[-1]


def test_start_without_release_confirmation_never_calls_up(monkeypatch: pytest.MonkeyPatch) -> None:
    module = helper()
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "check_models", lambda _: None)
    monkeypatch.setattr(module, "image_id", lambda _: "sha256:" + "1" * 64)

    def fake(args: list[str], env: dict[str, str] | None = None) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(module, "run", fake)
    with pytest.raises(ValueError, match="onaylanmamış"):
        module.compose(config(), "start")
    assert not any("up" in call for call in calls)


def test_source_package_has_no_weights_or_private_payload_and_is_self_contained(
    tmp_path: Path,
) -> None:
    module = helper()
    destination = tmp_path / "source.tar.gz"
    module.package(destination)
    with tarfile.open(destination) as archive:
        names = set(archive.getnames())
        extracted = tmp_path / "extracted"
        archive.extractall(extracted, filter="data")
    assert "scripts/deploy_learned_corners.py" in names
    assert "scripts/prepare_phase2d_context.py" in names
    assert "src/anpr_engine/detection/hyperlpr_corners.py" in names
    assert "configs/public/live-corners-template.json" in names
    assert "configs/public/live-lean-corners-template.json" in names
    assert "deploy/docker/inference.lean-corners.amd64.Dockerfile" in names
    assert all(not name.startswith(("/", "../")) for name in names)
    assert not any(Path(name).suffix in {".onnx", ".pt", ".pdiparams"} for name in names)
    assert not any(
        part in {"datasets", "private-governance", ".git", ".env"}
        for name in names
        for part in Path(name).parts
    )
    # Verify the exported source, not the original checkout, can recreate both
    # Docker contexts. No Docker daemon, dependency install, or image build.
    for service in ("web", "inference-paddleocr-amd64"):
        subprocess.run(
            [
                sys.executable,
                str(extracted / "scripts/prepare_phase2d_context.py"),
                service,
                str(tmp_path / service),
            ],
            check=True,
            capture_output=True,
        )
    with pytest.raises(FileExistsError):
        module.package(destination)


def test_build_config_does_not_require_live_credentials_or_models(tmp_path: Path) -> None:
    module = helper()
    path = tmp_path / "builder.json"
    path.write_text(
        json.dumps({"inference_image": "anpr-inference:test", "web_image": "anpr-web:test"})
    )
    path.chmod(0o600)
    assert "internal_credential" not in module.load_config(path, "build")


@pytest.mark.parametrize(
    "damage", [None, "corner", "detection", "recognition", "paddle", "extra", "manifest", "symlink"]
)
def test_all_models_are_checked_without_loading_a_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str | None
) -> None:
    module = helper()
    bundle, paddle = tmp_path / "bundle", tmp_path / "paddle"
    bundle.mkdir()
    paddle.mkdir()
    corner = tmp_path / "corners.onnx"
    corner.write_bytes(b"test-corners")
    (paddle / "weights").write_bytes(b"test-ocr")
    hashes = {}
    for role in ("detection", "recognition"):
        path = bundle / f"{role}.pt"
        path.write_bytes(role.encode())
        hashes[role] = hashlib.sha256(role.encode()).hexdigest()
    manifest = {
        "schema_version": "anpr-engine-private-model-bundle-v1",
        "classification": "PRIVATE_OWNER_ONLY",
        "models": [{"role": role, "sha256": sha} for role, sha in hashes.items()],
    }
    (bundle / "bundle-manifest.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256()
    name = b"weights"
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update(hashlib.sha256(b"test-ocr").digest())
    monkeypatch.setattr(module, "CORNER_SHA", hashlib.sha256(b"test-corners").hexdigest())
    monkeypatch.setattr(module, "BUNDLE_SHA", hashes)
    monkeypatch.setattr(module, "PADDLE_SHA", digest.hexdigest())
    if damage == "corner":
        corner.write_bytes(b"modified")
    elif damage in ("detection", "recognition"):
        (bundle / f"{damage}.pt").write_bytes(b"modified")
    elif damage == "paddle":
        (paddle / "weights").write_bytes(b"modified")
    elif damage == "extra":
        (paddle / "extra").write_bytes(b"unexpected")
    elif damage == "manifest":
        (bundle / "bundle-manifest.json").write_text("[]")
    elif damage == "symlink":
        (paddle / "alias").symlink_to(corner)
    paths = {"model_bundle": str(bundle), "paddle_model": str(paddle), "corner_model": str(corner)}
    if damage is None:
        module.check_models(paths)
        paths.pop("model_bundle")
        paths["runtime_profile"] = "lean_corners"
        module.check_models(paths)
    else:
        with pytest.raises(ValueError):
            module.check_models(paths)


def test_lean_start_selects_only_lean_compose_and_needs_no_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = helper()
    value = config()
    value.pop("model_bundle")
    value.update(runtime_profile="lean_corners", release_checks_confirmed=True)
    calls = []

    def fake(args: list[str], env: dict[str, str] | None = None) -> str:
        calls.append(args)
        assert env is not None
        assert env["ANPR_PRIVATE_MODEL_BUNDLE_SOURCE"] == ""
        assert env["ANPR_PUBLIC_ORIGIN"] == "https://plates.test"
        return ""

    monkeypatch.setattr(module, "check_models", lambda _: None)
    monkeypatch.setattr(module, "image_id", lambda _: "sha256:" + "1" * 64)
    monkeypatch.setattr(module, "run", fake)
    module.compose(value, "start")
    assert any("compose.lean-corners-candidate.yaml" in arg for arg in calls[0])
    assert not any("compose.public-candidate.yaml" in arg for arg in calls[0])


def test_lean_build_does_not_build_a_legacy_base(monkeypatch: pytest.MonkeyPatch) -> None:
    module = helper()
    value = config()
    value["runtime_profile"] = "lean_corners"
    calls: list[list[str]] = []

    def fake(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 1 if args[:3] == ["docker", "image", "inspect"] else 0
        )

    monkeypatch.setattr(module.subprocess, "run", fake)
    monkeypatch.setattr(module, "run", lambda *args: "")
    module.build(value)
    builds = [args for args in calls if args[:2] == ["docker", "build"]]
    assert len(builds) == 2
    assert any("inference.lean-corners.amd64.Dockerfile" in arg for arg in builds[0])
    assert not any("--build-arg" in args for args in builds)
