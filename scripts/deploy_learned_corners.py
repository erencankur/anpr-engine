#!/usr/bin/env python3
"""Explicit Linux/amd64 release helper; stdlib only, no implicit build or push."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CORNER_SHA = "0306de937471b87f56eb3f5620815e7e7058f8ab7428e0734fc64627cc4d716c"
PADDLE_SHA = "5fffa451df2efa707bafd3d1da7e94aaebf83c5d365d2c4b8a6d439b4ac48964"
BUNDLE_SHA = {
    "detection": "0d5e9b52fae5638e0ef1badafdeb53fe0771d056e43c4953d13cceed520e71ea",
    "recognition": "25b6e4af757b87da75162293a28274e3a0ecbb7f29bc55489b7132e353ad5ff7",
}
FILES = [
    "compose.public-candidate.yaml",
    "compose.paddleocr-candidate.yaml",
    "compose.learned-corners-candidate.yaml",
]


def doctor() -> dict[str, Any]:
    """Small, read-only host inventory; no Docker daemon, models or secrets."""
    memory: dict[str, int | None] = {"total_mib": None, "available_mib": None}
    if platform.system() == "Linux":
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                if name in ("MemTotal", "MemAvailable"):
                    values[name] = int(value.split()[0]) // 1024
            memory = {
                "total_mib": values.get("MemTotal"),
                "available_mib": values.get("MemAvailable"),
            }
        except (OSError, ValueError):
            pass
    compose_version = None
    docker_cli = shutil.which("docker") is not None
    if docker_cli:
        try:
            result = subprocess.run(
                ["docker", "compose", "version", "--short"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"v?\d+\.\d+\.\d+[a-zA-Z0-9.+-]*", version):
                compose_version = version
        except (OSError, subprocess.TimeoutExpired):
            pass
    warnings = []
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        warnings.append("Release hedefi Linux amd64; Mac yerel geliştirme için kullanılabilir.")
    if compose_version is None:
        warnings.append("Docker Compose sürümü doğrulanamadı.")
    available = memory["available_mib"]
    if available is None:
        warnings.append("Kullanılabilir RAM bilinmiyor; kaynak uygunluğu TBD.")
    elif available < 2560:
        warnings.append(
            "Boş RAM mevcut 2560 MiB toplam container tavanından az; bütçe incelenmeli."
        )
    return {
        "schema_version": "anpr-host-readiness-v1",
        "os": platform.system(),
        "architecture": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "host_memory": memory,
        "source_disk_free_mib": shutil.disk_usage(ROOT).free // (1024 * 1024),
        "docker_cli_found": docker_cli,
        "compose_version": compose_version,
        "container_limits_mib": {"inference": 2048, "web": 512},
        "production_readiness": "NOT_VERIFIED",
        "warnings": warnings,
        "notes": [
            "Host RAM cgroup/hosting kotası değildir; kotalar ayrıca doğrulanmalı.",
            "Docker daemon, build, model, DNS/TLS ve gerçek upload kontrolü yapılmadı.",
            "Boş disk kaynak dizininin filesystem'idir; Docker storage alanı ayrı olabilir.",
        ],
    }


def run(args: list[str], env: dict[str, str] | None = None) -> str:
    # Capture output: a failed Compose invocation must not echo resolved secrets.
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Komut başarısız: {' '.join(args[:3])}; servisler otomatik silinmedi.")
    return result.stdout.strip()


def load_config(path: Path, profile: str = "runtime") -> dict[str, Any]:
    if path.stat().st_mode & 0o077:
        raise ValueError("Özel config izinlerini chmod 600 ile sınırla.")
    config: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config bir JSON object olmalı.")
    if config.get("runtime_profile", "legacy") not in {"legacy", "lean_corners"}:
        raise ValueError("Geçersiz runtime_profile.")
    if profile == "models":
        return config
    for key in ("inference_image", "web_image"):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/@:-]+", config.get(key, "")):
            raise ValueError(f"Geçersiz {key}.")
    if profile == "build":
        # A separate builder must never need the live credential or model paths.
        if any(
            "@" in config[key] or config[key].startswith("sha256:")
            for key in ("inference_image", "web_image")
        ):
            raise ValueError("Build için digest yerine yeni image etiketi gerekli.")
        return config
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", config.get("project_name", "")):
        raise ValueError("Geçersiz project_name.")
    origin = urlsplit(config.get("public_origin", ""))
    if (
        origin.scheme != "https"
        or not origin.hostname
        or origin.username
        or origin.password
        or origin.path not in ("", "/")
        or origin.query
        or origin.fragment
        or origin.hostname.endswith(".example")
    ):
        raise ValueError("public_origin gerçek HTTPS domain'i olmalı.")
    secret = config.get("internal_credential", "")
    if not isinstance(secret, str) or len(secret) < 32 or "REPLACE" in secret:
        raise ValueError("En az 32 karakterlik yeni rastgele internal_credential gerekli.")
    port = config.get("web_port")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("web_port 1024–65535 olmalı.")
    return config


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_models(config: dict[str, Any]) -> None:
    keys = ["paddle_model", "corner_model"]
    if config.get("runtime_profile", "legacy") == "legacy":
        keys.append("model_bundle")
    for key in keys:
        path = Path(config[key])
        if not path.is_absolute() or not path.exists() or path.is_symlink():
            raise ValueError(f"Mevcut mutlak, symlink olmayan yol gerekli: {key}")
    corner = Path(config["corner_model"])
    if not corner.is_file() or corner.stat().st_size > 8_000_000 or file_sha(corner) != CORNER_SHA:
        raise ValueError("640 ONNX dosyası veya hash'i uygun değil.")
    if config.get("runtime_profile", "legacy") == "legacy":
        check_legacy_bundle(Path(config["model_bundle"]))
    paddle = Path(config["paddle_model"])
    if not paddle.is_dir() or not any(paddle.iterdir()):
        raise ValueError("Paddle model dizini boş/geçersiz.")
    digest = hashlib.sha256()
    for item in sorted(path for path in paddle.rglob("*") if path.is_file() or path.is_symlink()):
        if item.is_symlink():
            raise ValueError("Paddle model symlink içeremez.")
        relative = item.relative_to(paddle).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha(item)))
    if digest.hexdigest() != PADDLE_SHA:
        raise ValueError("Paddle model dizin hash'i uyuşmuyor.")


def check_legacy_bundle(bundle: Path) -> None:
    for name in ("detection.pt", "recognition.pt", "bundle-manifest.json"):
        if not (bundle / name).is_file() or (bundle / name).is_symlink():
            raise ValueError("Private bundle eksik.")
    manifest = json.loads((bundle / "bundle-manifest.json").read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "anpr-engine-private-model-bundle-v1"
        or manifest.get("classification") != "PRIVATE_OWNER_ONLY"
    ):
        raise ValueError("Private bundle manifest uygun değil.")
    records = manifest.get("models")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or any(not isinstance(item, dict) for item in records)
    ):
        raise ValueError("Private bundle model kaydı uygun değil.")
    identities = {item.get("role"): item.get("sha256") for item in records}
    for role, expected in BUNDLE_SHA.items():
        if identities.get(role) != expected or file_sha(bundle / f"{role}.pt") != expected:
            raise ValueError(f"Private bundle hash uyuşmazlığı: {role}")


def image_id(reference: str) -> str:
    metadata = json.loads(run(["docker", "image", "inspect", reference]))[0]
    if metadata["Os"] != "linux" or metadata["Architecture"] != "amd64":
        raise ValueError("Image Linux amd64 olmalı.")
    return str(metadata["Id"])


def package(destination: Path) -> None:
    if destination.resolve().is_relative_to(ROOT):
        raise ValueError("Paketi repo dışında yeni bir dosyaya yaz.")
    with tempfile.TemporaryDirectory(prefix="anpr-source-") as directory:
        contexts = [Path(directory) / "inference", Path(directory) / "web"]
        for service, target in zip(("inference-paddleocr-amd64", "web"), contexts, strict=True):
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_phase2d_context.py"),
                    service,
                    str(target),
                ]
            )
        entries: dict[str, Path] = {}
        for context in contexts:
            for path in context.rglob("*"):
                if path.is_file():
                    entries[path.relative_to(context).as_posix()] = path
        for name in [
            "scripts/deploy_learned_corners.py",
            "scripts/prepare_phase2d_context.py",
            "deploy/inference/runtime-allowlist.txt",
            "configs/public/live-corners-template.json",
            "configs/public/live-lean-corners-template.json",
            "deploy/docker/compose.lean-corners-candidate.yaml",
            "docs/operations/live-corners-quickstart.md",
            "docs/operations/lean-corners-runtime.md",
            "docs/operations/release-v9-controlled-rollout.md",
            "docs/operations/linux-lean-lab.md",
            "docs/operations/coolify-low-footprint.md",
            "docs/operations/learned-corners-release.md",
            "docs/operations/paddleocr-low-resource-release.md",
            "docs/reports/learned-corner-cpu-selection.md",
            "docs/reports/plate-robustness-review-20261023.md",
            "docs/reports/ocr-adaptive-contrast-review-20261023.md",
            *[f"deploy/docker/{name}" for name in FILES],
        ]:
            entries[name] = ROOT / name
        # Exclusive creation: never overwrite a previous release archive.
        with tarfile.open(destination, "x:gz") as archive:
            for name, path in sorted(entries.items()):
                archive.add(path, arcname=name, recursive=False)
    print(f"Kaynak paketi hazır: {destination}; private model/secret içermez.")


def build(config: dict[str, Any]) -> None:
    # Explicit only; no credentials/weights enter these allowlisted contexts.
    lean = config.get("runtime_profile") == "lean_corners"
    tags = [config["inference_image"], config["web_image"]]
    if not lean:
        tags.append(config["inference_image"] + "-base")
    for tag in tags:
        existing = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
        if existing.returncode == 0:
            raise ValueError(
                "Image etiketi mevcut; rollback'i korumak için yeni sürüm etiketi seç."
            )
    with tempfile.TemporaryDirectory(prefix="anpr-release-build-") as directory:
        context = Path(directory)
        for service, target in (("inference-paddleocr-amd64", "inference"), ("web", "web")):
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_phase2d_context.py"),
                    service,
                    str(context / target),
                ]
            )
        base = config["inference_image"] + "-base"
        steps = [
            ("inference", "inference.amd64.Dockerfile", base, []),
            (
                "inference",
                "inference.paddleocr.amd64.Dockerfile",
                config["inference_image"],
                ["--build-arg", f"ANPR_INFERENCE_BASE_IMAGE={base}"],
            ),
            ("web", "web.Dockerfile", config["web_image"], []),
        ]
        if lean:
            steps = [
                (
                    "inference",
                    "inference.lean-corners.amd64.Dockerfile",
                    config["inference_image"],
                    [],
                ),
                ("web", "web.Dockerfile", config["web_image"], []),
            ]
        for target, dockerfile, tag, extra in steps:
            print(f"Sırayla build: {target} / {dockerfile}", flush=True)
            result = subprocess.run(
                [
                    "docker",
                    "build",
                    "--platform",
                    "linux/amd64",
                    "-f",
                    str(context / target / "deploy/docker" / dockerfile),
                    "-t",
                    tag,
                    *extra,
                    str(context / target),
                ],
                cwd=ROOT,
            )
            if result.returncode:
                raise ValueError("Build başarısız; önceki image/container/volume silinmedi.")
    print("Image'lar hazır. Önce check, sonra start çalıştır.")


def compose(config: dict[str, Any], action: str) -> None:
    if action in ("check", "start"):
        check_models(config)
    env = dict(os.environ)
    # Do not inherit old Compose overrides or detector provider settings.
    for name in list(env):
        if name.startswith(("ANPR_", "COMPOSE_")):
            del env[name]
    env.update(
        {
            "ANPR_INTERNAL_SERVICE_CREDENTIAL": config["internal_credential"],
            "ANPR_APPLICATION_VERSION": config["application_version"],
            "ANPR_PRIVATE_MODEL_BUNDLE_SOURCE": config.get("model_bundle", ""),
            "ANPR_PUBLIC_ORIGIN": config["public_origin"].rstrip("/"),
            "ANPR_PADDLE_MODEL_SOURCE": config["paddle_model"],
            "ANPR_CORNER_MODEL_SOURCE": config["corner_model"],
            "ANPR_PADDLE_INFERENCE_IMAGE": (
                image_id(config["inference_image"])
                if action in ("check", "start")
                else config["inference_image"]
            ),
            "ANPR_PADDLE_WEB_IMAGE": (
                image_id(config["web_image"])
                if action in ("check", "start")
                else config["web_image"]
            ),
        }
    )
    override = {
        "services": {
            "web": {
                "environment": {"ANPR_PUBLIC_ORIGIN": config["public_origin"].rstrip("/")},
                "ports": [f"127.0.0.1:{config['web_port']}:3000"],
            }
        }
    }
    with tempfile.TemporaryDirectory(prefix="anpr-release-compose-") as directory:
        overlay = Path(directory) / "origin.json"
        overlay.write_text(json.dumps(override))
        command = ["docker", "compose", "--env-file", "/dev/null", "-p", config["project_name"]]
        files = (
            ["compose.lean-corners-candidate.yaml"]
            if config.get("runtime_profile") == "lean_corners"
            else FILES
        )
        for name in files:
            command += ["-f", str(ROOT / "deploy/docker" / name)]
        command += ["-f", str(overlay)]
        run([*command, "config", "--quiet"], env)
        if action == "start":
            if config.get("release_checks_confirmed") is not True:
                raise ValueError(
                    "Yayın kabul/lisans/RAM kontrolleri onaylanmamış; start yapılmadı."
                )
            run(
                [
                    *command,
                    "up",
                    "--detach",
                    "--no-build",
                    "--pull",
                    "never",
                    "--wait",
                    "--wait-timeout",
                    "240",
                ],
                env,
            )
            print(
                "Container health kontrolü geçti; gerçek upload ve HTTPS kontrolü ayrıca gerekli."
            )
        elif action == "status":
            print(run([*command, "ps"], env))
        elif action == "stop":
            run([*command, "stop", "--timeout", "120"], env)
            print("Yalnızca bu projenin servisleri durduruldu; volume/model silinmedi.")
        else:
            print(
                "Ön kontroller geçti. Servis başlatılmadı; gerçek-model kabul testi yerine geçmez."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("doctor", "package", "verify-models", "check", "build", "start", "status", "stop"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "doctor":
            print(json.dumps(doctor(), ensure_ascii=False, indent=2))
            return 0
        if args.action == "package":
            if args.output is None:
                raise ValueError("package için --output /repo/disinda/yeni-dosya.tar.gz gerekli.")
            package(args.output)
            return 0
        if args.config is None:
            raise ValueError("--config /ozel/live.json gerekli.")
        profile = (
            "models"
            if args.action == "verify-models"
            else ("build" if args.action == "build" else "runtime")
        )
        config = load_config(args.config, profile)
        if args.action == "verify-models":
            check_models(config)
            print("Seçilen runtime için tüm model hash'leri doğrulandı; model yüklenmedi.")
            return 0
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
            raise ValueError(
                "Bu yayın scripti Linux amd64 sunucu içindir; Mac yerel başlatıcısını kullan."
            )
        if not shutil.which("docker"):
            raise ValueError("Docker ve Compose v2 gerekli; otomatik kurulmaz.")
        run(["docker", "compose", "version"])
        run(["docker", "info", "--format", "{{.OSType}}"])
        if args.action == "build":
            build(config)
        else:
            compose(config, args.action)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"DURDU: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
