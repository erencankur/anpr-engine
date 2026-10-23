from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path


def test_local_launcher_exempts_only_its_loopback_client_ip() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/launch_local_website.py"
    tree = ast.parse(path.read_text())
    values = {
        node.arg: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and isinstance(node.value, ast.Constant)
    }
    assert values["ANPR_RATE_LIMIT_EXEMPT_IPS"] == "127.0.0.1"
    assert values["ANPR_LOCAL_DIRECT_MODE"] == "1"
    assert values["HOSTNAME"] == "127.0.0.1"


def test_local_launcher_preserves_virtualenv_python_symlink(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/launch_local_website.py"
    spec = importlib.util.spec_from_file_location("local_launcher_paths", path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    target = tmp_path / "base-python"
    target.touch()
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(target)
    normalized = launcher.absolute_runtime_path(Path("venv/bin/python"), tmp_path)
    assert normalized == venv_python
    assert normalized.is_symlink()


def test_local_launcher_stops_only_its_owned_process_group() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts/launch_local_website.py"
    spec = importlib.util.spec_from_file_location("local_website_launcher", path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    own = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        launcher.stop(own)
        assert own.poll() is not None
        assert other.poll() is None
    finally:
        for process in (own, other):
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
