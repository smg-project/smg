import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[1] / "tau2" / "costmap_shim"


def startup(tmp_path, mode, failure="ImportError", hook_failure="missing"):
    shim = tmp_path / "tau2" / "costmap_shim"
    shim.mkdir(parents=True)
    (shim / "sitecustomize.py").write_bytes((SHIM / "sitecustomize.py").read_bytes())
    (tmp_path / "litellm.py").write_text(
        f"""
from pathlib import Path
attempts = Path("attempts")
attempts.write_text(attempts.read_text() + "1" if attempts.exists() else "1")
if attempts.read_text() == "1" and {failure is not None!r}:
    raise {failure or "ImportError"}("recoverable SDK import failure")
def completion(**payload):
    Path("sent").touch()
def register_model(value):
    pass
"""
    )
    hook = 'raise ImportError("unavailable hook")\n'
    if hook_failure == "installer":
        hook = """
from pathlib import Path
def install(litellm, *, original):
    assert litellm.completion is not original
    Path("refusal_installed").touch()
    raise RuntimeError("installer failure")
"""
    elif hook_failure is None:
        hook = """
from pathlib import Path
def install(litellm, *, original):
    Path("hook_origin").write_text(__file__)
    def guarded(**payload):
        raise RuntimeError("sibling budget guard refused the request")
    litellm.completion = guarded
"""
    (shim.parent / "budget_hook.py").write_text(hook)
    # A wrong parents[] index must not pass by finding a hook on PYTHONPATH.
    (tmp_path / "budget_hook.py").write_text('raise AssertionError("decoy hook")\n')
    env = os.environ.copy()
    env.pop("SMG_TAU2_BUDGET_GUARD", None)
    if mode is not None:
        env["SMG_TAU2_BUDGET_GUARD"] = mode
    env.update(PYTHONPATH=os.pathsep.join((str(shim), str(tmp_path))), PYTHONNOUSERSITE="1")
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import json, sys; from pathlib import Path; import litellm; "
            'Path("startup_sys_path").write_text(json.dumps(sys.path)); '
            'litellm.completion(); print("main ran")',
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


@pytest.mark.parametrize("mode", [None, "", "0"])
def test_disabled_guard_preserves_recoverable_import(tmp_path, mode):
    result = startup(tmp_path, mode)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "attempts").read_text() == "11"
    assert (tmp_path / "sent").exists()


@pytest.mark.parametrize("failure", ["ImportError", "RuntimeError"])
def test_opt_in_cannot_recover_to_unbudgeted_completion(tmp_path, failure):
    result = startup(tmp_path, "1", failure)
    assert result.returncode != 0 and "main ran" not in result.stdout
    assert (tmp_path / "attempts").read_text() == "1"
    assert not (tmp_path / "sent").exists()


@pytest.mark.parametrize("hook_failure", ["missing", "installer"])
def test_opt_in_loader_failure_stops_real_startup(tmp_path, hook_failure):
    result = startup(tmp_path, "1", None, hook_failure)
    assert result.returncode != 0 and "main ran" not in result.stdout
    assert not (tmp_path / "sent").exists()
    if hook_failure == "installer":
        assert (tmp_path / "refusal_installed").exists()


def test_invalid_opt_in_stops_before_dependency_import(tmp_path):
    result = startup(tmp_path, "invalid")
    assert result.returncode != 0 and "main ran" not in result.stdout
    assert not (tmp_path / "attempts").exists() and not (tmp_path / "sent").exists()


def test_startup_loads_the_sibling_hook_instead_of_pythonpath_decoy(tmp_path):
    result = startup(tmp_path, "1", None, None)
    assert result.returncode != 0 and "sibling budget guard refused" in result.stderr
    assert Path((tmp_path / "hook_origin").read_text()) == tmp_path / "tau2/budget_hook.py"
    assert str(tmp_path / "tau2") not in json.loads((tmp_path / "startup_sys_path").read_text())
    assert not (tmp_path / "sent").exists()
