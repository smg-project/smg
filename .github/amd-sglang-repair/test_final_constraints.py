"""Final-shell environment controls only; no fake engine or pip compatibility proof."""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import environment  # noqa: E402


def entry(path):
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "text": data.decode(),
    }


def prepare(directory, monkeypatch, original):
    out = directory / "diagnostic-results"
    out.mkdir()
    monkeypatch.setattr(environment, "OUT", out)
    if original is None:
        monkeypatch.delenv("PIP_CONSTRAINT", raising=False)
    else:
        if original:
            original = str(directory / "image-constraints.txt")
            Path(original).write_text("torch==2.11.0\n")
        monkeypatch.setenv("PIP_CONSTRAINT", original)
    originals = [entry(Path(p)) for p in os.environ.get("PIP_CONSTRAINT", "").split()]
    (out / "original-constraints.json").write_text(json.dumps(originals))
    pins = out / "runtime-pins.txt"
    pins.write_text("apache-tvm-ffi==0.1.12+amdreadonlydict2\n")
    environment.pin_state("before")
    return out, pins, originals + [entry(pins)]


@pytest.mark.parametrize("original", [None, "", "image"])
@pytest.mark.parametrize("mode", ["valid", "omitted", "missing"])
def test_actual_cleanup_shell_requires_export_and_keeps_checks(
    tmp_path, monkeypatch, original, mode
):
    out, pins, expected = prepare(tmp_path, monkeypatch, original)
    before = json.loads((out / "runtime-pins-state-before.json").read_text())
    assert before["PIP_CONSTRAINT"] == os.environ.get("PIP_CONSTRAINT")
    workflow = (HERE.parent / "workflows/pr-test-rust.yml").read_text()
    block = (
        workflow.split("      - name: Clean owned processes", 1)[1]
        .split("        run: |\n", 1)[1]
        .split("      - name:", 1)[0]
    )
    shell = textwrap.dedent(block)
    if mode == "omitted":
        shell = "\n".join(
            line for line in shell.splitlines() if "export PIP_CONSTRAINT=" not in line
        )
    if mode == "missing":
        pins.unlink()
    (out / "install-started").touch()
    wrapper = tmp_path / "bin"
    wrapper.mkdir()
    # The subprocess records the real shell environment. Only final pin_state is real;
    # reaper/pip/package snapshot stubs do not prove any native or package behavior.
    script = (
        "#!"
        + sys.executable
        + "\n"
        + f"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, {str(HERE.resolve())!r})
import environment
out = Path("diagnostic-results")
with (out / "calls.jsonl").open("a") as log:
    log.write(json.dumps({{"argv": sys.argv[1:], "constraint": os.environ.get("PIP_CONSTRAINT")}})+"\\n")
if sys.argv[1].endswith("closure_audit.py"):
    rows=[]
    for filename in os.environ.get("PIP_CONSTRAINT", "").split():
        p=Path(filename); data=p.read_bytes()
        rows.append({{"path":str(p),"sha256":environment.sha(data),"text":data.decode()}})
    (out/"final.json").write_text(json.dumps({{"constraints":rows}}))
if sys.argv[1].endswith("environment.py"):
    environment.OUT=out
    environment.pin_state("final")
"""
    )
    (wrapper / "python3").write_text(script)
    (wrapper / "python3").chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", shell],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PATH=str(wrapper) + os.pathsep + os.environ["PATH"],
            E2E_JOB_MARKER="owned-control",
        ),
        text=True,
        capture_output=True,
    )
    assert (result.returncode == 0) is (mode == "valid"), result.stderr
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 4 and calls[-1]["argv"][-1] == "final"
    if mode == "valid":
        assert (
            json.loads((out / "runtime-pins-state-final.json").read_text())["constraints"]
            == expected
        )


@pytest.mark.parametrize("mutation", ["order", "hash", "text", "path", "pins"])
def test_actual_snapshot_and_runtime_pin_mismatch_rejected(tmp_path, monkeypatch, mutation):
    out, pins, expected = prepare(tmp_path, monkeypatch, "image")
    monkeypatch.setenv("PIP_CONSTRAINT", " ".join(item["path"] for item in expected))
    records = json.loads(json.dumps(expected))
    if mutation == "order":
        records.reverse()
    elif mutation == "pins":
        pins.write_text("apache-tvm-ffi==0.1.12\n")
    else:
        records[-1][mutation] = "wrong"
    (out / "final.json").write_text(json.dumps({"constraints": records}))
    with pytest.raises(
        ValueError, match="runtime pins changed|constraint path/hash/text/order differs"
    ):
        environment.pin_state("final")


@pytest.mark.parametrize("mutation", ["hash", "path"])
def test_loaded_registry_must_be_the_installed_code_fix(tmp_path, monkeypatch, mutation):
    from runtime_identity import RuntimeIdentity

    hashes = {
        "/venv/tvm_ffi/core.abi3.so": "48dbc71a92a0d134feaeb5f81d7e7760485e856404d1e26e776330fcc6cfea06",
        "/venv/tvm_ffi/lib/libtvm_ffi.so": "4f3e1f951aa56d2047b3353e75d6cea112f3f4ecf29fb4910b7b6ca11ca93c90",
        "/opt/tilelang/libtilelang.so": "observed",
        "/opt/tilelang/libtvm.so": "observed",
    }
    fixed = {
        "path": "/venv/tvm_ffi/registry.py",
        "sha256": "68e9cd4017191fb00bf9193e751b0d79b27047005dea5c01ff2babd7a527f902",
    }
    loaded = dict(fixed)
    loaded["sha256" if mutation == "hash" else "path"] = "wrong"
    value = {
        "loaded_libraries": [
            {"path": path, "exists": True, "sha256": digest} for path, digest in hashes.items()
        ],
        "loaded_modules": {"tvm_ffi.registry": loaded},
        "installed_registry": fixed,
    }
    observer = RuntimeIdentity(tmp_path)
    monkeypatch.setattr(observer, "capture", lambda phase: value)
    with pytest.raises(RuntimeError, match="actual imported registry differs"):
        observer.pytest_sessionfinish(None, 0)
