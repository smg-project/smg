"""The fixed SDK patch rejects unknown inputs and validates both outputs before writes."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "ci_sglang_mxfp4_fallback.py"


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fixed_sglang_patch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "sdk"
    root.mkdir()

    def digest(body):
        return hashlib.sha256(body).hexdigest()

    module.FILES = {name: (digest(b"before\n"), digest(b"after\n")) for name in ("a.py", "b.py")}
    for name in module.FILES:
        (root / name).write_bytes(b"before\n")
    patch = tmp_path / "fixed.patch"
    patch.write_text(
        "".join(
            f"--- a/python/sglang/{name}\n+++ b/python/sglang/{name}\n@@ -1 +1 @@\n-before\n+after\n"
            for name in module.FILES
        )
    )
    module.PATCH = patch
    module.PATCH_SHA = digest(patch.read_bytes())
    monkeypatch.setenv(module.ROW_ENV, "1")
    return module, root


def test_original_pair_installs_once_and_verifies_final_pair(sdk):
    module, root = sdk
    assert module.install(root)["state"] == "patched"
    assert module.install(root) == module.verify_installed(root)
    assert all((root / name).read_bytes() == b"after\n" for name in module.FILES)


@pytest.mark.parametrize("body", [b"unknown\n", b"after\n"])
def test_unknown_or_mixed_pair_never_changes_other_source(sdk, body):
    module, root = sdk
    (root / "a.py").write_bytes(body)
    with pytest.raises(RuntimeError, match="unknown or mixed"):
        module.install(root)
    assert (root / "a.py").read_bytes() == body and (root / "b.py").read_bytes() == b"before\n"


def test_second_output_mismatch_leaves_both_installed_sources_original(sdk):
    module, root = sdk
    module.FILES["b.py"] = (module.FILES["b.py"][0], "0" * 64)
    with pytest.raises(RuntimeError, match="output differs"):
        module.install(root)
    assert all((root / name).read_bytes() == b"before\n" for name in module.FILES)


def test_changed_patch_rejected_before_source_writes(sdk):
    module, root = sdk
    module.PATCH.write_bytes(b"changed patch")
    with pytest.raises(RuntimeError, match="patch bytes changed"):
        module.install(root)
    assert all((root / name).read_bytes() == b"before\n" for name in module.FILES)


def test_disabled_mode_requires_original_sources_and_invalid_mode_fails(sdk, monkeypatch):
    module, root = sdk
    monkeypatch.setenv(module.ROW_ENV, "0")
    assert module.install(root)["state"] == "original"
    (root / "a.py").write_bytes(b"after\n")
    with pytest.raises(RuntimeError, match="mixed"):
        module.verify_installed(root)
    monkeypatch.setenv(module.ROW_ENV, "yes")
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        module.install(root)
