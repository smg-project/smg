"""CPU contracts for original packed material identity and same-job CI wiring."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts/ci_sglang_mxfp4_numeric.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def numeric():
    return load("sglang_numeric", SCRIPT)


def test_original_keys_reject_absent_or_ambiguous_identity(numeric):
    good = {"model.layers.0.mlp.experts." + key: "shard.safetensors" for key in numeric.SUFFIXES}
    assert len(numeric.checkpoint_keys(good)) == 6
    with pytest.raises(ValueError):
        numeric.checkpoint_keys({})
    good["other.layers.0.mlp.experts.gate_up_proj_blocks"] = "other.safetensors"
    with pytest.raises(ValueError):
        numeric.checkpoint_keys(good)


def test_actual_cache_symlink_cannot_escape_repository(numeric, tmp_path):
    repo = tmp_path / "models--openai--gpt-oss-20b"
    root = repo / "snapshots" / numeric.REVISION
    root.mkdir(parents=True)
    blob = repo / "blobs" / "actual"
    blob.parent.mkdir()
    blob.write_bytes(b"fixture")
    (root / "a.safetensors").symlink_to(blob)
    assert numeric.checkpoint_file(root, "a.safetensors") == blob
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"fixture")
    (root / "bad.safetensors").symlink_to(foreign)
    for name in ("../foreign.safetensors", "/absolute.safetensors", "bad.safetensors"):
        with pytest.raises(ValueError):
            numeric.checkpoint_file(root, name)


def test_numeric_requires_the_exact_installed_patch_pair():
    installer = load("sglang_installer", ROOT / "scripts/ci_sglang_mxfp4_fallback.py")
    pins = json.loads(SCRIPT.with_name("ci_sglang_mxfp4_numeric_inputs.json").read_text())
    for name, (_, patched) in installer.FILES.items():
        assert pins["sdk_files"][name] == patched


def test_numeric_is_failclosed_in_same_final_environment_job_after_downloads():
    workflow = yaml.safe_load((ROOT / ".github/workflows/e2e-gpu-job.yml").read_text())
    steps = next(iter(workflow["jobs"].values()))["steps"]
    names = [step.get("name") for step in steps]
    index = names.index("Check original GPTOSS packed MXFP4 numeric path")
    step = steps[index]
    assert step["if"] == (
        "inputs.final_environment_check && inputs.engine == 'sglang' && inputs.gpu_vendor == 'amd'"
    )
    assert step["timeout-minutes"] == 3
    assert not step.get("continue-on-error", False)
    assert step["env"] == {
        "SGLANG_USE_AITER": "0",
        "SGLANG_GPTOSS_MXFP4_BF16_FALLBACK": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert names.index("Download models") < index
    assert names.index("Download extra models") < index < names.index("Run E2E tests")
    assert "python3 -I -B scripts/ci_sglang_mxfp4_numeric.py" in step["run"]
    assert "--snapshot" not in step["run"]
    assert "--output e2e-logs/gptoss-mxfp4-numeric" in step["run"]
    assert "||" not in step["run"]


@pytest.fixture
def checkpoint_layout(numeric, tmp_path, monkeypatch):
    import base64
    import hashlib
    import zlib

    fixtures = load(
        "checkpoint_metadata_fixtures", Path(__file__).with_name("test_ci_sglang_mxfp4_fallback.py")
    )
    root = tmp_path / "models"
    snapshot = root / "hub/models--openai--gpt-oss-20b/snapshots" / numeric.REVISION
    snapshot.mkdir(parents=True)
    for name, packed in fixtures._ORIGINAL_METADATA_ZLIB.items():
        (snapshot / name).write_bytes(zlib.decompress(base64.b64decode(packed)))
    index = json.loads((snapshot / "model.safetensors.index.json").read_bytes())
    for name in set(index["weight_map"].values()):
        (snapshot / name).write_bytes(b"canonical-fixture-not-real-weights")
    reference = snapshot.parents[1] / "refs/main"
    reference.parent.mkdir()
    reference.write_text(numeric.REVISION)
    alias = root / numeric.MODEL
    alias.parent.mkdir()
    monkeypatch.setenv("ROUTER_LOCAL_MODEL_PATH", str(root))
    monkeypatch.setenv("HF_HOME", str(root))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    patch = ROOT / "scripts/patches/sglang-0.5.21-gptoss-gfx942.patch"
    added = "\n".join(
        line[1:]
        for line in patch.read_text().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    helpers = added[
        added.index("_GPTOSS_REVISION =") : added.index("def _validate_gfx942_gptoss_bf16_runtime")
    ]
    sdk = tmp_path / "sdk"
    source = sdk / "srt/layers/quantization/mxfp4.py"
    source.parent.mkdir(parents=True)
    source.write_text(helpers)
    pins = {
        "sdk_files": {
            "srt/layers/quantization/mxfp4.py": hashlib.sha256(source.read_bytes()).hexdigest()
        }
    }
    return root, snapshot, alias, sdk, pins


def test_numeric_and_worker_choose_the_same_copy_instead_of_decoy_snapshot(
    numeric, checkpoint_layout
):
    import hashlib
    import shutil

    root, snapshot, alias, sdk, pins = checkpoint_layout
    shutil.copytree(snapshot, alias)
    shard = next(alias.glob("*.safetensors"))
    shard.write_bytes(b"selected-copy-fixture-not-real-weights")
    chosen = numeric.worker_checkpoint()
    assert chosen["worker_model_path"] == str(alias)
    checked = numeric.identity_validator(sdk, pins)(chosen["checkpoint_path"])
    assert checked["resolved_path"] == str(alias)
    actual = numeric.checkpoint_file(
        Path(checked["resolved_path"]), shard.name, Path(checked["repository"])
    )
    assert actual == shard and actual != snapshot / shard.name
    assert (
        hashlib.sha256(actual.read_bytes()).digest()
        != hashlib.sha256((snapshot / shard.name).read_bytes()).digest()
    )
    assert checked["metadata"]["config.json"]["path"] == str(alias / "config.json")
    assert checked["metadata"]["model.safetensors.index.json"]["path"] == str(
        alias / "model.safetensors.index.json"
    )


def test_numeric_accepts_symlinked_model_root_and_original_blob_links(
    numeric, checkpoint_layout, tmp_path, monkeypatch
):
    root, snapshot, alias, sdk, pins = checkpoint_layout
    alias.symlink_to(snapshot, target_is_directory=True)
    lexical_root = tmp_path / "mounted-models"
    lexical_root.symlink_to(root, target_is_directory=True)
    monkeypatch.setenv("ROUTER_LOCAL_MODEL_PATH", str(lexical_root))
    monkeypatch.setenv("HF_HOME", str(lexical_root))
    blobs = snapshot.parents[1] / "blobs"
    blobs.mkdir()
    shard = next(snapshot.glob("*.safetensors"))
    target = blobs / "original-blob-fixture"
    shard.rename(target)
    shard.symlink_to(target)
    chosen = numeric.worker_checkpoint()
    assert chosen["worker_model_path"] == str(lexical_root / numeric.MODEL)
    checked = numeric.identity_validator(sdk, pins)(chosen["checkpoint_path"])
    assert checked["resolved_path"] == str(snapshot)
    assert numeric.checkpoint_file(snapshot, shard.name, Path(checked["repository"])) == target


def test_hub_id_fallback_pins_worker_snapshot_independently_of_main_ref(numeric, checkpoint_layout):
    root, snapshot, alias, sdk, pins = checkpoint_layout
    assert not alias.exists()
    chosen = numeric.worker_checkpoint()
    assert chosen["harness_model_path"] == numeric.MODEL
    assert chosen["worker_model_path"] == str(snapshot)
    assert chosen["hub_main_ref"] == numeric.REVISION
    checked = numeric.identity_validator(sdk, pins)(chosen["checkpoint_path"])
    assert checked["resolved_path"] == str(snapshot)
    (snapshot.parents[1] / "refs/main").write_text("0" * 40)
    assert numeric.worker_checkpoint()["checkpoint_path"] == str(snapshot)
    assert numeric.worker_checkpoint()["hub_main_ref"] == "0" * 40
    (snapshot / "config.json").unlink()
    with pytest.raises(OSError):
        numeric.identity_validator(sdk, pins)(numeric.worker_checkpoint()["checkpoint_path"])


def test_copy_shard_cannot_use_unrelated_model_root_file(numeric, checkpoint_layout):
    import shutil

    root, snapshot, alias, sdk, pins = checkpoint_layout
    shutil.copytree(snapshot, alias)
    shard = next(alias.glob("*.safetensors"))
    shard.unlink()
    foreign = root / "foreign.safetensors"
    foreign.write_bytes(b"foreign-fixture")
    shard.symlink_to(foreign)
    with pytest.raises(ValueError, match="escapes"):
        numeric.identity_validator(sdk, pins)(str(alias))
    with pytest.raises(ValueError, match="escapes"):
        numeric.checkpoint_file(alias, shard.name, snapshot.parents[1])


def test_isolated_worker_resolution_loads_no_engine_or_infra_package(checkpoint_layout):
    import subprocess
    import sys

    root, snapshot, alias, sdk, pins = checkpoint_layout
    alias.symlink_to(snapshot, target_is_directory=True)
    code = (
        "import importlib.util,sys,json; "
        "spec=importlib.util.spec_from_file_location('numeric',sys.argv[1]); "
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); "
        "print(json.dumps(m.worker_checkpoint())); "
        "assert not any(n in sys.modules for n in ('sglang','torch','infra','e2e_test.infra'))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["worker_model_path"] == str(alias)


def test_numeric_and_worker_accept_the_same_anchored_shared_shards(numeric, checkpoint_layout):
    root, snapshot, alias, sdk, pins = checkpoint_layout
    fixtures = load(
        "shared_blob_fixtures", Path(__file__).with_name("test_ci_sglang_mxfp4_fallback.py")
    )
    links = fixtures.shared_blob_layout(snapshot)
    alias.symlink_to(snapshot, target_is_directory=True)
    selection = numeric.worker_checkpoint()
    checked = numeric.identity_validator(sdk, pins)(selection["checkpoint_path"])
    assert checked["resolved_path"] == str(snapshot)
    for name, (_, target) in links.items():
        actual = numeric.checkpoint_file(snapshot, name, Path(checked["repository"]))
        assert actual == target == (snapshot / name).resolve()


@pytest.mark.parametrize(
    "bad", ["direct_shared", "foreign_repository", "third_hop", "dangling", "nonregular"]
)
def test_numeric_shared_shards_preserve_namespace_and_regularfile_checks(
    numeric, checkpoint_layout, tmp_path, bad
):
    root, snapshot, alias, sdk, pins = checkpoint_layout
    fixtures = load(
        "shared_blob_negative_fixtures",
        Path(__file__).with_name("test_ci_sglang_mxfp4_fallback.py"),
    )
    links = fixtures.shared_blob_layout(snapshot)
    name = sorted(links)[0]
    original, target = links[name]
    shard = snapshot / name
    if bad in ("direct_shared", "foreign_repository"):
        shard.unlink()
        if bad == "direct_shared":
            shard.symlink_to(target)
        else:
            foreign = root / "hub/models--foreign--gpt-oss-20b/blobs" / original.name
            foreign.parent.mkdir(parents=True)
            foreign.symlink_to(target)
            shard.symlink_to(foreign)
    else:
        target.unlink()
        if bad == "third_hop":
            outside = tmp_path / "outside"
            outside.write_bytes(b"foreign")
            target.symlink_to(outside)
        elif bad == "nonregular":
            target.mkdir()
    with pytest.raises((ValueError, OSError)):
        numeric.checkpoint_file(snapshot, name, snapshot.parents[1])
