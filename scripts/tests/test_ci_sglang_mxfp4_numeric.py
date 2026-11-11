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
    assert (
        "--snapshot /models/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
        in step["run"]
    )
    assert "--output e2e-logs/gptoss-mxfp4-numeric" in step["run"]
    assert "||" not in step["run"]
