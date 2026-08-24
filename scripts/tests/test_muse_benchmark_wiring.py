from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _matrix_script(workflow: str) -> str:
    text = (ROOT / workflow).read_text(encoding="utf-8")
    marker = "python3 - <<'PY' >> \"$GITHUB_OUTPUT\"\n"
    script = text.split(marker, 1)[1].split("\n          PY", 1)[0]
    return textwrap.dedent(script)


def _run_matrix(workflow: str, monkeypatch, capsys, *, only: str = "") -> dict:
    monkeypatch.setenv("ONLY", only)
    monkeypatch.setenv("MODEL_OVERRIDE", "")
    monkeypatch.setenv("BFCL_OVERRIDE", "")
    exec(_matrix_script(workflow), {})
    line = capsys.readouterr().out.strip()
    assert line.startswith("matrix=")
    return json.loads(line.removeprefix("matrix="))


@pytest.mark.parametrize(
    ("launcher", "prefix"),
    [
        ("scripts/bfcl/launch_arm.sh", "BFCL"),
        ("scripts/tau2/launch_arms.sh", "TAU2"),
    ],
)
def test_launchers_support_the_same_sglang_backed_smg_arm(launcher: str, prefix: str) -> None:
    text = (ROOT / launcher).read_text(encoding="utf-8")

    assert f"{prefix}_ARM_B_WORKER" in text
    assert f"{prefix}_SGLANG_EXTRA" in text
    assert '"$SGLANG_PYTHON" -m sglang.launch_server' in text
    assert "--smg-grpc-mode" in text
    assert "free_low_port" in text
    assert "/readiness" in text


@pytest.mark.parametrize(
    ("workflow", "score_script"),
    [
        (".github/workflows/nightly-bfcl.yml", "scripts/bfcl/run_ab.py"),
        (".github/workflows/nightly-tau2.yml", "scripts/tau2/run_ab.py"),
    ],
)
def test_nightlies_use_the_same_muse_single_arm_contract(workflow: str, score_script: str) -> None:
    text = (ROOT / workflow).read_text(encoding="utf-8")

    assert '"name": "muse-glimmer"' in text
    assert '"model": "meta-models/Muse-Glimmer-30B"' in text
    assert '"arm_mode": "smg_only"' in text
    assert '"arm_b_worker": "sglang"' in text
    assert "uses: ./.github/actions/setup-sglang" in text
    assert f"python {score_script} --report-arm" in text
    assert f"gate python {score_script} --report-arm" not in text
    assert "REPORT_RC=0" in text
    assert "launch_arm" in text
    assert " stop 9>&- || true" in text


@pytest.mark.parametrize(
    ("workflow", "launcher"),
    [
        (".github/workflows/nightly-bfcl.yml", "scripts/bfcl/launch_arm.sh"),
        (".github/workflows/nightly-tau2.yml", "scripts/tau2/launch_arms.sh"),
    ],
)
def test_sequential_arms_keep_a_strict_teardown_boundary(workflow: str, launcher: str) -> None:
    text = (ROOT / workflow).read_text(encoding="utf-8")
    sequential = text.split('elif [ "$ARM_MODE" = sequential ]; then', 1)[1].split(
        "\n          else", 1
    )[0]

    assert f"bash {launcher} stop 9>&- || true" not in sequential
    assert sequential.count(f"bash {launcher} stop 9>&-") == 2


@pytest.mark.parametrize(
    "workflow",
    [
        ".github/workflows/nightly-bfcl.yml",
        ".github/workflows/nightly-tau2.yml",
    ],
)
def test_gpu_nightlies_share_a_host_mounted_lock(workflow: str) -> None:
    text = (ROOT / workflow).read_text(encoding="utf-8")

    assert 'mkdir -p "$ROUTER_LOCAL_MODEL_PATH/.locks"' in text
    assert 'exec 9>"$ROUTER_LOCAL_MODEL_PATH/.locks/smg-benchmark-host.lock"' in text
    assert "exec 9>/tmp/" not in text


def test_bfcl_teardown_backstop_uses_the_launch_run_directory() -> None:
    text = (ROOT / ".github/workflows/nightly-bfcl.yml").read_text(encoding="utf-8")
    teardown = text.split("- name: Teardown arms (leg-scoped backstop)", 1)[1]

    assert "BFCL_RUN_DIR: ${{ runner.temp }}/bfcl_run" in teardown


def test_bfcl_muse_handler_matches_the_served_model() -> None:
    text = (ROOT / ".github/workflows/nightly-bfcl.yml").read_text(encoding="utf-8")

    assert '"model": "meta-models/Muse-Glimmer-30B"' in text
    assert '"bfcl_model": "meta-models/Muse-Glimmer-30B-FC"' in text
    assert 'BFCL_OVERRIDE != f"{MODEL_OVERRIDE}-FC"' in text


@pytest.mark.parametrize(
    "workflow",
    [
        ".github/workflows/nightly-bfcl.yml",
        ".github/workflows/nightly-tau2.yml",
    ],
)
def test_matrix_builder_selects_one_complete_muse_leg(workflow: str, monkeypatch, capsys) -> None:
    matrix = _run_matrix(workflow, monkeypatch, capsys, only="muse-glimmer")

    assert len(matrix["include"]) == 1
    leg = matrix["include"][0]
    assert leg["model"] == "meta-models/Muse-Glimmer-30B"
    assert leg["arm_mode"] == "smg_only"
    assert leg["arm_b_worker"] == "sglang"
    assert leg["sglang_extra"] == ""
    assert leg["gpu_b"] == "0,1"


@pytest.mark.parametrize(
    "workflow",
    [
        ".github/workflows/nightly-bfcl.yml",
        ".github/workflows/nightly-tau2.yml",
    ],
)
def test_matrix_builder_defaults_every_other_leg_to_vllm(
    workflow: str, monkeypatch, capsys
) -> None:
    matrix = _run_matrix(workflow, monkeypatch, capsys)

    assert len(matrix["include"]) == 7
    for leg in matrix["include"]:
        expected = "sglang" if leg["name"] == "muse-glimmer" else "vllm"
        assert leg["arm_b_worker"] == expected
        assert "sglang_extra" in leg


def test_bfcl_matrix_rejects_a_mismatched_override_pair(monkeypatch) -> None:
    monkeypatch.setenv("ONLY", "muse-glimmer")
    monkeypatch.setenv("MODEL_OVERRIDE", "org/model")
    monkeypatch.setenv("BFCL_OVERRIDE", "org/other-FC")

    with pytest.raises(SystemExit, match="must exactly equal"):
        exec(_matrix_script(".github/workflows/nightly-bfcl.yml"), {})


def test_bespoke_eval_workflow_is_removed() -> None:
    assert not (ROOT / ".github/workflows/eval-accuracy.yml").exists()
    assert not (ROOT / "scripts/eval").exists()
