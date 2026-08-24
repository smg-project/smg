from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "bfcl" / "run_ab.py"
SPEC = importlib.util.spec_from_file_location("bfcl_run_ab", SCRIPT)
assert SPEC and SPEC.loader
run_ab = importlib.util.module_from_spec(SPEC)
sys.modules["bfcl_run_ab"] = run_ab
SPEC.loader.exec_module(run_ab)


def _args(tmp_path: Path, *, allow_incomplete: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        out=tmp_path / "report.md",
        json_out=tmp_path / "report.json",
        allow_incomplete=allow_incomplete,
        tolerance=0.02,
    )


def test_nonzero_bfcl_stage_marks_complete_looking_scores_incomplete(
    monkeypatch, tmp_path: Path
) -> None:
    returncodes = iter([7, 0])
    monkeypatch.setattr(run_ab, "_run", lambda *_args, **_kwargs: next(returncodes))
    monkeypatch.setattr(
        run_ab,
        "parse_scores",
        lambda *_args, **_kwargs: ({"simple_python": 1.0}, {"simple_python": 100}),
    )
    arm = run_ab.Arm(
        name="smg",
        base_url="http://localhost:8000",
        host="localhost",
        port="8000",
        project_root=tmp_path / "smg",
    )

    run_ab.run_bfcl(
        arm,
        bfcl="bfcl",
        model="meta-models/Muse-Glimmer-30B-FC",
        categories=["simple_python"],
        num_threads=1,
        temperature=0.001,
        skip_generate=False,
    )

    assert arm.scores == {"simple_python": 1.0}
    assert arm.failed == {"generate": "exited 7"}
    assert "generate (exited 7)" in (run_ab.incompleteness(arm, ["simple_python"]) or "")


def test_bfcl_failure_state_survives_score_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "scores.json"
    run_ab.save_scores(
        run_ab.Arm(
            name="smg",
            base_url="http://localhost:8000",
            scores={"parallel": 0.02},
            counts={"parallel": 100},
            failed={"evaluate": "exited 9"},
        ),
        path,
    )

    assert run_ab.load_scores(path).failed == {"evaluate": "exited 9"}


def test_single_arm_report_contains_absolute_scores_and_counts() -> None:
    arm = run_ab.Arm(
        name="smg",
        base_url="http://localhost:8000",
        scores={"simple_python": 0.875},
        counts={"simple_python": 40},
    )

    markdown, payload = run_ab.build_single_report(arm, ["simple_python"])

    assert "single arm" in markdown.lower()
    assert "| simple_python | 40 | 87.50 |" in markdown
    assert payload["mode"] == "single_arm"
    assert payload["overall_weighted"] == 0.875
    assert payload["incomplete"] is None


def test_single_arm_gate_rejects_a_score_with_zero_cases(tmp_path: Path) -> None:
    arm = run_ab.Arm(
        name="smg",
        base_url="http://localhost:8000",
        scores={"parallel": 0.0},
        counts={"parallel": 0},
    )

    rc = run_ab.write_single_report_and_gate(arm, ["parallel"], _args(tmp_path))

    assert rc == run_ab.EXIT_INCOMPLETE
    assert "zero cases" in (run_ab.incompleteness(arm, ["parallel"]) or "")
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "report.json").is_file()


def test_single_arm_gate_accepts_zero_accuracy_with_real_cases(tmp_path: Path) -> None:
    arm = run_ab.Arm(
        name="smg",
        base_url="http://localhost:8000",
        scores={"parallel": 0.0},
        counts={"parallel": 200},
    )

    rc = run_ab.write_single_report_and_gate(arm, ["parallel"], _args(tmp_path))

    assert rc == run_ab.EXIT_OK


def test_report_arm_cli_loads_saved_scores(monkeypatch, tmp_path: Path) -> None:
    scores = tmp_path / "scores.json"
    run_ab.save_scores(
        run_ab.Arm(
            name="smg",
            base_url="http://localhost:8000",
            scores={"multiple": 0.94},
            counts={"multiple": 100},
        ),
        scores,
    )
    report = tmp_path / "single.md"
    payload = tmp_path / "single.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--report-arm",
            str(scores),
            "--categories",
            "multiple",
            "--out",
            str(report),
            "--json-out",
            str(payload),
        ],
    )

    assert run_ab.main() == run_ab.EXIT_OK
    assert "94.00" in report.read_text(encoding="utf-8")
    assert payload.is_file()
