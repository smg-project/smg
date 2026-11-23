"""Official irrelevance credit must not disguise an inference failure.

The tiny fixtures exercise the real result/score JSONL boundary and, for the
regression, the real CLI subprocess path. They contain no model or credentials.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(
    os.environ.get(
        "BFCL_SCORING_TEST_SOURCE", Path(__file__).resolve().parents[1] / "bfcl/run_ab.py"
    )
)
SPEC = importlib.util.spec_from_file_location("bfcl_scoring_validity", SOURCE)
run_ab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_ab
SPEC.loader.exec_module(run_ab)
MODEL = "fixture/model-FC"
CATEGORY = "live_irrelevance"
TIMEOUT = "Error during inference: Request timed out."


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def scored_arm(tmp_path, *, result=TIMEOUT, credit_error=False):
    arm = run_ab.parse_arm("fixture=http://127.0.0.1:1", tmp_path)
    before, error = run_ab.result_paths(arm.project_root, MODEL, [CATEGORY])
    write_jsonl(
        arm.project_root / "result/fixture_model-FC/live/BFCL_v4_live_irrelevance_result.json",
        [{"id": CATEGORY + "_0", "result": result}, {"id": CATEGORY + "_1", "result": []}],
    )
    arm.score_directory = "score/run-fixture"
    path = (
        arm.project_root
        / arm.score_directory
        / "fixture_model-FC/live/BFCL_v4_live_irrelevance_score.json"
    )
    summary = {
        "accuracy": 1.0 if credit_error else 0.5,
        "correct_count": 2 if credit_error else 1,
        "total_count": 2,
    }
    write_jsonl(
        path, [summary] + ([] if credit_error else [{"id": CATEGORY + "_0", "valid": False}])
    )
    arm.scores, arm.counts = run_ab.parse_scores(
        arm.project_root / arm.score_directory, MODEL, [CATEGORY]
    )
    arm.result_diagnostics = run_ab.collect_result_diagnostics(
        arm.project_root, MODEL, [CATEGORY], before, error, arm.counts
    )
    arm.scoring_validity = run_ab.collect_scoring_validity(arm, [CATEGORY])
    return arm, path


@pytest.mark.parametrize("allow_incomplete", [False, True])
def test_actual_cli_rejects_timeout_receiving_irrelevance_credit(
    tmp_path, monkeypatch, allow_incomplete
):
    # The same shape as the two original live_irrelevance cases: a timeout
    # result is omitted from the official failure table, yielding accuracy 1.
    executable = tmp_path / "fixture-bfcl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "root = Path(os.environ['BFCL_PROJECT_ROOT'])\n"
        "if sys.argv[1] == 'generate':\n"
        " p = root / 'result/fixture_model-FC/live/BFCL_v4_live_irrelevance_result.json'\n"
        " rows = [{'id':'live_irrelevance_0','result':'Error during inference: Request timed out.'},"
        " {'id':'live_irrelevance_1','result':[]}]\n"
        "else:\n"
        " p = root / sys.argv[sys.argv.index('--score-dir')+1]"
        " / 'fixture_model-FC/live/BFCL_v4_live_irrelevance_score.json'\n"
        " rows = [{'accuracy':1.0,'correct_count':2,'total_count':2}]\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        "p.write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n"
    )
    executable.chmod(0o700)
    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    argv = [
        "run_ab.py",
        "--baseline",
        "vllm=http://127.0.0.1:1",
        "--candidate",
        "smg=http://127.0.0.1:2",
        "--bfcl-model",
        MODEL,
        "--categories",
        CATEGORY,
        "--bfcl",
        str(executable),
        "--project-root",
        str(tmp_path / "roots"),
        "--json-out",
        str(output),
        "--out",
        str(markdown),
    ]
    if allow_incomplete:
        argv.append("--allow-incomplete")
    monkeypatch.setattr(sys, "argv", argv)
    assert run_ab.main() == run_ab.EXIT_INCOMPLETE
    report = json.loads(output.read_text())
    assert report["overall"] == {"baseline": 1.0, "candidate": 1.0, "delta": 0.0}
    assert report["incomplete"] == {}
    assert report["comparison_validity"]["status"] == "invalid"
    assert report["comparison_validity"]["eligible"] is False
    for arm in ("baseline", "candidate"):
        row = report["comparison_validity"]["arms"][arm]["categories"][CATEGORY]
        assert row["inference_errors_receiving_credit"] == [CATEGORY + "_0"]
    assert "INVALID" in markdown.read_text()
    assert CATEGORY + "_0" in markdown.read_text()


@pytest.mark.parametrize("result", [[], None, {}, {"tool": {"message": TIMEOUT}}])
def test_legal_empty_or_opaque_arguments_and_normal_wrong_answer_are_not_transport_errors(
    tmp_path, result
):
    # A normal evaluated answer may be wrong. That is not invalid scoring.
    arm, _ = scored_arm(tmp_path, result=result)
    assert arm.scoring_validity["status"] == "valid"
    _, report = run_ab.build_report(arm, arm, [CATEGORY])
    assert report["comparison_validity"]["eligible"] is True
    assert arm.scores[CATEGORY] == 0.5


def test_inference_timeout_correctly_counted_as_official_failure_keeps_comparison_valid(tmp_path):
    arm, _ = scored_arm(tmp_path)
    assert arm.result_diagnostics["counts"]["other_inference_error"] == 1
    row = arm.scoring_validity["categories"][CATEGORY]
    assert row["inference_error_ids"] == [CATEGORY + "_0"]
    assert row["inference_errors_receiving_credit"] == []
    args = SimpleNamespace(out=None, json_out=None, tolerance=0.02, allow_incomplete=False)
    assert run_ab.write_report_and_gate(arm, arm, [CATEGORY], args) == run_ab.EXIT_OK


def test_duplicate_display_names_cannot_hide_one_invalid_arm(tmp_path):
    invalid, _ = scored_arm(tmp_path / "a", credit_error=True)
    valid, _ = scored_arm(tmp_path / "b")
    assert invalid.name == valid.name
    _, report = run_ab.build_report(invalid, valid, [CATEGORY])
    assert report["comparison_validity"]["status"] == "invalid"
    assert report["comparison_validity"]["eligible"] is False


@pytest.mark.parametrize("allow_incomplete", [False, True])
@pytest.mark.parametrize("remove_other_result_after_save", [False, True])
@pytest.mark.parametrize("inference_error", [False, True], ids=["pure-unknown", "known-invalid"])
def test_mixed_unknown_keeps_known_invalid_through_actual_sequential_cli(
    tmp_path, monkeypatch, allow_incomplete, remove_other_result_after_save, inference_error
):
    executable = tmp_path / "fixture-bfcl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "root = Path(os.environ['BFCL_PROJECT_ROOT'])\n"
        "if sys.argv[1] == 'generate':\n"
        " rows = [{'id':'live_irrelevance_0','result':"
        " 'Error during inference: Request timed out.' if os.environ['FIXTURE_ERROR']=='1' else []},"
        " {'id':'live_irrelevance_1','result':[]}]\n"
        " p = root / 'result/fixture_model-FC/live/BFCL_v4_live_irrelevance_result.json'\n"
        " p.parent.mkdir(parents=True, exist_ok=True)\n"
        " p.write_text(''.join(json.dumps(row)+'\\n' for row in rows))\n"
        " if os.environ['FIXTURE_OTHER_RESULT']=='1':\n"
        "  p = root / 'result/fixture_model-FC/non_live/BFCL_v4_simple_python_result.json'\n"
        "  p.parent.mkdir(parents=True, exist_ok=True)\n"
        "  p.write_text(json.dumps({'id':'simple_python_0','result':[]})+'\\n')\n"
        "else:\n"
        " for category, section, total in [('live_irrelevance','live',2),('simple_python','non_live',1)]:\n"
        "  p = root / sys.argv[sys.argv.index('--score-dir')+1]"
        " / 'fixture_model-FC' / section / ('BFCL_v4_'+category+'_score.json')\n"
        "  p.parent.mkdir(parents=True, exist_ok=True)\n"
        "  p.write_text(json.dumps({'accuracy':1.0,'correct_count':total,'total_count':total})+'\\n')\n"
    )
    executable.chmod(0o700)
    monkeypatch.setenv("FIXTURE_ERROR", str(int(inference_error)))
    monkeypatch.setenv("FIXTURE_OTHER_RESULT", str(int(remove_other_result_after_save)))
    saved = []
    for name, port in [("vllm", 1), ("smg", 2)]:
        output = tmp_path / f"scores_{name}.json"
        saved.append(output)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_ab.py",
                "--score-arm",
                f"{name}=http://127.0.0.1:{port}",
                "--bfcl-model",
                MODEL,
                "--categories",
                CATEGORY + ",simple_python",
                "--bfcl",
                str(executable),
                "--project-root",
                str(tmp_path / "roots"),
                "--scores-out",
                str(output),
            ],
        )
        assert run_ab.main() == run_ab.EXIT_OK  # Per-arm scoring persists; final diff qualifies.
        if remove_other_result_after_save:
            (
                tmp_path
                / "roots"
                / name
                / "result/fixture_model-FC/non_live/BFCL_v4_simple_python_result.json"
            ).unlink()
    output = tmp_path / "report.json"
    argv = [
        "run_ab.py",
        "--diff-baseline",
        str(saved[0]),
        "--diff-candidate",
        str(saved[1]),
        "--categories",
        CATEGORY + ",simple_python",
        "--json-out",
        str(output),
    ]
    if allow_incomplete:
        argv.append("--allow-incomplete")
    monkeypatch.setattr(sys, "argv", argv)
    expected = run_ab.EXIT_INCOMPLETE if inference_error else run_ab.EXIT_OK
    assert run_ab.main() == expected
    report = json.loads(output.read_text())
    assert report["overall"] == {"baseline": 1.0, "candidate": 1.0, "delta": 0.0}
    assert report["comparison_validity"]["status"] == ("invalid" if inference_error else "unknown")
    assert report["comparison_validity"]["eligible"] is False
    assert report["incomplete"] == {}
    if inference_error:
        for arm in ("baseline", "candidate"):
            assert report["comparison_validity"]["arms"][arm]["categories"][CATEGORY][
                "inference_errors_receiving_credit"
            ] == [CATEGORY + "_0"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_score",
        "bad_failure_id",
        "duplicate_failure_id",
        "partial_failures",
        "nonfinite",
        "count_mismatch",
    ],
)
def test_missing_or_unreconciled_official_failure_mapping_is_unknown_not_qualified(tmp_path, fault):
    arm, path = scored_arm(tmp_path)
    if fault == "missing_score":
        path.unlink()
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "bad_failure_id":
            rows[1]["id"] = CATEGORY + "_absent"
        elif fault == "duplicate_failure_id":
            rows.append(rows[1])
        elif fault == "partial_failures":
            rows = rows[:1]
        elif fault == "nonfinite":
            rows[0]["accuracy"] = float("nan")
        elif fault == "count_mismatch":
            rows[0]["total_count"] = 3
        write_jsonl(path, rows)
    arm.scoring_validity = run_ab.collect_scoring_validity(arm, [CATEGORY])
    assert arm.scoring_validity["status"] == "unknown"
    _, report = run_ab.build_report(arm, arm, [CATEGORY])
    assert report["comparison_validity"]["eligible"] is False


def test_saved_validity_is_recomputed_from_actual_score_and_result_sources(tmp_path):
    arm, path = scored_arm(tmp_path, credit_error=True)
    saved = tmp_path / "saved.json"
    run_ab.save_scores(arm, saved)
    data = json.loads(saved.read_text())
    data["scoring_validity"] = {"status": "valid"}  # Cannot whitewash the original files.
    saved.write_text(json.dumps(data))
    restored = run_ab.load_scores(saved)
    assert restored.scoring_validity["status"] == "invalid"
    assert restored.scores == arm.scores
    path.unlink()
    assert run_ab.load_scores(saved).scoring_validity["status"] == "unknown"


@pytest.mark.parametrize(
    "directory", [None, False, "../score/run-fixture", "/tmp/score/run-fixture"]
)
def test_legacy_missing_results_or_untrusted_score_directory_never_qualifies(tmp_path, directory):
    arm, _ = scored_arm(tmp_path)
    saved = tmp_path / "saved.json"
    run_ab.save_scores(arm, saved)
    data = json.loads(saved.read_text())
    data["score_directory"] = directory
    saved.write_text(json.dumps(data))
    loaded = run_ab.load_scores(saved)
    _, report = run_ab.build_report(loaded, loaded, [CATEGORY])
    assert report["comparison_validity"]["status"] == "unknown"
    assert report["comparison_validity"]["eligible"] is False


def test_macro_regression_exit_policy_is_unchanged_when_validity_is_unknown(tmp_path):
    baseline, candidate = run_ab.Arm("vllm", ""), run_ab.Arm("smg", "")
    baseline.scores, candidate.scores = {CATEGORY: 0.9}, {CATEGORY: 0.8}
    args = SimpleNamespace(
        out=None, json_out=tmp_path / "report.json", tolerance=0.02, allow_incomplete=False
    )
    assert (
        run_ab.write_report_and_gate(baseline, candidate, [CATEGORY], args)
        == run_ab.EXIT_REGRESSION
    )
    report = json.loads(args.json_out.read_text())
    assert report["overall"]["delta"] == pytest.approx(-0.1)
    assert report["comparison_validity"]["status"] == "unknown"
