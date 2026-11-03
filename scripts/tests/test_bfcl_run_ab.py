import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bfcl_run_ab", Path(__file__).resolve().parents[1] / "bfcl" / "run_ab.py"
)
run_ab = importlib.util.module_from_spec(SPEC)
sys.modules["bfcl_run_ab"] = run_ab
SPEC.loader.exec_module(run_ab)


def write_score(score_root, section, category, accuracy, total, prefix="BFCL_v4_"):
    path = score_root / "Qwen_Qwen3-4B-FC" / section / f"{prefix}{category}_score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"accuracy": accuracy, "correct_count": round(accuracy * total), "total_count": total}
    path.write_text(json.dumps(summary) + "\n" + json.dumps({"id": f"{category}_0"}) + "\n")


def test_a_category_is_not_read_from_a_longer_category_with_the_same_suffix(tmp_path):
    write_score(tmp_path, "non_live", "parallel_multiple", 0.72, 200)
    write_score(tmp_path, "live", "live_multiple", 0.60, 1053)
    write_score(tmp_path, "live", "live_parallel_multiple", 0.50, 24)
    write_score(tmp_path, "live", "live_parallel", 0.55, 16)
    write_score(tmp_path, "live", "live_irrelevance", 0.80, 884)

    assert run_ab._find_category_summary(tmp_path, "multiple") is None
    assert run_ab._find_category_summary(tmp_path, "parallel") is None
    assert run_ab._find_category_summary(tmp_path, "irrelevance") is None


def test_each_category_reads_its_own_score_file(tmp_path):
    scores = {
        "multiple": ("non_live", 0.865, 200),
        "parallel_multiple": ("non_live", 0.72, 200),
        "parallel": ("non_live", 0.71, 200),
        "irrelevance": ("non_live", 0.8333, 240),
        "live_multiple": ("live", 0.60, 1053),
        "live_parallel_multiple": ("live", 0.50, 24),
        "live_parallel": ("live", 0.55, 16),
        "live_irrelevance": ("live", 0.80, 884),
    }
    for category, (section, accuracy, total) in scores.items():
        write_score(tmp_path, section, category, accuracy, total)

    for category, (_, accuracy, total) in scores.items():
        assert run_ab._find_category_summary(tmp_path, category) == (accuracy, total)


def test_score_files_without_the_version_prefix_still_match(tmp_path):
    write_score(tmp_path, "non_live", "multiple", 0.865, 200, prefix="")
    write_score(tmp_path, "non_live", "parallel_multiple", 0.72, 200, prefix="")

    assert run_ab._find_category_summary(tmp_path, "multiple") == (0.865, 200)


def fake_bfcl(evaluations):
    """Stand-in for ``run_ab._run``. Each ``evaluate`` writes the next entry of
    ``evaluations`` (a list of ``(category, accuracy, total)``) where bfcl would:
    under ``--score-dir`` resolved against ``BFCL_PROJECT_ROOT``, else ``score/``."""
    score_dirs = []

    def run(cmd, env, label, timeout=0):
        if cmd[1] == "evaluate":
            rel = cmd[cmd.index("--score-dir") + 1] if "--score-dir" in cmd else "score"
            score_dirs.append(rel)
            for category, accuracy, total in evaluations.pop(0):
                write_score(
                    Path(env["BFCL_PROJECT_ROOT"]) / rel, "non_live", category, accuracy, total
                )
        return True

    return run, score_dirs


def score_arm(arm, monkeypatch, run):
    monkeypatch.setattr(run_ab, "_run", run)
    run_ab.run_bfcl(
        arm,
        bfcl="bfcl",
        model="Qwen/Qwen3-4B-FC",
        categories=["simple_python"],
        num_threads=1,
        temperature=0.0,
        skip_generate=True,
    )


def test_scores_come_only_from_this_evaluation(tmp_path, monkeypatch):
    arm = run_ab.parse_arm("smg=http://127.0.0.1:30000", tmp_path)
    # A score file left in the reused root by an earlier, older-BFCL run.
    write_score(
        arm.project_root / "score", "non_live", "simple_python", 0.10, 400, prefix="BFCL_v3_"
    )
    run, score_dirs = fake_bfcl([[("simple_python", 0.90, 400)]])

    score_arm(arm, monkeypatch, run)

    assert arm.scores == {"simple_python": 0.90}
    assert Path(score_dirs[0]).parts[0] == "score"


def test_a_category_this_evaluation_did_not_score_is_missing_not_stale(tmp_path, monkeypatch):
    arm = run_ab.parse_arm("smg=http://127.0.0.1:30000", tmp_path)
    run, score_dirs = fake_bfcl([[("simple_python", 0.90, 400)], []])

    score_arm(arm, monkeypatch, run)
    score_arm(arm, monkeypatch, run)

    assert arm.scores == {}
    assert run_ab.incompleteness(arm, ["simple_python"]) == (
        "no score for 1/1 categories; unscored: simple_python"
    )
    assert len(set(score_dirs)) == 2


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/stat").is_file(),
    reason="requires Linux /proc process identities",
)
@pytest.mark.parametrize("hang_after_summary", [False, True], ids=["complete", "timed-out"])
def test_actual_evaluate_timeout_is_incomplete_even_with_scores(
    tmp_path, monkeypatch, hang_after_summary
):
    # Exercise the real Popen/wait/kill/reap path without BFCL, a server or a
    # model. Only this private executable and its own temporary files are used.
    bfcl = tmp_path / "bounded-bfcl"
    bfcl.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import os
            import sys
            import time
            from pathlib import Path

            assert sys.argv[1] == "evaluate"
            root = Path(os.environ["BFCL_PROJECT_ROOT"])
            score_dir = sys.argv[sys.argv.index("--score-dir") + 1]
            target = root / score_dir / "Qwen_Qwen3-4B-FC" / "non_live"
            target.mkdir(parents=True, exist_ok=True)
            (target / "BFCL_v4_simple_python_score.json").write_text(
                json.dumps({"accuracy": 0.9, "total_count": 400}) + "\\n"
            )
            fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
            (root / "fixture-process.json").write_text(json.dumps({
                "pid": os.getpid(), "parent": os.getppid(),
                "pgid": os.getpgrp(), "start_ticks": int(fields[19]),
            }))
            if Path(__file__).with_suffix(".hang").exists():
                time.sleep(10)  # Self-bounded fallback; actual _run kills it at 3 s.
            """
        )
    )
    bfcl.chmod(0o755)
    if hang_after_summary:
        bfcl.with_suffix(".hang").touch()
    project = tmp_path / "project"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ab.py",
            "--baseline",
            "vllm=http://127.0.0.1:30000",
            "--candidate",
            "smg=http://127.0.0.1:30001",
            "--bfcl-model",
            "Qwen/Qwen3-4B-FC",
            "--categories",
            "simple_python",
            "--bfcl",
            str(bfcl),
            "--project-root",
            str(project),
            "--json-out",
            str(report),
            "--skip-generate",
            "--run-timeout",
            "3",  # Fixture-only; production defaults and original limits are unchanged.
        ],
    )

    exit_code = run_ab.main()
    payload = json.loads(report.read_text())
    for role in ("baseline", "candidate"):
        arm = payload[role]
        assert arm["scores"] == {"simple_python": 0.9}
        assert arm["counts"] == {"simple_python": 400}
        identity = json.loads((project / arm["name"] / "fixture-process.json").read_text())
        assert identity["parent"] == os.getpid()
        assert identity["pgid"] == identity["pid"]
        # Check disappearance by birth identity; never signal a pid from a file.
        stat = Path(f"/proc/{identity['pid']}/stat")
        if stat.exists():
            fields = stat.read_text().rsplit(")", 1)[1].split()
            assert int(fields[19]) != identity["start_ticks"]

    assert exit_code == (run_ab.EXIT_INCOMPLETE if hang_after_summary else run_ab.EXIT_OK)
    for role in ("baseline", "candidate"):
        assert payload[role]["timed_out"] == (["evaluate"] if hang_after_summary else [])
    if hang_after_summary:
        assert set(payload["incomplete"]) == {"vllm", "smg"}
        assert all(v == "timed out during: evaluate" for v in payload["incomplete"].values())
    else:
        assert payload["incomplete"] == {}


# Infrastructure diagnostics are separate from the official scorer and gates.
MODEL = "Qwen/Qwen3-4B-FC"
UNAVAILABLE = "Error during inference: Error code: 503 - no_available_workers"


def write_results(root, category, records, *, model=MODEL, section="non_live"):
    path = root / "result" / model.replace("/", "_") / section / f"BFCL_v4_{category}_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def diagnostics_before(root, categories=("simple_python",)):
    return run_ab.result_paths(root, MODEL, list(categories))


@pytest.mark.parametrize(
    "value,kind",
    [
        ("", "non_error_result"),
        ([], "non_error_result"),
        ("A legitimate answer mentioning503/no_available_workers", "non_error_result"),
        (UNAVAILABLE, "http_503_no_available_workers"),
        (["ok", UNAVAILABLE], "http_503_no_available_workers"),
        ("Error during inference: Request timed out.", "other_inference_error"),
    ],
)
def test_result_diagnostics_classification_and_source_bound_ids(tmp_path, value, kind):
    import hashlib

    before, error = diagnostics_before(tmp_path)
    path = write_results(
        tmp_path,
        "simple_python",
        [
            {"id": "simple_python_0", "result": []},
            {"id": "simple_python_1", "result": value},
        ],
    )
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error, {"simple_python": 2}
    )
    assert diagnostic["status"] == "observed"
    row = diagnostic["categories"]["simple_python"]
    assert row["source"] == {
        "path": str(path.relative_to(tmp_path)),
        "bytes": len(path.read_bytes()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert row["id_inventory"] == [
        {"id": "simple_python_0", "line": 1, "classification": "non_error_result"},
        {"id": "simple_python_1", "line": 2, "classification": kind},
    ]
    assert row["unique_ids"] == 2 and row["duplicate_ids"] == []
    assert sum(diagnostic["counts"].values()) == 2
    assert diagnostic["counts"][kind] == (2 if kind == "non_error_result" else 1)


def test_result_diagnostics_preexisting_rewrite_is_not_fresh(tmp_path):
    path = write_results(tmp_path, "simple_python", [{"id": "simple_python_0", "result": []}])
    before, error = diagnostics_before(tmp_path)
    path.write_text(json.dumps({"id": "simple_python_0", "result": UNAVAILABLE}) + "\n")
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error
    )
    assert diagnostic["status"] == "unknown" and diagnostic["counts"] is None
    assert diagnostic["categories"]["simple_python"]["reason"] == "preexisting_result_files"
    assert "source" not in diagnostic["categories"]["simple_python"]


@pytest.mark.parametrize(
    "state", ["missing", "malformed", "unsupported", "duplicate", "ambiguous", "linked"]
)
def test_result_diagnostics_unknown_evidence_never_becomes_zero_errors(tmp_path, state):
    before, error = diagnostics_before(tmp_path)
    if state != "missing":
        path = write_results(tmp_path, "simple_python", [{"id": "simple_python_0", "result": []}])
        if state == "malformed":
            path.write_text('{"id": "simple_python_0", "result":')
        elif state == "unsupported":
            path.write_text(json.dumps({"id": "simple_python_0", "result": 42}) + "\n")
        elif state == "duplicate":
            path.write_text(path.read_text() * 2)
        elif state == "ambiguous":
            write_results(
                tmp_path,
                "simple_python",
                [{"id": "simple_python_1", "result": []}],
                section="other",
            )
        elif state == "linked":
            outside = tmp_path / "outside-private.txt"
            outside.write_text("PRIVATE must not be read")
            path.unlink()
            path.symlink_to(outside)
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error
    )
    assert diagnostic["status"] == "unknown" and diagnostic["counts"] is None
    assert all(row["counts"] is None for row in diagnostic["categories"].values())
    assert "PRIVATE" not in json.dumps(diagnostic)
    if state == "duplicate":
        assert diagnostic["categories"]["simple_python"]["duplicate_ids"] == ["simple_python_0"]


def test_result_diagnostics_only_exact_model_and_category_are_observed(tmp_path):
    before, error = diagnostics_before(tmp_path, ("multiple",))
    write_results(tmp_path, "multiple", [{"id": "multiple_0", "result": []}])
    write_results(
        tmp_path, "parallel_multiple", [{"id": "parallel_multiple_0", "result": UNAVAILABLE}]
    )
    write_results(
        tmp_path, "multiple", [{"id": "multiple_9", "result": UNAVAILABLE}], model="Other/Model-FC"
    )
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["multiple"], before, error, {"multiple": 1}
    )
    assert diagnostic["status"] == "observed"
    assert diagnostic["counts"]["http_503_no_available_workers"] == 0
    assert diagnostic["categories"]["multiple"]["unique_ids"] == 1


def test_result_diagnostics_partial_result_count_is_unknown(tmp_path):
    before, error = diagnostics_before(tmp_path)
    write_results(tmp_path, "simple_python", [{"id": "simple_python_0", "result": []}])
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error, {"simple_python": 400}
    )
    assert diagnostic["status"] == "unknown" and diagnostic["counts"] is None
    assert diagnostic["categories"]["simple_python"]["reason"] == "result_and_score_counts_differ"
    assert diagnostic["categories"]["simple_python"]["unique_ids"] == 1


def test_result_diagnostics_automatic_collection_preserves_generate_contract(tmp_path, monkeypatch):
    arm = run_ab.parse_arm("smg=http://127.0.0.1:30000", tmp_path)
    calls = []

    def fake_run(cmd, env, label, timeout=0):
        calls.append((cmd, label, timeout))
        root = Path(env["BFCL_PROJECT_ROOT"])
        if cmd[1] == "generate":
            write_results(
                root,
                "simple_python",
                [
                    {"id": "simple_python_0", "result": []},
                    {"id": "simple_python_1", "result": UNAVAILABLE},
                ],
            )
        elif cmd[1] == "evaluate":
            score_dir = cmd[cmd.index("--score-dir") + 1]
            write_score(root / score_dir, "non_live", "simple_python", 0.9, 2)
        else:
            raise AssertionError("unknown synthetic BFCL command")
        return True

    monkeypatch.setattr(run_ab, "_run", fake_run)
    run_ab.run_bfcl(
        arm,
        bfcl="fixture-only",
        model=MODEL,
        categories=["simple_python"],
        num_threads=16,
        temperature=0.001,
        skip_generate=False,
        run_timeout=14400,
    )
    assert arm.scores == {"simple_python": 0.9} and arm.counts == {"simple_python": 2}
    assert arm.timed_out == []
    assert arm.result_diagnostics["counts"]["http_503_no_available_workers"] == 1
    assert [call[0][1] for call in calls] == ["generate", "evaluate"]
    assert all(call[2] == 14400 for call in calls)
    generate = calls[0][0]
    assert generate[generate.index("--num-threads") + 1] == "16"
    assert generate[generate.index("--temperature") + 1] == "0.001"
    run_ab.run_bfcl(
        arm,
        bfcl="fixture-only",
        model=MODEL,
        categories=["simple_python"],
        num_threads=16,
        temperature=0.001,
        skip_generate=True,
    )
    assert arm.result_diagnostics["status"] == "unknown"
    assert arm.result_diagnostics["reason"] == "generation_skipped"


def test_result_diagnostics_round_trip_legacy_and_quality_gate_are_separate(tmp_path):
    from types import SimpleNamespace

    baseline = run_ab.parse_arm("vllm=http://127.0.0.1:30000", tmp_path)
    candidate = run_ab.parse_arm("smg=http://127.0.0.1:30001", tmp_path)
    for arm, result in [(baseline, []), (candidate, UNAVAILABLE)]:
        before, error = diagnostics_before(arm.project_root)
        write_results(
            arm.project_root, "simple_python", [{"id": "simple_python_0", "result": result}]
        )
        arm.scores, arm.counts = {"simple_python": 0.9}, {"simple_python": 1}
        arm.result_diagnostics = run_ab.collect_result_diagnostics(
            arm.project_root, MODEL, ["simple_python"], before, error, arm.counts
        )
        saved = tmp_path / (arm.name + ".json")
        run_ab.save_scores(arm, saved)
        restored = run_ab.load_scores(saved)
        assert restored.result_diagnostics == arm.result_diagnostics
        assert restored.scores == arm.scores and restored.timed_out == []
    output = tmp_path / "report.json"
    args = SimpleNamespace(out=None, json_out=output, tolerance=0.02, allow_incomplete=False)
    assert (
        run_ab.write_report_and_gate(baseline, candidate, ["simple_python"], args) == run_ab.EXIT_OK
    )
    payload = json.loads(output.read_text())
    assert payload["overall"] == {"baseline": 0.9, "candidate": 0.9, "delta": 0.0}
    assert payload["incomplete"] == {}
    assert (
        payload["candidate"]["result_diagnostics"]["counts"]["http_503_no_available_workers"] == 1
    )
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"name": "old", "scores": {"simple_python": 0.9}}))
    restored = run_ab.load_scores(legacy)
    assert restored.result_diagnostics["status"] == "unknown"
    assert restored.result_diagnostics["counts"] is None


@pytest.mark.parametrize(
    "message,kind",
    [
        (
            "Error during inference: Error code: 429 - prompt mentioned503 no_available_workers",
            "other_inference_error",
        ),
        (
            "Error during inference: Error code: 500 - context: Error code: 503 no_available_workers",
            "other_inference_error",
        ),
        (
            "Error during inference: Error code: 5030 - no_available_workers",
            "other_inference_error",
        ),
        (
            "Error during inference: context mentioned503 no_available_workers",
            "other_inference_error",
        ),
        ("Error during inference: Error code: 503 - unrelated_failure", "other_inference_error"),
        (
            "Error during inference: HTTP 503 - no_available_workers",
            "http_503_no_available_workers",
        ),
    ],
)
def test_result_diagnostics_explicit_status_precedes_quoted_context(tmp_path, message, kind):
    before, error = diagnostics_before(tmp_path)
    write_results(tmp_path, "simple_python", [{"id": "simple_python_0", "result": message}])
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error, {"simple_python": 1}
    )
    assert diagnostic["status"] == "observed"
    assert diagnostic["counts"][kind] == 1
    assert diagnostic["counts"]["http_503_no_available_workers"] == (
        kind == "http_503_no_available_workers"
    )


@pytest.mark.parametrize("depth", [65, 1100])
def test_result_diagnostics_deep_data_is_unknown_without_aborting_score_report(
    tmp_path, monkeypatch, depth
):
    from types import SimpleNamespace

    arm = run_ab.parse_arm("smg=http://127.0.0.1:30000", tmp_path)

    def fake_run(cmd, env, label, timeout=0):
        root = Path(env["BFCL_PROJECT_ROOT"])
        if cmd[1] == "generate":
            path = write_results(root, "simple_python", [])
            nested = "[" * depth + '"ok"' + "]" * depth
            path.write_text('{"id":"simple_python_0","result":' + nested + "}\n")
        elif cmd[1] == "evaluate":
            score_dir = cmd[cmd.index("--score-dir") + 1]
            write_score(root / score_dir, "non_live", "simple_python", 0.9, 1)
        else:
            raise AssertionError("unknown synthetic BFCL command")
        return True

    monkeypatch.setattr(run_ab, "_run", fake_run)
    run_ab.run_bfcl(
        arm,
        bfcl="fixture-only",
        model=MODEL,
        categories=["simple_python"],
        num_threads=16,
        temperature=0.001,
        skip_generate=False,
    )
    assert arm.scores == {"simple_python": 0.9} and arm.timed_out == []
    assert arm.result_diagnostics["status"] == "unknown"
    assert arm.result_diagnostics["counts"] is None
    args = SimpleNamespace(
        out=None, json_out=tmp_path / "deep-report.json", tolerance=0.02, allow_incomplete=False
    )
    assert run_ab.write_report_and_gate(arm, arm, ["simple_python"], args) == run_ab.EXIT_OK
    payload = json.loads(args.json_out.read_text())
    assert payload["overall"]["delta"] == 0.0 and payload["incomplete"] == {}
    assert payload["candidate"]["result_diagnostics"]["status"] == "unknown"


@pytest.mark.parametrize(
    "state",
    [
        "unbound_dictionary",
        "missing_model",
        "bad_digest",
        "forged_id",
        "boolean_count",
        "stale_source",
        "missing_source",
        "escaped_root",
        "absolute_root",
        "moved_artifact",
    ],
)
def test_result_diagnostics_saved_observed_requires_actual_matching_source(tmp_path, state):
    arm = run_ab.parse_arm("smg=http://127.0.0.1:30000", tmp_path)
    before, error = diagnostics_before(arm.project_root)
    result = write_results(
        arm.project_root, "simple_python", [{"id": "simple_python_0", "result": []}]
    )
    arm.scores, arm.counts = {"simple_python": 0.9}, {"simple_python": 1}
    arm.result_diagnostics = run_ab.collect_result_diagnostics(
        arm.project_root, MODEL, ["simple_python"], before, error, arm.counts
    )
    saved = tmp_path / "cached.json"
    run_ab.save_scores(arm, saved)
    assert run_ab.load_scores(saved).result_diagnostics == arm.result_diagnostics
    data = json.loads(saved.read_text())
    row = data["result_diagnostics"]["categories"]["simple_python"]
    if state == "unbound_dictionary":
        data["result_diagnostics"] = {
            "status": "observed",
            "counts": {"http_503_no_available_workers": 0},
            "categories": {},
        }
    elif state == "missing_model":
        del data["result_diagnostics"]["model"]
    elif state == "bad_digest":
        row["source"]["sha256"] = "0" * 64
    elif state == "forged_id":
        row["id_inventory"][0]["id"] = "simple_python_99"
    elif state == "boolean_count":
        data["result_diagnostics"]["counts"]["http_503_no_available_workers"] = False
    elif state == "stale_source":
        result.write_text(json.dumps({"id": "simple_python_0", "result": UNAVAILABLE}) + "\n")
    elif state == "missing_source":
        result.unlink()
    elif state == "escaped_root":
        data["result_source_root"] = "../outside"
    elif state == "absolute_root":
        data["result_source_root"] = str(arm.project_root)
    elif state == "moved_artifact":
        saved = tmp_path / "new-location" / "cached.json"
        saved.parent.mkdir()
    saved.write_text(json.dumps(data))
    restored = run_ab.load_scores(saved)
    assert restored.scores == arm.scores and restored.counts == arm.counts
    assert restored.timed_out == arm.timed_out
    assert restored.result_diagnostics["status"] == "unknown"
    assert restored.result_diagnostics["counts"] is None


def test_result_diagnostics_valid_unknown_inventory_round_trip_stays_unknown(tmp_path):
    arm = run_ab.parse_arm("smg=http://127.0.0.1:30000", tmp_path)
    before, error = diagnostics_before(arm.project_root)
    write_results(
        arm.project_root, "simple_python", [{"id": "simple_python_0", "result": UNAVAILABLE}]
    )
    arm.scores, arm.counts = {"simple_python": 0.9}, {"simple_python": 400}
    arm.result_diagnostics = run_ab.collect_result_diagnostics(
        arm.project_root, MODEL, ["simple_python"], before, error, arm.counts
    )
    assert arm.result_diagnostics["status"] == "unknown"
    saved = tmp_path / "unknown.json"
    run_ab.save_scores(arm, saved)
    restored = run_ab.load_scores(saved)
    assert restored.result_diagnostics == arm.result_diagnostics
    assert restored.result_diagnostics["counts"] is None
    assert restored.result_diagnostics["categories"]["simple_python"]["unique_ids"] == 1


@pytest.mark.parametrize(
    "model_output",
    [
        [{"calculate_area": '{"base":10,"height":5}'}],
        None,
        [[{"first_call": '{"x":1}'}], [{"second_call": '{"y":2}'}]],
        [
            {
                "tool": {
                    "nested_argument": "Error during inference: Error code: 503 - no_available_workers"
                }
            }
        ],
        [{"tool": "Error during inference: Error code: 503 - no_available_workers"}],
        [None, {"tool": '{"empty":null}'}],
    ],
)
def test_real_bfcl_fc_dict_null_and_nested_outputs_keep_infra_counts(tmp_path, model_output):
    # Real pinned BFCL FC transcripts use list[dict[function, arguments]], and
    # multi-turn nests those lists. Function arguments are model output, never
    # a caught infrastructure error, even if their text resembles one.
    before, error = diagnostics_before(tmp_path)
    write_results(
        tmp_path,
        "simple_python",
        [
            {"id": "simple_python_0", "result": model_output},
            {"id": "simple_python_1", "result": UNAVAILABLE},
        ],
    )
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error, {"simple_python": 2}
    )
    assert diagnostic["status"] == "observed"
    assert diagnostic["counts"] == {
        "non_error_result": 1,
        "http_503_no_available_workers": 1,
        "other_inference_error": 0,
    }
    row = diagnostic["categories"]["simple_python"]
    assert row["unique_ids"] == 2 and row["duplicate_ids"] == []
    assert [item["classification"] for item in row["id_inventory"]] == [
        "non_error_result",
        "http_503_no_available_workers",
    ]


def test_result_diagnostics_missing_result_key_is_not_a_null_model_reply(tmp_path):
    before, error = diagnostics_before(tmp_path)
    write_results(tmp_path, "simple_python", [{"id": "simple_python_0"}])
    diagnostic = run_ab.collect_result_diagnostics(
        tmp_path, MODEL, ["simple_python"], before, error, {"simple_python": 1}
    )
    assert diagnostic["status"] == "unknown"
    assert diagnostic["counts"] is None
