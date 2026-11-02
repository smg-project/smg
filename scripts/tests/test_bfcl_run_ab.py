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
    assert run_ab.incompleteness(arm, ["simple_python"]) is not None
    assert len(set(score_dirs)) == 2


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
        assert all("timed out during: evaluate" in v for v in payload["incomplete"].values())
    else:
        assert payload["incomplete"] == {}
