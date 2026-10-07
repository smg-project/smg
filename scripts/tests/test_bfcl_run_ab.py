import importlib.util
import json
import sys
from pathlib import Path

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
