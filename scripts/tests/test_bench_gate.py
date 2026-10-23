"""The benchmark regression gate: its verdicts, its criterion reader and its run loop."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench_gate.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("bench_gate", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_gate"] = module
    spec.loader.exec_module(module)
    return module


GATE_TOML = """
repeats = 3
filter = "^(encode|decode)/"

[[gate]]
pattern = "^encode/"
threshold = 0.10

[[gate]]
pattern = "^decode/slow$"
threshold = 0.25
note = "the noisiest id"
"""


@pytest.fixture
def gates(mod, tmp_path):
    path = tmp_path / "gates.toml"
    path.write_text(GATE_TOML)
    return mod.load_gates(path)


def results(base: dict[str, list[float]], pr: dict[str, list[float]], repeats: int = 3) -> dict:
    def runs(values: dict[str, list[float]]) -> list[dict[str, float]]:
        n = max((len(v) for v in values.values()), default=0)
        return [{k: v[i] for k, v in values.items() if i < len(v)} for i in range(n)]

    return {
        "repeats": repeats,
        "base": {"runs": runs(base)},
        "pr": {"runs": runs(pr)},
    }


def verdicts(mod, gates, res) -> dict[str, str]:
    return {row.bench_id: row.verdict for row in mod.compare(res, gates)}


# ---------------------------------------------------------------- gate file


def test_gate_file_is_read_in_order(gates):
    assert gates.repeats == 3
    assert gates.filter == "^(encode|decode)/"
    assert gates.threshold_for("encode/short").threshold == 0.10
    assert gates.threshold_for("decode/slow").threshold == 0.25
    assert gates.threshold_for("decode/slow").note == "the noisiest id"
    assert gates.threshold_for("decode/other") is None


def test_gate_file_rejects_bad_values(mod, tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[[gate]]\npattern = "x"\nthreshold = 10\n')
    with pytest.raises(ValueError):
        mod.load_gates(path)
    path.write_text("repeats = 0\n")
    with pytest.raises(ValueError):
        mod.load_gates(path)
    # A bad gate file is a usage error (ValueError, exit 2), never a traceback or a silent
    # default: an unknown key, an entry without its pattern or threshold, a pattern or a
    # filter that does not compile.
    for text, named in [
        ('filters = "^encode/"\n', "unknown keys"),
        ("repeat = 5\n", "unknown keys"),
        ("[[gate]]\nthreshold = 0.1\n", "needs pattern and threshold"),
        ('[[gate]]\npattern = "x"\n', "needs pattern and threshold"),
        ('[[gate]]\npattern = "x"\nthreshold = 0.1\nthreshhold = 0.2\n', "unknown keys"),
        ('[[gate]]\npattern = "(unclosed"\nthreshold = 0.1\n', "does not compile"),
        ('filter = "(unclosed"\n', "does not compile"),
        # A value of the wrong TOML type is a usage error too, not a TypeError traceback.
        ('filter = ["^a/", "^b/"]\n', "filter must be a string"),
        ("filter = 5\n", "filter must be a string"),
        ("repeats = [3]\n", "repeats must be an integer"),
        ("bench_args = 5\n", "bench_args must be a list"),
        ("[[gate]]\npattern = 1\nthreshold = 0.1\n", "pattern must be a string"),
        ('[[gate]]\npattern = "x"\nthreshold = [0.1]\n', "threshold must be a number"),
        # A single [gate] table where the [[gate]] array belongs.
        ('[gate]\npattern = "x"\nthreshold = 0.1\n', "array of tables"),
    ]:
        path.write_text(text)
        with pytest.raises(ValueError, match=named):
            mod.load_gates(path)
    assert mod.main(["compare", "--gates", str(path), str(path)]) == 2


# ---------------------------------------------------------------- verdicts


def test_regression_needs_threshold_and_separation(mod, gates):
    res = results(
        {"encode/a": [100, 101, 102], "encode/b": [100, 101, 102], "encode/c": [100, 101, 102]},
        {
            "encode/a": [115, 116, 117],  # +16 %, every PR run slower: fails
            "encode/b": [115, 116, 101],  # +16 % by medians but one PR run inside the base runs
            "encode/c": [105, 106, 107],  # +6 %, separated, inside the 10 % threshold
        },
    )
    v = verdicts(mod, gates, res)
    assert v["encode/a"] == "regression"
    assert v["encode/b"] == "suspect (runs overlap)"
    assert v["encode/c"] == "ok"
    assert mod.report(res, gates, None, None) == 1


def test_improvement_and_report_only_never_fail(mod, gates):
    res = results(
        {"encode/a": [100, 101, 102], "decode/other": [100, 101, 102]},
        {"encode/a": [80, 81, 82], "decode/other": [200, 201, 202]},
    )
    v = verdicts(mod, gates, res)
    assert v["encode/a"] == "improvement"
    assert v["decode/other"] == "report only"
    assert mod.report(res, gates, None, None) == 0


def test_per_id_threshold_applies(mod, gates):
    res = results(
        {"decode/slow": [100, 101, 102]},
        {"decode/slow": [120, 121, 122]},  # +20 % is inside this id's 25 %
    )
    assert verdicts(mod, gates, res)["decode/slow"] == "ok"


def test_missing_gated_id_fails_and_new_id_is_reported(mod, gates):
    res = results(
        {"encode/a": [100, 101, 102], "decode/other": [5, 5, 5]},
        {"encode/new": [1, 1, 1]},
    )
    v = verdicts(mod, gates, res)
    assert v["encode/a"] == "missing"
    assert v["decode/other"] == "missing (report only)"
    assert v["encode/new"] == "new"
    assert mod.report(res, gates, None, None) == 1


def test_single_repeat_degenerates_to_a_threshold(mod, gates):
    res = results({"encode/a": [100]}, {"encode/a": [120]}, repeats=1)
    assert verdicts(mod, gates, res)["encode/a"] == "regression"


def test_spread_is_the_larger_side(mod, gates):
    res = results({"encode/a": [100, 100, 100]}, {"encode/a": [100, 110, 100]})
    row = mod.compare(res, gates)[0]
    assert row.spread == pytest.approx(0.10)
    assert row.change == pytest.approx(0.0)


# ---------------------------------------------------------------- markdown


def test_markdown_scales_units_and_marks_failures(mod, gates, tmp_path):
    res = results(
        {"encode/a": [66_040.0, 65_240.0, 76_270.0], "decode/other": [2.3e9, 2.3e9, 2.3e9]},
        {"encode/a": [80_000.0, 81_000.0, 82_000.0], "decode/other": [2.3e9, 2.3e9, 2.3e9]},
    )
    summary = tmp_path / "summary.md"
    assert mod.report(res, gates, tmp_path / "gate.md", str(summary)) == 1
    text = summary.read_text()
    assert text == (tmp_path / "gate.md").read_text()
    assert "### Benchmark gate: FAIL" in text
    assert "| encode/a | 66.04 µs (65.24 µs - 76.27 µs) | 81 µs (80 µs - 82 µs) | +22.7%" in text
    assert "| 2.3 s" in text
    assert "FAIL regression" in text
    assert "2 ids, 1 gated, 1 failed, 0 suspect" in text


def test_github_step_summary_is_appended(mod, gates, tmp_path, monkeypatch):
    summary = tmp_path / "step.md"
    summary.write_text("before\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    res = results({"encode/a": [1.0, 1.0, 1.0]}, {"encode/a": [1.0, 1.0, 1.0]})
    assert mod.report(res, gates, None, None) == 0
    assert summary.read_text().startswith("before\n### Benchmark gate: PASS")


# ---------------------------------------------------------------- criterion reader


def write_criterion(home: Path, values: dict[str, float]) -> None:
    for bench_id, median in values.items():
        new = home.joinpath(*bench_id.split("/")) / "new"
        new.mkdir(parents=True)
        (new / "benchmark.json").write_text(json.dumps({"full_id": bench_id}))
        (new / "estimates.json").write_text(
            json.dumps({"median": {"point_estimate": median}, "mean": {"point_estimate": 0}})
        )
    # a stale "base" directory criterion keeps next to "new" is not a run
    stale = home / "encode" / "stale" / "base"
    stale.mkdir(parents=True)
    (stale / "benchmark.json").write_text(json.dumps({"full_id": "encode/stale"}))
    (stale / "estimates.json").write_text(json.dumps({"median": {"point_estimate": 1}}))


def test_collect_reads_new_estimates_only(mod, tmp_path):
    write_criterion(tmp_path, {"encode/a": 10.5, "decode/slow": 2000.0})
    assert mod.collect(tmp_path) == {"encode/a": 10.5, "decode/slow": 2000.0}


# ---------------------------------------------------------------- build helper


def test_select_bench_executable(mod):
    messages = [
        {
            "reason": "compiler-artifact",
            "target": {"kind": ["lib"], "name": "smg"},
            "executable": None,
        },
        {
            "reason": "compiler-artifact",
            "target": {"kind": ["bench"], "name": "tokenizer_benchmark"},
            "executable": "/t/release/deps/tokenizer_benchmark-abc",
        },
        {
            "reason": "compiler-artifact",
            "target": {"kind": ["bench"], "name": "request_processing"},
            "executable": "/t/release/deps/request_processing-def",
        },
        {"reason": "build-finished", "success": True},
    ]
    assert (
        mod.select_bench_executable(messages, "tokenizer_benchmark")
        == "/t/release/deps/tokenizer_benchmark-abc"
    )
    with pytest.raises(ValueError, match="expected one bench executable"):
        mod.select_bench_executable(messages, None)
    assert (
        mod.select_bench_executable(messages[:2], None) == "/t/release/deps/tokenizer_benchmark-abc"
    )


# ---------------------------------------------------------------- run loop


FAKE_BENCH = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
home = Path(os.environ["CRITERION_HOME"])
assert sys.argv[1:3] == ["--bench", "--noplot"], sys.argv
assert sys.argv[-1] == "^(encode|decode)/", sys.argv
values = json.loads(Path(__file__).with_suffix(".json").read_text())
run = len([p for p in home.parent.iterdir() if p.is_dir()])  # 1-based index of this run
for bench_id, series in values.items():
    new = home.joinpath(*bench_id.split("/")) / "new"
    new.mkdir(parents=True)
    (new / "benchmark.json").write_text(json.dumps({"full_id": bench_id}))
    (new / "estimates.json").write_text(json.dumps({"median": {"point_estimate": series[run - 1]}}))
print("fake bench", sys.argv[1:], file=sys.stderr)
"""


def fake_bench(path: Path, values: dict[str, list[float]]) -> Path:
    path.write_text(FAKE_BENCH)
    path.with_suffix(".json").write_text(json.dumps(values))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_run_interleaves_and_compares(mod, gates, tmp_path, monkeypatch):
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100, 101, 102], "decode/x": [1, 1, 1]})
    pr = fake_bench(tmp_path / "pr.py", {"encode/a": [130, 131, 132], "decode/x": [1, 1, 1]})
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    out = tmp_path / "out"
    rc = mod.main(
        ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr), "--out", str(out)]
    )
    assert rc == 1
    saved = json.loads((out / "results.json").read_text())
    assert saved["order"] == ["base-1", "pr-1", "pr-2", "base-2", "base-3", "pr-3"]
    assert [run["encode/a"] for run in saved["base"]["runs"]] == [100, 101, 102]
    assert [run["encode/a"] for run in saved["pr"]["runs"]] == [130, 131, 132]
    assert (out / "base" / "run-2.log").read_text().startswith("+ ")
    assert "FAIL regression" in (out / "gate.md").read_text()
    # the same results re-evaluated against a looser gate pass
    loose = tmp_path / "loose.toml"
    loose.write_text(GATE_TOML.replace("threshold = 0.10", "threshold = 0.50"))
    assert mod.main(["compare", "--gates", str(loose), str(out / "results.json")]) == 0


def test_run_output_dir_is_resolved_before_the_bench_runs_under_cwd(mod, tmp_path, monkeypatch):
    """A relative --out must not land under --cwd: the bench gets it through the environment."""
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100], "decode/x": [1]})
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML.replace("repeats = 3", "repeats = 1"))
    bench_cwd = tmp_path / "crate"
    bench_cwd.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    rc = mod.main(
        [
            "run",
            "--gates",
            str(gate_file),
            "--base",
            str(base),
            "--pr",
            str(base),
            "--cwd",
            "crate",
            "--out",
            "gate-out",
        ]
    )
    assert rc == 0
    assert (tmp_path / "gate-out" / "results.json").exists()
    assert (tmp_path / "gate-out" / "base" / "run-1" / "encode" / "a" / "new").is_dir()
    assert not (bench_cwd / "gate-out").exists()


def test_run_reports_a_crashing_bench(mod, tmp_path, capsys):
    crash = tmp_path / "crash.py"
    crash.write_text("#!/usr/bin/env python3\nraise SystemExit(3)\n")
    crash.chmod(crash.stat().st_mode | stat.S_IXUSR)
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text("repeats = 1\n")
    rc = mod.main(
        [
            "run",
            "--gates",
            str(gate_file),
            "--base",
            str(crash),
            "--pr",
            str(crash),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert rc == 2
    assert "exited with 3" in capsys.readouterr().err


def test_interleaved_order(mod):
    assert mod.interleaved_order(1) == [(1, "base"), (1, "pr")]
    assert mod.interleaved_order(2) == [(1, "base"), (1, "pr"), (2, "pr"), (2, "base")]


def test_executable_bit(mod):
    assert os.access(SCRIPT, os.X_OK)
