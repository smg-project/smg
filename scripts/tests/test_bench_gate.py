"""The benchmark regression gate: its verdicts, its criterion reader and its run loop."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import types
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

SOURCES_TOML = GATE_TOML.replace(
    'filter = "^(encode|decode)/"',
    'filter = "^(encode|decode)/"\nsources = ["benches/a.rs", "benches/gone.rs"]',
)


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
    assert gates.sources == []


def test_gate_file_names_the_bench_sources(mod, tmp_path):
    path = tmp_path / "gates.toml"
    path.write_text(SOURCES_TOML)
    assert mod.load_gates(path).sources == ["benches/a.rs", "benches/gone.rs"]


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
        ('sources = "benches/a.rs"\n', "sources must be a list"),
        ("sources = [1]\n", "sources must be a list"),
        ('count = "yes"\n', "count must be true or false"),
        ("count = 1\n", "count must be true or false"),
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


# ---------------------------------------------------------------- bench sources


FAKE_CARGO = """#!/usr/bin/env python3
import json, sys
from pathlib import Path
a = Path("benches/a.rs")
text = a.read_text() if a.exists() else None
Path("seen.json").write_text(json.dumps({"a": text, "gone": Path("benches/gone.rs").exists()}))
if text and "FAIL" in text:
    sys.exit(1)
Path("built").write_text("bin")
message = {"reason": "compiler-artifact", "target": {"kind": ["bench"], "name": "t"}}
message["executable"] = str(Path("built").resolve())
print(json.dumps(message))
"""


def git(repo: Path, *args: str) -> str:
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    done = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE, text=True, env=env
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository at its base commit (detached, as the workflows check the base out), whose
    head commit changes benches/a.rs and removes benches/gone.rs; the fake cargo records what it
    saw of both in seen.json."""
    repo = tmp_path / "repo"
    (repo / "benches").mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "benches" / "a.rs").write_text("v1\n")
    (repo / "benches" / "gone.rs").write_text("only on the base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "base")
    (repo / "benches" / "a.rs").write_text("v2\n")
    (repo / "benches" / "gone.rs").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "head")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "HEAD^")
    cargo = tmp_path / "cargo.py"
    cargo.write_text(FAKE_CARGO)
    cargo.chmod(cargo.stat().st_mode | stat.S_IXUSR)
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(SOURCES_TOML)
    monkeypatch.chdir(repo)
    return repo, head, cargo, gate_file


def test_build_carries_the_bench_sources_of_the_named_commit(mod, repo, tmp_path):
    repo_dir, head, cargo, gate_file = repo
    out = tmp_path / "bin" / "base"
    rc = mod.main(
        [
            "build",
            "--gates",
            str(gate_file),
            "--sources-from",
            head,
            "--out",
            str(out),
            "--",
            str(cargo),
        ]
    )
    assert rc == 0
    # cargo saw the head's files: a.rs as the head has it and gone.rs absent
    assert json.loads((repo_dir / "seen.json").read_text()) == {"a": "v2\n", "gone": False}
    # and the checkout is its own again
    assert (repo_dir / "benches" / "a.rs").read_text() == "v1\n"
    assert (repo_dir / "benches" / "gone.rs").exists()
    assert os.access(out, os.X_OK)
    record = json.loads(mod.sources_record_path(out).read_text())
    assert record == {
        "rev": head,
        "applied": True,
        "files": {"benches/a.rs": hashlib.sha256(b"v2\n").hexdigest(), "benches/gone.rs": None},
    }
    # without a commit the record names the checkout's own files
    rc = mod.main(["build", "--gates", str(gate_file), "--out", str(out), "--", str(cargo)])
    assert rc == 0
    record = json.loads(mod.sources_record_path(out).read_text())
    assert record["applied"] is None
    assert record["files"]["benches/a.rs"] == hashlib.sha256(b"v1\n").hexdigest()
    assert record["files"]["benches/gone.rs"] == hashlib.sha256(b"only on the base\n").hexdigest()


def test_build_falls_back_to_the_checkout_sources_when_the_named_ones_do_not_build(
    mod, repo, tmp_path, capsys
):
    repo_dir, head, cargo, gate_file = repo
    base = git(repo_dir, "rev-parse", "HEAD")
    git(repo_dir, "checkout", "-q", "-b", "needs-the-head", head)
    (repo_dir / "benches" / "a.rs").write_text("v2 FAIL\n")
    git(repo_dir, "commit", "-q", "-am", "a row that builds on the head alone")
    failing = git(repo_dir, "rev-parse", "HEAD")
    git(repo_dir, "checkout", "-q", base)
    out = tmp_path / "bin" / "base"
    rc = mod.main(
        [
            "build",
            "--gates",
            str(gate_file),
            "--sources-from",
            failing,
            "--out",
            str(out),
            "--",
            str(cargo),
        ]
    )
    assert rc == 0
    assert json.loads((repo_dir / "seen.json").read_text()) == {"a": "v1\n", "gone": True}
    assert (repo_dir / "benches" / "a.rs").read_text() == "v1\n"
    record = json.loads(mod.sources_record_path(out).read_text())
    assert record["applied"] is False
    assert (
        record["reason"]
        == f"The bench sources of {failing} did not build here (cargo exited with 1)"
    )
    assert record["files"]["benches/a.rs"] == hashlib.sha256(b"v1\n").hexdigest()
    assert "did not build here" in capsys.readouterr().err
    # a revision the repository does not have is a usage error
    assert (
        mod.main(
            [
                "build",
                "--gates",
                str(gate_file),
                "--sources-from",
                "no-such",
                "--out",
                str(out),
                "--",
                str(cargo),
            ]
        )
        == 2
    )
    # and a gate file without sources has nothing to carry
    plain = tmp_path / "plain.toml"
    plain.write_text(GATE_TOML)
    assert (
        mod.main(
            [
                "build",
                "--gates",
                str(plain),
                "--sources-from",
                head,
                "--out",
                str(out),
                "--",
                str(cargo),
            ]
        )
        == 2
    )


# ---------------------------------------------------------------- run loop


FAKE_BENCH = """#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path
home = Path(os.environ["CRITERION_HOME"])
assert sys.argv[1:3] == ["--bench", "--noplot"], sys.argv
bench_filter = sys.argv[-1]  # the gate file's filter, or the exact ids of a re-measurement
values = json.loads(Path(r"__SERIES__").read_text())  # written by fake_bench; the copies read it too
run = len([p for p in home.parent.iterdir() if p.is_dir()])  # 1-based index of this run
for bench_id, series in values.items():
    if not re.search(bench_filter, bench_id):
        continue
    new = home.joinpath(*bench_id.split("/")) / "new"
    new.mkdir(parents=True)
    value = series[min(run, len(series)) - 1]  # a series shorter than the runs repeats its last
    (new / "benchmark.json").write_text(json.dumps({"full_id": bench_id}))
    (new / "estimates.json").write_text(json.dumps({"median": {"point_estimate": value}}))
print("fake bench", sys.argv[1:], "link", os.path.islink(__file__), file=sys.stderr)
"""


def fake_bench(path: Path, values: dict[str, list[float]]) -> Path:
    # Two sides are two different files: the gate passes binaries of the same bytes without a run.
    path.write_text(
        FAKE_BENCH.replace("__SERIES__", str(path.with_suffix(".json"))) + f"# {path.name}\n"
    )
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
    # three interleaved runs, then the failing id once more per side, the pull request first
    assert saved["order"] == [
        "base-1",
        "pr-1",
        "pr-2",
        "base-2",
        "base-3",
        "pr-3",
        "pr-4",
        "base-4",
    ]
    assert [run["encode/a"] for run in saved["base"]["runs"]] == [100, 101, 102, 102]
    assert [run["encode/a"] for run in saved["pr"]["runs"]] == [130, 131, 132, 132]
    assert saved["pr"]["runs"][3] == {"encode/a": 132}, (
        "the re-measurement runs the failing id alone"
    )
    assert saved["recheck"] == {"ids": ["encode/a"], "rounds": 1}
    assert saved["identical"] is False
    assert (out / "base" / "run-2.log").read_text().startswith("+ ")
    # each side ran from a fresh copy of its binary (a file, not a link) through paths of one
    # length, and the copies are gone after the runs: the output directory is the artifact
    for side in ("base", "head"):
        log = (out / side / "run-1.log").read_text()
        assert log.startswith(f"+ {out / side / 'bench'} --bench") and "link False" in log
        assert not (out / side / "bench").exists()
    assert (out / "head" / "run-1").is_dir()
    assert "FAIL regression" in (out / "gate.md").read_text()
    # the same results re-evaluated against a looser gate pass
    loose = tmp_path / "loose.toml"
    loose.write_text(GATE_TOML.replace("threshold = 0.10", "threshold = 0.50"))
    assert mod.main(["compare", "--gates", str(loose), str(out / "results.json")]) == 0


def test_run_names_the_bench_sources_the_two_binaries_carry(mod, tmp_path, monkeypatch):
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100, 101, 102], "decode/x": [1, 1, 1]})
    pr = fake_bench(tmp_path / "pr.py", {"encode/a": [130, 131, 132], "decode/x": [1, 1, 1]})
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    def record(binary: Path, files: dict, **more) -> None:
        mod.sources_record_path(binary).write_text(
            json.dumps({"rev": None, "applied": None, "files": files, **more})
        )

    args = ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr), "--out"]
    record(pr, {"benches/a.rs": "aa"})
    record(base, {"benches/a.rs": "aa"}, rev="h", applied=True)
    assert mod.main([*args, str(tmp_path / "same")]) == 1
    text = (tmp_path / "same" / "gate.md").read_text()
    assert (
        "Bench sources (1 named by the gate file): byte-identical in the two binaries, so the "
        "rows compare the measured code alone." in text
    )
    # the base could not carry the pull request's sources: the report says so and the
    # verdicts hold
    reason = "The bench sources of h did not build here (cargo exited with 101)"
    record(base, {"benches/a.rs": "bb"}, rev="h", applied=False, reason=reason)
    assert mod.main([*args, str(tmp_path / "differ")]) == 1
    text = (tmp_path / "differ" / "gate.md").read_text()
    assert (
        "Bench sources (1 named by the gate file): `benches/a.rs` differs between the two "
        f"binaries. {reason}, so the base's binary carries its own. A gated row that moves "
        "without a change to what it measures is the binary's inlining of a neighbour: a row "
        "this pull request adds goes to a bench target of its own." in text
    )
    assert "FAIL regression" in text
    saved = json.loads((tmp_path / "differ" / "results.json").read_text())
    assert saved["sources"]["base"]["applied"] is False
    assert saved["sources"]["pr"]["files"] == {"benches/a.rs": "aa"}
    # a binary without a record (a local build) says nothing
    mod.sources_record_path(base).unlink()
    assert mod.main([*args, str(tmp_path / "none")]) == 1
    assert "Bench sources" not in (tmp_path / "none" / "gate.md").read_text()


def test_run_output_dir_is_resolved_before_the_bench_runs_under_cwd(mod, tmp_path, monkeypatch):
    """A relative --out must not land under --cwd: the bench gets it through the environment."""
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100], "decode/x": [1]})
    pr = fake_bench(tmp_path / "pr.py", {"encode/a": [100], "decode/x": [1]})
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
            str(pr),
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
    other = tmp_path / "other.py"
    other.write_text("#!/usr/bin/env python3\nraise SystemExit(3)  # the other side\n")
    other.chmod(other.stat().st_mode | stat.S_IXUSR)
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
            str(other),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    assert rc == 2
    assert "exited with 3" in capsys.readouterr().err


def test_interleaved_order(mod):
    assert mod.interleaved_order(1) == [(1, "base"), (1, "pr")]
    assert mod.interleaved_order(2) == [(1, "base"), (1, "pr"), (2, "pr"), (2, "base")]
    assert mod.round_sides(4) == ("pr", "base")


def test_identical_binaries_pass_without_a_run(mod, tmp_path, monkeypatch):
    bench = fake_bench(tmp_path / "same.py", {"encode/a": [100, 101, 102]})
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    out = tmp_path / "out"
    args = ["run", "--gates", str(gate_file), "--base", str(bench), "--pr", str(bench)]
    assert mod.main([*args, "--out", str(out)]) == 0
    saved = json.loads((out / "results.json").read_text())
    assert saved["identical"] is True and saved["order"] == []
    assert saved["binaries"]["base"]["sha256"] == saved["binaries"]["pr"]["sha256"]
    text = (out / "gate.md").read_text()
    assert text.startswith("### Benchmark gate: PASS (identical binaries)")
    assert "the same bytes" in text
    assert not (out / "base").exists()
    assert mod.main(["compare", "--gates", str(gate_file), str(out / "results.json")]) == 0


def test_a_failing_id_is_re_measured_before_the_verdict(mod, tmp_path, monkeypatch):
    # The fourth value is the re-measurement, read within its round: a slow one confirms
    # the regression, a fast one does not, which is suspect, never a failure.
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100, 101, 102, 101]})
    burst = fake_bench(tmp_path / "burst.py", {"encode/a": [130, 131, 132, 95]})
    args = ["run", "--gates", str(gate_file), "--base", str(base)]
    out = tmp_path / "burst"
    assert mod.main([*args, "--pr", str(burst), "--out", str(out)]) == 0
    text = (out / "gate.md").read_text()
    assert "suspect (not confirmed by the re-measurement)" in text
    assert "Re-measured before the verdict" in text and "`encode/a`" in text
    assert (
        "pull request runs [130, 131, 132] ns; re-measured base 101 ns, pull request 95 ns (-5.9%)"
        in text
    )
    slow = fake_bench(tmp_path / "slow.py", {"encode/a": [130, 131, 132, 133]})
    out = tmp_path / "slow"
    assert mod.main([*args, "--pr", str(slow), "--out", str(out)]) == 1
    assert "FAIL regression" in (out / "gate.md").read_text()
    out = tmp_path / "none"
    assert mod.main([*args, "--pr", str(burst), "--out", str(out), "--recheck", "0"]) == 1
    saved = json.loads((out / "results.json").read_text())
    assert "recheck" not in saved and len(saved["order"]) == 6


def test_every_re_measured_round_runs_the_same_ids(mod, tmp_path, monkeypatch):
    # `--recheck 2`: a first round that does not confirm does not end the re-measurement;
    # both rounds run on the ids the full rounds failed, and the median reads them.
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100, 101, 102, 101, 100]})
    pr = fake_bench(tmp_path / "pr.py", {"encode/a": [130, 131, 132, 95, 131]})
    out = tmp_path / "two"
    args = ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr)]
    assert mod.main([*args, "--out", str(out), "--recheck", "2"]) == 1
    saved = json.loads((out / "results.json").read_text())
    assert saved["recheck"] == {"ids": ["encode/a"], "rounds": 2} and len(saved["order"]) == 10
    assert (
        "re-measured base 101 ns, pull request 95 ns (-5.9%); base 100 ns, pull request 131 ns (+31.0%)"
        in (out / "gate.md").read_text()
    )


def test_host_load_is_recorded_and_a_busy_host_is_not_judged(mod, tmp_path, monkeypatch):
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML.replace("repeats = 3", "repeats = 1"))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100], "decode/x": [1]})
    pr = fake_bench(tmp_path / "pr.py", {"encode/a": [100], "decode/x": [1]})
    args = ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr)]
    monkeypatch.setattr(mod, "host_load", lambda: 3.5)
    out = tmp_path / "quiet"
    assert mod.main([*args, "--out", str(out), "--busy-above", "4"]) == 0
    saved = json.loads((out / "results.json").read_text())
    assert saved["base"]["load"] == [3.5] and saved["pr"]["load"] == [3.5]
    assert (
        "Host load (one minute) before each run: base [3.5], pr [3.5]."
        in (out / "gate.md").read_text()
    )
    monkeypatch.setattr(mod, "host_load", lambda: 9.0)
    out = tmp_path / "busy"
    rc = mod.main([*args, "--out", str(out), "--busy-above", "4", "--busy-wait", "0"])
    assert rc == 2
    saved = json.loads((out / "results.json").read_text())
    assert saved["not_judged"].startswith("host busy: the one-minute load 9.0 stayed above 4")
    assert saved["order"] == []
    assert (out / "gate.md").read_text().startswith("### Benchmark gate: NOT JUDGED")
    # A host turning busy after the first run: the rows it left one-sided are not missing
    # ids, the saved file re-evaluates to 2, and the copies are gone.
    loads = iter([3.5, 3.5, 9.0, 9.0])
    monkeypatch.setattr(mod, "host_load", lambda: next(loads))
    out = tmp_path / "stopped"
    assert mod.main([*args, "--out", str(out), "--busy-above", "4", "--busy-wait", "0"]) == 2
    saved = json.loads((out / "results.json").read_text())
    assert saved["order"] == ["base-1"]
    text = (out / "gate.md").read_text()
    assert text.startswith("### Benchmark gate: NOT JUDGED")
    assert "FAIL missing" not in text and "| not judged (the run stopped) |" in text
    assert mod.report(saved, mod.load_gates(gate_file), None, None) == 2
    assert not (out / "base" / "bench").exists() and not (out / "head" / "bench").exists()


def test_cpu_pins_the_bench(mod, tmp_path, monkeypatch):
    if shutil.which("taskset") is None:
        pytest.skip("no taskset on this host")
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(GATE_TOML.replace("repeats = 3", "repeats = 1"))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base = fake_bench(tmp_path / "base.py", {"encode/a": [100], "decode/x": [1]})
    pr = fake_bench(tmp_path / "pr.py", {"encode/a": [100], "decode/x": [1]})
    out = tmp_path / "out"
    args = ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr), "--out"]
    assert mod.main([*args, str(out), "--cpu", "0"]) == 0
    assert (out / "base" / "run-1.log").read_text().startswith("+ taskset -c 0 ")


def test_per_run_values_are_listed_for_suspect_rows(mod, gates):
    res = results({"encode/a": [100, 101, 102]}, {"encode/a": [115, 116, 101]})
    text = mod.markdown(res, mod.compare(res, gates))
    assert (
        "- `encode/a`: suspect (runs overlap), +13.9%; base runs [100, 101, 102] ns, "
        "pull request runs [115, 116, 101] ns." in text
    )


def test_the_re_measured_round_is_read_within_itself(mod, gates):
    # A process that runs the failing ids alone sits on levels of its own: both sides read
    # 30 % faster in the re-measured round. Pooled, the runs would overlap; within the
    # round the regression stands.
    res = results({"encode/a": [10, 10, 10, 7.0]}, {"encode/a": [13, 13, 13, 9.1]})
    assert verdicts(mod, gates, res)["encode/a"] == "regression"
    text = mod.markdown(res, mod.compare(res, gates))
    assert "re-measured base 7 ns, pull request 9.1 ns (+30.0%)" in text
    # A re-measured round under half the threshold makes the row suspect.
    res = results({"encode/a": [10, 10, 10, 10]}, {"encode/a": [13, 13, 13, 10.3]})
    assert verdicts(mod, gates, res)["encode/a"] == "suspect (not confirmed by the re-measurement)"


def test_the_re_measured_round_rules_out_a_burst_not_the_size(mod, gates):
    # A real +30 % on a 25 % gate whose one re-measured pair swings to +21.8 % stays a
    # regression: the bar is half the threshold. Near zero, the failure was a burst.
    res = results({"decode/slow": [100, 102, 104, 101]}, {"decode/slow": [132, 133, 135, 123]})
    assert verdicts(mod, gates, res)["decode/slow"] == "regression"
    res = results({"decode/slow": [100, 102, 104, 101]}, {"decode/slow": [132, 133, 135, 103]})
    assert (
        verdicts(mod, gates, res)["decode/slow"] == "suspect (not confirmed by the re-measurement)"
    )
    # Two re-measured rounds are read on their median: one swing does not undo the other.
    res = results(
        {"decode/slow": [100, 102, 104, 101, 100]}, {"decode/slow": [132, 133, 135, 103, 131]}
    )
    assert verdicts(mod, gates, res)["decode/slow"] == "regression"


def test_per_run_values_print_millisecond_rows_without_an_exponent(mod):
    assert mod.fmt_runs([1500000.0, 2.5, 999999.4]) == "[1,500,000, 2.5, 999999]"


def test_executable_bit(mod):
    assert os.access(SCRIPT, os.X_OK)


# ---------------------------------------------------------------- other units


def test_a_zero_base_median_is_judged_on_the_absolute_values(mod, gates):
    gated_zero = results({"encode/zero": [0.0, 0.0, 0.0]}, {"encode/zero": [0.0, 0.0, 0.0]})
    assert verdicts(mod, gates, gated_zero) == {"encode/zero": "ok"}
    grown = results({"encode/zero": [0.0, 0.0, 0.0]}, {"encode/zero": [2.0, 2.0, 2.0]})
    assert verdicts(mod, gates, grown) == {"encode/zero": "regression"}
    text = mod.markdown(grown, mod.compare(grown, gates))
    assert "| - |" in text  # no relative change to print
    # An increase whose runs overlap the base's is suspect, as beyond a threshold.
    overlapping = results({"encode/zero": [0.0, 0.0, 0.0]}, {"encode/zero": [2.0, 0.0, 2.0]})
    assert verdicts(mod, gates, overlapping) == {"encode/zero": "suspect (runs overlap)"}
    # An idle-subtracted count can go below zero: separated, that is an improvement.
    below = results({"encode/zero": [0.0, 0.0, 0.0]}, {"encode/zero": [-1.0, -1.0, -1.0]})
    assert verdicts(mod, gates, below) == {"encode/zero": "improvement"}


def test_a_negative_base_median_keeps_the_direction(mod, gates):
    # An idle-subtracted count can sit below zero on both sides: a rise from -2 to -1 is a
    # rise still (a regression when the runs are separated), a fall an improvement, and the
    # spread is never negative; the change is read against the base median's magnitude.
    rose = results({"encode/neg": [-2.0, -2.0, -2.0]}, {"encode/neg": [-1.0, -1.0, -1.0]})
    assert mod.compare(rose, gates)[0].change == pytest.approx(0.5)
    assert verdicts(mod, gates, rose) == {"encode/neg": "regression"}
    fell = results({"encode/neg": [-1.0, -1.0, -1.0]}, {"encode/neg": [-2.0, -2.0, -2.0]})
    assert verdicts(mod, gates, fell) == {"encode/neg": "improvement"}
    noisy = results({"encode/neg": [-2.2, -2.0, -1.8]}, {"encode/neg": [-2.0, -2.0, -2.0]})
    assert mod.compare(noisy, gates)[0].spread == pytest.approx(0.2)


def test_a_results_file_that_names_its_unit_prints_plain_values(mod, gates):
    res = results({"encode/calls": [300.0, 301.0]}, {"encode/calls": [290.0, 289.0]})
    res["unit"] = ""
    text = mod.markdown(res, mod.compare(res, gates))
    assert "| 300.5 (300 - 301) | 289.5 (289 - 290) |" in text
    res["unit"] = "KiB"
    text = mod.markdown(res, mod.compare(res, gates))
    assert "| 300.5 KiB (300 KiB - 301 KiB) |" in text
    timed = results({"encode/t": [1500.0, 1600.0]}, {"encode/t": [1550.0, 1650.0]})
    assert "1.55 µs (1.5 µs - 1.6 µs)" in mod.markdown(timed, mod.compare(timed, gates))


# ---------------------------------------------------------------- count

COUNT_TOML = GATE_TOML.replace("repeats = 3\n", "repeats = 3\ncount = true\nrun_budget = 3600\n")
# The pull request's four runs of `encode/a` against a base of 100 to 102, on a 10 % threshold:
# SLOW is +14.9 %, within twice the threshold; OVER is +29.7 %, beyond it.
SLOW = (115, 116, 117, 118)
OVER = (130, 131, 132, 133)

# Stands in for valgrind's callgrind: one profile after each batch the listed counts name, the
# process's own at its end and one of a forked child, laid out as the real tool lays them out.
FAKE_VALGRIND = r"""#!/usr/bin/env python3
import json, os, re, sys, time
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ["--tool=callgrind", "--dump-instr=yes"], args
assert args[2].startswith("--dump-after=*criterion::bencher::"), args
out = args[3].split("=", 1)[1]
binary, rest = Path(args[4]), args[5:]
assert rest[:2] == ["--bench", "--noplot"] and rest[-3:-1] == ["--profile-time", "1"], rest
assert os.environ["CRITERION_HOME"] == str(Path(out).parent / "home")
bench_id = re.sub(r"\\(.)", r"\1", rest[-1][1:-1])
listed = Path(os.environ["FAKE_COUNTS"])
with listed.with_suffix(".calls").open("a") as calls:
    calls.write(f"{binary.parent.name} {bench_id}\n")
side = json.loads(listed.read_text())[binary.parent.name]
time.sleep(side.get("sleep", 0))  # a count that hangs
if "exit" in side:
    sys.exit(side["exit"])
PROFILE = '''# callgrind format
version: 1
positions: instr line
events: Ir

fn=(1) decision::decide
0x1000 0 3
cfn=(2) <criterion::bencher::Bencher>::iter::<u64, decision::decide::{closure#0}>
calls=CALLS 0x2000 0
* 0 COUNT
fn=(2)
0x2000 0 9
cfn=(3) <engine::Engine>::decide
calls=ITERS ADDRESS 0
+4 0 INSIDE
'''
batches = side[bench_id]
if batches and isinstance(batches[0], list):  # one list a count of this id on this side
    seen = listed.with_suffix(".calls").read_text().splitlines()
    batches = batches[min(seen.count(f"{binary.parent.name} {bench_id}"), len(batches)) - 1]
base = out.replace("%p", str(os.getpid()))
for n, count in enumerate(batches, start=1):
    text = PROFILE.replace("COUNT", str(count)).replace("INSIDE", str(count - 1))
    text = text.replace("CALLS", str(side.get("calls", 1))).replace("ITERS", str(2 ** (n - 1)))
    Path(f"{base}.{n}").write_text(text.replace("ADDRESS", hex(side.get("address", 0x5000))))
if side.get("garbage"):
    Path(f"{base}.1").write_text("fn=(1) main\ncalls=one 0x10 0\n* 0 5\n")
Path(base).write_text("# callgrind format\nversion: 1\nevents: Ir\n\nfn=(1) main\n16 5\n")
child = PROFILE.replace("COUNT", "1").replace("INSIDE", "1").replace("CALLS", "1")
child = child.replace("ITERS", "1").replace("ADDRESS", "0x1")
Path(out.replace("%p", "1")).write_text(child)
if side.get("second"):  # a second process that ran a batch of its own
    Path(out.replace("%p", "1") + ".1").write_text(child)
print("fake valgrind", bench_id)
"""


def count_run(
    mod, tmp_path, monkeypatch, counts, gate_text=COUNT_TOML, name="out", slow=SLOW, more=None
):
    """One gate run whose `encode/a` fails by time (`slow`: the pull request's four runs), under
    a valgrind on PATH that writes the listed batch counts per side directory (base, head).
    `more`: further ids, each with its base's runs and its pull request's.
    Returns the exit code, the output directory and the file that lists the fake tool's calls."""
    gate_file = tmp_path / "gates.toml"
    gate_file.write_text(gate_text)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base_runs = {"encode/a": [100, 101, 102, 101], "decode/x": [1, 1, 1]}
    pr_runs = {"encode/a": list(slow), "decode/x": [1, 1, 1]}
    for bench_id, (base_side, pr_side) in (more or {}).items():
        base_runs[bench_id], pr_runs[bench_id] = list(base_side), list(pr_side)
    base = fake_bench(tmp_path / "base.py", base_runs)
    pr = fake_bench(tmp_path / "pr.py", pr_runs)
    tools = tmp_path / "tools"
    if not tools.exists():
        tools.mkdir()
        tool = tools / "valgrind"
        tool.write_text(FAKE_VALGRIND)
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")
    listed = tmp_path / "counts.json"
    listed.write_text(json.dumps(counts))
    monkeypatch.setenv("FAKE_COUNTS", str(listed))
    out = tmp_path / name
    rc = mod.main(
        ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr), "--out", str(out)]
    )
    return rc, out, listed.with_suffix(".calls")


def test_gate_file_names_the_count_switch(mod, tmp_path, gates):
    assert gates.count is False
    path = tmp_path / "count.toml"
    path.write_text(COUNT_TOML)
    assert mod.load_gates(path).count is True


def test_gate_file_names_the_run_budget(mod, tmp_path, gates):
    # A gate file that does not count needs none; one that counts names the run's budget.
    assert gates.run_budget is None
    path = tmp_path / "budget.toml"
    path.write_text(COUNT_TOML)
    assert mod.load_gates(path).run_budget == 3600.0
    path.write_text(COUNT_TOML.replace("run_budget = 3600\n", ""))
    with pytest.raises(ValueError, match="count = true needs run_budget"):
        mod.load_gates(path)
    # 1e400 reads as inf, and an integer of 401 digits is past a float's range.
    for value in ("0", "-5", '"long"', "true", "nan", "inf", "1e400", "1" + "0" * 400):
        path.write_text(COUNT_TOML.replace("run_budget = 3600\n", f"run_budget = {value}\n"))
        with pytest.raises(ValueError, match="run_budget must be"):
            mod.load_gates(path)


def test_the_repository_gate_files_load(mod):
    # Every gate file loads as the workflows read it: one that counts names its run's budget.
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / ".github" / "bench-gates").glob("*.toml"))
    counting = [path.name for path in files if mod.load_gates(path).count]
    assert "policy-engine.toml" in counting


def test_a_profile_is_read_for_the_harness_timed_calls(mod, tmp_path):
    profile = tmp_path / "profile.1.3"
    profile.write_text(
        "# callgrind format\nversion: 1\npositions: instr line\nevents: Ir\nsummary: 900\n\n"
        "fn=(1) decision::decide\n0x1000 0 3\n"
        "cfn=(2) <criterion::bencher::Bencher>::iter::<u64, decision::decide::{closure#0}>\n"
        "calls=1 0x2000 0\n+2 0 700\n"
        "fn=(2)\n0x2000 0 9\n+3 0 5\n"
        "cfn=(3) <std::time::Instant>::now\ncalls=2 0x3000 75\n* 0 54\n"
        "cfn=(4) <engine::Engine>::decide\ncalls=4 0x5240 0\n+8 0 600\n"
        "fn=(4)\n0x5240 0 600\n"
        "totals: 900\n"
    )
    batch = mod.read_profile(profile)
    assert (batch.count, batch.calls) == (700, 1)
    assert (batch.unit, batch.address) == ("<engine::Engine>::decide", 0x5240)
    # The harness's async timing functions are timed calls too.
    asynced = profile.read_text().replace("bencher::Bencher>", "bencher::AsyncBencher<Runtime>>")
    profile.write_text(asynced)
    batch = mod.read_profile(profile)
    assert (batch.count, batch.calls, batch.address) == (700, 1, 0x5240)
    # A callee with less than half of the timed calls' instructions is not the unit: the
    # routine's own code sits in the harness's function, and the unit is that function.
    profile.write_text(asynced.replace("+8 0 600\n", "+8 0 300\n"))
    batch = mod.read_profile(profile)
    assert (batch.unit, batch.address) == (
        "<criterion::bencher::AsyncBencher<Runtime>>::iter::<u64, decision::decide::{closure#0}>",
        0x2000,
    )
    # The older mangling, a routine inlined into the harness's function (the unit is that
    # function) and positions without addresses (no place is read). A timing function that
    # calls another is one timed call, counted where it is entered from outside.
    profile.write_text(
        "version: 1\npositions: line\nevents: Ir\n\n"
        "fn=(1) bench::run\n12 3\n"
        "cfn=(2) criterion::bencher::Bencher<M>::iter_with_large_drop\ncalls=1 40\n12 500\n"
        "fn=(2)\n40 2\n"
        "cfn=(3) criterion::bencher::Bencher<M>::iter_batched\ncalls=1 60\n41 498\n"
        "fn=(3)\n60 478\ncfn=(4) std::time::Instant::now\ncalls=2 75\n61 20\n"
    )
    batch = mod.read_profile(profile)
    assert (batch.count, batch.calls) == (500, 1)
    assert (batch.unit, batch.address) == ("criterion::bencher::Bencher<M>::iter_batched", None)


def test_every_batch_both_sides_ran_after_the_first_decides_a_placement(mod, gates):
    res = results({"encode/a": [100, 101, 102]}, {"encode/a": [115, 116, 117]})
    # The measured function starts elsewhere in the pull request's executable and +14.9 % is
    # within twice the threshold: the count alone decides these rows.
    places = ({"unit": "decide", "address": 0x10C0}, {"unit": "decide", "address": 0x1240})

    def verdict(base, pr):
        res["count"] = {"encode/a": {"base": places[0] | base, "pr": places[1] | pr}}
        return verdicts(mod, gates, res)["encode/a"]

    # The first batch is never read: a first iteration pays what the process has not done.
    assert verdict({"batches": [9, 20, 40]}, {"batches": [7, 20, 40]}) == "placement"
    assert verdict({"batches": [9, 20, 40]}, {"batches": [9, 20, 41]}) == "regression"
    # The shorter side names the batches read: the fourth ran on one side alone.
    assert verdict({"batches": [9, 20, 40, 80]}, {"batches": [9, 20, 40]}) == "placement"
    # A batch in the middle that differs is a count that moved.
    assert verdict({"batches": [9, 20, 40, 80]}, {"batches": [9, 20, 41, 80]}) == "regression"
    # The second batch is read, and alone too: the first is the only one left out.
    assert verdict({"batches": [9, 20, 40]}, {"batches": [9, 21, 40]}) == "regression"
    assert verdict({"batches": [9, 20]}, {"batches": [9, 21]}) == "regression"
    assert verdict({"batches": [9, 20]}, {"batches": [7, 20]}) == "placement"
    # A side whose own two counts differ is not steady: its count says nothing of the change.
    unsteady = {"batches": [9, 20, 40], "again": [9, 20, 44]}
    # (two counts that agree are never taken again: the second reading is of counts that differ)
    assert verdict(unsteady, {"batches": [9, 20, 40]}) == "placement"
    assert verdict(unsteady, {"batches": [9, 20, 41]}) == "regression"
    assert mod.compare(res, gates)[0].counted == mod.Counted("unsteady", 4, 40, 41, "base", 0.1)
    # Not read on a side, or one batch alone: the time alone decides.
    assert verdict({"batches": [9, 20]}, {"error": "valgrind is not on this host"}) == "regression"
    assert verdict({"batches": [9]}, {"batches": [9]}) == "regression"
    # A count never makes a row that passes by time anything else.
    res["pr"]["runs"] = [{"encode/a": value} for value in (100, 101, 102)]
    assert verdict({"batches": [9, 20, 40]}, {"batches": [9, 20, 41]}) == "ok"


def test_a_failing_id_on_the_same_count_is_a_placement(mod, tmp_path, monkeypatch):
    unit = "<engine::Engine>::decide"
    rc, out, calls = count_run(
        mod,
        tmp_path,
        monkeypatch,
        {
            "base": {"encode/a": [5414, 8268, 16270], "address": 0x3BE0C0},
            "head": {"encode/a": [5680, 8268, 16270, 32274], "address": 0x3BE240},
        },
    )
    assert rc == 0
    # the id that failed after its re-measurement is counted, once a side, and no other
    assert calls.read_text().splitlines() == ["base encode/a", "head encode/a"]
    saved = json.loads((out / "results.json").read_text())
    assert saved["count"] == {
        "encode/a": {
            "base": {"batches": [5414, 8268, 16270], "unit": unit, "address": 0x3BE0C0},
            "pr": {"batches": [5680, 8268, 16270, 32274], "unit": unit, "address": 0x3BE240},
        }
    }
    text = (out / "gate.md").read_text()
    assert "### Benchmark gate: PASS" in text
    assert "| +14.9% | 2.0% | 10.0% | placement |" in text
    assert "0 failed, 0 suspect" in text and " 1 placement (" in text
    assert "Counted before the verdict, on both executables under callgrind: `encode/a`." in text
    assert (
        "- `encode/a`: placement, +14.9%; base runs [100, 101, 102] ns, pull request runs "
        "[115, 116, 117] ns; re-measured base 101 ns, pull request 118 ns (+16.8%). The two "
        "executables run it in the same instruction count in every batch after the first, "
        "16,270 over the 4 iterations of the last (4,067.5 an iteration): the time is where "
        f"the code sits in the bench executable, not work the pull request adds. `{unit}` "
        "starts at 0x0c0 of its 4 KiB page in the base's executable and at 0x240 in the pull "
        "request's."
    ) in text
    # the profiles are read and gone, the child's too; the log stays in the artifact
    for side in ("base", "head"):
        assert sorted(p.name for p in (out / side / "count-1-1").iterdir()) == ["count.log"]
        log = (out / side / "count-1-1" / "count.log").read_text()
        assert log.startswith("+ ") and f"{out / side / 'bench'} --bench --noplot" in log
    # the saved results carry the count: re-evaluated, the verdict is the same
    gate_file = str(tmp_path / "gates.toml")
    assert mod.main(["compare", "--gates", gate_file, str(out / "results.json")]) == 0


def test_a_count_that_moved_keeps_the_failure(mod, tmp_path, monkeypatch):
    rc, out, calls = count_run(
        mod,
        tmp_path,
        monkeypatch,
        {"base": {"encode/a": [5414, 8268, 16270]}, "head": {"encode/a": [5414, 8468, 16670]}},
    )
    assert rc == 1
    # two counts that differ are each taken once more
    assert calls.read_text().splitlines() == ["base encode/a", "head encode/a"] * 2
    saved = json.loads((out / "results.json").read_text())["count"]["encode/a"]
    assert saved["base"]["again"] == [5414, 8268, 16270]
    assert saved["pr"]["again"] == [5414, 8468, 16670]
    text = (out / "gate.md").read_text()
    assert "FAIL regression" in text and "1 failed" in text and "1 placement" not in text
    assert (
        "The count moved too: 16,270 instructions on the base and 16,670 on the pull request "
        "over a batch of 4 iterations (+2.459%, 100.0 instructions an iteration more), the same "
        "in two counts of each."
    ) in text
    assert (out / "head" / "count-1-2" / "count.log").exists()


def test_a_count_that_is_not_steady_keeps_the_failure(mod, tmp_path, monkeypatch):
    # The base's own second count differs from its first: the id draws a seed, or allocates
    # by the heap's state, and no count of it tells the change from the id's own movement.
    rc, out, _ = count_run(
        mod,
        tmp_path,
        monkeypatch,
        {
            "base": {"encode/a": [[5414, 8268, 16270], [5414, 8268, 16275]]},
            "head": {"encode/a": [5414, 8268, 16274]},
        },
    )
    assert rc == 1
    text = (out / "gate.md").read_text()
    assert "FAIL regression" in text and "The count moved too" not in text
    assert (
        "The count differs, 16,270 instructions on the base and 16,274 on the pull request over "
        "a batch of 4 iterations (+0.025%, 1.0 instructions an iteration more), and it is not "
        "steady on one executable alone: the base's own two counts differ by 0.031%. The time "
        "alone decides."
    ) in text


def test_a_count_that_was_not_read_keeps_the_failure(mod, tmp_path, monkeypatch):
    def run(counts, name):
        rc, out, _ = count_run(mod, tmp_path, monkeypatch, counts, name=name)
        text = (out / "gate.md").read_text()
        assert rc == 1 and "FAIL regression" in text
        return text

    # One batch alone: a first iteration pays what the process has not done yet.
    text = run({"base": {"encode/a": [5414]}, "head": {"encode/a": [5414]}}, "one")
    assert (
        "The count was not read (the first batch alone ran on both sides): the time alone decides."
    ) in text
    # The tool fails on one side.
    text = run({"base": {"encode/a": [1, 2]}, "head": {"exit": 3}}, "exit")
    assert "The count was not read (pull request: the count exited with 3; see " in text
    # A profile that holds two timed calls is not one batch.
    text = run({"base": {"encode/a": [1, 2], "calls": 2}, "head": {"encode/a": [1, 2]}}, "two")
    assert "The count was not read (base: a profile of the count does not hold one batch" in text
    # No profile of the harness at all: its timing function is not in the executable's names.
    text = run({"base": {"encode/a": []}, "head": {"encode/a": [1, 2]}}, "none")
    assert "The count was not read (base: the count holds no batch of the harness" in text
    # A profile that does not parse.
    text = run({"base": {"encode/a": [1, 2], "garbage": True}, "head": {"encode/a": [1, 2]}}, "bad")
    assert "The count was not read (base: a profile of the count does not parse (" in text
    # A second process that ran a batch of its own: no one count of the id.
    second = {"base": {"encode/a": [1, 2], "second": True}, "head": {"encode/a": [1, 2]}}
    text = run(second, "second")
    assert "The count was not read (base: two processes of the count ran batches; see " in text
    # No valgrind on the host.
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    text = run({"base": {}, "head": {}}, "absent")
    assert (
        "The count was not read (base: valgrind is not on this host; pull request: valgrind "
        "is not on this host): the time alone decides."
    ) in text


def test_the_count_runs_only_when_the_gate_file_asks(mod, tmp_path, monkeypatch):
    counts = {"base": {"encode/a": [1, 2]}, "head": {"encode/a": [1, 2]}}
    rc, out, calls = count_run(mod, tmp_path, monkeypatch, counts, gate_text=GATE_TOML)
    assert rc == 1 and not calls.exists()
    assert "count" not in json.loads((out / "results.json").read_text())
    assert "Counted before the verdict" not in (out / "gate.md").read_text()


def test_a_placement_needs_a_moved_function_within_twice_the_threshold(mod, gates):
    def counted(pr_runs, base_at, pr_at, unit="decide"):
        res = results({"encode/a": [100, 101, 102]}, {"encode/a": pr_runs})
        res["count"] = {
            "encode/a": {
                "base": {"batches": [9, 20, 40], "unit": "decide", "address": base_at},
                "pr": {"batches": [7, 20, 40], "unit": unit, "address": pr_at},
            }
        }
        return res

    def read(*args, **kwargs):
        row = mod.compare(counted(*args, **kwargs), gates)[0]
        return row.verdict, mod.fmt_count(row)

    same = (
        " The two executables run it in the same instruction count in every batch after the "
        "first, 40 over the 4 iterations of the last (10.0 an iteration)"
    )
    # The three facts: the same count, the function elsewhere, +14.9 % on a 10 % threshold.
    assert read([115, 116, 117], 0x3BE0C0, 0x3BE240) == (
        "placement",
        f"{same}: the time is where the code sits in the bench executable, not work the pull "
        "request adds. `decide` starts at 0x0c0 of its 4 KiB page in the base's executable and "
        "at 0x240 in the pull request's.",
    )
    # A whole page further on is another address.
    assert read([115, 116, 117], 0x3BE0C0, 0x3BF0C0) == (
        "placement",
        f"{same}: the time is where the code sits in the bench executable, not work the pull "
        "request adds. `decide` starts at 0x3be0c0 in the base's executable and at 0x3bf0c0 in "
        "the pull request's, the same place in its 4 KiB page.",
    )
    # The same count where nothing moved: where the code sits does not explain the time.
    assert read([115, 116, 117], 0x3BE0C0, 0x3BE0C0) == (
        "regression",
        f"{same}, and `decide` starts at 0x3be0c0 in both: where the code sits does not explain "
        "the time, and the failure stands.",
    )
    # No place read on a side, or another function on it: nothing says the code moved.
    nowhere = (
        f"{same}, but the count names no one function with a place on both sides: the failure "
        "stands."
    )
    assert read([115, 116, 117], 0x3BE0C0, None) == ("regression", nowhere)
    assert read([115, 116, 117], 0x3BE0C0, 0x3BE240, unit="other") == ("regression", nowhere)
    # Twice the threshold is the most a placement excuses: +19.8 % passes, +20.8 % fails.
    assert read([120, 121, 122], 0x3BE0C0, 0x3BE240)[0] == "placement"
    assert read([121, 122, 123], 0x3BE0C0, 0x3BE240) == (
        "regression",
        f"{same}. `decide` starts at 0x0c0 of its 4 KiB page in the base's executable and at "
        "0x240 in the pull request's. A placement excuses a change up to 2 times the threshold, "
        "20.0%: the failure stands.",
    )
    # A placement's line stands in the summary without a re-measurement too.
    res = counted([115, 116, 117], 0x3BE0C0, 0x3BE240)
    assert (
        "- `encode/a`: placement, +14.9%; base runs [100, 101, 102] ns, pull request runs "
        f"[115, 116, 117] ns.{same}: the time is where the code sits"
    ) in mod.markdown(res, mod.compare(res, gates))


def test_the_same_count_keeps_the_failure_unmoved_or_beyond_twice_the_threshold(
    mod, tmp_path, monkeypatch
):
    unit = "<engine::Engine>::decide"
    same = (
        "The two executables run it in the same instruction count in every batch after the "
        "first, 16,270 over the 4 iterations of the last (4,067.5 an iteration)"
    )
    counts = {
        "base": {"encode/a": [5414, 8268, 16270], "address": 0x3BE0C0},
        "head": {"encode/a": [5680, 8268, 16270, 32274], "address": 0x3BE240},
    }
    # +29.7 % on a 10 % threshold: the function moved, and a placement does not excuse that much.
    # A run counts no such id; results saved with its count still read so.
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, counts, name="over")
    assert rc == 0
    saved = json.loads((out / "results.json").read_text())
    for run, value in zip(saved["pr"]["runs"], OVER):
        run["encode/a"] = value
    (out / "results.json").write_text(json.dumps(saved))
    summary, gate_file = tmp_path / "over.md", str(tmp_path / "gates.toml")
    args = ["compare", "--gates", gate_file, "--summary", str(summary), str(out / "results.json")]
    assert mod.main(args) == 1
    text = summary.read_text()
    assert "### Benchmark gate: FAIL" in text and "1 failed" in text
    assert "FAIL regression" in text and " 1 placement (" not in text
    assert (
        "- `encode/a`: +29.7% against a 10.0% threshold; base runs [100, 101, 102] ns, pull "
        "request runs [130, 131, 132] ns; re-measured base 101 ns, pull request 133 ns (+31.7%). "
        f"{same}. `{unit}` starts at 0x0c0 of its 4 KiB page in the base's executable and at "
        "0x240 in the pull request's. A placement excuses a change up to 2 times the threshold, "
        "20.0%: the failure stands."
    ) in text
    # +14.9 % on the same count, with the function where it was.
    counts["head"]["address"] = 0x3BE0C0
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, counts, name="unmoved")
    assert rc == 1
    text = (out / "gate.md").read_text()
    assert "FAIL regression" in text and " 1 placement (" not in text
    assert (
        f"{same}, and `{unit}` starts at 0x3be0c0 in both: where the code sits does not explain "
        "the time, and the second measurement decides. Measured again from a fresh copy of each "
        "executable: base runs [101, 101, 101] ns, pull request runs [118, 118, 118] ns "
        "(+16.8%): the failure repeats."
    ) in text


def test_an_id_beyond_twice_its_threshold_is_counted_for_a_second_measurement(
    mod, tmp_path, monkeypatch
):
    # +29.7 % on a 10 % threshold: no count makes it a placement, and a count that reads level
    # still asks for the second measurement, which repeats here.
    counts = {"base": {"encode/a": [1, 2]}, "head": {"encode/a": [1, 2]}}
    rc, out, calls = count_run(mod, tmp_path, monkeypatch, counts, slow=OVER)
    assert rc == 1 and calls.read_text().splitlines() == ["base encode/a", "head encode/a"]
    saved = json.loads((out / "results.json").read_text())
    assert saved["count"]["encode/a"]["base"]["batches"] == [1, 2]
    assert saved["second"]["ids"] == ["encode/a"]
    text = (out / "gate.md").read_text()
    assert "### Benchmark gate: FAIL" in text and "FAIL regression" in text
    assert "Counted before the verdict" in text and "Not counted" not in text
    assert "pull request runs [133, 133, 133] ns (+31.7%): the failure repeats." in text
    # Results saved when the count passed such an id carry the reason, and read as they did.
    saved["count"], saved["second"] = {"encode/a": {"skipped": "beyond"}}, None
    (out / "results.json").write_text(json.dumps(saved))
    summary, gate_file = tmp_path / "old.md", str(tmp_path / "gates.toml")
    args = ["compare", "--gates", gate_file, "--summary", str(summary), str(out / "results.json")]
    assert mod.main(args) == 1
    said = "a change beyond 2 times the threshold, which no placement excuses."
    assert f"Not counted, `encode/a`: {said}" in summary.read_text()


def test_an_id_whose_base_median_is_zero_is_not_counted_and_says_so(mod, tmp_path, monkeypatch):
    # No change can be told against a base of zero, so no placement excuses it: its own reason.
    counts = {
        side: {"encode/a": [5414, 8268, 16270], "address": address}
        for side, address in (("base", 0x3BE0C0), ("head", 0x3BE240))
    }
    more = {"encode/z": ((0, 0, 0, 0), (5, 5, 5, 5))}
    rc, out, calls = count_run(mod, tmp_path, monkeypatch, counts, more=more)
    assert rc == 1 and "encode/z" not in calls.read_text()
    saved = json.loads((out / "results.json").read_text())
    # the saved results hold both kinds: the id not counted beside the counted one
    assert saved["count"]["encode/z"] == {"skipped": "no_base"}
    assert "batches" in saved["count"]["encode/a"]["base"]
    said = "a base median of zero, which leaves no change for a placement to excuse."
    assert f"Not counted, `encode/z`: {said}" in (out / "gate.md").read_text()


def test_a_change_of_exactly_twice_the_threshold_is_within_the_ceiling(mod, tmp_path, monkeypatch):
    # 100 to 120 on a 10 % threshold is twice the threshold to the bit: a placement can still
    # excuse it, so it is counted.
    counts = {side: {"encode/a": [1, 2], "encode/e": [1, 2]} for side in ("base", "head")}
    more = {"encode/e": ((100, 100, 100, 100), (120, 120, 120, 120))}
    rc, out, calls = count_run(mod, tmp_path, monkeypatch, counts, more=more)
    assert rc == 1 and "base encode/e" in calls.read_text().splitlines()
    saved = json.loads((out / "results.json").read_text())
    assert "batches" in saved["count"]["encode/e"]["base"]


def test_a_run_past_its_budget_counts_nothing_and_writes_the_report(mod, tmp_path, monkeypatch):
    # Two ids fail by time within twice the threshold. The budget runs from the run's start,
    # and here the timed rounds take it whole (a clock that moves 10 s at each read): no count
    # starts, both ids keep the time's verdict, and the report is written.
    ticks = iter(range(0, 10**9, 10))
    clock = types.SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda seconds: None)
    monkeypatch.setattr(mod, "time", clock)
    counts = {side: {"encode/a": [1, 2], "encode/b": [1, 2]} for side in ("base", "head")}
    spent = COUNT_TOML.replace("run_budget = 3600\n", "run_budget = 100\n")
    more = {"encode/b": ((100, 101, 102, 101), SLOW)}
    rc, out, calls = count_run(mod, tmp_path, monkeypatch, counts, gate_text=spent, more=more)
    assert rc == 1 and not calls.exists()
    saved = json.loads((out / "results.json").read_text())
    assert saved["count"] == {
        bench_id: {"skipped": "budget"} for bench_id in ("encode/a", "encode/b")
    }
    text = (out / "gate.md").read_text()
    assert "### Benchmark gate: FAIL" in text and "2 failed" in text
    assert "Counted before the verdict" not in text
    said = "the run's budget left no room for its count, and the time alone decides."
    assert f"Not counted, `encode/a`, `encode/b`: {said}" in text
    assert f"pull request 118 ns (+16.8%). Not counted: {said}" in text


def budget_count(mod, tmp_path, monkeypatch, reads, seconds, budget):
    """`count_ids` over `encode/a` and `encode/b` under a clock that only the stand-in count
    moves: a count takes `seconds` and returns the next of `reads` (a text there is the error
    it fails with), or, where its bound is shorter, takes the bound and fails as a count the
    bound ended. The run's budget ends `budget` seconds after the first count starts. Returns
    the counts and each count's bound."""
    clock, bounds = [1000.0], []

    def count_once(binary, work, cwd, gates, extra_args, bench_id, cpu, bound):
        bounds.append(bound)
        if seconds > bound:
            clock[0] += bound
            raise RuntimeError(f"the count ran over {bound:g} s")
        clock[0] += seconds
        read = reads.pop(0)
        if isinstance(read, str):
            raise RuntimeError(read)
        return {"batches": list(read), "unit": "decide", "address": "0x5000"}

    monkeypatch.setattr(mod, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(mod, "count_once", count_once)
    gates = mod.Gates(count=True, run_budget=budget)
    binaries = {"base": tmp_path / "base", "pr": tmp_path / "pr"}
    ids = ["encode/a", "encode/b"]
    counts = mod.count_ids(ids, binaries, tmp_path, tmp_path, gates, [], None, clock[0] + budget)
    return counts, bounds


def test_the_run_budget_is_read_before_each_count_and_caps_it(mod, tmp_path, monkeypatch):
    # 100 s are left and a count takes 40: the first id's two counts fit, each capped by what
    # was left at its start, and 20 s are under the floor: the second id's turn never starts.
    reads = [[1, 2], [1, 2]]
    counts, bounds = budget_count(mod, tmp_path, monkeypatch, reads, seconds=40, budget=100)
    assert bounds == [100, 60] and mod.COUNT_FLOOR_S == 30
    assert counts["encode/a"]["base"]["batches"] == [1, 2] == counts["encode/a"]["pr"]["batches"]
    assert counts["encode/b"] == {"skipped": "budget"}


def test_an_id_whose_turn_the_run_budget_cuts_is_not_counted(mod, tmp_path, monkeypatch):
    # The first id's two counts differ, so each side is owed one more: the budget has room for
    # the third count and none for the fourth. A turn cut short tells nothing: not counted.
    reads = [[1, 2, 3], [1, 2, 4], [1, 2, 3]]
    counts, bounds = budget_count(mod, tmp_path, monkeypatch, reads, seconds=40, budget=130)
    assert bounds == [120, 90, 50]
    assert counts == {bench_id: {"skipped": "budget"} for bench_id in ("encode/a", "encode/b")}


def test_a_count_the_run_budget_ends_is_not_a_count_that_was_not_read(mod, tmp_path, monkeypatch):
    # The second count starts with 30 s left and would take 70: the budget's cap ends it, which
    # is the budget's reason and not an error of the count.
    counts, bounds = budget_count(mod, tmp_path, monkeypatch, [[1, 2]], seconds=70, budget=100)
    assert bounds == [100, 30]
    assert counts == {bench_id: {"skipped": "budget"} for bench_id in ("encode/a", "encode/b")}


def test_a_count_that_fails_by_itself_under_the_cap_is_a_count_not_read(mod, tmp_path, monkeypatch):
    # The second count runs under the budget's cap of 60 s and fails by itself after 40: the
    # count's own error, on its side. The budget cut nothing of this id; it leaves the next.
    reads = [[1, 2], "the count exited with 3"]
    counts, bounds = budget_count(mod, tmp_path, monkeypatch, reads, seconds=40, budget=100)
    assert bounds == [100, 60]
    assert counts["encode/a"]["pr"] == {"error": "the count exited with 3"}
    assert counts["encode/b"] == {"skipped": "budget"}


def test_a_counting_gate_ends_its_count_before_its_gate_step_does(mod):
    # The run's budget and the gate step's bound stand in two files: every workflow's gate step
    # outlasts the budget of each counting gate file it names, by a minute for the report.
    root = Path(__file__).resolve().parents[2]
    step = re.compile(r"- name: Run the benchmark gate\n(?:.*\n)*?\s+timeout-minutes: (\d+)\n")
    checked = []
    for workflow in sorted((root / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text()
        found = step.search(text)
        if found is None:
            continue
        # A comment may name another workflow's gate file, as the store workflows name the
        # decision bench's for its placement flags. The files a workflow runs stand outside
        # its comment lines.
        runs = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        for name in sorted(set(re.findall(r"[\w-]+\.toml", runs))):
            path = root / ".github" / "bench-gates" / name
            if path.exists() and mod.load_gates(path).count:
                budget = mod.load_gates(path).run_budget
                assert budget <= int(found.group(1)) * 60 - 60, f"{workflow.name}: {name}"
                checked.append(name)
    # Every gate file that counts is held: one whose workflow's step went by another name
    # would leave this check in silence.
    counting = sorted(
        path.name
        for path in (root / ".github" / "bench-gates").glob("*.toml")
        if mod.load_gates(path).count
    )
    assert sorted(set(checked)) == counting
    assert "policy-engine.toml" in checked


def test_the_ids_are_counted_in_name_order(mod, tmp_path, monkeypatch):
    # The run's budget cuts the id whose turn it ends and the ids behind it: behind by name.
    same = [5414, 8268, 16270]
    ids = ("encode/a", "encode/b", "encode/0")
    counts = {side: dict.fromkeys(ids, same) for side in ("base", "head")}
    more = {bench_id: ([100, 101, 102, 101], SLOW) for bench_id in ("encode/b", "encode/0")}
    count_run(mod, tmp_path, monkeypatch, counts, more=more)
    calls = (tmp_path / "counts.calls").read_text().splitlines()
    assert calls == [f"{side} {bench_id}" for bench_id in sorted(ids) for side in ("base", "head")]


def test_the_timed_rounds_are_saved_before_the_count(mod, tmp_path, monkeypatch):
    saved = {}

    def cancelled(ids, binaries, out, *rest):
        # What a job cancelled in the count leaves: the file as it stands when the count starts.
        saved.update(json.loads((out / "results.json").read_text()))
        raise KeyboardInterrupt

    monkeypatch.setattr(mod, "count_ids", cancelled)
    with pytest.raises(KeyboardInterrupt):
        count_run(mod, tmp_path, monkeypatch, {"base": {}, "head": {}})
    assert [run["encode/a"] for run in saved["pr"]["runs"]] == [115, 116, 117, 118]
    assert saved["recheck"] == {"ids": ["encode/a"], "rounds": 1} and "count" not in saved
    assert json.loads((tmp_path / "out" / "results.json").read_text()) == saved


def test_a_count_that_hangs_is_ended_by_its_bound(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "COUNT_TIMEOUT_S", 0.5)
    counts = {"base": {"encode/a": [1, 2], "sleep": 30}, "head": {"encode/a": [1, 2]}}
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, counts)
    text = (out / "gate.md").read_text()
    assert rc == 1 and "FAIL regression" in text
    assert "The count was not read (base: the count ran over 0.5 s; see " in text


# ---------------------------------------------------------------- a whole pass

# The counting gate file with its `encode/` ids marked as ones that time a whole pass.
WHOLE_TOML = COUNT_TOML.replace(
    'pattern = "^encode/"\nthreshold = 0.10\n',
    'pattern = "^encode/"\nthreshold = 0.10\nwhole_pass = true\n',
)
LEVEL = "the 0.5% that reads as level for an id that times a whole pass"
NOT_WORK = (
    "The time is where the pass's code sits in the bench executable, not work the pull request "
    "adds."
)


def gates_of(mod, tmp_path, text):
    path = tmp_path / "marked.toml"
    path.write_text(text)
    return mod.load_gates(path)


def doubling(total, last):
    """The counts of batches of 1, 2, 4, ... iterations whose batch `last` counts `total`."""
    return [total * 2**n // 2**last for n in range(last + 1)]


def counted_twice(base, pr, last=2):
    """A saved count of two executables that differ, each counted the same a second time."""
    return {
        "base": {"batches": doubling(base, last), "again": doubling(base, last)},
        "pr": {"batches": doubling(pr, last), "again": doubling(pr, last)},
    }


def test_a_gate_entry_marks_an_id_that_times_a_whole_pass(mod, tmp_path):
    gates = gates_of(mod, tmp_path, WHOLE_TOML)
    assert gates.threshold_for("encode/a").whole_pass is True
    assert gates.threshold_for("decode/slow").whole_pass is False
    assert WHOLE_TOML != COUNT_TOML
    with pytest.raises(ValueError, match="whole_pass must be true or false, got 1"):
        gates_of(mod, tmp_path, WHOLE_TOML.replace("whole_pass = true", "whole_pass = 1"))
    # The mark is read at the count: a gate file that does not count refuses it.
    with pytest.raises(ValueError, match="whole_pass = true needs count = true"):
        gates_of(mod, tmp_path, WHOLE_TOML.replace("count = true\nrun_budget = 3600\n", ""))


def test_two_decide_ids_slower_on_a_level_count_are_placements(mod, tmp_path):
    # The figures of two decide ids a pull request read slower in the gate's build and faster
    # in a build of the same two sources without the alignment (2026-11-16): 6 and 2
    # instructions more an iteration, on counts of 21,676 and 4,631.
    res = results(
        {"encode/64": [1382.67, 1318.22, 1402.19], "encode/8": [301.34, 331.485, 327.946]},
        {"encode/64": [1556.38, 1498.28, 1547.46], "encode/8": [381.983, 353.995, 377.878]},
    )
    res["count"] = {
        "encode/64": counted_twice(177_570_058, 177_619_210, last=13),
        "encode/8": counted_twice(151_748_874, 151_814_410, last=15),
    }
    marked = mod.compare(res, gates_of(mod, tmp_path, WHOLE_TOML))
    assert [(row.bench_id, row.verdict) for row in marked] == [
        ("encode/64", "placement"),
        ("encode/8", "placement"),
    ]
    assert mod.fmt_count(marked[0]) == (
        " The count is level: 177,570,058 instructions on the base and 177,619,210 on the pull "
        "request over a batch of 8,192 iterations (+0.028%, 6.0 instructions an iteration more), "
        f"the same in two counts of each, within {LEVEL}. {NOT_WORK}"
    )
    assert "(+0.043%, 2.0 instructions an iteration more)" in mod.fmt_count(marked[1])
    text = mod.markdown(res, marked)
    assert "0 failed" in text and "FAIL" not in text
    assert (
        "2 placement (beyond the threshold within twice of it: on the same instruction count "
        "with the measured function moved, or for an id that times a whole pass on a level "
        "count; not failed)."
    ) in text
    assert (
        "An id that times a whole pass (`encode/64`, `encode/8`) is a placement on a level "
        "count instead: its two counts within 0.5% of each other, each executable's own two "
        "the same, and the change within twice its threshold."
    ) in text
    # The mark is the gate file's: the same results under an entry without it fail as before.
    plain = mod.compare(res, gates_of(mod, tmp_path, COUNT_TOML))
    assert [row.verdict for row in plain] == ["regression", "regression"]
    assert mod.fmt_count(plain[0]).startswith(" The count moved too: 177,570,058 instructions")
    assert "whole pass" not in mod.markdown(res, plain)


def whole_row(mod, tmp_path, count, pr_runs=(115, 116, 117)):
    """The `encode/a` row of saved results under the marked gate file, with its count."""
    res = results({"encode/a": [100, 101, 102]}, {"encode/a": list(pr_runs)})
    res["count"] = {"encode/a": count}
    row = mod.compare(res, gates_of(mod, tmp_path, WHOLE_TOML))[0]
    return row.verdict, mod.fmt_count(row)


def test_a_whole_pass_count_that_moved_beyond_the_level_keeps_the_failure(mod, tmp_path):
    verdict, said = whole_row(mod, tmp_path, counted_twice(400_000, 412_000))
    assert verdict == "regression"
    assert said == (
        " The count moved too: 400,000 instructions on the base and 412,000 on the pull request "
        "over a batch of 4 iterations (+3.000%, 3,000.0 instructions an iteration more), the "
        f"same in two counts of each, beyond {LEVEL}."
    )


def test_the_level_ends_at_half_a_percent_of_the_base_count(mod, tmp_path):
    assert mod.LEVEL_COUNT == 0.005
    for pr, verdict in [
        (402_000, "placement"),
        (402_004, "regression"),
        (398_000, "placement"),
        (397_996, "regression"),
    ]:
        assert whole_row(mod, tmp_path, counted_twice(400_000, pr))[0] == verdict, pr


def test_a_level_count_is_two_counts_of_each_executable_that_agree(mod, tmp_path):
    once = counted_twice(400_000, 400_024)
    del once["pr"]["again"]
    verdict, said = whole_row(mod, tmp_path, once)
    assert verdict == "regression"
    assert said.endswith(
        "(+0.006%, 6.0 instructions an iteration more). A level count is two counts of each "
        "executable that agree, and a second one was not read: the failure stands."
    )
    unsteady = counted_twice(400_000, 400_024)
    unsteady["pr"]["again"] = doubling(400_040, 2)
    verdict, said = whole_row(mod, tmp_path, unsteady)
    assert verdict == "regression"
    assert "it is not steady on one executable alone" in said and "level" not in said


def test_two_counts_that_differ_are_decided_at_a_batch_all_four_counts_ran(mod, tmp_path):
    # A second count that ended one batch early: the row is decided one batch earlier, where
    # every count ran, and its line names that batch.
    short = counted_twice(400_000, 400_024)
    short["pr"]["again"] = doubling(400_024, 2)[:2]
    verdict, said = whole_row(mod, tmp_path, short)
    assert verdict == "placement"
    assert (
        "200,000 instructions on the base and 200,012 on the pull request over a batch of 2 "
        "iterations (+0.006%, 6.0 instructions an iteration more), the same in two counts of each"
    ) in said
    # The base's second count one batch short bounds the row the same way.
    mirror = counted_twice(400_000, 400_024)
    mirror["base"]["again"] = doubling(400_000, 2)[:2]
    verdict, said = whole_row(mod, tmp_path, mirror)
    assert verdict == "placement" and "over a batch of 2 iterations" in said
    # The batch that decides is the last that differs, not the last every count ran: four
    # counts 1 % apart at the second batch and the same at the third keep the failure.
    early = counted_twice(400_000, 400_000)
    for key in ("batches", "again"):
        early["pr"][key][1] = 202_000
    verdict, said = whole_row(mod, tmp_path, early)
    assert verdict == "regression" and "over a batch of 2 iterations (+1.000%" in said
    # A second count of its first batch alone leaves no batch all four ran: the time decides.
    alone = counted_twice(400_000, 400_024)
    for side in ("base", "pr"):
        alone[side]["again"] = alone[side]["again"][:1]
    verdict, said = whole_row(mod, tmp_path, alone)
    assert verdict == "regression"
    assert said == (
        " The count was not read (a second count ended before the batches that differ): the "
        "time alone decides."
    )
    # First counts of two batches say the same: one ran past the first, and no second count.
    two = counted_twice(400_000, 400_024, last=1)
    for side in ("base", "pr"):
        two[side]["again"] = two[side]["again"][:1]
    assert whole_row(mod, tmp_path, two) == ("regression", said)


def test_a_batch_counted_once_grants_no_level_count_and_refuses_one(mod, tmp_path):
    def read_once(pr_last, short="pr"):
        """Four counts 12 instructions apart at the last batch all of them ran, a level count,
        and the pull request's `pr_last` at the batch the `short` side's second count left."""
        count = counted_twice(400_000, 400_024)
        for key in ("batches", "again"):
            count["pr"][key][2] = pr_last
        count[short]["again"].pop()
        return whole_row(mod, tmp_path, count)

    # Within the level, the batch all four ran decides as it did.
    for pr_last in (400_024, 402_000, 398_000):
        verdict, said = read_once(pr_last)
        assert verdict == "placement" and "over a batch of 2 iterations (+0.006%" in said, pr_last
    # Beyond it the placement is refused, whichever second count ended early.
    for pr_last in (402_004, 397_996):
        assert read_once(pr_last)[0] == "regression", pr_last
        assert read_once(pr_last, short="base")[0] == "regression", pr_last
    verdict, said = read_once(404_000)
    assert verdict == "regression"
    assert said == (
        " The count is level where all four counts ran: 200,000 instructions on the base and "
        "200,012 on the pull request over a batch of 2 iterations (+0.006%, 6.0 instructions an "
        f"iteration more), the same in two counts of each, within {LEVEL}. Past it the first "
        "counts read 400,000 and 404,000 over a batch of 4 iterations (+1.000%), beyond the "
        "level, and a second count did not run that batch: the failure stands."
    )
    # Work that grows with the iterations, +0.2, +0.4 and +0.8 % of a batch: the last batch
    # fails it with the second counts whole, and refuses the placement with one a batch short.
    grown = counted_twice(800_000, 800_000, last=3)
    for key in ("batches", "again"):
        grown["pr"][key] = [100_100, 200_400, 401_600, 806_400]
    assert whole_row(mod, tmp_path, grown)[0] == "regression"
    grown["pr"]["again"] = grown["pr"]["again"][:3]
    verdict, said = whole_row(mod, tmp_path, grown)
    assert verdict == "regression" and "over a batch of 8 iterations (+0.800%)" in said
    # The batch read is the last of the first counts that differs, so a row reads the same
    # whether a second count ran it or not: 1 % apart at the third batch, and beyond the level,
    # within it or the same at the fourth.
    for fourth, verdict in [
        (816_000, "regression"),
        (800_800, "placement"),
        (800_000, "regression"),
    ]:
        grown["pr"]["batches"][2:] = [404_000, fourth]
        grown["pr"]["again"] = [100_100, 200_400]
        short = whole_row(mod, tmp_path, grown)
        grown["pr"]["again"] = list(grown["pr"]["batches"])
        assert short[0] == whole_row(mod, tmp_path, grown)[0] == verdict, fourth
    # A batch one first count alone ran has no pair, and nothing is read there.
    longer = counted_twice(400_000, 400_024)
    longer["base"]["batches"].append(900_000)
    assert whole_row(mod, tmp_path, longer)[0] == "placement"


def test_a_whole_pass_id_on_the_same_count_asks_no_moved_function(mod, tmp_path):
    same = {"base": {"batches": [9, 20, 40]}, "pr": {"batches": [7, 20, 40]}}
    verdict, said = whole_row(mod, tmp_path, same)
    assert verdict == "placement"
    assert said == (
        " The two executables run it in the same instruction count in every batch after the "
        "first, 40 over the 4 iterations of the last (10.0 an iteration): the id times a whole "
        "pass, and the time is where the pass's code sits in the bench executable, not work "
        "the pull request adds."
    )
    res = results({"encode/a": [100, 101, 102]}, {"encode/a": [115, 116, 117]})
    res["count"] = {"encode/a": same}
    plain = mod.compare(res, gates_of(mod, tmp_path, COUNT_TOML))[0]
    assert plain.verdict == "regression"
    assert "names no one function with a place on both sides" in mod.fmt_count(plain)


def test_a_level_count_does_not_lift_the_ceiling(mod, tmp_path, monkeypatch):
    # Saved results can hold a count for an id beyond twice its threshold: the failure stands.
    verdict, said = whole_row(mod, tmp_path, counted_twice(400_000, 400_024), pr_runs=OVER)
    assert verdict == "regression"
    assert said.endswith(
        f"the same in two counts of each, within {LEVEL}. A placement excuses a change up to "
        "2 times the threshold, 20.0%: the failure stands."
    )
    same = {"base": {"batches": [9, 20, 40]}, "pr": {"batches": [7, 20, 40]}}
    verdict, said = whole_row(mod, tmp_path, same, pr_runs=OVER)
    assert verdict == "regression"
    assert said.endswith(
        "(10.0 an iteration). A placement excuses a change up to 2 times the "
        "threshold, 20.0%: the failure stands."
    )
    # A run counts such an id, marked or not, and measures it again on its level count: the
    # ceiling stands for the placement, and the second measurement decides.
    counts = {"base": {"encode/a": [9, 20, 40]}, "head": {"encode/a": [7, 20, 40]}}
    rc, out, _ = count_run(
        mod, tmp_path, monkeypatch, counts, gate_text=WHOLE_TOML, name="over", slow=OVER
    )
    assert rc == 1
    assert (
        "20.0%: the second measurement decides. Measured again from a fresh copy of each "
        "executable: base runs [101, 101, 101] ns, pull request runs [133, 133, 133] ns "
        "(+31.7%): the failure repeats."
    ) in (out / "gate.md").read_text()


def test_a_run_places_a_marked_id_on_a_level_count_and_fails_one_that_moved(
    mod, tmp_path, monkeypatch
):
    level = {"base": {"encode/a": [5414, 8268, 16270]}, "head": {"encode/a": [5414, 8270, 16274]}}
    rc, out, calls = count_run(mod, tmp_path, monkeypatch, level, gate_text=WHOLE_TOML)
    assert rc == 0
    # two counts that differ are each taken once more, and the level is read on both
    assert calls.read_text().splitlines() == ["base encode/a", "head encode/a"] * 2
    text = (out / "gate.md").read_text()
    assert "1 placement" in text and "0 failed" in text
    assert (
        "The count is level: 16,270 instructions on the base and 16,274 on the pull request "
        "over a batch of 4 iterations (+0.025%, 1.0 instructions an iteration more), the same "
        f"in two counts of each, within {LEVEL}. {NOT_WORK}"
    ) in text
    moved = {"base": {"encode/a": [5414, 8268, 16270]}, "head": {"encode/a": [5414, 8468, 16670]}}
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, moved, gate_text=WHOLE_TOML, name="moved")
    assert rc == 1
    text = (out / "gate.md").read_text()
    assert "FAIL regression" in text and "1 failed" in text and "1 placement" not in text
    assert f"the same in two counts of each, beyond {LEVEL}." in text


def test_the_repository_gate_files_mark_the_rows_that_time_a_whole_decision(mod):
    root = Path(__file__).resolve().parents[2] / ".github" / "bench-gates"
    marked = {
        "policy-engine.toml": {
            "policy_engine/decide/cache_aware/8": True,
            "policy_engine/decide/least_load_credited/8": True,
            "policy_engine/decide/cache_aware/512": True,
            "policy_engine/decide/passthrough/8": True,
            "policy_engine/decide/manual_keyed_miss/8": True,
            "policy_engine/decide/manual_keyed_hit/8": False,
            "policy_engine/decide/passthrough/64": False,
            "policy_engine/decide/power_of_two/512": False,
            "policy_engine/decide/random/64": False,
            "policy_engine/decide/round_robin/512": False,
            "policy_engine/decide/passthrough/512": False,
            "policy_engine/decide/power_of_two/64": False,
            "policy_engine/decide/random/512": False,
            "policy_engine/decide/round_robin/64": False,
            "policy_engine/view_build/64": False,
            "policy_engine/spec/parse_validate": False,
        },
        "policy-engine-affinity.toml": {
            "policy_affinity/decide/cache_aware/8": True,
            "policy_affinity/decide/cache_aware/512": True,
            "policy_affinity/view_build/64": False,
        },
        "policy-engine-prefix-hash.toml": {
            "policy_engine/prefix_hash/owner/512": True,
            "policy_engine/prefix_hash/levels_owner/512": True,
            "policy_engine/prefix_hash/levels_owner/64": True,
            "policy_engine/prefix_hash/passed_over/8": True,
            "policy_engine/prefix_hash/levels_passed_over/8": True,
        },
    }
    for name, ids in marked.items():
        gates = mod.load_gates(root / name)
        for bench_id, whole in ids.items():
            assert gates.threshold_for(bench_id).whole_pass is whole, f"{name}: {bench_id}"
    # The thresholds of the marked ids are the ones they had.
    engine = mod.load_gates(root / "policy-engine.toml")
    assert engine.threshold_for("policy_engine/decide/least_load_credited/8").threshold == 0.15
    assert engine.threshold_for("policy_engine/decide/cache_aware/8").threshold == 0.10
    assert engine.threshold_for("policy_engine/decide/random/64").threshold == 0.10
    prefix = mod.load_gates(root / "policy-engine-prefix-hash.toml")
    # The 512-row ids read above 5 % on the hosted runner and take 15 %.
    for path in ("owner", "passed_over", "levels_owner", "levels_passed_over"):
        assert prefix.threshold_for(f"policy_engine/prefix_hash/{path}/512").threshold == 0.15, path
    for bench_id in ("owner/64", "passed_over/64", "levels_owner/64", "passed_over/8"):
        assert prefix.threshold_for(f"policy_engine/prefix_hash/{bench_id}").threshold == 0.10, (
            bench_id
        )
    for name in ("policy-engine-power-of-two.toml", "policy-store.toml"):
        assert not any(gate.whole_pass for gate in mod.load_gates(root / name).gates), name


# ---------------------------------------------------------------- measured again

# Four jobs failed by time alone on 2026-11-18 and passed when run again: ids of 110 to 270 ns
# that no line of their pull requests reaches, the same instruction count on both executables
# at the same address, the re-measured round agreeing. Per id: the base's runs, the pull
# request's, the re-measured round and the count of the last batch.
TIME_ONLY = {
    "policy_engine/decide/random/512": (
        ([130.705, 130.803, 131.189], [152.681, 153.359, 151.732]),
        (130.953, 158.896),
        154_134_765,
    ),
    "policy_engine/decide/round_robin/512": (
        ([240.053, 245.549, 241.447], [273.386, 272.947, 273.457]),
        (244.616, 274.436),
        177_145_736,
    ),
    "policy_engine/power_of_two/least_requests/8": (
        ([111.903, 105.382, 107.504], [119.716, 120.697, 119.123]),
        (107.17, 121.202),
        303_081_870,
    ),
    "policy_engine/decide/random/64": (
        ([131.147, 129.616, 134.482], [152.831, 154.061, 147.547]),
        (132.948, 146.643),
        153_733_589,
    ),
}


def time_only(mod, bench_id, second=None):
    runs, recheck, count = TIME_ONLY[bench_id]
    side = {"batches": [count // 4 + 9, count // 2, count], "unit": "decide", "address": 0x3E5FC0}
    counts = {"base": dict(side), "pr": dict(side)}
    gate = mod.Gate("^policy_engine/", 0.10)
    row = mod.Row(bench_id, *runs, gate, rechecks=[recheck], counts=counts, second=second)
    row.verdict = mod.judge(row)
    return row


def test_four_jobs_red_by_time_on_the_same_count_ask_for_a_second_measurement(mod):
    for bench_id in TIME_ONLY:
        row = time_only(mod, bench_id)
        # The same count at the same address places nothing, and reads level.
        assert row.same_count and row.moved is False and not row.placed
        assert row.reads_level and row.repeated is None
        # Not measured again, the first measurement decides.
        assert row.verdict == "regression"


def test_a_second_measurement_decides_an_id_on_a_level_count(mod):
    bench_id = "policy_engine/power_of_two/least_requests/8"
    # The runs of the job's second attempt: +4.7 %, within the threshold.
    again = {"base": [150.575, 151.854, 150.691], "pr": [157.623, 157.849, 166.08]}
    row = time_only(mod, bench_id, again)
    assert row.repeated is False and row.verdict == "not repeated" and not row.fails
    # Beyond the threshold with the runs overlapping is not the failure a second time.
    row = time_only(mod, bench_id, {"base": [100, 100, 120], "pr": [118, 119, 121]})
    assert row.repeated is False and row.verdict == "not repeated"
    # Beyond it with every pull-request run worse than every base run: it repeats.
    row = time_only(mod, bench_id, {"base": [107, 108, 109], "pr": [120, 121, 122]})
    assert row.repeated is True and row.verdict == "regression" and row.fails


def test_a_count_unsteady_on_one_executable_alone_reads_level_within_the_level(mod):
    # One job's four counts of a failing id: the base's first read 0.023 % over the other three.
    first, rest = [44_198_696, 88_397_128, 176_793_992], [44_188_328, 88_376_392, 176_752_520]
    counts = {"base": {"batches": first, "again": rest}, "pr": {"batches": rest, "again": rest}}
    runs = [240.542, 237.072, 245.485], [271.799, 271.623, 265.3]
    gate = mod.Gate("^policy_engine/", 0.10)
    row = mod.Row("policy_engine/decide/round_robin/512", *runs, gate, counts=counts)
    assert row.counted.state == "unsteady" and not row.level_count and row.reads_level
    # Further apart than the level, it tells nothing and asks for nothing.
    counts["base"]["batches"] = [44_198_696, 88_397_128, 178_000_000]
    assert row.counted.state == "unsteady" and not row.reads_level


def test_a_count_that_moved_beyond_the_level_is_not_measured_again(mod, tmp_path, monkeypatch):
    counts = {"base": {"encode/a": [5, 10, 2000]}, "head": {"encode/a": [5, 10, 2100]}}
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, counts)
    assert rc == 1 and "second" not in json.loads((out / "results.json").read_text())
    assert "Measured again" not in (out / "gate.md").read_text()


def test_a_run_measures_an_id_again_from_fresh_copies_and_passes_one_that_does_not_repeat(
    mod, tmp_path, monkeypatch
):
    same = {"encode/a": [5414, 8268, 16270], "address": 0x3BE0C0}
    counts = {"base": dict(same), "head": dict(same)}
    # The pull request's runs: four over the threshold (the rounds and the re-measured one),
    # one the count's directory stands for, and the second measurement's three, level.
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, counts, slow=(*SLOW, 0, 101, 102, 100))
    assert rc == 0
    saved = json.loads((out / "results.json").read_text())
    second = saved["second"]
    assert second["ids"] == ["encode/a"] and "skipped" not in second
    assert [run["encode/a"] for run in second["pr"]["runs"]] == [101, 102, 100]
    assert second["order"] == ["base-1", "pr-1", "pr-2", "base-2", "base-3", "pr-3"]
    # The first measurement stands as it was read, and no copy is left behind.
    assert len(saved["pr"]["runs"]) == 4
    assert not (out / "head" / "second").exists() and not (out / "head" / "bench").exists()
    assert f"+ {out / 'head' / 'second'} --bench" in (out / "head" / "second-1.log").read_text()
    text = (out / "gate.md").read_text()
    assert "### Benchmark gate: PASS" in text and "| not repeated |" in text
    assert "Measured again before the verdict, 3 more run(s) per side from a fresh copy" in text
    assert " 1 not repeated (" in text and "0 failed" in text
    assert "pull request runs [101, 102, 100] ns (+0.0%): it does not repeat." in text
    # Saved, the results read the same.
    gate_file = str(tmp_path / "gates.toml")
    assert mod.main(["compare", "--gates", gate_file, str(out / "results.json")]) == 0


def test_a_second_measurement_starts_only_with_room_for_all_its_runs(mod, tmp_path, monkeypatch):
    # A clock that moves a second at each read: a run takes one, the second measurement's six
    # runs ask for six, and the count ends three before the run's budget does.
    clock = [0.0]

    def monotonic():
        clock[0] += 1.0
        return clock[0]

    def count_ids(ids, copies, out, cwd, gates, extra_args, cpu, deadline):
        clock[0] = deadline - 4.0
        side = {"batches": [9, 20, 40], "unit": "decide", "address": 0x5000}
        return {bench_id: {"base": dict(side), "pr": dict(side)} for bench_id in ids}

    stand_in = types.SimpleNamespace(monotonic=monotonic, sleep=lambda seconds: None)
    monkeypatch.setattr(mod, "time", stand_in)
    monkeypatch.setattr(mod, "count_ids", count_ids)
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, {})
    assert rc == 1
    saved = json.loads((out / "results.json").read_text())
    assert saved["second"]["skipped"] == "budget" and saved["second"]["base"]["runs"] == []
    assert not (out / "head" / "second-1").exists()
    text = (out / "gate.md").read_text()
    assert "Not measured again, `encode/a`: the run's budget left no room" in text
    assert "and the failure stands." in text and "FAIL regression" in text
    # With room for the six runs it is taken.
    clock[0] = 0.0
    monkeypatch.setattr(
        mod, "count_ids", lambda ids, *rest: count_ids(ids, *rest[:-1], rest[-1] - 96.0)
    )
    rc, out, _ = count_run(mod, tmp_path, monkeypatch, {}, name="room")
    saved = json.loads((out / "results.json").read_text())
    assert rc == 1 and len(saved["second"]["pr"]["runs"]) == 3


def counted_row(mod, counts, whole=False):
    """A failing row on which each executable counted what `counts` gives it."""
    gate = mod.Gate("^encode/", 0.10, whole_pass=whole)
    return mod.Row("encode/a", [100, 101, 102], [115, 116, 117], gate, counts=counts)


def level_row(mod, base, pr, whole=False):
    """A failing row whose count read `base` and `pr`: for each executable, the last batch of
    its first count and of its second."""

    def side(first, again):
        return {"batches": [5, 10, first], "again": [5, 10, again]}

    return counted_row(mod, {"base": side(*base), "pr": side(*pr)}, whole)


def test_a_count_reads_level_where_every_count_is_within_the_level(mod):
    # Not steady on the base alone, 0.02 % apart, the pull request's two counts the same.
    assert level_row(mod, (100_020, 100_000), (100_000, 100_000)).reads_level
    # Not steady on both, each inside the level.
    assert level_row(mod, (100_020, 100_000), (100_010, 100_000)).reads_level
    # The pull request's own two counts 10 % apart beside a base that is not steady either:
    # the base's drift says nothing of them. The mirrored shape reads the same.
    assert not level_row(mod, (100_020, 100_000), (100_000, 110_000)).reads_level
    assert not level_row(mod, (100_000, 110_000), (100_020, 100_000)).reads_level
    # The two executables further apart than the level.
    assert not level_row(mod, (100_020, 100_000), (101_000, 101_000)).reads_level
    # An executable counted once has no second count to be inside the level.
    once = level_row(mod, (100_020, 100_000), (100_000, 100_000))
    del once.counts["pr"]["again"]
    assert not once.reads_level
    # Each neighbour inside the level and the two second counts 1.4 % apart: every pair of the
    # four is compared.
    assert not level_row(mod, (100_000, 99_500), (100_400, 100_900)).reads_level
    # A count that moved, the same twice on each executable, is the same noise in its other
    # state: inside the level it reads level too, on an id that times a whole pass or none.
    moved = (100_000, 100_000), (100_020, 100_020)
    assert level_row(mod, *moved).reads_level
    assert level_row(mod, *moved, whole=True).reads_level
    assert not level_row(mod, (100_000, 100_000), (100_600, 100_600)).reads_level


def test_a_count_that_ended_a_batch_early_hides_no_other_from_the_level(mod):
    def level(base, pr):
        """Whether a row reads level on which each executable counted the two lists given."""
        sides = {"base": base, "pr": pr}
        counts = {side: dict(zip(("batches", "again"), two)) for side, two in sides.items()}
        return counted_row(mod, counts).reads_level

    short = [50_000, 100_000, 200_000], [50_000, 100_000]
    # One executable's second count ended a batch early, and the other's own two counts are
    # 10 % apart at the batch it left: the counts that ran that batch are read there, each way.
    far = [50_010, 100_020, 200_040], [50_010, 100_020, 220_040]
    assert not level(short, far)
    assert not level(far, short)
    # The same shape 0.005 % apart at that batch reads level.
    near = [50_010, 100_020, 200_040], [50_010, 100_020, 200_050]
    assert level(short, near)
    assert level(near, short)
    # Three counts ran the batch, each neighbour inside the level and the ends 0.9 % apart: the
    # ends are a pair that last differs there.
    ends = [50_000, 100_010, 200_900], [50_000, 100_010, 201_800]
    assert not level(short, ends)
    assert not level(ends, short)
    # The set: two counts last differ at the third batch, level there, where a third count that
    # ran it is 5 % off and level with both at the fourth. No pair's own last batch shows it:
    # the batch is compared on every count that ran it.
    whole = [50_000, 100_000, 200_000, 400_000], [50_000, 100_000]
    apart = [50_000, 100_010, 200_010, 400_000], [50_000, 100_010, 210_000, 400_001]
    assert not level(whole, apart)
    assert not level(apart, whole)
    # The two second counts alone ran the fourth batch, 1.5 % apart there: every two of the four
    # are read, over the batches the two ran.
    seconds = [5, 10, 100_000], [5, 10, 100_000, 200_000]
    further = [5, 10, 100_020], [5, 10, 100_020, 203_000]
    assert not level(seconds, further)
    assert not level(further, seconds)
    # The first counts 1 % apart at the last batch all four ran that differs, and level at the
    # batch after it, which one second count left: the batches all four ran are read too.
    early = [5, 100_000, 200_000, 400_000], [5, 100_000, 200_040]
    late = [5, 101_000, 200_000, 400_100], [5, 101_000, 200_000, 400_100]
    assert not level(early, late)
    assert not level(late, early)


def test_a_second_measurement_short_of_its_runs_leaves_the_first_to_decide(mod):
    bench_id = "policy_engine/power_of_two/least_requests/8"
    row = time_only(mod, bench_id, {"base": [100, 101, 102], "pr": [100, 101]})
    assert row.measured_again is None and row.repeated is None
    assert row.verdict == "regression"


# ---------------------------------------------------------------- the same program


def elf(image: bytes, build_id: bytes, unloaded: bytes) -> bytes:
    """A 64-bit little-endian ELF file of one loadable segment (its headers, two notes and
    `image`) with `unloaded` bytes behind it, and a section table its header names there."""
    # A note of another kind stands first: the walk steps over it to the build id.
    note = struct.pack("<III", 4, 16, 1) + b"GNU\x00" + bytes(16)
    note += struct.pack("<III", 4, len(build_id), 3) + b"GNU\x00" + build_id
    loaded = 64 + 2 * 56 + len(note) + len(image)
    sections = (loaded + len(unloaded), 64, len(unloaded) % 7, 1)
    header = struct.pack(
        "<6s10xHHIQQQIHHH", b"\x7fELF\x02\x01", 3, 62, 1, 0x1000, 64, sections[0], 0, 64, 56, 2
    )
    header += struct.pack("<HHH", *sections[1:])
    segments = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, loaded, loaded, 4096)
    segments += struct.pack("<IIQQQQQQ", 4, 4, 176, 176, 0, len(note), len(note), 4)
    return header + segments + note + image + unloaded


def test_two_files_that_load_the_same_image_are_the_same_program(mod, tmp_path, monkeypatch):
    # A change outside a bench's code moves its executable's build id and what no run reads.
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base, pr = tmp_path / "base.elf", tmp_path / "pr.elf"
    base.write_bytes(elf(b"code" * 64, b"A" * 20, b"symbols"))
    pr.write_bytes(elf(b"code" * 64, b"B" * 20, b"other symbols, longer"))
    assert base.read_bytes()[0x28:0x30] != pr.read_bytes()[0x28:0x30]
    assert mod.image_of(base) == mod.image_of(pr) is not None
    gate_file, out = tmp_path / "gates.toml", tmp_path / "out"
    gate_file.write_text(GATE_TOML)
    args = ["run", "--gates", str(gate_file), "--base", str(base), "--pr", str(pr)]
    assert mod.main([*args, "--out", str(out)]) == 0
    saved = json.loads((out / "results.json").read_text())
    assert saved["identical"] is True and saved["base"]["runs"] == []
    assert saved["binaries"]["base"]["image"] == saved["binaries"]["pr"]["image"]
    assert "### Benchmark gate: PASS (the same program)" in (out / "gate.md").read_text()
    # One byte of the image apart they are two programs.
    pr.write_bytes(elf(b"code" * 63 + b"cade", b"A" * 20, b"symbols"))
    assert mod.image_of(base) != mod.image_of(pr)
    # A file that is no such ELF has no image, nor has one whose headers do not add up.
    script = tmp_path / "bench.py"
    script.write_text("#!/usr/bin/env python3\n")
    assert mod.image_of(script) is None
    base.write_bytes(elf(b"code" * 64, b"A" * 20, b"")[:200])
    assert mod.image_of(base) is None


def test_a_file_whose_header_no_loadable_segment_covers_has_no_image(mod, tmp_path):
    def entered(at: int, covered: bool) -> Path:
        """A file whose entry address is `at`, its header inside the loadable segment or not."""
        data = bytearray(elf(b"code" * 64, b"A" * 20, b"symbols"))
        struct.pack_into("<Q", data, 0x18, at)
        if not covered:
            # The segment starts behind the file header: 64 bytes on, 64 bytes fewer.
            (size,) = struct.unpack_from("<Q", data, 64 + 32)
            struct.pack_into("<Q", data, 64 + 8, 64)
            struct.pack_into("<Q", data, 64 + 32, size - 64)
        path = tmp_path / f"bench-{at:x}-{covered}"
        path.write_bytes(bytes(data))
        return path

    # The loader maps the header with the first segment: the entry address is in the image.
    one, other = mod.image_of(entered(0x1000, True)), mod.image_of(entered(0x2000, True))
    assert one != other and None not in (one, other)
    # No segment covers it: two entry addresses would read as one image, so there is none.
    assert mod.image_of(entered(0x1000, False)) is None
    assert mod.image_of(entered(0x2000, False)) is None


def test_what_the_loader_reads_from_the_file_lies_inside_a_loadable_segment(mod, tmp_path):
    def image(named: bytes, held: bool) -> str | None:
        """The image of a file whose second program header, the interpreter's, names the two
        notes that hold `named`, inside the loadable segment or behind its end."""
        data = bytearray(elf(b"code" * 64, named, b"symbols"))
        struct.pack_into("<I", data, 64 + 56, 3)
        if not held:
            struct.pack_into("<Q", data, 64 + 32, 64 + 2 * 56)
        path = tmp_path / f"bench-{named[:1].decode()}-{held}"
        path.write_bytes(bytes(data))
        return mod.image_of(path)

    # Inside the segment the path is in the image: two paths, two images.
    one, other = image(b"A" * 20, True), image(b"B" * 20, True)
    assert one != other and None not in (one, other)
    # Behind the segment's end the loader still reads it, from the file: two paths would read as
    # one image, so there is none.
    assert image(b"A" * 20, False) is None
    assert image(b"B" * 20, False) is None
    # A segment that starts at the file's start holds the header only where it holds all of it.
    data = bytearray(elf(b"code" * 64, b"A" * 20, b"symbols"))
    # The note names no bytes here: behind so short a segment it would refuse the file itself.
    struct.pack_into("<Q", data, 64 + 56 + 32, 0)
    path = tmp_path / "bench-short"
    for size, held in ((63, False), (64, True)):
        struct.pack_into("<Q", data, 64 + 32, size)
        path.write_bytes(bytes(data))
        assert (mod.image_of(path) is not None) == held


def test_a_header_no_loader_reads_from_the_file_may_pass_its_segments_end(mod, tmp_path):
    def image(kind: int, held: bool) -> str | None:
        """The image of a file whose second program header is of `kind` and names the two
        notes, whole inside the loadable segment or from inside it past its end."""
        data = bytearray(elf(b"code" * 64, b"A" * 20, b"symbols"))
        struct.pack_into("<I", data, 64 + 56, kind)
        if not held:
            struct.pack_into("<Q", data, 64 + 32, 64 + 2 * 56 + 24)
        path = tmp_path / f"bench-{kind:x}-{held}"
        path.write_bytes(bytes(data))
        return mod.image_of(path)

    # A loader reads the interpreter's path, a note and the property note from the file: each
    # lies whole inside the segment, or the file has no image.
    for kind in (3, 4, 0x6474E553):
        assert image(kind, True) is not None and image(kind, False) is None
    # The range made read-only after relocation is read from memory alone, and a linker may
    # round its header's file size up past the segment's end: such a file keeps its image.
    assert image(0x6474E552, True) is not None and image(0x6474E552, False) is not None
