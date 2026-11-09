"""The benchmark regression gate: its verdicts, its criterion reader and its run loop."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
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
