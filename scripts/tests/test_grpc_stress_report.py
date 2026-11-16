"""The gRPC stress report's verdict: a cell against the bounds of its name."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

BED = Path(__file__).resolve().parents[1] / "bench" / "grpc_stress"
SCRIPT = BED / "report.py"
BOUNDS = BED / "bounds.toml"
RUN_ALL = BED / "run_all.sh"
NIGHTLY = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "nightly-grpc-stress.yml"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("grpc_stress_report", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["grpc_stress_report"] = module
    spec.loader.exec_module(module)
    return module


def _cell(name: str, **figures) -> dict:
    """An analyzed cell as `report.py` builds one: a quiet unary cell unless a figure says otherwise."""
    cell = {
        "cell": name,
        "failed": 0,
        "shed_admission": 0.0,
        "shed_overload": 0.0,
        "rss_anon_slope_mb_per_min": 1.5,
        "rss_anon_steady_slope_mb_per_min": 1.2,
        "cpu_ms_per_req": 0.4,
        "ttft_ms": {"p50": 100.0, "p99": 102.0},
        "e2e_ms": {"p50": 100.0, "p99": 102.0},
        "ttfb_ms": {},
        "gap_us": {},
        "rss_mb_max": 220.0,
    }
    cell.update(figures)
    return cell


def test_a_cell_within_its_bounds_is_no_breach(mod):
    bounds = tomllib.loads(BOUNDS.read_text())
    breaches, unbounded = mod.judge([_cell("unary-u1k-r1")], bounds)
    assert breaches == []
    assert unbounded == []


def test_a_breach_names_the_cell_the_figure_and_the_bound(mod):
    bounds = {"unary-u1k": {"rss_anon_steady_slope_mb_per_min_max": 31, "failed_max": 0}}
    breaches, _ = mod.judge(
        [_cell("unary-u1k-r2", rss_anon_steady_slope_mb_per_min=48.2, failed=3)], bounds
    )
    assert breaches == [
        ("unary-u1k-r2", "failed", 3, 0),
        ("unary-u1k-r2", "anon MB/min steady", 48.2, 31),
    ]
    text = mod.verdict(breaches, [])
    assert "`unary-u1k-r2`: anon MB/min steady 48.200 is above its bound 31" in text


def test_the_shed_bound_reads_both_shed_counters(mod):
    bounds = {"idle-default-idle10k": {"shed_max": 5803}}
    quiet = _cell("idle-default-idle10k-r1", shed_admission=5803.0, shed_overload=0.0)
    over = _cell("idle-default-idle10k-r1", shed_admission=5803.0, shed_overload=1.0)
    assert mod.judge([quiet], bounds)[0] == []
    assert mod.judge([over], bounds)[0] == [("idle-default-idle10k-r1", "shed", 5804, 5803)]


def test_a_cell_the_file_does_not_name_is_unbounded_not_red(mod):
    breaches, unbounded = mod.judge(
        [_cell("image-img4m-r1", rss_anon_steady_slope_mb_per_min=900.0)], {}
    )
    assert breaches == []
    assert unbounded == ["image-img4m-r1"]
    assert "Unbounded, judged by nothing: `image-img4m-r1`." in mod.verdict([], unbounded)


def test_a_bounded_figure_a_cell_lacks_is_a_breach_naming_it(mod):
    bounds = {"long-l128k": {"cpu_ms_per_req_max": 44.1, "ttft_p99_ms_max": 161}}
    cell = _cell("long-l128k-r1", cpu_ms_per_req=None, ttft_ms={})
    breaches, unbounded = mod.judge([cell], bounds)
    assert breaches == [
        ("long-l128k-r1", "CPU ms/req", None, 44.1),
        ("long-l128k-r1", "TTFT p99", None, 161),
    ]
    assert unbounded == []
    text = mod.verdict(breaches, [])
    assert "2 figure(s) outside the bounds or not reported:" in text
    assert "- `long-l128k-r1`: TTFT p99 is not reported, its bound is 161" in text


def test_every_nightly_cell_has_bounds(mod):
    """The cells `run_all.sh` defines for the families the nightly runs all have an entry."""
    families = re.search(r"run_all\.sh ((?:[a-z-]+ ?)+)", NIGHTLY.read_text()).group(1).split()
    text = RUN_ALL.read_text()
    cells = []
    for family in families:
        block = re.search(rf"^\s+{re.escape(family)}\)\n(.*?);;", text, re.S | re.M)
        assert block, f"no `{family})` case in run_all.sh"
        cells += [f"{family}-{m}" for m in re.findall(r'"([a-z0-9]+)=--', block.group(1))]
    assert cells, "no cells read from run_all.sh"
    bounds = mod.read_bounds(str(BOUNDS))  # every key of the shipped file is one the report judges
    assert set(cells) <= set(bounds), f"cells without bounds: {sorted(set(cells) - set(bounds))}"


def test_the_nightly_passes_the_bounds_and_run_all_reads_the_verdict():
    text = RUN_ALL.read_text()
    assert 'bounds_args=(--bounds "$BOUNDS")' in text
    assert '3) log "report: a cell is outside its bounds' in text
    assert '2) log "report: the bounds file is not usable' in text
    assert "BOUNDS: scripts/bench/grpc_stress/bounds.toml" in NIGHTLY.read_text()


def test_a_bounds_key_no_figure_answers_to_is_a_setup_error(mod, tmp_path):
    path = tmp_path / "bounds.toml"
    path.write_text("[unary-u1k]\ncpu_ms_per_request_max = 1.5\n")
    with pytest.raises(mod.BoundsError, match=r"\[unary-u1k\] names no figure the report judges"):
        mod.read_bounds(str(path))


def test_a_bad_bounds_file_leaves_the_tables_and_exits_2(mod, tmp_path):
    path = tmp_path / "bounds.toml"
    path.write_text("[unary-u1k]\ncpu_ms_per_request_max = 1.5\n")
    text, out, code = mod.render([_cell("unary-u1k-r1")], str(path))
    assert code == 2
    assert text.startswith("## Cells") and "## Mean over repeats" in text
    assert "No verdict: " in text and "cpu_ms_per_request_max" in text
    assert "cpu_ms_per_request_max" in out["setup_error"] and out["cells"]
    for name, body in (
        ("malformed.toml", b"[unary-u1k\nfailed_max = 0\n"),
        ("not-a-number.toml", b'[unary-u1k]\nfailed_max = "0"\n'),
        ("a-bool.toml", b"[unary-u1k]\nfailed_max = true\n"),
        ("nan.toml", b"[unary-u1k]\nfailed_max = nan\n"),
        ("inf.toml", b"[unary-u1k]\nfailed_max = inf\n"),
        ("not-utf8.toml", b"[unary-u1k]\nfailed_max = 0 # \xff\n"),
    ):
        (tmp_path / name).write_bytes(body)
        text, out, code = mod.render([_cell("unary-u1k-r1")], str(tmp_path / name))
        assert code == 2 and "No verdict: " in text and name in out["setup_error"], name
    text, out, code = mod.render([_cell("unary-u1k-r1")], str(tmp_path / "missing.toml"))
    assert code == 2 and "No verdict: " in text and "missing.toml" in out["setup_error"]


def test_shed_is_absent_not_zero_without_both_counters(mod):
    cell = _cell("unary-u1k-r1", shed_admission=None, shed_overload=None)
    assert mod.flat(cell)["shed"] is None
    assert mod.judge([cell], {"unary-u1k": {"shed_max": 0}})[0] == [
        ("unary-u1k-r1", "shed", None, 0)
    ]


def test_a_missing_empty_or_falling_scrape_makes_a_counter_absent(mod):
    name = "smg_admission_queue_rejected_total"
    before = f"{name} 3\n"
    assert mod.counter_delta(before, f"{name} 10\n", name) == 7.0
    # went down: a reset or a rename between the scrapes
    assert mod.counter_delta(before, "other_total 9\n", name) is None
    # published at its first increment, inside the cell
    assert mod.counter_delta("other_total 1\n", f"{name} 10\n", name) == 10.0
    assert mod.counter_delta(before, "", name) is None
    assert mod.counter_delta(before, "\n", name) is None
    assert mod.counter_delta(None, f"{name} 10\n", name) is None


def _write_cell(tmp_path, name, anon_kb, warmup=5, duration=40):
    """A cell directory as `cell.sh` leaves one: `cell.json` and one sample a second with the anon RSS given,
    from the cell's start through its window."""
    d = tmp_path / name
    d.mkdir()
    t0 = 1_000_000.0
    cell = {
        "name": name,
        "t0": t0,
        "t1": t0 + warmup + duration,
        "warmup": warmup,
        "duration": duration,
    }
    (d / "cell.json").write_text(json.dumps(cell))
    with (d / "samples.csv").open("w") as f:
        f.write("t,cpu_s,vmrss_kb,rssanon_kb\n")
        for i, kb in enumerate(anon_kb):
            f.write(f"{t0 + i},{i * 0.01},{kb + 40_000},{kb}\n")
    return d


def test_the_judged_slope_is_the_steady_state_past_the_warm_up(mod, tmp_path):
    # Anon RSS climbs by 4 MiB a second through the window's first half and
    # holds through the second: the whole-window slope records the warm-up,
    # the steady slope the verdict judges is near zero.
    warmup, duration = 5, 40
    kb = [
        100_000 + 4096 * max(0, min(i - warmup, duration // 2))
        for i in range(warmup + duration + 1)
    ]
    cell = mod.analyze(str(_write_cell(tmp_path, "unary-u1k-r1", kb, warmup, duration)))
    assert cell["rss_anon_slope_mb_per_min"] > 100
    assert abs(cell["rss_anon_steady_slope_mb_per_min"]) < 1


def test_a_leak_through_the_window_breaches_the_steady_bound(mod, tmp_path):
    # The forged leak: 4 MiB a sample through the whole window is 240 MB/min in
    # both slopes, above the shipped bound of every cell.
    warmup, duration = 5, 40
    kb = [100_000 + 4096 * i for i in range(warmup + duration + 1)]
    cell = mod.analyze(str(_write_cell(tmp_path, "unary-u1k-r1", kb, warmup, duration)))
    assert cell["rss_anon_steady_slope_mb_per_min"] == pytest.approx(240, rel=0.01)
    breaches, _ = mod.judge([cell], mod.read_bounds(str(BOUNDS)))
    steady = [b for b in breaches if b[1] == "anon MB/min steady"]
    assert steady == [("unary-u1k-r1", "anon MB/min steady", pytest.approx(240, rel=0.01), 41)]
