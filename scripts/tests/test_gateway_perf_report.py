"""The gateway perf reports: the Numbers table, the gate verdicts and the capacity rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "gateway_perf" / "gateway_perf.py"


@pytest.fixture(scope="module")
def gp():
    spec = importlib.util.spec_from_file_location("gateway_perf", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["gateway_perf"] = module
    spec.loader.exec_module(module)
    return module


def results(
    base: dict[str, list[float]], pr: dict[str, list[float]], unit: str | None = ""
) -> dict:
    def runs(values):
        n = max((len(v) for v in values.values()), default=0)
        return [{k: v[i] for k, v in values.items() if i < len(v)} for i in range(n)]

    out = {"repeats": 3, "base": {"runs": runs(base)}, "pr": {"runs": runs(pr)}}
    if unit is not None:
        out["unit"] = unit
    return out


# ---------------------------------------------------------------- the Numbers table


def test_numbers_table_has_the_pull_request_shape_with_medians_change_and_spread(gp):
    res = results(
        {
            "chat/user_instructions_per_request_M": [0.500, 0.502, 0.498],
            "chat/allocation_calls_per_request": [300, 301, 300],
            "chat/rss_mb_after_window": [100.0, 101.0, 100.0],
        },
        {
            "chat/user_instructions_per_request_M": [0.480, 0.481, 0.479],
            "chat/allocation_calls_per_request": [290, 290, 291],
            "chat/rss_mb_after_window": [100.5, 100.0, 100.0],
        },
    )
    table = gp.numbers_table(res, "10,000 requests per cell")
    lines = table.splitlines()
    assert lines[0] == "| metric | before | after | setup |"
    assert lines[1] == "|---|---|---|---|"
    assert (
        lines[2] == "| chat: user instructions per request | 0.500 M | 0.480 M (-4.0 %) "
        "| 10,000 requests per cell; 3+3 interleaved cells; base spread 0.80 % |"
    )
    assert lines[3].startswith("| chat: allocation calls per request | 300 | 290 (-10) |")
    assert lines[4].startswith("| chat: RSS after the window | 100.0 MB | 100.0 MB (+0.0 %) |")


def test_numbers_table_skips_unknown_metrics_and_one_sided_ids(gp):
    res = results({"chat/odd_metric": [1.0], "chat/hwm_mb": [110.0]}, {"chat/odd_metric": [1.0]})
    assert gp.numbers_table(res, "x").splitlines()[2:] == []


# ---------------------------------------------------------------- the gate rule


GATES = """
repeats = 3

[[gate]]
pattern = "/allocation_calls_per_request$"
threshold = 0.02

[[gate]]
pattern = "/user_instructions_per_request_M$"
threshold = 0.02
"""


def test_gate_fails_only_a_separated_change_beyond_the_threshold(gp, tmp_path):
    gates = tmp_path / "g.toml"
    gates.write_text(GATES)
    separated = results(
        {"chat/allocation_calls_per_request": [300, 301, 300]},
        {"chat/allocation_calls_per_request": [320, 321, 319]},
    )
    md, rc = gp.gate_report(separated, gates)
    assert rc == 1 and "FAIL regression" in md and "### Benchmark gate: FAIL" in md
    assert "| 300 (300 - 301) | 320 (319 - 321) | +6.7% |" in md
    overlapping = results(
        {"chat/allocation_calls_per_request": [300, 330, 300]},
        {"chat/allocation_calls_per_request": [320, 321, 319]},
    )
    md, rc = gp.gate_report(overlapping, gates)
    assert rc == 0 and "suspect (runs overlap)" in md
    small = results(
        {"chat/allocation_calls_per_request": [300, 301, 300]},
        {"chat/allocation_calls_per_request": [303, 303, 302]},
    )
    md, rc = gp.gate_report(small, gates)
    assert rc == 0 and "| ok |" in md


def test_gate_reports_ungated_ids_and_improvements_without_failing(gp, tmp_path):
    gates = tmp_path / "g.toml"
    gates.write_text(GATES)
    res = results(
        {
            "chat/user_instructions_per_request_M": [0.50, 0.50, 0.50],
            "chat/cpu_us_per_request": [700, 720, 690],
        },
        {
            "chat/user_instructions_per_request_M": [0.45, 0.45, 0.45],
            "chat/cpu_us_per_request": [900, 910, 905],
        },
    )
    md, rc = gp.gate_report(res, gates)
    assert rc == 0
    assert "faster improvement" in md
    assert (
        "| chat/cpu_us_per_request | 700 (690 - 720) | 905 (900 - 910) | +29.3% |" in md
        and "report only" in md
    )


def test_gate_without_a_file_is_report_only(gp):
    md, rc = gp.gate_report(results({"a/x": [1.0, 1.0, 1.0]}, {"a/x": [2.0, 2.0, 2.0]}), None)
    assert rc == 0 and "report only" in md and "0 gated" in md


# ---------------------------------------------------------------- the capacity rule


def step(gp, offered: float, ttft_p99: float = 5.0, achieved: float | None = None, failed: int = 0):
    summary = {
        "sent": 1_000,
        "ok": 1_000 - failed,
        "failed": failed,
        "skipped": 0,
        "achieved_rps": offered if achieved is None else achieved,
        "ttft_ms": {"p50": 1.0, "p99": ttft_p99},
        "e2e_ms": {"p50": 2.0, "p99": 10.0},
    }
    slo = {"ttft_p99_ms": 100.0, "e2e_p99_ms": 1000.0, "error_rate": 0.0, "achieved_fraction": 0.95}
    return gp.judge_step(summary, 0.5, offered, slo)


def test_capacity_is_the_highest_holding_step_with_the_strict_reading_beside(gp):
    steps = [
        step(gp, 200),
        step(gp, 400, ttft_p99=500),
        step(gp, 800),
        step(gp, 1200, ttft_p99=500),
    ]
    cap = gp.capacity(steps)
    assert cap["hold"] == 800
    assert cap["strict"] == 200
    assert cap["failing_steps_below_hold"] == [400]
    assert cap["at_ceiling"] is False
    assert "highest offered step" in cap["rule"]
    assert gp.capacity([step(gp, 200), step(gp, 400)])["at_ceiling"] is True


def test_a_step_fails_on_errors_or_an_achieved_shortfall(gp):
    assert step(gp, 400, failed=1)["holds"] is False
    short = step(gp, 400, achieved=300)
    assert short["holds"] is False and "achieved 300" in short["reasons"][0]
    good = step(gp, 400)
    assert good["holds"] is True and good["cpu_ms_per_request"] == pytest.approx(0.5)
    assert gp.capacity([step(gp, 100, ttft_p99=900)])["hold"] is None
