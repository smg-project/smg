"""The gateway perf driver: its arguments, its counters' arithmetic and its results file."""

from __future__ import annotations

import importlib.util
import json
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


# ---------------------------------------------------------------- arguments


def test_every_subcommand_parses_with_defaults(gp):
    cpu = gp.parse_args(["cpu", "--base", "/b", "--head", "/h"])
    assert cpu.command == "cpu" and cpu.repeats == 3 and cpu.requests == 10_000
    assert cpu.paths == gp.DEFAULT_PATHS and cpu.counters == "auto"
    alloc = gp.parse_args(["alloc", "--base", "/b", "--rate-scale", "0.5", "--repeats", "5"])
    assert alloc.rate_scale == 0.5 and alloc.repeats == 5
    mem = gp.parse_args(["mem", "--base", "/b"])
    assert mem.paths == "stream" and mem.windows == 5 and mem.burst == 2.0 and mem.repeats == 1
    live = gp.parse_args(["live", "--rates", "100,200"])
    assert live.rates == "100,200" and live.slo_ttft_p99_ms == 100.0 and live.base == ""
    bed = gp.parse_args(["bed", "smoke"])
    assert bed.action == "smoke"
    report = gp.parse_args(["report", "--out", "/tmp/x"])
    assert report.out == "/tmp/x"


def test_paths_are_checked_and_ordered(gp):
    assert [p.name for p in gp.parse_paths("stream, chat")] == ["stream", "chat"]
    with pytest.raises(ValueError, match="unknown path"):
        gp.parse_paths("chat,nosuch")
    with pytest.raises(ValueError, match="no path"):
        gp.parse_paths(" , ")


def test_every_path_has_a_rate_a_client_mode_and_an_answer(gp):
    for spec in gp.PATHS.values():
        assert spec.rate > 0
        assert "--mode" in spec.client
        assert "--seed" in spec.client
        assert spec.answer and spec.finish in ("stop", "length")
    assert gp.PATHS["stream"].finish == "length" and "--stream" in gp.PATHS["stream"].client
    assert json.loads(gp.PATHS["tools"].answer)["name"].startswith("lookup_")
    assert set(json.loads(gp.PATHS["json"].answer)) == {"answer", "confidence", "status"}


def test_answers_are_one_per_request_with_slack_served_one_token_a_frame(gp):
    items = gp.answers(gp.PATHS["stream"], 12)
    assert len(items) == 12
    assert items[0] == {
        "output_text": gp.PATHS["stream"].answer,
        "finish_reason": "length",
        "chunk_tokens": 1,
        "delay_ms": 0,
    }
    assert items[0] is not items[1]
    assert len(gp.PATHS["stream"].answer.split()) == 128
    assert len(gp.PATHS["chat"].answer.split()) == 8


def test_interleaving_alternates_which_side_goes_first(gp):
    assert gp.interleaved_order(3) == [
        (1, "base"),
        (1, "pr"),
        (2, "pr"),
        (2, "base"),
        (3, "base"),
        (3, "pr"),
    ]


def test_port_block_is_free_and_laid_out_as_the_bed_expects(gp):
    ports = gp.pick_ports(0, 4)
    base = ports["gateway"]
    assert ports == {
        "gateway": base,
        "metrics": base + 1,
        "admin": base + 9,
        "mock_base": base + 10,
    }
    assert gp.pick_ports(29800, 2)["admin"] == 29809
    with pytest.raises(RuntimeError, match="no free port block"):
        gp.pick_ports(0, 4, first=29700, last=29705)


# ---------------------------------------------------------------- counters


def test_perf_csv_skips_unsupported_and_comments(gp):
    text = (
        "# started\n123,,instructions:u,1000,100.00,,\n"
        "<not supported>,,cycles,0,100.00,,\n4.5,msec,task-clock,1,100.00,,\n"
    )
    assert gp.parse_perf_csv(text) == {"instructions:u": 123.0, "task-clock": 4.5}


JEMALLOC_LOG = (
    "gateway lines\n"
    '{"jemalloc": {"stats": {"allocated": 10, "resident": 20},'
    ' "arenas": {"bin": [{"size": 8}, {"size": 16}], "lextent": [{"size": 16384}]},'
    ' "stats.arenas": {"merged": {"small": {"nrequests": 30, "nmalloc": 20}, "large": {"nrequests": 2, "nmalloc": 2},'
    ' "bins": [{"nrequests": 10}, {"nrequests": 20}], "lextents": [{"nrequests": 2}]}}}}\n'
    "trailer\n"
)


def test_jemalloc_exit_statistics_count_requests_and_bytes_by_size_class(gp):
    je = gp.parse_jemalloc(JEMALLOC_LOG)
    assert je["nrequests"] == 32 and je["small_nrequests"] == 30 and je["large_nrequests"] == 2
    assert je["bytes_requested"] == 10 * 8 + 20 * 16 + 2 * 16384
    assert je["nmalloc"] == 22 and je["allocated"] == 10
    assert "error" in gp.parse_jemalloc("no statistics here")
    assert "error" in gp.parse_jemalloc('{"jemalloc": {broken')


def idle_cell(side="base"):
    return {
        "side": side,
        "path": "idle",
        "label": "idle",
        "window_s": 10.0,
        "cpu_s": 0.1,
        "perf": {"instructions:u": 1e8, "instructions": 2e8, "cycles": 3e8},
        "jemalloc": {"nrequests": 1_000, "bytes_requested": 100_000, "large_nrequests": 1},
    }


def loaded_cell(side="base", label="r1", **overrides):
    cell = {
        "side": side,
        "path": "chat",
        "label": label,
        "sent": 1_000,
        "window_s": 20.0,
        "cpu_s": 0.7,
        "status_1": {"VmRSS": 102_400, "VmHWM": 112_640},
        "perf": {"instructions:u": 7e8, "instructions": 13e8, "cycles": 16e8},
        "jemalloc": {
            "nrequests": 301_000,
            "bytes_requested": 100_000 + 1_000 * 1024 * 120,
            "large_nrequests": 1,
        },
    }
    cell.update(overrides)
    return cell


def test_per_request_subtracts_the_idle_rate_over_the_window(gp):
    r = gp.per_request(loaded_cell(), idle_cell())
    # 7e8 user instructions minus 20 s of the idle rate (1e8 / 10 s) = 5e8 over 1,000 requests
    assert r["user_instructions_per_request_M"] == pytest.approx(0.5)
    assert r["instructions_per_request_M"] == pytest.approx(0.9)
    assert r["cycles_per_request_M"] == pytest.approx(1.0)
    assert r["cpu_us_per_request"] == pytest.approx((0.7 - 0.2) / 1_000 * 1e6)
    assert r["allocation_calls_per_request"] == pytest.approx(300.0)
    assert r["kib_requested_per_request"] == pytest.approx(120.0)
    assert r["large_allocations_per_request"] == 0.0
    assert r["rss_mb_after_window"] == 100.0 and r["hwm_mb"] == 110.0


def test_per_request_without_counters_or_idle_still_gives_cpu_time_and_rss(gp):
    cell = loaded_cell(perf={}, jemalloc={"error": "none"})
    r = gp.per_request(cell, None)
    assert set(r) == {"cpu_us_per_request", "rss_mb_after_window", "hwm_mb"}
    assert r["cpu_us_per_request"] == pytest.approx(700.0)
    assert gp.per_request(dict(cell, sent=0), None) == {}


def test_results_file_has_one_run_per_repeat_in_bench_gate_shape(gp):
    cells = [idle_cell("base"), idle_cell("pr")]
    for i, side in gp.interleaved_order(3):
        allocs = 301_000 if side == "base" else 291_000
        cells.append(
            loaded_cell(
                side,
                f"r{i}",
                jemalloc={"nrequests": allocs, "bytes_requested": 100_000, "large_nrequests": 1},
            )
        )
    results = gp.results_from_cells(cells, 3, {"base": Path("/b"), "pr": Path("/h")}, note="n")
    assert results["unit"] == "" and results["repeats"] == 3 and results["note"] == "n"
    assert results["order"] == ["base-1", "pr-1", "pr-2", "base-2", "base-3", "pr-3"]
    assert len(results["base"]["runs"]) == 3 and len(results["pr"]["runs"]) == 3
    assert results["base"]["runs"][0]["chat/allocation_calls_per_request"] == pytest.approx(300.0)
    assert results["pr"]["runs"][0]["chat/allocation_calls_per_request"] == pytest.approx(290.0)
    assert results["base"]["binary"] == "/b"


def test_outputs_are_written_and_a_warned_cell_fails_the_run(gp, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    settings = gp.Settings(out=tmp_path, bed=None, binaries={}, paths=[])
    cells = [idle_cell("base"), idle_cell("pr"), loaded_cell("base"), loaded_cell("pr")]
    results = gp.results_from_cells(cells, 1, {})
    args = SimpleNamespace(gates="")
    assert gp.write_outputs(settings, cells, results, args, "setup") == 0
    assert (tmp_path / "numbers.md").exists() and (tmp_path / "results.json").exists()
    cells[-1]["warnings"] = ["3 failed requests"]
    assert gp.write_outputs(settings, cells, results, args, "setup") == 2
    assert "chat: allocation calls per request" in (tmp_path / "numbers.md").read_text()


def test_mem_soak_record_warns_on_failed_requests_and_script_misses(gp, tmp_path, monkeypatch):
    class FakeBed:
        def __init__(self, cfg, binary, directory, jemalloc_stats=True):
            self.cfg, self.pid = cfg, 1

        def up(self):
            pass

        def down(self):
            pass

    sample = {
        "t": 0.0,
        "cpu_s": 0.0,
        "status": {"VmRSS": 1024, "VmHWM": 2048},
        "gauges": {},
        "loadavg": 0.0,
    }
    monkeypatch.setattr(gp, "Bed", FakeBed)
    monkeypatch.setattr(gp, "snapshot", lambda bed: dict(sample))
    monkeypatch.setattr(gp, "enqueue", lambda bed, answers: None)
    monkeypatch.setattr(gp, "answers", lambda spec, n: [])
    monkeypatch.setattr(gp.time, "sleep", lambda seconds: None)
    counts = {"hits": 18, "misses": 2, "unreadable": 1}
    monkeypatch.setattr(gp, "script_counts", lambda bed: counts)
    summaries = iter([{"sent": 10, "ok": 9, "failed": 1}, {"sent": 10, "ok": 8, "failed": 2}])
    monkeypatch.setattr(gp, "run_client", lambda *a, **k: next(summaries))
    settings = gp.Settings(
        out=tmp_path, bed=None, binaries={"pr": Path("/h")}, paths=[], requests=10
    )
    spec = gp.parse_paths("stream")[0]
    record = gp.mem_soak(settings, "pr", spec, "r1", windows=1, burst=2.0)
    assert record["sent"] == 20 and record["failed"] == 3 and record["script"] == counts
    assert record["warnings"] == [
        "3 failed requests",
        "2 script misses (answers served by the engine simulator)",
        "1 script counters unreadable (misses unknown)",
    ]
    soak = json.loads((tmp_path / "cells" / "pr" / "mem-stream" / "r1" / "soak.json").read_text())
    assert soak["warnings"] == record["warnings"]


def test_spread_is_max_minus_min_over_median(gp):
    assert gp.spread([100.0, 102.0, 98.0]) == pytest.approx(0.04)
    assert gp.spread([]) is None
    assert gp.spread([0.0, 0.0]) is None
    # An idle-subtracted value can sit below zero: the spread reads against the magnitude.
    assert gp.spread([-2.2, -2.0, -1.8]) == pytest.approx(0.2)


def test_script_counts_report_the_reads_that_failed_instead_of_zero_misses(gp, monkeypatch):
    import urllib.error
    from types import SimpleNamespace

    bed = SimpleNamespace(
        admin_url="http://127.0.0.1:1", cfg=SimpleNamespace(workers=2, ports={"mock_base": 40})
    )
    answers = {
        "/admin/script/grpc:40": json.dumps({"hits": 5, "misses": 1}),
        "/admin/script/all": json.dumps({"queued": 3}),
    }

    def fake_http(method, url, body=None, timeout=30):
        path = url[len(bed.admin_url) :]
        if path not in answers:
            raise urllib.error.URLError("unknown worker")
        return answers[path]

    monkeypatch.setattr(gp, "http", fake_http)
    assert gp.script_counts(bed) == {"hits": 5, "misses": 1, "unreadable": 1, "queued_left": 3}
    answers.clear()
    assert gp.script_counts(bed)["unreadable"] == 3


def test_a_failed_bed_up_takes_the_partial_bed_down(gp, tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_run(cmd, env=None, stdout=None, stderr=None, **kwargs):
        calls.append(cmd[-1])
        if cmd[-1] == "up":
            stdout.write("bed: the probe failed\n")
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(gp.subprocess, "run", fake_run)
    cfg = gp.BedConfig(
        mock_bin=Path("/m"),
        load_bin=Path("/l"),
        tokenizer="/t",
        ports={"gateway": 1, "metrics": 2, "admin": 3, "mock_base": 4},
    )
    bed = gp.Bed(cfg, Path("/h"), tmp_path / "bed")
    with pytest.raises(RuntimeError, match="bed up failed"):
        bed.up()
    assert calls == ["up", "down"]
    # A bed that is up already (its pid file under the run directory) is not this call's
    # to stop: the refusal comes before the script runs, and nothing is taken down.
    calls.clear()
    (tmp_path / "bed" / "bed").mkdir(parents=True)
    (tmp_path / "bed" / "bed" / "gateway.pid").write_text("1\n")
    with pytest.raises(RuntimeError, match="up already"):
        bed.up()
    assert calls == []
