"""Failure evidence must exclude a cell from error-free benchmark comparisons."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def summary():
    path = ROOT / "e2e_test/benchmarks/nightly_summarize.py"
    spec = importlib.util.spec_from_file_location("nightly_summary_error_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def experiment(summary, tmp_path, agg, records=None, protocol="grpc", runtime="vllm"):
    folder = tmp_path / f"nightly_llama_{protocol}_{runtime}_multi"
    folder.mkdir()
    (folder / "experiment_metadata.json").write_text(
        json.dumps({"model": "llama", "server_engine": runtime})
    )
    data = {"aggregated_metrics": {"scenario": "D(2000,200)", "num_concurrency": 128, **agg}}
    if records is not None:
        data["individual_request_metrics"] = records
    (folder / "result.json").write_text(json.dumps(data))
    return summary.parse_experiment(folder)


# Lossless projection of error_code sequences from weekly run41283282398,
# artifact13079091616: D2000_200 C128/C256 and D7800_200 C64 respectively.
@pytest.mark.parametrize(
    "scenario,concurrency,completed,errors,runs",
    [
        (
            "D(2000,200)",
            128,
            128,
            284,
            [(None, 1), (503, 168), (None, 84), (503, 2), (None, 43), (503, 114)],
        ),
        ("D(2000,200)", 256, 256, 143, [(503, 143), (None, 256)]),
        (
            "D(7800,200)",
            64,
            252,
            50,
            [(None, 43), (503, 39), (None, 11), (503, 11), (None, 198)],
        ),
    ],
)
def test_original_nov1_false_zero_cells(
    summary, tmp_path, scenario, concurrency, completed, errors, runs
):
    total = completed + errors
    identity = {"scenario": scenario, "num_concurrency": concurrency}
    records = [{"error_code": code} for code, count in runs for _ in range(count)]
    bad = experiment(
        summary,
        tmp_path,
        {
            "error_rate": 0,
            "num_requests": total,
            "num_completed_requests": completed,
            "num_error_requests": errors,
            "error_codes_frequency": {"503": errors},
            **identity,
        },
        records,
    )
    run = bad.runs[0]
    assert run.error_rate == pytest.approx(errors / total)
    assert not run.is_error_free
    assert "Reported error_rate disagrees" in run.error_note
    clean_http = experiment(summary, tmp_path, {"error_rate": 0, **identity}, protocol="http")
    clean_sg = experiment(summary, tmp_path, {"error_rate": 0, **identity}, runtime="sglang")
    assert summary.build_comparisons([bad, clean_http]) == []
    assert summary.build_runtime_comparisons([bad, clean_sg]) == []
    report = summary.generate_summary(tmp_path).markdown
    assert f"{errors}/{total} failed requests" in report
    assert f"{errors / total:.1%}" in report
    assert "Reported error_rate disagrees" in report
    assert "completed with 0% error rate" not in report


@pytest.mark.parametrize(
    "agg,records,rate",
    [
        ({"error_rate": 0}, None, 0),
        ({"error_rate": "0.25"}, None, 0.25),
        ({"num_requests": 2, "num_error_requests": 0}, None, 0),
        ({"num_completed_requests": 1, "num_error_requests": 1}, None, 0.5),
        ({"error_rate": 0}, [{"error_code": None}, {"error_code": 503}], 0.5),
        ({"num_requests": 4, "error_codes_frequency": {"503": 1}}, None, 0.25),
        ({"num_requests": 4, "num_completed_requests": 3}, None, 0.25),
    ],
)
def test_valid_current_and_legacy_metrics(summary, tmp_path, agg, records, rate):
    exp = experiment(summary, tmp_path, agg, records)
    run = exp.runs[0]
    assert run.error_rate == pytest.approx(rate)
    assert run.is_error_free == (rate == 0)


def test_zero_error_cells_keep_both_comparison_paths(summary, tmp_path):
    grpc = experiment(summary, tmp_path, {"error_rate": 0})
    http = experiment(
        summary, tmp_path, {"num_requests": 4, "num_error_requests": 0}, protocol="http"
    )
    sg = experiment(summary, tmp_path, {}, [{"error_code": None}] * 4, runtime="sglang")
    assert len(summary.build_comparisons([grpc, http])) == 1
    assert len(summary.build_runtime_comparisons([grpc, sg])) == 1
    assert summary._section_error_rates([grpc, http, sg]) == []


@pytest.mark.parametrize(
    "agg,records,note",
    [
        ({}, None, "unavailable"),
        ({"error_rate": 0, "num_error_requests": 1}, None, "attempted-request"),
        ({"num_requests": 0, "num_error_requests": 0}, None, "attempted"),
        ({"error_rate": 0, "num_requests": 0}, None, "attempted"),
        ({"error_rate": 0, "num_requests": 2, "num_error_requests": -1}, None, "Invalid"),
        ({"error_rate": 0, "num_requests": 2, "num_error_requests": True}, None, "Invalid"),
        (
            {"error_rate": 0, "num_requests": 2, "num_error_requests": 0},
            [{"error_code": 503}] * 2,
            "disagree",
        ),
        (
            {"error_rate": 0, "num_requests": 3, "num_error_requests": 1},
            [{"error_code": 503}] * 2,
            "disagree",
        ),
        (
            {
                "error_rate": 0,
                "num_requests": 2,
                "num_error_requests": 0,
                "error_codes_frequency": {"503": 1},
            },
            None,
            "disagree",
        ),
        ({"error_rate": 0, "error_codes_frequency": {"503": "1"}}, None, "Invalid"),
        ({"error_rate": 0}, [{"ttft": 1}], "Invalid"),
        ({"error_rate": "nan"}, None, "Invalid"),
        ({"error_rate": 2}, None, "Invalid"),
    ],
)
def test_missing_malformed_and_conflicting_metrics_are_not_clean(
    summary, tmp_path, agg, records, note
):
    exp = experiment(summary, tmp_path, agg, records)
    run = exp.runs[0]
    assert run.error_rate is None
    assert not run.is_error_free
    assert note in run.error_note
    report = summary.generate_summary(tmp_path).markdown
    assert "unknown" in report
    assert "completed with 0% error rate" not in report


def test_nonzero_reported_rate_cannot_be_silently_discarded(summary, tmp_path):
    exp = experiment(
        summary, tmp_path, {"error_rate": 0.5, "num_requests": 2, "num_error_requests": 0}
    )
    run = exp.runs[0]
    assert run.error_rate == 0
    assert not run.is_error_free
    assert "disagrees" in run.error_note
    assert "Reported error_rate disagrees" in summary.generate_summary(tmp_path).markdown


def test_completed_count_above_attempted_is_a_conflict(summary, tmp_path):
    exp = experiment(summary, tmp_path, {"num_requests": 100, "num_completed_requests": 120})
    run = exp.runs[0]
    assert run.error_rate is None
    assert not run.is_error_free
    assert run.error_note == "Request/error counts disagree"


@pytest.mark.parametrize(
    "agg,field",
    [
        ({"num_requests": -1}, "num_requests"),
        ({"num_error_requests": True}, "num_error_requests"),
        ({"num_completed_requests": "1"}, "num_completed_requests"),
        ({"error_codes_frequency": {"503": -1}}, "error_codes_frequency[503]"),
    ],
)
def test_invalid_count_diagnostics_name_the_field(summary, tmp_path, agg, field):
    run = experiment(summary, tmp_path, agg).runs[0]
    assert run.error_rate is None
    assert run.error_note == f"Invalid error metrics: {field}: expected a nonnegative integer"
