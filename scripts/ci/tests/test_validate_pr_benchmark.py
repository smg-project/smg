"""Raw benchmark qualification must fail closed before advisory reporting."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "validate_pr_benchmark.py"
spec = importlib.util.spec_from_file_location("validate_pr_benchmark", SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def valid():
    return {
        "aggregated_metrics": {
            "num_requests": 2,
            "num_completed_requests": 2,
            "num_error_requests": 0,
            "error_codes_frequency": {},
            "error_rate": 0,
            "stats": {metric: {"mean": 1.5} for metric in validator.METRICS},
        },
        "individual_request_metrics": [
            {"error_code": None, "error_message": None},
            {"error_code": None, "error_message": None},
        ],
    }


def write(tmp_path, data, protocol="http"):
    folder = tmp_path / f"benchmark_round_robin_regular_{protocol}"
    folder.mkdir(exist_ok=True)
    path = folder / "requests.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "data", [{}, {"aggregated_metrics": {}}, {"aggregated_metrics": {"stats": {}}}, [], None]
)
def test_missing_metrics_fail_closed(tmp_path, data):
    assert not validator.qualify_file(write(tmp_path, data))["qualified"]


@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), -float("inf"), 0, -1, True, "1.5"]
)
@pytest.mark.parametrize("metric", validator.METRICS)
def test_invalid_metric_rejected(tmp_path, metric, value):
    data = valid()
    data["aggregated_metrics"]["stats"][metric]["mean"] = value
    assert not validator.qualify_file(write(tmp_path, data))["qualified"]


@pytest.mark.parametrize(
    "change",
    [
        {"num_completed_requests": 0},
        {"num_completed_requests": 3},
        {"num_requests": 3},
        {"num_error_requests": 1},
        {"num_requests": True},
        {"error_codes_frequency": {"500": 1}},
        {"error_rate": 0.5},
    ],
)
def test_inconsistent_or_failed_counts_rejected(tmp_path, change):
    data = valid()
    data["aggregated_metrics"].update(change)
    assert not validator.qualify_file(write(tmp_path, data))["qualified"]


def test_zero_requests_rejected(tmp_path):
    data = valid()
    data["aggregated_metrics"].update(num_requests=0, num_completed_requests=0)
    data["individual_request_metrics"] = []
    assert not validator.qualify_file(write(tmp_path, data))["qualified"]


def test_individual_errors_cannot_hide_behind_zero_rate(tmp_path):
    data = valid()
    data["individual_request_metrics"][0]["error_message"] = "failed"
    assert not validator.qualify_file(write(tmp_path, data))["qualified"]


def test_rate_only_cannot_prove_successful_requests(tmp_path):
    data = valid()
    data["aggregated_metrics"] = {"stats": data["aggregated_metrics"]["stats"], "error_rate": 0}
    del data["individual_request_metrics"]
    assert not validator.qualify_file(write(tmp_path, data))["qualified"]


def test_complete_two_protocols_and_cli_report(tmp_path):
    for protocol in validator.PROTOCOLS:
        write(tmp_path, valid(), protocol)
    report = tmp_path / "qualification.json"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--output", str(report)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(report.read_text())
    assert result["qualified"]
    for protocol in validator.PROTOCOLS:
        assert result["protocols"][protocol]["files"][0]["completed_requests"] == 2


def test_missing_protocol_and_invalid_additional_file_fail(tmp_path):
    write(tmp_path, valid())
    assert not validator.qualify(tmp_path)["qualified"]
    write(tmp_path, valid(), "grpc")
    folder = tmp_path / "benchmark_round_robin_regular_http"
    (folder / "unknown-schema.json").write_text("{}")
    assert not validator.qualify(tmp_path)["qualified"]


def test_metadata_ignored_but_empty_result_not_accepted(tmp_path):
    for protocol in validator.PROTOCOLS:
        path = write(tmp_path, valid(), protocol)
        (path.parent / "experiment_metadata_2026.json").write_text("{}")
        (path.parent / "gpu_utilization_2026.json").write_text("{}")
    assert validator.qualify(tmp_path)["qualified"]
    (tmp_path / "benchmark_round_robin_regular_http" / "requests.json").write_text("")
    assert not validator.qualify(tmp_path)["qualified"]


def test_threshold_miss_is_valid_data(tmp_path):
    data = copy.deepcopy(valid())
    data["aggregated_metrics"]["stats"]["ttft"]["mean"] = 200
    data["aggregated_metrics"]["stats"]["output_throughput"]["mean"] = 0.001
    assert validator.qualify_file(write(tmp_path, data))["qualified"]


def test_cli_writes_failure_report_and_exits_nonzero(tmp_path):
    write(tmp_path, valid(), "http")
    report = tmp_path / "qualification.json"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), "--output", str(report)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1, run.stderr
    result = json.loads(report.read_text())
    assert not result["qualified"]
    assert result["protocols"]["grpc"]["error"] == "no raw benchmark result JSON"
