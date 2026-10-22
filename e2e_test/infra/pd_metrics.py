"""Read scheduler work from HTTP workers without treating missing data as idle."""

from __future__ import annotations

import json
import math
import re

import httpx

_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z_0-9]*)=("(?:[^"\\]|\\[\\"n])*")\s*(?:,|$)')
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"


def metric_samples(text: str, name: str, **required: str) -> dict[tuple, float]:
    pattern = re.compile(rf"{re.escape(name)}(?:\{{(.*)\}})?\s+({_NUMBER})(?:\s+[+-]?[0-9]+)?")
    samples = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(name) or line[len(name) : len(name) + 1] not in ("", "{", " ", "\t"):
            continue
        match = pattern.fullmatch(line)
        assert match, f"malformed {name} sample: {line}"
        raw_labels, raw_value = match.groups()
        labels, offset = {}, 0
        while raw_labels and offset < len(raw_labels):
            label = _LABEL.match(raw_labels, offset)
            assert label, f"malformed {name} labels: {line}"
            key, value = label.groups()
            assert key not in labels, f"duplicate {name} label: {line}"
            labels[key] = json.loads(value)
            offset = label.end()
        value = float(raw_value)
        assert math.isfinite(value) and value >= 0, f"invalid {name} value: {line}"
        identity = tuple(sorted(labels.items()))
        assert identity not in samples, f"duplicate {name} series: {line}"
        samples[identity] = value
    samples = {
        key: value for key, value in samples.items() if required.items() <= dict(key).items()
    }
    assert samples, f"no {name} sample with {required}"
    return samples


def worker_metrics(worker, *names: str, **labels: str) -> dict[str, dict[tuple, float]]:
    response = httpx.get(f"{worker.http_url}/metrics", timeout=5.0)
    response.raise_for_status()
    return {name: metric_samples(response.text, name, **labels) for name in names}


def vllm_work_snapshot(gateway) -> tuple[bool, list[dict]]:
    """One scrape per worker; zero gauges do not prove KV memory was freed."""
    workers = gateway.prefill_workers + gateway.decode_workers
    assert workers, "no PD workers to observe"
    assert len({w.base_url for w in workers}) == len(workers), "duplicate PD worker URL"
    names = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    states = []
    for worker in workers:
        assert worker.engine == "vllm", "HTTP scheduler observation requires vLLM metrics"
        samples = worker_metrics(worker, *names)
        assert all(samples[name].keys() == samples[names[0]].keys() for name in names), (
            f"{worker.base_url} exports different scheduler series"
        )
        for series in samples[names[0]]:
            labels = dict(series)
            assert labels.get("model_name", "").strip() and re.fullmatch(
                r"0|[1-9][0-9]*", labels.get("engine", "")
            ), f"{worker.base_url} exports invalid vLLM engine labels: {labels}"
        states.append(
            {
                "worker": worker.base_url,
                "num_running_reqs": sum(samples[names[0]].values()),
                "num_waiting_reqs": sum(samples[names[1]].values()),
            }
        )
    assert all(
        math.isfinite(s[key]) for s in states for key in ("num_running_reqs", "num_waiting_reqs")
    ), f"non-finite scheduler totals: {states}"
    return all(s["num_running_reqs"] == s["num_waiting_reqs"] == 0 for s in states), states
