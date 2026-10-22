"""CPU checks for HTTP scheduler observations; no workers are launched."""

from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from infra import pd_metrics as metrics

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"


def _worker(name="decode", engine="vllm"):
    return SimpleNamespace(base_url=f"http://{name}", http_url=f"http://{name}", engine=engine)


def _text(running=(0, 0), waiting=(0, 0)):
    return "\n".join(
        f'{name}{{model_name="meta-llama/Llama-3.1-8B-Instruct",engine="{i}"}} {value}'
        for name, values in ((RUNNING, running), (WAITING, waiting))
        for i, value in enumerate(values)
    )


def test_metric_labels_timestamp_and_prefix():
    text = "\n".join(
        [
            "# HELP counter A token counter.",
            "counter_created 100",
            r'counter{engine="1",model_name="model\\path\"name\n",source="local_compute"} 2e1 123',
            'counter{source="remote",engine="0"} 5',
        ]
    )
    samples = metrics.metric_samples(text, "counter", source="local_compute")
    assert len(samples) == 1
    labels, value = next(iter(samples.items()))
    assert dict(labels) == {
        "engine": "1",
        "model_name": 'model\\path"name\n',
        "source": "local_compute",
    }
    assert value == 20


@pytest.mark.parametrize(
    "text",
    [
        "",
        "counter_other 1",
        'counter{engine="0"}',
        "counter{engine=0} 1",
        'counter{engine="0",engine="1"} 1',
        'counter{engine="0"} 1\ncounter{engine="0"} 2',
        'counter{b="2",a="1"} 1\ncounter{a="1",b="2"} 2',
        "counter NaN",
        "counter +Inf",
        "counter -Inf",
        "counter -1",
        "counter nonsense",
        'counter{engine="0"} 1 trailing',
        r'counter{engine="bad\tescape"} 0',
        "counter 0_0",
        "counter ٠",
        "counter 1e٢",
        "counter 0 1_0",
    ],
)
def test_missing_malformed_or_duplicate_metric_is_not_zero(text):
    with pytest.raises((AssertionError, ValueError)):
        metrics.metric_samples(text, "counter")


def test_required_label_must_be_exported():
    with pytest.raises(AssertionError, match="no counter sample"):
        metrics.metric_samples('counter{source="remote"} 1', "counter", source="local_compute")


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(500, text="failed", request=httpx.Request("GET", "http://decode/metrics")),
        httpx.ReadTimeout("metrics timeout"),
    ],
)
def test_metrics_http_failure_propagates(monkeypatch, failure):
    def get(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(metrics.httpx, "get", get)
    with pytest.raises(httpx.HTTPError):
        metrics.worker_metrics(_worker(), RUNNING)


@pytest.mark.parametrize(
    "running,waiting,idle",
    [((0, 0), (0, 0), True), ((0, 1), (0, 0), False), ((0, 0), (1, 0), False)],
)
def test_every_engine_series_counts_in_one_worker_scrape(monkeypatch, running, waiting, idle):
    worker = _worker()
    gateway = SimpleNamespace(prefill_workers=[], decode_workers=[worker])
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, text=_text(running, waiting), request=httpx.Request("GET", url))

    monkeypatch.setattr(metrics.httpx, "get", get)
    observed, states = metrics.vllm_work_snapshot(gateway)
    assert observed == idle
    assert calls == ["http://decode/metrics"]
    assert states == [
        {
            "worker": "http://decode",
            "num_running_reqs": sum(running),
            "num_waiting_reqs": sum(waiting),
        }
    ]


@pytest.mark.parametrize(
    "text",
    [
        _text().replace("vllm:num_requests_waiting{", "unrelated{"),
        _text().replace(
            "vllm:num_requests_waiting{model_name=", "vllm:num_requests_waiting{other="
        ),
        _text().replace("model_name=", "other="),
        _text().replace('engine="0"', 'engine=""'),
        _text().replace('engine="0"', 'engine="-1"'),
        _text().replace('engine="0"', 'other="0"'),
        _text().replace("meta-llama/Llama-3.1-8B-Instruct", ""),
    ],
)
def test_each_worker_must_export_valid_matching_scheduler_series(monkeypatch, text):
    monkeypatch.setattr(
        metrics.httpx,
        "get",
        lambda url, **kwargs: httpx.Response(200, text=text, request=httpx.Request("GET", url)),
    )
    with pytest.raises(AssertionError):
        metrics.vllm_work_snapshot(SimpleNamespace(prefill_workers=[], decode_workers=[_worker()]))


@pytest.mark.parametrize("workers", [[], [_worker(), _worker()], [_worker(engine="sglang")]])
def test_unknown_empty_or_duplicate_workers_fail(workers):
    with pytest.raises(AssertionError):
        metrics.vllm_work_snapshot(SimpleNamespace(prefill_workers=[], decode_workers=workers))


def test_missing_metric_on_one_worker_does_not_hide_behind_idle_peer(monkeypatch):
    monkeypatch.setattr(
        metrics.httpx,
        "get",
        lambda url, **kwargs: httpx.Response(
            200,
            text=_text() if "prefill" in url else f"{RUNNING} 0",
            request=httpx.Request("GET", url),
        ),
    )
    with pytest.raises(AssertionError, match=f"no {WAITING} sample"):
        metrics.vllm_work_snapshot(
            SimpleNamespace(prefill_workers=[_worker("prefill")], decode_workers=[_worker()])
        )


def test_overflowed_scheduler_total_fails(monkeypatch):
    first = (("model_name", "model"), ("engine", "0"))
    second = (("model_name", "model"), ("engine", "1"))
    monkeypatch.setattr(
        metrics,
        "worker_metrics",
        lambda *args: {
            RUNNING: {first: 1e308, second: 1e308},
            WAITING: {first: 0, second: 0},
        },
    )
    with pytest.raises(AssertionError, match="non-finite scheduler totals"):
        metrics.vllm_work_snapshot(SimpleNamespace(prefill_workers=[], decode_workers=[_worker()]))


@pytest.mark.parametrize(
    "lines",
    [
        [],
        ["data: [DONE]"],
        ['data: {"error":{"message":"bad"}}'],
        ['data: {"choices":[{"delta":{"role":"assistant"}}]}'],
        ['data: {"choices":[{"delta":{"content":""}}]}'],
        ['data: {"choices":[{"delta":{"content":"done"},"finish_reason":"length"}]}'],
        ["data: malformed"],
    ],
)
def test_stream_without_unfinished_generated_text_fails(lines):
    with pytest.raises((AssertionError, ValueError)):
        metrics.wait_for_generated_text(lines)


@pytest.mark.parametrize("key", ["content", "reasoning_content", "reasoning"])
def test_stream_waits_past_role_chunk_for_generated_text(key):
    metrics.wait_for_generated_text(
        [
            ": heartbeat",
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            f'data: {{"choices":[{{"delta":{{"{key}":"one token"}}}}]}}',
        ]
    )


@pytest.mark.parametrize("runtime", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "method",
    ["test_abandoned_streams_free_both_sides", "test_aborts_during_prefill_free_both_sides"],
)
def test_ordinary_http_cancellation_remains_skipped(monkeypatch, runtime, method):
    from router import test_pd_topologies as topology

    monkeypatch.setattr(topology, "get_runtime", lambda: runtime)
    monkeypatch.setattr(topology, "_moriio_lane", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="load reports need gRPC workers"):
        getattr(topology.TestPDTopology(), method)(("pd_http", "model", None, None))


@pytest.mark.parametrize(
    "kind,status,generated",
    [
        ("stream", 200, True),
        ("stream", 200, False),
        ("stream", 500, True),
        ("prefill", 200, True),
        ("prefill", 400, True),
        ("prefill", 500, True),
    ],
)
def test_cancel_test_rejects_unstarted_or_http_error_requests(monkeypatch, kind, status, generated):
    from router import test_pd_topologies as topology

    checks = []
    monkeypatch.setattr(topology, "_assert_fleet_idle_within", lambda *a: checks.append("idle"))
    monkeypatch.setattr(topology, "_wait_until_served", lambda *a, **k: checks.append("served"))
    if kind == "stream":

        @contextmanager
        def stream(*args, **kwargs):
            data = '{"choices":[{"delta":{"content":"token"}}]}' if generated else "[DONE]"
            yield httpx.Response(status, text=f"data: {data}\n\n")

        monkeypatch.setattr(topology.httpx, "stream", stream)
        method = "test_abandoned_streams_free_both_sides"
    else:
        monkeypatch.setattr(
            topology.httpx,
            "post",
            lambda *a, **k: httpx.Response(status, text="response"),
        )
        method = "test_aborts_during_prefill_free_both_sides"
    run = getattr(topology.TestPDTopology(), method)
    setup = ("pd_grpc", "model", None, SimpleNamespace(base_url="http://gateway"))
    if status == 200 and generated:
        run(setup)
        assert checks == ["idle", "served"]
    else:
        with pytest.raises(AssertionError, match="failed before they could be abandoned"):
            run(setup)
        assert checks == []
