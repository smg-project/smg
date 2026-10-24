"""The Python servicer's Prometheus metrics: the families the Rust servicer
exposes, with the same names, labels, bucket edges, HELP lines and meaning
(the Rust module is read from the checkout and compared), recorded through
the servicers' own Generate paths over engine stand-ins, served over HTTP by
the endpoint the launchers bind before the engine starts, and accepted by
the launchers under the Python implementation."""

from __future__ import annotations

import asyncio
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from socketserver import BaseServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest
from smg_grpc_servicer import metrics, rust_lifecycle
from smg_grpc_servicer.metrics import NO_METRICS, MetricsEndpoint, ServicerMetrics
from test_sglang_server_info import servicer_mod as _sglang_servicer_fixture
from test_sglang_server_startup import _serving_server as _sglang_serving_server
from test_sglang_server_startup import server_mod as _sglang_server_fixture
from test_tokenspeed_server_lifecycle import _stop_event_of
from test_tokenspeed_server_lifecycle import _wire as _tokenspeed_wire
from test_tokenspeed_server_lifecycle import server_mod as _tokenspeed_server_fixture

# The SGLang servicer module loaded over engine stand-ins (the server-info
# test's fixture), under a name of its own beside the TokenSpeed one, and the
# two engines' server modules over their stand-ins (the start-up and
# lifecycle tests' fixtures).
sglang_servicer_mod = _sglang_servicer_fixture
sglang_server_mod = _sglang_server_fixture
tokenspeed_server_mod = _tokenspeed_server_fixture

RUST_MODULE = Path(__file__).parents[2] / "crates" / "engine_servicer" / "src" / "metrics.rs"

# The type of every family, as the Rust recorder registers it.
TYPES = {
    "REQUESTS_TOTAL": "counter",
    "REQUESTS_REFUSED_TOTAL": "counter",
    "REQUESTS_ACTIVE": "gauge",
    "REQUEST_ERRORS_TOTAL": "counter",
    "PROMPT_TOKENS_TOTAL": "counter",
    "PROMPT_TOKENS_CACHED_TOTAL": "counter",
    "GENERATION_TOKENS_TOTAL": "counter",
    "STRING_STOPS_TOTAL": "counter",
    "TIME_TO_FIRST_TOKEN_SECONDS": "histogram",
    "E2E_REQUEST_LATENCY_SECONDS": "histogram",
    "REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS": "histogram",
    "INTER_TOKEN_LATENCY_SECONDS": "histogram",
    "REQUEST_PROMPT_TOKENS": "histogram",
    "REQUEST_GENERATION_TOKENS": "histogram",
    "REQUEST_PARAMS_MAX_TOKENS": "histogram",
    "REQUEST_PARAMS_N": "histogram",
    "STATE": "gauge",
    "START_TIME_SECONDS": "gauge",
    "METRICS_DROPPED_TOTAL": "counter",
    "METRICS_SCRAPE_SECONDS": "gauge",
    "METRICS_SCRAPES_TOTAL": "counter",
}


def _rust_source() -> str:
    if not RUST_MODULE.exists():
        pytest.skip("the Rust servicer's metrics module is not in this tree")
    return RUST_MODULE.read_text(encoding="utf-8")


def _sample(body: str, name: str, *labels: str) -> float | None:
    """The value of the first sample line of ``name`` carrying every label."""
    prefix = f"{name}{{"
    for line in body.splitlines():
        if line.startswith(prefix) and all(label in line for label in labels):
            return float(line.rsplit(" ", 1)[1])
    return None


def _families(body: str) -> dict[str, str]:
    """Every family the exposition declares, with its type."""
    return dict(re.findall(r"^# TYPE (\S+) (\S+)$", body, re.M))


class _Clock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _registry(**facts) -> tuple[ServicerMetrics, _Clock]:
    clock = _Clock()
    registry = ServicerMetrics("served", "test", clock=clock)
    registry.engine_facts(
        **{"ranks": 1, "max_model_len": 100, "matches_string_stops": True, **facts}
    )
    return registry, clock


# ---------------------------------------------------------------------------
# The contract: the Rust module's names, labels, edges and HELP lines
# ---------------------------------------------------------------------------


def test_the_family_names_are_the_rust_servicers():
    source = _rust_source()
    rust = dict(
        re.findall(
            r'pub\(crate\) const ([A-Z0-9_]+): &str =\s*"(smg_servicer_[a-z0-9_]+)";', source
        )
    )
    assert rust == metrics.NAMES
    assert set(TYPES) == set(metrics.NAMES)


def test_the_bucket_edges_and_label_values_are_the_rust_servicers():
    source = _rust_source()

    def edges(name: str) -> tuple[float, ...]:
        block = re.search(rf"const {name}: &\[f64\] = &\[(.*?)\];", source, re.S).group(1)
        return tuple(float(value) for value in re.findall(r"[0-9.]+", block))

    def labels(enum: str) -> tuple[str, ...]:
        block = re.search(rf"impl {enum} \{{(.*?)\n\}}", source, re.S).group(1)
        return tuple(re.findall(r'Self::\w+ => "([a-z_]+)"', block))

    assert edges("TTFT_BUCKETS") == metrics.TTFT_BUCKETS
    assert edges("INTER_TOKEN_BUCKETS") == metrics.INTER_TOKEN_BUCKETS
    assert edges("REQUEST_LATENCY_BUCKETS") == metrics.REQUEST_LATENCY_BUCKETS
    assert edges("PARAMS_N_BUCKETS") == metrics.PARAMS_N_BUCKETS
    assert labels("FinishReason") == metrics.FINISH_REASONS
    assert labels("Refusal") == metrics.REFUSALS
    assert labels("ServicerState") == metrics.STATES
    default = re.search(r"const DEFAULT_MAX_MODEL_LEN: u32 = ([0-9_]+);", source).group(1)
    assert int(default.replace("_", "")) == metrics.DEFAULT_MAX_MODEL_LEN
    # The token-count series, as the Rust test pins it.
    assert metrics.one_two_five_buckets(100) == (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
    assert metrics.one_two_five_buckets(0) == (1.0,)
    assert len(metrics.one_two_five_buckets(4096)) == 11


def test_the_help_lines_are_the_rust_servicers():
    source = _rust_source()
    described = {}
    for key, text in re.findall(
        r'describe_\w+\(\s*names::([A-Z0-9_]+),\s*"((?:[^"\\]|\\.)*)"', source, re.S
    ):
        described[key] = re.sub(r"\\\n\s*", "", text)
    assert described == metrics.HELP


# ---------------------------------------------------------------------------
# What the request path records
# ---------------------------------------------------------------------------


def test_a_request_records_on_its_pins_place_among_the_managed_ranks():
    # A servicer serving rank 2 alone (a load balancer's slice): its one
    # series takes the unpinned requests and the pin 2, labelled by its
    # place; a pin outside the slice is a rank beyond the width.
    assert metrics.rank_among([2], None) is None
    assert metrics.rank_among([2], 2) == 0
    assert metrics.rank_among([2], 3) == 1
    assert metrics.rank_among([0, 1, 2, 3], 2) == 2
    registry, clock = _registry(ranks=1)
    for pinned, prompt_tokens in ((2, 3), (None, 4)):
        observed = registry.request(
            rank=metrics.rank_among([2], pinned), prompt_tokens=prompt_tokens
        )
        observed.chunk(1, prompt_tokens, 0)
        observed.complete("stop", prompt_tokens, 0)
        observed.end()
    body = registry.render(metrics.SERVING)
    assert _sample(body, metrics.NAMES["PROMPT_TOKENS_TOTAL"], 'rank="0"') == 7
    assert 'rank="other"' not in body


def test_a_request_records_its_families_at_its_end():
    registry, clock = _registry()
    observed = registry.request(rank=0, prompt_tokens=3, max_tokens=8, n=1, arrival=clock.now)
    clock.now += 0.5
    observed.chunk(1, 3, 1)
    clock.now += 0.2
    observed.chunk(2, 3, 1)
    observed.complete("length", 3, 1)
    clock.now += 0.3
    observed.end()
    body = registry.render(metrics.SERVING)

    n = metrics.NAMES
    assert _sample(body, n["REQUESTS_TOTAL"], 'finish_reason="length"') == 1
    assert _sample(body, n["REQUESTS_TOTAL"], 'finish_reason="stop"') == 0
    assert _sample(body, n["PROMPT_TOKENS_TOTAL"], 'rank="0"') == 3
    assert _sample(body, n["PROMPT_TOKENS_CACHED_TOTAL"], 'rank="0"') == 1
    assert _sample(body, n["GENERATION_TOKENS_TOTAL"], 'rank="0"') == 3
    assert _sample(body, n["STRING_STOPS_TOTAL"], 'rank="0"') == 0
    assert _sample(body, n["REQUESTS_ACTIVE"]) == 0
    # The first output 0.5 s after the arrival lands in le="0.5" (inclusive).
    assert _sample(body, n["TIME_TO_FIRST_TOKEN_SECONDS"] + "_bucket", 'le="0.25"') == 0
    assert _sample(body, n["TIME_TO_FIRST_TOKEN_SECONDS"] + "_bucket", 'le="0.5"') == 1
    assert _sample(body, n["TIME_TO_FIRST_TOKEN_SECONDS"] + "_count") == 1
    # 0.2 s between the chunks over the second chunk's 2 tokens.
    assert _sample(body, n["INTER_TOKEN_LATENCY_SECONDS"] + "_count") == 1
    assert _sample(body, n["INTER_TOKEN_LATENCY_SECONDS"] + "_sum") == pytest.approx(0.1)
    # The mean over the 2 tokens after the first, between the first and last output.
    assert _sample(body, n["REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS"] + "_sum") == pytest.approx(0.1)
    assert _sample(body, n["E2E_REQUEST_LATENCY_SECONDS"] + "_sum") == pytest.approx(1.0)
    assert _sample(body, n["E2E_REQUEST_LATENCY_SECONDS"] + "_bucket", 'le="1"') == 1
    assert _sample(body, n["REQUEST_PROMPT_TOKENS"] + "_sum") == 3
    assert _sample(body, n["REQUEST_GENERATION_TOKENS"] + "_sum") == 3
    assert _sample(body, n["REQUEST_PARAMS_MAX_TOKENS"] + "_sum") == 8
    assert _sample(body, n["REQUEST_PARAMS_N"] + "_sum") == 1
    assert _sample(body, n["STATE"], 'state="serving"') == 1
    assert _sample(body, n["STATE"], 'state="starting"') == 0
    assert "# HELP smg_servicer_requests_total " in body
    # The token-count edges follow the context length (100): the 1-2-5 series up to it.
    assert _sample(body, n["REQUEST_PROMPT_TOKENS"] + "_bucket", 'le="100"') == 1
    assert 'le="200"' not in body


def test_every_family_renders_with_the_rust_names_types_and_labels():
    registry, clock = _registry()
    observed = registry.request(rank=0, prompt_tokens=3, max_tokens=8, n=1)
    observed.chunk(1, 3, 1)
    observed.complete("stop", 3, 1)
    observed.end()
    failed = registry.request(rank=0)
    failed.failed(grpc.StatusCode.INTERNAL)
    failed.end()
    registry.request_not_started(grpc.StatusCode.INVALID_ARGUMENT, refusal="invalid_argument")
    registry.scrape(metrics.SERVING)
    body = registry.render(metrics.SERVING)

    families = _families(body)
    assert families == {metrics.NAMES[key]: kind for key, kind in TYPES.items()}
    base = 'model="served",wire="test"'
    for line in body.splitlines():
        if line and not line.startswith("#"):
            assert base in line, line
    assert _sample(body, metrics.NAMES["REQUEST_ERRORS_TOTAL"], 'code="internal"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="error"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_REFUSED_TOTAL"], 'reason="invalid_argument"') == 1
    for reason in metrics.REFUSALS:
        assert (
            _sample(body, metrics.NAMES["REQUESTS_REFUSED_TOTAL"], f'reason="{reason}"') is not None
        )
    for state in metrics.STATES:
        assert _sample(body, metrics.NAMES["STATE"], f'state="{state}"') is not None
    assert _sample(body, metrics.NAMES["METRICS_SCRAPES_TOTAL"], 'code="200"') == 1
    assert (
        _sample(body, metrics.NAMES["METRICS_DROPPED_TOTAL"], 'family="inter_token_latency"') == 0
    )


def test_a_servicer_whose_engine_matches_string_stops_has_no_string_stop_family():
    registry, _ = _registry(matches_string_stops=False)
    registry.request(rank=0).end()
    assert metrics.NAMES["STRING_STOPS_TOTAL"] not in _families(registry.render(metrics.SERVING))


def test_a_fan_out_counts_one_request_with_the_last_reason_and_an_abort_winning():
    registry, clock = _registry()
    observed = registry.request(rank=0, prompt_tokens=3, n=2)
    observed.chunk(1, 3, 0, index=0)
    observed.chunk(1, 3, 0, index=1)
    observed.complete("stop", 3, 0, index=0)
    observed.complete("length", 3, 0, index=1)
    observed.end()
    observed = registry.request(rank=0, prompt_tokens=3, n=2)
    observed.ended("abort")
    observed.complete("stop", 3, 0, index=1)
    observed.end()
    body = registry.render(metrics.SERVING)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="length"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="abort"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="stop"') == 0
    assert _sample(body, metrics.NAMES["TIME_TO_FIRST_TOKEN_SECONDS"] + "_count") == 2
    assert _sample(body, metrics.NAMES["REQUEST_PARAMS_N"] + "_sum") == 4


def test_a_request_left_open_counts_as_an_abort_or_as_cut_while_draining():
    registry, _ = _registry()
    registry.request(rank=0).end()
    registry.draining = True
    registry.request(rank=0).end()
    body = registry.render(metrics.DRAINING)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="abort"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="cut"') == 1


def test_the_prompt_counts_once_at_the_first_non_zero_report():
    registry, _ = _registry()
    observed = registry.request(rank=0, prompt_tokens=3)
    observed.chunk(1, 0, 0)
    observed.chunk(1, 5, 2)
    observed.chunk(1, 5, 2)
    observed.complete("stop", 5, 2)
    observed.end()
    body = registry.render(metrics.SERVING)
    assert _sample(body, metrics.NAMES["PROMPT_TOKENS_TOTAL"]) == 5
    assert _sample(body, metrics.NAMES["PROMPT_TOKENS_CACHED_TOTAL"]) == 2
    assert _sample(body, metrics.NAMES["REQUEST_PROMPT_TOKENS"] + "_sum") == 5


def test_a_failure_before_the_submission_is_a_refusal_or_an_error_after_it_a_failed_stream():
    registry, _ = _registry()
    registry.failed(None, grpc.StatusCode.UNAVAILABLE, refusal="media")
    registry.failed(None, grpc.StatusCode.INVALID_ARGUMENT, refusal="invalid_argument")
    registry.failed(None, grpc.StatusCode.UNAVAILABLE)
    observed = registry.request(rank=0)
    registry.failed(observed, grpc.StatusCode.FAILED_PRECONDITION)
    observed.end()
    body = registry.render(metrics.SERVING)
    refused = metrics.NAMES["REQUESTS_REFUSED_TOTAL"]
    errors = metrics.NAMES["REQUEST_ERRORS_TOTAL"]
    assert _sample(body, refused, 'reason="media"') == 1
    assert _sample(body, refused, 'reason="invalid_argument"') == 1
    assert _sample(body, errors, 'code="unavailable"') == 1
    assert _sample(body, errors, 'code="failed_precondition"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="error"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="abort"') == 0


def test_the_rank_label_is_the_pin_else_the_only_rank_else_other():
    one, _ = _registry(ranks=1)
    one.request(rank=None).end()
    body = one.render(metrics.SERVING)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'rank="0"', 'finish_reason="abort"') == 1
    assert 'rank="other"' not in body
    many, _ = _registry(ranks=2)
    many.request(rank=1).end()
    many.request(rank=None).end()
    many.request(rank=7).end()
    body = many.render(metrics.SERVING)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'rank="1"', 'finish_reason="abort"') == 1
    assert (
        _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'rank="other"', 'finish_reason="abort"') == 2
    )


def test_without_an_endpoint_nothing_is_recorded():
    observed = NO_METRICS.request(rank=0, prompt_tokens=3, max_tokens=8, n=1)
    observed.chunk(1, 3, 1)
    observed.complete("stop", 3, 1)
    observed.failed(grpc.StatusCode.INTERNAL)
    observed.end()
    NO_METRICS.failed(None, grpc.StatusCode.INTERNAL, refusal="media")
    NO_METRICS.request_not_started(grpc.StatusCode.INTERNAL)
    NO_METRICS.engine_facts(ranks=2)


def test_the_active_gauge_follows_the_requests_in_flight():
    registry, _ = _registry()
    first = registry.request(rank=0)
    second = registry.request(rank=0)
    assert _sample(registry.render(metrics.SERVING), metrics.NAMES["REQUESTS_ACTIVE"]) == 2
    first.end()
    assert _sample(registry.render(metrics.SERVING), metrics.NAMES["REQUESTS_ACTIVE"]) == 1
    second.end()
    assert _sample(registry.render(metrics.SERVING), metrics.NAMES["REQUESTS_ACTIVE"]) == 0


# ---------------------------------------------------------------------------
# The endpoint: the listener, the lifecycle state, the reference parser
# ---------------------------------------------------------------------------


# A local listener is reached directly, never through a proxy the
# environment may name (an `http_proxy` without `::1` in `no_proxy` would
# send the IPv6 case there).
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get(address: str, path: str = "/metrics"):
    return _DIRECT.open(f"http://{address}{path}", timeout=5)


def test_the_endpoint_serves_the_exposition_and_reads_the_lifecycle():
    endpoint = MetricsEndpoint("127.0.0.1:0", model="served", wire="test").start()
    try:
        assert endpoint.address != "127.0.0.1:0"
        response = _get(endpoint.address)
        assert response.status == 200
        assert response.headers["Content-Type"] == metrics.CONTENT_TYPE
        body = response.read().decode()
        # Before the servicer exists: starting, with the process families only.
        assert _sample(body, metrics.NAMES["STATE"], 'state="starting"') == 1
        assert metrics.NAMES["STATE"] in _families(body)
        assert metrics.NAMES["REQUESTS_TOTAL"] not in _families(body)

        endpoint.metrics.engine_facts(ranks=1, max_model_len=100)
        state = {"now": metrics.WARMING_UP}
        endpoint.serve_from(lambda: state["now"])
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["STATE"], 'state="warming_up"') == 1
        assert metrics.NAMES["REQUESTS_TOTAL"] in _families(body)
        state["now"] = metrics.ENGINE_DOWN
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["STATE"], 'state="engine_down"') == 1
        endpoint.note_draining()
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["STATE"], 'state="draining"') == 1
        assert _sample(body, metrics.NAMES["METRICS_SCRAPES_TOTAL"], 'code="200"') == 3

        with pytest.raises(urllib.error.HTTPError) as refused:
            _get(endpoint.address, "/other")
        assert refused.value.code == 404
    finally:
        endpoint.stop()
    with pytest.raises(urllib.error.URLError):
        _get(endpoint.address)


def test_a_scrape_failure_answers_500_once_and_is_counted(monkeypatch):
    endpoint = MetricsEndpoint("127.0.0.1:0", model="served", wire="test").start()
    try:
        render = endpoint.metrics.render

        def failing(state):
            monkeypatch.setattr(endpoint.metrics, "render", render)
            raise RuntimeError("one bad render")

        monkeypatch.setattr(endpoint.metrics, "render", failing)
        with pytest.raises(urllib.error.HTTPError) as failed:
            _get(endpoint.address)
        assert failed.value.code == 500
        # A scrape counts itself once it rendered: the next one shows it.
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["METRICS_SCRAPES_TOTAL"], 'code="500"') == 1
        assert _sample(body, metrics.NAMES["METRICS_SCRAPES_TOTAL"], 'code="200"') == 0
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["METRICS_SCRAPES_TOTAL"], 'code="200"') == 1
    finally:
        endpoint.stop()


def test_a_raising_lifecycle_probe_is_a_counted_500_too():
    # The probe reads engine objects from the listener thread (a closed
    # process object raises): it is read under the scrape's own guard, so
    # the client gets the counted 500, not a closed connection.
    endpoint = MetricsEndpoint("127.0.0.1:0", model="served", wire="test").start()
    try:

        def closed_process() -> str:
            raise ValueError("process object is closed")

        endpoint.serve_from(closed_process)
        with pytest.raises(urllib.error.HTTPError) as failed:
            _get(endpoint.address)
        assert failed.value.code == 500
        endpoint.serve_from(lambda: metrics.SERVING)
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["METRICS_SCRAPES_TOTAL"], 'code="500"') == 1
        assert _sample(body, metrics.NAMES["STATE"], 'state="serving"') == 1
    finally:
        endpoint.stop()


def test_an_ipv6_bind_address_binds_bracketed():
    try:
        endpoint = MetricsEndpoint("[::1]:0", model="served", wire="test").start()
    except OSError:
        pytest.skip("no IPv6 loopback on this host")
    try:
        assert endpoint.address.startswith("[::1]:")
        assert _get(endpoint.address).status == 200
    finally:
        endpoint.stop()


def test_the_exposition_parses_with_the_reference_client():
    parser = pytest.importorskip("prometheus_client.parser")
    registry, _ = _registry()
    observed = registry.request(rank=0, prompt_tokens=3, max_tokens=8, n=1)
    observed.chunk(2, 3, 1)
    observed.complete("stop", 3, 1)
    observed.end()
    registry.request_not_started(grpc.StatusCode.INTERNAL)
    families = {
        # The parser names a counter family without its `_total` suffix.
        (family.name + "_total" if family.type == "counter" else family.name): family
        for family in parser.text_string_to_metric_families(registry.render(metrics.SERVING))
    }
    assert set(families) == set(metrics.NAMES.values())
    for key, kind in TYPES.items():
        assert families[metrics.NAMES[key]].type == kind, key
    ttft = families[metrics.NAMES["TIME_TO_FIRST_TOKEN_SECONDS"]]
    edges = [
        float(sample.labels["le"])
        for sample in ttft.samples
        if sample.name.endswith("_bucket") and sample.labels["le"] != "+Inf"
    ]
    assert tuple(edges) == metrics.TTFT_BUCKETS


# ---------------------------------------------------------------------------
# The launchers: the port is served under the Python implementation
# ---------------------------------------------------------------------------


def test_the_refusal_of_a_metrics_port_under_the_python_servicer_is_gone():
    assert not hasattr(rust_lifecycle, "check_metrics_port_needs_rust")


def test_start_metrics_endpoint_binds_from_the_variable_and_refuses_a_busy_port():
    args = SimpleNamespace(host="127.0.0.1", port=50051)
    assert (
        rust_lifecycle.start_metrics_endpoint(
            args, port_env="X_PORT", host_env="X_HOST", model="m", wire="w", environ={}
        )
        is None
    )
    endpoint = rust_lifecycle.start_metrics_endpoint(
        args, port_env="X_PORT", host_env="X_HOST", model="m", wire="w", environ={"X_PORT": "0"}
    )
    try:
        assert endpoint is not None
        host, port = metrics.split_bind_address(endpoint.address)
        assert (host, port > 0) == ("127.0.0.1", True)
        assert _get(endpoint.address).status == 200
        with pytest.raises(rust_lifecycle.ServicerConfigError, match="cannot bind"):
            rust_lifecycle.start_metrics_endpoint(
                args,
                port_env="X_PORT",
                host_env="X_HOST",
                model="m",
                wire="w",
                environ={"X_PORT": str(port)},
            )
    finally:
        endpoint.stop()


def test_the_vllm_switch_binds_the_endpoint_before_the_launcher_runs_and_stops_it_after(
    monkeypatch, tmp_path
):
    """The launcher's own coroutine finds the endpoint bound (`starting`),
    the servicer it builds reaches it through `launcher_endpoint`, and the
    endpoint closes when the coroutine returns."""
    from smg_grpc_servicer.vllm import launcher_switch

    monkeypatch.setenv("SMG_VLLM_SERVICER_METRICS_PORT", "0")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", str(tmp_path))
    seen = {}

    async def serve_grpc(args):
        endpoint = metrics.launcher_endpoint()
        seen["address"] = endpoint.address
        seen["body"] = _get(endpoint.address).read().decode()
        seen["model"] = endpoint.metrics.model
        return "served"

    switched = launcher_switch.with_servicer_switch(serve_grpc)
    args = SimpleNamespace(
        host="127.0.0.1", port=50051, model="org/m", served_model_name=["served-m"]
    )
    assert asyncio.run(switched(args)) == "served"
    assert _sample(seen["body"], metrics.NAMES["STATE"], 'state="starting"') == 1
    assert seen["model"] == "served-m"
    with pytest.raises(urllib.error.URLError):
        _get(seen["address"])
    metrics.remember_launcher_endpoint(None)


def _listener_thread_stays_real(monkeypatch, real_thread):
    """The server wirings stand the warm-up thread in with a mock on the
    threading module itself; the metrics listener's own threads (its serving
    thread, one per connection) stay real."""
    fake_thread = threading.Thread

    def thread(*args, **kwargs):
        listener = getattr(kwargs.get("target"), "__self__", None)
        if kwargs.get("name") == "smg-servicer-metrics" or isinstance(listener, BaseServer):
            return real_thread(*args, **kwargs)
        return fake_thread(*args, **kwargs)

    monkeypatch.setattr(threading, "Thread", thread)


def _recording_start(monkeypatch, server_mod, seen):
    """`start_metrics_endpoint` as the server module calls it, keeping the
    endpoint and a scrape taken the moment it was bound."""
    start = server_mod.start_metrics_endpoint

    def start_and_scrape(*args, **kwargs):
        endpoint = start(*args, **kwargs)
        seen["endpoint"] = endpoint
        seen["at_bind"] = _get(endpoint.address).read().decode()
        return endpoint

    monkeypatch.setattr(server_mod, "start_metrics_endpoint", start_and_scrape)


def _assert_served_lifecycle(seen, wire: str):
    n = metrics.NAMES
    # Bound before the engine: `starting`, the process families only.
    assert _sample(seen["at_bind"], n["STATE"], 'state="starting"') == 1
    assert n["REQUESTS_TOTAL"] not in _families(seen["at_bind"])
    # Serving: the engine's facts completed the families; no string-stop
    # family, the engine matches its own.
    assert _sample(seen["serving"], n["STATE"], 'state="serving"') == 1
    assert f'model="served",wire="{wire}",rank="0"' in seen["serving"]
    assert n["REQUESTS_TOTAL"] in _families(seen["serving"])
    assert n["STRING_STOPS_TOTAL"] not in _families(seen["serving"])
    # The shutdown: the drain noted, the listener closed.
    assert seen["endpoint"].metrics.draining
    with pytest.raises(urllib.error.URLError):
        _get(seen["endpoint"].address)


def test_the_tokenspeed_server_binds_the_endpoint_before_the_engine_and_closes_it_after(
    monkeypatch, tokenspeed_server_mod
):
    """The server binds the listener before its engine launches, hands the
    servicer the registry, completes the families with the engine's facts
    and serves the lifecycle, notes the drain at the shutdown and closes."""
    server_mod = tokenspeed_server_mod
    events: list = []
    real_thread = threading.Thread
    server_args = _tokenspeed_wire(monkeypatch, server_mod, events)
    _listener_thread_stays_real(monkeypatch, real_thread)
    server_args.served_model_name = "served"
    monkeypatch.setenv("SMG_TOKENSPEED_SERVICER_METRICS_PORT", "0")

    class Health(server_mod.TokenSpeedHealthServicer):
        serving = True  # the warm-up ran

    monkeypatch.setattr(server_mod, "TokenSpeedHealthServicer", Health)
    seen: dict = {}
    _recording_start(monkeypatch, server_mod, seen)

    async def exercise():
        task = asyncio.create_task(server_mod.serve_grpc(server_args))
        while "server-start" not in events:
            if task.done():
                task.result()
            await asyncio.sleep(0)
        seen["serving"] = _get(seen["endpoint"].address).read().decode()
        _stop_event_of(task).set()
        return await task

    assert asyncio.run(exercise()) is None
    _assert_served_lifecycle(seen, "tokenspeed")
    servicer = server_mod.TokenSpeedSchedulerServicer.instances[-1]
    assert servicer.kwargs["metrics"] is seen["endpoint"].metrics


def test_the_sglang_server_binds_the_endpoint_before_the_schedulers_and_closes_it_after(
    monkeypatch, sglang_server_mod
):
    """The same lifecycle on the SGLang server: bound before the schedulers
    launch, the servicer built with the registry, the families completed,
    the drain noted, the listener closed."""
    server_mod = sglang_server_mod
    events: list = []
    real_thread = threading.Thread
    server_args = _sglang_serving_server(monkeypatch, server_mod, events, scheduler_exited=False)
    _listener_thread_stays_real(monkeypatch, real_thread)
    server_args.served_model_name = "served"
    monkeypatch.setenv("SMG_SGLANG_SERVICER_METRICS_PORT", "0")
    # Scheduler processes that are alive, a health service past its warm-up.
    procs = [SimpleNamespace(is_alive=lambda: True)]

    async def launch(**_kwargs):
        return {}, object(), procs, object()

    monkeypatch.setattr(server_mod, "_launch_scheduler_with_request_manager", launch)
    server_mod.SGLangHealthServicer.return_value.serving = True
    seen: dict = {}
    _recording_start(monkeypatch, server_mod, seen)

    async def scrape_then_stop(*_args):
        # The server is serving when it waits on its schedulers.
        seen["serving"] = _get(seen["endpoint"].address).read().decode()
        return False

    monkeypatch.setattr(
        server_mod, "wait_for_scheduler_shutdown", AsyncMock(side_effect=scrape_then_stop)
    )
    assert asyncio.run(server_mod.serve_grpc(server_args, model_info={})) is None
    _assert_served_lifecycle(seen, "sglang")
    built = server_mod.SGLangSchedulerServicer.call_args.kwargs
    assert built["metrics"] is seen["endpoint"].metrics


def test_the_vllm_plugin_accepts_a_metrics_port_with_the_python_servicer(monkeypatch):
    import argparse

    from smg_grpc_servicer.vllm import plugin

    monkeypatch.delenv("SMG_VLLM_SERVICER_IMPL", raising=False)
    monkeypatch.delenv("SMG_VLLM_SERVICER_METRICS_PORT", raising=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--grpc", action="store_true")
    plugin.add_metrics_arguments(parser)
    # The namespace as the flags fill it (a parse would also export the port
    # under the variable, which stays out of this test's environment: the
    # flag tests cover the export).
    parsed = argparse.Namespace(
        grpc=True,
        servicer_impl=None,
        metrics_port=29100,
        metrics_host=None,
        host="127.0.0.1",
        port=50051,
    )
    assert plugin.check_servicer_impl(parser, parsed) == "python"
    parsed.metrics_port = 50051
    with pytest.raises(SystemExit) as refused:
        plugin.check_servicer_impl(parser, parsed)
    assert refused.value.code == rust_lifecycle.CONFIG_ERROR_EXIT_CODE


# ---------------------------------------------------------------------------
# Through the servicers' own Generate, over engine stand-ins
# ---------------------------------------------------------------------------


def _served(registry: ServicerMetrics, *, errors: bool = False) -> str:
    """The exposition of a servicer whose engine matches string stops, with
    every family the Rust servicer would render for the same traffic (the
    error family exists once an error happened, on both)."""
    body = registry.render(metrics.SERVING)
    left_out = {"STRING_STOPS_TOTAL"} | (set() if errors else {"REQUEST_ERRORS_TOTAL"})
    expected = {metrics.NAMES[key]: kind for key, kind in TYPES.items() if key not in left_out}
    assert _families(body) == expected
    return body


def test_the_tokenspeed_servicer_records_a_streamed_request(tokenspeed_servicer_mod):
    pytest.importorskip("msgspec")
    pytest.importorskip("zmq")
    pytest.importorskip("numpy")
    from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

    outputs = [
        {
            "index": 0,
            "output_ids": [11],
            "meta_info": {"prompt_tokens": 3, "cached_tokens": 1, "completion_tokens": 1},
        },
        {
            "index": 0,
            "output_ids": [11, 12, 13],
            "meta_info": {
                "prompt_tokens": 3,
                "cached_tokens": 1,
                "completion_tokens": 2,
                "finish_reason": {"type": "length"},
            },
        },
    ]

    class _AsyncLlm:
        rid_to_state: dict = {}

        async def generate_request(self, request):
            for output in outputs:
                yield output

    cls = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    servicer = cls.__new__(cls)
    servicer.async_llm = _AsyncLlm()
    servicer._build_generate_req = lambda request: SimpleNamespace(rid="engine-rid")
    registry, _ = _registry(matches_string_stops=False)
    servicer._metrics = registry
    request = tokenspeed_scheduler_pb2.GenerateRequest(
        request_id="request-1",
        stream=True,
        tokenized=tokenspeed_scheduler_pb2.TokenizedInput(input_ids=[7, 8, 9]),
        sampling_params=tokenspeed_scheduler_pb2.SamplingParams(max_new_tokens=8),
    )

    async def collect():
        return [response async for response in servicer.Generate(request, context=None)]

    responses = asyncio.run(collect())
    assert [response.WhichOneof("response") for response in responses] == [
        "chunk",
        "chunk",
        "complete",
    ]
    body = _served(registry)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="length"') == 1
    assert _sample(body, metrics.NAMES["GENERATION_TOKENS_TOTAL"]) == 3
    assert _sample(body, metrics.NAMES["PROMPT_TOKENS_TOTAL"]) == 3
    assert _sample(body, metrics.NAMES["PROMPT_TOKENS_CACHED_TOTAL"]) == 1
    assert _sample(body, metrics.NAMES["TIME_TO_FIRST_TOKEN_SECONDS"] + "_count") == 1
    assert _sample(body, metrics.NAMES["INTER_TOKEN_LATENCY_SECONDS"] + "_count") == 1
    assert _sample(body, metrics.NAMES["REQUEST_PARAMS_MAX_TOKENS"] + "_sum") == 8
    assert _sample(body, metrics.NAMES["REQUESTS_ACTIVE"]) == 0


def test_the_tokenspeed_servicer_counts_an_invalid_request_as_a_refusal(
    tokenspeed_servicer_mod,
):
    pytest.importorskip("msgspec")
    pytest.importorskip("zmq")
    pytest.importorskip("numpy")
    from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

    class _Context:
        async def abort(self, code, message):
            raise grpc.aio.AbortError(f"{code}: {message}")

    cls = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    servicer = cls.__new__(cls)

    def refuse(request):
        raise ValueError("bad request")

    servicer._build_generate_req = refuse
    registry, _ = _registry(matches_string_stops=False)
    servicer._metrics = registry
    request = tokenspeed_scheduler_pb2.GenerateRequest(request_id="request-2")

    async def drive():
        with pytest.raises(grpc.aio.AbortError):
            async for _ in servicer.Generate(request, context=_Context()):
                pass

    asyncio.run(drive())
    body = registry.render(metrics.SERVING)
    assert _sample(body, metrics.NAMES["REQUESTS_REFUSED_TOTAL"], 'reason="invalid_argument"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_ACTIVE"]) == 0


def test_the_sglang_servicer_records_a_streamed_request(sglang_servicer_mod):
    from smg_grpc_proto import sglang_scheduler_pb2

    outputs = [
        {
            "index": 0,
            "token_ids": [11],
            "meta_info": {"prompt_tokens": 3, "cached_tokens": 1, "completion_tokens": 1},
        },
        {
            "index": 0,
            "token_ids": [12, 13],
            "finished": True,
            "meta_info": {
                "prompt_tokens": 3,
                "cached_tokens": 1,
                "completion_tokens": 3,
                "finish_reason": {"type": "length"},
            },
        },
    ]

    class _Manager:
        async def generate_request(self, *, obj, request_id, grpc_context):
            for output in outputs:
                yield output

    cls = sglang_servicer_mod.SGLangSchedulerServicer
    servicer = cls.__new__(cls)
    servicer.request_manager = _Manager()
    servicer._convert_generate_request = lambda request: SimpleNamespace(rid=request.request_id)
    servicer._handle_epd_disaggregation_encode_request = lambda request, tokenized: None
    registry, _ = _registry(matches_string_stops=False)
    servicer._metrics = registry
    request = sglang_scheduler_pb2.GenerateRequest(
        request_id="request-1",
        stream=True,
        tokenized=sglang_scheduler_pb2.TokenizedInput(input_ids=[7, 8, 9]),
        sampling_params=sglang_scheduler_pb2.SamplingParams(max_new_tokens=8, n=1),
    )

    async def collect():
        return [response async for response in servicer.Generate(request, None)]

    responses = asyncio.run(collect())
    assert [response.WhichOneof("response") for response in responses] == [
        "chunk",
        "chunk",
        "complete",
    ]
    body = _served(registry)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="length"') == 1
    assert _sample(body, metrics.NAMES["GENERATION_TOKENS_TOTAL"]) == 3
    assert _sample(body, metrics.NAMES["PROMPT_TOKENS_TOTAL"]) == 3
    assert _sample(body, metrics.NAMES["TIME_TO_FIRST_TOKEN_SECONDS"] + "_count") == 1
    assert _sample(body, metrics.NAMES["REQUEST_PARAMS_MAX_TOKENS"] + "_sum") == 8
    assert _sample(body, metrics.NAMES["REQUESTS_ACTIVE"]) == 0


def test_the_sglang_servicer_counts_a_scheduler_abort_as_an_error(sglang_servicer_mod):
    from smg_grpc_proto import sglang_scheduler_pb2

    class _Manager:
        async def generate_request(self, *, obj, request_id, grpc_context):
            yield {
                "index": 0,
                "token_ids": [],
                "error": "queue is full",
                "meta_info": {"finish_reason": {"type": "abort", "status_code": 503}},
            }

    class _Context:
        async def abort(self, code, message):
            raise grpc.aio.AbortError(f"{code}: {message}")

    cls = sglang_servicer_mod.SGLangSchedulerServicer
    servicer = cls.__new__(cls)
    servicer.request_manager = _Manager()
    servicer._convert_generate_request = lambda request: SimpleNamespace(rid=request.request_id)
    servicer._handle_epd_disaggregation_encode_request = lambda request, tokenized: None
    registry, _ = _registry(matches_string_stops=False)
    servicer._metrics = registry
    request = sglang_scheduler_pb2.GenerateRequest(request_id="request-3", stream=True)

    async def drive():
        with pytest.raises(grpc.aio.AbortError):
            async for _ in servicer.Generate(request, _Context()):
                pass

    asyncio.run(drive())
    body = _served(registry, errors=True)
    assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="error"') == 1
    assert _sample(body, metrics.NAMES["REQUEST_ERRORS_TOTAL"], 'code="unavailable"') == 1
    assert _sample(body, metrics.NAMES["REQUESTS_ACTIVE"]) == 0


def test_a_media_failure_is_a_media_refusal_only_when_the_requests_own_fetch_failed():
    pytest.importorskip("vllm")
    from smg_grpc_servicer.vllm import mm_processor
    from smg_grpc_servicer.vllm import servicer as vllm_servicer

    # The Rust servicer counts `media` for the request's own fetch failure
    # alone; a shed, a sidecar failure or a processor error is the worker's,
    # an error under its code, and an invalid request is a refusal of its own.
    assert (
        vllm_servicer._media_refusal(mm_processor.MmFetchFailed("media_refs[0]: fetch failed"))
        == "media"
    )
    assert (
        vllm_servicer._media_refusal(mm_processor.MmProcessorUnavailable("worker is saturated"))
        is None
    )
    assert (
        vllm_servicer._media_refusal(mm_processor.MmProcessorUnavailable("sidecar_unavailable"))
        is None
    )
    assert vllm_servicer._refusal(grpc.StatusCode.INVALID_ARGUMENT) == "invalid_argument"
    assert vllm_servicer._refusal(grpc.StatusCode.INTERNAL) is None
    assert vllm_servicer._refusal(grpc.StatusCode.UNAVAILABLE) is None


def test_the_vllm_servicer_records_a_request_through_the_launchers_endpoint(monkeypatch):
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    from smg_grpc_proto import vllm_engine_pb2
    from smg_grpc_servicer.vllm.servicer import VllmEngineServicer
    from vllm.outputs import CompletionOutput, RequestOutput

    async def generate(*, prompt, sampling_params, request_id, **kwargs):
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=[7, 8, 9, 10, 11],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[42, 43],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="length",
                )
            ],
            finished=True,
        )

    engine = SimpleNamespace(
        generate=generate,
        renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
        model_config=SimpleNamespace(
            is_multimodal_model=False, dtype=torch.float32, max_model_len=100
        ),
        vllm_config=SimpleNamespace(kv_events_config=None),
        errored=False,
    )
    monkeypatch.delenv("SMG_VLLM_MM_PROCESSOR", raising=False)
    endpoint = MetricsEndpoint("127.0.0.1:0", model="served", wire="vllm").start()
    metrics.remember_launcher_endpoint(endpoint)
    try:
        servicer = VllmEngineServicer(engine, start_time=0.0)
        request = vllm_engine_pb2.GenerateRequest(
            request_id="req-1",
            stream=True,
            tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[7, 8, 9]),
            sampling_params=vllm_engine_pb2.SamplingParams(max_tokens=2),
        )
        context = SimpleNamespace(invocation_metadata=lambda: ())

        async def collect():
            return [response async for response in servicer.Generate(request, context)]

        responses = asyncio.run(collect())
        assert [response.WhichOneof("response") for response in responses] == [
            "chunk",
            "complete",
        ]
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["STATE"], 'state="serving"') == 1
        assert _sample(body, metrics.NAMES["REQUESTS_TOTAL"], 'finish_reason="length"') == 1
        assert _sample(body, metrics.NAMES["PROMPT_TOKENS_TOTAL"], 'rank="0"') == 5
        assert _sample(body, metrics.NAMES["GENERATION_TOKENS_TOTAL"], 'rank="0"') == 2
        assert _sample(body, metrics.NAMES["REQUEST_PARAMS_MAX_TOKENS"] + "_sum") == 2
        assert _sample(body, metrics.NAMES["TIME_TO_FIRST_TOKEN_SECONDS"] + "_count") == 1
        assert _sample(body, metrics.NAMES["REQUEST_PROMPT_TOKENS"] + "_bucket", 'le="100"') == 1
        assert _sample(body, metrics.NAMES["REQUESTS_ACTIVE"]) == 0
        engine.errored = True
        body = _get(endpoint.address).read().decode()
        assert _sample(body, metrics.NAMES["STATE"], 'state="engine_down"') == 1
    finally:
        metrics.remember_launcher_endpoint(None)
        endpoint.stop()
