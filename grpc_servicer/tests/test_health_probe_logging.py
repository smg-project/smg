"""A stuck scheduler is reported once, not on every probe.

Kubernetes re-probes on its own interval for as long as the condition holds, so
a line per probe buries whatever else the scheduler is trying to report.

Run with: pytest grpc_servicer/tests/test_health_probe_logging.py
"""

import asyncio
import importlib.util
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
from grpc_health.v1 import health_pb2  # noqa: E402

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING

_BACKENDS = Path(__file__).resolve().parent.parent / "smg_grpc_servicer"


def _load(backend):
    path = _BACKENDS / backend / "health_servicer.py"
    spec = importlib.util.spec_from_file_location(f"{backend}_health_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine():
    return SimpleNamespace(gracefully_exit=False, last_receive_tstamp=time.time(), rid_to_state={})


def _sglang():
    module = _load("sglang")
    engine = _engine()
    servicer = module.SGLangHealthServicer(engine, {})
    servicer.set_serving()
    return servicer, engine, module.SGLangHealthServicer.SGLANG_SERVICE


def _tokenspeed():
    module = _load("tokenspeed")
    engine = _engine()
    servicer = module.TokenSpeedHealthServicer(engine, {})
    servicer.set_serving()
    return servicer, engine, module.TokenSpeedHealthServicer.TOKENSPEED_SERVICE


BACKENDS = [pytest.param(_sglang, id="sglang"), pytest.param(_tokenspeed, id="tokenspeed")]


def _probe(servicer, service):
    return asyncio.run(servicer.Check(SimpleNamespace(service=service), None))


def _stall(engine):
    engine.last_receive_tstamp = time.time() - 600
    engine.rid_to_state = {"r1": object()}


def _recover(engine):
    engine.last_receive_tstamp = time.time()


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("build", BACKENDS)
def test_a_healthy_scheduler_says_nothing(build, caplog):
    servicer, _, service = build()
    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            assert _probe(servicer, service).status == SERVING
    assert _warnings(caplog) == []


@pytest.mark.parametrize("build", BACKENDS)
def test_a_stuck_scheduler_is_reported_once_however_long_it_lasts(build, caplog):
    servicer, engine, service = build()
    _stall(engine)
    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            assert _probe(servicer, service).status == NOT_SERVING
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize("build", BACKENDS)
def test_recovery_is_reported_and_a_second_stall_is_reported_again(build, caplog):
    servicer, engine, service = build()
    with caplog.at_level(logging.DEBUG):
        _stall(engine)
        assert _probe(servicer, service).status == NOT_SERVING
        _recover(engine)
        assert _probe(servicer, service).status == SERVING
        _stall(engine)
        assert _probe(servicer, service).status == NOT_SERVING

    assert len(_warnings(caplog)) == 2, "the stall was reported again after it cleared"
    recovered = [
        r for r in caplog.records if r.levelno == logging.INFO and "responsive" in r.getMessage()
    ]
    assert len(recovered) == 1


@pytest.mark.parametrize("build", BACKENDS)
def test_pending_work_is_what_makes_silence_a_stall(build, caplog):
    # An idle server has not received anything for a long time either. Reporting
    # that would flag every quiet deployment as broken.
    servicer, engine, service = build()
    engine.last_receive_tstamp = time.time() - 600
    with caplog.at_level(logging.DEBUG):
        assert _probe(servicer, service).status == SERVING
    assert _warnings(caplog) == []
