"""The frame every engine's health servicer shares: two names, shutdown first,
a known service's own status, an unknown one refused on Check and reported on
Watch without a code on the stream."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("smg_grpc_proto")
import grpc  # noqa: E402
from smg_grpc_servicer.health_frame import (  # noqa: E402
    NOT_SERVING,
    SERVICE_UNKNOWN,
    SERVING,
    HealthFrame,
)


class _Engine(HealthFrame):
    ENGINE_SERVICE = "test.grpc.Engine"

    def __init__(self):
        self.serving = True
        self.closing = False
        self.asked: list[str] = []

    def shutting_down(self) -> bool:
        return self.closing

    async def engine_status(self, service_name: str) -> int:
        self.asked.append(service_name)
        return SERVING if self.serving else NOT_SERVING


class _Context:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def _check(servicer, service):
    context = _Context()
    response = asyncio.run(servicer.Check(SimpleNamespace(service=service), context))
    return response.status, context


def _watch(servicer, service):
    context = _Context()

    async def collect():
        return [r async for r in servicer.Watch(SimpleNamespace(service=service), context)]

    (response,) = asyncio.run(collect())
    return response.status, context


@pytest.mark.parametrize("service", ["", "test.grpc.Engine"])
def test_a_known_service_answers_the_engines_status(service):
    engine = _Engine()
    assert _check(engine, service)[0] == SERVING
    assert _watch(engine, service)[0] == SERVING
    engine.serving = False
    assert _check(engine, service)[0] == NOT_SERVING
    assert _watch(engine, service)[0] == NOT_SERVING
    assert engine.asked == [service] * 4


def test_an_unknown_service_is_refused_on_check_and_reported_on_watch():
    engine = _Engine()
    status, context = _check(engine, "no.such.Service")
    assert status == SERVICE_UNKNOWN
    assert context.code == grpc.StatusCode.NOT_FOUND
    assert context.details == "Unknown service: no.such.Service"
    status, context = _watch(engine, "no.such.Service")
    assert status == SERVICE_UNKNOWN
    assert (context.code, context.details) == (None, None), "a code on a stream fails it"
    assert engine.asked == [], "the engine is not asked about a service it does not have"


@pytest.mark.parametrize("service", ["", "test.grpc.Engine", "no.such.Service"])
def test_shutting_down_answers_not_serving_for_every_service_before_the_name(service):
    engine = _Engine()
    engine.closing = True
    status, context = _check(engine, service)
    assert (status, context.code) == (NOT_SERVING, None)
    status, context = _watch(engine, service)
    assert (status, context.code) == (NOT_SERVING, None)
    assert engine.asked == []
