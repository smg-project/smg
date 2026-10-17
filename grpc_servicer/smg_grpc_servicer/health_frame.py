"""The frame every engine's gRPC health servicer shares.

Each engine package serves ``grpc.health.v1.Health`` with the same frame around
one decision of its own. The frame: two service names (the overall server,
``""``, for liveness; the engine's service for readiness); a server that is
shutting down answers NOT_SERVING for every service; a known service answers
with the engine's own status; an unknown one is refused with NOT_FOUND on a
unary ``Check`` and answered with a SERVICE_UNKNOWN status on a streaming
``Watch``, since a code set on a stream fails it; and ``Watch`` yields the
current status once (Kubernetes probes use ``Check``). An engine's servicer
subclasses the frame, names its service, implements ``engine_status`` and,
where it has one, ``shutting_down``, and keeps its own serving flips and their
log lines.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

SERVING = health_pb2.HealthCheckResponse.SERVING
NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


class HealthFrame(health_pb2_grpc.HealthServicer):
    """The health protocol around one engine's status decision."""

    #: The overall server: liveness probes ask for it.
    OVERALL_SERVER = ""
    #: The engine's own service: readiness probes ask for it. Set by the
    #: engine's servicer.
    ENGINE_SERVICE: str

    def shutting_down(self) -> bool:
        """Whether the server is shutting down; then every service is NOT_SERVING."""
        return False

    async def engine_status(self, service_name: str) -> int:
        """The status of a known service, the overall server or the engine's."""
        raise NotImplementedError

    def is_known(self, service_name: str) -> bool:
        return service_name in (self.OVERALL_SERVER, self.ENGINE_SERVICE)

    async def status_of(self, service_name: str) -> int | None:
        """The status the frame answers for ``service_name``; ``None`` for an
        unknown service, which ``Check`` refuses and ``Watch`` reports."""
        if self.shutting_down():
            return NOT_SERVING
        if self.is_known(service_name):
            return await self.engine_status(service_name)
        return None

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        service_name = request.service
        status = await self.status_of(service_name)
        if status is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Unknown service: {service_name}")
            return health_pb2.HealthCheckResponse(status=SERVICE_UNKNOWN)
        return health_pb2.HealthCheckResponse(status=status)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        # Kubernetes probes use Check; Watch yields the current status once,
        # and never a code on the stream: an unknown service is a status.
        status = await self.status_of(request.service)
        yield health_pb2.HealthCheckResponse(status=SERVICE_UNKNOWN if status is None else status)
