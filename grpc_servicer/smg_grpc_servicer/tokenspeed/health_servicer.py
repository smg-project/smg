"""Standard ``grpc.health.v1.Health`` servicer for the TokenSpeed backend.

Advertises ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` so the router
can auto-detect the backend via the health probe.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

from smg_grpc_servicer.health_probe import StallReport

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

# Sourced from the proto descriptor so a package/service rename stays in sync.
TOKENSPEED_SCHEDULER_SERVICE_NAME = tokenspeed_scheduler_pb2.DESCRIPTOR.services_by_name[
    "TokenSpeedScheduler"
].full_name


class TokenSpeedHealthServicer(health_pb2_grpc.HealthServicer):
    """Health servicer that tracks TokenSpeed's AsyncLLM liveness.

    Advertises two service levels:

    * ``""`` (empty) — overall server health, flipped to SERVING once the
      warmup request succeeds and back to NOT_SERVING on shutdown.
    * ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` — readiness: the
      base status, plus a scheduler-responsiveness check (if there are
      pending requests but the scheduler hasn't pushed output for >30s,
      report NOT_SERVING).
    """

    OVERALL_SERVER = ""
    TOKENSPEED_SERVICE = TOKENSPEED_SCHEDULER_SERVICE_NAME

    def __init__(self, async_llm: AsyncLLM, scheduler_info: dict):
        self.async_llm = async_llm
        self.scheduler_info = scheduler_info
        self._serving_status: dict[str, int] = {
            self.OVERALL_SERVER: health_pb2.HealthCheckResponse.NOT_SERVING,
            self.TOKENSPEED_SERVICE: health_pb2.HealthCheckResponse.NOT_SERVING,
        }
        self._stall = StallReport(logger)
        logger.info("TokenSpeed gRPC health service initialized")

    def set_serving(self) -> None:
        """Flip both services to SERVING (call after successful warmup)."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.SERVING
        self._serving_status[self.TOKENSPEED_SERVICE] = health_pb2.HealthCheckResponse.SERVING
        logger.info("TokenSpeed gRPC health status -> SERVING")

    def set_not_serving(self) -> None:
        """Flip both services to NOT_SERVING (call on shutdown)."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.TOKENSPEED_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING
        logger.info("TokenSpeed gRPC health status -> NOT_SERVING")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        service_name = request.service
        logger.debug("Health check request for service=%r", service_name)

        if self.async_llm.gracefully_exit:
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        if service_name == self.OVERALL_SERVER:
            return health_pb2.HealthCheckResponse(
                status=self._serving_status.get(
                    self.OVERALL_SERVER, health_pb2.HealthCheckResponse.NOT_SERVING
                )
            )

        if service_name == self.TOKENSPEED_SERVICE:
            base = self._serving_status.get(
                self.TOKENSPEED_SERVICE, health_pb2.HealthCheckResponse.NOT_SERVING
            )
            if base != health_pb2.HealthCheckResponse.SERVING:
                return health_pb2.HealthCheckResponse(status=base)

            # A scheduler silent beyond the threshold with requests pending is stuck.
            if self._stall.stuck(
                self.async_llm.last_receive_tstamp, len(self.async_llm.rid_to_state)
            ):
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Unknown service: {service_name}")
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        # K8s probes use Check, not Watch — we emit the current status once,
        # computed as Check computes it (a server shutting down answers
        # NOT_SERVING for every service) but without Check's refusal on the
        # context: a NOT_FOUND code set on a streaming call fails the stream,
        # while the protocol answers an unknown service with SERVICE_UNKNOWN.
        if self.async_llm.gracefully_exit or request.service in (
            self.OVERALL_SERVER,
            self.TOKENSPEED_SERVICE,
        ):
            yield await self.Check(request, context)
        else:
            yield health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
            )
