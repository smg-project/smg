"""Standard ``grpc.health.v1.Health`` servicer for the TokenSpeed backend.

Advertises ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` so the router
can auto-detect the backend via the health probe. The protocol's frame is the
engine servicers' shared one (:mod:`smg_grpc_servicer.health_frame`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from grpc_health.v1 import health_pb2
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2

from smg_grpc_servicer.health_frame import NOT_SERVING, SERVING, HealthFrame
from smg_grpc_servicer.health_probe import StallReport

if TYPE_CHECKING:
    from tokenspeed.runtime.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

# Sourced from the proto descriptor so a package/service rename stays in sync.
TOKENSPEED_SCHEDULER_SERVICE_NAME = tokenspeed_scheduler_pb2.DESCRIPTOR.services_by_name[
    "TokenSpeedScheduler"
].full_name


class TokenSpeedHealthServicer(HealthFrame):
    """Health servicer that tracks TokenSpeed's AsyncLLM liveness.

    Advertises two service levels:

    * ``""`` (empty) — overall server health, flipped to SERVING once the
      warmup request succeeds and back to NOT_SERVING on shutdown.
    * ``tokenspeed.grpc.scheduler.TokenSpeedScheduler`` — readiness: the
      base status, plus a scheduler-responsiveness check (if there are
      pending requests but the scheduler hasn't pushed output for >30s,
      report NOT_SERVING).
    """

    TOKENSPEED_SERVICE = TOKENSPEED_SCHEDULER_SERVICE_NAME
    ENGINE_SERVICE = TOKENSPEED_SERVICE

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

    def shutting_down(self) -> bool:
        return self.async_llm.gracefully_exit

    async def engine_status(self, service_name: str) -> int:
        logger.debug("Health check request for service=%r", service_name)
        if service_name == self.OVERALL_SERVER:
            return self._serving_status.get(self.OVERALL_SERVER, NOT_SERVING)
        base = self._serving_status.get(self.TOKENSPEED_SERVICE, NOT_SERVING)
        if base != SERVING:
            return base
        # A scheduler silent beyond the threshold with requests pending is stuck.
        if self._stall.stuck(self.async_llm.last_receive_tstamp, len(self.async_llm.rid_to_state)):
            return NOT_SERVING
        return SERVING
