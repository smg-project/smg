"""
Standard gRPC health check service for vLLM Kubernetes probes.

Implements grpc.health.v1.Health protocol, delegating health status
to AsyncLLM.check_health() from the vLLM EngineClient protocol.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)


class VllmHealthServicer(health_pb2_grpc.HealthServicer):
    """
    Standard gRPC health check service for Kubernetes probes.
    Implements grpc.health.v1.Health protocol.

    Supports two service levels:
    1. Overall server health (service="") - for liveness probes
    2. VllmEngine service health (service="vllm.grpc.engine.VllmEngine")
       - for readiness probes

    Health is determined by calling async_llm.check_health(), the same
    EngineClient protocol method used by vLLM's HTTP /health endpoint.
    """

    OVERALL_SERVER = ""
    VLLM_SERVICE = "vllm.grpc.engine.VllmEngine"

    def __init__(self, async_llm: "AsyncLLM"):
        """
        Initialize health servicer.

        Args:
            async_llm: AsyncLLM instance for checking engine health
        """
        self.async_llm = async_llm
        self._shutting_down = False
        self._failure_logged = False
        logger.info("Standard gRPC health service initialized")

    def set_not_serving(self):
        """Mark all services as NOT_SERVING during graceful shutdown."""
        self._shutting_down = True
        logger.info("Health service status set to NOT_SERVING")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        """
        Standard health check for Kubernetes probes.

        Args:
            request: Contains service name ("" for overall, or specific service)
            context: gRPC context

        Returns:
            HealthCheckResponse with SERVING/NOT_SERVING/SERVICE_UNKNOWN status
        """
        service_name = request.service

        if self._shutting_down:
            return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.NOT_SERVING)

        if service_name in (self.OVERALL_SERVER, self.VLLM_SERVICE):
            try:
                await self.async_llm.check_health()
                return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)
            except Exception:
                self._log_failure_once(service_name)
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Unknown service: {service_name}")
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN)

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """
        Streaming health check - sends current status once.

        For now, sends current status once (Kubernetes doesn't use Watch).
        A full implementation would monitor status changes and stream updates.

        Args:
            request: Contains service name
            context: gRPC context

        Yields:
            HealthCheckResponse messages
        """
        service_name = request.service

        # Inline status computation to avoid Check()'s context.set_code()
        # side effect, which would incorrectly set the RPC status on the
        # streaming response for unknown services.
        status = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
        if self._shutting_down:
            status = health_pb2.HealthCheckResponse.NOT_SERVING
        elif service_name in (self.OVERALL_SERVER, self.VLLM_SERVICE):
            try:
                await self.async_llm.check_health()
                status = health_pb2.HealthCheckResponse.SERVING
            except Exception:
                self._log_failure_once(service_name)
                status = health_pb2.HealthCheckResponse.NOT_SERVING

        yield health_pb2.HealthCheckResponse(status=status)

    def _log_failure_once(self, service_name: str) -> None:
        """Record the first refused probe and stay quiet about the rest.

        Probes arrive on a fixed interval for as long as the pod lives, and an
        engine that fails this check does not recover, so every later probe
        would repeat the same traceback and bury the one that explains it.
        """
        if self._failure_logged:
            return
        self._failure_logged = True
        logger.exception("Health check is now failing for service '%s'", service_name)
