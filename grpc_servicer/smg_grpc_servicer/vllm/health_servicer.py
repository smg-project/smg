"""
Standard gRPC health check service for vLLM Kubernetes probes.

Implements grpc.health.v1.Health protocol, delegating health status
to AsyncLLM.check_health() from the vLLM EngineClient protocol, on the frame
the engine servicers share (:mod:`smg_grpc_servicer.health_frame`).
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from smg_grpc_servicer.health_frame import NOT_SERVING, SERVING, HealthFrame

# The launcher imports this module before it defines serve_grpc: the moment
# the servicer switch has to be in place (see launcher_switch).
from smg_grpc_servicer.vllm.launcher_switch import install_launcher_switch

install_launcher_switch()

if TYPE_CHECKING:
    from vllm.v1.engine.async_llm import AsyncLLM

logger = init_logger(__name__)


class VllmHealthServicer(HealthFrame):
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

    VLLM_SERVICE = "vllm.grpc.engine.VllmEngine"
    ENGINE_SERVICE = VLLM_SERVICE

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

    def shutting_down(self) -> bool:
        return self._shutting_down

    async def engine_status(self, service_name: str) -> int:
        try:
            await self.async_llm.check_health()
        except Exception:
            self._log_failure_once(service_name)
            return NOT_SERVING
        return SERVING

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
