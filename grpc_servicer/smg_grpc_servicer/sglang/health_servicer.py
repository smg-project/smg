"""
Standard gRPC health check service implementation for Kubernetes probes.

This module implements the grpc.health.v1.Health service protocol, enabling
native Kubernetes gRPC health probes for liveness and readiness checks, on
the frame the engine servicers share (:mod:`smg_grpc_servicer.health_frame`).
"""

import logging

from grpc_health.v1 import health_pb2

from smg_grpc_servicer.health_frame import NOT_SERVING, SERVING, HealthFrame
from smg_grpc_servicer.health_probe import StallReport

logger = logging.getLogger(__name__)


class SGLangHealthServicer(HealthFrame):
    """
    Standard gRPC health check service implementation for Kubernetes probes.
    Implements grpc.health.v1.Health protocol.

    Supports two service levels:
    1. Overall server health (service="") - for liveness probes
    2. SGLang service health (service="sglang.grpc.scheduler.SglangScheduler") - for readiness probes

    Health status lifecycle:
    - NOT_SERVING: Initial state, model loading, or shutting down
    - SERVING: Model loaded and ready to serve requests
    """

    # Service names we support
    SGLANG_SERVICE = "sglang.grpc.scheduler.SglangScheduler"
    ENGINE_SERVICE = SGLANG_SERVICE

    def __init__(self, request_manager, scheduler_info: dict):
        """
        Initialize health servicer.

        Args:
            request_manager: GrpcRequestManager instance for checking server state
            scheduler_info: Dict containing scheduler metadata
        """
        self.request_manager = request_manager
        self.scheduler_info = scheduler_info
        self._serving_status = {}
        self._stall = StallReport(logger)

        # Initially set to NOT_SERVING until model is loaded
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING

        logger.info("Standard gRPC health service initialized")

    def set_serving(self):
        """Mark services as SERVING - call this after model is loaded."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.SERVING
        logger.info("Health service status set to SERVING")

    def set_not_serving(self):
        """Mark services as NOT_SERVING - call this during shutdown."""
        self._serving_status[self.OVERALL_SERVER] = health_pb2.HealthCheckResponse.NOT_SERVING
        self._serving_status[self.SGLANG_SERVICE] = health_pb2.HealthCheckResponse.NOT_SERVING
        logger.info("Health service status set to NOT_SERVING")

    def shutting_down(self) -> bool:
        return self.request_manager.gracefully_exit

    async def engine_status(self, service_name: str) -> int:
        logger.debug(f"Health check request for service: '{service_name}'")
        if service_name == self.OVERALL_SERVER:
            # Overall server health - just check if process is alive
            status = self._serving_status.get(self.OVERALL_SERVER, NOT_SERVING)
            logger.debug(
                f"Overall health check: {health_pb2.HealthCheckResponse.ServingStatus.Name(status)}"
            )
            return status
        # Specific service health - check if ready to serve: the base status
        # first, then the scheduler's responsiveness.
        base_status = self._serving_status.get(self.SGLANG_SERVICE, NOT_SERVING)
        if base_status != SERVING:
            logger.debug("Service health check: NOT_SERVING (base status)")
            return base_status
        # A scheduler silent beyond the threshold with requests pending is stuck.
        if self._stall.stuck(
            self.request_manager.last_receive_tstamp, len(self.request_manager.rid_to_state)
        ):
            return NOT_SERVING
        logger.debug("Service health check: SERVING")
        return SERVING
