"""
Standard gRPC health check service for MLX.

Implements grpc.health.v1.Health protocol with simple liveness tracking, on
the frame the engine servicers share (:mod:`smg_grpc_servicer.health_frame`).
"""

import logging

from smg_grpc_servicer.health_frame import NOT_SERVING, SERVING, HealthFrame

logger = logging.getLogger(__name__)


class MlxHealthServicer(HealthFrame):
    """Standard gRPC health check for MLX inference server."""

    MLX_SERVICE = "mlx.grpc.engine.MlxEngine"
    ENGINE_SERVICE = MLX_SERVICE

    def __init__(self):
        self._serving = False
        logger.info("MlxHealthServicer initialized")

    def set_serving(self):
        self._serving = True
        logger.info("Health status set to SERVING")

    def set_not_serving(self):
        self._serving = False
        logger.info("Health status set to NOT_SERVING")

    async def engine_status(self, service_name: str) -> int:
        return SERVING if self._serving else NOT_SERVING
