"""Serialized, opt-in request-directory profiling on vLLM 0.31 AsyncLLM."""

import asyncio
from pathlib import Path
from uuid import uuid4

import grpc
from smg_grpc_proto.generated import common_pb2


class ProfileStateError(RuntimeError):
    pass


class RequestProfiler:
    def __init__(self, engine):
        self.engine, self.lock, self.phase = engine, asyncio.Lock(), "idle"
        self.owner, self.ranks = None, None

    def check_ranks(self, rows, phase):
        config = self.engine.vllm_config.parallel_config
        expected = config.tensor_parallel_size * config.pipeline_parallel_size
        if config.data_parallel_size != 1 or not isinstance(rows, list) or len(rows) != expected:
            raise RuntimeError("Request profiling requires all TP/PP ranks and DP1")
        ranks = {row["rank"] for row in rows}
        if ranks != set(range(expected)) or any(
            row["owner"] != self.owner or row["phase"] != phase for row in rows
        ):
            raise RuntimeError("Profiler rank/owner/transition evidence differs")
        if self.ranks is not None and ranks != self.ranks:
            raise RuntimeError("Profiler ranks changed inside the session")
        self.ranks = ranks

    async def transition(self, start, directory, budget):
        async with self.lock:
            if (start and self.phase != "idle") or (not start and self.phase != "active"):
                raise ProfileStateError("Profiler is already active, absent, or faulted")
            config = self.engine.vllm_config.parallel_config
            if (
                config.data_parallel_size != 1
                or config.tensor_parallel_size < 1
                or config.pipeline_parallel_size < 1
            ):
                raise ProfileStateError("Request profiling requires positive TP/PP ranks and DP1")
            extension = (
                "smg_grpc_servicer.vllm.request_profile_worker.RequestProfileWorkerExtension"
            )
            if config.worker_extension_cls != extension:
                raise ProfileStateError(
                    "Select the exact request-profile worker extension; "
                    "refusing to replace another extension"
                )
            if start and self.engine.profiler is not None:
                raise ProfileStateError(
                    "A frontend profiler already exists; refusing to replace it"
                )
            self.phase = "starting" if start else "stopping"
            try:
                if start:
                    self.owner, self.ranks = uuid4().hex, None
                    rows = await self.engine.collective_rpc(
                        "smg_request_profile",
                        timeout=budget,
                        args=("prepare", self.owner, directory),
                    )
                    self.check_ranks(rows, "prepared")
                    await self.engine.start_profile()
                    rows = await self.engine.collective_rpc(
                        "smg_request_profile", timeout=budget, args=("started", self.owner),
                    )
                    self.check_ranks(rows, "active")
                    self.phase = "active"
                else:
                    await self.engine.stop_profile()
                    rows = await self.engine.collective_rpc(
                        "smg_request_profile", timeout=budget, args=("finish", self.owner),
                    )
                    self.check_ranks(rows, "finished")
                    self.phase, self.owner, self.ranks = "idle", None, None
            except BaseException:
                # A cancelled local future cannot establish the state of remote ranks.
                # Keep their owner/config/profiler and require engine recovery; never reset.
                self.phase = "faulted"
                raise

    async def run(self, start, request, context, timeout):
        directory = None
        if start:
            unsupported = [
                field.name for field, value in request.ListFields() if field.name != "output_dir"
            ]
            if unsupported or not request.HasField("output_dir") or not request.output_dir:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "This path accepts only a nonempty output_dir",
                )
            if "://" in request.output_dir:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "output_dir must name a local directory"
                )
            directory = str(Path(request.output_dir).expanduser().resolve())
        remaining = context.time_remaining()
        budget = timeout if remaining is None else min(timeout, remaining)
        try:
            await asyncio.wait_for(self.transition(start, directory, budget), timeout=budget)
        except ProfileStateError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041
            await context.abort(
                grpc.StatusCode.DEADLINE_EXCEEDED,
                "Profiler deadline exceeded; remote state may be unknown",
            )
        except Exception as error:
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Real profiler operation failed: {error}"
            )
        return common_pb2.ProfileResponse(
            success=True, message="Profiling started" if start else "Profiling stopped"
        )


def session(engine):
    existing = getattr(engine, "_smg_request_profiler", None)
    if existing is None:
        existing = RequestProfiler(engine)
        engine._smg_request_profiler = existing
    return existing
