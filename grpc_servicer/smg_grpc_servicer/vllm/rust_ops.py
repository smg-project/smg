"""Thin event adapter to vLLM's statistics, collectors and metrics HTTP handler.

Rust decides which outputs were consumed and why a request finished. All metric
formulas, descriptors, plugins and multiprocess aggregation remain in vLLM.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import AsyncExitStack, suppress
from types import SimpleNamespace as NS
from typing import Any

logger = logging.getLogger(__name__)
MAX_PENDING_OUTPUTS = 65536


def _fields(struct: Any, value: Any) -> dict[str, Any]:
    """Decode either msgspec array-like or map-like wire fields without tensor frames."""
    return value if isinstance(value, dict) else dict(zip(struct.__struct_fields__, value))


class RustOps:
    def __init__(self, config: Any, port: int, *, log_stats: bool, host: str = "0.0.0.0") -> None:
        """Prepare native collectors and multiprocess storage before engine processes start."""
        import msgspec
        from prometheus_client import values
        from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest
        from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
        from vllm.v1.metrics.stats import LoRARequestStates

        if log_stats:
            self._check_compatibility()
        setup_multiprocess_prometheus()
        # prometheus_client can be imported before the launcher enables mmap.
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR") and not values.ValueClass._multiprocess:
            values.ValueClass = values.MultiProcessValue()
        self.port, self.host = port, host
        self.closed = False
        self.error: Exception | None = None
        self.lock = threading.RLock()
        self.stack = AsyncExitStack()
        self.stats_task: asyncio.Task | None = None
        self.mm_cache_stats: Any = None
        self.requests: dict[str, Any] = {}
        self.parents: dict[str, Any] = {}
        self.children: dict[str, str] = {}
        self.arrivals: dict[str, float] = {}
        self.pending_outputs = 0
        self.event_time = 0.0
        self.decoder = msgspec.msgpack.Decoder(ext_hook=lambda code, data: None)
        self.request_type, self.output_type, self.batch_type = (
            EngineCoreRequest,
            EngineCoreOutput,
            EngineCoreOutputs,
        )
        self.lora = LoRARequestStates(log_stats)
        parallel = config.parallel_config
        # The Rust launcher localizes the complete DP group.
        ranks = list(
            range(
                parallel.data_parallel_rank,
                parallel.data_parallel_rank + parallel.data_parallel_size,
            )
        )
        self.config, self.ranks, self.log_stats = config, ranks, log_stats
        self.stats = None
        # Reuse vLLM's post-handshake DP sums, hybrid cache alignment and
        # auto-fitted context lengths before constructing metric collectors.
        self.ready_state = NS(
            vllm_config=config,
            _effective_attention_block_sizes=set(),
            stats_update_address=None,
            _kv_event_sources={},
        )

    @staticmethod
    def _check_compatibility() -> None:
        """Reject incompatible native statistics APIs before starting engine workers."""
        try:
            from vllm.v1.engine.core_client import MPClient
            from vllm.v1.engine.parallel_sampling import ParentRequest
            from vllm.v1.metrics.stats import IterationStats

            inspect.signature(MPClient._apply_ready_response).bind(None, b"")
            inspect.signature(IterationStats.update_from_output).bind(*([None] * 7))
            inspect.signature(IterationStats.update_from_finished_request).bind(
                None,
                finish_reason=None,
                request_id="probe",
                num_prompt_tokens=0,
                max_tokens_param=None,
                req_stats=None,
                num_cached_tokens=0,
            )
            ParentRequest(
                NS(request_id="probe", external_req_id="probe", params=NS(n=2, output_kind=0))
            )
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "--metrics-port is incompatible with this vLLM statistics API"
            ) from error

    def observe(self, event: str, payload: bytes, timestamp: float | None = None) -> None:
        """Apply one ordered Rust event; poison the scrape on incompatible engine data."""
        with self.lock:
            if self.closed or self.error is not None or not self.log_stats:
                return
            try:
                self.event_time = time.time() if timestamp is None else timestamp
                if event == "overflow":
                    raise RuntimeError("vLLM metrics observer queue overflowed")
                if event == "ready":
                    from vllm.v1.engine.core_client import MPClient

                    MPClient._apply_ready_response(self.ready_state, payload)
                elif event == "ready_done":
                    from vllm.v1.metrics.loggers import StatLoggerManager

                    if self.stats is None:
                        self.stats = StatLoggerManager(self.config, engine_idxs=self.ranks)
                        self.stats.log_engine_initialized()
                elif event == "add":
                    self._add(self.decoder.decode(payload))
                elif event == "output":
                    self._output(self.decoder.decode(payload))
                elif event == "consume":
                    self._consume(payload.decode())
                elif event in ("finish", "processed"):
                    parent, index, *reason = json.loads(payload)
                    child = f"{parent}-{index}"
                    request_id = child if child in self.children else parent
                    if reason:
                        self._finish(request_id, *reason)
                    else:
                        self._processed(request_id)
                elif event == "parent":
                    from vllm.v1.engine.parallel_sampling import ParentRequest

                    parent, count, *arrival = json.loads(payload)
                    if arrival:
                        self.arrivals.update(
                            {
                                child: arrival[0]
                                for child in (
                                    [f"{parent}-{i}" for i in range(count)]
                                    if count > 1
                                    else [parent]
                                )
                            }
                        )
                    if count > 1:
                        group = ParentRequest(
                            NS(
                                request_id=parent,
                                external_req_id=parent,
                                params=NS(n=count, output_kind=0),
                            )
                        )
                        group.child_requests = {f"{parent}-{i}" for i in range(count)}
                        self.parents[parent] = group
                        self.children.update({child: parent for child in group.child_requests})
                elif event == "abort":
                    self._forget(payload.decode())
                elif event == "forget_parent":
                    parent = payload.decode()
                    group = self.parents.get(parent)
                    for child in list(group.child_requests) if group else [parent]:
                        self._forget(child)
            except Exception as error:
                self.error = error
                self.requests.clear()
                self.parents.clear()
                self.children.clear()
                self.arrivals.clear()
                self.pending_outputs = 0
                logger.exception("vLLM metrics collection failed")

    def _add(self, raw: Any) -> None:
        """Track an admitted child with frontend arrival time and native request statistics."""
        from vllm.v1.metrics.stats import RequestStateStats

        request = _fields(self.request_type, raw)
        lora = request.get("lora_request")
        self.requests[request["request_id"]] = NS(
            stats=RequestStateStats(
                arrival_time=self.arrivals.pop(request["request_id"], request["arrival_time"])
            ),
            external_id=request.get("external_req_id")
            or self.children.get(request["request_id"], request["request_id"]),
            prompt_len=len(request.get("prompt_token_ids") or []),
            max_tokens=(request.get("sampling_params") or {}).get("max_tokens"),
            lora_name=lora.get("lora_name")
            if isinstance(lora, dict)
            else lora[0]
            if lora
            else None,
            prefilling=True,
            cached=0,
            pending=deque(),
            active=None,
            pooling=request.get("pooling_params") is not None,
        )

    def _output(self, raw: Any) -> None:
        """Queue an engine batch until its live request outputs have been consumed."""
        import msgspec
        from vllm.v1.metrics.stats import IterationStats, SchedulerStats

        fields = _fields(self.batch_type, raw)
        scheduler = fields.get("scheduler_stats")
        if not fields.get("outputs") and scheduler is None:
            return
        batch = NS(
            rank=fields.get("engine_index", 0),
            timestamp=fields["timestamp"],
            scheduler=msgspec.convert(scheduler, SchedulerStats) if scheduler is not None else None,
            iteration=IterationStats() if fields.get("outputs") else None,
            remaining=0,
            started=False,
        )
        # Publish engine gauges on arrival, independently of stream consumption.
        self.lora.update_scheduler_stats(batch.scheduler)
        self.stats.record(
            scheduler_stats=batch.scheduler,
            iteration_stats=None,
            mm_cache_stats=self.mm_cache_stats() if self.mm_cache_stats else None,
            engine_idx=batch.rank,
        )
        for output in fields.get("outputs", []):
            output = _fields(self.output_type, output)
            request = self.requests.get(output["request_id"])
            if request is not None:
                request.pending.append((batch, output))
                batch.remaining += 1
                self.pending_outputs += 1
                if self.pending_outputs > MAX_PENDING_OUTPUTS:
                    raise RuntimeError("vLLM metrics pending output limit exceeded")
        self._record(batch)

    def _consume(self, request_id: str) -> None:
        """Apply native token and timing formulas when Rust consumes an output."""
        import msgspec
        from vllm.v1.engine import EngineCoreEvent, FinishReason
        from vllm.v1.metrics.stats import PrefillStats

        request = self.requests.get(request_id)
        if request is None or not request.pending:
            return
        batch, output = request.pending.popleft()
        request.active = batch
        request.engine_reason = output.get("finish_reason")
        if not batch.started:
            batch.iteration.iteration_timestamp = self.event_time
            batch.started = True
        prefill = output.get("prefill_stats")
        prefill = msgspec.convert(prefill, PrefillStats) if prefill is not None else None
        batch.iteration.update_from_output(
            NS(
                request_id=request_id,
                new_token_ids=output["new_token_ids"],
                prefill_stats=prefill,
                events=msgspec.convert(output.get("events"), list[EngineCoreEvent] | None),
                num_nans_in_logits=output.get("num_nans_in_logits", 0),
            ),
            batch.timestamp,
            request.prefilling,
            request.stats,
            self.lora,
            request.lora_name,
        )
        if request.prefilling and prefill is not None:
            request.cached = prefill.num_cached_tokens
        request.prefilling = False
        # Pooling has no Rust string-stop stage or GenerateComplete event.
        if request.engine_reason == FinishReason.ERROR or (
            request.pooling and request.engine_reason is not None
        ):
            self._finish(request_id, str(FinishReason(request.engine_reason)))

    def _finish(self, request_id: str, reason: str, frontend_stop: bool = False) -> None:
        """Record the effective frontend finish reason once, then release child state."""
        from vllm.v1.engine import FinishReason
        from vllm.v1.engine.parallel_sampling import ParentRequest

        request = self.requests.get(request_id)
        if request is None:
            return
        if request.active is not None and (
            reason != "abort" or request.engine_reason == FinishReason.ABORT
        ):
            # The gRPC adapter maps engine repetition to public "stop"; metrics
            # retain vLLM's distinct reason unless a frontend string stop won.
            if (
                reason == "stop"
                and not frontend_stop
                and request.engine_reason == FinishReason.REPETITION
            ):
                reason = "repetition"
            iteration = request.active.iteration
            iteration.update_from_finished_request(
                finish_reason=FinishReason[reason.upper()],
                request_id=request.external_id,
                num_prompt_tokens=request.prompt_len,
                max_tokens_param=request.max_tokens,
                req_stats=request.stats,
                num_cached_tokens=request.cached,
            )
            parent = self.parents.get(self.children.get(request_id))
            if parent is not None:
                parent.child_requests.discard(request_id)
            ParentRequest.observe_finished_request(
                parent, iteration, request.stats.num_generation_tokens
            )
        self._forget(request_id)

    def _processed(self, request_id: str) -> None:
        """Release the consumed output so its completed batch can be published."""
        request = self.requests.get(request_id)
        if request is not None and request.active is not None:
            batch, request.active = request.active, None
            batch.remaining -= 1
            self.pending_outputs -= 1
            self._record(batch)

    def _forget(self, request_id: str) -> None:
        """Discard abandoned outputs and remove all state owned by this child."""
        self.arrivals.pop(request_id, None)
        request = self.requests.get(request_id)
        if request is not None:
            self.lora.request_finished(request_id, request.lora_name)
            self._processed(request_id)
            del self.requests[request_id]
            for batch, _ in request.pending:
                batch.remaining -= 1
                self.pending_outputs -= 1
                self._record(batch)
        parent_id = self.children.pop(request_id, None)
        parent = self.parents.get(parent_id)
        if parent is not None:
            parent.child_requests.discard(request_id)
            if not parent.child_requests:
                del self.parents[parent_id]

    def _record(self, batch: Any) -> None:
        """Publish completed iterations without waiting for unrelated slow streams."""
        if batch.remaining == 0 and batch.iteration is not None:
            self.stats.record(
                scheduler_stats=None,
                iteration_stats=batch.iteration,
                mm_cache_stats=None,
                engine_idx=batch.rank,
            )
            batch.iteration = None

    async def start(self, server: Any) -> None:
        """Bind the scrape listener and attach the observer before launching the engine."""
        from aiohttp import web
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        from vllm.v1.metrics.prometheus import get_prometheus_registry

        @web.middleware
        async def health(request: Any, handler: Any) -> Any:
            """Refuse stale successful scrapes after collection has failed."""
            if request.path == "/metrics" and self.error is not None:
                raise web.HTTPServiceUnavailable(text="vLLM metrics collection failed")
            return await handler(request)

        app = web.Application(middlewares=[health])

        async def metrics(request: Any) -> Any:
            """Render vLLM's registry with the public Prometheus exposition API."""
            return web.Response(
                body=generate_latest(get_prometheus_registry()),
                headers={"Content-Type": CONTENT_TYPE_LATEST},
            )

        app.router.add_get("/metrics", metrics)
        runner = web.AppRunner(app, access_log=None)
        self.stack.push_async_callback(runner.cleanup)
        await runner.setup()
        await web.TCPSite(runner, self.host, self.port).start()
        if self.log_stats:
            server.set_metrics_observer(self.observe)
            self.stats_task = asyncio.create_task(self._log_stats())
        logger.info("Metrics HTTP server listening on %s:%d", self.host, self.port)

    async def _log_stats(self) -> None:
        """Run native periodic loggers under the same lock as request updates."""
        from vllm import envs

        while True:
            await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
            try:
                with self.lock:
                    if self.stats is not None:
                        self.stats.log()
            except Exception:
                logger.exception("vLLM statistics logging failed")

    async def close(self) -> None:
        """Stop collection and release the listener; repeated closes are harmless."""
        if self.closed:
            return
        with self.lock:
            self.closed = True
            self.requests.clear()
            self.parents.clear()
            self.children.clear()
            self.arrivals.clear()
            self.pending_outputs = 0
        if self.stats_task is not None:
            self.stats_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.stats_task
        try:
            await self.stack.aclose()
        finally:
            from vllm.v1.metrics.prometheus import shutdown_prometheus

            shutdown_prometheus()
