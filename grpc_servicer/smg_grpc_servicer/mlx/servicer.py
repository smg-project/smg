"""
MLX Engine gRPC Servicer

Implements the MlxEngine proto service backed by mlx-lm's BatchGenerator
for Apple Silicon inference.
"""

import asyncio
import logging
import os
import threading
import time

import grpc
import mlx.core as mx
from mlx_lm.generate import BatchGenerator, SequenceStateMachine
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from smg_grpc_proto import mlx_engine_pb2, mlx_engine_pb2_grpc

from smg_grpc_servicer.tokenizer_bundle import tokenizer_bundle, tokenizer_chunks

logger = logging.getLogger(__name__)


def _set_future_result_safe(future: asyncio.Future, result) -> None:
    """``future.set_result`` that no-ops on already-completed futures.

    Used by the gen thread to wake a Generate awaiter via
    ``call_soon_threadsafe``. Generate's task may be cancelled (which
    cancels the future) between the time the gen thread schedules this
    callback and the time the loop runs it; ``set_result`` would then
    raise ``InvalidStateError``. Cleanup of the inserted batch slot is
    handled by Generate's CancelledError path.
    """
    if not future.done():
        future.set_result(result)


def _set_future_exception_safe(future: asyncio.Future, exc: BaseException) -> None:
    """``future.set_exception`` that no-ops on already-completed futures.

    Same race as :func:`_set_future_result_safe`.
    """
    if not future.done():
        future.set_exception(exc)


class _PendingRequest:
    """A Generate() call queued to enter the next fresh batch.

    Holds the inputs we need to feed BatchGenerator.insert() plus an
    asyncio.Future that the generation thread resolves with the assigned
    uid once the request actually enters the batch. Generate() awaits
    that future before it starts pulling tokens off ``queue``.
    """

    __slots__ = (
        "token_ids",
        "max_tokens",
        "sampler",
        "logits_processors",
        "state_machine",
        "queue",
        "uid_future",
        "request_id",
    )

    def __init__(
        self,
        token_ids,
        max_tokens,
        sampler,
        logits_processors,
        state_machine,
        queue,
        uid_future,
        request_id,
    ):
        self.token_ids = token_ids
        self.max_tokens = max_tokens
        self.sampler = sampler
        self.logits_processors = logits_processors
        self.state_machine = state_machine
        self.queue = queue
        self.uid_future = uid_future
        self.request_id = request_id


class MlxEngineServicer(mlx_engine_pb2_grpc.MlxEngineServicer):
    """gRPC servicer implementing the MlxEngine service for MLX backends.

    Concurrency model: per-step admission (mlx-lm.server-style)
    ----------------------------------------------------------
    The earlier drain-and-batch model — wait for ``_active_uids`` to be
    empty before allowing inserts — was a workaround for a
    cross-thread mlx-state corruption that surfaced as

        ValueError: [rope] offset must be a scalar or vector with N
            elements but has shape (N-1).

    inside ``mx.fast.rope`` (PR #1414). The drain wait paid for that
    correctness with a TTFT regression at high concurrency-to-batch-size
    ratio: a request arriving mid-decode of a 4-way batch had to wait
    for all four to finish (~3 s for chat) before its prefill could
    start.

    The actual root cause turned out to be threading, not insert timing.
    ``mlx_lm.generate.generation_stream`` is allocated by
    ``mx.new_thread_local_stream(...)``; mlx's ``mx.stream(s)`` context
    is per-thread. When the BatchGenerator is constructed on thread A
    (the asyncio main thread, in the original design) and ``next()``
    runs on thread B (the gen thread), the stream object's per-thread
    binding doesn't follow it — mx kernel calls and ``mx.async_eval``
    continuations later raise "no Stream(gpu, 1) in current thread".
    That's the same threading bug that made concurrent insert-during-
    decode unsafe in our setup, but mlx-lm.server doesn't hit either
    failure because it runs all mlx state on a single dispatch thread.

    This servicer now mirrors mlx-lm.server's design:

      * The BatchGenerator is constructed inside ``_generation_loop``
        on the gen thread, so its thread-local stream binds to that
        thread for the lifetime of the process. All ``insert()``,
        ``next()``, and ``remove()`` calls run on that same thread.
      * Per-step admission. Each iteration of the loop drains
        ``_pending`` (regardless of whether the batch is empty), calls
        ``insert()``, then advances by exactly one ``next()`` step.
        Worst-case admission delay is one decode step (~50 ms),
        matching mlx-lm.server's main loop.

    Flow:

      * Incoming ``Generate`` calls build a :class:`_PendingRequest` and
        push it onto ``self._pending``, then await ``uid_future``.
      * The gen thread, every iteration, drains ``_pending`` and calls
        ``BatchGenerator.insert()`` (which only appends to the
        ``_unprocessed_sequences`` deque — fast, no batch shape
        mutation). Then ``BatchGenerator.next()`` pulls from that deque
        into the prefill batch and advances generation by one token.
      * Each request's ``uid_future`` is resolved as soon as its uid is
        known so ``Generate`` can register its uid for ``Abort`` and
        start consuming tokens from its per-uid queue.

    Thread-safety: mlx-lm.server's signal pattern
    ---------------------------------------------
    Every mlx-state mutation — ``insert``, ``next``, ``remove`` on
    ``BatchGenerator``, plus all mutations of ``_active_uids`` /
    ``_uid_queues`` / ``_request_uid_map`` — runs on the gen thread.
    Event-loop callers (``Generate``'s ``finally`` / ``CancelledError``
    handler, ``Abort``) communicate with the gen thread by appending
    request_ids to ``self._aborted_request_ids`` (a set guarded by
    ``_pending_lock``). The gen thread drains that set at the start
    of each iteration and does all the cleanup work itself.

    This mirrors ``mlx_lm/server.py`` (``ctx.stop()`` flips
    ``_should_stop``; the gen thread observes it on the next
    iteration). Their pattern works because everything runs on one
    thread; ours works because the asyncio→gen-thread channel is a
    single shared set, and the gen thread reads it at a known point
    in each iteration.

    Concrete consequences:

      * ``Abort`` is non-blocking. It just adds to a set and returns;
        cleanup happens within one decode-step (~50 ms) on the gen
        thread.
      * No ``_gen_lock``. There is no shared mutable state between
        the gen thread and event-loop callers other than the
        ``_pending_lock``-guarded fields.
      * The race the prior lock-based fix closed (``Abort`` arriving
        between gen's ``_pending_by_request_id.pop`` and
        ``_request_uid_map[rid] = uid``) goes away naturally:
        ``Abort`` records the rid, the gen thread observes it
        ≥1 iteration later, by which time the prior iteration's
        transition is fully committed and ``_request_uid_map[rid]``
        is reliably populated.

    ``self._pending_lock`` is the only lock. It guards
    ``_pending`` / ``_pending_by_request_id`` (request submission)
    and ``_aborted_request_ids`` (cleanup signal). Held only briefly
    each time, never nested with anything else.

    Cost model: ``Abort`` returns within microseconds (one
    set-add). Cleanup of an aborted request lags by at most one
    decode step (~10–50 ms on M-series).
    """

    def __init__(
        self,
        *,
        model,
        completion_batch_size: int,
        prefill_batch_size: int,
        model_path,
        model_dir,
        model_config,
        eos_token_ids,
        start_time,
    ):
        # The BatchGenerator is constructed lazily on the gen thread (see
        # class docstring). Until then `batch_generator is None`.
        self._model = model
        self._completion_batch_size = completion_batch_size
        self._prefill_batch_size = prefill_batch_size
        self.batch_generator = None
        self.model_path = model_path
        self.model_dir = model_dir
        self.model_config = model_config
        self._eos_token_ids = eos_token_ids
        self.start_time = start_time
        self._active_requests = 0
        # Gen-thread-only state — every mutation happens on the gen
        # thread. Event-loop callers never touch these dicts/set
        # directly; they signal the gen thread via
        # ``_aborted_request_ids`` and it does the cleanup.
        self._request_uid_map: dict[str, int] = {}
        self._uid_queues: dict[int, asyncio.Queue] = {}
        self._active_uids: set[int] = set()
        self._shutdown_event = threading.Event()
        # Set by the gen thread once BatchGenerator is constructed and
        # warmup has completed; ``server.serve_grpc`` waits on this
        # before flipping the health check to SERVING so that no
        # Generate RPC arrives before there's a BatchGenerator to
        # insert into. ``_construction_failed`` lets ``wait_ready``
        # report failure to the startup path even though the event
        # itself was set (so waiters unblock instead of hanging).
        self._ready_event = threading.Event()
        self._construction_failed = False
        self._loop = None
        self._gen_thread = None
        # Per-step admission state. New ``Generate`` calls land here
        # and the gen thread drains them at the top of every iteration
        # (regardless of ``_active_uids``). Indexed by request_id so
        # ``Abort`` can cancel a request that hasn't entered the
        # batch yet.
        self._pending: list[_PendingRequest] = []
        self._pending_by_request_id: dict[str, _PendingRequest] = {}
        # mlx-lm.server-style abort signal. Event-loop callers
        # (``Generate``'s ``finally`` / ``CancelledError`` handler,
        # ``Abort``) add request_ids here; the gen thread drains the
        # set between phase 1 (insert) and phase 2 (next) and does
        # all cleanup work itself, keeping every mlx-state mutation
        # on one thread. See class docstring for the analogy to
        # mlx_lm/server.py's ``ctx.stop()`` pattern.
        self._aborted_request_ids: set[str] = set()
        self._pending_lock = threading.Lock()
        # Admission coalescing. When a fresh batch is forming (nothing
        # generating yet) and only part of a concurrent burst has landed in
        # _pending, a single next() prefill chunk (seconds long for a
        # multi-thousand-token prompt) blocks admission of the siblings that
        # arrive microseconds later — splitting one wave into misaligned
        # prefill groups and inflating TTFT ~50% at concurrency. We wait for
        # the burst to finish arriving before kicking off the first chunk.
        #
        # Adaptive: poll _pending every _coalesce_tick; proceed as soon as it
        # stops growing (burst done) or the batch is full, capped at
        # _coalesce_cap. A lone request therefore pays ~one tick, not the full
        # cap, while a real burst is fully coalesced. Both are negligible next
        # to a multi-second prefill, and the wait is skipped once generating.
        self._coalesce_cap = float(os.environ.get("MLX_COALESCE_MS", "50")) / 1000.0
        self._coalesce_tick = float(os.environ.get("MLX_COALESCE_TICK_MS", "5")) / 1000.0
        # Prompt prefill chunk size (tokens per next() prefill step). Smaller
        # chunks shorten the admission-blocking window, but benchmarking found
        # this insufficient alone (still above floor) and costlier in
        # throughput from extra kernel launches — admission coalescing above is
        # the real fix. Exposed as a tuning lever; default matches mlx-lm (2048)
        # so behaviour is unchanged unless explicitly overridden.
        self._prefill_step_size = int(os.environ.get("MLX_PREFILL_STEP", "2048"))
        # Resolve context length once — config doesn't change at runtime,
        # and Generate was previously scanning these keys on every request.
        self._ctx_limit = 0
        for key in ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length"):
            val = model_config.get(key)
            if isinstance(val, int) and val > 0:
                self._ctx_limit = val
                break
        logger.info("MlxEngineServicer initialized for model %s", model_path)

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until the gen thread has constructed BatchGenerator + warmed up.

        Called from ``server.serve_grpc`` (in an executor thread so the
        asyncio loop isn't blocked) before flipping the health probe
        to SERVING. Returns ``True`` when ready; ``False`` if the
        servicer was shut down before becoming ready, or if
        BatchGenerator construction raised on the gen thread (in which
        case the gen thread sets ``_construction_failed`` then sets the
        event to unblock this waiter).
        """
        # Poll so a shutdown signal during warmup unblocks the waiter.
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while not self._ready_event.is_set():
            if self._shutdown_event.is_set():
                return False
            wait = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            self._ready_event.wait(wait)
        return not self._construction_failed

    def _warmup(self) -> None:
        """Run one end-to-end token through the batch generator so the
        first real request doesn't pay JIT/kernel compilation cost.

        Runs ON the gen thread, after BatchGenerator construction and
        before the main per-step loop, so the warmup also exercises
        the same thread-local stream binding the bench traffic will use.
        """
        logger.info("Running warmup generation...")
        uids = None
        try:
            uids = self.batch_generator.insert(prompts=[[1]], max_tokens=[1])
            for _ in range(10):
                _, gen_responses = self.batch_generator.next()
                if any(r.finish_reason is not None for r in gen_responses if r.uid == uids[0]):
                    break
            logger.info("Warmup complete")
        except Exception:
            logger.warning("Warmup failed (non-fatal)", exc_info=True)
        finally:
            # Always clean up the warmup probe even if next() raised
            # mid-iteration. Otherwise the warmup uid leaks inside
            # BatchGenerator and the first real request runs against
            # corrupted batch state.
            if uids is not None:
                try:
                    self.batch_generator.remove(uids)
                except Exception:
                    logger.warning("Warmup cleanup failed", exc_info=True)

    @staticmethod
    def _build_sampler(sampling_params):
        """Convert proto SamplingParams to an mlx-lm sampler callable."""
        # When temperature is unset, default to 1.0 to match vLLM/SGLang/TRT-LLM
        # behavior. mlx-lm's make_sampler defaults to 0.0 (greedy), which would
        # silently diverge for requests that omit temperature.
        temp = sampling_params.temperature if sampling_params.HasField("temperature") else 1.0
        return make_sampler(
            temp=temp,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
        )

    @staticmethod
    def _build_logits_processors(sampling_params):
        """Convert proto SamplingParams to a list of mlx-lm logits processors."""
        logit_bias = dict(sampling_params.logit_bias) if sampling_params.logit_bias else None
        rep_pen = sampling_params.repetition_penalty if sampling_params.repetition_penalty else None
        freq_pen = sampling_params.frequency_penalty if sampling_params.frequency_penalty else None
        pres_pen = sampling_params.presence_penalty if sampling_params.presence_penalty else None
        return make_logits_processors(
            logit_bias=logit_bias,
            repetition_penalty=rep_pen,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
        )

    @staticmethod
    def _build_state_machine(sampling_params, eos_token_ids):
        """Build a SequenceStateMachine from stop_token_ids and EOS tokens."""
        stop_sequences = []

        if not sampling_params.ignore_eos:
            for eos_id in eos_token_ids:
                stop_sequences.append(((eos_id,), None))

        for tid in sampling_params.stop_token_ids:
            stop_sequences.append(((tid,), None))

        if not stop_sequences:
            return SequenceStateMachine()

        return SequenceStateMachine(
            transitions={"normal": stop_sequences},
            initial="normal",
        )

    @staticmethod
    def _matched_stop_token(response):
        """Return the matched stop token id if the response matched a single-token stop."""
        ms = response.match_sequence
        return ms[0] if ms and len(ms) == 1 else None

    @staticmethod
    def _build_output_logprobs(token_id, logprobs_array, num_logprobs):
        """Build OutputLogProbs proto from an mlx logprobs array."""
        # num_logprobs == 0 would make top_k == 0 and `[-0:]` would slice the
        # entire vocabulary — guard explicitly.
        if num_logprobs is None or num_logprobs <= 0:
            return None

        token_logprob = logprobs_array[token_id].item()

        top_k = min(num_logprobs, logprobs_array.shape[0])
        top_indices = mx.argpartition(logprobs_array, kth=-top_k)[-top_k:]
        top_values = logprobs_array[top_indices]
        sort_order = mx.argsort(top_values)[::-1]
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]

        top_logprobs = mlx_engine_pb2.TopLogProbs(
            token_ids=[int(i) for i in top_indices.tolist()],
            values=[float(v) for v in top_values.tolist()],
        )

        return mlx_engine_pb2.OutputLogProbs(
            token_ids=[token_id],
            token_logprobs=[token_logprob],
            top_logprobs=[top_logprobs],
        )

    @staticmethod
    def _chunk_response(
        token_ids, prompt_tokens, completion_tokens, cached_tokens, index, output_logprobs=None
    ):
        """Build a GenerateStreamChunk response."""
        chunk = mlx_engine_pb2.GenerateStreamChunk(
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            index=index,
        )
        if output_logprobs is not None:
            chunk.output_logprobs.CopyFrom(output_logprobs)
        return mlx_engine_pb2.GenerateResponse(chunk=chunk)

    @staticmethod
    def _complete_response(
        output_ids,
        finish_reason,
        prompt_tokens,
        completion_tokens,
        cached_tokens,
        index,
        output_logprobs=None,
        matched_token_id=None,
    ):
        """Build a GenerateComplete response."""
        kwargs = {}
        if matched_token_id is not None:
            kwargs["matched_stop_token_id"] = matched_token_id

        complete = mlx_engine_pb2.GenerateComplete(
            output_ids=output_ids,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            index=index,
            **kwargs,
        )
        if output_logprobs is not None:
            complete.output_logprobs.CopyFrom(output_logprobs)
        return mlx_engine_pb2.GenerateResponse(complete=complete)

    async def GetModelInfo(
        self,
        request: mlx_engine_pb2.GetModelInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> mlx_engine_pb2.GetModelInfoResponse:
        config = self.model_config

        # Reuse the resolved EOS IDs so GetModelInfo agrees with the stop
        # behavior we actually apply in generation (server.py falls back to
        # tokenizer-derived IDs when config.json has none).
        eos_token_ids = list(self._eos_token_ids)

        # Use the pre-resolved context limit so GetModelInfo reports the
        # same value Generate enforces (config keys vary across model
        # families — see __init__).
        return mlx_engine_pb2.GetModelInfoResponse(
            model_path=self.model_path,
            is_generation=True,
            max_context_length=self._ctx_limit,
            vocab_size=config.get("vocab_size", 0),
            served_model_name=self.model_path,
            model_type=config.get("model_type", ""),
            architectures=config.get("architectures", []),
            eos_token_ids=eos_token_ids,
            pad_token_id=config.get("pad_token_id") or 0,
            bos_token_id=config.get("bos_token_id") or 0,
            max_req_input_len=self._ctx_limit,
        )

    async def GetServerInfo(
        self,
        request: mlx_engine_pb2.GetServerInfoRequest,
        context: grpc.aio.ServicerContext,
    ) -> mlx_engine_pb2.GetServerInfoResponse:
        return mlx_engine_pb2.GetServerInfoResponse(
            server_type="mlx-grpc",
            active_requests=self._active_requests,
            uptime_seconds=time.time() - self.start_time,
        )

    def start_generation_loop(self):
        self._loop = asyncio.get_running_loop()
        self._gen_thread = threading.Thread(
            target=self._generation_loop, daemon=True, name="mlx-gen-loop"
        )
        self._gen_thread.start()
        logger.info("Generation loop started")

    def stop_generation_loop(self):
        self._shutdown_event.set()
        if self._gen_thread and self._gen_thread.is_alive():
            self._gen_thread.join(timeout=5.0)
        logger.info("Generation loop stopped")

    def _generation_loop(self):
        # Construct the BatchGenerator HERE on the gen thread so its
        # thread-local mlx stream binds to this thread for life. All
        # subsequent insert/next/remove calls happen on this same
        # thread, matching mlx-lm.server's single-threaded mlx state
        # invariant. See class docstring for why cross-thread mlx
        # state was the underlying cause of both the rope crash from
        # PR #1414 and the "no Stream(gpu, 1) in current thread"
        # RuntimeError seen at concurrency 4.
        try:
            self.batch_generator = BatchGenerator(
                self._model,
                completion_batch_size=self._completion_batch_size,
                prefill_batch_size=self._prefill_batch_size,
                prefill_step_size=self._prefill_step_size,
            )
        except Exception:
            logger.exception("BatchGenerator construction failed")
            # Flag the failure BEFORE setting the event so wait_ready
            # observes the flag (the event is set last, after the
            # write — readers re-check the flag once unblocked).
            self._construction_failed = True
            self._ready_event.set()
            return
        logger.info(
            "BatchGenerator created on gen thread (prefill=%d, completion=%d)",
            self._prefill_batch_size,
            self._completion_batch_size,
        )

        # Warmup before signalling ready so the first real Generate RPC
        # doesn't pay JIT/kernel compilation cost.
        self._warmup()
        self._ready_event.set()

        # Per-step admission loop. Every iteration:
        #   1. Drain _pending into BatchGenerator.insert(...) (deque
        #      append on the gen thread — fast, no batch shape mutation).
        #   2. Advance the batch by exactly one BatchGenerator.next()
        #      step. next() pulls from _unprocessed_sequences into the
        #      prefill batch, runs one prefill chunk and one decode
        #      token, and returns responses.
        # Worst-case admission delay for a request that arrives just
        # after a next() call begins: one decode step (~50 ms on
        # M-series), matching mlx-lm.server's loop.
        while not self._shutdown_event.is_set():
            prompt_responses: list = []
            gen_responses: list = []
            try:
                # Phase 0: coalesce a concurrent burst into one prefill batch.
                # Only when idle (a fresh batch is forming): poll _pending and
                # proceed as soon as it stops growing or fills a prefill batch,
                # capped at _coalesce_cap. Skipped entirely once a batch is
                # generating, preserving the per-step (mlx-lm.server-style)
                # mid-decode admission latency.
                if self._coalesce_cap and self._coalesce_tick and not self._active_uids:
                    with self._pending_lock:
                        n_pending = len(self._pending)
                    waited = 0.0
                    while 0 < n_pending < self._prefill_batch_size and waited < self._coalesce_cap:
                        time.sleep(self._coalesce_tick)
                        waited += self._coalesce_tick
                        with self._pending_lock:
                            n_now = len(self._pending)
                        if n_now == n_pending:
                            break  # burst stopped growing — admit what we have
                        n_pending = n_now

                # Phase 1: admit pending. NOT gated on _active_uids —
                # pending requests can join while a batch is mid-decode,
                # the whole point of mlx-lm.server-style scheduling.
                with self._pending_lock:
                    batch = self._pending[:]
                    self._pending.clear()
                    # Don't pop from _pending_by_request_id yet:
                    # keeping pending entries indexed until insert()
                    # succeeds lets Abort() cancel a request that
                    # lost the insert race.

                batch = [p for p in batch if not p.uid_future.cancelled()]

                if batch:
                    try:
                        uids = self.batch_generator.insert(
                            prompts=[p.token_ids for p in batch],
                            max_tokens=[p.max_tokens for p in batch],
                            samplers=[p.sampler for p in batch],
                            logits_processors=[p.logits_processors for p in batch],
                            state_machines=[p.state_machine for p in batch],
                        )
                    except Exception as e:
                        # Wake every waiter with a real error so
                        # Generate exits with INTERNAL instead of
                        # hanging on uid_future forever.
                        logger.exception("BatchGenerator.insert failed for batch of %d", len(batch))
                        with self._pending_lock:
                            for p in batch:
                                self._pending_by_request_id.pop(p.request_id, None)
                        for p in batch:
                            self._loop.call_soon_threadsafe(
                                _set_future_exception_safe, p.uid_future, e
                            )
                        continue

                    # insert() succeeded — finalize each request.
                    with self._pending_lock:
                        for p in batch:
                            self._pending_by_request_id.pop(p.request_id, None)
                    for uid, p in zip(uids, batch):
                        self._uid_queues[uid] = p.queue
                        self._active_uids.add(uid)
                        self._request_uid_map[p.request_id] = uid
                        self._loop.call_soon_threadsafe(_set_future_result_safe, p.uid_future, uid)

                # Phase 1.5: drain abort signals (mlx-lm.server-style
                # ctx.stop() equivalent). Generate.finally,
                # Generate.CancelledError, and Abort all add request_ids
                # to _aborted_request_ids. We do all the cleanup here
                # so every mlx-state mutation stays on the gen thread.
                with self._pending_lock:
                    aborted = self._aborted_request_ids
                    self._aborted_request_ids = set()

                for rid in aborted:
                    # Case A: request still pending — never inserted.
                    with self._pending_lock:
                        pending = self._pending_by_request_id.pop(rid, None)
                        if pending is not None:
                            try:
                                self._pending.remove(pending)
                            except ValueError:
                                pass
                    if pending is not None:
                        # Cancel the future so Generate's await raises
                        # CancelledError. Already-done futures (e.g.,
                        # natural completion of a request that never
                        # got past the insert race) are no-ops.
                        if not pending.uid_future.done():
                            self._loop.call_soon_threadsafe(pending.uid_future.cancel)
                        continue

                    # Case B: already inserted (or never registered —
                    # e.g., Generate.finally called for a request that
                    # was never accepted). Look up uid + clean up.
                    uid = self._request_uid_map.pop(rid, None)
                    if uid is None:
                        # Either never inserted, or already cleaned up
                        # by a prior abort drain. Nothing to do.
                        continue

                    queue = self._uid_queues.pop(uid, None)
                    if queue is not None:
                        # Drain buffered tokens so a still-streaming
                        # consumer stops emitting immediately rather
                        # than flushing stale chunks before seeing the
                        # sentinel.
                        while not queue.empty():
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        self._loop.call_soon_threadsafe(queue.put_nowait, None)

                    if uid in self._active_uids:
                        try:
                            self.batch_generator.remove([uid])
                            self._active_uids.discard(uid)
                        except Exception:
                            logger.exception(
                                "BatchGenerator.remove failed for uid %d during abort drain",
                                uid,
                            )
                    # else: gen thread already removed the uid inline
                    # via the finish_reason path on a prior iteration.
                    # Generate.finally fired afterward; nothing to do
                    # at the backend level.

                # Phase 2: advance one step. Skip when nothing is in
                # flight — next() on an empty BatchGenerator is wasted
                # work.
                if self._active_uids:
                    # BatchGenerator.next() wraps itself in
                    # `with mx.stream(self._stream):` internally
                    # (mlx_lm/generate.py:1847), so no outer wrap
                    # needed.
                    prompt_responses, gen_responses = self.batch_generator.next()

                    for r in gen_responses:
                        queue = self._uid_queues.get(r.uid)
                        if queue is not None:
                            self._loop.call_soon_threadsafe(queue.put_nowait, r)
                        if r.finish_reason is not None:
                            # Only discard from _active_uids on a
                            # successful remove; if remove fails for a
                            # real backend reason the uid stays
                            # tracked so accounting isn't lost.
                            try:
                                self.batch_generator.remove([r.uid])
                                self._active_uids.discard(r.uid)
                            except Exception:
                                logger.exception("BatchGenerator.remove failed for uid %d", r.uid)
            except Exception:
                logger.exception("Error in generation loop")
                continue

            # Idle sleep only when there's truly nothing to do.
            if not prompt_responses and not gen_responses and not self._active_uids:
                with self._pending_lock:
                    nothing_to_do = len(self._pending) == 0 and len(self._aborted_request_ids) == 0
                if nothing_to_do:
                    time.sleep(0.001)

        # Shutdown — runs on the gen thread, after the main loop
        # exits. No lock needed: every mlx-state field below is
        # gen-thread-only mutated, and event-loop callers can only
        # add to _pending / _aborted_request_ids under _pending_lock,
        # which we acquire briefly when clearing those fields.
        # Clearing self.batch_generator unconditionally — any
        # event-loop caller that races past server.stop()'s grace
        # period only adds to _aborted_request_ids; it never touches
        # batch_generator directly.
        with self._pending_lock:
            pending = self._pending[:]
            self._pending.clear()
            self._aborted_request_ids.clear()
            for p in pending:
                self._pending_by_request_id.pop(p.request_id, None)

        # Wake any RPCs still waiting. Without this, Generate calls
        # blocked on `await uid_future` (not yet inserted) or
        # `await queue.get()` (mid-stream) would hang until the
        # client deadline or transport cancellation.
        shutdown_exc = RuntimeError("MlxEngineServicer is shutting down")
        for p in pending:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(
                    _set_future_exception_safe, p.uid_future, shutdown_exc
                )

        queues = list(self._uid_queues.values())
        self._uid_queues.clear()
        self._request_uid_map.clear()
        self._active_uids.clear()
        for queue in queues:
            # Sentinel — Generate's stream/non-stream loops both
            # treat None as "Abort received, stop emitting".
            if self._loop is not None:
                self._loop.call_soon_threadsafe(queue.put_nowait, None)

        if self.batch_generator is not None:
            try:
                self.batch_generator.close()
            except Exception:
                logger.warning("BatchGenerator.close raised", exc_info=True)
            finally:
                self.batch_generator = None

    async def Generate(self, request, context):
        request_id = request.request_id
        try:
            input_type = request.WhichOneof("input")
            if input_type != "tokenized":
                raise ValueError("MLX servicer requires tokenized input")

            token_ids = list(request.tokenized.input_ids)
            sp = request.sampling_params

            sampler = self._build_sampler(sp)
            logits_processors = self._build_logits_processors(sp)
            state_machine = self._build_state_machine(sp, self._eos_token_ids)
            # When max_tokens is unset, cap at remaining context (matches
            # vLLM/SGLang semantics: unbounded within model limits, not a
            # silent 256-token truncation). Fall back to 256 if the model
            # config didn't advertise a context length.
            if sp.HasField("max_tokens"):
                max_tokens = sp.max_tokens
            elif self._ctx_limit > 0:
                max_tokens = max(self._ctx_limit - len(token_ids), 1)
            else:
                max_tokens = 256
            num_logprobs = sp.logprobs if sp.HasField("logprobs") else None

            if sp.HasField("seed"):
                mx.random.seed(sp.seed)

            queue: asyncio.Queue = asyncio.Queue()
            uid_future: asyncio.Future = asyncio.get_running_loop().create_future()
            pending = _PendingRequest(
                token_ids=token_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                state_machine=state_machine,
                queue=queue,
                uid_future=uid_future,
                request_id=request_id,
            )
            # Hand off to the gen thread. It'll insert this request into
            # the next fresh batch and resolve uid_future once the uid
            # is assigned. We register on _pending_by_request_id first so
            # an Abort that races with this append can still find us.
            with self._pending_lock:
                self._pending_by_request_id[request_id] = pending
                self._pending.append(pending)
            try:
                # Wait for the gen thread to actually insert this
                # request. The uid value isn't needed by Generate
                # anymore (gen thread owns _request_uid_map /
                # _uid_queues / _active_uids); we just need the
                # synchronization that the gen thread has admitted us.
                await uid_future
            except asyncio.CancelledError:
                # Signal the gen thread to clean up — whether the
                # request was still pending, already inserted, or in
                # the transient gap between the two. The gen thread
                # observes this on its next iteration and does all
                # the necessary backend cleanup itself, so we don't
                # touch _request_uid_map / _uid_queues / batch_generator
                # from the asyncio thread.
                with self._pending_lock:
                    self._aborted_request_ids.add(request_id)
                raise
            self._active_requests += 1
            prompt_tokens = len(token_ids)

            try:
                if request.stream:
                    completion_tokens = 0
                    while True:
                        r = await queue.get()
                        if r is None:
                            # Sentinel from Abort — terminate the stream.
                            break
                        completion_tokens += 1
                        yield self._chunk_response(
                            token_ids=[r.token],
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            cached_tokens=0,
                            index=0,
                            output_logprobs=self._build_output_logprobs(
                                r.token, r.logprobs, num_logprobs
                            ),
                        )
                        if r.finish_reason is not None:
                            yield self._complete_response(
                                output_ids=[],
                                finish_reason=r.finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                cached_tokens=0,
                                index=0,
                                matched_token_id=self._matched_stop_token(r),
                            )
                            break
                else:
                    all_output_ids = []
                    # Aggregate per-token logprobs across the whole sequence so
                    # the final GenerateComplete carries logprobs for every
                    # generated token (not just the last step).
                    agg_token_ids: list[int] = []
                    agg_token_logprobs: list[float] = []
                    agg_top: list = []
                    while True:
                        r = await queue.get()
                        if r is None:
                            # Sentinel from Abort — terminate without emitting.
                            break
                        all_output_ids.append(r.token)
                        step = self._build_output_logprobs(r.token, r.logprobs, num_logprobs)
                        if step is not None:
                            agg_token_ids.extend(step.token_ids)
                            agg_token_logprobs.extend(step.token_logprobs)
                            agg_top.extend(step.top_logprobs)
                        if r.finish_reason is not None:
                            seq_logprobs = None
                            if agg_token_ids:
                                seq_logprobs = mlx_engine_pb2.OutputLogProbs(
                                    token_ids=agg_token_ids,
                                    token_logprobs=agg_token_logprobs,
                                    top_logprobs=agg_top,
                                )
                            yield self._complete_response(
                                output_ids=all_output_ids,
                                finish_reason=r.finish_reason,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=len(all_output_ids),
                                cached_tokens=0,
                                index=0,
                                output_logprobs=seq_logprobs,
                                matched_token_id=self._matched_stop_token(r),
                            )
                            break
            finally:
                self._active_requests -= 1
                # Signal the gen thread to clean up
                # _request_uid_map / _uid_queues / (if still in
                # _active_uids) BatchGenerator. Natural-completion
                # requests already had `_active_uids.discard(uid)` run
                # inline on the gen thread when finish_reason fired;
                # the abort drain just pops the asyncio-side index
                # entries. Disconnect-mid-stream requests get the full
                # cleanup including `batch_generator.remove`. Either
                # way, all mlx-state mutations stay on the gen thread.
                with self._pending_lock:
                    self._aborted_request_ids.add(request_id)

        except ValueError as e:
            logger.warning("Generate invalid request %s: %s", request_id, e)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("Generate failed for request %s", request_id)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def Abort(self, request, context):
        # mlx-lm.server-style: just record the request_ids in the
        # abort signal set and return. The gen thread observes the
        # set on its next iteration and does all the cleanup work
        # — pending lookup, uid lookup, queue wake, BatchGenerator
        # remove. Abort returns within microseconds; the actual
        # cleanup lags by at most one decode step.
        with self._pending_lock:
            for request_id in request.request_ids:
                self._aborted_request_ids.add(request_id)
        return mlx_engine_pb2.AbortResponse()

    async def HealthCheck(self, request, context):
        # Reflect actual servicer state so the router can stop routing to us
        # when the generation thread is dead or we're shutting down.
        if self._shutdown_event.is_set():
            return mlx_engine_pb2.HealthCheckResponse(
                healthy=False, message="servicer shutting down"
            )
        if self._gen_thread is None:
            return mlx_engine_pb2.HealthCheckResponse(
                healthy=False, message="generation loop not started"
            )
        if not self._gen_thread.is_alive():
            return mlx_engine_pb2.HealthCheckResponse(
                healthy=False, message="generation thread exited"
            )
        return mlx_engine_pb2.HealthCheckResponse(healthy=True, message="OK")

    async def GetTokenizer(self, request, context):
        """Stream the model directory's tokenizer files as the shared ZIP
        bundle: the selection, order and chunking the other engines' servicers
        send and the Router's loader expects, the archive's SHA-256 on the last
        chunk. The archive is built in a worker thread the first time and
        served from memory after that, so the streams keep flowing meanwhile.
        """
        try:
            zip_data, sha256 = await tokenizer_bundle(self.model_dir)
        except Exception as e:
            logger.exception("GetTokenizer failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
        for chunk in tokenizer_chunks(zip_data, sha256):
            yield chunk
