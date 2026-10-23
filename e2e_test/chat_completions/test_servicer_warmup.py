"""The servicer's planned warm-up at its full level: once per engine and run.

Every gRPC servicer runs a planned warm-up before it reports SERVING. The
e2e legs start their workers with it at the ``fast`` level (the job sets
``SMG_<ENGINE>_SERVICER_WARMUP=fast``), so a chat suite of eight worker
starts does not pay the full pass eight times. This case starts one worker
at the shipped default, ``full``, and checks what the legs' level no longer
shows: the pass plans at that level, runs its steps to their end inside its
budget with no failed request, and the servicer answers SERVING only once it
is over. An engine whose plan also visits every prefill length up to its
chunk size (the chunk remainders, after every step) may have the budget end
those, and only those: the case pins that engine's chunk size and accepts
such an end.
"""

from __future__ import annotations

import logging
import os

import pytest
from infra import (
    DEFAULT_STARTUP_TIMEOUT,
    ConnectionMode,
    effective_startup_timeout,
    get_connection_mode_override,
    get_pool,
    get_runtime,
    start_workers,
    stop_workers,
)
from infra.constants import ENV_SHOW_WORKER_LOGS
from infra.model_specs import get_model_spec
from infra.servicer_warmup import (
    LEVEL_FULL,
    SERVICER_LOG_ENGINE_ARGS,
    SERVICER_WARMUP_ENV,
    watch_warmup,
)

logger = logging.getLogger(__name__)

# The suite's smallest worker on each of these engines (test_validation.py
# runs it on all three): the full pass costs what the model costs.
MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.engine("sglang", "vllm", "tokenspeed")
@pytest.mark.gpu(1)
@pytest.mark.e2e
@pytest.mark.model(MODEL)
class TestFullWarmupBeforeServing:
    def test_the_full_pass_ends_inside_its_budget_before_the_servicer_serves(self):
        if get_connection_mode_override() == ConnectionMode.ZMQ:
            pytest.skip("a ZMQ engine has no health service of its own: the gateway gates it")
        if os.environ.get(ENV_SHOW_WORKER_LOGS, "0") == "1":
            pytest.skip(f"{ENV_SHOW_WORKER_LOGS}=1 sends the worker's output to the terminal")
        engine = get_runtime()
        knob = SERVICER_WARMUP_ENV[engine]

        # The session pool may hold the previous class's worker on this GPU
        # slice; a closed pool is replaced on the next acquire.
        pool = get_pool()
        if pool.current_workers():
            pool.cleanup()

        # The engine's own start is bounded as any worker's start is; the pass
        # then gets its own budget on top, read off the worker's log, which
        # the engine must write at the level the servicer's lines carry.
        spec = get_model_spec(MODEL)
        engine_start_secs = effective_startup_timeout(
            spec.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)
        )
        # The engine writes its log at the level the servicer's lines carry
        # (the table's args), so the pass's lines can be read off the worker's log.
        extra_engine_args = list(SERVICER_LOG_ENGINE_ARGS[engine])
        # TokenSpeed's full plan runs the chunk remainders, one-token requests
        # at every prefill length up to the chunk size, last and under the same
        # budget; the harness pins no chunk size, so the engine's own step
        # (8,192, or its prefill budget) would hand thousands of them to the
        # 900 s budget. Pinned, the count is known and a normal run ends complete.
        if engine == "tokenspeed":
            extra_engine_args += ["--chunked-prefill-size", "2048"]
        worker = start_workers(
            MODEL,
            engine,
            mode=ConnectionMode.GRPC,
            count=1,
            wait_ready=False,
            extra_engine_args=extra_engine_args,
            extra_env={knob: LEVEL_FULL},
        )[0]
        try:
            outcome = watch_warmup(worker, engine_start_secs=engine_start_secs)
        finally:
            stop_workers([worker])
        logger.info("%s=%s on %s: %s", knob, LEVEL_FULL, engine, outcome.describe())

        assert outcome.start.level == LEVEL_FULL, (
            f"the worker's own {knob}={LEVEL_FULL} did not win over the job's level: "
            f"{outcome.describe()}"
        )
        # A failed request leaves its step uncovered, so the pass would read as
        # not complete too: the failures are named first.
        assert outcome.end.failed_requests == 0, (
            f"the pass had failed requests: {outcome.describe()}"
        )
        # The chunk remainders are the plan's best effort under the budget: a
        # pass the budget ended there, every step before them covered and no
        # request failed, is the finished warm-up this case checks for.
        assert outcome.end.complete or outcome.end.budget_ended_in_the_remainders, (
            f"the pass did not run to its end: {outcome.describe()}"
        )
