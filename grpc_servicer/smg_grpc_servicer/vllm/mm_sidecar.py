"""Reference media-processing sidecar for vLLM gRPC workers (GPU-free).

Pops jobs from Redis, fetches the media with vLLM's MediaConnector, runs vLLM's
own multimodal processor over the unexpanded prompt, and pushes full-tensor
results back. Built the way ``vllm launch render`` builds its renderer, so the
tensors match what the worker would have produced itself.

Usage:
    python -m smg_grpc_servicer.vllm.mm_sidecar --model <model> \\
        --redis-url redis://127.0.0.1:6379/0 [--namespace NS] [--concurrency N] \\
        [AsyncEngineArgs flags: --dtype --allowed-media-domains --media-io-kwargs ...]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from smg_grpc_servicer.mm_sidecar_protocol import (
    CODE_RESULT_PUSH_FAILED,
    CODE_RESULT_TOO_LARGE,
    DEFAULT_MAX_RESULT_BYTES,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_REDIS_URL,
    ENV_MAX_RESULT_BYTES,
    ENV_MAX_VIDEO_FRAMES,
    HELLO_REFRESH_S,
    HELLO_TTL_S,
    RESULT_TTL_S,
    SCHEMA_VERSION,
    Fingerprint,
    Job,
    JobResult,
    Keys,
    Placeholder,
    Timing,
    decode_job,
    encode_result,
    failure,
    resolve_namespace,
)
from smg_grpc_servicer.vllm.media_refs import advertised_schemes, parse_scheme_list, url_scheme
from smg_grpc_servicer.vllm.mm_processor import (
    clamp_video_frames,
    env_int,
    fingerprint_from_model_config,
    vllm_default_video_frames,
)

logger = logging.getLogger("smg_grpc_servicer.vllm.mm_sidecar")

# How long a worker waits for the next job before looking again.
JOB_WAIT_S = 5
# The margin on top of that before the wait is treated as a dead connection
# rather than an empty queue.
JOB_WAIT_MARGIN_S = 2
# How long a worker holds off after a refused or unanswered wait.
RECONNECT_PAUSE_S = 1
# The least time a finished job's answer gets to reach the requester.
PUSH_FLOOR_S = 5
# A failure notice is small; the second push gets at most this long.
FAILURE_PUSH_S = 5.0
# Redis's per-value cap; a push at or above it is refused, not queued.
REDIS_BULK_LIMIT = "proto-max-bulk-len"


def build_config(args: argparse.Namespace):
    """vLLM config for preprocessing only: no quantized kernels, no KV cache."""
    from vllm import envs
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import AsyncEngineArgs

    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    model_config.quantization = None
    envs.VLLM_CPU_KVCACHE_SPACE = 0
    return VllmConfig(model_config=model_config)


def config_fingerprint(vllm_config) -> Fingerprint:
    return fingerprint_from_model_config(vllm_config.model_config)


# Codes come from exception types where vLLM offers one. The two substring
# checks pin message wording from vLLM 0.27: connector.py's allowed-domain
# rejection ("... domain ...") and processing/context.py's per-prompt limit
# ("At most N ..."); test_vllm_mm_sidecar.py holds the wording.
def classify_fetch_error(exc: BaseException) -> str:
    if isinstance(exc, ValueError) and "domain" in str(exc).lower():
        return "domain_not_allowed"
    return "fetch_failed"


def classify_process_error(exc: BaseException) -> str:
    if isinstance(exc, RuntimeError):
        return "placeholder_mismatch"
    if isinstance(exc, ValueError):
        return "limit_exceeded" if "at most" in str(exc).lower() else "decode_failed"
    return "processor_error"


class Sidecar:
    def __init__(self, vllm_config, renderer, client, *, namespace: str | None, concurrency: int):
        from vllm import TokensPrompt, envs
        from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY
        from vllm.transformers_utils.processor import get_video_processor_cls_name
        from vllm.v1.serial_utils import MsgpackEncoder

        model_config = vllm_config.model_config
        mm_config = model_config.get_multimodal_config()
        self._renderer = renderer
        self._tokens_prompt = TokensPrompt
        # All tensors inline: one buffer per kwargs item.
        self._encoder = MsgpackEncoder(size_threshold=2**62)
        self._client = client
        self._concurrency = max(1, concurrency)
        self._fingerprint = config_fingerprint(vllm_config)
        self._keys = Keys.for_namespace(resolve_namespace(self._fingerprint, namespace))
        # Lowered further by redis's own cap once connected (`_learn_result_limit`).
        self._max_result_bytes = env_int(os.environ, ENV_MAX_RESULT_BYTES, DEFAULT_MAX_RESULT_BYTES)
        self._max_video_frames = env_int(
            os.environ, ENV_MAX_VIDEO_FRAMES, DEFAULT_MAX_VIDEO_FRAMES, minimum=0
        )
        # The fingerprint keeps the engine's kwargs: the budget bounds this
        # sidecar's sampling, it does not describe a different engine.
        self._connector = MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=clamp_video_frames(
                mm_config.media_io_kwargs, self._max_video_frames, vllm_default_video_frames()
            ),
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        self._video_processor = get_video_processor_cls_name(model_config)
        self._schemes = advertised_schemes(model_config.allowed_local_media_path)
        self._accepted = parse_scheme_list(self._schemes)
        self._started_at = time.time()
        if not model_config.allowed_media_domains:
            logger.warning(
                "no --allowed-media-domains: this sidecar will fetch media from any host"
            )

    async def run(self) -> None:
        await self._learn_result_limit()
        logger.info(
            "media sidecar serving under %s (concurrency=%d) fingerprint=%s",
            self._keys.prefix,
            self._concurrency,
            self._fingerprint.to_hello(),
        )
        tasks = [asyncio.create_task(self._heartbeat())]
        tasks += [asyncio.create_task(self._worker(i)) for i in range(self._concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()

    async def _learn_result_limit(self) -> None:
        """Cap results at redis's bulk limit when it is lower than the configured one.

        Managed redis often refuses CONFIG; the configured limit then stands.
        """
        try:
            reply = await asyncio.wait_for(self._client.config_get(REDIS_BULK_LIMIT), HELLO_TTL_S)
            (value,) = reply.values()
            redis_limit = int(value)
        except Exception as e:  # noqa: BLE001 - CONFIG unavailable or unreadable: keep the configured limit
            logger.info(
                "result limit %d bytes (%s not readable: %s)",
                self._max_result_bytes,
                REDIS_BULK_LIMIT,
                e,
            )
            return
        self._max_result_bytes = min(self._max_result_bytes, redis_limit)
        logger.info(
            "result limit %d bytes (%s=%d)", self._max_result_bytes, REDIS_BULK_LIMIT, redis_limit
        )

    async def _heartbeat(self) -> None:
        mapping = {
            **self._fingerprint.to_hello(),
            "schema": str(SCHEMA_VERSION),
            "schemes": self._schemes,
            "started_at": str(int(self._started_at)),
            "max_result_bytes": str(self._max_result_bytes),
        }
        while True:
            try:
                # One transaction: a hello can never outlive its TTL. The TTL
                # is also the longest this is worth waiting on, since a refresh
                # that lands later than that has already lapsed.
                pipe = self._client.pipeline(transaction=True)
                pipe.hset(self._keys.hello, mapping=mapping)
                pipe.expire(self._keys.hello, HELLO_TTL_S)
                await asyncio.wait_for(pipe.execute(), HELLO_TTL_S)
            except Exception as e:  # noqa: BLE001 - keep advertising through redis blips
                logger.warning("hello refresh failed: %s", e)
            await asyncio.sleep(HELLO_REFRESH_S)

    async def _worker(self, index: int) -> None:
        while True:
            try:
                # Redis gives up on its own, but only while it is still
                # answering. Without the outer bound a connection that goes
                # quiet takes this worker with it, and a sidecar whose workers
                # are all gone keeps advertising itself as ready.
                popped = await asyncio.wait_for(
                    self._client.brpop(self._keys.jobs, timeout=JOB_WAIT_S),
                    JOB_WAIT_S + JOB_WAIT_MARGIN_S,
                )
            except TimeoutError:
                logger.warning("worker %d: redis stopped answering; reconnecting", index)
                await asyncio.sleep(RECONNECT_PAUSE_S)
                continue
            except Exception as e:  # noqa: BLE001 - reconnect on the next iteration
                logger.warning("worker %d: brpop failed: %s", index, e)
                await asyncio.sleep(RECONNECT_PAUSE_S)
                continue
            if popped is None:
                continue
            _, raw = popped
            try:
                job = decode_job(raw)
            except Exception as e:  # noqa: BLE001 - a malformed job has no result key to answer on
                logger.error("worker %d: undecodable job dropped: %s", index, e)
                continue
            try:
                result = await self.handle(job)
            except Exception as e:  # noqa: BLE001 - one bad job must not kill the worker
                logger.exception("worker %d: job %s failed", index, job.job_id)
                result = failure(job.job_id, "processor_error", repr(e))
            raw = encode_result(result)
            if len(raw) >= self._max_result_bytes:
                # Redis would refuse the value; the requester gets told why instead.
                logger.warning(
                    "worker %d: result for %s is %d bytes, at or above the %d-byte transport "
                    "limit (%d items: %s); answering %s",
                    index,
                    job.job_id,
                    len(raw),
                    self._max_result_bytes,
                    len(job.items),
                    ",".join(sorted({item.modality for item in job.items})),
                    CODE_RESULT_TOO_LARGE,
                )
                result = failure(
                    job.job_id,
                    CODE_RESULT_TOO_LARGE,
                    f"encoded media result is {len(raw)} bytes, transport limit "
                    f"{self._max_result_bytes} bytes; reduce video length, fps or resolution",
                )
                raw = encode_result(result)
            await self._deliver(index, job, raw)

    async def _deliver(self, index: int, job: Job, raw: bytes) -> None:
        """Answer on the job's result key; a refused answer is replaced by a small one."""
        key = self._keys.result(job.job_id)
        try:
            await self._push(key, raw, self._push_budget(job))
        except Exception as first:  # noqa: BLE001 - the requester is told, not left to time out
            logger.warning("worker %d: result push failed for %s: %s", index, job.job_id, first)
            notice = failure(
                job.job_id, CODE_RESULT_PUSH_FAILED, f"{type(first).__name__}: {first}"
            )
            try:
                await self._push(
                    key, encode_result(notice), min(self._push_budget(job), FAILURE_PUSH_S)
                )
            except Exception as second:  # noqa: BLE001 - the servicer times out and retries
                logger.error(
                    "worker %d: result push failed twice for %s: %s; then %s",
                    index,
                    job.job_id,
                    first,
                    second,
                )

    async def _push(self, key: str, raw: bytes, budget: float) -> None:
        pipe = self._client.pipeline(transaction=True)
        pipe.lpush(key, raw)
        pipe.expire(key, RESULT_TTL_S)
        # An answer is worth only as long as the requester is still waiting for
        # it, and this push carries the whole payload, so it gets the time the
        # job has left and no more. Unbounded, a worker that lands on a stalled
        # connection is gone for good while the sidecar goes on advertising it.
        await asyncio.wait_for(pipe.execute(), budget)

    @staticmethod
    def _push_budget(job: Job) -> float:
        """Seconds left before the requester gives up, never less than a moment.

        A job already past its deadline still gets an attempt: the answer may
        be a failure code, and delivering it ends the wait sooner than letting
        it lapse.
        """
        return max(PUSH_FLOOR_S, job.deadline_ms / 1000 - time.time())

    async def handle(self, job: Job) -> JobResult:
        started = time.time()
        now_ms = int(started * 1000)
        if now_ms > job.deadline_ms:
            return failure(job.job_id, "expired", "job deadline passed before processing")
        mismatches = self._fingerprint.mismatches(job.fingerprint)
        if mismatches:
            return failure(job.job_id, "fingerprint_mismatch", ",".join(mismatches))

        multi_modal_data: dict[str, list] = {}
        for index, item in enumerate(job.items):
            scheme = url_scheme(item.url)
            if scheme not in self._accepted:
                return failure(
                    job.job_id,
                    "scheme_not_accepted",
                    f"item {index}: scheme '{scheme or 'none'}' not in {self._schemes}",
                )
            try:
                media = await self._fetch(item.modality, item.url)
            except Exception as e:  # noqa: BLE001 - fetch failures are client-visible codes
                return failure(job.job_id, classify_fetch_error(e), f"item {index}: {e}")
            multi_modal_data.setdefault(item.modality, []).append(media)
        fetched = time.time()

        prompt = self._tokens_prompt(
            prompt_token_ids=list(job.prompt_token_ids), multi_modal_data=multi_modal_data
        )
        if job.prompt:
            prompt["prompt"] = job.prompt
        try:
            engine_input = await self._renderer.process_for_engine_async(
                prompt, arrival_time=0.0, skip_mm_cache=True
            )
        except Exception as e:  # noqa: BLE001 - classified into a result code
            code = classify_process_error(e)
            if code == "processor_error":
                logger.exception("processing failed for job %s", job.job_id)
            return failure(job.job_id, code, str(e))
        processed = time.time()

        encoder = self._encoder
        mm_kwargs: dict[str, list[bytes]] = {}
        for modality, items in engine_input["mm_kwargs"].items():
            blobs = []
            for item in items:
                bufs = encoder.encode(item)
                if len(bufs) != 1:
                    return failure(job.job_id, "processor_error", "tensor not inlined")
                blobs.append(bytes(bufs[0]))
            mm_kwargs[modality] = blobs
        mm_placeholders = {
            modality: [
                Placeholder(
                    offset=r.offset,
                    length=r.length,
                    is_embed=r.is_embed.tolist() if r.is_embed is not None else None,
                )
                for r in ranges
            ]
            for modality, ranges in engine_input["mm_placeholders"].items()
        }
        return JobResult(
            v=SCHEMA_VERSION,
            job_id=job.job_id,
            ok=True,
            fingerprint=self._fingerprint,
            prompt_token_ids=list(engine_input["prompt_token_ids"]),
            mm_hashes={m: list(h) for m, h in engine_input["mm_hashes"].items()},
            mm_placeholders=mm_placeholders,
            mm_kwargs=mm_kwargs,
            timing=Timing(
                queue_ms=max(0, now_ms - job.enqueued_ms),
                fetch_ms=int((fetched - started) * 1000),
                process_ms=int((processed - fetched) * 1000),
            ),
        )

    async def _fetch(self, modality: str, url: str):
        if modality == "image":
            return await self._connector.fetch_image_async(url)
        if modality == "video":
            return await self._connector.fetch_video_async(
                url, video_processor=self._video_processor
            )
        raise ValueError(f"unsupported media modality {modality!r}")


async def serve(args: argparse.Namespace) -> None:
    import redis.asyncio as redis_asyncio
    from vllm.renderers.registry import renderer_from_config

    vllm_config = build_config(args)
    renderer = renderer_from_config(vllm_config)
    # No read deadline of the client's own: waiting for the next job is meant
    # to sit on the socket for JOB_WAIT_S, and a client-wide deadline at or
    # under that turns every quiet stretch into a failed wait. The caller
    # bounds each call instead. redis 8 made this explicit by starting to
    # default it to five seconds. Reaching the server in the first place is a
    # different question and stays bounded: there is nothing to wait for yet.
    client = redis_asyncio.from_url(
        args.redis_url, decode_responses=False, socket_connect_timeout=1.0, socket_timeout=None
    )
    sidecar = Sidecar(
        vllm_config, renderer, client, namespace=args.namespace, concurrency=args.concurrency
    )
    await sidecar.run()


def main() -> None:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    logging.basicConfig(level=logging.INFO)
    parser = FlexibleArgumentParser(description="smg media-processing sidecar for vLLM")
    parser.add_argument("--redis-url", default=DEFAULT_REDIS_URL)
    parser.add_argument("--namespace", default=None, help="override the derived key namespace")
    parser.add_argument("--concurrency", type=int, default=2)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
