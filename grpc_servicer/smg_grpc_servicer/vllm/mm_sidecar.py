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
import json
import logging
import time

from smg_grpc_servicer.mm_sidecar_protocol import (
    DEFAULT_REDIS_URL,
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

logger = logging.getLogger("smg_grpc_servicer.vllm.mm_sidecar")


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
    import vllm
    from vllm import envs

    model_config = vllm_config.model_config
    mm_config = model_config.get_multimodal_config()
    return Fingerprint(
        model=model_config.model,
        vllm_version=vllm.__version__,
        dtype=str(model_config.dtype),
        video_backend=envs.VLLM_VIDEO_LOADER_BACKEND,
        media_io_kwargs=json.dumps(mm_config.media_io_kwargs or {}, sort_keys=True, default=str),
        mm_processor_kwargs=json.dumps(
            mm_config.mm_processor_kwargs or {}, sort_keys=True, default=str
        ),
    )


def _url_scheme(url: str) -> str:
    scheme, sep, _ = url.partition(":")
    return scheme.lower() if sep else ""


class Sidecar:
    def __init__(self, vllm_config, renderer, client, *, namespace: str | None, concurrency: int):
        from vllm import envs
        from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY
        from vllm.transformers_utils.processor import get_video_processor_cls_name

        from smg_grpc_servicer.vllm.media_refs import advertised_schemes, parse_scheme_list

        model_config = vllm_config.model_config
        mm_config = model_config.get_multimodal_config()
        self._renderer = renderer
        self._client = client
        self._concurrency = max(1, concurrency)
        self._fingerprint = config_fingerprint(vllm_config)
        self._keys = Keys.for_namespace(resolve_namespace(self._fingerprint, namespace))
        self._connector = MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=mm_config.media_io_kwargs,
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
        logger.info(
            "media sidecar serving %s under %s (concurrency=%d)",
            self._fingerprint.model,
            self._keys.prefix,
            self._concurrency,
        )
        tasks = [asyncio.create_task(self._heartbeat())]
        tasks += [asyncio.create_task(self._worker(i)) for i in range(self._concurrency)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()

    async def _heartbeat(self) -> None:
        mapping = {
            **self._fingerprint.to_hello(),
            "schema": str(SCHEMA_VERSION),
            "schemes": self._schemes,
            "started_at": str(int(self._started_at)),
        }
        while True:
            try:
                await self._client.hset(self._keys.hello, mapping=mapping)
                await self._client.expire(self._keys.hello, HELLO_TTL_S)
            except Exception as e:  # noqa: BLE001 - keep advertising through redis blips
                logger.warning("hello refresh failed: %s", e)
            await asyncio.sleep(HELLO_REFRESH_S)

    async def _worker(self, index: int) -> None:
        while True:
            try:
                popped = await self._client.brpop(self._keys.jobs, timeout=5)
            except Exception as e:  # noqa: BLE001 - reconnect on the next iteration
                logger.warning("worker %d: brpop failed: %s", index, e)
                await asyncio.sleep(1)
                continue
            if popped is None:
                continue
            _, raw = popped
            try:
                job = decode_job(raw)
            except Exception as e:  # noqa: BLE001 - a malformed job has no result key to answer on
                logger.error("worker %d: undecodable job dropped: %s", index, e)
                continue
            result = await self.handle(job)
            try:
                key = self._keys.result(job.job_id)
                await self._client.lpush(key, encode_result(result))
                await self._client.expire(key, RESULT_TTL_S)
            except Exception as e:  # noqa: BLE001 - the servicer times out and retries
                logger.warning("worker %d: result push failed for %s: %s", index, job.job_id, e)

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
            scheme = _url_scheme(item.url)
            if scheme not in self._accepted:
                return failure(
                    job.job_id,
                    "scheme_not_accepted",
                    f"item {index}: scheme '{scheme or 'none'}' not in {self._schemes}",
                )
            try:
                media = await self._fetch(item.modality, item.url)
            except ValueError as e:
                code = "domain_not_allowed" if "domain" in str(e).lower() else "fetch_failed"
                return failure(job.job_id, code, f"item {index}: {e}")
            except Exception as e:  # noqa: BLE001 - network/decoder errors are client-visible
                return failure(job.job_id, "fetch_failed", f"item {index}: {e}")
            multi_modal_data.setdefault(item.modality, []).append(media)
        fetched = time.time()

        from vllm import TokensPrompt
        from vllm.v1.serial_utils import MsgpackEncoder

        prompt = TokensPrompt(
            prompt_token_ids=list(job.prompt_token_ids), multi_modal_data=multi_modal_data
        )
        if job.prompt:
            prompt["prompt"] = job.prompt
        try:
            engine_input = await self._renderer.process_for_engine_async(
                prompt, arrival_time=0.0, skip_mm_cache=True
            )
        except RuntimeError as e:
            return failure(job.job_id, "placeholder_mismatch", str(e))
        except ValueError as e:
            code = "limit_exceeded" if "at most" in str(e).lower() else "decode_failed"
            return failure(job.job_id, code, str(e))
        except Exception as e:  # noqa: BLE001 - processor bugs are retryable elsewhere
            logger.exception("processing failed for job %s", job.job_id)
            return failure(job.job_id, "processor_error", str(e))
        processed = time.time()

        encoder = MsgpackEncoder(size_threshold=2**62)
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
    client = redis_asyncio.from_url(args.redis_url, decode_responses=False)
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
