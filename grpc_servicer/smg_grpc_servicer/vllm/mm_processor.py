"""Worker-side multimodal processing backends for the vLLM servicer.

The router forwards `media_refs` (URLs) with unexpanded placeholder anchors; a
backend fetches the media and runs vLLM's own multimodal processor, returning a
fully processed engine input. The module-level helpers are engine-free; vLLM is
imported lazily inside the backends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from smg_grpc_servicer.mm_sidecar_protocol import (
    CLIENT_ERROR_CODES,
    DEFAULT_MAX_QUEUE,
    DEFAULT_REDIS_URL,
    DEFAULT_TIMEOUT_MS,
    ENV_MAX_QUEUE,
    ENV_NAMESPACE,
    ENV_REDIS_URL,
    ENV_TIMEOUT_MS,
    SCHEMA_VERSION,
    Fingerprint,
    Job,
    JobItem,
    JobResult,
    Keys,
    decode_result,
    encode_job,
    hello_schemes,
    resolve_namespace,
)

logger = logging.getLogger(__name__)

ENV_PROCESSOR = "SMG_VLLM_MM_PROCESSOR"
ENV_MAX_INFLIGHT = "SMG_VLLM_MM_MAX_INFLIGHT"
ENV_MAX_ITEM_BYTES = "SMG_VLLM_MM_MAX_ITEM_BYTES"

MODE_OFF = "off"
MODE_INPROCESS = "inprocess"
MODE_REDIS = "redis"
VALID_MODES = (MODE_OFF, MODE_INPROCESS, MODE_REDIS)

DEFAULT_MAX_INFLIGHT = 64
DEFAULT_MAX_ITEM_BYTES = 32 * 1024 * 1024


class MmProcessorUnavailable(Exception):
    """Retryable backend failure (sidecar down, overloaded, timed out)."""


def resolve_mm_processor_mode(env: Mapping[str, str] = os.environ) -> str:
    raw = (env.get(ENV_PROCESSOR) or MODE_OFF).strip().lower()
    if raw not in VALID_MODES:
        raise ValueError(f"{ENV_PROCESSOR}={raw!r} is not one of {'|'.join(VALID_MODES)}")
    return raw


def env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{key}={raw!r} is not an integer") from e
    if value <= 0:
        raise ValueError(f"{key}={raw!r} must be positive")
    return value


def data_url_payload_bytes(url: str) -> int | None:
    """Approximate decoded byte size of a data: URL; None for other URLs."""
    if not url[:5].lower() == "data:":
        return None
    header, _, payload = url.partition(",")
    if ";base64" in header.lower():
        return (len(payload) * 3) // 4
    return len(payload)


def enforce_item_bytes(items: Sequence[Any], max_bytes: int) -> None:
    for index, item in enumerate(items):
        size = data_url_payload_bytes(item.url)
        if size is not None and size > max_bytes:
            raise ValueError(
                f"media_refs[{index}]: inline payload is {size} bytes, above the "
                f"{max_bytes}-byte cap ({ENV_MAX_ITEM_BYTES})"
            )


class InProcessMediaProcessor:
    """Fetch with vLLM's MediaConnector and process with the engine's own renderer."""

    name = MODE_INPROCESS

    def __init__(self, engine, *, max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES) -> None:
        from vllm import envs
        from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY
        from vllm.transformers_utils.processor import get_video_processor_cls_name

        from smg_grpc_servicer.vllm.media_refs import advertised_schemes, parse_scheme_list

        model_config = engine.model_config
        mm_config = model_config.get_multimodal_config()
        self._engine = engine
        self._max_item_bytes = max_item_bytes
        # Same construction as vLLM's OpenAI frontend, so the engine-level
        # allowlists and media_io_kwargs apply to refs fetched here.
        self._connector = MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=mm_config.media_io_kwargs,
            allowed_local_media_path=model_config.allowed_local_media_path,
            allowed_media_domains=model_config.allowed_media_domains,
        )
        self._video_processor = get_video_processor_cls_name(model_config)
        self.schemes = advertised_schemes(model_config.allowed_local_media_path)
        self.accepted_schemes = parse_scheme_list(self.schemes)
        if not model_config.allowed_media_domains:
            logger.warning(
                "%s=inprocess with no --allowed-media-domains: this worker will fetch media "
                "from any host the router forwards",
                ENV_PROCESSOR,
            )

    async def probe(self) -> bool:
        return True

    async def process(
        self,
        prompt_token_ids: list[int],
        prompt_text: str | None,
        items: Sequence[Any],
        arrival_time: float,
        *,
        request_id: str = "",
    ):
        from vllm import TokensPrompt

        enforce_item_bytes(items, self._max_item_bytes)
        fetched = await asyncio.gather(*(self._fetch(item) for item in items))
        multi_modal_data: dict[str, list[Any]] = {}
        for item, media in zip(items, fetched):
            multi_modal_data.setdefault(item.modality, []).append(media)

        prompt = TokensPrompt(prompt_token_ids=prompt_token_ids, multi_modal_data=multi_modal_data)
        if prompt_text:
            prompt["prompt"] = prompt_text
        # skip_mm_cache routes through the renderer's read-only processor cache,
        # which returns full items on a hit: the engine never sees data=None.
        return await self._engine.renderer.process_for_engine_async(
            prompt, arrival_time=arrival_time, skip_mm_cache=True
        )

    async def _fetch(self, item):
        if item.modality == "image":
            return await self._connector.fetch_image_async(item.url)
        if item.modality == "video":
            return await self._connector.fetch_video_async(
                item.url, video_processor=self._video_processor
            )
        raise ValueError(f"unsupported media modality {item.modality!r}")


def engine_fingerprint(engine) -> Fingerprint:
    """What the sidecar must match to process media for this engine."""
    import vllm
    from vllm import envs

    model_config = engine.model_config
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


def _cast_floats(data, dtype):
    """Cast floating tensors to the model dtype, as the HF processor path does."""
    import torch

    if torch.is_tensor(data):
        return data.to(dtype=dtype) if data.is_floating_point() else data
    if isinstance(data, list):
        return [_cast_floats(part, dtype) for part in data]
    return data


def _parse_schemes(value: str) -> set[str]:
    return {scheme.strip().lower() for scheme in value.split(",") if scheme.strip()}


class RedisMediaProcessor:
    """Hand jobs to a media-processing sidecar over Redis lists.

    The sidecar fetches and runs vLLM's processor; the result carries full
    tensors, rebuilt here into the same pre-rendered engine input the
    preprocessed path uses.
    """

    name = MODE_REDIS

    def __init__(
        self,
        engine,
        fingerprint: Fingerprint,
        *,
        redis_url: str = DEFAULT_REDIS_URL,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_queue: int = DEFAULT_MAX_QUEUE,
        namespace: str | None = None,
        max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES,
        client=None,
    ) -> None:
        self._engine = engine
        self._fingerprint = fingerprint
        self._timeout_ms = timeout_ms
        self._max_queue = max_queue
        self._max_item_bytes = max_item_bytes
        self._keys = Keys.for_namespace(resolve_namespace(fingerprint, namespace))
        self._client = client if client is not None else _redis_client(redis_url)
        self._probe_logged = False
        # Until the sidecar says otherwise, assume the default fetch schemes.
        self.schemes = "http,https,data"
        self.accepted_schemes = _parse_schemes(self.schemes)
        logger.info("Redis media sidecar keys under %s", self._keys.prefix)

    async def probe(self) -> bool:
        """Whether a sidecar with a matching fingerprint is alive."""
        try:
            hello = await self._client.hgetall(self._keys.hello)
        except Exception as e:  # noqa: BLE001 - any transport failure means "not advertised"
            self._log_probe_once("redis unreachable: %s", e)
            return False
        remote = Fingerprint.from_hello(hello) if hello else None
        if remote is None:
            self._log_probe_once("no sidecar hello at %s", self._keys.hello)
            return False
        mismatches = self._fingerprint.mismatches(remote)
        if mismatches:
            self._log_probe_once(
                "sidecar fingerprint mismatch on %s (worker=%s sidecar=%s)",
                ",".join(mismatches),
                self._fingerprint.to_hello(),
                remote.to_hello(),
            )
            return False
        schemes = hello_schemes(hello)
        if schemes:
            self.schemes = schemes
            self.accepted_schemes = _parse_schemes(schemes)
        self._probe_logged = False
        return True

    def _log_probe_once(self, message: str, *args) -> None:
        if not self._probe_logged:
            logger.error("Media sidecar not advertised: " + message, *args)
            self._probe_logged = True

    async def process(
        self,
        prompt_token_ids: list[int],
        prompt_text: str | None,
        items: Sequence[Any],
        arrival_time: float,
        *,
        request_id: str = "",
    ):
        enforce_item_bytes(items, self._max_item_bytes)
        now_ms = int(time.time() * 1000)
        job = Job(
            v=SCHEMA_VERSION,
            job_id=uuid.uuid4().hex,
            request_id=request_id,
            fingerprint=self._fingerprint,
            prompt_token_ids=list(prompt_token_ids),
            prompt=prompt_text,
            items=[JobItem(modality=item.modality, url=item.url) for item in items],
            enqueued_ms=now_ms,
            deadline_ms=now_ms + self._timeout_ms,
        )
        result = await self._submit_and_wait(job)
        return self._rebuild(result, prompt_text, arrival_time)

    async def _submit_and_wait(self, job: Job) -> JobResult:
        """Transport only: queue the job and wait for its result."""
        try:
            depth = await self._client.llen(self._keys.jobs)
            if depth >= self._max_queue:
                raise MmProcessorUnavailable(
                    f"sidecar_overloaded: {depth} jobs queued (cap {self._max_queue})"
                )
            await self._client.lpush(self._keys.jobs, encode_job(job))
            popped = await self._client.brpop(
                self._keys.result(job.job_id), timeout=self._timeout_ms / 1000
            )
        except MmProcessorUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - redis transport failures are retryable
            raise MmProcessorUnavailable(f"sidecar_unavailable: {e}") from e
        if popped is None:
            raise MmProcessorUnavailable(
                f"sidecar_timeout: no result for job {job.job_id} within {self._timeout_ms} ms"
            )
        _, raw = popped
        result = decode_result(raw)
        if result.job_id != job.job_id:
            raise MmProcessorUnavailable(
                f"sidecar_protocol: result for job {result.job_id} on key of {job.job_id}"
            )
        if not result.ok:
            if result.code in CLIENT_ERROR_CODES:
                raise ValueError(f"media processing failed ({result.code}): {result.message}")
            raise MmProcessorUnavailable(f"{result.code or 'processor_error'}: {result.message}")
        if result.fingerprint is not None:
            mismatches = self._fingerprint.mismatches(result.fingerprint)
            if mismatches:
                raise MmProcessorUnavailable(f"fingerprint_mismatch: {','.join(mismatches)}")
        return result

    def _rebuild(self, result: JobResult, prompt_text: str | None, arrival_time: float):
        """Turn a sidecar result into the pre-rendered engine input."""
        import torch
        from vllm.inputs.engine import mm_input
        from vllm.multimodal.inputs import (
            MultiModalKwargsItem,
            MultiModalKwargsItems,
            PlaceholderRange,
        )
        from vllm.v1.serial_utils import MsgpackDecoder

        decoder = MsgpackDecoder(t=MultiModalKwargsItem)
        dtype = self._engine.model_config.dtype
        mm_kwargs: dict[str, list[MultiModalKwargsItem]] = {}
        for modality, blobs in result.mm_kwargs.items():
            items = []
            for blob in blobs:
                item = decoder.decode(blob)
                for elem in item.values():
                    elem.data = _cast_floats(elem.data, dtype)
                items.append(item)
            mm_kwargs[modality] = items

        mm_placeholders = {
            modality: [
                PlaceholderRange(
                    offset=p.offset,
                    length=p.length,
                    is_embed=torch.tensor(p.is_embed, dtype=torch.bool) if p.is_embed else None,
                )
                for p in ranges
            ]
            for modality, ranges in result.mm_placeholders.items()
        }
        prompt = mm_input(
            prompt_token_ids=list(result.prompt_token_ids),
            mm_kwargs=MultiModalKwargsItems(mm_kwargs),
            mm_hashes={modality: list(hashes) for modality, hashes in result.mm_hashes.items()},
            mm_placeholders=mm_placeholders,
            prompt=prompt_text,
        )
        prompt["arrival_time"] = arrival_time
        return prompt


def _redis_client(redis_url: str):
    import redis.asyncio as redis_asyncio

    return redis_asyncio.from_url(redis_url, decode_responses=False, socket_connect_timeout=1.0)


def build_mm_processor(engine, *, env: Mapping[str, str] = os.environ):
    """Construct the configured backend, or None when worker-side processing is off."""
    mode = resolve_mm_processor_mode(env)
    if mode == MODE_OFF:
        return None
    model_config = getattr(engine, "model_config", None)
    if model_config is None or not getattr(model_config, "is_multimodal_model", False):
        logger.warning("%s=%s ignored: the served model is not multimodal", ENV_PROCESSOR, mode)
        return None
    max_item_bytes = env_int(env, ENV_MAX_ITEM_BYTES, DEFAULT_MAX_ITEM_BYTES)
    if mode == MODE_INPROCESS:
        return InProcessMediaProcessor(engine, max_item_bytes=max_item_bytes)
    return RedisMediaProcessor(
        engine,
        engine_fingerprint(engine),
        redis_url=env.get(ENV_REDIS_URL) or DEFAULT_REDIS_URL,
        timeout_ms=env_int(env, ENV_TIMEOUT_MS, DEFAULT_TIMEOUT_MS),
        max_queue=env_int(env, ENV_MAX_QUEUE, DEFAULT_MAX_QUEUE),
        namespace=env.get(ENV_NAMESPACE),
        max_item_bytes=max_item_bytes,
    )
