"""Worker-side multimodal processing backends for the vLLM servicer.

The router forwards `media_refs` (URLs) with unexpanded placeholder anchors; a
backend fetches the media and runs vLLM's own multimodal processor, returning a
fully processed engine input. The module-level helpers are engine-free; vLLM is
imported lazily inside the backends.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from typing import Any

from smg_grpc_servicer.mm_sidecar_protocol import (
    CLIENT_ERROR_CODES,
    CODE_RESULT_PUSH_FAILED,
    CODE_RESULT_TOO_LARGE,
    DEFAULT_MAX_QUEUE,
    DEFAULT_MAX_VIDEO_FRAMES,
    DEFAULT_REDIS_URL,
    DEFAULT_TIMEOUT_MS,
    ENV_MAX_QUEUE,
    ENV_MAX_VIDEO_FRAMES,
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
    hello_field,
    hello_schemes,
    resolve_namespace,
)
from smg_grpc_servicer.vllm.media_refs import (
    BASE_SCHEMES,
    FETCHABLE_MODALITIES,
    advertised_schemes,
    parse_scheme_list,
)

logger = logging.getLogger(__name__)

ENV_PROCESSOR = "SMG_VLLM_MM_PROCESSOR"
PROCESSOR_FLAG = "--mm-processor"
ENV_MAX_INFLIGHT = "SMG_VLLM_MM_MAX_INFLIGHT"
ENV_MAX_ITEM_BYTES = "SMG_VLLM_MM_MAX_ITEM_BYTES"
ENV_MAX_ITEMS = "SMG_VLLM_MM_MAX_ITEMS"

MODE_OFF = "off"
MODE_INPROCESS = "inprocess"
MODE_REDIS = "redis"
VALID_MODES = (MODE_OFF, MODE_INPROCESS, MODE_REDIS)

DEFAULT_MAX_INFLIGHT = 64
DEFAULT_MAX_ITEM_BYTES = 32 * 1024 * 1024
# How long a bookkeeping round trip to the sidecar's Redis may take before the
# worker calls it unreachable instead of waiting on it.
CONTROL_TIMEOUT_S = 1.0
# First vLLM release with renderer.process_for_engine_async(skip_mm_cache=).
MIN_VLLM_VERSION = "0.20.0"


class MmProcessorUnavailable(Exception):
    """Retryable backend failure (sidecar down, overloaded, timed out)."""


def resolve_mm_processor_mode(env: Mapping[str, str] = os.environ) -> str:
    raw = (env.get(ENV_PROCESSOR) or MODE_OFF).strip().lower()
    if raw not in VALID_MODES:
        raise ValueError(
            f"{PROCESSOR_FLAG} / {ENV_PROCESSOR}={raw!r} is not one of {'|'.join(VALID_MODES)}"
        )
    return raw


def env_int(env: Mapping[str, str], key: str, default: int, *, minimum: int = 1) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{key}={raw!r} is not an integer") from e
    if value < minimum:
        bound = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{key}={raw!r} must be {bound}")
    return value


def env_int_opt(env: Mapping[str, str], key: str) -> int | None:
    """Same as `env_int`, but an unset variable means "no override"."""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    return env_int(env, key, 0)


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


def clamp_video_frames(
    media_io_kwargs: Mapping[str, Any] | None, max_frames: int, default_frames: int | None = None
) -> Mapping[str, Any] | None:
    """vLLM's media kwargs with the video frame count capped at `max_frames`.

    A copy: the engine config keeps its own value. `default_frames` is what vLLM
    samples when the kwargs set nothing; unknown, and nothing set, sampling is
    left alone, since a maximum must never raise it. A non-positive count means
    every frame to vLLM, so it is capped as well.
    """
    if max_frames <= 0:
        return media_io_kwargs
    kwargs = dict(media_io_kwargs or {})
    video = dict(kwargs.get("video") or {})
    current = video.get("num_frames", default_frames)
    if current is None:
        logger.warning(
            "%s=%d not applied: vLLM's default video frame count is unknown and "
            "media_io_kwargs sets none; sampling is left as vLLM decides",
            ENV_MAX_VIDEO_FRAMES,
            max_frames,
        )
        return media_io_kwargs
    unbounded = int(current) <= 0
    video["num_frames"] = max_frames if unbounded else min(int(current), max_frames)
    kwargs["video"] = video
    return kwargs


def vllm_default_video_frames() -> int | None:
    """The frame count vLLM's video loader falls back to."""
    try:
        # Re-exported from vllm.multimodal.media.video; vllm.multimodal.video
        # is not a module on the vLLM this servicer targets.
        from vllm.multimodal.media import VideoMediaIO

        default = inspect.signature(VideoMediaIO.__init__).parameters["num_frames"].default
    except Exception:  # noqa: BLE001 - an unknown default leaves sampling untouched
        return None
    return default if isinstance(default, int) else None


def item_limits(mm_config, override: int | None = None) -> dict[str, int]:
    """How many items of each kind one request may carry.

    The engine refuses a prompt that exceeds its own per-prompt limits, so the
    same numbers bound the fetch: a worker never turns away what it was
    configured to accept, and never fetches media the engine will not take.
    """
    if override is not None:
        return dict.fromkeys(FETCHABLE_MODALITIES, override)
    return {modality: mm_config.get_limit_per_prompt(modality) for modality in FETCHABLE_MODALITIES}


def enforce_item_count(items: Sequence[Any], limits: Mapping[str, int]) -> None:
    """Bound per-request fetch fan-out before any fetch task is created."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item.modality] = counts.get(item.modality, 0) + 1
    for modality, count in sorted(counts.items()):
        limit = limits.get(modality, 0)
        if count > limit:
            raise ValueError(
                f"media_refs carries {count} {modality} items, above this worker's "
                f"limit of {limit} (--limit-mm-per-prompt, or {ENV_MAX_ITEMS} to override)"
            )


async def _fetch_all(coros: Sequence[Awaitable[Any]]) -> list[Any]:
    """Await all fetches; on the first failure cancel and reap the rest.

    A bare gather leaves siblings running after one raises, outside the
    in-flight bound. Outer cancellation still propagates to every task.
    """
    tasks = [asyncio.ensure_future(coro) for coro in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _require_inprocess_apis(engine) -> None:
    """Fail at construction, naming the fix, when vLLM lacks the APIs used here."""
    try:
        import vllm
        from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY  # noqa: F401
        from vllm.transformers_utils.processor import (  # noqa: F401
            get_video_processor_cls_name,
        )
    except ImportError as e:
        raise ValueError(f"{PROCESSOR_FLAG}=inprocess needs vllm>={MIN_VLLM_VERSION} ({e})") from e
    process = getattr(getattr(engine, "renderer", None), "process_for_engine_async", None)
    if process is None or "skip_mm_cache" not in inspect.signature(process).parameters:
        raise ValueError(
            f"{PROCESSOR_FLAG}=inprocess needs vllm>={MIN_VLLM_VERSION} "
            f"(installed {vllm.__version__}: renderer.process_for_engine_async lacks skip_mm_cache)"
        )


class InProcessMediaProcessor:
    """Fetch with vLLM's MediaConnector and process with the engine's own renderer."""

    name = MODE_INPROCESS

    def __init__(
        self,
        engine,
        *,
        max_item_bytes: int = DEFAULT_MAX_ITEM_BYTES,
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        max_items: int | None = None,
        max_video_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
    ) -> None:
        _require_inprocess_apis(engine)
        from vllm import envs
        from vllm.exceptions import VLLMClientError
        from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY
        from vllm.transformers_utils.processor import get_video_processor_cls_name

        model_config = engine.model_config
        mm_config = model_config.get_multimodal_config()
        self._engine = engine
        self._max_item_bytes = max_item_bytes
        self._item_limits = item_limits(mm_config, max_items)
        self.max_inflight = max_inflight
        # The fetcher already decides which failures are the caller's.
        self._caller_error = VLLMClientError
        # Same construction as vLLM's OpenAI frontend, so the engine-level
        # allowlists and media_io_kwargs apply to refs fetched here.
        self._connector = MEDIA_CONNECTOR_REGISTRY.load(
            envs.VLLM_MEDIA_CONNECTOR,
            media_io_kwargs=clamp_video_frames(
                mm_config.media_io_kwargs, max_video_frames, vllm_default_video_frames()
            ),
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
                PROCESSOR_FLAG,
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
        enforce_item_count(items, self._item_limits)
        enforce_item_bytes(items, self._max_item_bytes)
        fetched = await _fetch_all([self._fetch(index, item) for index, item in enumerate(items)])
        multi_modal_data: dict[str, list[Any]] = {}
        for item, media in zip(items, fetched):
            multi_modal_data.setdefault(item.modality, []).append(media)

        # A vllm.TokensPrompt is a TypedDict; a plain dict keeps this engine-free.
        prompt: dict[str, Any] = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": multi_modal_data,
        }
        if prompt_text:
            prompt["prompt"] = prompt_text
        # skip_mm_cache routes through the renderer's read-only processor cache,
        # which returns full items on a hit: the engine never sees data=None.
        try:
            return await self._engine.renderer.process_for_engine_async(
                prompt, arrival_time=arrival_time, skip_mm_cache=True
            )
        except RuntimeError as e:
            # vLLM's placeholder validation (anchor/count mismatch): terminal client error.
            raise ValueError(f"multimodal placeholder validation failed: {e}") from e

    async def _fetch(self, index: int, item):
        """Fetch one item, keeping the fetcher's own verdict on whose fault it is.

        A bad URL or unusable media is the caller's and terminal. A timeout, a
        refused connection or an origin 5xx is not: the same request can
        succeed elsewhere or later, so it leaves here as retryable.
        """
        try:
            if item.modality == "image":
                return await self._connector.fetch_image_async(item.url)
            if item.modality == "video":
                return await self._connector.fetch_video_async(
                    item.url, video_processor=self._video_processor
                )
        except (self._caller_error, ValueError):
            raise
        except Exception as e:
            logger.warning("media_refs[%d]: fetch failed for %s: %s", index, item.modality, e)
            raise MmProcessorUnavailable(f"media_refs[{index}]: fetch failed: {e}") from e
        raise ValueError(f"unsupported media modality {item.modality!r}")


# A modality name vLLM never configures: its resolved count is the unset default.
_UNSET_MODALITY = "__unset__"


def _limit_options(options) -> dict[str, Any]:
    """A validated per-modality limit as a comparable dict, unset fields dropped.

    Unknown shapes are hashed by their attributes, never reduced to a count:
    the two ends must fail closed rather than falsely match.
    """
    if dataclasses.is_dataclass(options) and not isinstance(options, type):
        data = dataclasses.asdict(options)
    elif isinstance(options, Mapping):
        data = dict(options)
    elif isinstance(options, int):  # legacy count-only form
        data = {"count": options}
    elif hasattr(options, "model_dump"):  # a pydantic model
        data = dict(options.model_dump())
    elif hasattr(options, "__dict__"):
        logger.warning("limit_per_prompt value of type %s: hashing its attributes", type(options))
        data = dict(vars(options))
    else:
        raise TypeError(f"unsupported limit_per_prompt value: {options!r}")
    return {key: value for key, value in data.items() if value is not None}


def resolved_mm_limits(mm_config) -> dict[str, dict[str, Any]]:
    """Per-modality prompt limits as vLLM validated them: the unset default under
    "*", then each configured modality's effective count (get_limit_per_prompt)
    plus its sibling options, omitting entries that only spell out the default,
    so equivalent --limit-mm-per-prompt spellings match and a real skew fails
    closed."""
    unset = {"count": mm_config.get_limit_per_prompt(_UNSET_MODALITY)}
    limits: dict[str, dict[str, Any]] = {"*": unset}
    configured = getattr(mm_config, "limit_per_prompt", None) or {}
    for modality in sorted(configured):
        extra = _limit_options(configured[modality])
        extra.pop("count", None)
        entry = {"count": mm_config.get_limit_per_prompt(modality), **extra}
        if entry != unset:
            limits[modality] = entry
    return limits


def fingerprint_from_model_config(model_config) -> Fingerprint:
    """The one derivation the worker and the sidecar share."""
    import vllm
    from vllm import envs

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
        limit_per_prompt=json.dumps(resolved_mm_limits(mm_config), sort_keys=True, default=str),
    )


def engine_fingerprint(engine) -> Fingerprint:
    """What the sidecar must match to process media for this engine."""
    return fingerprint_from_model_config(engine.model_config)


def _cast_floats(data, dtype):
    """Cast floating tensors to the model dtype, as the HF processor path does."""
    import torch

    if torch.is_tensor(data):
        return data.to(dtype=dtype) if data.is_floating_point() else data
    if isinstance(data, list):
        return [_cast_floats(part, dtype) for part in data]
    return data


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
        max_inflight: int = DEFAULT_MAX_INFLIGHT,
        max_items: int | None = None,
        client=None,
    ) -> None:
        self._engine = engine
        self._fingerprint = fingerprint
        self._timeout_ms = timeout_ms
        self._max_queue = max_queue
        self._max_item_bytes = max_item_bytes
        self._item_limits = item_limits(engine.model_config.get_multimodal_config(), max_items)
        self.max_inflight = max_inflight
        self._keys = Keys.for_namespace(resolve_namespace(fingerprint, namespace))
        self._client = client if client is not None else _redis_client(redis_url)
        self._probe_logged = False
        # Until the sidecar says otherwise, assume the default fetch schemes.
        self.schemes = ",".join(BASE_SCHEMES)
        self.accepted_schemes = parse_scheme_list(self.schemes)
        logger.info("Redis media sidecar keys under %s", self._keys.prefix)

    async def probe(self) -> bool:
        """Whether a sidecar with a matching fingerprint is alive."""
        try:
            hello = await asyncio.wait_for(
                self._client.hgetall(self._keys.hello), CONTROL_TIMEOUT_S
            )
        except Exception as e:  # noqa: BLE001 - any transport failure means "not advertised"
            self._log_probe_once("redis unreachable: %s", e)
            return False
        schema = hello_field(hello, "schema") if hello else ""
        if hello and schema != str(SCHEMA_VERSION):
            self._log_probe_once(
                "sidecar speaks protocol v%s, worker speaks v%s", schema or "?", SCHEMA_VERSION
            )
            return False
        remote = Fingerprint.from_hello(hello) if hello else None
        if remote is None:
            # A sidecar built for any other fingerprint lives under another
            # namespace, so this is the message every disagreement produces.
            self._log_probe_once(
                "no sidecar hello at %s (worker fingerprint=%s)",
                self._keys.hello,
                self._fingerprint.to_hello(),
            )
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
            self.accepted_schemes = parse_scheme_list(schemes)
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
        enforce_item_count(items, self._item_limits)
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
        wait_s = self._timeout_ms / 1000
        try:
            depth = await asyncio.wait_for(self._client.llen(self._keys.jobs), CONTROL_TIMEOUT_S)
            if depth >= self._max_queue:
                raise MmProcessorUnavailable(
                    f"sidecar_overloaded: {depth} jobs queued (cap {self._max_queue})"
                )
            await asyncio.wait_for(
                self._client.lpush(self._keys.jobs, encode_job(job)), CONTROL_TIMEOUT_S
            )
            # Redis stops waiting on its own, but only if it is still answering;
            # the outer bound is what covers a connection that has gone quiet.
            popped = await asyncio.wait_for(
                self._client.brpop(self._keys.result(job.job_id), timeout=wait_s),
                wait_s + CONTROL_TIMEOUT_S,
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
        try:
            result = decode_result(raw)
        except Exception as e:  # noqa: BLE001 - an undecodable result means "try another worker"
            raise MmProcessorUnavailable(f"sidecar_protocol: undecodable result: {e}") from e
        if result.v != SCHEMA_VERSION:
            raise MmProcessorUnavailable(
                f"sidecar_protocol: result schema v{result.v}, worker speaks v{SCHEMA_VERSION}"
            )
        if result.job_id != job.job_id:
            raise MmProcessorUnavailable(
                f"sidecar_protocol: result for job {result.job_id} on key of {job.job_id}"
            )
        if not result.ok:
            if result.code == CODE_RESULT_TOO_LARGE:
                raise ValueError(f"media_too_large: {result.message}")
            if result.code == CODE_RESULT_PUSH_FAILED:
                raise MmProcessorUnavailable(f"sidecar_push_failed: {result.message}")
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
    try:
        import redis.asyncio as redis_asyncio
    except ImportError as e:
        raise ValueError(
            f"{PROCESSOR_FLAG}=redis requires the redis client: "
            "pip install smg-grpc-servicer[vllm,vllm-redis]"
        ) from e

    # No read deadline of the client's own. Every call here is already bounded
    # by the caller, against the wait that call actually asked for, and a
    # client-wide deadline cannot know that number: waiting for a result runs
    # as long as the configured job timeout, so a shorter one would abandon
    # every job that outlives it and report the sidecar as unreachable.
    # redis 8 made this explicit by starting to default it to five seconds.
    return redis_asyncio.from_url(
        redis_url, decode_responses=False, socket_connect_timeout=1.0, socket_timeout=None
    )


SOURCE_FLAG = "flag"
SOURCE_ENV = "env"
SOURCE_DEFAULT = "default"

# Setting name -> (launcher flag, env var, default). `None` defaults are
# settings that may legitimately stay unset.
_MM_SETTING_SPECS: dict[str, tuple[str, str, Any]] = {
    "processor": ("--mm-processor", ENV_PROCESSOR, MODE_OFF),
    "max_inflight": ("--mm-max-inflight", ENV_MAX_INFLIGHT, DEFAULT_MAX_INFLIGHT),
    "max_item_bytes": ("--mm-max-item-bytes", ENV_MAX_ITEM_BYTES, DEFAULT_MAX_ITEM_BYTES),
    "max_items": ("--mm-max-items", ENV_MAX_ITEMS, None),
    "redis_url": ("--mm-redis-url", ENV_REDIS_URL, DEFAULT_REDIS_URL),
    "sidecar_timeout_ms": ("--mm-sidecar-timeout-ms", ENV_TIMEOUT_MS, DEFAULT_TIMEOUT_MS),
    "sidecar_max_queue": ("--mm-sidecar-max-queue", ENV_MAX_QUEUE, DEFAULT_MAX_QUEUE),
    "sidecar_namespace": ("--mm-sidecar-namespace", ENV_NAMESPACE, None),
}
_MM_INT_SETTINGS = frozenset(
    {"max_inflight", "max_item_bytes", "max_items", "sidecar_timeout_ms", "sidecar_max_queue"}
)


@dataclasses.dataclass(frozen=True)
class MmSettings:
    """The engine-side media settings, as requested or as resolved.

    A field left `None` was not asked for; `resolve` fills it as flag > env >
    default and records each value's source. `sources` is empty until then.
    """

    processor: str | None = None
    max_inflight: int | None = None
    max_item_bytes: int | None = None
    max_items: int | None = None
    redis_url: str | None = None
    sidecar_timeout_ms: int | None = None
    sidecar_max_queue: int | None = None
    sidecar_namespace: str | None = None
    sources: Mapping[str, str] = dataclasses.field(default_factory=dict, compare=False)

    @classmethod
    def from_args(cls, args) -> MmSettings:
        """The `--mm-*` values of a launcher namespace; absent flags ask for nothing."""
        return cls(**{name: getattr(args, f"mm_{name}", None) for name in _MM_SETTING_SPECS})

    @property
    def resolved(self) -> bool:
        """Resolved at least once; `sources` names which settings, so a
        subset-resolved object is not a full one (see `build_mm_processor`)."""
        return bool(self.sources)

    @property
    def source(self) -> str:
        """Where the processor mode came from."""
        return self.sources.get("processor", SOURCE_DEFAULT)

    def resolve(
        self,
        env: Mapping[str, str] = os.environ,
        *,
        only: Iterable[str] | None = None,
        flags: Mapping[str, str] | None = None,
    ) -> MmSettings:
        """Flag > env > default; already resolved settings come back unchanged.

        `only` limits resolution to the named settings (the rest stay unset
        and unvalidated), for a process that uses a subset; `flags` renames
        the flag a deprecation line points at, for a parser with its own.
        """
        if self.resolved:
            return self
        values: dict[str, Any] = {}
        sources: dict[str, str] = {}
        wanted = set(_MM_SETTING_SPECS if only is None else only)
        if unknown := wanted - _MM_SETTING_SPECS.keys():
            raise ValueError(f"unknown mm settings: {sorted(unknown)}")
        for name, (flag, env_name, default) in _MM_SETTING_SPECS.items():
            if name not in wanted:
                continue
            flag = (flags or {}).get(name, flag)
            requested = getattr(self, name)
            if requested is not None:
                values[name] = _validate_flag(name, flag, requested)
                sources[name] = SOURCE_FLAG
                continue
            from_env = _read_env(name, env, env_name)
            if from_env is not None:
                logger.warning(
                    "%s is deprecated in favour of %s; env support ends in the next minor release",
                    env_name,
                    flag,
                )
                values[name] = from_env
                sources[name] = SOURCE_ENV
                continue
            values[name] = default
            sources[name] = SOURCE_DEFAULT
        return MmSettings(**values, sources=sources)


def _validate_flag(name: str, flag: str, value: Any) -> Any:
    if name == "processor":
        mode = str(value).strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"{flag}={value!r} is not one of {'|'.join(VALID_MODES)}")
        return mode
    if name in _MM_INT_SETTINGS:
        if int(value) <= 0:
            raise ValueError(f"{flag}={value} must be positive")
        return int(value)
    return value


def _read_env(name: str, env: Mapping[str, str], env_name: str) -> Any:
    """The env's value for `name`, validated as before; `None` when unset."""
    if name == "processor":
        raw = env.get(env_name)
        return resolve_mm_processor_mode(env) if raw is not None and raw.strip() else None
    if name in _MM_INT_SETTINGS:
        return env_int_opt(env, env_name)
    raw = env.get(env_name)
    return raw if raw else None


def build_mm_processor(
    engine, *, env: Mapping[str, str] = os.environ, settings: MmSettings | None = None
):
    """Construct the configured backend, or None when worker-side processing is off.

    `settings` are the launcher's flags (already resolved or not); without them
    everything comes from the env, as before.
    """
    resolved = (settings or MmSettings()).resolve(env)
    if missing := _MM_SETTING_SPECS.keys() - resolved.sources.keys():
        raise ValueError(f"mm settings resolved without {sorted(missing)}")
    mode = resolved.processor
    if mode == MODE_OFF:
        return None
    model_config = getattr(engine, "model_config", None)
    if model_config is None or not getattr(model_config, "is_multimodal_model", False):
        logger.warning("mm_processor=%s ignored: the served model is not multimodal", mode)
        return None
    # The frame budget is not one of the eight launcher flags yet: env only.
    max_video_frames = env_int(env, ENV_MAX_VIDEO_FRAMES, DEFAULT_MAX_VIDEO_FRAMES, minimum=0)
    if mode == MODE_INPROCESS:
        return InProcessMediaProcessor(
            engine,
            max_item_bytes=resolved.max_item_bytes,
            max_inflight=resolved.max_inflight,
            max_items=resolved.max_items,
            max_video_frames=max_video_frames,
        )
    return RedisMediaProcessor(
        engine,
        engine_fingerprint(engine),
        redis_url=resolved.redis_url,
        timeout_ms=resolved.sidecar_timeout_ms,
        max_queue=resolved.sidecar_max_queue,
        namespace=resolved.sidecar_namespace,
        max_item_bytes=resolved.max_item_bytes,
        max_inflight=resolved.max_inflight,
        max_items=resolved.max_items,
    )
