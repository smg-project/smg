"""Worker-side multimodal processing backends for the vLLM servicer.

The router forwards `media_refs` (URLs) with unexpanded placeholder anchors; a
backend fetches the media and runs vLLM's own multimodal processor, returning a
fully processed engine input. The module-level helpers are engine-free; vLLM is
imported lazily inside the backends.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

ENV_PROCESSOR = "SMG_VLLM_MM_PROCESSOR"
ENV_MAX_INFLIGHT = "SMG_VLLM_MM_MAX_INFLIGHT"
ENV_MAX_ITEM_BYTES = "SMG_VLLM_MM_MAX_ITEM_BYTES"
ENV_MAX_ITEMS = "SMG_VLLM_MM_MAX_ITEMS"

MODE_OFF = "off"
MODE_INPROCESS = "inprocess"
MODE_REDIS = "redis"
VALID_MODES = (MODE_OFF, MODE_INPROCESS, MODE_REDIS)

DEFAULT_MAX_INFLIGHT = 64
DEFAULT_MAX_ITEM_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_ITEMS = 16
# First vLLM release with renderer.process_for_engine_async(skip_mm_cache=).
MIN_VLLM_VERSION = "0.20.0"


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


def enforce_item_count(items: Sequence[Any], max_items: int) -> None:
    """Bound per-request fetch fan-out before any fetch task is created."""
    if len(items) > max_items:
        raise ValueError(
            f"media_refs carries {len(items)} items, above the {max_items}-item cap ({ENV_MAX_ITEMS})"
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
        raise ValueError(f"{ENV_PROCESSOR}=inprocess needs vllm>={MIN_VLLM_VERSION} ({e})") from e
    process = getattr(getattr(engine, "renderer", None), "process_for_engine_async", None)
    if process is None or "skip_mm_cache" not in inspect.signature(process).parameters:
        raise ValueError(
            f"{ENV_PROCESSOR}=inprocess needs vllm>={MIN_VLLM_VERSION} "
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
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        _require_inprocess_apis(engine)
        from vllm import envs
        from vllm.multimodal.media.connector import MEDIA_CONNECTOR_REGISTRY
        from vllm.transformers_utils.processor import get_video_processor_cls_name

        from smg_grpc_servicer.vllm.media_refs import advertised_schemes, parse_scheme_list

        model_config = engine.model_config
        mm_config = model_config.get_multimodal_config()
        self._engine = engine
        self._max_item_bytes = max_item_bytes
        self._max_items = max_items
        self.max_inflight = max_inflight
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
    ):
        enforce_item_count(items, self._max_items)
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
        """Fetch one item; every fetch failure is the caller's (a terminal 400)."""
        try:
            if item.modality == "image":
                return await self._connector.fetch_image_async(item.url)
            if item.modality == "video":
                return await self._connector.fetch_video_async(
                    item.url, video_processor=self._video_processor
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning("media_refs[%d]: fetch failed for %s: %s", index, item.modality, e)
            raise ValueError(f"media_refs[{index}]: fetch failed: {e}") from e
        raise ValueError(f"unsupported media modality {item.modality!r}")


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
    max_inflight = env_int(env, ENV_MAX_INFLIGHT, DEFAULT_MAX_INFLIGHT)
    max_items = env_int(env, ENV_MAX_ITEMS, DEFAULT_MAX_ITEMS)
    if mode == MODE_INPROCESS:
        return InProcessMediaProcessor(
            engine, max_item_bytes=max_item_bytes, max_inflight=max_inflight, max_items=max_items
        )
    raise ValueError(f"{ENV_PROCESSOR}={mode} is not available in this servicer build")
