"""Unit tests for the worker-side media processor configuration (engine-free).

Run with: pytest grpc_servicer/tests/test_vllm_mm_processor.py
"""

import asyncio
import importlib.util
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_processor.py"
_spec = importlib.util.spec_from_file_location("mm_processor", _MODULE_PATH)
mm_processor = importlib.util.module_from_spec(_spec)
# Registered so dataclass decorators can resolve the module by name.
sys.modules[_spec.name] = mm_processor
_spec.loader.exec_module(mm_processor)


@dataclass
class _Item:
    modality: str
    url: str


class TestResolveMode:
    def test_defaults_to_off(self):
        assert mm_processor.resolve_mm_processor_mode({}) == "off"
        assert mm_processor.resolve_mm_processor_mode({"SMG_VLLM_MM_PROCESSOR": ""}) == "off"

    def test_normalizes_case_and_whitespace(self):
        assert mm_processor.resolve_mm_processor_mode({"SMG_VLLM_MM_PROCESSOR": " InProcess "}) == (
            "inprocess"
        )

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="SMG_VLLM_MM_PROCESSOR='sidecar'"):
            mm_processor.resolve_mm_processor_mode({"SMG_VLLM_MM_PROCESSOR": "sidecar"})


class TestEnvInt:
    def test_default_when_unset(self):
        assert mm_processor.env_int({}, "K", 7) == 7
        assert mm_processor.env_int({"K": "  "}, "K", 7) == 7

    def test_parses_positive(self):
        assert mm_processor.env_int({"K": "12"}, "K", 7) == 12

    @pytest.mark.parametrize("raw", ["0", "-1", "abc"])
    def test_rejects_invalid(self, raw):
        with pytest.raises(ValueError, match="K="):
            mm_processor.env_int({"K": raw}, "K", 7)

    def test_zero_is_allowed_only_when_asked_for(self):
        assert mm_processor.env_int({"K": "0"}, "K", 7, minimum=0) == 0
        with pytest.raises(ValueError, match="K='-1' must be at least 0"):
            mm_processor.env_int({"K": "-1"}, "K", 7, minimum=0)


class TestVideoFrameBudget:
    def test_no_budget_leaves_the_kwargs_alone(self):
        kwargs = {"video": {"num_frames": 40, "fps": 2}}
        assert mm_processor.clamp_video_frames(kwargs, 0) is kwargs
        assert mm_processor.clamp_video_frames(None, 0) is None

    def test_a_budget_caps_an_explicit_frame_count(self):
        kwargs = {"video": {"num_frames": 40, "fps": 2}, "image": {"x": 1}}
        assert mm_processor.clamp_video_frames(kwargs, 16) == {
            "video": {"num_frames": 16, "fps": 2},
            "image": {"x": 1},
        }
        assert kwargs["video"]["num_frames"] == 40, "the engine config is not mutated"

    def test_a_budget_never_raises_the_frame_count(self):
        kwargs = {"video": {"num_frames": 8}}
        assert mm_processor.clamp_video_frames(kwargs, 16) == {"video": {"num_frames": 8}}
        assert mm_processor.clamp_video_frames({}, 16, default_frames=32) == {
            "video": {"num_frames": 16}
        }
        assert mm_processor.clamp_video_frames({}, 64, default_frames=32) == {
            "video": {"num_frames": 32}
        }

    def test_an_explicit_every_frame_setting_is_capped_too(self):
        # vLLM reads num_frames <= 0 as "every frame": the very case the budget is for.
        for every in (-1, 0):
            assert mm_processor.clamp_video_frames({"video": {"num_frames": every}}, 16) == {
                "video": {"num_frames": 16}
            }

    def test_an_unknown_default_leaves_sampling_untouched(self, caplog):
        # A budget caps sampling and never raises it: with vLLM's default
        # unknown and no explicit count, writing the budget could do just that.
        assert mm_processor.clamp_video_frames(None, 16) is None
        kwargs = {"image": {}}
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert mm_processor.clamp_video_frames(kwargs, 16, default_frames=None) is kwargs
        assert sum("SMG_VLLM_MM_MAX_VIDEO_FRAMES" in r.getMessage() for r in caplog.records) == 1

    def test_vllms_default_is_read_from_the_media_package(self, monkeypatch):
        # vllm.multimodal.media re-exports VideoMediaIO; vllm.multimodal.video
        # does not exist on the vLLM this servicer targets.
        class _VideoMediaIO:
            def __init__(self, image_io, num_frames: int = 24, **kwargs):
                pass

        media = types.ModuleType("vllm.multimodal.media")
        media.VideoMediaIO = _VideoMediaIO
        monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
        monkeypatch.setitem(sys.modules, "vllm.multimodal", types.ModuleType("vllm.multimodal"))
        monkeypatch.setitem(sys.modules, "vllm.multimodal.media", media)
        monkeypatch.delitem(sys.modules, "vllm.multimodal.video", raising=False)
        assert mm_processor.vllm_default_video_frames() == 24
        del media.VideoMediaIO
        assert mm_processor.vllm_default_video_frames() is None


class TestItemBytes:
    def test_non_data_urls_are_unbounded(self):
        assert mm_processor.data_url_payload_bytes("https://a/1.png") is None
        assert mm_processor.data_url_payload_bytes("file:///a.png") is None

    def test_base64_payload_estimate(self):
        assert mm_processor.data_url_payload_bytes("data:image/png;base64,AAAAAAAA") == 6

    def test_plain_payload_length(self):
        assert mm_processor.data_url_payload_bytes("DATA:text/plain,hello") == 5

    def test_enforce_caps_data_urls_only(self):
        items = [
            _Item("image", "https://a/1.png"),
            _Item("image", "data:image/png;base64,AAAAAAAA"),
        ]
        mm_processor.enforce_item_bytes(items, 6)
        with pytest.raises(ValueError, match="media_refs\\[1\\].*6 bytes, above the 5-byte cap"):
            mm_processor.enforce_item_bytes(items, 5)


class TestItemCount:
    class _MmConfig:
        def __init__(self, limits):
            self._limits = limits

        def get_limit_per_prompt(self, modality):
            return self._limits[modality]

    def test_within_cap_passes(self):
        items = [_Item("image", f"https://a/{i}.png") for i in range(3)]
        mm_processor.enforce_item_count(items, {"image": 3, "video": 1})

    def test_over_cap_is_rejected(self):
        items = [_Item("image", f"https://a/{i}.png") for i in range(4)]
        with pytest.raises(ValueError, match="4 image items, above this worker's limit of 3"):
            mm_processor.enforce_item_count(items, {"image": 3, "video": 1})

    def test_each_kind_is_counted_on_its_own(self):
        items = [_Item("image", "https://a/1.png"), _Item("video", "https://a/1.mp4")]
        mm_processor.enforce_item_count(items, {"image": 1, "video": 1})
        with pytest.raises(ValueError, match="1 video items, above this worker's limit of 0"):
            mm_processor.enforce_item_count(items, {"image": 1, "video": 0})

    def test_limits_follow_the_engine_by_default(self):
        cfg = self._MmConfig({"image": 200, "video": 4})
        assert mm_processor.item_limits(cfg) == {"image": 200, "video": 4}

    def test_override_applies_to_every_kind(self):
        cfg = self._MmConfig({"image": 200, "video": 4})
        assert mm_processor.item_limits(cfg, 2) == {"image": 2, "video": 2}


class TestBuildProcessor:
    class _ModelConfig:
        def __init__(self, multimodal: bool):
            self.is_multimodal_model = multimodal

    class _Engine:
        def __init__(self, multimodal: bool = True):
            self.model_config = TestBuildProcessor._ModelConfig(multimodal)

    def test_off_returns_none(self):
        assert mm_processor.build_mm_processor(self._Engine(), env={}) is None

    def test_non_multimodal_model_returns_none(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess"}
        assert mm_processor.build_mm_processor(self._Engine(multimodal=False), env=env) is None

    def test_invalid_item_cap_is_rejected_before_construction(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess", "SMG_VLLM_MM_MAX_ITEM_BYTES": "0"}
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_ITEM_BYTES"):
            mm_processor.build_mm_processor(self._Engine(), env=env)

    def test_invalid_inflight_is_rejected_even_while_off(self):
        # The cap also bounds router-preprocessed media, so it is read either way.
        env = {"SMG_VLLM_MM_MAX_INFLIGHT": "sixty-four"}
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_INFLIGHT"):
            mm_processor.build_mm_processor(self._Engine(), env=env)

    def test_invalid_inflight_is_rejected_when_on(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess", "SMG_VLLM_MM_MAX_INFLIGHT": "0"}
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_INFLIGHT"):
            mm_processor.build_mm_processor(self._Engine(), env=env)

    def test_invalid_item_count_is_rejected_when_on(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess", "SMG_VLLM_MM_MAX_ITEMS": "-1"}
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_ITEMS"):
            mm_processor.build_mm_processor(self._Engine(), env=env)

    def test_flag_settings_beat_the_env(self):
        # --mm-processor off keeps the processor unset even with the env asking for one.
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess"}
        settings = mm_processor.MmSettings(processor="off")
        assert mm_processor.build_mm_processor(self._Engine(), env=env, settings=settings) is None

    def test_invalid_frame_budget_is_rejected_when_on(self):
        env = {"SMG_VLLM_MM_PROCESSOR": "inprocess", "SMG_VLLM_MM_MAX_VIDEO_FRAMES": "-4"}
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_VIDEO_FRAMES"):
            mm_processor.build_mm_processor(self._Engine(), env=env)


class TestMmSettings:
    """Flag > env > default, each value remembering where it came from."""

    LOGGER = "mm_processor"

    def test_defaults_when_nothing_is_set(self):
        resolved = mm_processor.MmSettings().resolve(env={})
        assert resolved.processor == "off"
        assert resolved.max_inflight == mm_processor.DEFAULT_MAX_INFLIGHT
        assert resolved.max_item_bytes == mm_processor.DEFAULT_MAX_ITEM_BYTES
        assert resolved.max_items is None
        assert resolved.redis_url == "redis://127.0.0.1:6379/0"
        assert resolved.sidecar_timeout_ms == 30_000
        assert resolved.sidecar_max_queue == 256
        assert resolved.sidecar_namespace is None
        assert resolved.source == "default"
        assert set(resolved.sources.values()) == {"default"}
        assert resolved.resolved

    def test_env_fills_what_the_flags_left_unset(self, caplog):
        env = {
            "SMG_VLLM_MM_PROCESSOR": " Redis ",
            "SMG_VLLM_MM_MAX_ITEM_BYTES": "4096",
            "SMG_VLLM_MM_SIDECAR_TIMEOUT_MS": "1000",
            "SMG_VLLM_MM_MAX_ITEMS": "3",
        }
        with caplog.at_level("WARNING", logger=self.LOGGER):
            resolved = mm_processor.MmSettings().resolve(env=env)
        assert resolved.processor == "redis"
        assert resolved.max_item_bytes == 4096
        assert resolved.sidecar_timeout_ms == 1000
        assert resolved.max_items == 3
        assert resolved.sources["processor"] == "env"
        assert resolved.sources["max_item_bytes"] == "env"
        assert resolved.sources["max_inflight"] == "default"
        assert resolved.source == "env"
        messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert (
            "SMG_VLLM_MM_PROCESSOR is deprecated in favour of --mm-processor; env support ends"
            " in the next minor release"
        ) in messages
        assert sum("SMG_VLLM_MM_PROCESSOR is deprecated" in m for m in messages) == 1
        assert sum("is deprecated in favour of" in m for m in messages) == 4

    def test_flags_win_over_env_and_log_no_deprecation(self, caplog):
        requested = mm_processor.MmSettings(
            processor="inprocess", max_item_bytes=64, sidecar_timeout_ms=5
        )
        env = {
            "SMG_VLLM_MM_PROCESSOR": "redis",
            "SMG_VLLM_MM_MAX_ITEM_BYTES": "4096",
            "SMG_VLLM_MM_SIDECAR_TIMEOUT_MS": "1000",
        }
        with caplog.at_level("WARNING", logger=self.LOGGER):
            resolved = requested.resolve(env=env)
        assert resolved.processor == "inprocess"
        assert resolved.max_item_bytes == 64
        assert resolved.sidecar_timeout_ms == 5
        assert resolved.sources["processor"] == "flag"
        assert resolved.sources["sidecar_timeout_ms"] == "flag"
        assert resolved.source == "flag"
        assert not [r for r in caplog.records if "deprecated" in r.getMessage()]

    def test_from_args_reads_the_launcher_namespace(self):
        args = types.SimpleNamespace(
            mm_processor="redis",
            mm_max_inflight=None,
            mm_max_item_bytes=None,
            mm_max_items=None,
            mm_redis_url="redis://cache:6379/1",
            mm_sidecar_timeout_ms=None,
            mm_sidecar_max_queue=None,
            mm_sidecar_namespace="ns",
            model="m",
        )
        settings = mm_processor.MmSettings.from_args(args)
        assert settings == mm_processor.MmSettings(
            processor="redis", redis_url="redis://cache:6379/1", sidecar_namespace="ns"
        )
        # An older launcher namespace without the flags asks for nothing.
        assert mm_processor.MmSettings.from_args(types.SimpleNamespace(model="m")) == (
            mm_processor.MmSettings()
        )

    def test_flag_values_are_validated_like_env_values(self):
        with pytest.raises(ValueError, match="--mm-processor='sidecar' is not one of"):
            mm_processor.MmSettings(processor="sidecar").resolve(env={})
        with pytest.raises(ValueError, match="--mm-max-item-bytes=0 must be positive"):
            mm_processor.MmSettings(max_item_bytes=0).resolve(env={})
        with pytest.raises(ValueError, match="SMG_VLLM_MM_MAX_INFLIGHT"):
            mm_processor.MmSettings().resolve(env={"SMG_VLLM_MM_MAX_INFLIGHT": "0"})

    def test_a_subset_resolves_only_itself_under_its_own_flag_names(self, caplog):
        env = {
            "SMG_VLLM_MM_REDIS_URL": "redis://env:6379/3",
            "SMG_VLLM_MM_MAX_INFLIGHT": "not-a-number",  # worker-only, must not be read
            "SMG_VLLM_MM_PROCESSOR": "redis",  # worker-only, must not warn
        }
        with caplog.at_level("WARNING", logger=self.LOGGER):
            resolved = mm_processor.MmSettings(sidecar_namespace="ns").resolve(
                env=env,
                only=("redis_url", "sidecar_namespace", "sidecar_timeout_ms"),
                flags={"redis_url": "--redis-url", "sidecar_namespace": "--namespace"},
            )
        assert resolved.redis_url == "redis://env:6379/3"
        assert resolved.sidecar_namespace == "ns"
        assert resolved.sidecar_timeout_ms == 30_000
        assert resolved.max_inflight is None and resolved.processor is None
        assert resolved.sources == {
            "redis_url": "env",
            "sidecar_namespace": "flag",
            "sidecar_timeout_ms": "default",
        }
        messages = [r.getMessage() for r in caplog.records]
        assert messages == [
            "SMG_VLLM_MM_REDIS_URL is deprecated in favour of --redis-url; env support ends"
            " in the next minor release"
        ]

    def test_an_unknown_subset_name_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown mm settings: \\['redis-url'\\]"):
            mm_processor.MmSettings().resolve(env={}, only=("redis-url",))

    def test_a_subset_resolved_object_cannot_build_a_processor(self):
        subset = mm_processor.MmSettings(processor="redis").resolve(env={}, only=("processor",))
        with pytest.raises(ValueError, match="resolved without"):
            mm_processor.build_mm_processor(
                types.SimpleNamespace(model_config=None), env={}, settings=subset
            )

    def test_resolving_twice_is_stable_and_quiet(self, caplog):
        env = {"SMG_VLLM_MM_PROCESSOR": "redis"}
        with caplog.at_level("WARNING", logger=self.LOGGER):
            once = mm_processor.MmSettings().resolve(env=env)
            again = once.resolve(env={})
        assert again == once
        assert sum("deprecated" in r.getMessage() for r in caplog.records) == 1


class TestInProcessConstruction:
    """The frame budget reaches the connector the in-process processor fetches with."""

    @staticmethod
    def _stub_vllm(monkeypatch, loaded, video_io=True):
        def module(name, **attrs):
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

        class _Registry:
            @staticmethod
            def load(name, **kwargs):
                loaded.update(kwargs)
                return object()

        class _VideoMediaIO:
            def __init__(self, image_io, num_frames: int = 32, **kwargs):
                pass

        module(
            "vllm", __version__="0.29.0", envs=types.SimpleNamespace(VLLM_MEDIA_CONNECTOR="http")
        )
        module("vllm.exceptions", VLLMClientError=ValueError)
        module("vllm.multimodal")
        media = module("vllm.multimodal.media")
        if video_io:
            media.VideoMediaIO = _VideoMediaIO
            module("vllm.multimodal.media.video", VideoMediaIO=_VideoMediaIO)
        module("vllm.multimodal.media.connector", MEDIA_CONNECTOR_REGISTRY=_Registry)
        module("vllm.transformers_utils")
        module("vllm.transformers_utils.processor", get_video_processor_cls_name=lambda cfg: "vp")

    @staticmethod
    def _engine(media_io_kwargs):
        mm_config = types.SimpleNamespace(
            media_io_kwargs=media_io_kwargs, get_limit_per_prompt=lambda modality: 4
        )
        model_config = types.SimpleNamespace(
            get_multimodal_config=lambda: mm_config,
            allowed_local_media_path="",
            allowed_media_domains=["example.com"],
        )

        class _Renderer:
            async def process_for_engine_async(self, prompt, *, arrival_time, skip_mm_cache):
                return prompt

        return types.SimpleNamespace(model_config=model_config, renderer=_Renderer())

    def test_budget_caps_the_connectors_video_frames(self, monkeypatch):
        loaded = {}
        self._stub_vllm(monkeypatch, loaded)
        engine = self._engine({"video": {"num_frames": 40}})
        mm_processor.InProcessMediaProcessor(engine, max_video_frames=16)
        assert loaded["media_io_kwargs"] == {"video": {"num_frames": 16}}
        assert engine.model_config.get_multimodal_config().media_io_kwargs == {
            "video": {"num_frames": 40}
        }

    def test_budget_above_vllms_default_keeps_the_default(self, monkeypatch):
        loaded = {}
        self._stub_vllm(monkeypatch, loaded)
        mm_processor.InProcessMediaProcessor(self._engine(None), max_video_frames=64)
        assert loaded["media_io_kwargs"] == {"video": {"num_frames": 32}}

    def test_no_budget_passes_the_engine_kwargs_through(self, monkeypatch):
        loaded = {}
        self._stub_vllm(monkeypatch, loaded)
        mm_processor.InProcessMediaProcessor(self._engine({"video": {"num_frames": 40}}))
        assert loaded["media_io_kwargs"] == {"video": {"num_frames": 40}}

    def test_unknown_default_leaves_num_frames_absent(self, monkeypatch):
        loaded = {}
        self._stub_vllm(monkeypatch, loaded, video_io=False)
        mm_processor.InProcessMediaProcessor(self._engine({"image": {}}), max_video_frames=64)
        assert loaded["media_io_kwargs"] == {"image": {}}


class TestRedisClient:
    @staticmethod
    def _stub(monkeypatch):
        calls = {}
        asyncio_mod = types.ModuleType("redis.asyncio")
        asyncio_mod.from_url = lambda url, **kwargs: calls.setdefault("kwargs", kwargs)
        redis_mod = types.ModuleType("redis")
        redis_mod.asyncio = asyncio_mod
        monkeypatch.setitem(sys.modules, "redis", redis_mod)
        monkeypatch.setitem(sys.modules, "redis.asyncio", asyncio_mod)
        return calls

    def test_reads_have_no_deadline_of_their_own(self, monkeypatch):
        calls = self._stub(monkeypatch)
        mm_processor._redis_client("redis://127.0.0.1:6379")
        # Redis 8 defaults this to five seconds, which is shorter than the waits
        # this client is built to make; the caller bounds each call instead.
        assert calls["kwargs"]["socket_timeout"] is None

    def test_connecting_still_gives_up(self, monkeypatch):
        calls = self._stub(monkeypatch)
        mm_processor._redis_client("redis://127.0.0.1:6379")
        assert calls["kwargs"]["socket_connect_timeout"] == 1.0

    def test_a_missing_client_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "redis", None)
        with pytest.raises(ValueError, match="vllm-redis"):
            mm_processor._redis_client("redis://127.0.0.1:6379")


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestFetchAll:
    def test_returns_results_in_order(self):
        async def value(v, delay):
            await asyncio.sleep(delay)
            return v

        assert run(mm_processor._fetch_all([value("a", 0.02), value("b", 0.0)])) == ["a", "b"]

    def test_first_failure_cancels_siblings(self):
        state = {"slow_cancelled": False, "slow_finished": False}

        async def slow():
            try:
                await asyncio.sleep(5)
                state["slow_finished"] = True
            except asyncio.CancelledError:
                state["slow_cancelled"] = True
                raise

        async def bad():
            await asyncio.sleep(0)
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run(mm_processor._fetch_all([slow(), bad()]))
        assert state["slow_cancelled"] and not state["slow_finished"]

    def test_outer_cancellation_reaches_the_fetches(self):
        state = {"cancelled": False}

        async def slow():
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

        async def scenario():
            outer = asyncio.ensure_future(mm_processor._fetch_all([slow()]))
            await asyncio.sleep(0.01)
            outer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await outer

        run(scenario())
        assert state["cancelled"]


class TestFetchErrorClassification:
    class _CallerError(Exception):
        """Stands in for vLLM's own "the caller sent something unusable" error."""

    class _Connector:
        def __init__(self, exc):
            self.exc = exc

        async def fetch_image_async(self, url):
            raise self.exc

    def processor(self, exc):
        p = mm_processor.InProcessMediaProcessor.__new__(mm_processor.InProcessMediaProcessor)
        p._connector = self._Connector(exc)
        p._video_processor = None
        p._caller_error = self._CallerError
        return p

    def test_transport_failures_are_retryable(self):
        p = self.processor(TimeoutError("image fetch timed out"))
        with pytest.raises(
            mm_processor.MmProcessorUnavailable,
            match="media_refs\\[3\\]: fetch failed: image fetch timed out",
        ):
            run(p._fetch(3, _Item("image", "https://a/1.png")))

    def test_caller_errors_pass_through(self):
        p = self.processor(self._CallerError("415 Unsupported Media Type"))
        with pytest.raises(self._CallerError, match="^415 Unsupported Media Type$"):
            run(p._fetch(0, _Item("image", "https://a/1.png")))

    def test_value_errors_pass_through(self):
        p = self.processor(ValueError("domain not allowed"))
        with pytest.raises(ValueError, match="^domain not allowed$"):
            run(p._fetch(0, _Item("image", "https://a/1.png")))

    def test_cancellation_is_not_swallowed(self):
        p = self.processor(asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            run(p._fetch(0, _Item("image", "https://a/1.png")))


class TestProcess:
    """The engine-free half of process(): caps run before any fetch, and vLLM's
    placeholder validation error is the client's."""

    class _CallerError(Exception):
        """Stands in for vLLM's own "the caller sent something unusable" error."""

    class _Connector:
        def __init__(self, exc=None):
            self.fetched: list[str] = []
            self.exc = exc

        async def fetch_image_async(self, url):
            self.fetched.append(url)
            if self.exc is not None:
                raise self.exc
            return object()

    class _Renderer:
        def __init__(self, exc=None):
            self.exc = exc
            self.calls: list[dict] = []

        async def process_for_engine_async(self, prompt, *, arrival_time, skip_mm_cache):
            self.calls.append(prompt)
            if self.exc is not None:
                raise self.exc
            return {"prompt": prompt, "skip_mm_cache": skip_mm_cache}

    def processor(self, renderer, *, max_items=16, fetch_exc=None):
        p = mm_processor.InProcessMediaProcessor.__new__(mm_processor.InProcessMediaProcessor)
        p._engine = type("E", (), {"renderer": renderer})()
        p._connector = self._Connector(fetch_exc)
        p._video_processor = None
        p._max_item_bytes = mm_processor.DEFAULT_MAX_ITEM_BYTES
        p._item_limits = dict.fromkeys(("image", "video"), max_items)
        p._caller_error = self._CallerError
        return p

    def test_item_cap_rejects_before_any_fetch(self):
        p = self.processor(self._Renderer(), max_items=1)
        items = [_Item("image", "https://a/1.png"), _Item("image", "https://a/2.png")]
        with pytest.raises(ValueError, match="2 image items, above this worker's limit of 1"):
            run(p.process([1, 2, 3], None, items, 0.0))
        assert p._connector.fetched == []

    def test_unusable_media_stays_the_callers_fault(self):
        p = self.processor(self._Renderer(), fetch_exc=self._CallerError("not an image"))
        with pytest.raises(self._CallerError):
            run(p.process([1, 2, 3], None, [_Item("image", "https://a/1.png")], 0.0))

    def test_transient_fetch_failure_is_retryable(self):
        p = self.processor(self._Renderer(), fetch_exc=TimeoutError("origin timed out"))
        with pytest.raises(mm_processor.MmProcessorUnavailable, match="fetch failed"):
            run(p.process([1, 2, 3], None, [_Item("image", "https://a/1.png")], 0.0))

    def test_fetched_media_reaches_the_renderer_uncached(self):
        renderer = self._Renderer()
        p = self.processor(renderer)
        out = run(p.process([1, 2, 3], "hi", [_Item("image", "https://a/1.png")], 0.0))
        assert p._connector.fetched == ["https://a/1.png"]
        assert out["skip_mm_cache"] is True
        assert renderer.calls[0]["prompt"] == "hi"
        assert renderer.calls[0]["prompt_token_ids"] == [1, 2, 3]
        assert len(renderer.calls[0]["multi_modal_data"]["image"]) == 1

    def test_placeholder_validation_is_a_client_error(self):
        p = self.processor(self._Renderer(RuntimeError("Expected 1 image placeholders, found 0")))
        with pytest.raises(ValueError, match="multimodal placeholder validation failed: Expected"):
            run(p.process([1, 2, 3], None, [_Item("image", "https://a/1.png")], 0.0))


class TestServicerWiring:
    """With vLLM installed, the servicer constructor stays two-argument and reads the env."""

    def test_default_off_keeps_processor_unset(self, monkeypatch):
        pytest.importorskip("vllm")
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

        monkeypatch.delenv("SMG_VLLM_MM_PROCESSOR", raising=False)

        class _Engine:
            vllm_config = type("VC", (), {"kv_events_config": None})()
            model_config = type("MC", (), {"is_multimodal_model": False})()

        servicer = VllmEngineServicer(_Engine(), start_time=0.0)
        assert servicer._mm_processor is None
        assert servicer._mm_settings.source == "default"

    def test_launcher_settings_take_precedence_and_name_their_source(self, monkeypatch, caplog):
        pytest.importorskip("vllm")
        from smg_grpc_servicer.vllm.servicer import VllmEngineServicer

        monkeypatch.setenv("SMG_VLLM_MM_PROCESSOR", "inprocess")
        monkeypatch.setenv("SMG_VLLM_MM_MAX_INFLIGHT", "3")

        class _Engine:
            vllm_config = type("VC", (), {"kv_events_config": None})()
            model_config = type("MC", (), {"is_multimodal_model": False})()

        settings = mm_processor.MmSettings(processor="off")
        with caplog.at_level("INFO", logger="smg_grpc_servicer.vllm.servicer"):
            servicer = VllmEngineServicer(_Engine(), start_time=0.0, mm_settings=settings)
        assert servicer._mm_processor is None
        assert servicer._mm_settings.source == "flag"
        assert servicer._mm_limit == 3, "an env-supplied cap still applies when the flag is silent"
        assert any(
            "VllmEngineServicer initialized (mm_processor=off, source=flag)" in r.getMessage()
            for r in caplog.records
        )
