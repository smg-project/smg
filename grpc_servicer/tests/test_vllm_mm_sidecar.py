"""Unit tests for the reference media sidecar loop (engine-free, fake Redis).

Run with: pytest grpc_servicer/tests/test_vllm_mm_sidecar.py
"""

import asyncio
import sys
import time
import types
from dataclasses import dataclass

import pytest

pytest.importorskip("msgspec")
from smg_grpc_servicer import mm_sidecar_protocol as proto  # noqa: E402
from smg_grpc_servicer.vllm import mm_processor, mm_sidecar  # noqa: E402


class FakePipeline:
    def __init__(self, store):
        self.store = store
        self.ops = []

    def hset(self, key, mapping=None):
        self.ops.append(("hset", key, dict(mapping or {})))

    def lpush(self, key, value):
        self.ops.append(("lpush", key, value))

    def expire(self, key, seconds):
        self.ops.append(("expire", key, seconds))

    async def execute(self):
        self.store.executed.append(list(self.ops))
        if self.store.execute_fail is not None:
            raise self.store.execute_fail
        return [True] * len(self.ops)


class FakeRedis:
    """Records transactions; brpop yields queued jobs then stops the loop."""

    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.executed = []
        self.execute_fail = None

    def pipeline(self, transaction=True):
        assert transaction
        return FakePipeline(self)

    async def brpop(self, key, timeout=0):
        if self.jobs:
            return (key.encode(), proto.encode_job(self.jobs.pop(0)))
        raise asyncio.CancelledError


def stall_the_first_transaction(client):
    """Make the client's first transaction never answer; returns the stall flag."""
    stalled = []

    class Stalls(FakePipeline):
        async def execute(self):
            if not stalled:
                stalled.append(True)
                await asyncio.sleep(3600)
            return await super().execute()

    client.pipeline = lambda transaction=True: Stalls(client)
    return stalled


def fail_transactions_on(client, key, exc, times=None):
    """Make every transaction touching `key` raise `exc` (the first `times` of them if given)."""
    failures = []

    class Fails(FakePipeline):
        async def execute(self):
            self.store.executed.append(list(self.ops))
            if any(op[1] == key for op in self.ops) and (times is None or len(failures) < times):
                failures.append(exc)
                raise exc
            return [True] * len(self.ops)

    client.pipeline = lambda transaction=True: Fails(client)
    return failures


def pushed_values(client):
    return [op[2] for ops in client.executed for op in ops if op[0] == "lpush"]


def fingerprint():
    return proto.Fingerprint(
        model="m",
        vllm_version="0.27.1",
        dtype="torch.bfloat16",
        video_backend="opencv",
        media_io_kwargs="{}",
        mm_processor_kwargs="{}",
        limit_per_prompt="{}",
    )


# Shaped like vLLM's validated limit_per_prompt values (pydantic dataclasses).
@dataclass
class _BaseOptions:
    count: int = 999


@dataclass
class _ImageOptions(_BaseOptions):
    width: int | None = None
    height: int | None = None


@dataclass
class _VideoOptions(_BaseOptions):
    num_frames: int | None = None


class _MmConfig:
    def __init__(self, limit_per_prompt):
        self.limit_per_prompt = limit_per_prompt
        self.media_io_kwargs = {"video": {"num_frames": 8}}
        self.mm_processor_kwargs = None

    def get_limit_per_prompt(self, modality):
        options = self.limit_per_prompt.get(modality)
        return 999 if options is None else options.count


class _ModelConfig:
    def __init__(self, limit_per_prompt):
        self.model = "m"
        self.dtype = "torch.bfloat16"
        self._mm = _MmConfig(limit_per_prompt)

    def get_multimodal_config(self):
        return self._mm


def _fake_vllm(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.__version__ = "0.27.1"
    fake.envs = types.SimpleNamespace(VLLM_VIDEO_LOADER_BACKEND="opencv")
    monkeypatch.setitem(sys.modules, "vllm", fake)


class TestFingerprintDerivation:
    def test_worker_and_sidecar_derive_the_same_fingerprint(self, monkeypatch):
        _fake_vllm(monkeypatch)
        limits = {"image": _ImageOptions(count=8)}
        engine = types.SimpleNamespace(model_config=_ModelConfig(limits))
        vllm_config = types.SimpleNamespace(model_config=_ModelConfig(limits))
        worker = mm_processor.engine_fingerprint(engine)
        assert worker == mm_sidecar.config_fingerprint(vllm_config)
        assert worker.limit_per_prompt == '{"*": {"count": 999}, "image": {"count": 8}}'

    def test_equivalent_limit_spellings_share_a_namespace(self, monkeypatch):
        _fake_vllm(monkeypatch)
        implicit = mm_processor.fingerprint_from_model_config(_ModelConfig({}))
        for spelled_out in (
            {"image": _ImageOptions(count=999)},
            {"point_cloud": _BaseOptions(count=999)},
            {"video": _VideoOptions(count=999), "audio": _BaseOptions()},
        ):
            assert mm_processor.fingerprint_from_model_config(_ModelConfig(spelled_out)) == implicit
        for skewed in (
            {"image": _ImageOptions(count=8)},
            {"image": _ImageOptions(count=999, width=4)},
            {"point_cloud": _BaseOptions(count=2)},
        ):
            fp = mm_processor.fingerprint_from_model_config(_ModelConfig(skewed))
            assert fp.namespace() != implicit.namespace(), skewed

    def test_sibling_options_and_extra_modalities_stay_in_the_fingerprint(self):
        limits = {
            "video": _VideoOptions(count=2, num_frames=16),
            "image": _ImageOptions(count=8, width=4),
            "point_cloud": _BaseOptions(count=1),
        }
        assert mm_processor.resolved_mm_limits(_MmConfig(limits)) == {
            "*": {"count": 999},
            "image": {"count": 8, "width": 4},
            "point_cloud": {"count": 1},
            "video": {"count": 2, "num_frames": 16},
        }

    def test_unknown_limit_shapes_are_loud_not_lossy(self):
        class _Model:
            def __init__(self):
                self.count = 2
                self.num_frames = 16

            def model_dump(self):
                return {"count": self.count, "num_frames": self.num_frames}

        assert mm_processor._limit_options(_Model()) == {"count": 2, "num_frames": 16}
        plain = types.SimpleNamespace(count=2, num_frames=16, width=None)
        assert mm_processor._limit_options(plain) == {"count": 2, "num_frames": 16}
        assert mm_processor._limit_options(7) == {"count": 7}
        assert mm_processor._limit_options({"count": 3, "length": 9}) == {"count": 3, "length": 9}
        with pytest.raises(TypeError, match="unsupported limit_per_prompt value"):
            mm_processor._limit_options(object())

    def test_pydantic_dataclass_options_resolve_like_vllm(self):
        # vLLM's *DummyOptions are pydantic dataclasses; mirror the decorator so the
        # is_dataclass/asdict path runs without vLLM installed.
        pydantic = pytest.importorskip("pydantic")
        from pydantic.dataclasses import dataclass as pydantic_dataclass

        @pydantic_dataclass
        class _Base:
            count: int = pydantic.Field(999, ge=0)

        @pydantic_dataclass(config=pydantic.ConfigDict(extra="forbid"))
        class _Video(_Base):
            num_frames: int | None = pydantic.Field(None, gt=0)
            width: int | None = pydantic.Field(None, gt=0)

        cfg = _MmConfig({"video": _Video(count=2, num_frames=16), "image": _Base(count=999)})
        assert mm_processor.resolved_mm_limits(cfg) == {
            "*": {"count": 999},
            "video": {"count": 2, "num_frames": 16},
        }

    def test_resolved_limits_match_vllm(self):
        pytest.importorskip("vllm")
        from vllm.config import MultiModalConfig

        # The derivation drops entries equal to one probed unset default, which
        # is only right while that default is uniform across the options classes
        # and an unknown key resolves to it instead of raising.
        cfg = MultiModalConfig()
        modalities = ("image", "video", "audio")
        assert {m: cfg.get_limit_per_prompt(m) for m in modalities} == dict.fromkeys(
            modalities, 999
        )
        assert cfg.get_limit_per_prompt(mm_processor._UNSET_MODALITY) == 999
        unset = mm_processor.resolved_mm_limits(cfg)
        assert unset == {"*": {"count": 999}}
        for modality in modalities:
            for spelled_out in (999, {}):
                explicit = MultiModalConfig(limit_per_prompt={modality: spelled_out})
                assert mm_processor.resolved_mm_limits(explicit) == unset, (modality, spelled_out)
        resolved = mm_processor.resolved_mm_limits(
            MultiModalConfig(limit_per_prompt={"image": 8, "video": {"count": 2, "num_frames": 16}})
        )
        assert resolved["image"] == {"count": 8}
        assert resolved["video"] == {"count": 2, "num_frames": 16}


def sidecar(client):
    s = mm_sidecar.Sidecar.__new__(mm_sidecar.Sidecar)
    s._client = client
    s._fingerprint = fingerprint()
    s._keys = proto.Keys.for_namespace("t")
    s._schemes = "http,https,data"
    s._accepted = {"http", "https", "data"}
    s._started_at = time.time()
    s._concurrency = 1
    s._max_result_bytes = proto.DEFAULT_MAX_RESULT_BYTES
    s._max_video_frames = proto.DEFAULT_MAX_VIDEO_FRAMES
    return s


def job(job_id="j1"):
    return proto.Job(
        v=proto.SCHEMA_VERSION,
        job_id=job_id,
        request_id="r1",
        fingerprint=fingerprint(),
        prompt_token_ids=[1, 2, 3],
        prompt=None,
        items=[proto.JobItem(modality="image", url="https://a/1.png")],
        enqueued_ms=0,
        deadline_ms=int(time.time() * 1000) + 60_000,
    )


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestClassification:
    def test_fetch_errors(self):
        assert mm_sidecar.classify_fetch_error(ValueError("domain not allowed")) == (
            "domain_not_allowed"
        )
        assert mm_sidecar.classify_fetch_error(ValueError("bad base64")) == "fetch_failed"
        assert mm_sidecar.classify_fetch_error(TimeoutError("slow")) == "fetch_failed"

    def test_process_errors(self):
        assert mm_sidecar.classify_process_error(RuntimeError("placeholder")) == (
            "placeholder_mismatch"
        )
        # vLLM 0.27 wording: "At most 1 image(s) may be provided in one prompt."
        assert (
            mm_sidecar.classify_process_error(
                ValueError("At most 1 image(s) may be provided in one prompt.")
            )
            == "limit_exceeded"
        )
        assert mm_sidecar.classify_process_error(ValueError("cannot identify image")) == (
            "decode_failed"
        )
        assert mm_sidecar.classify_process_error(KeyError("mm_kwargs")) == "processor_error"


class TestWorkerLoop:
    def test_a_failing_handle_answers_and_keeps_the_loop_alive(self, monkeypatch):
        client = FakeRedis(jobs=[job("j1"), job("j2")])
        s = sidecar(client)
        calls = []

        async def handle(j):
            calls.append(j.job_id)
            if j.job_id == "j1":
                raise KeyError("mm_kwargs")
            return proto.JobResult(v=1, job_id=j.job_id, ok=True)

        monkeypatch.setattr(s, "handle", handle)
        with pytest.raises(asyncio.CancelledError):
            run(s._worker(0))
        assert calls == ["j1", "j2"], "the second job was still served"
        first, second = client.executed
        assert first[0][0] == "lpush" and first[0][1] == s._keys.result("j1")
        answered = proto.decode_result(first[0][2])
        assert not answered.ok and answered.code == "processor_error"
        assert first[1] == ("expire", s._keys.result("j1"), proto.RESULT_TTL_S)
        assert proto.decode_result(second[0][2]).ok

    def test_a_silent_redis_does_not_take_the_worker_with_it(self, monkeypatch):
        class Hangs(FakeRedis):
            def __init__(self, jobs=()):
                super().__init__(jobs)
                self.hung = 0

            async def brpop(self, key, timeout=0):
                if self.hung == 0:
                    self.hung += 1
                    await asyncio.sleep(3600)
                return await super().brpop(key, timeout)

        client = Hangs(jobs=[job("j4")])
        s = sidecar(client)
        monkeypatch.setattr(mm_sidecar, "JOB_WAIT_S", 0.01)
        monkeypatch.setattr(mm_sidecar, "JOB_WAIT_MARGIN_S", 0.01)
        monkeypatch.setattr(mm_sidecar, "RECONNECT_PAUSE_S", 0)
        served = []

        async def handle(j):
            served.append(j.job_id)
            return proto.JobResult(v=1, job_id=j.job_id, ok=True)

        monkeypatch.setattr(s, "handle", handle)
        with pytest.raises(asyncio.CancelledError):
            run(s._worker(0))
        assert client.hung == 1
        assert served == ["j4"], "the worker came back and served the next job"

    def test_a_stalled_result_push_does_not_take_the_worker_with_it(self, monkeypatch):
        client = FakeRedis(jobs=[job("j5"), job("j6")])
        stalled = stall_the_first_transaction(client)
        s = sidecar(client)
        served = []

        async def handle(j):
            served.append(j.job_id)
            return proto.JobResult(v=1, job_id=j.job_id, ok=True)

        monkeypatch.setattr(s, "handle", handle)
        monkeypatch.setattr(s, "_push_budget", lambda _job: 0.01)
        with pytest.raises(asyncio.CancelledError):
            run(s._worker(0))
        assert stalled, "the first push hung"
        assert served == ["j5", "j6"], "the worker came back and served the next job"

    def test_the_push_budget_is_what_the_job_has_left(self):
        pending = job("j7")
        pending.deadline_ms = int(time.time() * 1000) + 30_000
        assert 25 < mm_sidecar.Sidecar._push_budget(pending) <= 30

        lapsed = job("j8")
        lapsed.deadline_ms = int(time.time() * 1000) - 60_000
        assert mm_sidecar.Sidecar._push_budget(lapsed) == mm_sidecar.PUSH_FLOOR_S

    def test_result_push_is_one_transaction(self, monkeypatch):
        client = FakeRedis(jobs=[job("j3")])
        s = sidecar(client)

        async def handle(j):
            return proto.JobResult(v=1, job_id=j.job_id, ok=True)

        monkeypatch.setattr(s, "handle", handle)
        with pytest.raises(asyncio.CancelledError):
            run(s._worker(0))
        (ops,) = client.executed
        assert [op[0] for op in ops] == ["lpush", "expire"]


class TestResultDelivery:
    """A popped job is always answered on its result key, with something redis takes."""

    @staticmethod
    def _serve(client, results, monkeypatch, *, max_result_bytes=None):
        s = sidecar(client)
        if max_result_bytes is not None:
            s._max_result_bytes = max_result_bytes

        async def handle(j):
            return results[j.job_id]

        monkeypatch.setattr(s, "handle", handle)
        monkeypatch.setattr(s, "_push_budget", lambda _job: 0.5)
        with pytest.raises(asyncio.CancelledError):
            run(s._worker(0))
        return s

    def test_an_oversized_result_is_answered_with_result_too_large(self, monkeypatch, caplog):
        video_job = job("j1")
        video_job.items = [proto.JobItem(modality="video", url="https://a/1.mp4")]
        client = FakeRedis(jobs=[video_job])
        big = proto.JobResult(
            v=1,
            job_id="j1",
            ok=True,
            mm_kwargs={"video": [b"\x80" * 4096]},
            mm_placeholders={"video": [proto.Placeholder(offset=1, length=4)]},
        )
        with caplog.at_level("WARNING", logger="smg_grpc_servicer.vllm.mm_sidecar"):
            self._serve(client, {"j1": big}, monkeypatch, max_result_bytes=1024)
        (raw,) = pushed_values(client)
        assert len(raw) < 1024, "the oversized payload never reached redis"
        answered = proto.decode_result(raw)
        assert not answered.ok
        assert answered.code == proto.CODE_RESULT_TOO_LARGE
        assert "transport limit 1024 bytes" in answered.message
        assert "reduce video length" in answered.message
        (record,) = [r for r in caplog.records if r.levelname == "WARNING"]
        assert "j1" in record.getMessage() and "video" in record.getMessage()

    def test_a_result_under_the_limit_is_pushed_as_is(self, monkeypatch):
        client = FakeRedis(jobs=[job("j1")])
        fine = proto.JobResult(v=1, job_id="j1", ok=True, mm_kwargs={"image": [b"\x80" * 64]})
        self._serve(client, {"j1": fine}, monkeypatch)
        (raw,) = pushed_values(client)
        assert proto.decode_result(raw).ok

    def test_a_failed_push_is_answered_with_result_push_failed(self, monkeypatch):
        client = FakeRedis(jobs=[job("j1")])
        s = sidecar(client)
        key = s._keys.result("j1")
        failures = fail_transactions_on(
            client, key, ConnectionResetError("Connection reset by peer"), times=1
        )
        self._serve(client, {"j1": proto.JobResult(v=1, job_id="j1", ok=True)}, monkeypatch)
        assert len(failures) == 1
        first, second = pushed_values(client)
        assert proto.decode_result(first).ok, "the real result was tried first"
        answered = proto.decode_result(second)
        assert not answered.ok
        assert answered.code == proto.CODE_RESULT_PUSH_FAILED
        assert answered.message == "ConnectionResetError: Connection reset by peer"

    def test_two_failed_pushes_log_one_error_and_keep_the_loop_alive(self, monkeypatch, caplog):
        client = FakeRedis(jobs=[job("j1"), job("j2")])
        s = sidecar(client)
        fail_transactions_on(client, s._keys.result("j1"), ConnectionResetError("reset"))
        results = {
            "j1": proto.JobResult(v=1, job_id="j1", ok=True),
            "j2": proto.JobResult(v=1, job_id="j2", ok=True),
        }
        with caplog.at_level("WARNING", logger="smg_grpc_servicer.vllm.mm_sidecar"):
            self._serve(client, results, monkeypatch)
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert "j1" in errors[0].getMessage()
        assert proto.decode_result(pushed_values(client)[-1]).job_id == "j2"
        assert proto.decode_result(pushed_values(client)[-1]).ok


class TestResultLimit:
    """The transport limit is redis's bulk-length cap or the configured one, whichever is lower."""

    class _Config(FakeRedis):
        def __init__(self, reply=None, fail=None):
            super().__init__()
            self.reply = reply
            self.fail = fail
            self.asked = []

        async def config_get(self, name):
            self.asked.append(name)
            if self.fail is not None:
                raise self.fail
            return self.reply

    def test_a_lower_redis_cap_wins(self):
        client = self._Config(reply={b"proto-max-bulk-len": b"1048576"})
        s = sidecar(client)
        run(s._learn_result_limit())
        assert client.asked == ["proto-max-bulk-len"]
        assert s._max_result_bytes == 1048576

    def test_a_higher_redis_cap_keeps_the_configured_limit(self):
        client = self._Config(reply={"proto-max-bulk-len": str(2**40)})
        s = sidecar(client)
        s._max_result_bytes = 4096
        run(s._learn_result_limit())
        assert s._max_result_bytes == 4096

    def test_config_get_failing_keeps_the_configured_limit(self, caplog):
        client = self._Config(fail=ConnectionError("CONFIG is disabled"))
        s = sidecar(client)
        with caplog.at_level("INFO", logger="smg_grpc_servicer.vllm.mm_sidecar"):
            run(s._learn_result_limit())
        assert s._max_result_bytes == proto.DEFAULT_MAX_RESULT_BYTES
        assert any(str(proto.DEFAULT_MAX_RESULT_BYTES) in r.getMessage() for r in caplog.records)

    def test_an_unreadable_reply_keeps_the_configured_limit(self):
        client = self._Config(reply={b"proto-max-bulk-len": b"lots"})
        s = sidecar(client)
        run(s._learn_result_limit())
        assert s._max_result_bytes == proto.DEFAULT_MAX_RESULT_BYTES


class TestHeartbeat:
    def test_hello_and_ttl_are_one_transaction(self, monkeypatch):
        client = FakeRedis()
        s = sidecar(client)
        monkeypatch.setattr(mm_sidecar, "HELLO_REFRESH_S", 0)

        async def stop_after_two():
            task = asyncio.ensure_future(s._heartbeat())
            while len(client.executed) < 2:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        run(stop_after_two())
        ops = client.executed[0]
        assert ops[0][0] == "hset" and ops[0][1] == s._keys.hello
        assert ops[0][2]["schemes"] == "http,https,data"
        assert ops[0][2]["model"] == "m"
        assert ops[0][2]["max_result_bytes"] == str(proto.DEFAULT_MAX_RESULT_BYTES)
        assert ops[1] == ("expire", s._keys.hello, proto.HELLO_TTL_S)

    def test_heartbeat_survives_a_stalled_refresh(self, monkeypatch):
        client = FakeRedis()
        stalled = stall_the_first_transaction(client)
        s = sidecar(client)
        monkeypatch.setattr(mm_sidecar, "HELLO_REFRESH_S", 0)
        monkeypatch.setattr(mm_sidecar, "HELLO_TTL_S", 0.01)

        async def stop_after_two():
            task = asyncio.ensure_future(s._heartbeat())
            while len(client.executed) < 2:
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        run(stop_after_two())
        assert stalled, "the first refresh hung"
        assert len(client.executed) >= 2, "the heartbeat came back and refreshed again"

    def test_heartbeat_survives_a_failed_refresh(self, monkeypatch):
        client = FakeRedis()
        client.execute_fail = ConnectionError("redis down")
        s = sidecar(client)
        monkeypatch.setattr(mm_sidecar, "HELLO_REFRESH_S", 0)

        async def stop_after_two():
            task = asyncio.ensure_future(s._heartbeat())
            while len(client.executed) < 2:
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        run(stop_after_two())
        assert len(client.executed) >= 2


class TestServe:
    """The entrypoint builds the client that actually runs in a deployment."""

    @staticmethod
    def _stub(monkeypatch):
        seen = {}

        def module(name, **attrs):
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, mod)
            return mod

        redis_asyncio = module("redis.asyncio", from_url=lambda url, **kw: seen.update(kw))
        module("redis", asyncio=redis_asyncio)
        module("vllm")
        module("vllm.renderers")
        module("vllm.renderers.registry", renderer_from_config=lambda config: object())

        async def returns_at_once():
            return None

        monkeypatch.setattr(mm_sidecar, "build_config", lambda args: object())
        monkeypatch.setattr(
            mm_sidecar, "Sidecar", lambda *a, **kw: types.SimpleNamespace(run=returns_at_once)
        )
        return seen

    def _serve(self, monkeypatch):
        seen = self._stub(monkeypatch)
        args = types.SimpleNamespace(
            redis_url="redis://cache:6379/0", namespace="ns", concurrency=2
        )
        run(mm_sidecar.serve(args))
        return seen

    def test_waiting_for_a_job_has_no_read_deadline(self, monkeypatch):
        # The wait for the next job is meant to sit on the socket for JOB_WAIT_S.
        assert self._serve(monkeypatch)["socket_timeout"] is None

    def test_reaching_the_server_still_gives_up(self, monkeypatch):
        assert self._serve(monkeypatch)["socket_connect_timeout"] == 1.0
