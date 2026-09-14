"""Unit tests for the reference media sidecar loop (engine-free, fake Redis).

Run with: pytest grpc_servicer/tests/test_vllm_mm_sidecar.py
"""

import asyncio
import sys
import time
import types

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


class _MmConfig:
    def __init__(self, limit_per_prompt):
        self.limit_per_prompt = limit_per_prompt
        self.media_io_kwargs = {"video": {"num_frames": 8}}
        self.mm_processor_kwargs = None

    def get_limit_per_prompt(self, modality):
        # Mirrors vLLM: 999 when unset, otherwise the validated options' count.
        value = self.limit_per_prompt.get(modality, 999)
        return value["count"] if isinstance(value, dict) else value


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
        engine = types.SimpleNamespace(model_config=_ModelConfig({"image": 8}))
        vllm_config = types.SimpleNamespace(model_config=_ModelConfig({"image": 8}))
        worker = mm_processor.engine_fingerprint(engine)
        assert worker == mm_sidecar.config_fingerprint(vllm_config)
        assert worker.limit_per_prompt == '{"audio": 999, "image": 8, "video": 999}'

    def test_equivalent_limit_spellings_share_a_namespace(self, monkeypatch):
        _fake_vllm(monkeypatch)
        explicit = mm_processor.fingerprint_from_model_config(_ModelConfig({"image": 999}))
        implicit = mm_processor.fingerprint_from_model_config(_ModelConfig({}))
        nested = mm_processor.fingerprint_from_model_config(
            _ModelConfig({"image": {"count": 999, "max_pixels": 4}})
        )
        assert explicit == implicit == nested
        assert (
            explicit.namespace()
            != mm_processor.fingerprint_from_model_config(_ModelConfig({"image": 8})).namespace()
        )


def sidecar(client):
    s = mm_sidecar.Sidecar.__new__(mm_sidecar.Sidecar)
    s._client = client
    s._fingerprint = fingerprint()
    s._keys = proto.Keys.for_namespace("t")
    s._schemes = "http,https,data"
    s._accepted = {"http", "https", "data"}
    s._started_at = time.time()
    s._concurrency = 1
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
        assert ops[1] == ("expire", s._keys.hello, proto.HELLO_TTL_S)

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
