"""Unit tests for RedisMediaProcessor transport semantics (engine-free, fake Redis).

Run with: pytest grpc_servicer/tests/test_vllm_redis_media_processor.py
"""

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("msgspec")
from smg_grpc_servicer import mm_sidecar_protocol as proto  # noqa: E402

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_processor.py"
_spec = importlib.util.spec_from_file_location("mm_processor", _MODULE_PATH)
mm_processor = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mm_processor
_spec.loader.exec_module(mm_processor)


@dataclass
class _Item:
    modality: str
    url: str


class _ModelConfig:
    dtype = "bf16"
    is_multimodal_model = True


class _Engine:
    model_config = _ModelConfig()


class FakeRedis:
    """Just enough of redis.asyncio for the servicer side of the protocol."""

    def __init__(self, *, hello=None, queue_depth=0, responder=None, fail=None):
        self.hello = hello or {}
        self.queue_depth = queue_depth
        self.responder = responder
        self.fail = fail
        self.pushed = []
        self.expired = {}

    async def hgetall(self, key):
        if self.fail:
            raise self.fail
        return dict(self.hello)

    async def llen(self, key):
        if self.fail:
            raise self.fail
        return self.queue_depth

    async def lpush(self, key, value):
        self.pushed.append((key, value))

    async def brpop(self, key, timeout=0):
        if self.responder is None:
            return None
        job = proto.decode_job(self.pushed[-1][1])
        result = self.responder(job)
        return (key.encode(), proto.encode_result(result))

    async def expire(self, key, seconds):
        self.expired[key] = seconds


def fingerprint(**overrides):
    values = {
        "model": "m",
        "vllm_version": "0.27.1",
        "dtype": "torch.bfloat16",
        "video_backend": "opencv",
        "media_io_kwargs": "{}",
        "mm_processor_kwargs": "{}",
    }
    values.update(overrides)
    return proto.Fingerprint(**values)


def processor(client, **kwargs):
    return mm_processor.RedisMediaProcessor(
        _Engine(), fingerprint(), client=client, timeout_ms=50, max_queue=4, **kwargs
    )


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def ok_result(job, **extra):
    return proto.JobResult(
        v=proto.SCHEMA_VERSION,
        job_id=job.job_id,
        ok=True,
        fingerprint=extra.pop("fingerprint", fingerprint()),
        prompt_token_ids=[1, 5, 5, 3],
        mm_hashes={"image": ["h1"]},
        mm_placeholders={"image": [proto.Placeholder(offset=1, length=2)]},
        mm_kwargs={"image": [b"\x80"]},
        **extra,
    )


class TestProbe:
    def test_no_hello_is_not_advertised(self):
        assert run(processor(FakeRedis()).probe()) is False

    def test_redis_error_is_not_advertised(self):
        assert run(processor(FakeRedis(fail=ConnectionError("down"))).probe()) is False

    def test_fingerprint_mismatch_is_not_advertised(self):
        hello = {k.encode(): v.encode() for k, v in fingerprint(dtype="x").to_hello().items()}
        assert run(processor(FakeRedis(hello=hello)).probe()) is False

    def test_matching_hello_advertises_and_adopts_schemes(self):
        hello = {k.encode(): v.encode() for k, v in fingerprint().to_hello().items()}
        hello[b"schemes"] = b"http,https,data,file"
        p = processor(FakeRedis(hello=hello))
        assert run(p.probe()) is True
        assert p.schemes == "http,https,data,file"
        assert "file" in p.accepted_schemes


class TestSubmitAndWait:
    def job(self, job_id="j1"):
        return proto.Job(
            v=proto.SCHEMA_VERSION,
            job_id=job_id,
            request_id="r1",
            fingerprint=fingerprint(),
            prompt_token_ids=[1, 2, 3],
            prompt=None,
            items=[proto.JobItem(modality="image", url="https://a/1.png")],
            enqueued_ms=0,
            deadline_ms=50,
        )

    def test_queue_over_cap_fails_fast(self):
        client = FakeRedis(queue_depth=4)
        with pytest.raises(mm_processor.MmProcessorUnavailable, match="sidecar_overloaded"):
            run(processor(client)._submit_and_wait(self.job()))
        assert client.pushed == []

    def test_timeout_is_retryable(self):
        client = FakeRedis()
        with pytest.raises(mm_processor.MmProcessorUnavailable, match="sidecar_timeout"):
            run(processor(client)._submit_and_wait(self.job()))
        key, raw = client.pushed[0]
        assert key.endswith(":jobs")
        assert proto.decode_job(raw).job_id == "j1"

    def test_transport_error_is_retryable(self):
        client = FakeRedis(fail=ConnectionError("refused"))
        with pytest.raises(mm_processor.MmProcessorUnavailable, match="sidecar_unavailable"):
            run(processor(client)._submit_and_wait(self.job()))

    def test_client_error_codes_become_value_errors(self):
        client = FakeRedis(
            responder=lambda job: proto.failure(job.job_id, "domain_not_allowed", "x")
        )
        with pytest.raises(ValueError, match="domain_not_allowed"):
            run(processor(client)._submit_and_wait(self.job()))

    def test_retryable_error_codes_stay_unavailable(self):
        client = FakeRedis(responder=lambda job: proto.failure(job.job_id, "expired", "late"))
        with pytest.raises(mm_processor.MmProcessorUnavailable, match="expired"):
            run(processor(client)._submit_and_wait(self.job()))

    def test_result_fingerprint_mismatch_is_unavailable(self):
        client = FakeRedis(
            responder=lambda job: ok_result(job, fingerprint=fingerprint(vllm_version="0.1"))
        )
        with pytest.raises(mm_processor.MmProcessorUnavailable, match="fingerprint_mismatch"):
            run(processor(client)._submit_and_wait(self.job()))

    def test_success_returns_full_result(self):
        client = FakeRedis(responder=ok_result)
        result = run(processor(client)._submit_and_wait(self.job()))
        assert result.ok
        assert result.prompt_token_ids == [1, 5, 5, 3]
        assert result.mm_kwargs["image"] == [b"\x80"]


class TestProcess:
    def test_process_mints_a_fresh_job_per_attempt(self, monkeypatch):
        seen = []

        def responder(job):
            seen.append(job)
            return ok_result(job)

        client = FakeRedis(responder=responder)
        p = processor(client)
        monkeypatch.setattr(p, "_rebuild", lambda result, text, arrival: ("built", result.job_id))
        items = [_Item("image", "https://a/1.png")]
        first = run(p.process([1, 2, 3], "text", items, 1.0, request_id="req"))
        second = run(p.process([1, 2, 3], "text", items, 2.0, request_id="req"))
        assert first[0] == "built" and second[0] == "built"
        assert first[1] != second[1], "each attempt gets its own job id"
        assert [job.request_id for job in seen] == ["req", "req"]
        assert seen[0].items[0].url == "https://a/1.png"
        assert seen[0].deadline_ms - seen[0].enqueued_ms == 50

    def test_process_caps_inline_payloads_before_queueing(self):
        client = FakeRedis(responder=ok_result)
        p = processor(client, max_item_bytes=4)
        items = [_Item("image", "data:image/png;base64,AAAAAAAAAAAA")]
        with pytest.raises(ValueError, match="above the 4-byte cap"):
            run(p.process([1], None, items, 0.0))
        assert client.pushed == []


class TestBuild:
    def test_redis_mode_builds_processor(self, monkeypatch):
        monkeypatch.setattr(mm_processor, "engine_fingerprint", lambda engine: fingerprint())
        monkeypatch.setattr(mm_processor, "_redis_client", lambda url: FakeRedis())
        env = {
            "SMG_VLLM_MM_PROCESSOR": "redis",
            "SMG_VLLM_MM_SIDECAR_TIMEOUT_MS": "1000",
            "SMG_VLLM_MM_SIDECAR_MAX_QUEUE": "8",
            "SMG_VLLM_MM_SIDECAR_NAMESPACE": "ns-1",
        }
        p = mm_processor.build_mm_processor(_Engine(), env=env)
        assert p.name == "redis"
        assert p._timeout_ms == 1000
        assert p._max_queue == 8
        assert p._keys.prefix == "smg:mm:v1:ns-1"
