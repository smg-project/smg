"""Unit tests for the Redis media-sidecar protocol (engine-free, needs msgspec).

Run with: pytest grpc_servicer/tests/test_vllm_mm_sidecar_protocol.py
"""

import pytest

pytest.importorskip("msgspec")
from smg_grpc_servicer import mm_sidecar_protocol as proto  # noqa: E402


def fingerprint(**overrides) -> proto.Fingerprint:
    values = {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "vllm_version": "0.27.1",
        "dtype": "torch.bfloat16",
        "video_backend": "opencv",
        "media_io_kwargs": "{}",
        "mm_processor_kwargs": "{}",
    }
    values.update(overrides)
    return proto.Fingerprint(**values)


class TestFingerprint:
    def test_namespace_is_deterministic_and_dtype_sensitive(self):
        a = fingerprint().namespace()
        assert a == fingerprint().namespace()
        assert len(a) == 16
        assert a != fingerprint(dtype="torch.float16").namespace()
        # Kwargs are not part of the namespace; they are checked by fingerprint.
        assert a == fingerprint(media_io_kwargs='{"video":{"num_frames":8}}').namespace()

    def test_override_wins(self):
        assert proto.resolve_namespace(fingerprint(), " prod-a ") == "prod-a"
        assert proto.resolve_namespace(fingerprint(), "") == fingerprint().namespace()

    def test_mismatches_name_fields(self):
        assert fingerprint().mismatches(fingerprint()) == []
        assert fingerprint().mismatches(fingerprint(vllm_version="0.28.0", dtype="x")) == [
            "vllm_version",
            "dtype",
        ]

    def test_hello_roundtrip_with_bytes_mapping(self):
        hello = {k.encode(): v.encode() for k, v in fingerprint().to_hello().items()}
        hello[b"schemes"] = b"http,https,data"
        hello[b"schema"] = b"1"
        assert proto.Fingerprint.from_hello(hello) == fingerprint()
        assert proto.hello_schemes(hello) == "http,https,data"

    def test_hello_missing_field_is_none(self):
        hello = {b"model": b"m"}
        assert proto.Fingerprint.from_hello(hello) is None
        assert proto.hello_schemes(hello) == ""


class TestKeys:
    def test_key_layout(self):
        keys = proto.Keys.for_namespace("abc")
        assert keys.hello == "smg:mm:v1:abc:hello"
        assert keys.jobs == "smg:mm:v1:abc:jobs"
        assert keys.result("j1") == "smg:mm:v1:abc:res:j1"


class TestMessages:
    def test_job_roundtrip(self):
        job = proto.Job(
            v=proto.SCHEMA_VERSION,
            job_id="j1",
            request_id="r1",
            fingerprint=fingerprint(),
            prompt_token_ids=[1, 2, 3],
            prompt="hi <|image_pad|>",
            items=[proto.JobItem(modality="image", url="https://a/1.png")],
            enqueued_ms=100,
            deadline_ms=30100,
        )
        assert proto.decode_job(proto.encode_job(job)) == job

    def test_result_roundtrip_keeps_blobs_and_masks(self):
        result = proto.JobResult(
            v=proto.SCHEMA_VERSION,
            job_id="j1",
            ok=True,
            fingerprint=fingerprint(),
            prompt_token_ids=[1, 5, 5, 5, 3],
            mm_hashes={"image": ["abc"]},
            mm_placeholders={
                "image": [proto.Placeholder(offset=1, length=3, is_embed=[True, False, True])]
            },
            mm_kwargs={"image": [b"\x93\x01\x02\x03"]},
            timing=proto.Timing(queue_ms=1, fetch_ms=2, process_ms=3),
            cache="miss",
        )
        parsed = proto.decode_result(proto.encode_result(result))
        assert parsed == result
        assert parsed.mm_kwargs["image"][0] == b"\x93\x01\x02\x03"
        assert parsed.mm_placeholders["image"][0].is_embed == [True, False, True]

    def test_failure_defaults(self):
        result = proto.failure("j2", "expired", "too late")
        assert not result.ok
        assert result.code == "expired"
        assert result.mm_kwargs == {}
        assert proto.decode_result(proto.encode_result(result)) == result

    def test_error_code_sets_are_disjoint(self):
        assert not (proto.CLIENT_ERROR_CODES & proto.RETRYABLE_ERROR_CODES)
        assert "placeholder_mismatch" in proto.CLIENT_ERROR_CODES
        assert "fingerprint_mismatch" in proto.RETRYABLE_ERROR_CODES
