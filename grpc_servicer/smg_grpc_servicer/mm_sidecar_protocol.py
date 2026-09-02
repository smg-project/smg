"""Servicer <-> sidecar protocol for worker-side media processing over Redis.

Engine-free: no vLLM import. Redis is a job transport plus the sidecar's own
cache; the servicer always receives full tensors, so nothing here has to stay
consistent with vLLM's processor caches.

Keys under ``smg:mm:v1:{namespace}``:
- ``:hello``          HASH, refreshed by the sidecar (fingerprint + schemes)
- ``:jobs``           LIST, servicer LPUSH / sidecar BRPOP
- ``:res:{job_id}``   LIST, sidecar LPUSH + EXPIRE / servicer BRPOP
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import msgspec

SCHEMA_VERSION = 1
KEY_ROOT = "smg:mm:v1"

ENV_REDIS_URL = "SMG_VLLM_MM_REDIS_URL"
ENV_TIMEOUT_MS = "SMG_VLLM_MM_SIDECAR_TIMEOUT_MS"
ENV_MAX_QUEUE = "SMG_VLLM_MM_SIDECAR_MAX_QUEUE"
ENV_NAMESPACE = "SMG_VLLM_MM_SIDECAR_NAMESPACE"

DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_TIMEOUT_MS = 30_000
DEFAULT_MAX_QUEUE = 256

HELLO_TTL_S = 15
HELLO_REFRESH_S = 5
RESULT_TTL_S = 120

# Client-caused failures become INVALID_ARGUMENT (terminal 400).
CLIENT_ERROR_CODES = frozenset(
    {
        "fetch_failed",
        "domain_not_allowed",
        "scheme_not_accepted",
        "limit_exceeded",
        "decode_failed",
        "placeholder_mismatch",
    }
)
# Everything else is UNAVAILABLE (503) so the router retries elsewhere.
RETRYABLE_ERROR_CODES = frozenset({"expired", "processor_error", "fingerprint_mismatch"})


class Fingerprint(msgspec.Struct, frozen=True):
    """What must match between a worker and the sidecar processing for it."""

    model: str
    vllm_version: str
    dtype: str
    video_backend: str
    media_io_kwargs: str
    mm_processor_kwargs: str

    def namespace(self) -> str:
        digest = hashlib.sha256(
            f"{self.model}|{self.vllm_version}|{self.dtype}".encode()
        ).hexdigest()
        return digest[:16]

    def mismatches(self, other: Fingerprint) -> list[str]:
        return [
            name for name in self.__struct_fields__ if getattr(self, name) != getattr(other, name)
        ]

    def to_hello(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__struct_fields__}

    @classmethod
    def from_hello(cls, mapping: Mapping[bytes | str, bytes | str]) -> Fingerprint | None:
        decoded = {_text(key): _text(value) for key, value in mapping.items()}
        try:
            return cls(**{name: decoded[name] for name in cls.__struct_fields__})
        except KeyError:
            return None


def _text(value: bytes | str) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def hello_schemes(mapping: Mapping[bytes | str, bytes | str]) -> str:
    for key, value in mapping.items():
        if _text(key) == "schemes":
            return _text(value)
    return ""


def resolve_namespace(fingerprint: Fingerprint, override: str | None) -> str:
    override = (override or "").strip()
    return override if override else fingerprint.namespace()


class Keys(msgspec.Struct, frozen=True):
    prefix: str

    @classmethod
    def for_namespace(cls, namespace: str) -> Keys:
        return cls(prefix=f"{KEY_ROOT}:{namespace}")

    @property
    def hello(self) -> str:
        return f"{self.prefix}:hello"

    @property
    def jobs(self) -> str:
        return f"{self.prefix}:jobs"

    def result(self, job_id: str) -> str:
        return f"{self.prefix}:res:{job_id}"


class JobItem(msgspec.Struct):
    modality: str
    url: str


class Job(msgspec.Struct):
    v: int
    job_id: str
    request_id: str
    fingerprint: Fingerprint
    prompt_token_ids: list[int]
    prompt: str | None
    items: list[JobItem]
    enqueued_ms: int
    deadline_ms: int


class Placeholder(msgspec.Struct):
    offset: int
    length: int
    is_embed: list[bool] | None = None


class Timing(msgspec.Struct):
    queue_ms: int = 0
    fetch_ms: int = 0
    process_ms: int = 0


class JobResult(msgspec.Struct):
    v: int
    job_id: str
    ok: bool
    fingerprint: Fingerprint | None = None
    prompt_token_ids: list[int] = msgspec.field(default_factory=list)
    mm_hashes: dict[str, list[str]] = msgspec.field(default_factory=dict)
    mm_placeholders: dict[str, list[Placeholder]] = msgspec.field(default_factory=dict)
    # Per item: one vLLM MsgpackEncoder buffer of a MultiModalKwargsItem.
    mm_kwargs: dict[str, list[bytes]] = msgspec.field(default_factory=dict)
    code: str = ""
    message: str = ""
    timing: Timing | None = None
    cache: str = ""


def failure(job_id: str, code: str, message: str) -> JobResult:
    return JobResult(v=SCHEMA_VERSION, job_id=job_id, ok=False, code=code, message=message)


_job_encoder = msgspec.msgpack.Encoder()
_job_decoder = msgspec.msgpack.Decoder(type=Job)
_result_encoder = msgspec.msgpack.Encoder()
_result_decoder = msgspec.msgpack.Decoder(type=JobResult)


def encode_job(job: Job) -> bytes:
    return _job_encoder.encode(job)


def decode_job(raw: bytes) -> Job:
    return _job_decoder.decode(raw)


def encode_result(result: JobResult) -> bytes:
    return _result_encoder.encode(result)


def decode_result(raw: bytes) -> JobResult:
    return _result_decoder.decode(raw)
