"""Admission correlation only: no cancellation or trace-completeness verdict."""

import re
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestIdentity:
    run: str
    attempt: str
    request_id: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.run, self.attempt)):
            raise ValueError("explicit run and attempt are required")
        if (
            not isinstance(self.request_id, str)
            or re.fullmatch(r"[!-~]{1,128}", self.request_id) is None
        ):
            raise ValueError("request ID must satisfy the gateway header contract")

    @classmethod
    def new(cls, run, attempt):
        return cls(run, attempt, f"smg-pd-{uuid.uuid4().hex}")

    @property
    def headers(self):
        return {"x-request-id": self.request_id}

    def admitted_id(self, record, *, worker, role, instance):
        if (
            not isinstance(worker, str)
            or not worker
            or role not in ("prefill", "decode")
            or not isinstance(instance, str)
            or not instance
        ):
            raise ValueError("explicit worker, role and scheduler instance are required")
        if not isinstance(record, dict) or not isinstance(record.get("request"), dict):
            raise ValueError("unknown request admission record")
        expected = {
            "run": self.run,
            "attempt": self.attempt,
            "worker": worker,
            "role": role,
            "instance": instance,
            "schema": 1,
            "event": "add_request_return",
        }
        if (
            record.get("registered") is not True
            or type(record.get("schema")) is not int
            or any(record.get(key) != value for key, value in expected.items())
        ):
            raise ValueError("unknown request admission context")
        internal = record.get("request", {}).get("request_id")
        if (
            not isinstance(internal, str)
            or re.fullmatch(rf"chatcmpl-{re.escape(self.request_id)}-[0-9a-f]{{8}}", internal)
            is None
        ):
            raise ValueError("unknown internal request ID")
        return internal
