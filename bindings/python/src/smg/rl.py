"""Client for the SMG RL control plane (`/v1/rl/*`). Standard library only.

from smg.rl import RL
rl = RL("http://smg:30000")
rl.fanout("pause_generation", selector="engine=sglang")
rl.fanout("update_weights_from_disk", {"model_path": p, "weight_version": "42"}, selector="engine=sglang")
rl.fanout("continue_generation", selector="engine=sglang")
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class RlError(Exception):
    """An SMG-side error response from /v1/rl (4xx/5xx with a JSON body)."""

    def __init__(self, status: int, payload: dict[str, Any]):
        self.status = status
        self.code = str(payload.get("error", "unknown"))
        self.payload = payload
        super().__init__(f"{self.code} (HTTP {status}): {payload.get('message', '')}")


@dataclass
class Worker:
    id: str
    url: str
    base_url: str
    engine: str
    engine_version: str | None
    model_id: str
    worker_type: str
    connection_mode: str
    tp_size: int | None
    dp_size: int | None
    pp_size: int | None
    dp_ranks: int
    role: str | None
    health: str
    weight_version: str | None
    labels: dict[str, str] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Worker:
        kwargs = {k: d.get(k) for k in cls.__dataclass_fields__}
        kwargs["labels"] = d.get("labels") or {}
        kwargs["capabilities"] = d.get("capabilities") or {}
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass
class CallResult:
    worker_id: str
    url: str
    status: int
    latency_ms: int
    body: Any

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CallResult:
        return cls(
            worker_id=str(d.get("worker_id", "")),
            url=str(d.get("url", "")),
            status=int(d.get("status", 0)),
            latency_ms=int(d.get("latency_ms", 0)),
            body=d.get("body"),
        )


@dataclass
class FanoutFailure:
    worker_id: str
    url: str
    status: int | None
    error: str
    message: str


@dataclass
class FanoutResult:
    results: dict[str, CallResult]
    failed: list[FanoutFailure]
    total: int
    succeeded: int

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> FanoutResult:
        results = {
            wid: CallResult.from_json({"worker_id": wid, **r})
            for wid, r in (d.get("results") or {}).items()
        }
        failed = [
            FanoutFailure(
                worker_id=str(f.get("worker_id", "")),
                url=str(f.get("url", "")),
                status=f.get("status"),
                error=str(f.get("error", "")),
                message=str(f.get("message", "")),
            )
            for f in (d.get("failed") or [])
        ]
        return cls(
            results=results,
            failed=failed,
            total=int(d.get("total", 0)),
            succeeded=int(d.get("succeeded", 0)),
        )


class FanoutError(RlError):
    """Raised by `RL.fanout` when any target failed and `allow_partial` is False."""

    def __init__(self, result: FanoutResult, status: int = 207):
        self.result = result
        ids = ", ".join(f.worker_id for f in result.failed)
        super().__init__(status, {"error": "fanout_partial", "message": f"failed workers: {ids}"})


class RL:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # -- transport -----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = None
        headers = {"accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["content-type"] = "application/json"
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return resp.status, _decode(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, _decode(e.read())

    # -- API -----------------------------------------------------------------

    def workers(self) -> list[Worker]:
        status, payload = self._request("GET", "/v1/rl/workers")
        _raise_for(status, payload)
        return [Worker.from_json(w) for w in payload.get("workers", [])]

    def worker(self, worker_id: str) -> Worker:
        status, payload = self._request("GET", f"/v1/rl/workers/{worker_id}")
        _raise_for(status, payload)
        return Worker.from_json(payload)

    def call(
        self,
        worker_id: str,
        path: str,
        body: Any = None,
        *,
        method: str = "POST",
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> CallResult:
        route = f"/v1/rl/workers/{worker_id}/engine/{path.lstrip('/')}"
        status, payload = self._request(method, route, body, params, timeout)
        if isinstance(payload, dict) and "worker_id" in payload:
            return CallResult.from_json(payload)
        _raise_for(status, payload)
        raise RlError(status, {"error": "unexpected_response", "message": str(payload)})

    def fanout(
        self,
        path: str,
        body: Any = None,
        *,
        selector: str,
        method: str = "POST",
        params: dict[str, Any] | None = None,
        timeout: float | None = None,
        allow_partial: bool = False,
    ) -> FanoutResult:
        query = dict(params or {})
        query["selector"] = selector
        route = f"/v1/rl/engine/{path.lstrip('/')}"
        status, payload = self._request(method, route, body, query, timeout)
        if status in (200, 207) and isinstance(payload, dict) and "results" in payload:
            result = FanoutResult.from_json(payload)
            if result.failed and not allow_partial:
                raise FanoutError(result, status)
            return result
        _raise_for(status, payload)
        raise RlError(status, {"error": "unexpected_response", "message": str(payload)})


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", "replace")


def _raise_for(status: int, payload: Any) -> None:
    if status >= 400:
        if not isinstance(payload, dict):
            payload = {"error": "http_error", "message": str(payload)}
        raise RlError(status, payload)
