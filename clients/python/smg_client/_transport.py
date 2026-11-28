"""HTTP transport layer with retry logic."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

from smg_client._errors import ConnectionError, TimeoutError, raise_for_status

if TYPE_CHECKING:
    from smg_client._config import ClientConfig

_RETRYABLE_STATUS_CODES = {429, 500, 503}


def _build_headers(config: ClientConfig) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "smg-client-python/0.1.0",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    headers.update(config.default_headers)
    return headers


def _should_retry(status_code: int, attempt: int, max_retries: int) -> bool:
    return attempt < max_retries and status_code in _RETRYABLE_STATUS_CODES


def _retry_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Compute retry delay with exponential backoff and Retry-After support."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                value = float(retry_after)
                if 0.0 <= value <= 300.0:
                    return value
            except (ValueError, OverflowError):
                pass
    return min(2**attempt * 0.5, 30.0)


class SyncTransport:
    """Synchronous HTTP transport."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            headers=_build_headers(config),
            timeout=config.timeout,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        stream: bool = False,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        last_exc: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                if stream:
                    req = self._client.build_request(
                        method, path, json=json, headers=headers, params=params
                    )
                    resp = self._client.send(req, stream=True)
                    if resp.status_code >= 400:
                        try:
                            body = resp.read().decode("utf-8", errors="replace")
                        finally:
                            resp.close()
                        if _should_retry(resp.status_code, attempt, self._config.max_retries):
                            time.sleep(_retry_delay(attempt, resp))
                            continue
                        raise_for_status(resp.status_code, body)
                    return resp

                response_obj = self._client.request(
                    method, path, json=json, headers=headers, params=params
                )
                if response_obj.status_code >= 400:
                    if _should_retry(response_obj.status_code, attempt, self._config.max_retries):
                        time.sleep(_retry_delay(attempt, response_obj))
                        continue
                    raise_for_status(response_obj.status_code, response_obj.text)
                return response_obj

            except httpx.ConnectError as e:
                last_exc = e
                if attempt < self._config.max_retries:
                    time.sleep(_retry_delay(attempt))
                    continue
                raise ConnectionError(str(e)) from e
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt < self._config.max_retries:
                    time.sleep(_retry_delay(attempt))
                    continue
                raise TimeoutError(str(e)) from e

        # Should not reach here, but just in case
        if last_exc:
            raise ConnectionError(str(last_exc)) from last_exc
        raise ConnectionError("Request failed after retries")

    def close(self) -> None:
        self._client.close()


class AsyncTransport:
    """Async HTTP transport."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers=_build_headers(config),
            timeout=config.timeout,
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        stream: bool = False,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        import asyncio

        last_exc: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                if stream:
                    req = self._client.build_request(
                        method, path, json=json, headers=headers, params=params
                    )
                    resp = await self._client.send(req, stream=True)
                    if resp.status_code >= 400:
                        try:
                            body = (await resp.aread()).decode("utf-8", errors="replace")
                        finally:
                            await resp.aclose()
                        if _should_retry(resp.status_code, attempt, self._config.max_retries):
                            await asyncio.sleep(_retry_delay(attempt, resp))
                            continue
                        raise_for_status(resp.status_code, body)
                    return resp

                response_obj = await self._client.request(
                    method, path, json=json, headers=headers, params=params
                )
                if response_obj.status_code >= 400:
                    if _should_retry(response_obj.status_code, attempt, self._config.max_retries):
                        await asyncio.sleep(_retry_delay(attempt, response_obj))
                        continue
                    raise_for_status(response_obj.status_code, response_obj.text)
                return response_obj

            except httpx.ConnectError as e:
                last_exc = e
                if attempt < self._config.max_retries:
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise ConnectionError(str(e)) from e
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt < self._config.max_retries:
                    await asyncio.sleep(_retry_delay(attempt))
                    continue
                raise TimeoutError(str(e)) from e

        if last_exc:
            raise ConnectionError(str(last_exc)) from last_exc
        raise ConnectionError("Request failed after retries")

    async def close(self) -> None:
        await self._client.aclose()
