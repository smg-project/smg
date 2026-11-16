"""Registration barrier for a fresh gateway with exactly one static MCP server."""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from .gateway import Gateway


async def _registration_sample(response: httpx.Response) -> float | None:
    # Read bounded chunks under the caller's cancellation scope. A read timeout
    # alone is insufficient: a peer can keep sending bytes past the deadline.
    body = bytearray()
    async for chunk in response.aiter_raw(chunk_size=4096):
        if len(body) + len(chunk) > 1024 * 1024:
            return None
        body.extend(chunk)
        await asyncio.sleep(0)

    samples = []
    for index, line in enumerate(body.splitlines()):
        if index % 64 == 0:
            await asyncio.sleep(0)
        fields = line.split()
        if fields[:1] == [b"smg_mcp_servers_active"]:
            samples.append(fields[1:])
            if len(samples) > 1:
                return None
    if len(samples) == 1 and len(samples[0]) == 1:
        try:
            count = float(samples[0][0])
        except ValueError:
            return None
        return count if math.isfinite(count) else None
    return None


async def _wait_for_registration(gateway: Gateway, *, deadline: float) -> None:
    last = "registration not observed"

    async def poll() -> None:
        nonlocal last
        # This fresh local gateway does not use an environment proxy. Keep the
        # client and response stream owned here so cancellation closes both.
        async with httpx.AsyncClient(trust_env=False) as client:
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    async with client.stream(
                        "GET",
                        f"{gateway.metrics_url}/metrics",
                        headers={"Accept-Encoding": "identity"},
                        timeout=min(1.0, remaining),
                    ) as response:
                        if response.status_code != 200:
                            last = f"metrics HTTP {response.status_code}"
                        elif response.headers.get("content-encoding", "identity") != "identity":
                            last = "unexpected compressed metrics"
                        else:
                            count = await _registration_sample(response)
                            last = "missing or invalid smg_mcp_servers_active"
                            if count is not None:
                                last = f"smg_mcp_servers_active={count:g}, expected 1"
                                if count == 1 and time.monotonic() < deadline:
                                    if not gateway.is_running:
                                        raise RuntimeError("Gateway exited during MCP registration")
                                    return
                except httpx.RequestError as error:
                    last = type(error).__name__
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    await asyncio.sleep(min(0.1, remaining))
            raise TimeoutError(f"MCP registration missed the router-start deadline; {last}")

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"MCP registration missed the router-start deadline; {last}")
    task = asyncio.create_task(poll())
    try:
        # One total deadline covers connection, headers, body and parsing. The
        # periodic check also cancels an in-flight request if the gateway exits.
        async with asyncio.timeout(remaining):
            while not task.done():
                if not gateway.is_running:
                    raise RuntimeError(f"Gateway exited before MCP registration; {last}")
                await asyncio.wait({task}, timeout=0.05)
            await task
    except TimeoutError:
        raise TimeoutError(f"MCP registration missed the router-start deadline; {last}") from None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def wait_for_single_mcp_registration(gateway: Gateway, *, deadline: float) -> None:
    """Wait within the caller's existing router-start deadline.

    This is only for a fresh fixture with one configured MCP server and no
    requests yet. The gateway sets this gauge after static registration and
    FormatRegistry population, unlike worker readiness. It is not a general
    per-server inventory check or proof that a subsequent tool call succeeds.
    """
    asyncio.run(_wait_for_registration(gateway, deadline=deadline))
