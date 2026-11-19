"""Worker-side media fetches refuse the hosts the router's own fetches refuse.

The address classes mirror ``crates/multimodal/src/fetch_policy.rs``; the
session tests run a loopback HTTP server, which a guarded fetch must refuse
by name, by IP literal and as a redirect target, and must reach when private
hosts are allowed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import sys
import types

import pytest
from smg_grpc_servicer.vllm import media_host_guard as guard
from smg_grpc_servicer.vllm.media_host_guard import (
    MediaHostPolicy,
    MediaHostRefused,
    non_public_address_class,
)


@pytest.mark.parametrize(
    "address,cls",
    [
        ("0.0.0.0", "unspecified"),
        ("0.1.2.3", "unspecified"),
        ("127.0.0.1", "loopback"),
        ("10.42.0.7", "private"),
        ("172.16.0.1", "private"),
        ("192.168.1.1", "private"),
        ("169.254.169.254", "link-local"),
        ("100.64.0.1", "shared address space"),
        ("192.0.0.8", "reserved"),
        ("198.18.0.1", "benchmarking"),
        ("224.0.0.1", "multicast"),
        ("255.255.255.255", "broadcast"),
        ("240.0.0.1", "reserved"),
        ("::", "unspecified"),
        ("::1", "loopback"),
        ("fd00::1", "unique local"),
        ("fe80::1", "link-local"),
        ("ff02::1", "multicast"),
        ("::ffff:169.254.169.254", "link-local"),
        ("::10.0.0.1", "private"),
        ("64:ff9b::7f00:1", "loopback"),
        ("64:ff9b:1::a00:1", "private"),
        ("64:ff9b:1:ff00::1", "NAT64 local-use with an unreadable layout"),
        ("2002:a9fe:a9fe::1", "link-local"),
        ("8.8.8.8", None),
        ("172.32.0.1", None),
        ("100.128.0.1", None),
        ("2606:4700::1111", None),
        ("64:ff9b::808:808", None),
    ],
)
def test_address_classes_match_the_router(address, cls):
    assert non_public_address_class(ipaddress.ip_address(address)) == cls


def test_allow_list_is_checked_by_name_on_every_hop():
    policy = MediaHostPolicy([" Images.Example.com ", ""], allow_private_hosts=True)
    assert policy.check_name("images.example.com") == "images.example.com"
    with pytest.raises(MediaHostRefused, match="allowed-media-domains"):
        policy.check_name("evil.example.com")
    assert MediaHostPolicy([], allow_private_hosts=False).allowed_domains is None


aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402


async def _serve(handler_routes):
    app = web.Application()
    app.add_routes(handler_routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    return runner, port


def _routes(state):
    async def media(_request):
        state["hits"] += 1
        return web.Response(body=b"img")

    async def bounce(request):
        raise web.HTTPFound(request.query["to"])

    async def bounce_uri(request):
        # The legacy header aiohttp follows when there is no Location.
        return web.Response(status=302, headers={"URI": request.query["to"]})

    return [
        web.get("/media", media),
        web.get("/bounce", bounce),
        web.get("/bounce-uri", bounce_uri),
    ]


async def _fetch(policy, url):
    session = guard.guarded_session(policy)
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.read()
    finally:
        await session.close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "no_proxy",
        "NO_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_loopback_is_refused_by_name_by_literal_and_as_a_redirect_target():
    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        strict = MediaHostPolicy(None, allow_private_hosts=False)
        try:
            for url in (
                f"http://localhost:{port}/media",
                f"http://127.0.0.1:{port}/media",
                f"http://[::ffff:127.0.0.1]:{port}/media",
            ):
                with pytest.raises(MediaHostRefused):
                    await _fetch(strict, url)
            assert state["hits"] == 0

            # A redirect hop is checked like the first: allowed by name, the
            # first hop reaches the server, the loopback target does not.
            permissive_first = MediaHostPolicy(None, allow_private_hosts=True)
            target = f"http://127.0.0.1:{port}/media"
            assert (
                await _fetch(permissive_first, f"http://127.0.0.1:{port}/bounce?to={target}")
                == b"img"
            )
            assert state["hits"] == 1
            only_listed = MediaHostPolicy(["localhost"], allow_private_hosts=True)
            with pytest.raises(MediaHostRefused, match="allowed-media-domains"):
                await _fetch(only_listed, f"http://localhost:{port}/bounce?to={target}")
            assert state["hits"] == 1
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("bounce", ["bounce", "bounce-uri"])
def test_a_redirect_to_a_non_public_literal_is_refused_before_it_is_sent(monkeypatch, bounce):
    # The first hop's name passes (as a public name would); its redirect names
    # a loopback literal, which is checked before the hop and never reached,
    # whether the target came in Location or in URI.
    real = guard.check_address

    def check_address(host, address):
        if host != "localhost":
            real(host, address)

    monkeypatch.setattr(guard, "check_address", check_address)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        strict = MediaHostPolicy(None, allow_private_hosts=False)
        try:
            target = f"http://127.0.0.1:{port}/media"
            with pytest.raises(MediaHostRefused, match="loopback"):
                await _fetch(strict, f"http://localhost:{port}/{bounce}?to={target}")
            assert state["hits"] == 0
            # Decimal spelling of 127.0.0.1, which aiohttp takes for a literal.
            numeric = f"http://2130706433:{port}/media"
            with pytest.raises(MediaHostRefused, match="loopback"):
                await _fetch(strict, f"http://localhost:{port}/{bounce}?to={numeric}")
            assert state["hits"] == 0
            listed = MediaHostPolicy(["localhost"], allow_private_hosts=True)
            with pytest.raises(MediaHostRefused, match="allowed-media-domains"):
                await _fetch(listed, f"http://localhost:{port}/{bounce}?to={target}")
            assert state["hits"] == 0
            assert await _fetch(strict, f"http://localhost:{port}/media") == b"img"
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize(
    "host,cls",
    [
        ("2130706433", "loopback"),
        ("127.1", "loopback"),
        ("127.0.1", "loopback"),
        ("0177.0.0.1", "loopback"),
        ("017700000001", "loopback"),
        ("0x7f.0.0.1", "loopback"),
        ("0x7F000001", "loopback"),
        ("127.0.0.1.", "loopback"),
        ("2852039166", "link-local"),
        ("0xa9.0xfe.0xa9.0xfe", "link-local"),
        ("10.1", "private"),
        ("0", "unspecified"),
    ],
)
def test_legacy_numeric_ipv4_hosts_are_read_as_the_address_they_connect_to(host, cls):
    yarl = pytest.importorskip("yarl")
    strict = MediaHostPolicy(None, allow_private_hosts=False)
    with pytest.raises(MediaHostRefused, match=cls):
        asyncio.run(strict.check_hop(yarl.URL(f"http://{host}:80/x"), resolve=False))


@pytest.mark.parametrize("host", ["999.1.1.1", "1.2.3.4.5", "1..2", "08.0.0.1"])
def test_a_numeric_host_that_is_no_address_is_refused_not_passed_on(host, monkeypatch):
    # aiohttp would hand these to the socket as literals, unresolved here.
    yarl = pytest.importorskip("yarl")

    async def lookup(_host):  # pragma: no cover - must not be reached
        raise AssertionError("a numeric host is never resolved")

    monkeypatch.setattr(guard, "_lookup", lookup)
    strict = MediaHostPolicy(None, allow_private_hosts=False)
    with pytest.raises(MediaHostRefused, match="not a valid IPv4"):
        asyncio.run(strict.check_hop(yarl.URL(f"http://{host}:80/x"), resolve=False))


def test_public_numeric_hosts_pass_and_names_stay_names():
    yarl = pytest.importorskip("yarl")
    strict = MediaHostPolicy(None, allow_private_hosts=False)
    for host in ("134744072", "8.8.2056", "0x8.0x8.0x8.0x8"):
        asyncio.run(strict.check_hop(yarl.URL(f"http://{host}:80/x"), resolve=False))
    for name in ("0x1g.1", "face.b00c", "1.2.3.example"):
        assert guard._ip_literal(name) is None  # noqa: SLF001


class _LoopbackResolver:
    """aiohttp's resolver answering loopback for every name: the rebinding
    answer a connection would get after a public answer was checked."""

    def __init__(self, *args, **kwargs):
        pass

    async def resolve(self, host, port=0, family=0):
        return [
            {
                "hostname": host,
                "host": "127.0.0.1",
                "port": port,
                "family": 0,
                "proto": 0,
                "flags": 0,
            }
        ]

    async def close(self):
        pass


@pytest.mark.parametrize(
    "scheme,env",
    [
        # NO_PROXY exempts the host: aiohttp connects it directly.
        ("http", {"HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "rebind.example"}),
        # Only an HTTP proxy: aiohttp connects an https URL directly.
        ("https", {"HTTP_PROXY": "http://127.0.0.1:9"}),
    ],
)
def test_a_direct_hop_stays_pinned_while_a_proxy_is_configured(monkeypatch, scheme, env):
    # The name resolves to a public address when checked and to loopback when
    # the connection resolves it: a direct hop is checked by the resolver it
    # connects through, so the rebinding answer is refused.
    import aiohttp.resolver

    for name, value in env.items():
        monkeypatch.setenv(name, value)

    async def lookup(_host):
        return ["93.184.216.34"]

    monkeypatch.setattr(guard, "_lookup", lookup)
    monkeypatch.setattr(aiohttp.resolver, "DefaultResolver", _LoopbackResolver)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        strict = MediaHostPolicy(None, allow_private_hosts=False)
        try:
            with pytest.raises(MediaHostRefused, match="loopback"):
                await _fetch(strict, f"{scheme}://rebind.example:{port}/media")
            assert state["hits"] == 0
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_only_the_hops_own_proxy_is_resolved_unchecked(monkeypatch):
    # The loopback server is the egress proxy, named by a name that resolves
    # to loopback. A hop through it reaches it (its target checked here); a
    # direct hop to the same name, NO_PROXY'd, is checked and refused.
    async def lookup(host):
        assert host == "public.example"
        return ["93.184.216.34"]

    monkeypatch.setattr(guard, "_lookup", lookup)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        strict = MediaHostPolicy(None, allow_private_hosts=False)
        try:
            monkeypatch.setenv("HTTP_PROXY", f"http://localhost:{port}")
            assert await _fetch(strict, "http://public.example/media") == b"img"
            assert state["hits"] == 1
            monkeypatch.setenv("NO_PROXY", "localhost")
            with pytest.raises(MediaHostRefused, match="loopback"):
                await _fetch(strict, f"http://localhost:{port}/media")
            assert state["hits"] == 1
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_behind_a_proxy_the_hop_name_is_resolved_and_checked_here(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")

    async def run():
        with pytest.raises(MediaHostRefused, match="loopback"):
            await _fetch(MediaHostPolicy(None, allow_private_hosts=False), "http://localhost:1/x")

    asyncio.run(run())


def test_a_public_name_passes_the_resolver_and_a_private_answer_does_not():
    resolver_cls = guard._resolver_class()  # noqa: SLF001

    class Fake:
        def __init__(self, answers):
            self.answers = answers

        async def resolve(self, host, port, family):
            return [
                {"hostname": host, "host": a, "port": port, "family": 0, "proto": 0, "flags": 0}
                for a in self.answers
            ]

        async def close(self):
            pass

    async def run():
        resolver = resolver_cls()
        resolver._inner = Fake(["93.184.216.34"])  # noqa: SLF001
        assert (await resolver.resolve("example.com", 80))[0]["host"] == "93.184.216.34"
        # One non-public address among public ones refuses the host.
        resolver._inner = Fake(["93.184.216.34", "10.0.0.5"])  # noqa: SLF001
        with pytest.raises(MediaHostRefused, match="private"):
            await resolver.resolve("rebind.example", 80)

    asyncio.run(run())


def test_the_connection_hands_vllm_the_guarded_session(monkeypatch):
    connections = types.ModuleType("vllm.connections")

    class HTTPConnection:
        async def get_async_client(self):  # pragma: no cover - replaced
            raise AssertionError("the guard must replace the session")

    connections.HTTPConnection = HTTPConnection
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.connections", connections)

    async def run():
        kwargs = guard.media_connector_kwargs(None, allow_private_hosts=False)
        connection = kwargs["connection"]
        session = await connection.get_async_client()
        try:
            assert await connection.get_async_client() is session
            with pytest.raises(MediaHostRefused):
                async with session.get("http://127.0.0.1:1/x"):
                    pass
        finally:
            await session.close()
        with pytest.raises(RuntimeError):
            connection.get_sync_client()

    asyncio.run(run())
