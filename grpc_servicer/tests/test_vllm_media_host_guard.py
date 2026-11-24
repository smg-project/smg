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


async def _serve(handler_routes, host="127.0.0.1"):
    app = web.Application()
    app.add_routes(handler_routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, 0)
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
                with pytest.raises(MediaHostRefused) as raised:
                    await _fetch(strict, url)
                # The URL the request named: refused as sent, not a redirect.
                assert type(raised.value) is MediaHostRefused
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
            with pytest.raises(guard.MediaRedirectRefused, match="allowed-media-domains"):
                await _fetch(only_listed, f"http://localhost:{port}/bounce?to={target}")
            assert state["hits"] == 1
            # The next request in the same task starts at its own first hop.
            with pytest.raises(MediaHostRefused) as raised:
                await _fetch(only_listed, f"http://127.0.0.1:{port}/media")
            assert type(raised.value) is MediaHostRefused
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
            with pytest.raises(guard.MediaRedirectRefused, match="loopback"):
                await _fetch(strict, f"http://localhost:{port}/{bounce}?to={target}")
            assert state["hits"] == 0
            # Decimal spelling of 127.0.0.1, which aiohttp takes for a literal.
            numeric = f"http://2130706433:{port}/media"
            with pytest.raises(guard.MediaRedirectRefused, match="loopback"):
                await _fetch(strict, f"http://localhost:{port}/{bounce}?to={numeric}")
            assert state["hits"] == 0
            listed = MediaHostPolicy(["localhost"], allow_private_hosts=True)
            with pytest.raises(guard.MediaRedirectRefused, match="allowed-media-domains"):
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


def _hop(policy, url):
    yarl = pytest.importorskip("yarl")
    asyncio.run(policy.check_hop(yarl.URL(url), resolve=False))


def test_ipv6_allow_list_entries_are_bracketed_and_compared_by_address():
    # vLLM's first-URL check (urllib3's hostname) and the router's URL parser
    # both name an IPv6 host bracketed, so that is how the list names it.
    for allow_private in (False, True):
        for entry in ("[2606:4700::1111]", "[2606:4700:0:0::1111]", " [2606:4700::1111] "):
            policy = MediaHostPolicy([entry], allow_private_hosts=allow_private)
            _hop(policy, "http://[2606:4700::1111]/x")
            _hop(policy, "http://[2606:4700:0:0::1111]/x")
            with pytest.raises(MediaHostRefused, match="allowed-media-domains"):
                _hop(policy, "http://[2606:4700::2222]/x")
        # A bare IPv6 entry names no host, as for vLLM and the router.
        with pytest.raises(MediaHostRefused, match="allowed-media-domains"):
            _hop(
                MediaHostPolicy(["2606:4700::1111"], allow_private_hosts=allow_private),
                "http://[2606:4700::1111]/x",
            )
    # Listed, a loopback address is still refused by address...
    with pytest.raises(MediaHostRefused, match="loopback"):
        _hop(MediaHostPolicy(["[::1]"], allow_private_hosts=False), "http://[::1]/x")
    # ...and reached once private hosts are allowed.
    _hop(MediaHostPolicy(["[::1]"], allow_private_hosts=True), "http://[::1]/x")


@pytest.mark.parametrize("allow_private", [False, True])
@pytest.mark.parametrize(
    "url", ["http://[fe80::1%25eth0]/x", "http://[::1%25lo]/x", "http://[2606:4700::1111%25eth0]/x"]
)
def test_a_host_with_an_ipv6_zone_id_is_refused_whatever_the_policy(url, allow_private):
    # The router's URL parser refuses a zone ID; so does every hop here,
    # public address, listed host or private hosts allowed.
    for entries in (None, ["[fe80::1%eth0]", "[::1]", "[2606:4700::1111]"]):
        policy = MediaHostPolicy(entries, allow_private_hosts=allow_private)
        with pytest.raises(MediaHostRefused, match="zone ID"):
            _hop(policy, url)


def test_a_redirect_to_a_name_the_resolver_refuses_is_a_redirect_refusal(monkeypatch):
    # The first hop's literal passes (as a public address would); its
    # redirect names a host that resolves to loopback, which the resolver
    # refuses as that hop connects: still the redirect's refusal.
    real = guard.check_address

    def check_address(host, address):
        if host != "127.0.0.1":
            real(host, address)

    monkeypatch.setattr(guard, "check_address", check_address)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        strict = MediaHostPolicy(None, allow_private_hosts=False)
        try:
            target = f"http://localhost:{port}/media"
            with pytest.raises(guard.MediaRedirectRefused, match="loopback"):
                await _fetch(strict, f"http://127.0.0.1:{port}/bounce?to={target}")
            assert state["hits"] == 0
            # The same name as the URL the request named: refused as sent.
            with pytest.raises(MediaHostRefused, match="loopback") as raised:
                await _fetch(strict, target)
            assert type(raised.value) is MediaHostRefused
            assert state["hits"] == 0
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("allow_private", [False, True])
@pytest.mark.parametrize("bounce", ["bounce", "bounce-uri"])
def test_a_redirect_to_a_host_with_a_zone_id_is_refused(monkeypatch, allow_private, bounce):
    # The first hop's name passes (as a public name would); the redirect's
    # zone ID is refused before the hop is sent.
    real = guard.check_address

    def check_address(host, address):
        if host != "localhost":
            real(host, address)

    monkeypatch.setattr(guard, "check_address", check_address)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        policy = MediaHostPolicy(None, allow_private_hosts=allow_private)
        try:
            for target in (f"http://[::1%25lo]:{port}/media", f"http://[fe80::1%25eth0]:{port}/x"):
                with pytest.raises(guard.MediaRedirectRefused, match="zone ID"):
                    await _fetch(policy, f"http://localhost:{port}/{bounce}?to={target}")
            with pytest.raises(MediaHostRefused, match="zone ID") as raised:
                await _fetch(policy, f"http://[::1%25lo]:{port}/media")
            assert type(raised.value) is MediaHostRefused
            assert state["hits"] == 0
        finally:
            await runner.cleanup()

    asyncio.run(run())


@pytest.fixture
def stub_vllm_connections(monkeypatch):
    connections = types.ModuleType("vllm.connections")

    class HTTPConnection:
        async def get_async_client(self):  # pragma: no cover - replaced
            raise AssertionError("the guard must replace the session")

    connections.HTTPConnection = HTTPConnection
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.connections", connections)


def test_an_allow_list_entry_with_a_zone_id_is_ignored_with_a_warning(
    stub_vllm_connections, caplog
):
    with caplog.at_level("WARNING", logger=guard.__name__):
        kwargs = guard.media_connector_kwargs(
            ["[fe80::1%ETH0]", "Example.com"], allow_private_hosts=True
        )
    assert "'[fe80::1%ETH0]' names an IPv6 zone ID" in caplog.text
    assert kwargs["allowed_media_domains"] == ["example.com"]
    policy = MediaHostPolicy(["[fe80::1%ETH0]", "example.com"], allow_private_hosts=True)
    assert policy.allowed_domains == frozenset({"example.com"})


@pytest.mark.parametrize("entries", [[" ", ""], ["[fe80::1%eth0]"]])
def test_a_list_that_names_no_usable_host_refuses_every_host(
    stub_vllm_connections, caplog, entries
):
    # Normalising never allows more than the list as configured did: vLLM
    # refused every host for a list of blanks, and so does the worker now.
    with caplog.at_level("WARNING", logger=guard.__name__):
        kwargs = guard.media_connector_kwargs(entries, allow_private_hosts=True)
    assert "every media URL is refused" in caplog.text
    # vLLM reads an empty list as none: it is handed the list as configured.
    assert kwargs["allowed_media_domains"] == entries
    for allow_private in (False, True):
        policy = MediaHostPolicy(entries, allow_private_hosts=allow_private)
        assert policy.allowed_domains == frozenset()
        for url in ("http://example.com/x", "http://localhost/x", "http://[::1]/x"):
            with pytest.raises(MediaHostRefused, match="allowed-media-domains"):
                _hop(policy, url)
    assert guard.effective_allowed_domains(None) is None
    assert guard.effective_allowed_domains([]) is None


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


def test_the_connections_download_marks_what_it_raises(stub_vllm_connections, monkeypatch):
    """vLLM's one network step: what it raises is told apart from a failure
    to decode the bytes it delivered, whatever vLLM later wraps it in."""
    failure = ValueError("HTTP 403")
    answers = [b"bytes", failure]

    async def async_get_bytes(self, url, **kwargs):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    connections = sys.modules["vllm.connections"]
    monkeypatch.setattr(
        connections.HTTPConnection, "async_get_bytes", async_get_bytes, raising=False
    )
    connection = guard.media_connector_kwargs(None, allow_private_hosts=False)["connection"]
    assert asyncio.run(connection.async_get_bytes("https://a/1.png", timeout=1)) == b"bytes"
    with pytest.raises(ValueError) as raised:
        asyncio.run(connection.async_get_bytes("https://a/1.png", timeout=1))
    assert raised.value is failure
    assert guard.raised_by_fetch(failure)
    assert not guard.raised_by_fetch(ValueError("Failed to load image"))


async def _download(self, url, **kwargs):
    """vLLM's HTTPConnection.async_get_bytes, as far as these tests need."""
    client = await self.get_async_client()
    async with client.get(url) as response:
        response.raise_for_status()
        return await response.read()


def test_a_named_url_aiohttp_will_not_send_is_refused_as_sent(stub_vllm_connections, monkeypatch):
    # No host, or one that does not parse: aiohttp refuses the URL the request
    # named before any hop is sent (some before the request even starts), the
    # request's reference as sent. Its refusal of a redirect target is the
    # fetch's, and does not leak into the next download in the same task.
    connections = sys.modules["vllm.connections"]
    monkeypatch.setattr(connections.HTTPConnection, "async_get_bytes", _download, raising=False)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        connection = guard.media_connector_kwargs(None, allow_private_hosts=True)["connection"]
        try:
            for redirect_first in (False, True):
                if redirect_first:
                    with pytest.raises(aiohttp.NonHttpUrlClientError) as raised:
                        await connection.async_get_bytes(
                            f"http://127.0.0.1:{port}/bounce?to=ftp://media.example/a.png"
                        )
                    assert guard.raised_by_fetch(raised.value)
                for url in ("http://", "http://[::1"):
                    with pytest.raises(aiohttp.InvalidURL) as raised:
                        await connection.async_get_bytes(url)
                    assert not guard.raised_by_fetch(raised.value), url
            assert state["hits"] == 0
        finally:
            await connection._guarded.close()  # noqa: SLF001
            await runner.cleanup()

    asyncio.run(run())


def test_concurrent_downloads_on_one_session_keep_their_own_hop(stub_vllm_connections, monkeypatch):
    # One guarded session, many downloads at once: a redirect target refused
    # in one never makes another's refusal of its named URL a redirect's, and
    # the other way round.
    connections = sys.modules["vllm.connections"]
    monkeypatch.setattr(connections.HTTPConnection, "async_get_bytes", _download, raising=False)

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        connection = guard.media_connector_kwargs(["localhost"], allow_private_hosts=True)[
            "connection"
        ]
        target = f"http://127.0.0.1:{port}/media"
        redirected = f"http://localhost:{port}/bounce?to={target}"
        try:
            urls = [redirected, target] * 16
            results = await asyncio.gather(
                *(connection.async_get_bytes(url) for url in urls), return_exceptions=True
            )
            for url, result in zip(urls, results, strict=True):
                assert isinstance(result, MediaHostRefused), (url, result)
                expected = guard.MediaRedirectRefused if url == redirected else MediaHostRefused
                assert type(result) is expected, (url, result)
            assert state["hits"] == 0
        finally:
            await connection._guarded.close()  # noqa: SLF001
            await runner.cleanup()

    asyncio.run(run())


def test_the_connection_hands_vllm_the_guarded_session(stub_vllm_connections):
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


class _RawBytes:
    """A MediaIO that hands back the fetched bytes."""

    def get_max_bytes(self):
        # Asked by later vLLM releases before a fetch; no cap here.
        return None

    def load_bytes(self, data):
        return data


async def _ipv6_loopback_server(state):
    try:
        return await _serve(_routes(state), host="::1")
    except OSError:
        pytest.skip("no IPv6 loopback here")


@pytest.mark.parametrize(
    "entries,host,v6",
    [
        # vLLM was handed the list as given: an exact-match miss on its side.
        ([" LocalHost "], "localhost", False),
        (["Images.Example.com", "localhost", ""], "localhost", False),
        # The guard compared a bare IPv6 host with the bracketed entry.
        (["[::1]"], "[::1]", True),
        (["[0:0::1]"], "[0:0::1]", True),
    ],
)
def test_vllms_connector_and_the_guard_read_the_allow_list_alike(entries, host, v6):
    connector_module = pytest.importorskip("vllm.multimodal.media.connector")
    vllm_exceptions = pytest.importorskip("vllm.exceptions")

    async def run():
        state = {"hits": 0}
        if v6:
            runner, port = await _ipv6_loopback_server(state)
        else:
            runner, port = await _serve(_routes(state))
        connector = connector_module.MediaConnector(
            **guard.media_connector_kwargs(entries, allow_private_hosts=True)
        )
        try:
            url = f"http://{host}:{port}/media"
            assert await connector.load_from_url_async(url, _RawBytes()) == b"img"
            assert state["hits"] == 1
            # A redirect off the list is still refused by the guard (which
            # later vLLM releases re-raise as their own client error).
            elsewhere = f"http://127.0.0.2:{port}/media"
            refused = (ValueError, getattr(vllm_exceptions, "VLLMClientError", ValueError))
            with pytest.raises(refused, match="127.0.0.2|Invalid URL"):
                await connector.load_from_url_async(
                    f"http://{host}:{port}/bounce?to={elsewhere}", _RawBytes()
                )
            assert state["hits"] == 1
        finally:
            session = connector.connection._guarded  # noqa: SLF001
            if session is not None:
                await session.close()
            await runner.cleanup()

    asyncio.run(run())


def test_vllms_connector_wraps_a_named_host_and_a_redirect_refusal_apart(monkeypatch):
    # Through vLLM's own loader, which re-raises a download's error as its
    # client error: the named URL refused is the request's reference as sent,
    # the redirect target refused a fetch that failed, as the router answers.
    connector_module = pytest.importorskip("vllm.multimodal.media.connector")
    from smg_grpc_servicer.vllm import mm_processor

    real = guard.check_address

    def check_address(host, address):
        if host != "localhost":
            real(host, address)

    monkeypatch.setattr(guard, "check_address", check_address)
    processor = mm_processor.InProcessMediaProcessor.__new__(mm_processor.InProcessMediaProcessor)

    async def code_of(connector, url):
        with pytest.raises(Exception) as raised:  # noqa: B017 - vLLM's own wrapping
            await connector.load_from_url_async(url, _RawBytes())
        return processor._client_media_code(raised.value)  # noqa: SLF001

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        connector = connector_module.MediaConnector(
            **guard.media_connector_kwargs(None, allow_private_hosts=False)
        )
        try:
            target = f"http://127.0.0.1:{port}/media"
            assert await code_of(connector, target) == "invalid_multimodal_request"
            bounce = f"http://localhost:{port}/bounce?to={target}"
            assert await code_of(connector, bounce) == "media_fetch_failed"
            assert state["hits"] == 0
        finally:
            session = connector.connection._guarded  # noqa: SLF001
            if session is not None:
                await session.close()
            await runner.cleanup()

    asyncio.run(run())


def test_vllms_connector_refuses_a_url_with_no_host_as_sent():
    # A named URL aiohttp will not send at all is the request's reference as
    # sent, as the router answers it; the same refusal of a redirect target
    # (a scheme it does not fetch) is a fetch that failed.
    connector_module = pytest.importorskip("vllm.multimodal.media.connector")
    from smg_grpc_servicer.vllm import mm_processor

    processor = mm_processor.InProcessMediaProcessor.__new__(mm_processor.InProcessMediaProcessor)

    async def code_of(connector, url):
        with pytest.raises(Exception) as raised:  # noqa: B017 - vLLM's own wrapping
            await connector.load_from_url_async(url, _RawBytes())
        return processor._client_media_code(raised.value)  # noqa: SLF001

    async def run():
        state = {"hits": 0}
        runner, port = await _serve(_routes(state))
        connector = connector_module.MediaConnector(
            **guard.media_connector_kwargs(None, allow_private_hosts=True)
        )
        try:
            bounce = f"http://127.0.0.1:{port}/bounce?to=ftp://media.example/a.png"
            assert await code_of(connector, bounce) == "media_fetch_failed"
            assert await code_of(connector, "http://") == "invalid_multimodal_request"
            assert state["hits"] == 0
        finally:
            session = connector.connection._guarded  # noqa: SLF001
            if session is not None:
                await session.close()
            await runner.cleanup()

    asyncio.run(run())


def test_vllms_connector_refuses_a_host_off_the_list_before_the_guard():
    connector_module = pytest.importorskip("vllm.multimodal.media.connector")
    connector = connector_module.MediaConnector(
        **guard.media_connector_kwargs(["[::1]"], allow_private_hosts=True)
    )
    with pytest.raises(ValueError, match="allowed domains"):
        asyncio.run(connector.load_from_url_async("http://localhost:1/x", _RawBytes()))


def test_a_mixed_ipv6_spelling_passes_the_guard_but_not_vllms_exact_check():
    # vLLM checks the first URL's host as the URL spells it, for exact
    # membership; the guard compares an IPv6 hop with the entry by address.
    connector_module = pytest.importorskip("vllm.multimodal.media.connector")
    vllm_exceptions = pytest.importorskip("vllm.exceptions")

    async def run():
        state = {"hits": 0}
        runner, port = await _ipv6_loopback_server(state)
        connector = connector_module.MediaConnector(
            **guard.media_connector_kwargs(["[0:0::1]"], allow_private_hosts=True)
        )
        try:
            refused = (ValueError, getattr(vllm_exceptions, "VLLMClientError", ValueError))
            with pytest.raises(refused, match="allowed domains"):
                await connector.load_from_url_async(f"http://[::1]:{port}/media", _RawBytes())
            assert state["hits"] == 0
            target = f"http://[::1]:{port}/media"
            first = f"http://[0:0::1]:{port}/bounce?to={target}"
            assert await connector.load_from_url_async(first, _RawBytes()) == b"img"
            assert state["hits"] == 1
        finally:
            session = connector.connection._guarded  # noqa: SLF001
            if session is not None:
                await session.close()
            await runner.cleanup()

    asyncio.run(run())
