"""Which hosts a worker fetches client media from: the router's fetch policy
(``crates/multimodal/src/fetch_policy.rs``) for the fetches a worker makes.

vLLM's MediaConnector checks only the first URL's host name against
``--allowed-media-domains`` and follows redirects, so a worker that fetches
the URLs a router forwards (``--mm-processor inprocess`` or ``redis``) would
otherwise fetch from loopback, link-local (cloud metadata) and private
addresses that the router itself refuses. The guard here hands the connector
an HTTP connection whose session:

* resolves every host through :class:`PublicOnlyResolver`, which refuses a
  name that resolves to any non-public address; the connection goes to the
  addresses it checked, so a DNS answer cannot change between the check and
  the connect;
* checks every hop (the URL and each redirect target) before it is sent, in
  a client middleware, which aiohttp runs on the request it is about to send:
  the URL it follows, whichever header (``Location`` or ``URI``) named it,
  and the proxy it chose for that URL (``*_PROXY`` and ``NO_PROXY``, per hop).
  An IP literal is checked by address (aiohttp never resolves one), the
  legacy IPv4 spellings ``inet_aton`` accepts (``2130706433``, ``127.1``,
  ``0177.0.0.1``, ``0x7f.1``) included, read as the router's URL parser reads
  them; the host name is checked against the allow-list when one is
  configured, read as the router reads it (an IPv6 address named bracketed,
  as vLLM's own first-URL check names it); a host with an IPv6 zone ID is
  refused, as the router's URL parser refuses it. A hop sent through an egress proxy is resolved by the proxy, so
  its name is resolved and checked here instead, without a pin, as the router
  does; the resolver passes that hop's proxy name, and only that one,
  unchecked. A hop sent directly is resolved and pinned as above.

A refusal is a :class:`MediaHostRefused` (a ``ValueError``): the request's
media, terminal. A refusal of the URL the request named is answered as
``invalid_multimodal_request``; one of a redirect target, a
:class:`MediaRedirectRefused`, as ``media_fetch_failed``, as the router
answers a redirect its policy refuses.
"""

from __future__ import annotations

import asyncio
import contextvars
import ipaddress
import logging
import re
import socket
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class MediaHostRefused(ValueError):
    """A media URL's host the policy refuses: the caller's, never retried."""


class MediaRedirectRefused(MediaHostRefused):
    """A redirect target the policy refuses: the URL the request named was
    admitted and its origin sent the fetch somewhere the worker does not go,
    a fetch that failed (as the router answers a redirect its policy
    refuses), not a reference refused as the request sent it."""


# Set on whatever the guarded connection's download raised (a hop refused, an
# origin's error status, a broken transfer, a size cap), so a caller can tell
# a failed fetch from a failure to read the bytes a fetch delivered, whatever
# vLLM wraps the download's error in.
_RAISED_BY_FETCH_ATTR = "smg_raised_by_media_fetch"


def mark_raised_by_fetch(exc: BaseException) -> BaseException:
    """`exc`, marked as raised by a media download."""
    setattr(exc, _RAISED_BY_FETCH_ATTR, True)
    return exc


def raised_by_fetch(exc: BaseException) -> bool:
    """Whether `exc` itself was raised by a media download (see
    :func:`mark_raised_by_fetch`); the exceptions it was raised from are the
    caller's to walk."""
    return getattr(exc, _RAISED_BY_FETCH_ATTR, False) is True


def _ipv4_class(address: ipaddress.IPv4Address) -> str | None:
    first, second, third, _ = address.packed
    if first == 0:
        return "unspecified"
    if address.is_loopback:
        return "loopback"
    if address.is_link_local:
        return "link-local"
    if first == 100 and 64 <= second < 128:
        return "shared address space"
    if first == 192 and second == 0 and third == 0:
        return "reserved"
    if first == 198 and second in (18, 19):
        return "benchmarking"
    if address.is_multicast:
        return "multicast"
    if address == ipaddress.IPv4Address("255.255.255.255"):
        return "broadcast"
    if first >= 240:
        return "reserved"
    if any(address in network for network in _IPV4_PRIVATE):
        return "private"
    return None


_IPV4_PRIVATE = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
)
_NAT64_LOCAL_USE = ipaddress.IPv6Network("64:ff9b:1::/48")


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 address leads a translator or relay to, as the
    router reads it: mapped, compatible, NAT64 (/96 well-known and local-use)
    and 6to4."""
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    segments = [int.from_bytes(address.packed[i : i + 2], "big") for i in range(0, 16, 2)]
    low_32 = ipaddress.IPv4Address(address.packed[12:16])
    if segments[:6] == [0] * 6:
        return None if segments[6] == 0 and segments[7] in (0, 1) else low_32
    if segments[:6] == [0x64, 0xFF9B, 0, 0, 0, 0] or segments[:6] == [0x64, 0xFF9B, 1, 0, 0, 0]:
        return low_32
    if segments[0] == 0x2002:
        return ipaddress.IPv4Address(address.packed[2:6])
    return None


def _ipv6_class(address: ipaddress.IPv6Address) -> str | None:
    if address.is_unspecified:
        return "unspecified"
    if address.is_loopback:
        return "loopback"
    if address in ipaddress.IPv6Network("fc00::/7"):
        return "unique local"
    if address in ipaddress.IPv6Network("fe80::/10"):
        return "link-local"
    if address.is_multicast:
        return "multicast"
    if address in _NAT64_LOCAL_USE:
        # A layout whose IPv4 address this side cannot read (see the router).
        return "NAT64 local-use with an unreadable layout"
    return None


def non_public_address_class(address: IpAddress) -> str | None:
    """The class of an address media is not fetched from; None when public."""
    if isinstance(address, ipaddress.IPv4Address):
        return _ipv4_class(address)
    carried = _embedded_ipv4(address)
    return _ipv4_class(carried) if carried is not None else _ipv6_class(address)


def check_address(host: str, address: str | IpAddress) -> None:
    """Refuse `host` when `address`, one it names or resolves to, is not public."""
    ip = ipaddress.ip_address(address.split("%", 1)[0] if isinstance(address, str) else address)
    if (cls := non_public_address_class(ip)) is not None:
        raise MediaHostRefused(f"media host {host!r} is a {cls} address ({ip}), not fetched")


# A host every label of which is a number (decimal, octal or hex), which the
# socket takes for an IPv4 address rather than a name to resolve, as the
# router's URL parser does; aiohttp's connector also skips the resolver for
# any host of digits and dots.
_NUMERIC_IPV4 = re.compile(r"(?:0x[0-9a-f]*|[0-9]+)(?:\.(?:0x[0-9a-f]*|[0-9]+))*\.*", re.IGNORECASE)


def _ip_literal(host: str) -> IpAddress | None:
    """The address `host` names as an IP literal, read as the connection
    would read it; None for a name, which is resolved. A literal is never
    resolved (aiohttp hands anything with a colon, or of digits and dots, to
    the socket as it is), so a host that looks like one and does not parse is
    refused rather than passed on unchecked."""
    bare = host.strip("[]")
    try:
        return ipaddress.ip_address(bare.split("%", 1)[0])
    except ValueError:
        pass
    if ":" in bare:
        raise MediaHostRefused(f"media host {host!r} is not a valid IP address")
    if bare.replace(".", "").isdigit() or _NUMERIC_IPV4.fullmatch(bare):
        # The legacy forms inet_aton (and so the socket) accepts: one to four
        # parts, each decimal, octal (leading 0) or hex (0x), the last filling
        # the remaining bytes. A trailing dot names the same address.
        try:
            return ipaddress.IPv4Address(socket.inet_aton(bare.rstrip(".")))
        except (OSError, ValueError):
            raise MediaHostRefused(f"media host {host!r} is not a valid IPv4 address") from None
    return None


def _names_a_zone(entry: str) -> bool:
    """Whether an allow-list entry names an IPv6 address with a zone ID,
    which no media URL may carry (:meth:`MediaHostPolicy.check_hop`)."""
    return ":" in entry and "%" in entry


def effective_allowed_domains(allowed_domains: Iterable[str] | None) -> list[str] | None:
    """An allow-list as both of a worker's checks read it, vLLM's own
    first-URL check and the policy's check of every hop: each entry trimmed
    and lowercased, empty entries and entries with an IPv6 zone ID dropped,
    each once, as the router reads its list (``effective_allowed_domains``
    in ``fetch_policy.rs``). None when no list is configured, which allows
    any host. A configured list none of whose entries is left is an empty
    list, which refuses every host: a reading never allows more than the
    list as configured did (the router would read it as no list)."""
    configured = list(allowed_domains or ())
    if not configured:
        return None
    return sorted(
        {d.strip().lower() for d in configured if d and d.strip() and not _names_a_zone(d)}
    )


def _bracketed_ipv6(text: str) -> str | None:
    """`text`, an IPv6 literal with or without brackets, as ``[address]`` in
    its compressed spelling; None when it is no IPv6 address."""
    bare = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    if ":" not in bare:
        return None
    try:
        return f"[{ipaddress.IPv6Address(bare)}]"
    except ValueError:
        return None


def _allow_list_key(entry: str) -> str:
    """An allow-list entry as the policy compares hops with it. An IPv6
    address is named bracketed, as vLLM's check (urllib3's hostname) and the
    router's URL parser spell it. The policy compares it by address, so any
    spelling of the entry matches any spelling of a hop; vLLM's first-URL
    check is exact, so the first URL must spell it as the entry does. A bare
    IPv6 entry names no host, as for vLLM and the router."""
    if entry.startswith("["):
        return _bracketed_ipv6(entry) or entry
    return entry


class MediaHostPolicy:
    """The allow-list and the address policy one worker's fetches follow."""

    def __init__(self, allowed_domains: Iterable[str] | None, allow_private_hosts: bool) -> None:
        domains = effective_allowed_domains(allowed_domains)
        self.allowed_domains = (
            frozenset(_allow_list_key(d) for d in domains) if domains is not None else None
        )
        self.allow_private_hosts = allow_private_hosts

    def check_name(self, host: str | None, *spellings: str | None) -> str:
        """`host` against the allow-list, by name; `spellings` are other
        names of the same host (its IDNA form) the list may give instead."""
        if not host:
            raise MediaHostRefused("media URL has no host")
        host = host.lower()
        if self.allowed_domains is not None and not any(
            name.lower() in self.allowed_domains for name in (host, *spellings) if name
        ):
            raise MediaHostRefused(f"media host {host!r} is not in --allowed-media-domains")
        return host

    async def check_hop(self, url: Any, *, resolve: bool) -> None:
        """One hop's URL (a yarl URL), before the request is sent: its host
        as the connection uses it (`raw_host`, IDNA-encoded) by address.
        `resolve` checks a name by its addresses here (a hop sent through a
        proxy, where the resolver never sees it); otherwise the resolver
        checks names as it connects."""
        raw_host = url.raw_host
        if raw_host and ":" in raw_host:
            # An IPv6 literal, which yarl gives bare and compressed: named
            # bracketed on the allow-list, judged bare by address. A zone ID
            # (an interface of this host) is refused whatever the policy, as
            # the router's URL parser refuses it.
            if "%" in raw_host:
                raise MediaHostRefused(
                    f"media host {raw_host!r} carries an IPv6 zone ID, not fetched"
                )
            name = _bracketed_ipv6(raw_host.lower())
            if name is None:
                raise MediaHostRefused(f"media host {raw_host!r} is not a valid IP address")
            self.check_name(name)
            host = raw_host.lower()
        else:
            host = self.check_name(raw_host, url.host)
        if self.allow_private_hosts:
            return
        if (ip := _ip_literal(host)) is not None:
            check_address(host, ip)
        elif resolve:
            for address in await _lookup(host):
                check_address(host, address)


async def _lookup(host: str) -> list[str]:
    """The addresses `host` resolves to here, for a hop a proxy connects."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


# The proxy host of the hop being connected, when that hop goes through a
# proxy: the one name the resolver resolves without checking it (an egress
# proxy is usually an internal address). Set by the middleware around the
# connect of each hop, so a direct hop, a later hop, or a concurrent request
# never inherits it.
_HOP_PROXY_HOST: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "media_hop_proxy_host", default=None
)


# Whether the hop being sent is a redirect target rather than the URL the
# request named: cleared when a request starts and set when it follows a
# redirect (aiohttp's trace signals, sent in the request's own context around
# each hop), so the middleware and the resolver of that hop read it.
_HOP_IS_REDIRECT: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "media_hop_is_redirect", default=False
)


def _same_host(a: str, b: str) -> bool:
    return a.rstrip(".").lower() == b.rstrip(".").lower()


def _resolver_class():
    from aiohttp.abc import AbstractResolver
    from aiohttp.resolver import DefaultResolver

    class PublicOnlyResolver(AbstractResolver):
        """Resolves through aiohttp's default resolver and refuses a host any
        of whose addresses is not public; the connection uses exactly the
        answer checked here."""

        def __init__(self) -> None:
            self._inner = DefaultResolver()

        async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
            results = await self._inner.resolve(host, port, family)
            proxy_host = _HOP_PROXY_HOST.get()
            if proxy_host is not None and _same_host(host, proxy_host):
                # This hop's proxy, which connects the target (checked by
                # the middleware): an operator's address, not the client's.
                return results
            for result in results:
                check_address(host, result["host"])
            return results

        async def close(self) -> None:
            await self._inner.close()

    return PublicOnlyResolver


def guarded_session(policy: MediaHostPolicy):
    """An aiohttp session (vLLM's own: `trust_env=True`) under `policy`."""
    import aiohttp

    async def guard_hop(req, handler):
        # Run on every hop, the first URL and each redirect target, on the
        # request aiohttp is about to send: `req.url` is the URL it follows
        # (from Location or URI, joined as it joins them) and `req.proxy` the
        # proxy it chose for that URL, NO_PROXY applied.
        proxy = req.proxy
        try:
            await policy.check_hop(req.url, resolve=proxy is not None)
            token = _HOP_PROXY_HOST.set(proxy.raw_host if proxy is not None else None)
            try:
                return await handler(req)
            finally:
                _HOP_PROXY_HOST.reset(token)
        except MediaHostRefused as e:
            # Refused here or by the resolver as this hop connects.
            if not _HOP_IS_REDIRECT.get() or isinstance(e, MediaRedirectRefused):
                raise
            raise MediaRedirectRefused(f"redirect refused: {e}") from e

    async def request_started(session, ctx, params) -> None:
        _HOP_IS_REDIRECT.set(False)

    async def redirected(session, ctx, params) -> None:
        _HOP_IS_REDIRECT.set(True)

    hops = aiohttp.TraceConfig()
    hops.on_request_start.append(request_started)
    hops.on_request_redirect.append(redirected)

    connector = None
    if not policy.allow_private_hosts:
        # No DNS cache: every connect resolves (and is checked) in the
        # request's own context, never from an answer another hop resolved.
        connector = aiohttp.TCPConnector(resolver=_resolver_class()(), use_dns_cache=False)
    return aiohttp.ClientSession(
        trust_env=True, connector=connector, middlewares=(guard_hop,), trace_configs=[hops]
    )


def guarded_connection(policy: MediaHostPolicy):
    """A vLLM HTTPConnection whose async session is :func:`guarded_session`."""
    from vllm.connections import HTTPConnection

    class GuardedHTTPConnection(HTTPConnection):
        _guarded: Any = None

        async def get_async_client(self):
            if self._guarded is None or self._guarded.closed:
                self._guarded = guarded_session(policy)
            return self._guarded

        def get_sync_client(self):
            raise RuntimeError("worker media fetches are async only")

        async def async_get_bytes(self, *args: Any, **kwargs: Any) -> bytes:
            # The one network step of vLLM's loader: what fails here is the
            # fetch, never the decode of the bytes it delivered. Except a URL
            # the request named that aiohttp will not send at all (no host, a
            # scheme it does not fetch): refused as the request sent it, so
            # left unmarked; the same refusal of a redirect target is the
            # fetch's. aiohttp may refuse the named URL before the request's
            # start resets the hop, so the download resets it here first.
            from aiohttp import InvalidURL, NonHttpUrlClientError

            _HOP_IS_REDIRECT.set(False)
            try:
                return await super().async_get_bytes(*args, **kwargs)
            except Exception as e:
                unsendable = isinstance(e, (InvalidURL, NonHttpUrlClientError))
                if not unsendable or _HOP_IS_REDIRECT.get():
                    mark_raised_by_fetch(e)
                raise

    return GuardedHTTPConnection()


def media_connector_kwargs(
    allowed_media_domains: Iterable[str] | None, allow_private_hosts: bool
) -> dict[str, Any]:
    """The `connection=` and `allowed_media_domains=` a worker's
    MediaConnector is built with: vLLM checks the first URL against the same
    reading of the allow-list the connection checks every hop against."""
    configured = list(allowed_media_domains or ())
    for entry in configured:
        if entry and _names_a_zone(entry):
            logger.warning(
                "--allowed-media-domains entry %r names an IPv6 zone ID, which no media URL "
                "may carry: ignored",
                entry,
            )
    domains = effective_allowed_domains(configured)
    if domains == []:
        logger.warning(
            "--allowed-media-domains %r names no host a media URL may name: every media URL "
            "is refused",
            configured,
        )
    policy = MediaHostPolicy(configured, allow_private_hosts)
    if allow_private_hosts:
        logger.warning(
            "worker media fetches may reach private, loopback and link-local hosts "
            "(--mm-allow-private-media-hosts)"
        )
    return {
        "connection": guarded_connection(policy),
        # A list that names no host stays as configured for vLLM, whose check
        # reads an empty list as none; the connection refuses every hop.
        "allowed_media_domains": domains or (configured or None),
    }
