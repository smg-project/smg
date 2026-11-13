"""The launcher's arguments and the Rust router's constructor agree.

A router argument is written in several places by hand: the ``RouterArgs``
dataclass and its command-line flag, the constructor signature of the Rust
``Router`` and the mapping into the router configuration. ``Router.from_args``
hands every dataclass field the Rust constructor takes to it as a keyword, so
a field added on one side only fails at launch. These tests pin the two sides
against each other, and build the real router, with no engine behind it.
"""

from __future__ import annotations

import inspect
import warnings
from unittest.mock import patch

import pytest
from smg.router import Router
from smg.router_args import RouterArgs
from smg.smg_rs import Router as _Router

REGULAR = RouterArgs(worker_urls=["http://worker:8000"])
# A ZMQ worker without --backend: the launcher pins no wire and the gateway's
# detection step decides, as it does for the Rust CLI.
ZMQ_UNPINNED = RouterArgs(worker_urls=["ipc:///tmp/smg-zmq/engine-0"])
# The Gemini provider route, which only the Rust binary could select before.
GEMINI = RouterArgs(backend="gemini", worker_urls=["https://generativelanguage.googleapis.com"])
PREFILL_DECODE = RouterArgs(
    worker_urls=[],
    pd_disaggregation=True,
    prefill_urls=[("http://prefill:8000", None)],
    decode_urls=["http://decode:8000"],
)
# The file, Slurm and Consul providers are the cases where the launcher builds
# the keyword-only ``discovery`` mapping; Kubernetes passes
# ``service_discovery`` and the selectors instead.
FILE_DISCOVERY = RouterArgs(
    worker_urls=[],
    discovery_provider="file",
    discovery_file="/run/smg/workers.json",
    discovery_check_interval_secs=5,
)
SLURM_DISCOVERY = RouterArgs(
    worker_urls=[],
    discovery_provider="slurm",
    slurm_job_name=["smg-worker"],
    slurm_port=8000,
    discovery_check_interval_secs=5,
)
CONSUL_DISCOVERY = RouterArgs(
    worker_urls=[],
    discovery_provider="consul",
    consul_service="smg-worker",
    consul_address="http://consul.internal:8500",
    discovery_check_interval_secs=5,
)


def constructor_parameters() -> set[str]:
    return set(inspect.signature(_Router).parameters)


def keywords_from_args(args: RouterArgs) -> set[str]:
    """The keywords ``from_args`` hands the constructor, captured before the call."""
    with patch("smg.router._Router") as spy:
        Router.from_args(args)
    return set(spy.call_args.kwargs)


@pytest.mark.parametrize(
    "args",
    [REGULAR, FILE_DISCOVERY, SLURM_DISCOVERY, CONSUL_DISCOVERY],
    ids=["regular", "file-discovery", "slurm-discovery", "consul-discovery"],
)
def test_every_keyword_the_launcher_passes_is_a_constructor_parameter(args: RouterArgs):
    unknown = keywords_from_args(args) - constructor_parameters()
    assert not unknown, (
        f"the launcher passes keywords the Rust constructor does not take: {sorted(unknown)}"
    )


def test_every_constructor_parameter_is_set_by_the_launcher():
    unset = constructor_parameters() - keywords_from_args(FILE_DISCOVERY)
    assert not unset, (
        f"the Rust constructor takes parameters the launcher never sets: {sorted(unset)}"
    )


@pytest.mark.parametrize(
    "args",
    [
        REGULAR,
        ZMQ_UNPINNED,
        PREFILL_DECODE,
        GEMINI,
        FILE_DISCOVERY,
        SLURM_DISCOVERY,
        CONSUL_DISCOVERY,
    ],
    ids=[
        "regular",
        "zmq-unpinned",
        "prefill-decode",
        "gemini",
        "file-discovery",
        "slurm-discovery",
        "consul-discovery",
    ],
)
def test_the_launcher_builds_the_real_router(args: RouterArgs):
    # The constructor only keeps the configuration (the file, Slurm and Consul
    # providers' mappings are parsed, no file read, no command run and no
    # request sent); nothing is bound.
    assert Router.from_args(args) is not None


def test_no_backend_reaches_the_native_constructor_as_none():
    """Without ``--backend`` the launcher pins no wire: the constructor gets None
    and the gateway's detection step decides, as it does for the Rust CLI."""
    with patch("smg.router._Router") as constructor:
        Router.from_args(ZMQ_UNPINNED)
    assert constructor.call_args.kwargs["backend"] is None


@pytest.mark.parametrize("settle", [None, 0, 30, 120])
def test_shutdown_settle_reaches_native_constructor(settle):
    """The optional propagation window reaches the native router unchanged."""
    args = RouterArgs(worker_urls=["http://worker:8000"], shutdown_settle_secs=settle)
    with patch("smg.router._Router") as constructor:
        Router.from_args(args)
    assert constructor.call_args.kwargs["shutdown_settle_secs"] == settle
    assert isinstance(Router.from_args(args), Router)


def test_dp_aware_is_accepted_with_a_deprecation_line_and_changes_nothing():
    """The retired ``dp_aware`` keyword still parses, draws one deprecation line
    that names the release it leaves in, and reaches the native constructor
    beside otherwise identical keywords: the gateway discovers data-parallel
    ranks at every registration whatever it says."""
    kwargs = {}
    for dp_aware in (False, True):
        args = RouterArgs(worker_urls=["http://worker:8000"], dp_aware=dp_aware)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("smg.router._Router") as constructor:
                Router.from_args(args)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        if dp_aware:
            assert len(deprecations) == 1, [str(w.message) for w in caught]
            message = str(deprecations[0].message)
            assert "dp_aware is deprecated and ignored" in message
            assert "removed in 1.17" in message
        else:
            assert deprecations == []
        kwargs[dp_aware] = dict(constructor.call_args.kwargs)
    assert kwargs[True].pop("dp_aware") is True
    assert kwargs[False].pop("dp_aware") is False
    assert kwargs[True] == kwargs[False]
    assert isinstance(Router.from_args(RouterArgs(worker_urls=["http://worker:8000"])), Router)
