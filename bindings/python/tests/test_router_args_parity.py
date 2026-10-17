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
from unittest.mock import patch

import pytest
from smg.router import Router
from smg.router_args import RouterArgs
from smg.smg_rs import Router as _Router

REGULAR = RouterArgs(worker_urls=["http://worker:8000"])
PREFILL_DECODE = RouterArgs(
    worker_urls=[],
    pd_disaggregation=True,
    prefill_urls=[("http://prefill:8000", None)],
    decode_urls=["http://decode:8000"],
)
# The file provider is the one case where the launcher builds the keyword-only
# ``discovery`` mapping; every other provider passes ``service_discovery`` and
# the selectors instead.
FILE_DISCOVERY = RouterArgs(
    worker_urls=[],
    discovery_provider="file",
    discovery_file="/run/smg/workers.json",
    discovery_check_interval_secs=5,
)


def constructor_parameters() -> set[str]:
    return set(inspect.signature(_Router).parameters)


def keywords_from_args(args: RouterArgs) -> set[str]:
    """The keywords ``from_args`` hands the constructor, captured before the call."""
    with patch("smg.router._Router") as spy:
        Router.from_args(args)
    return set(spy.call_args.kwargs)


@pytest.mark.parametrize("args", [REGULAR, FILE_DISCOVERY], ids=["regular", "file-discovery"])
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
    [REGULAR, PREFILL_DECODE, FILE_DISCOVERY],
    ids=["regular", "prefill-decode", "file-discovery"],
)
def test_the_launcher_builds_the_real_router(args: RouterArgs):
    # The constructor only keeps the configuration (the file provider's mapping
    # is parsed, its file not read); nothing is bound.
    assert Router.from_args(args) is not None
