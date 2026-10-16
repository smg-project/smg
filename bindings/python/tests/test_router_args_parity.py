"""The launcher's arguments and the Rust router's constructor agree.

A router argument is written in several places by hand: the ``RouterArgs``
dataclass and its command-line flag, the constructor signature of the Rust
``Router`` and the mapping into the router configuration. ``Router.from_args``
hands every dataclass field the Rust constructor takes to it as a keyword, so
a field added on one side only fails at launch. These tests pin the two sides
against each other with the real constructor, with no engine behind it.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest
from smg.router import Router
from smg.router_args import RouterArgs
from smg.smg_rs import Router as _Router

# The constructor parameters the launcher never sets: the keyword-only
# ``discovery`` mapping is for programmatic callers; the launcher passes
# ``service_discovery`` and the selectors instead (the two are refused together).
CONSTRUCTOR_ONLY: set[str] = {"discovery"}


def constructor_parameters() -> set[str]:
    return set(inspect.signature(_Router).parameters)


def keywords_from_args(args: RouterArgs) -> dict:
    """The keywords ``from_args`` hands the real constructor, which must take them."""
    real = _Router
    with patch("smg.router._Router", side_effect=real) as spy:
        Router.from_args(args)
    return spy.call_args.kwargs


def test_every_keyword_the_launcher_passes_is_a_constructor_parameter():
    passed = set(keywords_from_args(RouterArgs(worker_urls=["http://worker:8000"])))
    unknown = passed - constructor_parameters()
    assert not unknown, (
        f"the launcher passes keywords the Rust constructor does not take: {sorted(unknown)}"
    )


def test_every_constructor_parameter_is_set_by_the_launcher():
    passed = set(keywords_from_args(RouterArgs(worker_urls=["http://worker:8000"])))
    unset = constructor_parameters() - passed - CONSTRUCTOR_ONLY
    assert not unset, (
        f"the Rust constructor takes parameters the launcher never sets: {sorted(unset)}"
    )


@pytest.mark.parametrize(
    "args",
    [
        RouterArgs(worker_urls=["http://worker:8000"]),
        RouterArgs(
            worker_urls=[],
            pd_disaggregation=True,
            prefill_urls=[("http://p:8000", None)],
            decode_urls=["http://d:8000"],
        ),
    ],
    ids=["regular", "prefill-decode"],
)
def test_the_launcher_builds_the_real_router(args: RouterArgs):
    # The constructor only keeps the configuration; nothing is bound.
    assert Router.from_args(args) is not None
