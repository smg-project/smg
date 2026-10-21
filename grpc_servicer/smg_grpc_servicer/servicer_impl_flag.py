"""The ``--servicer-impl`` flag the engine launchers register: its name, its
choices and the one registration the three plugins share.

Stdlib only on purpose: the plugins are loaded in every engine process and
import nothing heavier than the package init; the lifecycle module that acts
on the choice is imported by the launchers, later and lazily.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

IMPLS = ("python", "rust")
SERVICER_IMPL_FLAG = "--servicer-impl"


def add_servicer_impl_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str,
    action: type[argparse.Action] | None = None,
    applies: Callable[[argparse.ArgumentParser], bool] | None = None,
    on_add: Callable[[], None] | None = None,
) -> bool:
    """Add ``--servicer-impl`` (the choices in :data:`IMPLS`, no default) to a
    launcher's parser that lacks it; returns whether it was added. ``applies``
    refuses a parser that does not serve gRPC; ``on_add`` runs before the
    argument goes in (a launcher that remembers an earlier parse drops it);
    ``action`` replaces argparse's store action when the launcher needs its
    own. Each engine keeps its help text, which names its server class and
    the flag the choice rides under."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if SERVICER_IMPL_FLAG in actions or (applies is not None and not applies(parser)):
        return False
    if on_add is not None:
        on_add()
    kwargs: dict[str, Any] = {
        "dest": "servicer_impl",
        "choices": list(IMPLS),
        "default": None,
        "help": help,
    }
    if action is not None:
        kwargs["action"] = action
    parser.add_argument(SERVICER_IMPL_FLAG, **kwargs)
    return True
