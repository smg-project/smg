"""Marker helper utilities for E2E tests.

This module provides helper functions for extracting values from pytest markers.
"""

from __future__ import annotations

from typing import Any

import pytest


def resolve_class_marker(
    item: pytest.Item,
    marker_name: str,
) -> pytest.Mark | None:
    """Resolve a marker with correct MRO precedence for inherited classes.

    Walks the class MRO (child-first) so that a child class marker overrides
    a parent class marker.  Falls back to ``get_closest_marker`` for
    non-class (module/function-level) tests.

    Works on pytest ``Item`` nodes (not ``FixtureRequest``).
    """
    if hasattr(item, "cls") and item.cls is not None:
        for cls in item.cls.__mro__:
            if hasattr(cls, "pytestmark"):
                markers = cls.pytestmark if isinstance(cls.pytestmark, list) else [cls.pytestmark]
                for marker in markers:
                    if marker.name == marker_name:
                        return marker
    return item.get_closest_marker(marker_name)


def get_marker_value(
    request: pytest.FixtureRequest,
    marker_name: str,
    arg_index: int = 0,
    default: Any = None,
) -> Any:
    """Get a value from a pytest marker.

    For class-based tests, walks the MRO (child-first) so that a child class
    marker overrides a parent class marker.

    Args:
        request: The pytest fixture request.
        marker_name: Name of the marker to look for.
        arg_index: Index of positional argument to extract.
        default: Default value if marker not found.

    Returns:
        The marker argument value or default.
    """
    marker = resolve_class_marker(request.node, marker_name)
    if marker is None:
        return default
    if marker.args and len(marker.args) > arg_index:
        return marker.args[arg_index]
    return default


def get_marker_kwargs(
    request: pytest.FixtureRequest,
    marker_name: str,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get keyword arguments from a pytest marker.

    For class-based tests, walks the MRO (child-first) so that a child class
    marker overrides a parent class marker.

    Args:
        request: The pytest fixture request.
        marker_name: Name of the marker to look for.
        defaults: Default values if marker not found or missing kwargs.

    Returns:
        Dict of keyword arguments merged with defaults.
    """
    result = dict(defaults) if defaults else {}
    marker = resolve_class_marker(request.node, marker_name)
    if marker is not None:
        result.update(marker.kwargs)
    return result


def model_id_for_engine(marker: pytest.Mark | None, engine: str | None, default: Any = None) -> Any:
    """Resolve the model id a ``@pytest.mark.model`` marker names for ``engine``.

    ``@pytest.mark.model("meta-llama/Llama-3.1-8B-Instruct", tokenspeed="Qwen/Qwen3.5-9B")``
    keeps one class portable across engines whose supported model sets differ:
    the positional id is the default and a keyword named after an engine
    overrides it for that engine only.
    """
    if marker is None:
        return default
    if engine and engine in marker.kwargs:
        return marker.kwargs[engine]
    if marker.args:
        return marker.args[0]
    return default
