"""Unit tests for env-filter deselection reporting and the selection floor (no GPU).

Mirrors the ``_FakeItem`` harness in ``test_hooks_zmq_filter.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fixtures import hooks


class _FakeItem:
    """Minimal stand-in for a pytest ``Item`` for the collection helpers.

    Exposes only what the env filter, ``_filter_zmq_items`` and
    ``_pool_sort_key`` touch: ``nodeid``, ``callspec.params``, ``cls`` (always
    None so marker resolution falls back to ``get_closest_marker``), and a
    name->marker mapping served through ``get_closest_marker``.
    """

    def __init__(self, nodeid, params=None, markers=None):
        self.nodeid = nodeid
        self.cls = None
        self.callspec = SimpleNamespace(params=params) if params is not None else None
        self._markers = markers or {}

    def get_closest_marker(self, name):
        return self._markers.get(name)


class _FakeConfig:
    """Fake ``pytest.Config`` recording ``pytest_deselected`` calls."""

    def __init__(self):
        self.stash = pytest.Stash()
        self.deselected = []
        self.hook = SimpleNamespace(pytest_deselected=self._record)

    def _record(self, items):
        self.deselected.extend(items)


def _item(nodeid, engine=None, vendor=None, gpu=None, setup_backend=None):
    markers = {}
    if engine is not None:
        markers["engine"] = pytest.mark.engine(*engine).mark
    if vendor is not None:
        markers["vendor"] = pytest.mark.vendor(*vendor).mark
    if gpu is not None:
        markers["gpu"] = pytest.mark.gpu(gpu).mark
    params = {"setup_backend": setup_backend} if setup_backend is not None else None
    return _FakeItem(nodeid, params=params, markers=markers)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate every test from the ambient lane environment."""
    for var in (
        "E2E_ENGINE",
        "E2E_VENDOR",
        "E2E_GPU_TIER",
        "E2E_CONNECTION_MODE",
        "E2E_RUNTIME",
        "E2E_ZMQ_ENGINE_COUNT",
        "E2E_MIN_SELECTED",
    ):
        monkeypatch.delenv(var, raising=False)


def _run(config, items):
    hooks.pytest_collection_modifyitems(config, items)
    return items


def _summary(config):
    return hooks._SelectionReporter().pytest_report_collectionfinish(config)


# ---------------------------------------------------------------------------
# _filter_env_items
# ---------------------------------------------------------------------------


def test_env_filter_attributes_drops_to_first_rejecting_dimension():
    match = _item("t.py::test_a", engine=("sglang",), gpu=1)
    wrong_engine = _item("t.py::test_b", engine=("vllm",), gpu=1)
    no_marker = _item("t.py::test_c")
    wrong_tier = _item("t.py::test_d", engine=("sglang",), gpu=4)
    selected, deselected_by = hooks._filter_env_items(
        [match, wrong_engine, no_marker, wrong_tier],
        engine="sglang",
        vendor=None,
        gpu_tier="1",
    )
    assert selected == [match]
    assert deselected_by["engine"] == [wrong_engine, no_marker]
    assert deselected_by["vendor"] == []
    assert deselected_by["tier"] == [wrong_tier]


def test_env_filter_gpu_defaults_to_one_without_marker():
    unmarked = _item("t.py::test_a")
    selected, deselected_by = hooks._filter_env_items(
        [unmarked], engine=None, vendor=None, gpu_tier="1"
    )
    assert selected == [unmarked]
    assert deselected_by["tier"] == []


def test_env_filter_by_vendor():
    openai = _item("t.py::test_a", vendor=("openai",))
    xai = _item("t.py::test_b", vendor=("xai",))
    selected, deselected_by = hooks._filter_env_items(
        [openai, xai], engine=None, vendor="openai", gpu_tier=None
    )
    assert selected == [openai]
    assert deselected_by["vendor"] == [xai]


# ---------------------------------------------------------------------------
# pytest_collection_modifyitems: deselection reporting
# ---------------------------------------------------------------------------


def test_env_filtered_items_are_reported_deselected(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    match = _item("t.py::test_a", engine=("sglang",))
    wrong = _item("t.py::test_b", engine=("vllm",))
    unmarked = _item("t.py::test_c")
    config = _FakeConfig()
    items = _run(config, [match, wrong, unmarked])
    assert items == [match]
    assert config.deselected == [wrong, unmarked]


def test_no_env_filter_deselects_nothing():
    a = _item("t.py::test_a")
    b = _item("t.py::test_b")
    config = _FakeConfig()
    items = _run(config, [a, b])
    assert items == [a, b]
    assert config.deselected == []


def test_env_and_zmq_filters_both_report(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_CONNECTION_MODE", "zmq")
    grpc = _item("t.py::test_a[grpc]", engine=("sglang",), setup_backend="grpc")
    http = _item("t.py::test_a[http]", engine=("sglang",), setup_backend="http")
    wrong = _item("t.py::test_b", engine=("vllm",))
    config = _FakeConfig()
    items = _run(config, [grpc, http, wrong])
    assert items == [grpc]
    assert set(config.deselected) == {http, wrong}


# ---------------------------------------------------------------------------
# Summary line
# ---------------------------------------------------------------------------


def test_summary_line_counts_and_prefix(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_GPU_TIER", "1")
    match = _item("t.py::test_a", engine=("sglang",), gpu=1)
    wrong_engine = _item("t.py::test_b", engine=("vllm",), gpu=1)
    unmarked = _item("t.py::test_c")
    wrong_tier = _item("t.py::test_d", engine=("sglang",), gpu=4)
    config = _FakeConfig()
    _run(config, [match, wrong_engine, unmarked, wrong_tier])
    line = _summary(config)
    assert line == (
        "e2e selection: engine=sglang tier=1: selected 1 of 4 collected "
        "(2 deselected by engine, 1 by tier, 0 by zmq-dedup)"
    )


def test_summary_line_without_filters(monkeypatch):
    a = _item("t.py::test_a")
    config = _FakeConfig()
    _run(config, [a])
    line = _summary(config)
    assert line == (
        "e2e selection: engine=- tier=-: selected 1 of 1 collected "
        "(0 deselected by engine, 0 by tier, 0 by zmq-dedup)"
    )


def test_summary_line_includes_vendor_and_zmq_counts(monkeypatch):
    monkeypatch.setenv("E2E_VENDOR", "openai")
    monkeypatch.setenv("E2E_CONNECTION_MODE", "zmq")
    kept = _item("t.py::test_a[grpc]", vendor=("openai",), setup_backend="grpc")
    twin = _item("t.py::test_a[http]", vendor=("openai",), setup_backend="http")
    other_vendor = _item("t.py::test_b", vendor=("xai",))
    config = _FakeConfig()
    _run(config, [kept, twin, other_vendor])
    line = _summary(config)
    assert line == (
        "e2e selection: engine=- vendor=openai tier=-: selected 1 of 3 collected "
        "(0 deselected by engine, 1 by vendor, 0 by tier, 1 by zmq-dedup)"
    )


def test_summary_is_none_before_collection():
    config = _FakeConfig()
    assert _summary(config) is None


# ---------------------------------------------------------------------------
# E2E_MIN_SELECTED floor
# ---------------------------------------------------------------------------


def test_floor_unset_is_noop():
    hooks._enforce_selection_floor(0)  # must not raise


def test_floor_blank_is_noop(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "  ")
    hooks._enforce_selection_floor(0)  # must not raise


def test_floor_passes_at_and_above(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "2")
    hooks._enforce_selection_floor(2)
    hooks._enforce_selection_floor(3)


def test_floor_fails_below(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "5")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(1)
    msg = str(excinfo.value)
    assert "only 1 test selected" in msg  # singular for a one-test selection
    assert "floor of 5" in msg
    assert "E2E_ENGINE" in msg  # points at the env filter as the likely cause


def test_floor_message_pluralizes_above_one(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "5")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(2)
    assert "only 2 tests selected" in str(excinfo.value)


def test_floor_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("E2E_MIN_SELECTED", "lots")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(100)
    assert "not an integer" in str(excinfo.value)


def test_floor_rejects_negative(monkeypatch):
    """A negative floor can never trip, so take it as a typo, not "no floor"."""
    monkeypatch.setenv("E2E_MIN_SELECTED", "-1")
    with pytest.raises(pytest.exit.Exception) as excinfo:
        hooks._enforce_selection_floor(0)
    assert "must be non-negative" in str(excinfo.value)


def test_floor_zero_is_armed_but_unbreakable(monkeypatch):
    """Zero is a legal floor: explicitly armed, and no selection can fall under it."""
    monkeypatch.setenv("E2E_MIN_SELECTED", "0")
    hooks._enforce_selection_floor(0)


def test_floor_enforced_from_collection_hook(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_MIN_SELECTED", "2")
    match = _item("t.py::test_a", engine=("sglang",))
    wrong = _item("t.py::test_b", engine=("vllm",))
    config = _FakeConfig()
    with pytest.raises(pytest.exit.Exception):
        hooks.pytest_collection_modifyitems(config, [match, wrong])


def test_floor_satisfied_from_collection_hook(monkeypatch):
    monkeypatch.setenv("E2E_ENGINE", "sglang")
    monkeypatch.setenv("E2E_MIN_SELECTED", "1")
    match = _item("t.py::test_a", engine=("sglang",))
    wrong = _item("t.py::test_b", engine=("vllm",))
    config = _FakeConfig()
    items = _run(config, [match, wrong])
    assert items == [match]
