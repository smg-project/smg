"""Unit tests for the ZMQ-lane collection filter (no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fixtures import hooks


class _FakeItem:
    """Minimal stand-in for a pytest ``Item`` for the collection helpers.

    Exposes only what ``_filter_zmq_items`` / ``_zmq_dedup_key`` touch:
    ``nodeid``, ``callspec.params``, ``cls`` (always None so marker resolution
    falls back to ``get_closest_marker``), and a ``workers`` marker.
    """

    def __init__(self, nodeid, params=None, workers=None):
        self.nodeid = nodeid
        self.cls = None
        self.callspec = SimpleNamespace(params=params) if params is not None else None
        self._workers = workers

    def get_closest_marker(self, name):
        if name == "workers":
            return self._workers
        return None


def _item(nodeid, setup_backend=None, extra_params=None, workers=None):
    params = None
    if setup_backend is not None or extra_params is not None:
        params = {}
        if setup_backend is not None:
            params["setup_backend"] = setup_backend
        if extra_params:
            params.update(extra_params)
    return _FakeItem(nodeid, params=params, workers=workers)


def _workers_marker(**kwargs):
    return pytest.mark.workers(**kwargs).mark


# ---------------------------------------------------------------------------
# _zmq_dedup_key
# ---------------------------------------------------------------------------


def test_dedup_key_ignores_setup_backend_value():
    grpc = _item("t.py::test_x[grpc]", setup_backend="grpc")
    http = _item("t.py::test_x[http]", setup_backend="http")
    assert hooks._zmq_dedup_key(grpc) == hooks._zmq_dedup_key(http)


def test_dedup_key_keeps_other_params_apart():
    a = _item("t.py::test_x[grpc-a]", setup_backend="grpc", extra_params={"api_client": "a"})
    b = _item("t.py::test_x[grpc-b]", setup_backend="grpc", extra_params={"api_client": "b"})
    assert hooks._zmq_dedup_key(a) != hooks._zmq_dedup_key(b)


def test_dedup_key_splits_nodeid_at_bracket():
    key, _others = hooks._zmq_dedup_key(_item("t.py::test_x[grpc]", setup_backend="grpc"))
    assert key == "t.py::test_x"


# ---------------------------------------------------------------------------
# _filter_zmq_items
# ---------------------------------------------------------------------------


def test_grpc_http_twins_collapse_to_grpc():
    grpc = _item("t.py::test_x[grpc]", setup_backend="grpc")
    http = _item("t.py::test_x[http]", setup_backend="http")
    kept, deselected = hooks._filter_zmq_items([grpc, http])
    assert kept == [grpc]
    assert deselected == [http]


def test_http_only_case_is_retained():
    http = _item("t.py::test_x[http]", setup_backend="http")
    kept, deselected = hooks._filter_zmq_items([http])
    assert kept == [http]
    assert deselected == []


def test_distinct_other_params_keep_both_wires():
    # A grpc/http pair that differs on another param is NOT a twin.
    grpc = _item("t.py::test_x[grpc-a]", setup_backend="grpc", extra_params={"api_client": "a"})
    http = _item("t.py::test_x[http-b]", setup_backend="http", extra_params={"api_client": "b"})
    kept, deselected = hooks._filter_zmq_items([grpc, http])
    assert kept == [grpc, http]
    assert deselected == []


@pytest.mark.parametrize("param", ["pd_grpc", "epd_grpc", ("epd_grpc", (1, 1, 1)), ()])
def test_non_local_wire_families_are_deselected(param):
    item = _item("t.py::test_x[p]", setup_backend=param)
    kept, deselected = hooks._filter_zmq_items([item])
    assert kept == []
    assert deselected == [item]


def test_multi_worker_case_is_deselected():
    item = _item("t.py::test_x[grpc]", setup_backend="grpc", workers=_workers_marker(count=2))
    kept, deselected = hooks._filter_zmq_items([item])
    assert kept == []
    assert deselected == [item]


def test_pd_worker_topology_is_deselected():
    item = _item(
        "t.py::test_x[grpc]",
        setup_backend="grpc",
        workers=_workers_marker(prefill=1, decode=1),
    )
    kept, deselected = hooks._filter_zmq_items([item])
    assert kept == []
    assert deselected == [item]


def test_items_without_setup_backend_are_untouched():
    item = _item("t.py::test_plain")  # no callspec / params
    kept, deselected = hooks._filter_zmq_items([item])
    assert kept == [item]
    assert deselected == []
