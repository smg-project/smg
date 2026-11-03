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


# ---------------------------------------------------------------------------
# _filter_sglang_rust_items
# ---------------------------------------------------------------------------


def test_sglang_rust_lane_keeps_prefill_decode_topologies():
    """The Rust servicer carries the prefill/decode bootstrap fields: the PD
    topologies run on its lane like the local wires."""
    pd_param = _item("t.py::test_x[pd_grpc]", setup_backend="pd_grpc")
    pd_http = _item("t.py::test_x[pd_http]", setup_backend="pd_http")
    # The topology suites carry the counts in a tuple behind the mode.
    pd_topology = _item("t.py::test_x[1p2d]", setup_backend=("pd_grpc", (1, 2)))
    pd_asymmetric = _item("t.py::test_x[1p2d-ptp2-dtp1]", setup_backend=("pd_grpc", (1, 2, 2, 1)))
    pd_workers = _item(
        "t.py::test_x[grpc]",
        setup_backend="grpc",
        workers=_workers_marker(prefill=1, decode=1),
    )
    plain = _item("t.py::test_x[grpc]", setup_backend="grpc")
    items = [pd_param, pd_http, pd_topology, pd_asymmetric, pd_workers, plain]
    kept, deselected = hooks._filter_sglang_rust_items(items)
    assert kept == items
    assert deselected == []


@pytest.mark.parametrize("param", ["epd_grpc", ("epd_grpc", (1, 1, 1)), ()])
def test_sglang_rust_lane_drops_epd_and_foreign_wires(param):
    item = _item("t.py::test_x[p]", setup_backend=param)
    encode_workers = _item(
        "t.py::test_x[grpc]",
        setup_backend="grpc",
        workers=_workers_marker(encode=1, prefill=1, decode=1),
    )
    kept, deselected = hooks._filter_sglang_rust_items([item, encode_workers])
    assert kept == []
    assert deselected == [item, encode_workers]


def test_sglang_rust_lane_keeps_multimodal_cases():
    """The Rust servicer carries the preprocessed media to the scheduler: the
    multimodal cases run on its lane over the servicer and over SGLang's own
    server alike."""
    grpc = _item("chat_completions/test_multimodal.py::test_x[grpc]", setup_backend="grpc")
    http = _item("chat_completions/test_multimodal.py::test_x[http]", setup_backend="http")
    kept, deselected = hooks._filter_sglang_rust_items([grpc, http])
    assert kept == [grpc, http]
    assert deselected == []
