"""The harness labels a registered worker whose servicer cannot advertise a
fact itself (``Gateway.label_workers``, ``PATCH /workers/{id}``), and waits
for the queued update to show on ``GET /workers``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from infra import constants
from infra import gateway as gw
from infra.constants import ENV_TRTLLM_SERVICER_IMPL


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self.text = str(payload)
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _gateway(
    monkeypatch, registered: list[tuple[str, str]], patches: list, status: int = 200, lag: int = 1
):
    """A gateway whose ``GET /workers`` lists ``registered`` (id, url) pairs; a PATCH is accepted with
    ``status`` and its labels show up ``lag`` listings later, as the queued update lands."""
    monkeypatch.setattr(gw.Gateway, "base_url", property(lambda self: "http://gw"), raising=False)
    queued: dict[str, list] = {}
    listings: list[int] = []

    def get(url, **kwargs):
        workers = []
        for wid, wurl in registered:
            worker = {"id": wid, "url": wurl, "is_healthy": True}
            if wid in queued:
                if queued[wid][0] > 0:
                    queued[wid][0] -= 1
                else:
                    worker["labels"] = queued[wid][1]
            workers.append(worker)
        listings.append(len(listings) + 1)
        return _Response(200, {"workers": workers})

    def patch(url, json, timeout):
        patches.append((url, json))
        if status in (200, 202):
            queued[url.rsplit("/", 1)[-1]] = [lag, json["labels"]]
        return _Response(status, {})

    monkeypatch.setattr(gw.httpx, "get", get)
    monkeypatch.setattr(gw.httpx, "patch", patch)
    gateway = gw.Gateway.__new__(gw.Gateway)
    return gateway, listings


def test_label_workers_patches_the_registered_worker_by_its_url_and_waits_for_the_label(
    monkeypatch,
):
    patches: list = []
    gateway, listings = _gateway(
        monkeypatch, [("w1", "grpc://127.0.0.1:5"), ("w2", "grpc://127.0.0.1:6")], patches
    )
    worker = SimpleNamespace(engine="trtllm", base_url="grpc://127.0.0.1:6")

    gateway.label_workers([worker], {"structural_tag_vocabulary": "older"})

    assert patches == [("http://gw/workers/w2", {"labels": {"structural_tag_vocabulary": "older"}})]
    # The listing before the PATCH, one without the label (the update is queued), one with it.
    assert len(listings) == 3


def test_label_workers_does_nothing_without_workers_and_refuses_an_unregistered_one(monkeypatch):
    patches: list = []
    gateway, listings = _gateway(monkeypatch, [("w1", "grpc://127.0.0.1:5")], patches)

    gateway.label_workers([], {"structural_tag_vocabulary": "older"})
    assert patches == [] and listings == []

    stranger = SimpleNamespace(engine="trtllm", base_url="grpc://127.0.0.1:9")
    with pytest.raises(RuntimeError, match="not registered"):
        gateway.label_workers([stranger], {"structural_tag_vocabulary": "older"})
    assert patches == []


def test_label_workers_reports_a_refused_patch(monkeypatch):
    patches: list = []
    gateway, _ = _gateway(monkeypatch, [("w1", "grpc://127.0.0.1:5")], patches, status=400)
    worker = SimpleNamespace(engine="trtllm", base_url="grpc://127.0.0.1:5")

    with pytest.raises(RuntimeError, match="PATCH /workers/w1 returned 400"):
        gateway.label_workers([worker], {"structural_tag_vocabulary": "older"})


def test_the_hand_set_label_goes_to_trtllm_workers_on_the_engines_own_servicer(monkeypatch):
    """The engine's own gRPC server advertises no structural-tag vocabulary
    (the lane's default, an explicit ``python``, the package's own default);
    the Rust servicer's launcher advertises it, so nothing is set by hand."""
    trt = SimpleNamespace(engine="trtllm", base_url="grpc://127.0.0.1:5")
    other = SimpleNamespace(engine="vllm", base_url="grpc://127.0.0.1:6")
    monkeypatch.delenv(ENV_TRTLLM_SERVICER_IMPL, raising=False)
    monkeypatch.setattr(constants, "_package_servicer_switch", lambda: None)
    assert gw.workers_labelled_by_hand([other, trt]) == [trt]
    assert gw.workers_labelled_by_hand([other]) == []
    assert gw.workers_labelled_by_hand(None) == []
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "python")
    assert gw.workers_labelled_by_hand([trt]) == [trt]
    monkeypatch.setenv(ENV_TRTLLM_SERVICER_IMPL, "rust")
    assert gw.workers_labelled_by_hand([other, trt]) == []
    monkeypatch.delenv(ENV_TRTLLM_SERVICER_IMPL)
    monkeypatch.setattr(
        constants,
        "_package_servicer_switch",
        lambda: lambda variable, environ=None, **_: ("rust", "default"),
    )
    assert gw.workers_labelled_by_hand([trt]) == []


def test_label_workers_raises_when_the_queued_label_never_lands(monkeypatch):
    patches: list = []
    gateway, listings = _gateway(monkeypatch, [("w1", "grpc://127.0.0.1:5")], patches, lag=10**6)
    worker = SimpleNamespace(engine="trtllm", base_url="grpc://127.0.0.1:5")

    with pytest.raises(RuntimeError, match="are not on \\['w1'\\] after 0.0s"):
        gateway.label_workers([worker], {"structural_tag_vocabulary": "older"}, timeout=0.0)
    assert len(patches) == 1 and len(listings) >= 2
