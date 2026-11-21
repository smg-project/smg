"""Operator-label PD compatibility oracle through the real harness API adapter."""

import ast
import json
import logging
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    function = next(
        node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name
    )
    function.decorator_list = []
    # Annotations refer to runtime-only infrastructure; they have no role in
    # the code path tested here.
    for arg in function.args.args:
        arg.annotation = None
    function.returns = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "defect", [None, "wrong_error_code", "mismatch_allowed", "empty_match", "stale_keys"]
)
def test_real_label_adapter_and_pairing_oracle(defect):
    workers = [
        SimpleNamespace(base_url="http://prefill:1", id="p", labels={}),
        SimpleNamespace(base_url="http://decode:2", id="d", labels={}),
    ]
    patches, queries, stopped, placements = [], [], [], []

    def list_workers(strict=True):
        return [
            SimpleNamespace(
                url=w.base_url,
                id=w.id,
                labels=w.labels,
                metadata={"pd_pairing": w.labels.get("pairing_protocol", "default")},
            )
            for w in workers
        ]

    def patch(url, *, json, timeout):
        worker = next(worker for worker in workers if url.endswith("/" + worker.id))
        assert json.keys() == {"labels"} and json["labels"].keys() == {"pairing_protocol"}
        worker.labels.update(json["labels"])
        patches.append((worker.id, dict(json["labels"])))
        return SimpleNamespace(status_code=202, text="queued")

    def get(url, *, timeout):
        assert url.endswith("/workers")
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "workers": [{"id": worker.id, "labels": worker.labels} for worker in workers],
            },
        )

    gateway = SimpleNamespace(
        base_url="http://gateway:3",
        prefill_workers=workers[:1],
        decode_workers=workers[1:],
        list_workers=list_workers,
        shutdown=lambda: stopped.append("gateway"),
    )
    adapter = load_function(
        ROOT / "infra/gateway.py",
        "label_workers",
        {"httpx": SimpleNamespace(patch=patch, get=get), "time": time},
        class_name="Gateway",
    )
    gateway.label_workers = MethodType(adapter, gateway)
    namespace = {"time": time}
    path = ROOT / "router/test_pd_topologies.py"
    load_function(path, "_pairing_keys", namespace)
    wait = load_function(path, "_wait_for_pairing_protocols", namespace)

    def wait_keys(gateway, expected, *, timeout):
        if defect == "stale_keys":
            namespace["_pairing_keys"] = lambda gateway: {"stale": "metadata"}
            return wait(gateway, expected, timeout=0)
        return wait(gateway, expected, timeout=timeout)

    def raw_chat(gateway, model, prompt, *, timeout):
        assert workers[0].labels["pairing_protocol"] != workers[1].labels["pairing_protocol"]
        queries.append("mismatch")
        return SimpleNamespace(
            status_code=200 if defect == "mismatch_allowed" else 503,
            text=json.dumps(
                {
                    "error": {
                        "code": "wrong" if defect == "wrong_error_code" else "no_compatible_pd_pair"
                    }
                }
            ),
        )

    def ask(client, model, prompt, *, max_tokens):
        if not patches:
            queries.append("baseline")
            return {"content": "10837"}
        assert len(patches) == 3 and workers[0].labels == workers[1].labels
        queries.append("match")
        placements.append((workers[0].base_url, workers[1].base_url))
        return {"content": "" if defect == "empty_match" else "59281"}

    namespace.update(
        _wait_for_pairing_protocols=wait_keys,
        _ask=ask,
        _raw_chat=raw_chat,
        _answer_text=lambda message: message["content"],
        _error_code=lambda response: json.loads(response.text)["error"]["code"],
        _wait_for_healthy_workers=lambda *args, **kwargs: None,
        _pairs_logged=lambda: list(placements),
        _wait_for_pairs=lambda count: list(placements),
        stop_workers=lambda owned: stopped.extend(owned),
        logger=logging.getLogger(__name__),
    )
    test = load_function(
        path,
        "test_operator_protocol_mismatch_refuses_and_matching_pair_serves",
        namespace,
        class_name="TestHTTPMoRILabelPairingProtocol",
    )
    if defect:
        with pytest.raises(AssertionError):
            test(None, ("pd_http", "model", None, gateway))
    else:
        test(None, ("pd_http", "model", None, gateway))
        assert queries == ["baseline", "mismatch", "match"]
        assert patches == [
            ("p", {"pairing_protocol": "http-mori-v1"}),
            ("d", {"pairing_protocol": "http-mori-v2"}),
            ("d", {"pairing_protocol": "http-mori-v1"}),
        ]
    assert stopped == ["gateway", *workers]
