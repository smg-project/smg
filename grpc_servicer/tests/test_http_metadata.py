"""Engine-free checks of the installed module's plain ASGI implementation.

The source package is loaded under an isolated name, like other engine-free
servicer tests, so an unrelated installed servicer/version cannot mask it.
No vLLM, GPU, gRPC proto or pytest-asyncio import/skip is needed.
"""

import asyncio
import gzip
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "smg_grpc_servicer"
spec = importlib.util.spec_from_file_location(
    "_smg_http_metadata_cpu", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
)
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
metadata = importlib.import_module("_smg_http_metadata_cpu.http_metadata")


def scope(path="/v1/models", method="GET", headers=()):
    return {"type": "http", "path": path, "method": method, "headers": list(headers)}


def model_list(**extra):
    return json.dumps(
        {"object": "list", "data": [{"id": "m"}], **extra}, ensure_ascii=False
    ).encode()


def run(
    monkeypatch, chunks, *, headers=None, status=200, request=None, trailers=False, env=" kv-v1 "
):
    if env is None:
        monkeypatch.delenv("SMG_PAIRING_PROTOCOL", raising=False)
    else:
        monkeypatch.setenv("SMG_PAIRING_PROTOCOL", env)
    observed = []
    sent = []
    if headers is None:
        headers = [(b"content-type", b"application/json")]

    async def app(inner_scope, receive, send):
        observed.append(inner_scope)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
                "trailers": trailers,
            }
        )
        for index, chunk in enumerate(chunks):
            await send(
                {"type": "http.response.body", "body": chunk, "more_body": index < len(chunks) - 1}
            )
        if trailers:
            await send({"type": "http.response.trailers", "headers": [(b"digest", b"old")]})

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(metadata.PairingProtocolMiddleware(app)(request or scope(), receive, send))
    return observed, sent


def answer(sent):
    start, body = sent
    assert body["more_body"] is False
    lengths = [value for key, value in start["headers"] if key.lower() == b"content-length"]
    assert lengths == [str(len(body["body"])).encode()]
    return start, json.loads(body["body"])


def test_chunked_json_exposes_only_trimmed_engine_value_and_repairs_headers(monkeypatch):
    monkeypatch.setenv("UNRELATED_SECRET", "never-expose-this")
    body = model_list()
    original = scope(headers=[(b"accept-encoding", b"gzip"), (b"Accept-Encoding", b"br")])
    headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode()),
        (b"etag", b"old"),
        (b"digest", b"old"),
        (b"content-encoding", b"identity"),
    ]
    observed, sent = run(monkeypatch, [body[:7], body[7:]], headers=headers, request=original)
    start, payload = answer(sent)
    assert start["status"] == 200
    assert [value for key, value in start["headers"] if key.lower() == b"content-type"] == [
        b"application/json"
    ]
    assert payload == {"object": "list", "data": [{"id": "m"}], "pairing_protocol": "kv-v1"}
    assert observed[0]["headers"] == [(b"accept-encoding", b"identity")]
    assert original["headers"] == [(b"accept-encoding", b"gzip"), (b"Accept-Encoding", b"br")]
    assert not any(
        key.lower() in (b"etag", b"digest", b"content-encoding") for key, _ in start["headers"]
    )
    assert "never-expose-this" not in repr(sent)


@pytest.mark.parametrize("env", [None, "", "   "])
def test_blank_environment_is_exact_passthrough(monkeypatch, env):
    body = model_list()
    _, sent = run(monkeypatch, [body[:3], body[3:]], env=env)
    assert len(sent) == 3
    assert sent[1]["body"] + sent[2]["body"] == body
    assert "pairing_protocol" not in json.loads(body)


@pytest.mark.parametrize(
    "path,method", [("/v1/chat/completions", "POST"), ("/v1/models", "POST"), ("/metrics", "GET")]
)
def test_other_routes_keep_original_chunks_and_headers(monkeypatch, path, method):
    original = scope(path, method, [(b"accept-encoding", b"gzip")])
    observed, sent = run(monkeypatch, [b"data: one\n\n", b"data: [DONE]\n\n"], request=original)
    assert observed == [original]
    assert len(sent) == 3
    assert sent[1]["more_body"] is True
    assert sent[1]["body"] == b"data: one\n\n"


@pytest.mark.parametrize(
    "fault",
    [
        "gzip",
        "br",
        "html",
        "trailers",
        "bad_json",
        "bad_shape",
        "conflict",
        "wrong_type",
        "too_large",
        "nan",
    ],
)
def test_bad_metadata_is_a_valid_non200_not_silent_missing_protocol(monkeypatch, fault):
    body = model_list()
    headers = [(b"content-type", b"application/json")]
    if fault in ("gzip", "br"):
        headers.append((b"content-encoding", fault.encode()))
        body = gzip.compress(body)
    elif fault == "html":
        headers = [(b"content-type", b"text/html")]
    elif fault == "bad_json":
        body = b"not JSON"
    elif fault == "bad_shape":
        body = b'{"object":"list","data":[3]}'
    elif fault == "conflict":
        body = model_list(pairing_protocol="other-protocol")
    elif fault == "wrong_type":
        body = model_list(pairing_protocol=4)
    elif fault == "nan":
        body = model_list(extra=float("nan"))
    elif fault == "too_large":
        monkeypatch.setattr(metadata, "MAX_METADATA_BYTES", 4)
    _, sent = run(monkeypatch, [body], headers=headers, trailers=fault == "trailers")
    start, payload = answer(sent)
    assert start["status"] == 502
    assert payload["error"]["code"] == "smg_http_metadata_invalid"
    assert len(sent) == 2


def test_identical_extension_and_unicode_chunks_are_valid(monkeypatch):
    body = model_list(pairing_protocol="kv-v1", note="模型")
    split = body.index("模".encode()) + 1
    _, sent = run(monkeypatch, [body[:split], body[split:]], env="kv-v1")
    start, payload = answer(sent)
    assert start["status"] == 200
    assert payload["note"] == "模型"


def test_auth_error_is_preserved_without_exposing_engine_metadata(monkeypatch):
    _, sent = run(monkeypatch, [b"unauthorized"], status=401)
    assert sent[0]["status"] == 401
    assert sent[1]["body"] == b"unauthorized"


def test_inference_first_chunk_is_not_buffered(monkeypatch):
    monkeypatch.setenv("SMG_PAIRING_PROTOCOL", "kv-v1")

    async def check():
        sent = []
        first = asyncio.Event()
        finish = asyncio.Event()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"first", "more_body": True})
            first.set()
            await finish.wait()
            await send({"type": "http.response.body", "body": b"last"})

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(
            metadata.PairingProtocolMiddleware(app)(
                scope("/v1/chat/completions", "POST"), None, send
            )
        )
        await asyncio.wait_for(first.wait(), 1)
        assert len(sent) == 2
        finish.set()
        await task

    asyncio.run(check())


def test_cancellation_unwinds_current_metadata_request_without_a_reader_task(monkeypatch):
    monkeypatch.setenv("SMG_PAIRING_PROTOCOL", "kv-v1")

    async def check():
        entered = asyncio.Event()
        unwound = asyncio.Event()
        sent = []

        async def app(scope, receive, send):
            try:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": b"{", "more_body": True})
                entered.set()
                await asyncio.Event().wait()
            finally:
                unwound.set()

        async def send(message):
            sent.append(message)

        task = asyncio.create_task(metadata.PairingProtocolMiddleware(app)(scope(), None, send))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert unwound.is_set()
        assert sent == []  # no partial200 or fabricated successful body

    asyncio.run(check())
