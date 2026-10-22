"""Offline controls only: no real server, artifact API, model, or GPU work."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import qwen_vision as vision


def test_png_has_identical_PIL_decoded_pixels(tmp_path):
    media = vision.images(tmp_path)
    info = json.loads((tmp_path / "inputs.json").read_text())
    assert media["jpeg"][1] != media["pil-png"][1]
    assert info["png_same_PIL_decoded_rgb_pixels"] and not info["png_is_original_jpeg_acceptance"]


def test_missing_snapshot_fails_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "OUT", tmp_path / "out")
    with pytest.raises(ValueError, match="absent"):
        vision.snapshot(tmp_path)


def test_dog_assertion_matches_original_ast():
    original = ast.parse((vision.ROOT / "e2e_test/chat_completions/test_multimodal.py").read_text())
    method = next(
        n
        for n in ast.walk(original)
        if isinstance(n, ast.FunctionDef) and n.name == "test_single_image_base64"
    )
    assertion = next(n for n in method.body if isinstance(n, ast.Assert))
    helper = ast.parse(Path(vision.__file__).read_text())
    helper = next(
        n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == "assert_dog"
    )
    assert ast.dump(assertion, include_attributes=False) == ast.dump(
        helper.body[0], include_attributes=False
    )
    with pytest.raises(AssertionError, match="Expected dog-related"):
        vision.assert_dog("Based on the animal The animal The animal")
    vision.assert_dog("A black Labrador puppy")


@pytest.mark.parametrize("raw", [b'data: {"choices":[]}\n\n', b"data: [DONE]\n\ndata: [DONE]\n\n"])
def test_incomplete_or_repeated_stream_completion_rejected(raw):
    with pytest.raises(ValueError):
        vision.parse(raw, True)


@pytest.mark.parametrize("kind", ["http-error", "wrong-answer", "success", "read-error"])
def test_request_is_once_and_always_records_raw(kind, tmp_path, monkeypatch):
    calls = []
    client_type = httpx.Client

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("lost stream")

    def handle(request):
        calls.append(request)
        if kind == "http-error":
            return httpx.Response(503, json={"error": "unavailable"})
        if kind == "read-error":
            return httpx.Response(200, stream=Broken())
        answer = "dog" if kind == "success" else "The animal The animal"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": answer}}], "usage": {"total_tokens": 7}}
        )

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        kwargs["transport"].close()
        return client_type(transport=httpx.MockTransport(handle))

    monkeypatch.setattr(vision.httpx, "Client", client)
    request = vision.payload("frozen-model", "image/jpeg", b"image", False)
    if kind == "success":
        vision.request_once("http://127.0.0.1:1", request, tmp_path)
    else:
        with pytest.raises((httpx.HTTPStatusError, httpx.ReadError, AssertionError)):
            vision.request_once("http://127.0.0.1:1", request, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(calls) == result["attempts"] == 1 and result["automatic_retries"] == 0
    assert (tmp_path / "response.raw").read_bytes()
    assert result["dog_assertion_passed"] == (kind == "success")
    if kind == "read-error":
        assert (tmp_path / "response.raw").read_bytes() == b"partial"


def test_real_infra_entrypoints_cleanup_on_gateway_failure(tmp_path, monkeypatch):
    import infra
    from infra.model_specs import MODEL_SPECS

    calls = []
    old = MODEL_SPECS[vision.INPUTS["model"]]["model"]
    worker = SimpleNamespace(
        mode=infra.ConnectionMode.HTTP, base_url="http://127.0.0.1:1", is_alive=lambda: False
    )

    class Pool:
        def acquire(self, *, model_id, engine, mode, count, timeout, log_dir):
            assert (
                model_id == vision.INPUTS["model"]
                and engine == "sglang"
                and mode == infra.ConnectionMode.HTTP
            )
            assert count == 1 and timeout == 300 and log_dir == str(tmp_path)
            calls.append("acquire")
            return [worker]

        def cleanup(self):
            calls.append("pool-cleanup")

    class Gateway:
        def start(self, *, worker_urls, model_path, timeout, log_dir, show_output, extra_args):
            assert worker_urls == [worker.base_url] and model_path == "/frozen"
            assert timeout == 120 and log_dir == str(tmp_path) and show_output is False
            assert extra_args == ["--disable-retries"]
            calls.append("gateway-start")
            raise RuntimeError("startup failed")

        def shutdown(self):
            calls.append("gateway-shutdown")

    monkeypatch.setattr(infra, "WorkerPool", Pool)
    monkeypatch.setattr(infra, "Gateway", Gateway)
    with (
        pytest.raises(RuntimeError, match="startup failed"),
        vision.backend("http", tmp_path, Path("/frozen")),
    ):
        pytest.fail("must not yield after startup failure")
    assert calls == ["acquire", "gateway-start", "gateway-shutdown", "pool-cleanup"]
    assert MODEL_SPECS[vision.INPUTS["model"]]["model"] == old
    assert json.loads((tmp_path / "cleanup.json").read_text())["pool_cleanup"]
