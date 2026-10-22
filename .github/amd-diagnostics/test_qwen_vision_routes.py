"""Diagnostic only: twelve independent, unretried observations; no row acceptance."""

import os
import platform

import pytest
import qwen_vision as vision


@pytest.fixture(scope="session")
def media():
    return vision.images(vision.OUT / "inputs")


@pytest.fixture(scope="class")
def backend(request):
    import PIL
    import sglang
    import torch

    assert sglang.__version__ == "0.5.21" and torch.version.hip
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"
    vision.save(
        vision.OUT / "runtime.json",
        {
            "python": platform.python_version(),
            "sglang": sglang.__version__,
            "torch": torch.__version__,
            "torch_hip_compilation_string": torch.version.hip,
            "Pillow": PIL.__version__,
            "runner_name": os.environ["RUNNER_NAME"],
            "consumer_sha": os.environ["GITHUB_SHA"],
            "consumer_run": os.environ["GITHUB_RUN_ID"],
            "declared_container_image": vision.INPUTS["image"],
            "offline": True,
        },
    )
    path = vision.snapshot()
    with vision.backend(request.param, vision.OUT / request.param, path) as endpoints:
        yield endpoints, path


@pytest.mark.parametrize("backend", ["http"], indirect=True)
@pytest.mark.parametrize("route", ["native-http", "smg-http"])
@pytest.mark.parametrize("fmt", ["jpeg", "pil-png"])
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
class TestHTTP:
    def test_dog(self, backend, route, fmt, stream, media):
        endpoints, model = backend
        mime, data = media[fmt]
        vision.request_once(
            endpoints[route],
            vision.payload(model, mime, data, stream),
            vision.OUT / f"{route}-{fmt}-{'stream' if stream else 'non-stream'}",
        )


@pytest.mark.parametrize("backend", ["grpc"], indirect=True)
@pytest.mark.parametrize("fmt", ["jpeg", "pil-png"])
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
class TestGRPC:
    def test_dog(self, backend, fmt, stream, media):
        endpoints, model = backend
        mime, data = media[fmt]
        vision.request_once(
            endpoints["smg-grpc"],
            vision.payload(model, mime, data, stream),
            vision.OUT / f"smg-grpc-{fmt}-{'stream' if stream else 'non-stream'}",
        )
