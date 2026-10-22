"""One-attempt JPEG/PNG comparison through existing SMG worker/gateway infra."""

import base64
import hashlib
import io
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
INPUTS = json.loads(Path(__file__).with_name("qwen-vision-inputs.json").read_text())
OUT = ROOT / "diagnostic-results"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def images(out):
    jpeg = (ROOT / "e2e_test/fixtures/images/dog.jpg").read_bytes()
    assert hashlib.sha256(jpeg).hexdigest() == INPUTS["dog_jpeg_sha256"]
    with Image.open(io.BytesIO(jpeg)) as image:
        rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    png = buffer.getvalue()
    with Image.open(io.BytesIO(png)) as image:
        assert image.convert("RGB").tobytes() == rgb.tobytes()
    out.mkdir(parents=True, exist_ok=True)
    (out / "dog.jpg").write_bytes(jpeg)
    (out / "dog-pil-rgb.png").write_bytes(png)
    save(
        out / "inputs.json",
        {
            "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
            "png_sha256": hashlib.sha256(png).hexdigest(),
            "PIL_rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "dimensions": list(rgb.size),
            "png_same_PIL_decoded_rgb_pixels": True,
            "png_is_original_jpeg_acceptance": False,
        },
    )
    return {"jpeg": ("image/jpeg", jpeg), "pil-png": ("image/png", png)}


def snapshot(cache=Path("/models/hub")):
    path = cache / "models--Qwen--Qwen3-VL-8B-Instruct/snapshots" / INPUTS["model_snapshot"]
    config = path / "config.json"
    if not config.is_file():
        raise ValueError("frozen model snapshot is absent; no download permitted")
    index = path / "model.safetensors.index.json"
    names = (
        set(json.loads(index.read_text())["weight_map"].values())
        if index.exists()
        else {"model.safetensors"}
    )
    if not names or any(
        Path(n).name != n or not (path / n).is_file() or (path / n).stat().st_size == 0
        for n in names
    ):
        raise ValueError("frozen snapshot has missing/invalid weight shards")
    save(
        OUT / "model.json",
        {
            "model_id": INPUTS["model"],
            "snapshot": INPUTS["model_snapshot"],
            "model_path": str(path),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "weight_files": {n: (path / n).stat().st_size for n in sorted(names)},
            "full_weight_hashes_verified": False,
            "offline": True,
        },
    )
    return path


@contextmanager
def backend(mode, out, model_path):
    from infra import ConnectionMode, Gateway, WorkerPool
    from infra.model_specs import MODEL_SPECS

    pool, gateway = WorkerPool(), Gateway()
    spec = MODEL_SPECS[INPUTS["model"]]
    old_path, spec["model"] = spec["model"], str(model_path)
    workers = []
    cleanup = {"mode": mode, "gateway_shutdown": False, "pool_cleanup": False}
    try:
        workers = pool.acquire(
            model_id=INPUTS["model"],
            engine="sglang",
            mode=ConnectionMode(mode),
            count=1,
            timeout=300,
            log_dir=str(out),
        )
        assert len(workers) == 1 and workers[0].mode == ConnectionMode(mode)
        gateway.start(
            worker_urls=[workers[0].base_url],
            model_path=str(model_path),
            timeout=120,
            log_dir=str(out),
            show_output=False,
            extra_args=["--disable-retries"],
        )
        endpoints = {"smg-" + mode: gateway.base_url}
        if mode == "http":
            endpoints["native-http"] = workers[0].base_url
        save(
            out / "backend.json",
            {
                "mode": mode,
                "endpoints": endpoints,
                "worker_url": workers[0].base_url,
                "worker_pid": workers[0].process.pid,
                "gpu_ids": workers[0].gpu_ids,
                "worker_log": workers[0].log_path,
                "gateway_log": gateway.log_path,
                "model_path": str(model_path),
                "gateway_retries_disabled": True,
            },
        )
        yield endpoints
    finally:
        try:
            try:
                gateway.shutdown()
                cleanup["gateway_shutdown"] = True
            finally:
                pool.cleanup()
                cleanup["pool_cleanup"] = True
                cleanup["worker_alive_after_cleanup"] = [w.is_alive() for w in workers]
        finally:
            spec["model"] = old_path
            save(out / "cleanup.json", cleanup)


def payload(model_path, mime, data, stream):
    image = "data:" + mime + ";base64," + base64.b64encode(data).decode()
    return {
        "model": str(model_path),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What animal is in this image?"},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 100,
        "stream": stream,
    }


def parse(raw, streaming):
    if not streaming:
        body = json.loads(raw)
        return body["choices"][0]["message"]["content"], [body]
    frames, done = [], False
    for line in raw.decode().splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if done:
            raise ValueError("data after stream completion")
        if data == "[DONE]":
            done = True
            continue
        frames.append(json.loads(data))
    if not done or not frames:
        raise ValueError("incomplete/empty SSE response")
    text = "".join(c["delta"].get("content") or "" for f in frames for c in f.get("choices", []))
    return text, frames


def assert_dog(text):
    assert any(k in text.lower() for k in ["dog", "puppy", "labrador"]), (
        f"Expected dog-related content, got: {text}"
    )


def request_once(endpoint, request, out):
    raw = bytearray()
    record = {
        "endpoint": endpoint + "/v1/chat/completions",
        "request": request,
        "attempts": 1,
        "automatic_retries": 0,
        "started_at_unix": time.time(),
    }
    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(retries=0), timeout=60, follow_redirects=False
        ) as client:
            with client.stream("POST", record["endpoint"], json=request) as response:
                record["status_code"] = response.status_code
                record["response_headers"] = dict(response.headers)
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > 4 * 1024 * 1024:
                        raise ValueError("response exceeds diagnostic limit")
                response.raise_for_status()
        text, frames = parse(bytes(raw), request["stream"])
        record.update(
            text=text,
            json_frames=frames,
            usage=[f["usage"] for f in frames if f.get("usage") is not None],
        )
        assert_dog(text)
        record["dog_assertion_passed"] = True
    except BaseException as exc:
        record.update(error_type=type(exc).__name__, error=str(exc), dog_assertion_passed=False)
        raise
    finally:
        record["finished_at_unix"] = time.time()
        record["consumer_sha"] = os.environ.get("GITHUB_SHA")
        record["model_snapshot"] = INPUTS["model_snapshot"]
        record["container_image"] = INPUTS["image"]
        record["original_jpeg"] = request["messages"][0]["content"][1]["image_url"][
            "url"
        ].startswith("data:image/jpeg;base64,")
        out.mkdir(parents=True, exist_ok=True)
        (out / "response.raw").write_bytes(raw)
        save(out / "result.json", record)
