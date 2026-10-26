"""One bounded operation, launched by run_paired_ab.py in its own process group.

SOURCE ONLY until root admits a new window. The parent, not a socket timeout,
enforces the absolute wall deadline. Never follow redirects or inherited proxies.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import urllib.error
import urllib.request

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
PUG_URL = "https://raw.githubusercontent.com/smg-project/smg/main/e2e_test/fixtures/images/pug.jpg"
IMAGES = {"dog.jpg": "e1cd91db28149f21f3c410ffa074d0fb8bc8950740ba140c7eaae130f0493464",
          "pug.jpg": "9601bd492fd14ccea9adfe401135794adbf682f3354b3cdfed099fc20d6b3510"}
BODY_CAP = 4 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused")


def payload(fixture_dir, model_value, case):
    raw = {name: (fixture_dir / name).read_bytes() for name in IMAGES}
    if any(digest(raw[name]) != expected for name, expected in IMAGES.items()):
        raise ValueError("fixture bytes changed")
    image = lambda value: {"type": "image_url", "image_url": {"url": value}}
    def media(value):
        kind, name = value.split(":", 1)
        if kind == "base64":
            return "data:image/jpeg;base64," + base64.b64encode(raw[name]).decode()
        if kind != "URL" or name not in IMAGES:
            raise ValueError("unknown original image descriptor")
        return "https://raw.githubusercontent.com/smg-project/smg/main/e2e_test/fixtures/images/" + name
    value = {"model": model_value, "messages": [{"role": "user", "content": [
        {"type": "text", "text": case["user_text"]},
        *[image(media(item)) for item in case["images_in_order"]]]}],
        "temperature": case["temperature"], "max_tokens": case["max_tokens"]}
    # The two original nonstream methods omit this argument; single-image
    # parametrized methods explicitly pass False or True. Do not add stream_options.
    if case["order"] >= 3:
        value["stream"] = case["stream"]
    return json.dumps(value, separators=(",", ":")).encode()


def oracles(value, case, stream_text=None, nonempty_chunks=None):
    text = stream_text if case["stream"] else value["choices"][0]["message"]["content"]
    lower = text.lower() if isinstance(text, str) else ""
    result = {"nonempty": isinstance(text, str) and bool(text)}
    if case["order"] != 2:
        result["dog"] = any(x in lower for x in ["dog", "puppy", "labrador"])
    if case["order"] == 1:
        result["pug"] = any(x in lower for x in ["pug", "blanket", "wrapped"])
        result["duplicate"] = any(x in lower for x in ["same", "identical", "duplicate", "copy", "copies", "repeated", "twice", "similar", "both"])
    if case["order"] == 1 or (case["order"] >= 3 and not case["stream"]):
        result["prompt_usage_positive"] = value["usage"]["prompt_tokens"] > 0
        result["completion_usage_positive"] = value["usage"]["completion_tokens"] > 0
    if case["stream"]:
        result["multiple_nonempty_content_chunks"] = nonempty_chunks > 1
    return result


def read_sse(response, out, record):
    """Persist each received bounded raw line before decoding or testing it."""
    size, event_lines, chunks, text, ids = 0, [], [], [], []
    record["SSE_done"] = False
    with (out / "response.body").open("xb") as body, (out / "chunks.jsonl").open("x") as events:
        def accept_event():
            if not event_lines:
                return
            data = b"\n".join(event_lines).decode("utf-8")
            event_lines.clear()
            if data == "[DONE]":
                record["SSE_done"] = True
                return
            value = json.loads(data)
            row = {"sequence": len(chunks), "received_monotonic": time.monotonic(), "value": value}
            events.write(json.dumps(row, ensure_ascii=False) + "\n")
            events.flush()
            chunks.append(value)
            if len(chunks) > 1024:
                raise ValueError("SSE event cap")
            if value.get("id") is not None:
                ids.append(value["id"])
            for choice in value.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if choice.get("index", 0) == 0 and content:
                    if not isinstance(content, str):
                        raise ValueError("nontext content delta")
                    text.append(content)
        while True:
            line = response.readline(65537)
            if not line:
                break
            if size + len(line) > BODY_CAP:
                body.write(line[:max(0, BODY_CAP - size)])
                body.flush()
                record["body_exceeded_cap"] = True
                raise ValueError("response body cap")
            body.write(line)
            body.flush()
            size += len(line)
            if len(line) > 65536:
                raise ValueError("SSE line cap")
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                accept_event()
            elif stripped.startswith(b"data:"):
                event_lines.append(stripped[5:].lstrip(b" "))
        accept_event()
    record.update(SSE_events=len(chunks), response_ids=ids,
                  response_id=ids[0] if ids else None,
                  finish_reasons=[c.get("finish_reason") for v in chunks for c in v.get("choices", []) if c.get("finish_reason") is not None],
                  usage=[v["usage"] for v in chunks if v.get("usage") is not None],
                  text="".join(text), nonempty_content_chunks=len(text))
    if not record["SSE_done"] or not chunks:
        raise ValueError("incomplete SSE response; actual bytes retained")
    return "".join(text), len(text)


def http_once(url, data, out, request_id, parse_json=True, case=None):
    out.mkdir(mode=0o700)
    record = {"started_at_utc": utc(), "started_monotonic": time.monotonic(),
              "url": url, "request_id_header": request_id, "passed": False}
    if data is not None:
        (out / "request.json").write_bytes(data)
        request_value = json.loads(data)
        record.update(request_sha256=digest(data), case=case,
                      actual_request_model=request_value["model"],
                      request_semantics_sha256=digest(json.dumps({key: value for key, value in request_value.items()
                          if key != "model"}, separators=(",", ":")).encode()))
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        headers = {"Content-Type": "application/json", "X-Request-ID": request_id}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            response = opener.open(req, timeout=90 if data is not None else 10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            record["status"] = response.code
            record["response_headers"] = {k: v[:1024] for k, v in response.headers.items()
                                           if k.lower() in {"content-type", "content-length", "etag", "last-modified", "x-request-id", "request-id", "x-correlation-id"}}
            if case and case["stream"] and 200 <= response.code < 300:
                text, chunk_count = read_sse(response, out, record)
                body = (out / "response.body").read_bytes()
                record["oracles"] = oracles({}, case, text, chunk_count)
            else:
                record["body_exceeded_cap"] = False
                received = 0
                # read1 performs at most one underlying read: retain every available
                # prefix before a later socket timeout or the parent's deadline kill.
                with (out / "response.body").open("xb") as saved:
                    while True:
                        chunk = response.read1(min(65536, BODY_CAP + 1 - received))
                        if not chunk:
                            break
                        remaining = BODY_CAP - received
                        saved.write(chunk[:remaining])
                        saved.flush()
                        received += len(chunk)
                        if received > BODY_CAP:
                            record["body_exceeded_cap"] = True
                            break
                body = (out / "response.body").read_bytes()
        record.update(response_sha256=digest(body), response_bytes=len(body))
        if record.get("body_exceeded_cap"):
            raise ValueError("response body cap")
        if not 200 <= record["status"] < 300:
            raise ValueError("non2xx response retained")
        name = url.rsplit("/", 1)[-1]
        if url.startswith("https://raw.githubusercontent.com/smg-project/smg/main/e2e_test/fixtures/images/"):
            if name not in IMAGES or digest(body) != IMAGES[name]:
                raise ValueError("original URL bytes differ from checked-in fixture")
            record["URL_probe_scope"] = "real original URL byte fetch, not native Rust downloader internals"
        elif parse_json and not (case and case["stream"]):
            value = json.loads(body)
            record["value"] = value if data is None else None
            record["response_id"] = value.get("id")
            if case:
                record.update(text=value["choices"][0]["message"]["content"],
                              finish_reasons=[c.get("finish_reason") for c in value.get("choices", [])],
                              usage=value.get("usage"), oracles=oracles(value, case))
        if case and not all(record["oracles"].values()):
            raise ValueError("original request oracle failed")
        record["passed"] = True
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error)[:1024],
                      traceback=traceback.format_exc()[:65536])
    finally:
        body_path = out / "response.body"
        if body_path.exists():
            raw = body_path.read_bytes()
            record.update(response_sha256=digest(raw), response_bytes=len(raw))
        record.update(finished_at_utc=utc(), finished_monotonic=time.monotonic())
        (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def native_URL(out, expected_source, url):
    """Real saved4420 helper, raw bytes only: do not decode or read GPU values."""
    import sglang.srt.utils.common as common
    out.mkdir(mode=0o700)
    record = {"started_at_utc": utc(), "url": url, "passed": False}
    try:
        origin = Path(common.__file__).resolve()
        if digest(origin.read_bytes()) != expected_source:
            raise ValueError("installed native download helper source differs")
        common.configure_media_url_security(["raw.githubusercontent.com"], max_file_size_mb=4)
        body = common.download_remote_media(url, timeout=10)
        if len(body) > BODY_CAP or digest(body) != IMAGES[url.rsplit("/", 1)[-1]]:
            raise ValueError("actual native download bytes differ")
        (out / "response.body").write_bytes(body)
        record.update(passed=True, response_sha256=digest(body), response_bytes=len(body),
                      actual_source={"path": str(origin), "sha256": expected_source},
                      scope="actual SGLang download_remote_media fetch; not B Rust downloader internal attestation")
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error)[:1024])
    record["finished_at_utc"] = utc()
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def grpc_info(target, out, health_only=False):
    # Exact c9d source defines these real RPCs; no inert/fake server is used.
    import grpc
    from google.protobuf.json_format import MessageToDict
    from smg_grpc_proto import sglang_scheduler_pb2 as pb, sglang_scheduler_pb2_grpc as rpc
    out.mkdir(mode=0o700)
    record = {"started_at_utc": utc(), "passed": False}
    try:
        with grpc.insecure_channel(target) as channel:
            stub = rpc.SglangSchedulerStub(channel)
            if health_only:
                health = stub.HealthCheck(pb.HealthCheckRequest(), timeout=5)
                record.update(health=MessageToDict(health, preserving_proto_field_name=True), passed=health.healthy is True)
            else:
                model = stub.GetModelInfo(pb.GetModelInfoRequest(), timeout=3)
                server = stub.GetServerInfo(pb.GetServerInfoRequest(), timeout=3)
                record.update(model=MessageToDict(model, preserving_proto_field_name=True),
                              server=MessageToDict(server, preserving_proto_field_name=True), passed=True)
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error)[:1024])
    record["finished_at_utc"] = utc()
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["request", "json", "health", "URL", "URL-native", "grpc", "grpc-health", "metrics"])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--installed-common-sha")
    parser.add_argument("--model-value")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--metrics-source-sha")
    args = parser.parse_args()
    if args.mode in {"URL", "URL-native"}:
        assert args.endpoint in [PUG_URL, PUG_URL.replace("pug.jpg", "dog.jpg")]
    elif args.mode in {"grpc", "grpc-health"}:
        assert args.endpoint.startswith("127.0.0.1:")
    else:
        from urllib.parse import urlparse
        parsed = urlparse(args.endpoint)
        assert parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port
        assert not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
    if args.mode in {"grpc", "grpc-health"}:
        result = grpc_info(args.endpoint, args.out, health_only=args.mode == "grpc-health")
    elif args.mode == "URL-native":
        result = native_URL(args.out, args.installed_common_sha, args.endpoint)
    else:
        case = None
        if args.mode == "request":
            assert args.model_value and args.case_index in range(1, 7)
            matrix = json.loads((Path(__file__).resolve().parent / "six-request-matrix.json").read_text())
            case = matrix["requests"][args.case_index - 1]
            assert case["order"] == args.case_index
        result = http_once(args.endpoint, payload(args.fixtures, args.model_value, case) if case else None,
                           args.out, args.request_id, parse_json=args.mode not in {"health", "metrics"}, case=case)
        if args.mode == "metrics" and result["passed"]:
            try:
                import infra.mm_processing as mm
                origin = Path(mm.__file__).resolve()
                assert digest(origin.read_bytes()) == args.metrics_source_sha, "actual source metric oracle changed"
                samples = [(labels, value) for labels, value in mm.parse_mm_processing_samples(
                    (args.out / "response.body").read_text()) if value > 0]
                assert samples, "gateway recorded no multimodal processing samples"
                assert {labels["mode"] for labels, value in samples} == {"router"}, samples
                result.update(original_SGLang_metric_oracle=True, samples=samples,
                              source={"path": str(origin), "sha256": args.metrics_source_sha})
            except Exception as error:
                result.update(passed=False, error_type=type(error).__name__, error=str(error)[:1024],
                              traceback=traceback.format_exc()[:65536])
            (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
