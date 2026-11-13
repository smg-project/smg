"""Exercise six real HTTP/GPU cases; timings are smoke-test observations, not an A/B benchmark.

Requires Pillow and an already running managed gateway with real E/P/D workers.
The test fails on content, protocol or usage errors; no mock or media fallback is used.
"""

import argparse
import base64
import io
import json
import time
import urllib.request

from PIL import Image


def main():
    """Fail on any protocol, content or usage regression in the six real-worker cases."""
    parser = argparse.ArgumentParser(
        description="Real managed EPD HTTP/GPU acceptance for text, images, streaming and usage."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/") + "/v1/chat/completions"
    for stream in (False, True):
        for count in (0, 1, 2):
            content = [
                {
                    "type": "text",
                    "text": "Name only the colors in these images, in order."
                    if count
                    else "What is 2 plus 2? Answer with one digit.",
                }
            ]
            for color in ["red", "blue"][:count]:
                b = io.BytesIO()
                Image.new("RGB", (224, 224), color).save(b, format="PNG")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(b.getvalue()).decode()
                        },
                    }
                )
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 64,
                "stream": stream,
                "n": 1,
            }
            if stream:
                body["stream_options"] = {"include_usage": True}
            start = time.monotonic()
            text = ""
            first = None
            usage = None
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if stream:
                        for line in response:
                            if not line.startswith(b"data: "):
                                continue
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                break
                            event = json.loads(data)
                            if "error" in event:
                                raise RuntimeError(event)
                            for choice in event.get("choices", []):
                                part = choice.get("delta", {}).get("content") or ""
                                if part and first is None:
                                    first = time.monotonic() - start
                                text += part
                            usage = event.get("usage") or usage
                    else:
                        event = json.load(response)
                        print("RESPONSE", event, flush=True)
                        text = event["choices"][0]["message"]["content"]
                        usage = event.get("usage")
                print(
                    "HTTP",
                    stream,
                    count,
                    "seconds",
                    time.monotonic() - start,
                    "ttft",
                    first,
                    "text",
                    repr(text),
                    "usage",
                    usage,
                    flush=True,
                )
                assert usage and usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, (
                    usage
                )
                assert (
                    ("4" in text)
                    if count == 0
                    else all(c in text.lower() for c in ["red", "blue"][:count])
                )
            except urllib.error.HTTPError as e:
                print("HTTP_ERROR", e.code, e.read().decode(), flush=True)
                raise


if __name__ == "__main__":
    main()
