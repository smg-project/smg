#!/usr/bin/env python3
"""One credential-safe Brave Search request; no SMG, MCP, or GPU test."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

URL = "https://api.search.brave.com/res/v1/web/search?q=AMD&count=1"
OUTPUT = Path("brave-search-auth-result.json")
MAX_BYTES = 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a subscription token to another URL.


def main():
    key = os.environ.get("BRAVE_API_KEY", "")
    report = {
        "status": None,
        "count": 0,
        "source_sha": os.environ.get("GITHUB_SHA", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "credential_available": bool(key),
    }
    if key:
        try:
            request = urllib.request.Request(
                URL, headers={"X-Subscription-Token": key, "Accept": "application/json"},
                method="GET",
            )
            opener = urllib.request.build_opener(NoRedirect)
            with opener.open(request, timeout=20) as response:
                report["status"] = response.status
                raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                report["error_type"] = "ResponseTooLarge"
            else:
                payload = json.loads(raw)
                web = payload.get("web") if isinstance(payload, dict) else None
                results = web.get("results") if isinstance(web, dict) else None
                if isinstance(results, list):
                    report["count"] = len(results)
        except urllib.error.HTTPError as error:
            report["status"] = error.code
            error.close()  # Do not read, print, or persist its response body.
        except Exception as error:
            report["error_type"] = type(error).__name__  # No repr/message/body.
    safe_json = json.dumps(report, sort_keys=True) + "\n"
    OUTPUT.write_text(safe_json)
    print(safe_json, end="")
    return 0 if report["status"] == 200 and report["count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
