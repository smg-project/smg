"""Exercise the actual Ruby program embedded in the cluster trigger manifest."""

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

MANIFEST = Path(__file__).resolve().parents[1] / "k8s-runner-resources/ci-node-health-trigger.yaml"
pytestmark = pytest.mark.skipif(
    shutil.which("ruby") is None, reason="Ruby is required by the trigger"
)


@pytest.fixture
def trigger():
    cron = next(d for d in yaml.safe_load_all(MANIFEST.read_text()) if d["kind"] == "CronJob")
    code = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["args"][0]

    def run(test_code, **data):
        harness = "input = JSON.parse(STDIN.read)\n"
        harness += 'eval(input.fetch("code"), TOPLEVEL_BINDING, "ci_node_health_trigger", 1)\n'
        harness += test_code
        result = subprocess.run(
            ["ruby", "-rjson", "-rnet/http", "-ropenssl", "-rbase64", "-rtime", "-e", harness],
            input=json.dumps({"code": code, **data}),
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return json.loads(result.stdout)

    return run


def _runner(name, phase="Failed", repo="smg", pool="1-gpu-h100"):
    return {
        "metadata": {
            "name": name,
            "labels": {
                "actions.github.com/organization": "smg-project",
                "actions.github.com/repository": repo,
                "actions.github.com/scale-set-name": pool,
            },
        },
        "status": {
            "phase": phase,
            "reason": "InvalidPod",
            "message": "x" * 1000,
            "runnerJITConfig": "must-not-be-forwarded",
        },
        "spec": {"token": "must-not-be-forwarded"},
    }


def test_summary_counts_all_failures_but_bounds_examples_and_omits_credentials(trigger):
    items = [_runner(f"runner-{i}") for i in range(7)]
    items += [
        _runner("active", phase="Running"),
        _runner("foreign", repo="other"),
        _runner("other-pool", pool="unknown"),
    ]
    pools = trigger('puts JSON.generate(summarize_runners(input.fetch("items")))', items=items)
    assert pools[0]["failed"] == 7
    assert [e["name"] for e in pools[0]["examples"]] == ["runner-0", "runner-1", "runner-2"]
    assert len(pools[0]["examples"][0]["message"]) == 500
    assert [p["failed"] for p in pools[1:]] == [0, 0, 0]
    assert "must-not-be-forwarded" not in json.dumps(pools)


def test_snapshot_reads_all_pages_before_reporting_clean_pools(trigger):
    result = trigger(
        """
        File.define_singleton_method(:read) { |*args| "dummy-token" }
        requests = []
        define_singleton_method(:api) do |method, path, token, **options|
          requests << { method: method::METHOD, path: path, options: options }
          if requests.length == 1
            { "items" => [], "metadata" => { "continue" => "next/page" } }
          else
            { "items" => input.fetch("items"), "metadata" => {} }
          end
        end
        snapshot = runner_snapshot
        puts JSON.generate(snapshot: snapshot, requests: requests)
        """,
        items=[_runner("failed-on-second-page")],
    )
    assert len(result["requests"]) == 2
    assert all(r["method"] == "GET" for r in result["requests"])
    assert "continue=next%2Fpage" in result["requests"][1]["path"]
    assert result["requests"][0]["options"] == {
        "base": "https://kubernetes.default.svc",
        "ca_file": "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    }
    assert result["snapshot"]["pools"][0]["failed"] == 1
    assert "checked_at" in result["snapshot"]


def test_snapshot_failure_is_not_an_empty_healthy_result(trigger):
    snapshot = trigger("""
        File.define_singleton_method(:read) { |*args| raise "token-bearing-details-must-not-leak" }
        puts JSON.generate(runner_snapshot)
    """)
    assert "error" in snapshot and "pools" not in snapshot
    assert "token-bearing-details" not in snapshot["error"]


def test_repeated_pagination_token_reports_unreadable_snapshot(trigger):
    snapshot = trigger("""
        File.define_singleton_method(:read) { |*args| "dummy-token" }
        define_singleton_method(:api) do |*args, **options|
          { "items" => [], "metadata" => { "continue" => "same-page" } }
        end
        puts JSON.generate(runner_snapshot)
    """)
    assert "error" in snapshot and "pools" not in snapshot


@pytest.mark.skipif(shutil.which("openssl") is None, reason="OpenSSL verifies the test signature")
def test_app_jwt_signature_and_claims_are_valid(trigger, tmp_path):
    key = tmp_path / "private-key.pem"
    (tmp_path / "app-id").write_text("12345")
    subprocess.run(["openssl", "genrsa", "-out", str(key), "2048"], check=True, capture_output=True)
    jwt = trigger(
        """
        Time.define_singleton_method(:now) { Time.at(1000) }
        puts JSON.generate(app_jwt(input.fetch("directory")))
        """,
        directory=str(tmp_path),
    )
    header, claims, signature = jwt.split(".")

    def decode(value):
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))

    assert json.loads(decode(header)) == {"alg": "RS256", "typ": "JWT"}
    assert json.loads(decode(claims)) == {"iat": 940, "exp": 1540, "iss": "12345"}
    public = subprocess.run(
        ["openssl", "rsa", "-in", str(key), "-pubout"], check=True, capture_output=True
    ).stdout
    (tmp_path / "public.pem").write_bytes(public)
    (tmp_path / "signature").write_bytes(decode(signature))
    verified = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            str(tmp_path / "public.pem"),
            "-signature",
            str(tmp_path / "signature"),
        ],
        input=f"{header}.{claims}".encode(),
        capture_output=True,
    )
    assert verified.returncode == 0


def test_dispatch_failure_still_revokes_installation_token(trigger):
    result = trigger("""
        File.define_singleton_method(:read) { |*args| "123" }
        define_singleton_method(:app_jwt) { |*args| "dummy-jwt" }
        define_singleton_method(:runner_snapshot) { { "error" => "unavailable" } }
        requests = []
        define_singleton_method(:api) do |method, path, token, body = nil|
          requests << { method: method::METHOD, path: path, body: body }
          if path.end_with?("/access_tokens")
            { "token" => "dummy-installation-token" }
          elsif path.end_with?("/dispatches")
            raise "dispatch failed"
          else
            {}
          end
        end
        begin
          main
        rescue => error
          puts JSON.generate(error: error.message, requests: requests)
        end
    """)
    assert result["error"] == "dispatch failed"
    assert result["requests"][0]["body"] == {
        "repositories": ["smg"],
        "permissions": {"actions": "write"},
    }
    assert json.loads(result["requests"][1]["body"]["inputs"]["runner_failures_json"]) == {
        "error": "unavailable"
    }
    assert result["requests"][-1]["method"] == "DELETE"
    assert result["requests"][-1]["path"] == "/installation/token"


def test_snapshot_input_stays_bounded_with_unicode_in_every_example(trigger):
    pools = ("1-gpu-h100", "2-gpu-h100", "4-gpu-h100", "k8s-runner-cpu")
    items = [_runner(f"runner-{pool}-{i}", pool=pool) for pool in pools for i in range(3)]
    for item in items:
        item["status"]["message"] = "😀" * 500
    serialized = trigger(
        """
        snapshot = { "checked_at" => "2026-10-05T00:00:00+00:00", "pools" => summarize_runners(input.fetch("items")) }
        puts JSON.generate(serialize_snapshot(snapshot))
        """,
        items=items,
    )
    assert len(serialized.encode()) <= 48000
    snapshot = json.loads(serialized)
    assert [p["failed"] for p in snapshot["pools"]] == [3, 3, 3, 3]
    assert snapshot["pools"][0]["examples"][0]["message"] == "😀" * 500


def test_oversize_snapshot_reports_blindness_without_blocking_dispatch(trigger):
    serialized = trigger(
        'puts JSON.generate(serialize_snapshot(input.fetch("snapshot")))',
        snapshot={"error": "x" * 48001},
    )
    assert len(serialized.encode()) <= 48000
    assert json.loads(serialized) == {"error": "Runner snapshot exceeded dispatch size limit"}


def test_shared_api_requests_json_and_verifies_tls_without_retries(trigger):
    result = trigger("""
        requests = []
        Net::HTTP.define_singleton_method(:start) do |host, port, **options, &block|
          http = Struct.new(:max_retries).new
          http.define_singleton_method(:request) do |request|
            requests << { host: host, accept: request["Accept"], options: options, retries: max_retries }
            response = Net::HTTPOK.new("1.1", "200", "OK")
            response.define_singleton_method(:body) { "{}" }
            response
          end
          block.call(http)
        end
        api(Net::HTTP::Get, "/runners", "dummy-token", base: "https://kubernetes.default.svc", ca_file: "/ca.crt")
        api(Net::HTTP::Post, "/dispatches", "dummy-token")
        puts JSON.generate(requests)
    """)
    assert [r["accept"] for r in result] == ["application/json", "application/json"]
    assert [r["retries"] for r in result] == [0, 0]
    assert result[0]["options"]["ca_file"] == "/ca.crt"
    assert all(r["options"]["verify_mode"] == 1 for r in result)
