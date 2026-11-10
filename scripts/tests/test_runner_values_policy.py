"""The committed runner values keep API keys in the key-proxy sidecar only, and privilege in dind only."""

import sys
from pathlib import Path

import pytest
import yaml

RESOURCES = Path(__file__).resolve().parents[2] / "scripts/k8s-runner-resources"
VALUES = sorted(RESOURCES.glob("runner-values-*.yaml"))
CPU = RESOURCES / "runner-values-cpu.yaml"


def containers(values):
    spec = values.get("template", {}).get("spec", {})
    return spec.get("containers", []) + spec.get("initContainers", [])


# The containers that may hold a credential: the key-proxy sidecar holds provider API keys and
# nothing else may; the GPU lanes' runner keeps the read-only Hugging Face token it downloads the
# gated Llama checkpoint with. See README "No API keys in the runner container".
KEY_HOLDER = "ai-key-proxy"
ALLOWED_RUNNER_CREDENTIALS = {"HF_TOKEN"}


@pytest.mark.parametrize("path", VALUES, ids=lambda p: p.name)
def test_api_keys_live_in_the_key_proxy_sidecar_only(path):
    for container in containers(yaml.safe_load(path.read_text())):
        for env in container.get("env") or []:
            name = env["name"]
            if container["name"] == KEY_HOLDER:
                continue
            if name in ALLOWED_RUNNER_CREDENTIALS and path != CPU:
                continue
            assert not any(word in name for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")), (
                f"{path.name}: {container['name']} carries {name}"
            )


@pytest.mark.parametrize("path", VALUES, ids=lambda p: p.name)
def test_only_a_dind_sidecar_may_be_privileged(path):
    for container in containers(yaml.safe_load(path.read_text())):
        privileged = (container.get("securityContext") or {}).get("privileged", False)
        assert not privileged or container["name"] == "dind", f"{path.name}: {container['name']}"


def test_the_cpu_set_the_reviews_run_on_has_an_empty_runner_and_a_localhost_key_proxy():
    values = yaml.safe_load(CPU.read_text())
    assert values["runnerScaleSetName"] == "k8s-runner-cpu"
    spec = values["template"]["spec"]
    runner = next(c for c in spec["containers"] if c["name"] == "runner")
    assert [e["name"] for e in runner["env"]] == [
        "DOCKER_HOST",
        "RUNNER_WAIT_FOR_DOCKER_IN_SECONDS",
    ]
    assert not (runner.get("securityContext") or {}).get("privileged")
    assert values["maxRunners"] == 48
    proxy = next(c for c in spec["initContainers"] if c["name"] == KEY_HOLDER)
    assert proxy["restartPolicy"] == "Always"
    assert "@sha256:" in proxy["image"]
    env = {e["name"]: e for e in proxy["env"]}
    assert env["ANTHROPIC_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "anthropic-api-key"
    assert env["OPENAI_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "openai-api-key"
    template = env["NGINX_TEMPLATE"]["value"]
    assert "listen 127.0.0.1:8787;" in template
    assert "proxy_ssl_verify on;" in template
    assert 'proxy_set_header Authorization "";' in template
    assert (
        "unset ANTHROPIC_API_KEY OPENAI_API_KEY XAI_API_KEY NGINX_TEMPLATE" in proxy["command"][2]
    )
    command = proxy["command"][2]
    assert "sed -i 's/^worker_processes .*/worker_processes 1;/' /etc/nginx/nginx.conf" in command
    # a -g worker_processes duplicates the directive in nginx.conf and nginx refuses to start
    assert (
        "-g 'daemon off;'" in command
        and "worker_processes 1;'" not in command.split("exec nginx")[1]
    )


def test_the_codex_app_key_lives_in_the_key_proxy_and_is_only_exchanged_for_an_hour_token():
    values = yaml.safe_load(CPU.read_text())
    spec = values["template"]["spec"]
    volume = next(v for v in spec["volumes"] if v["name"] == "codex-github-app")
    assert volume["secret"]["secretName"] == "codex-github-app"
    assert volume["secret"]["defaultMode"] == 0o444  # the nginx workers run as nginx, not root
    mounts = {
        c["name"]: [m for m in c.get("volumeMounts", []) if m["name"] == "codex-github-app"]
        for c in containers(values)
    }
    assert mounts.pop(KEY_HOLDER) == [
        {"name": "codex-github-app", "mountPath": "/run/secrets/codex-github-app", "readOnly": True}
    ]
    assert all(not m for m in mounts.values()), "only the key proxy may see the App key"
    proxy = next(c for c in spec["initContainers"] if c["name"] == KEY_HOLDER)
    assert "load_module modules/ngx_http_js_module.so;" in proxy["command"][2]
    env = {e["name"]: e for e in proxy["env"]}
    template = env["NGINX_TEMPLATE"]["value"]
    assert "location = /github/app-token" in template and "js_content" in template
    assert "js_fetch_trusted_certificate /etc/ssl/certs/ca-certificates.crt;" in template
    script = env["NGINX_JS"]["value"]
    assert "/run/secrets/codex-github-app/" in script
    assert "RSASSA-PKCS1-v1_5" in script and "/access_tokens" in script
    assert "exp: now + 540" in script  # a JWT good for nine minutes buys a token good for an hour


def test_the_token_endpoint_admits_only_a_job_that_presents_a_github_oidc_token():
    values = yaml.safe_load(CPU.read_text())
    proxy = next(c for c in values["template"]["spec"]["initContainers"] if c["name"] == KEY_HOLDER)
    script = next(e for e in proxy["env"] if e["name"] == "NGINX_JS")["value"]
    # the caller's GitHub Actions OIDC token, checked against GitHub's signing keys and pinned to
    # this repository, this workflow file and the audience the post job asks for
    assert "https://token.actions.githubusercontent.com/.well-known/jwks" in script
    assert "'https://token.actions.githubusercontent.com'" in script
    assert "'ai-key-proxy'" in script
    assert "'smg-project/smg'" in script
    assert "'.github/workflows/codex-code-review.yml@'" in script
    assert "crypto.subtle.verify" in script
    assert "r.headersIn.Authorization" in script
    assert "r.return(401" in script


def test_optional_xai_key_is_sidecar_only_and_missing_key_fails_closed():
    proxy = next(c for c in containers(yaml.safe_load(CPU.read_text())) if c["name"] == KEY_HOLDER)
    env = {e["name"]: e for e in proxy["env"]}
    assert env["XAI_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "key": "XAI_API_KEY",
        "name": "xai-api-key",
        "optional": True,
    }
    template = env["NGINX_TEMPLATE"]["value"]
    xai = template.split("location /xai/ {", 1)[1]
    assert 'if ($xai_key = "") { return 503' in xai
    assert "proxy_pass https://api.x.ai/;" in xai
    assert 'proxy_set_header Authorization "Bearer $xai_key";' in xai


def test_template_substitution_keeps_nginx_websocket_variables(monkeypatch):
    import os
    import shutil
    import subprocess

    if shutil.which("envsubst") is None:
        pytest.skip(
            "real envsubst is installed in the proxy image; unavailable in this CPU environment"
        )
    proxy = next(c for c in containers(yaml.safe_load(CPU.read_text())) if c["name"] == KEY_HOLDER)
    template = next(e["value"] for e in proxy["env"] if e["name"] == "NGINX_TEMPLATE")
    # Exercise the actual restricted envsubst allowlist. An unrestricted expansion would erase
    # runtime Nginx variables and break Upgrade, Connection and optional-key routing.
    command = proxy["command"][2]
    allowlist = command.split("envsubst '", 1)[1].split("'", 1)[0]
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "RESOLVER"):
        monkeypatch.setenv(name, "test-" + name)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    rendered = subprocess.run(
        ["envsubst", allowlist],
        input=template,
        text=True,
        env=os.environ.copy(),
        capture_output=True,
        check=True,
    ).stdout
    assert "${" not in rendered
    assert 'set $xai_key "";' in rendered
    assert "map $http_upgrade $smg_connection_upgrade" in rendered
    assert "proxy_set_header Upgrade $http_upgrade;" in rendered
    assert "proxy_set_header Connection $smg_connection_upgrade;" in rendered
    assert 'proxy_set_header Authorization "Bearer test-OPENAI_API_KEY";' in rendered
    assert 'proxy_set_header Authorization "Bearer $xai_key";' in rendered


def test_vendor_job_cannot_restore_real_keys_into_the_runner_environment():
    workflow = RESOURCES.parents[1] / ".github/workflows/pr-test-rust.yml"
    job = yaml.safe_load(workflow.read_text())["jobs"]["e2e-vendor"]
    for env in [job.get("env", {})] + [step.get("env", {}) for step in job["steps"]]:
        for name, value in env.items():
            assert "secrets." not in str(value), f"vendor job exports a real credential: {name}"


@pytest.mark.parametrize("scope", ["job", "step"])
def test_vendor_secret_alias_is_rejected(tmp_path, monkeypatch, scope):
    workflow = RESOURCES.parents[1] / ".github/workflows/pr-test-rust.yml"
    document = yaml.safe_load(workflow.read_text())
    job = document["jobs"]["e2e-vendor"]
    target = job if scope == "job" else job["steps"][0]
    target.setdefault("env", {})["OPENAI_TOKEN"] = "${{ secrets.OPENAI_API_KEY }}"
    fixture = tmp_path / ".github/workflows/pr-test-rust.yml"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(
        sys.modules[__name__],
        "RESOURCES",
        tmp_path / "scripts/k8s-runner-resources",
    )
    with pytest.raises(AssertionError, match="vendor job exports a real credential: OPENAI_TOKEN"):
        test_vendor_job_cannot_restore_real_keys_into_the_runner_environment()
