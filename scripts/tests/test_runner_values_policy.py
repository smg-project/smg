"""The committed runner values keep API keys in the key-proxy sidecar only, and privilege in dind only."""

from pathlib import Path

import pytest
import yaml

RESOURCES = Path(__file__).resolve().parents[2] / "scripts/k8s-runner-resources"
VALUES = sorted(RESOURCES.glob("runner-values-*.yaml"))
CPU = RESOURCES / "runner-values-cpu.yaml"


def containers(values):
    spec = values.get("template", {}).get("spec", {})
    return spec.get("containers", []) + spec.get("initContainers", [])


# The containers that may hold a credential: the key-proxy sidecar holds the two API keys and
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
    assert "unset ANTHROPIC_API_KEY OPENAI_API_KEY NGINX_TEMPLATE" in proxy["command"][2]
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
