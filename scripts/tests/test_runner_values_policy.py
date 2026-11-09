"""The committed runner values carry no API key, and only the dind sidecar is privileged."""

from pathlib import Path

import pytest
import yaml

RESOURCES = Path(__file__).resolve().parents[2] / "scripts/k8s-runner-resources"
VALUES = sorted(RESOURCES.glob("runner-values-*.yaml"))
CPU = RESOURCES / "runner-values-cpu.yaml"


def containers(values):
    spec = values.get("template", {}).get("spec", {})
    return spec.get("containers", []) + spec.get("initContainers", [])


# The one credential a runner pod may carry: the read-only Hugging Face token the GPU lanes need
# to download the gated Llama checkpoint. API keys never; see README "No API keys in the runner pods".
ALLOWED_CREDENTIALS = {"HF_TOKEN"}


@pytest.mark.parametrize("path", VALUES, ids=lambda p: p.name)
def test_no_runner_values_file_puts_an_api_key_in_the_environment(path):
    for container in containers(yaml.safe_load(path.read_text())):
        for env in container.get("env") or []:
            name = env["name"]
            if name in ALLOWED_CREDENTIALS and path != CPU:
                continue
            assert not any(word in name for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")), (
                f"{path.name}: {container['name']} carries {name}"
            )


@pytest.mark.parametrize("path", VALUES, ids=lambda p: p.name)
def test_only_a_dind_sidecar_may_be_privileged(path):
    for container in containers(yaml.safe_load(path.read_text())):
        privileged = (container.get("securityContext") or {}).get("privileged", False)
        assert not privileged or container["name"] == "dind", f"{path.name}: {container['name']}"


def test_the_cpu_set_the_reviews_run_on_carries_no_credential_at_all():
    values = yaml.safe_load(CPU.read_text())
    assert values["runnerScaleSetName"] == "k8s-runner-cpu"
    runner = next(c for c in values["template"]["spec"]["containers"] if c["name"] == "runner")
    assert [e["name"] for e in runner["env"]] == [
        "DOCKER_HOST",
        "RUNNER_WAIT_FOR_DOCKER_IN_SECONDS",
    ]
    assert not (runner.get("securityContext") or {}).get("privileged")
    assert values["maxRunners"] == 48
