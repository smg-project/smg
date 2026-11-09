"""The committed runner values carry no credentials, and the review set carries no Docker."""

from pathlib import Path

import pytest
import yaml

RESOURCES = Path(__file__).resolve().parents[2] / "scripts/k8s-runner-resources"
VALUES = sorted(RESOURCES.glob("runner-values-*.yaml"))
REVIEW = RESOURCES / "runner-values-ai-review-cpu.yaml"


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
            if name in ALLOWED_CREDENTIALS and path != REVIEW:
                continue
            assert not any(word in name for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")), (
                f"{path.name}: {container['name']} carries {name}"
            )


@pytest.mark.parametrize("path", VALUES, ids=lambda p: p.name)
def test_only_a_dind_sidecar_may_be_privileged(path):
    for container in containers(yaml.safe_load(path.read_text())):
        privileged = (container.get("securityContext") or {}).get("privileged", False)
        assert not privileged or container["name"] == "dind", f"{path.name}: {container['name']}"


def test_the_review_set_has_no_docker_and_no_privilege():
    values = yaml.safe_load(REVIEW.read_text())
    assert values["runnerScaleSetName"] == "ai-review-cpu"
    assert values["githubConfigUrl"] == "https://github.com/smg-project/smg"
    spec = values["template"]["spec"]
    assert "initContainers" not in spec
    assert [c["name"] for c in spec["containers"]] == ["runner"]
    runner = spec["containers"][0]
    assert runner["env"] == []
    assert runner["securityContext"]["privileged"] is False
    assert runner["securityContext"]["runAsNonRoot"] is True
    assert all("dind" not in v["name"] for v in spec["volumes"])
