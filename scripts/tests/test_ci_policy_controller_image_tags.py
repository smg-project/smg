"""The policy controller's image is pushed under every tag of the router's.

A release and a nightly publish the controller's image as the router's tag
with `-policy-controller` appended: one version names both images. A run that
pushed a router tag without that twin would break the pairing for whoever
names the controller by the router's version. These cases run each workflow's
own tags step and read what it writes, then read the build steps that take
the two lists."""

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SUFFIX = "-policy-controller"
SHA = "0123456789abcdef0123456789abcdef01234567"


def steps(workflow: str) -> list[dict]:
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    return document["jobs"]["build-and-push"]["steps"]


def written(workflow: str, environment: dict[str, str], tmp_path: Path) -> dict:
    """The workflow's tags step run under bash, and what it wrote to its
    output file: a `name=value` line as a string, a `name<<EOF` block as a
    list of lines."""
    (step,) = [step for step in steps(workflow) if step.get("id") == "tags"]
    file = tmp_path / "output"
    file.write_text("")
    subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(file), **environment},
        check=True,
        capture_output=True,
    )
    found: dict = {}
    lines = file.read_text().splitlines()
    while lines:
        line = lines.pop(0)
        if "<<" in line:
            name, mark = line.split("<<")
            end = lines.index(mark)
            found[name], lines = lines[:end], lines[end + 1 :]
        else:
            name, _, value = line.partition("=")
            found[name] = value
    return found


@pytest.mark.parametrize("version", ["1.16.0", "1.16.0-rc.1"], ids=["a final", "a candidate"])
def test_a_release_pushes_the_controller_under_every_router_tag(tmp_path, version):
    found = written("release-docker.yml", {"VERSION": version, "OWNER": "example"}, tmp_path)
    assert f"lightseekorg/smg:{version}" in found["tags"]
    assert ("lightseekorg/smg:latest" in found["tags"]) == (found["latest"] == "true")
    assert found["controller_tags"] == [tag + SUFFIX for tag in found["tags"]]


@pytest.mark.parametrize("express", ["false", "true"], ids=["a full run", "an express run"])
def test_a_nightly_pushes_the_controller_under_every_router_tag(tmp_path, express):
    environment = {"OWNER": "example", "EXPRESS": express, "GITHUB_SHA": SHA}
    found = written("nightly-docker.yml", environment, tmp_path)
    assert ("ghcr.io/example/smg:nightly" in found["list"]) == (express == "false")
    assert found["controller_list"] == [tag + SUFFIX for tag in found["list"]]


@pytest.mark.parametrize(
    "workflow,router,controller",
    [
        ("release-docker.yml", "tags", "controller_tags"),
        ("nightly-docker.yml", "list", "controller_list"),
    ],
)
def test_the_controller_is_built_and_pushed_as_the_router_is(workflow, router, controller):
    """The two build steps take the two lists of one tags step, and the
    controller's is pushed when the router's is, for the same platforms. It
    runs when the router's step succeeded, whatever a step between them did."""
    by_list = {
        step["with"]["tags"]: step for step in steps(workflow) if "tags" in step.get("with", {})
    }
    of_the_router = by_list["${{ steps.tags.outputs." + router + " }}"]
    of_the_controller = by_list["${{ steps.tags.outputs." + controller + " }}"]
    assert of_the_router["with"]["file"] == "docker/Dockerfile"
    assert of_the_controller["with"]["file"] == "docker/policy-controller.Dockerfile"
    assert of_the_controller["with"]["target"] == "policy-controller-image"
    for key in ("push", "platforms", "context"):
        assert of_the_controller["with"][key] == of_the_router["with"][key], key
    router_built = "steps." + of_the_router["id"] + ".outcome == 'success'"
    assert of_the_controller["if"] == "${{ !cancelled() && " + router_built + " }}"
