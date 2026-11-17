"""The Helm chart's policy controller block: off by default, and when on,
every argument its Deployment passes is one the controller binary takes.

The controller is a Rust binary (``smg-policy-controller``), so no Python
parser stands in for it here: the binary cases run the real binary, named
by the ``SMG_POLICY_CONTROLLER_BIN`` environment variable, and are skipped
without it. The render cases need the ``helm`` binary alone and are skipped
without it.

This module renders the chart with its own few lines and not through
``test_helm_chart_args``: that module imports the gateway's launcher, which
the job that runs these cases on the image's binary does not install.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
CHART = REPO / "deploy" / "helm" / "smg"
# The job that builds the controller image runs these cases on its binary.
IMAGE_WORKFLOW = REPO / ".github" / "workflows" / "pr-policy-controller-image.yml"
# One run of each kind the two publishing workflows have, by the environment
# its tags step reads.
PUBLISHING_RUNS = {
    "a final": ("release-docker.yml", {"VERSION": "1.16.0", "OWNER": "example"}),
    "a candidate": ("release-docker.yml", {"VERSION": "1.16.0-rc.1", "OWNER": "example"}),
    "a full nightly": ("nightly-docker.yml", {"OWNER": "example", "EXPRESS": "false"}),
    "an express nightly": ("nightly-docker.yml", {"OWNER": "example", "EXPRESS": "true"}),
}
VALUES = CHART / "values.yaml"
SCHEMA = CHART / "values.schema.json"
EXAMPLES = CHART / "examples"
# The chart needs Kubernetes >= 1.26 and `helm template` checks it against
# its default version, which older helm releases put below that.
KUBE_VERSION = "1.30.0"
BINARY_VARIABLE = "SMG_POLICY_CONTROLLER_BIN"
COMPONENT = "policy-controller"
NAME = "smg-policy-controller"
# The block turned on. The switch is all it takes: every other value holds a
# default, and the image's is the router's.
ON = ("policyController.enabled=true",)
# The block with the election turned on.
ELECTED = (*ON, "policyController.election.enabled=true")
# The spellings of an election that is off.
OFF = ((), ("policyController.election.enabled=false",), ("policyController.election=null",))
# The strategy of a controller with the election off: no pod more at a
# time, so the old pod is told to stop before the new one is made. The type
# is the rolling update's, as with the election on.
LONE_STRATEGY = {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1}}
# The variable the elected controller's address is read from: the kubelet
# writes the pod's address in place of $(POD_IP).
POD_IP = {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}
# The settle window the controller takes when the chart passes none.
DEFAULT_SETTLE_SECONDS = 5
# A policy spec document the stock controller runs, another one, and one no
# build runs: its filter is none the controller knows.
LEAST_LOAD = {
    "apiVersion": "smg.policy/v1",
    "profiles": {"default": {"filter": ["healthy"], "score": ["least-load"], "pick": "best"}},
}
RANDOM = {
    "apiVersion": "smg.policy/v1",
    "profiles": {"default": {"filter": ["healthy"], "pick": "random"}},
}
UNKNOWN_FILTER = {
    "apiVersion": "smg.policy/v1",
    "profiles": {"default": {"filter": ["no-such-filter"], "pick": "random"}},
}
# Where the pod mounts the document's ConfigMap, and the file under it that
# the controller is passed.
SPEC_MOUNT = "/etc/smg/policy-spec"
SPEC_FILE = f"{SPEC_MOUNT}/spec.yaml"
# The started controller is read on a loopback port: a proxy named in the
# environment is not asked for it.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# A pod's environment names none of the variables the controller's log reads
# for its color or its level.
LOG_VARIABLES = ("NO_COLOR", "FORCE_COLOR", "CLICOLOR_FORCE", "RUST_LOG")

needs_helm = pytest.mark.skipif(
    shutil.which("helm") is None or not CHART.exists(), reason="helm or the chart not present"
)


def helm_template(
    settings: tuple[str, ...],
    files: tuple[Path, ...] = (),
    chart: Path = CHART,
    namespace: str = "default",
    skip_schema: bool = False,
):
    command = ["helm", "template", "smg", str(chart), "--kube-version", KUBE_VERSION]
    # Pinned so the render never follows the kube context.
    command += ["--namespace", namespace]
    if skip_schema:
        # The helper's own refusals are read with the schema's check off.
        command.append("--skip-schema-validation")
    for file in files:
        command += ["-f", str(file)]
    for setting in settings:
        command += ["--set", setting]
    return subprocess.run(command, capture_output=True, text=True)


def render(
    *settings: str,
    files: tuple[Path, ...] = (),
    chart: Path = CHART,
    namespace: str = "default",
    skip_schema: bool = False,
) -> list[dict]:
    """The documents the chart renders with ``--set`` pairs."""
    completed = helm_template(settings, files, chart, namespace, skip_schema)
    assert completed.returncode == 0, completed.stderr
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def render_error(
    *settings: str, files: tuple[Path, ...] = (), chart: Path = CHART, skip_schema: bool = False
) -> str:
    """Helm's error for ``--set`` pairs the chart must refuse."""
    completed = helm_template(settings, files, chart, skip_schema=skip_schema)
    assert completed.returncode != 0, "the chart rendered values it should refuse"
    return completed.stderr


def component(document: dict) -> str | None:
    return document["metadata"].get("labels", {}).get("app.kubernetes.io/component")


def of_the_controller(documents: list[dict]) -> list[dict]:
    """The documents that carry the controller's component or its name."""
    return [
        document
        for document in documents
        if component(document) == COMPONENT or COMPONENT in document["metadata"]["name"]
    ]


def one(documents: list[dict], kind: str, name: str = NAME) -> dict:
    (document,) = [d for d in documents if d["kind"] == kind and d["metadata"]["name"] == name]
    return document


def container(documents: list[dict]) -> dict:
    (found,) = one(documents, "Deployment")["spec"]["template"]["spec"]["containers"]
    return found


def controller_args(*settings: str) -> list[str]:
    """The arguments the rendered Deployment passes to the controller."""
    return container(render(*ON, *settings))["args"]


def spec_values(directory: Path, document: object, name: str = "spec.yaml") -> Path:
    """A values file that holds ``document`` as the controller's spec."""
    values = directory / name
    values.write_text(yaml.safe_dump({"policyController": {"spec": document}}))
    return values


def selects(selector: dict[str, str], labels: dict[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in selector.items())


@needs_helm
def test_the_controller_is_off_by_default():
    assert of_the_controller(render()) == []
    assert of_the_controller(render("policyController.enabled=false")) == []
    # An election asked for with the controller off renders nothing of it.
    assert of_the_controller(render("policyController.election.enabled=true")) == []


@pytest.fixture
def chart_with_no_block(tmp_path) -> Path:
    """A copy of the chart whose values file holds no controller block: what
    an upgrade that reuses the values of a release from before the block
    hands the templates in place of the chart's own values."""
    values = yaml.safe_load(VALUES.read_text())
    del values["policyController"]
    chart = tmp_path / "smg"
    shutil.copytree(CHART, chart)
    (chart / "values.yaml").write_text(yaml.safe_dump(values))
    return chart


@needs_helm
def test_a_controller_that_is_off_asks_nothing_of_the_values(chart_with_no_block):
    """With no block, a null block over the chart's values or over none, the
    switch alone named off or null, or one value or a spec document and no
    switch, the chart renders what its defaults render, byte for byte.
    Each again with the router's monitor on: the controller's monitor stands
    behind that switch."""
    cells = {
        "no block": (chart_with_no_block, ()),
        "a null block": (CHART, ("policyController=null",)),
        "a null block over no block": (chart_with_no_block, ("policyController=null",)),
        "the switch alone, off": (chart_with_no_block, ("policyController.enabled=false",)),
        "the switch alone, null": (chart_with_no_block, ("policyController.enabled=null",)),
        "one value and no switch": (chart_with_no_block, ("policyController.ports.admin=9000",)),
        "a document and no switch": (
            chart_with_no_block,
            ("policyController.spec.apiVersion=smg.policy/v1",),
        ),
    }
    wrong = {}
    for monitor in ((), ("router.metrics.serviceMonitor.enabled=true",)):
        default = helm_template(monitor)
        assert default.returncode == 0, default.stderr
        for name, (chart, settings) in cells.items():
            rendered = helm_template((*monitor, *settings), chart=chart)
            if rendered.returncode != 0:
                wrong[name, monitor] = rendered.stderr
            elif rendered.stdout != default.stdout:
                wrong[name, monitor] = "renders other bytes than the defaults"
    assert wrong == {}
    # Every list of required values stands under the schema's condition: none
    # holds while the controller is off.
    assert '"required"' not in json.dumps(schema_block()["properties"])


def answered_as_the_schema_names(chart: Path) -> tuple[list[set[str]], str, dict]:
    """The controller turned on over ``chart`` and given, round by round, the
    chart's own default for each value the schema names, until it renders:
    the names of each round, the render, and the block that was given."""
    defaults = yaml.safe_load(VALUES.read_text())["policyController"]
    lists = schema_block()["then"]["properties"]
    given: dict = {"enabled": True}
    rounds: list[set[str]] = []
    answers = chart.parent / "answers.yaml"
    while len(rounds) < 4:
        answers.write_text(yaml.safe_dump({"policyController": given}))
        completed = helm_template((), files=(answers,), chart=chart)
        if completed.returncode == 0:
            return rounds, completed.stdout, given
        named = re.findall(r"policyController(?:\.(\w+))?: (\w+) is required", completed.stderr)
        assert named, completed.stderr
        rounds.append({f"{parent}.{name}" if parent else name for parent, name in named})
        for parent, name in named:
            if parent:
                given[parent][name] = defaults[parent][name]
            else:
                # An object with required members is named member by member
                # in the next round.
                given[name] = {} if "required" in lists.get(name, {}) else defaults[name]
    raise AssertionError(f"the controller did not render after {rounds}")


@needs_helm
def test_turned_on_over_values_with_no_block_renders_a_fresh_install(chart_with_no_block):
    """An upgrade that reuses the values of a release from before the block
    and turns the controller on has none of the block's defaults. The schema
    names every value that holds one, round by round. Answered with the
    chart's own values until it renders, the upgrade renders a fresh install,
    byte for byte: nothing is left to another default."""
    rounds, answered, given = answered_as_the_schema_names(chart_with_no_block)
    assert rounds[0] == set(schema_block()["then"]["required"])
    fresh = helm_template(ON)
    assert fresh.returncode == 0, fresh.stderr
    assert answered == fresh.stdout
    # Over values with no block a null is not removed with a default: it is
    # there as a null, and refused as one.
    for name in ("securityContext", "runtimeWorkerThreads"):
        null = helm_template((ON[0], f"policyController.{name}=null"), chart=chart_with_no_block)
        assert f"policyController.{name}: Invalid type" in null.stderr
    # No default stands under a map given there either: an empty security
    # context runs the container with none, where over the chart's own
    # values the same answers keep the default.
    given["securityContext"] = {}
    no_context = chart_with_no_block.parent / "no-context.yaml"
    no_context.write_text(yaml.safe_dump({"policyController": given}))
    bare = container(render(files=(no_context,), chart=chart_with_no_block))
    assert "securityContext" not in bare
    assert helm_template((), files=(no_context,)).stdout == fresh.stdout


@needs_helm
def test_only_its_own_example_turns_the_controller_on():
    examples = sorted(EXAMPLES.glob("*.yaml"))
    assert examples, "the chart has examples"
    on = [example.name for example in examples if of_the_controller(render(files=(example,)))]
    assert on == ["with-policy-controller.yaml"]


@needs_helm
def test_an_example_on_a_release_with_no_controller_image_needs_the_tag_named():
    """The controller turned on over each example: it renders on the default
    tag, or the example pins the router at a release with no controller
    image and the chart says so until the controller's tag is named."""
    for example in sorted(EXAMPLES.glob("*.yaml")):
        completed = helm_template(ON, files=(example,))
        if completed.returncode != 0:
            assert "policyController.image.tag is needed" in completed.stderr, example.name
            named = render(*ON, "policyController.image.tag=mine", files=(example,))
            assert container(named)["image"].endswith(":mine"), example.name


def router_image(documents: list[dict]) -> str:
    (deployment,) = [d for d in documents if d["kind"] == "Deployment" and component(d) == "router"]
    (found,) = deployment["spec"]["template"]["spec"]["containers"]
    return found["image"]


@needs_helm
@pytest.mark.parametrize(
    "router",
    [
        (),
        ("global.image.tag=1.16.1",),
        ("global.image.registry=registry.example.com", "global.image.repository=mirror/smg"),
    ],
    ids=["the chart's own", "a tag", "a mirror"],
)
def test_the_image_follows_the_routers(router):
    """With no part of it named, the controller's image is the router's with
    -policy-controller appended to the tag, the tag a release publishes the
    controller under: one version names both images. A named registry or tag
    wins alone, an image of the operator's own names its repository and its
    tag, and the router's own image never moves."""
    of_the_router = router_image(render(*router))
    registry, _, path = of_the_router.partition("/")
    repository = path.rpartition(":")[0]

    def image(*settings: str) -> str:
        documents = render("policyController.enabled=true", *router, *settings)
        assert router_image(documents) == of_the_router
        return container(documents)["image"]

    assert image() == f"{of_the_router}-policy-controller"
    assert image("policyController.image=null") == f"{of_the_router}-policy-controller"
    assert image("policyController.image.registry=ghcr.io") == f"ghcr.io/{path}-policy-controller"
    assert image("policyController.image.tag=dev") == f"{registry}/{repository}:dev"
    own = ("policyController.image.repository=example/controller", "policyController.image.tag=dev")
    assert image(*own) == f"{registry}/example/controller:dev"


@needs_helm
def test_a_release_names_the_image_it_publishes():
    """The chart's own values at a release: the router's image on Docker Hub
    and the controller's beside it, under the suffixed tag."""
    documents = render("policyController.enabled=true", "global.image.tag=1.16.0")
    assert router_image(documents) == "docker.io/lightseekorg/smg:1.16.0"
    assert container(documents)["image"] == "docker.io/lightseekorg/smg:1.16.0-policy-controller"


@needs_helm
def test_a_router_on_its_slurm_variant_takes_the_releases_controller():
    """A release publishes one controller image. The Slurm variant is the
    router's alone, so its suffix is cut before the controller's is appended:
    the chart's own Slurm example, with the controller on. The variant's
    suffix ends a tag: the same letters inside one are the tag's own."""
    example = EXAMPLES / "with-slurm-discovery.yaml"
    documents = render("policyController.enabled=true", files=(example,))
    of_the_router = router_image(documents)
    assert of_the_router.endswith("-slurm")
    release = of_the_router.removesuffix("-slurm")
    assert container(documents)["image"] == release + "-policy-controller"
    inside = render("policyController.enabled=true", "global.image.tag=1.16.0-slurm-rc.1")
    assert container(inside)["image"] == router_image(inside) + "-policy-controller"


def published(workflow: str, environment: dict[str, str], tmp_path: Path) -> dict[str, list[str]]:
    """The lists of image references a publishing workflow's own tags step
    writes for one run, the step run under bash: each multi-line output, by
    its name."""
    document = yaml.safe_load((IMAGE_WORKFLOW.parent / workflow).read_text())
    (step,) = [s for s in document["jobs"]["build-and-push"]["steps"] if s.get("id") == "tags"]
    output = tmp_path / "output"
    output.write_text("")
    ran = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        env={"PATH": "/usr/bin:/bin", "GITHUB_OUTPUT": str(output), "GITHUB_SHA": "0" * 40}
        | environment,
        capture_output=True,
        text=True,
    )
    assert ran.returncode == 0, f"the tags step of {workflow} failed:\n{ran.stderr}"
    lists: dict[str, list[str]] = {}
    lines = output.read_text().splitlines()
    while lines:
        name, mark, end = lines.pop(0).partition("<<")
        if mark:
            stop = lines.index(end)
            lists[name], lines = lines[:stop], lines[stop + 1 :]
    return lists


@needs_helm
@pytest.mark.parametrize("run", list(PUBLISHING_RUNS))
def test_every_published_router_tag_names_a_published_controller(run: str, tmp_path: Path):
    """For every tag a run publishes the router's image under, its own list
    and each variant's, the chart's default names a controller image the
    same run publishes: the workflow's own tags step against the render. A
    variant a workflow gains and the helper does not know is red here."""
    lists = published(*PUBLISHING_RUNS[run], tmp_path)
    (of_the_controller,) = [name for name in lists if name.startswith("controller")]
    routers = [tag for name in lists if name != of_the_controller for tag in lists[name]]
    assert routers, "the tags step wrote no router tag"
    for reference in routers:
        path, _, tag = reference.rpartition(":")
        registry, repository = path.split("/", 1) if path.count("/") == 2 else ("docker.io", path)
        documents = render(
            "policyController.enabled=true",
            f"global.image.registry={registry}",
            f"global.image.repository={repository}",
            f"global.image.tag={tag}",
        )
        assert router_image(documents).removeprefix("docker.io/") == reference
        image = container(documents)["image"].removeprefix("docker.io/")
        assert image in lists[of_the_controller], f"{reference}: {image} is not published"


@needs_helm
def test_an_image_the_suffix_cannot_name_is_refused():
    """The default tag belongs to the router's repository and stands on the
    router's tag: a repository of the operator's own needs its tag, and a
    router named by a digest leaves the suffix nothing to stand on."""
    on = "policyController.enabled=true"
    error = render_error(on, "policyController.image.repository=example/controller")
    assert "policyController.image.repository needs policyController.image.tag" in error
    digest = "global.image.tag=1.16.0@sha256:" + "0" * 64
    error = render_error(on, digest)
    assert "policyController.image.tag is needed" in error
    assert "global.image.tag" in error
    named = render(on, digest, "policyController.image.tag=1.16.0-policy-controller")
    assert container(named)["image"] == "docker.io/lightseekorg/smg:1.16.0-policy-controller"


@needs_helm
def test_a_release_older_than_the_first_with_a_controller_image_is_refused():
    """No release older than the first that publishes a controller image has
    one, and a release tag says which release it is: under such a tag the
    chart refuses where its default would name a tag no registry holds, the
    variant's suffix on it or not, until the controller's tag is named. The
    README's table names the same first release. From that release on the
    default stands, and under a tag that is no release's: only the three
    numbers a release tag starts with are compared, so no tag fails the
    render for its spelling."""
    on = "policyController.enabled=true"
    row = r"-policy-controller`, beside every release image from (\S+) on"
    (first,) = re.findall(row, (CHART / "README.md").read_text())
    for older in ("1.15.2", "1.15.2-slurm", "1.15.99-rc.1", "1.9.0", "0.9.0", "1.15.2-my_build"):
        router = f"global.image.tag={older}"
        error = render_error(on, router)
        assert f"image.tag is needed: global.image.tag {older} is older than {first}" in error
        named = render(on, router, "policyController.image.tag=mine")
        assert container(named)["image"] == "docker.io/lightseekorg/smg:mine"
        assert not of_the_controller(render(router))
    newer = (first, f"{first}-rc.1", f"{first}-dev", f"{first}-my_build", f"{first}-0115")
    no_release = ("nightly", "v1.15.2", "1.15", "1.15.2.1", "build-1.15.2", "9" * 20 + ".0.0")
    for tag in newer + ("1.100.0", "2.0.0") + no_release:
        documents = render(on, f"global.image.tag={tag}")
        assert container(documents)["image"] == router_image(documents) + "-policy-controller", tag


@needs_helm
def test_the_block_renders_two_objects_and_the_election_four_more():
    """No ConfigMap without a spec document. The election's four objects
    stand behind the election's switch."""
    documents = of_the_controller(render(*ON))
    assert sorted(document["kind"] for document in documents) == ["Deployment", "Service"]
    assert {document["metadata"]["name"] for document in documents} == {NAME}

    elected = of_the_controller(render(*ELECTED))
    assert sorted(document["kind"] for document in elected) == [
        "Deployment",
        "Lease",
        "Role",
        "RoleBinding",
        "Service",
        "ServiceAccount",
    ]
    assert {document["metadata"]["name"] for document in elected} == {NAME}

    switched = of_the_controller(
        render(
            *ON,
            "router.metrics.serviceMonitor.enabled=true",
            "policyController.podDisruptionBudget.enabled=true",
        )
    )
    assert sorted(document["kind"] for document in switched) == [
        "Deployment",
        "PodDisruptionBudget",
        "Service",
        "ServiceMonitor",
    ]


@needs_helm
def test_a_spec_document_renders_its_configmap_its_mount_and_its_flag(tmp_path):
    """A document in the values is the one key of one ConfigMap under the
    controller's name. The pod mounts that ConfigMap whole and read-only,
    and the controller is passed the key's file under the mount. A null and
    an empty map are no document: none of the three is rendered."""
    documents = render(*ON, files=(spec_values(tmp_path, LEAST_LOAD),))
    assert sorted(document["kind"] for document in of_the_controller(documents)) == [
        "ConfigMap",
        "Deployment",
        "Service",
    ]
    config = one(documents, "ConfigMap")
    assert component(config) == COMPONENT
    assert list(config["data"]) == [Path(SPEC_FILE).name]
    assert yaml.safe_load(config["data"]["spec.yaml"]) == LEAST_LOAD
    pod = one(documents, "Deployment")["spec"]["template"]["spec"]
    found = container(documents)
    # The whole ConfigMap, whose updates the kubelet projects (a subPath gets
    # none), and not optional: the pod waits for it.
    assert pod["volumes"] == [{"name": "policy-spec", "configMap": {"name": NAME}}]
    assert found["volumeMounts"] == [
        {"name": "policy-spec", "mountPath": SPEC_MOUNT, "readOnly": True}
    ]
    # The flag once, after the chart's other flags.
    assert found["args"][-2:] == ["--spec-file", SPEC_FILE]
    assert found["args"].count("--spec-file") == 1
    # The read-only root filesystem stands: the mount is a volume of its own.
    assert found["securityContext"]["readOnlyRootFilesystem"] is True
    # A null and an empty map are no document: the render is the one of no
    # spec value, byte for byte.
    plain = helm_template(ON)
    assert plain.returncode == 0, plain.stderr
    assert helm_template((*ON, "policyController.spec=null")).stdout == plain.stdout
    no_document = spec_values(tmp_path, {}, "empty.yaml")
    assert helm_template(ON, files=(no_document,)).stdout == plain.stdout


def without_the_document(documents: list[dict]) -> list[dict]:
    """Every rendered object but the spec document's ConfigMap."""
    config = one(documents, "ConfigMap")
    return [document for document in documents if document is not config]


@needs_helm
@pytest.mark.parametrize("switches", [ON, ELECTED], ids=["alone", "elected"])
def test_a_changed_document_changes_its_configmap_and_no_pod(switches, tmp_path):
    """The pod template holds no checksum of the document. An upgrade to
    another document renders another ConfigMap and every other object as it
    was, so it moves no pod: the kubelet turns the mounted file and the
    running controller reads it. A first document does move the pod: the
    mount and the flag arrive with it."""
    first = render(*switches, files=(spec_values(tmp_path, LEAST_LOAD, "first.yaml"),))
    second = render(*switches, files=(spec_values(tmp_path, RANDOM, "second.yaml"),))
    assert one(first, "ConfigMap")["data"] != one(second, "ConfigMap")["data"]
    assert without_the_document(first) == without_the_document(second)
    assert "annotations" not in one(first, "Deployment")["spec"]["template"]["metadata"]
    assert one(first, "Deployment") != one(render(*switches), "Deployment")


@needs_helm
def test_the_document_reaches_the_file_as_the_values_hold_it(tmp_path):
    """The chart reads no line of the document and writes none, so the file
    reads back as the map the values hold. The map here is no document a
    controller runs: it asks what the chart carries. Text that reads as a
    number, a boolean or a null stays text; a list keeps its order; a null,
    an empty list and an empty map stay what they are; a text of several
    lines keeps its empty line; braces are no template. No generation line
    is written, and one the document writes is handed on."""
    carried = {
        **LEAST_LOAD,
        "order": ["c", "a", "b"],
        "text": {
            "number": "123",
            "boolean": "true",
            "null": "null",
            "lines": "first\n\nthird\n",
            "braces": "{{ .Values.none }}",
        },
        "numbers": {"whole": 67108864, "ratio": 0.3},
        "nothing": None,
        "empty": {"list": [], "map": {}},
    }

    def file(document: dict) -> str:
        documents = render(*ON, files=(spec_values(tmp_path, document),))
        return one(documents, "ConfigMap")["data"]["spec.yaml"]

    written = file(carried)
    assert yaml.safe_load(written) == carried
    numbered = file({**carried, "generation": 7})
    assert yaml.safe_load(numbered) == {**carried, "generation": 7}
    # By value a whole number equals its float form. The text of its line
    # holds the form: the controller's reader takes no float there.
    assert "\n  whole: 67108864\n" in written
    assert "\ngeneration: 7\n" in numbered


@needs_helm
@pytest.mark.parametrize("written", ["false", "0", '""', "[]", '"apiVersion: smg.policy/v1"'])
def test_a_document_that_is_no_map_is_refused_though_it_reads_as_empty(
    written: str, tmp_path: Path
):
    """With the schema's check skipped, a document that is no map is refused
    as it was written. One that reads as empty never renders as no document,
    and a document handed as one text is not written into the file as a
    text: null and the empty map alone are no document."""
    values = tmp_path / "values.yaml"
    values.write_text(f"policyController:\n  spec: {written}\n")
    refused = render_error(*ON, files=(values,), skip_schema=True)
    assert f"policyController.spec is {written}: the policy spec document as a map" in refused
    # The schema refuses each first, by its type.
    assert "policyController.spec: Invalid type" in render_error(*ON, files=(values,))


@needs_helm
def test_no_other_selector_takes_the_controllers_pods():
    documents = render(
        *ON,
        "router.metrics.serviceMonitor.enabled=true",
        "router.podDisruptionBudget.enabled=true",
        "policyController.podDisruptionBudget.enabled=true",
        "router.adminPort=9001",
    )
    labels = one(documents, "Deployment")["spec"]["template"]["metadata"]["labels"]
    assert labels["app.kubernetes.io/component"] == COMPONENT
    takers = []
    for document in documents:
        spec = document.get("spec", {})
        selector = spec.get("selector")
        if document["kind"] == "Service":
            matcher = selector
        elif isinstance(selector, dict) and "matchLabels" in selector:
            matcher = selector["matchLabels"]
        else:
            continue
        if selects(matcher, labels):
            takers.append((document["kind"], document["metadata"]["name"]))
    assert sorted(takers) == [
        ("Deployment", NAME),
        ("PodDisruptionBudget", NAME),
        ("Service", NAME),
        ("ServiceMonitor", NAME),
    ]
    router = one(documents, "Deployment", "smg-router")["spec"]["template"]["metadata"]["labels"]
    assert not selects(one(documents, "Service")["spec"]["selector"], router)


@needs_helm
def test_the_controllers_selectors_take_the_controller_and_nothing_else():
    """The Service and the budget select pods and the monitor selects
    Services: each takes the controller's and no other of a release that also
    has a router and a worker. A second budget over a router pod makes its
    eviction fail; a monitor over every Service scrapes the router and the
    workers twice."""
    documents = render(
        *ON,
        # The example's router is older than the first controller image.
        "policyController.image.tag=1.16.0-policy-controller",
        "router.metrics.serviceMonitor.enabled=true",
        "router.podDisruptionBudget.enabled=true",
        "policyController.podDisruptionBudget.enabled=true",
        "router.adminPort=9001",
        files=(EXAMPLES / "with-vllm-http.yaml",),
    )
    pods = {
        document["metadata"]["name"]: document["spec"]["template"]["metadata"]["labels"]
        for document in documents
        if document["kind"] == "Deployment"
    }
    services = {
        document["metadata"]["name"]: document["metadata"]["labels"]
        for document in documents
        if document["kind"] == "Service"
    }
    assert len(pods) == 3 and len(services) == 4, "a router and a worker beside the controller"
    selectors = {
        "Service": (one(documents, "Service")["spec"]["selector"], pods),
        "PodDisruptionBudget": (
            one(documents, "PodDisruptionBudget")["spec"]["selector"]["matchLabels"],
            pods,
        ),
        "ServiceMonitor": (
            one(documents, "ServiceMonitor")["spec"]["selector"]["matchLabels"],
            services,
        ),
    }
    for kind, (selector, candidates) in selectors.items():
        taken = sorted(name for name, labels in candidates.items() if selects(selector, labels))
        assert taken == [NAME], f"the controller's {kind} selects {taken}"


@needs_helm
def test_the_pod_runs_under_the_routers_security_context_and_its_own_probes():
    documents = render(*ON)
    pod = one(documents, "Deployment")["spec"]["template"]["spec"]
    found = container(documents)
    (router,) = one(documents, "Deployment", "smg-router")["spec"]["template"]["spec"]["containers"]
    assert found["securityContext"] == router["securityContext"]
    assert found["securityContext"]["readOnlyRootFilesystem"] is True
    assert found["securityContext"]["runAsUser"] == 65532
    # The controller calls no Kubernetes API: no token is mounted.
    assert pod["automountServiceAccountToken"] is False
    assert "serviceAccountName" not in pod
    assert "volumes" not in pod and "volumeMounts" not in found
    assert found["livenessProbe"]["httpGet"] == {"path": "/liveness", "port": "admin"}
    assert found["readinessProbe"]["httpGet"] == {"path": "/readiness", "port": "admin"}
    assert {port["name"]: port["containerPort"] for port in found["ports"]} == {
        "gateway": 30100,
        "control": 30101,
        "replication": 30102,
        "admin": 30103,
    }
    # The controller's grace period and the margin over it.
    assert pod["terminationGracePeriodSeconds"] == 30 + 10


@needs_helm
def test_the_arguments_follow_the_values():
    assert controller_args() == [
        "--gateway-listen",
        "0.0.0.0:30100",
        "--control-listen",
        "0.0.0.0:30101",
        "--replication-listen",
        "0.0.0.0:30102",
        "--admin-listen",
        "0.0.0.0:30103",
        "--shutdown-grace-period-secs",
        "30",
        "--runtime-worker-threads",
        "2",
        "--log-level",
        "info",
    ]
    documents = render(
        *ON,
        "policyController.ports.gateway=31100",
        "policyController.ports.control=31101",
        "policyController.ports.replication=31102",
        "policyController.ports.admin=31103",
        "policyController.identityAsTold={gateway,control}",
        "policyController.shutdownGracePeriodSeconds=60",
        "policyController.shutdownSettleSeconds=20",
        "policyController.terminationGraceMarginSeconds=15",
        "policyController.runtimeWorkerThreads=8",
        "policyController.logging.level=debug",
        "policyController.logging.json=true",
        "policyController.extraArgs={--ingestion-listen,0.0.0.0:31104,--load-monitor-interval,2}",
    )
    assert container(documents)["args"] == [
        "--gateway-listen",
        "0.0.0.0:31100",
        "--control-listen",
        "0.0.0.0:31101",
        "--replication-listen",
        "0.0.0.0:31102",
        "--admin-listen",
        "0.0.0.0:31103",
        "--identity-as-told",
        "gateway,control",
        "--shutdown-grace-period-secs",
        "60",
        "--shutdown-settle-secs",
        "20",
        "--runtime-worker-threads",
        "8",
        "--log-level",
        "debug",
        "--log-json",
        "--ingestion-listen",
        "0.0.0.0:31104",
        "--load-monitor-interval",
        "2",
    ]
    pod = one(documents, "Deployment")["spec"]["template"]["spec"]
    assert pod["terminationGracePeriodSeconds"] == 60 + 15
    # The controller's id is the pod's name, which the flag reads from
    # HOSTNAME: the chart passes neither the flag nor the variable.
    assert "--controller-id" not in container(documents)["args"]
    assert "env" not in container(documents)
    # 0 leaves the flag out: the controller then starts one worker thread
    # for every CPU the pod may use.
    unset = controller_args("policyController.runtimeWorkerThreads=0")
    assert "--runtime-worker-threads" not in unset
    # A settle window of zero is a window: the flag is passed, not left to
    # the controller's default.
    zero = controller_args("policyController.shutdownSettleSeconds=0")
    assert zero[zero.index("--shutdown-settle-secs") + 1] == "0"
    # No listener takes a caller's stated name unless the values name it: an
    # empty list and a null pass no flag, and the controller then names none.
    assert "--identity-as-told" not in controller_args()
    assert "--identity-as-told" not in controller_args("policyController.identityAsTold=null")
    told = controller_args("policyController.identityAsTold={replication}")
    assert told[told.index("--identity-as-told") + 1] == "replication"
    # The value is a set: it renders in the listeners' own order however it
    # was written, so a reorder rolls no pod.
    reordered = controller_args("policyController.identityAsTold={replication,gateway}")
    assert reordered[reordered.index("--identity-as-told") + 1] == "gateway,replication"


@needs_helm
def test_the_values_the_chart_hands_on_reach_their_objects(tmp_path):
    """The values the block hands on unread, set through one values file and
    read in the object each lands in."""
    handed_on = {
        "replicas": 3,
        "resources": {"limits": {"cpu": "2"}},
        "nodeSelector": {"pool": "control"},
        "tolerations": [{"key": "dedicated", "operator": "Exists"}],
        "podAnnotations": {"example.com/scrape": "true"},
        "podLabels": {"team": "routing"},
        "extraEnv": [{"name": "EXAMPLE", "value": "1"}],
        "securityContext": {"runAsUser": 1000},
        "service": {"type": "NodePort", "annotations": {"example.com/zone": "a"}},
        "podDisruptionBudget": {"enabled": True, "minAvailable": 2},
        "livenessProbe": {"initialDelaySeconds": 7, "periodSeconds": 11},
        "readinessProbe": {"initialDelaySeconds": 3, "periodSeconds": 13},
    }
    values = tmp_path / "values.yaml"
    values.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "image": {"pullPolicy": "Always"},
                    "imagePullSecrets": [{"name": "registry-key"}],
                },
                "policyController": handed_on,
            }
        )
    )
    # Three replicas render with the election on alone.
    documents = render(*ELECTED, files=(values,))
    deployment = one(documents, "Deployment")["spec"]
    pod = deployment["template"]["spec"]
    found = container(documents)
    service = one(documents, "Service")
    assert deployment["replicas"] == 3
    assert found["resources"] == handed_on["resources"]
    assert pod["nodeSelector"] == handed_on["nodeSelector"]
    assert pod["tolerations"] == handed_on["tolerations"]
    assert deployment["template"]["metadata"]["annotations"] == handed_on["podAnnotations"]
    assert deployment["template"]["metadata"]["labels"]["team"] == "routing"
    assert found["env"] == [POD_IP, *handed_on["extraEnv"]]
    # A map of the values file joins the default's: the other keys stay.
    assert found["securityContext"]["runAsUser"] == 1000
    assert found["securityContext"]["readOnlyRootFilesystem"] is True
    assert found["imagePullPolicy"] == "Always"
    assert pod["imagePullSecrets"] == [{"name": "registry-key"}]
    assert service["spec"]["type"] == "NodePort"
    assert service["metadata"]["annotations"] == handed_on["service"]["annotations"]
    assert one(documents, "PodDisruptionBudget")["spec"]["minAvailable"] == 2
    for probe in ("livenessProbe", "readinessProbe"):
        for number, value in handed_on[probe].items():
            assert found[probe][number] == value, f"{probe}.{number}"


@needs_helm
def test_the_service_carries_three_of_the_four_listeners():
    documents = render(*ON, "policyController.ports.admin=31103")
    service = one(documents, "Service")["spec"]
    assert service["type"] == "ClusterIP"
    assert [(p["name"], p["port"], p["targetPort"]) for p in service["ports"]] == [
        ("grpc", 30100, "gateway"),
        ("control", 30101, "control"),
        ("metrics", 31103, "admin"),
    ]
    monitored = render(*ON, "router.metrics.serviceMonitor.enabled=true")
    (endpoint,) = one(monitored, "ServiceMonitor")["spec"]["endpoints"]
    assert endpoint["port"] == "metrics"


@needs_helm
def test_replicas_prefer_a_node_each_unless_an_affinity_is_given():
    single = one(render(*ON), "Deployment")["spec"]
    assert single["replicas"] == 1
    assert "affinity" not in single["template"]["spec"]

    pair = one(render(*ELECTED, "policyController.replicas=2"), "Deployment")["spec"]
    (term,) = pair["template"]["spec"]["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    assert term["podAffinityTerm"]["topologyKey"] == "kubernetes.io/hostname"
    assert (
        term["podAffinityTerm"]["labelSelector"]["matchLabels"] == pair["selector"]["matchLabels"]
    )

    own = one(
        render(
            *ELECTED,
            "policyController.replicas=2",
            "policyController.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution"
            ".nodeSelectorTerms[0].matchExpressions[0].key=zone",
            "policyController.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution"
            ".nodeSelectorTerms[0].matchExpressions[0].operator=Exists",
        ),
        "Deployment",
    )["spec"]["template"]["spec"]["affinity"]
    assert list(own) == ["nodeAffinity"]

    # With the schema's check skipped, a count given as text that the
    # manifest's reader takes as two prefers a node each as the number does.
    for count in ("2e0", " 2"):
        text = one(
            render(*ELECTED, f"policyController.replicas={count}", skip_schema=True), "Deployment"
        )["spec"]
        assert text["template"]["spec"]["affinity"] == pair["template"]["spec"]["affinity"], count


@needs_helm
def test_the_election_names_its_lease_and_the_pods_own_address():
    """On, the controller is passed the Lease's name and its pod's address;
    the variable the address is read from is the chart's, ahead of the
    operator's. No flag names a namespace: the controller's is its pod's."""
    assert controller_args("policyController.election.enabled=true")[-4:] == [
        "--lease-name",
        NAME,
        "--advertise-address",
        "$(POD_IP)",
    ]
    found = container(
        render(
            *ELECTED,
            "policyController.election.leaseName=routing",
            "policyController.extraEnv[0].name=EXAMPLE",
            "policyController.extraEnv[0].value=set",
            "policyController.extraArgs={--lease-duration-ms,2000}",
        )
    )
    assert found["args"][-6:] == [
        "--lease-name",
        "routing",
        "--advertise-address",
        "$(POD_IP)",
        "--lease-duration-ms",
        "2000",
    ]
    assert found["env"] == [POD_IP, {"name": "EXAMPLE", "value": "set"}]
    assert "--lease-namespace" not in found["args"]


def rest(documents: list[dict]) -> list[dict]:
    """What the chart renders beside the controller's own objects."""
    own = of_the_controller(documents)
    return [document for document in documents if document not in own]


@needs_helm
def test_the_election_grants_the_controller_its_one_lease_and_nothing_else():
    """The Role holds a read and a replace of the one Lease the chart
    renders: no create, no list, no watch, no other name. It is bound to the
    controller's own account, whose token is mounted in the controller's
    pods alone; every other object of the release is the same with the
    election on and off."""
    off, on = render(*ON), render(*ELECTED)
    assert rest(on) == rest(off)
    assert one(on, "Role")["rules"] == [
        {
            "apiGroups": ["coordination.k8s.io"],
            "resources": ["leases"],
            "resourceNames": [NAME],
            "verbs": ["get", "update"],
        }
    ]
    binding = one(on, "RoleBinding")
    assert binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": NAME,
    }
    assert binding["subjects"] == [{"kind": "ServiceAccount", "name": NAME, "namespace": "default"}]
    assert one(on, "ServiceAccount")["metadata"]["name"] == NAME
    pod = one(on, "Deployment")["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == pod["serviceAccount"] == NAME
    assert pod["automountServiceAccountToken"] is True
    for document in rest(on):
        other = document.get("spec", {}).get("template", {}).get("spec", {})
        assert other.get("serviceAccountName") != NAME, document["metadata"]["name"]
    # Off, the pod names no account and mounts no token.
    lone = one(off, "Deployment")["spec"]["template"]["spec"]
    assert "serviceAccountName" not in lone and "serviceAccount" not in lone
    assert lone["automountServiceAccountToken"] is False
    # A name of the operator's own is the Lease's, the grant's and the flag's.
    named = render(*ELECTED, "policyController.election.leaseName=routing")
    assert one(named, "Lease", "routing")["metadata"]["name"] == "routing"
    assert one(named, "Role")["rules"][0]["resourceNames"] == ["routing"]
    assert "routing" in container(named)["args"]
    # A name no Lease can carry is refused at render: the cluster reads a
    # name label by label.
    for name in ("Routing", "routing_a", "-routing", "r" * 254, "a..b", "a.-b", "a-.b"):
        assert "policyController.election.leaseName" in render_error(
            *ELECTED, f"policyController.election.leaseName={name}"
        )
    for name in ("a.b", "routing-2.smg"):
        named = render(*ELECTED, f"policyController.election.leaseName={name}")
        assert one(named, "Lease", name)["metadata"]["name"] == name


@needs_helm
def test_turning_the_election_off_takes_the_account_off_the_live_pod_template():
    """The server fills `serviceAccount`, the deprecated alias of `serviceAccountName`, on the live object and reads the
    name back from the alias when the name is gone. An upgrade or a rollback deletes only what the old manifest named
    and the new one does not. The elected manifest names both, so with the election off neither stands."""
    elected = one(render(*ELECTED), "Deployment")["spec"]["template"]["spec"]
    lone = one(render(*ON), "Deployment")["spec"]["template"]["spec"]
    live = {**elected, "serviceAccount": elected["serviceAccountName"]}
    for field in elected.keys() - lone.keys():
        del live[field]
    live.update(lone)
    assert "serviceAccountName" not in live
    assert "serviceAccount" not in live


@needs_helm
def test_a_name_yaml_reads_as_a_number_a_boolean_or_null_stays_text(tmp_path):
    """A Lease may be named 123, true or null, and so may a namespace: text
    the cluster admits, which YAML reads as a number, a boolean or null
    where it is rendered bare. The Lease's name is the text given, the same
    as the grant's and the flag's, and the binding's subject is in the
    release's namespace by its text."""
    values = tmp_path / "values.yaml"
    for name in ("123", "007", "0x1f", "1.5", "true", "on", "no", "null"):
        values.write_text(yaml.safe_dump({"policyController": {"election": {"leaseName": name}}}))
        named = render(*ELECTED, files=(values,))
        assert one(named, "Lease", name)["metadata"]["name"] == name
        assert one(named, "Role")["rules"][0]["resourceNames"] == [name]
        args = container(named)["args"]
        assert args[args.index("--lease-name") + 1] == name
    for namespace in ("123", "on", "true", "routing"):
        (subject,) = one(render(*ELECTED, namespace=namespace), "RoleBinding")["subjects"]
        assert subject["namespace"] == namespace


@needs_helm
def test_the_rendered_lease_holds_no_record():
    """The Lease is rendered with its name and its labels and nothing else:
    the holder, the times and the term are the controllers', so an upgrade
    has nothing to say of them."""
    lease = one(render(*ELECTED), "Lease")
    assert set(lease) == {"apiVersion", "kind", "metadata"}
    assert lease["apiVersion"] == "coordination.k8s.io/v1"
    assert set(lease["metadata"]) == {"name", "labels"}


@needs_helm
def test_the_rollout_follows_the_election():
    """On, one pod more at a time and none unavailable: a new pod stands by
    before an old one stops. One elected replica is a leader with no
    standby: allowed."""
    elected = one(render(*ELECTED, "policyController.replicas=2"), "Deployment")["spec"]
    assert elected["strategy"] == {
        "type": "RollingUpdate",
        "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0},
    }
    assert one(render(*ELECTED), "Deployment")["spec"]["replicas"] == 1


@needs_helm
def test_with_the_election_off_an_upgrade_tells_the_old_pod_to_stop_first():
    """Off, every pod leads alone, and the server's own strategy starts the
    new pod beside the old one: two leaders until the new one is ready. The
    lone controller rolls with no pod more at a time. Its strategy keeps the
    rolling update's type and names both of its fields, as the elected one
    does. The API refuses those fields beside Recreate, and a server-side
    apply takes off a live object only what it put there: the server's own,
    on a Deployment that named no strategy, would stay. So no upgrade
    between the three strategies has a field to take off."""
    for off in OFF:
        assert one(render(*ON, *off), "Deployment")["spec"]["strategy"] == LONE_STRATEGY, off
    served = {
        "type": "RollingUpdate",
        "rollingUpdate": {"maxSurge": "25%", "maxUnavailable": "25%"},
    }
    elected = one(render(*ELECTED), "Deployment")["spec"]["strategy"]
    lone = one(render(*ON), "Deployment")["spec"]["strategy"]
    for rendered in (lone, elected):
        for live in (served, lone, elected):
            # An apply that takes nothing off: a live field the manifest
            # does not name stays beside the ones it names.
            fields = {**live["rollingUpdate"], **rendered["rollingUpdate"]}
            applied = {**live, **rendered, "rollingUpdate": fields}
            assert applied == rendered, (rendered, live)
            assert applied["type"] == "RollingUpdate", (rendered, live)


@needs_helm
def test_no_hook_delays_the_stop_signal():
    """The lone controller's order rests on the stop signal reaching the old
    pod at its delete: the old pod decides nothing from that signal on. A
    preStop hook runs before the signal, and the old pod would lead beside
    the new one for the hook's length: the container names no hook, with
    the election on or off."""
    for values in (ON, ELECTED):
        assert "lifecycle" not in container(render(*values)), values


@needs_helm
def test_with_the_election_off_nothing_of_it_is_rendered():
    """A release that does not turn the election on renders, in each spelling
    of off, no object of the election, and on the controller's Deployment
    of one replica the lone controller's strategy, no account, no variable
    and neither flag."""
    for off in OFF:
        documents = render(*ON, *off)
        kinds = sorted(document["kind"] for document in of_the_controller(documents))
        assert kinds == ["Deployment", "Service"], off
        deployment = one(documents, "Deployment")["spec"]
        assert deployment["replicas"] == 1, off
        assert deployment["strategy"] == LONE_STRATEGY, off
        pod = deployment["template"]["spec"]
        assert "serviceAccountName" not in pod, off
        assert "serviceAccount" not in pod, off
        assert pod["automountServiceAccountToken"] is False, off
        lone = container(documents)
        assert "env" not in lone, off
        assert not {"--lease-name", "--advertise-address"} & set(lone["args"]), off


@needs_helm
def test_more_than_one_replica_needs_the_election(tmp_path):
    """With the election off every replica is a controller of its own: two
    would each lead alone. The render refuses them in each spelling of off
    and says the two ways out; with the election on they render. The
    refusal reads the count as written, and judges it as a reader of the
    manifest takes it: a number from a values file, a fraction above one,
    and a text that reads there as a number above one."""
    for off in OFF:
        for replicas in (2, 3):
            refused = render_error(*ON, f"policyController.replicas={replicas}", *off)
            assert (
                f"policyController.replicas is {replicas} with policyController.election.enabled"
                " off: each replica would lead alone. Turn the election on, or keep one replica"
            ) in refused, off
    pair = one(render(*ELECTED, "policyController.replicas=2"), "Deployment")["spec"]
    assert pair["replicas"] == 2
    # A number from a values file, whole or a fraction above one: the
    # refusal reads it as written.
    values = tmp_path / "values.yaml"
    for replicas, skip_schema in ((2, False), (2.5, True)):
        values.write_text(yaml.safe_dump({"policyController": {"replicas": replicas}}))
        refused = render_error(*ON, files=(values,), skip_schema=skip_schema)
        assert (
            f"policyController.replicas is {replicas} with policyController.election.enabled off"
        ) in refused, replicas
    # A text the manifest's reader takes as a number above one is refused as
    # that number is. With the election on it is written as asked, for the
    # API to judge.
    for count in ("2e0", " 2"):
        asked = f"policyController.replicas={count}"
        refused = render_error(*ON, asked, skip_schema=True)
        assert (
            f"policyController.replicas is {count} with policyController.election.enabled off"
        ) in refused, count
        written = one(render(*ELECTED, asked, skip_schema=True), "Deployment")["spec"]["replicas"]
        assert written == yaml.safe_load(f"count: {count}")["count"], count


@needs_helm
def test_a_replica_count_the_schema_refuses_is_written_as_asked(tmp_path):
    """With the schema's check skipped the helper compares the count and
    writes the value as the values hold it. A null stays the API's default
    of one replica and a fraction, as text or as a number, stays for the API
    to refuse: neither is written as zero or as its whole part."""
    unset = render(*ON, "policyController.replicas=null", skip_schema=True)
    assert one(unset, "Deployment")["spec"]["replicas"] is None
    fraction = render(*ON, "policyController.replicas=1.5", skip_schema=True)
    assert one(fraction, "Deployment")["spec"]["replicas"] == 1.5
    values = tmp_path / "values.yaml"
    values.write_text(yaml.safe_dump({"policyController": {"replicas": 1.5}}))
    number = render(*ON, files=(values,), skip_schema=True)
    assert one(number, "Deployment")["spec"]["replicas"] == 1.5


@pytest.fixture
def chart_with_no_election(tmp_path) -> Path:
    """A copy of the chart whose controller block holds no election: what an
    upgrade that reuses the values of a release from before the election
    hands the templates."""
    values = yaml.safe_load(VALUES.read_text())
    del values["policyController"]["election"]
    chart = tmp_path / "smg"
    shutil.copytree(CHART, chart)
    (chart / "values.yaml").write_text(yaml.safe_dump(values))
    return chart


@needs_helm
def test_the_election_over_reused_values_renders_a_fresh_install(chart_with_no_election):
    """Over values that hold no election the controller renders what a fresh
    install renders, byte for byte: off with nothing set, and on at one
    replica and at two, with the Lease under the controller's own name or a
    given one."""
    cells = (
        ON,
        ELECTED,
        (*ELECTED, "policyController.replicas=2"),
        (*ELECTED, "policyController.election.leaseName=routing"),
    )
    for settings in cells:
        fresh = helm_template(settings)
        assert fresh.returncode == 0, fresh.stderr
        reused = helm_template(settings, chart=chart_with_no_election)
        assert reused.returncode == 0, reused.stderr
        assert reused.stdout == fresh.stdout, settings


@needs_helm
def test_the_chart_refuses_what_the_controller_refuses_at_start():
    shared = render_error(*ON, "policyController.ports.control=30100")
    assert "policyController.ports.control 30100 equals policyController.ports.gateway" in shared
    assert "policyController.ports.admin 30102 equals policyController.ports.replication" in (
        render_error(*ON, "policyController.ports.admin=30102")
    )

    over = render_error(
        *ON,
        "policyController.shutdownGracePeriodSeconds=30",
        "policyController.shutdownSettleSeconds=16",
    )
    assert "a settle window of 16 s is over half the grace period of 30 s" in over
    # With no window of the chart's, the controller's own default counts.
    default = render_error(
        *ON, f"policyController.shutdownGracePeriodSeconds={2 * DEFAULT_SETTLE_SECONDS - 1}"
    )
    assert f"a settle window of {DEFAULT_SETTLE_SECONDS} s is over half" in default
    # Half the grace period is the bound, and a window of zero is a window.
    render(*ON, f"policyController.shutdownGracePeriodSeconds={2 * DEFAULT_SETTLE_SECONDS}")
    render(
        *ON,
        "policyController.shutdownGracePeriodSeconds=1",
        "policyController.shutdownSettleSeconds=0",
    )
    # A name that is none of the chart's three listeners is refused by the
    # schema, and by the helper where the schema's check is skipped: it never
    # renders as no flag at all.
    for names in ("{admin}", "{ingestion}", "{gateway,admin}"):
        unknown = render_error(*ON, f"policyController.identityAsTold={names}", skip_schema=True)
        assert "policyController.identityAsTold names" in unknown
        assert "none of the chart's listeners (gateway, control, replication)" in unknown
    assert 'policyController.identityAsTold is "gateway"' in render_error(
        *ON, "policyController.identityAsTold=gateway", skip_schema=True
    )
    # The names the schema takes render the same with its check skipped.
    told = container(
        render(*ON, "policyController.identityAsTold={replication,gateway}", skip_schema=True)
    )["args"]
    assert told[told.index("--identity-as-told") + 1] == "gateway,replication"
    # A null is the value left out, with the check skipped too.
    unset = render(*ON, "policyController.identityAsTold=null", skip_schema=True)
    assert "--identity-as-told" not in container(unset)["args"]


@needs_helm
@pytest.mark.parametrize("written", ["false", "0", '""', "{}"])
def test_a_told_value_that_is_no_list_is_refused_though_it_reads_as_empty(
    written: str, tmp_path: Path
):
    """With the schema's check skipped, a value that is no list is refused as
    it was written. One that reads as empty never renders as no listener
    told: null alone is the value left out."""
    values = tmp_path / "values.yaml"
    values.write_text(f"policyController:\n  identityAsTold: {written}\n")
    refused = render_error(*ON, files=(values,), skip_schema=True)
    assert f"policyController.identityAsTold is {written}: a list of listeners' names" in refused
    # The schema refuses each first.
    assert "policyController.identityAsTold" in render_error(*ON, files=(values,))


def schema_block() -> dict:
    return json.loads(SCHEMA.read_text())["properties"]["policyController"]


def unlisted(values: dict, schema: dict, path: str) -> list[str]:
    """The keys of ``values`` the schema's ``properties`` do not list, and
    the listed ones ``values`` lacks; an object the schema leaves open (a
    resources block, a security context) is not walked."""
    properties = schema.get("properties")
    if properties is None:
        return []
    found = [
        f"{path}.{key} is in one of values.yaml and the schema only"
        for key in set(values) ^ set(properties)
    ]
    for key in set(values) & set(properties):
        if isinstance(values[key], dict):
            found += unlisted(values[key], properties[key], f"{path}.{key}")
    return found


def test_the_schema_lists_every_value_of_the_block_and_no_other():
    values = yaml.safe_load(VALUES.read_text())["policyController"]
    assert unlisted(values, schema_block(), "policyController") == []
    assert schema_block()["additionalProperties"] is False


@needs_helm
def test_the_schema_refuses_a_value_the_block_does_not_have():
    assert "replica" in render_error(*ON, "policyController.replica=2")
    assert "policyController.logging.level" in render_error(
        *ON, "policyController.logging.level=loud"
    )
    # The election's block holds its two values, and its switch is no text.
    assert "Additional property lease is not allowed" in render_error(
        *ON, "policyController.election.lease=routing"
    )
    assert "policyController.election.enabled" in render_error(
        *ON, "policyController.election.enabled=yes"
    )
    assert "policyController.replicas" in render_error(*ON, "policyController.replicas=0")
    for threads in (-1, 4097):
        assert "policyController.runtimeWorkerThreads" in render_error(
            *ON, f"policyController.runtimeWorkerThreads={threads}"
        )
    assert "policyController.ports.admin" in render_error(*ON, "policyController.ports.admin=65536")
    # A listener is named by one of the chart's three names, once, in a list:
    # the controller refuses a name that is no listener at its start.
    for names in ("{admin}", "{ingestion}", "{gateway,gateway}", "gateway"):
        assert "policyController.identityAsTold" in render_error(
            *ON, f"policyController.identityAsTold={names}"
        )
    # An empty path is no path: the kubelet would ask the root.
    for probe in ("livenessProbe", "readinessProbe"):
        assert f"policyController.{probe}.path" in render_error(
            *ON, f"policyController.{probe}.path="
        )
    # An empty Service type or budget number renders a field with no value,
    # which leaves it to the cluster.
    assert "policyController.service.type" in render_error(*ON, "policyController.service.type=")
    assert "policyController.podDisruptionBudget.minAvailable" in render_error(
        *ON,
        "policyController.podDisruptionBudget.enabled=true",
        "policyController.podDisruptionBudget.minAvailable=",
    )


def block_keys(values: dict, schema: dict, prefix: str = "") -> list[str]:
    """The keys of the block as the values file has them, with the members
    of each object whose members the schema lists."""
    keys = []
    for key, value in values.items():
        keys.append(f"{prefix}{key}")
        listed = schema["properties"].get(key, {})
        if "properties" in listed:
            keys += block_keys(value, listed, f"{prefix}{key}.")
    return keys


# A null removes a value's default. These nulls are legal and render what
# the defaults render: each default is empty, and an empty part of the image
# is the router's.
SAME_AS_THE_DEFAULT = {
    "image",
    "image.registry",
    "image.repository",
    "image.tag",
    "service.annotations",
    "logging.json",
    "resources",
    "nodeSelector",
    "tolerations",
    "affinity",
    "podAnnotations",
    "podLabels",
    "extraArgs",
    "extraEnv",
    "identityAsTold",
    "shutdownSettleSeconds",
    # No document: the controller runs no spec.
    "spec",
    # The election off, and the Lease under the controller's own name.
    "election",
    "election.enabled",
    "election.leaseName",
}
# These are legal too: each is a switch, and its null turns it off.
SWITCHED_OFF_BY_A_NULL = {
    "enabled": lambda documents: of_the_controller(documents) == [],
    "podDisruptionBudget.enabled": lambda documents: (
        "PodDisruptionBudget" not in [document["kind"] for document in documents]
    ),
}


def default_of(block: dict, key: str):
    for name in key.split("."):
        block = block[name]
    return block


def empty(default) -> bool:
    """An empty default: nothing, or a map of nothing but empty members."""
    if isinstance(default, dict):
        return all(empty(member) for member in default.values())
    return not default


@needs_helm
def test_a_null_is_refused_or_renders_what_this_case_names(chart_with_no_block):
    """Every key of the block set to null, one at a time, over the chart's
    own values and over values with no block where the block is given whole
    in a file: the schema refuses it, or it is one of the legal nulls above,
    the same over both. Without the refusals the chart
    would render a listener on port 0, a flag with no value, a probe with no
    path, a grace period of zero or a budget with no number, or leave the
    replica count, the worker threads, the grace margin, a probe's timing or
    the Service's type to another default. A key added to the block fails
    here until it is one or the other, and a legal null is for an empty
    default or a switch: a value that holds a default is refused."""
    block = yaml.safe_load(VALUES.read_text())["policyController"]
    for key in SAME_AS_THE_DEFAULT:
        assert empty(default_of(block, key)), f"{key} holds a default: refuse its null"
    for key in SWITCHED_OFF_BY_A_NULL:
        assert default_of(block, key) is False, f"{key} is no switch: refuse its null"
    switches = (*ON, "policyController.podDisruptionBudget.enabled=true")
    default = of_the_controller(render(*switches))
    # Over values with no block a null is not removed with a default: it
    # reaches the schema as a null, and a required value refuses it by type.
    whole = chart_with_no_block.parent / "block.yaml"
    whole.write_text(yaml.safe_dump({"policyController": block}))
    sides: dict[str, dict] = {
        "the chart's own values": {},
        "values with no block": {"files": (whole,), "chart": chart_with_no_block},
    }
    for side, over in sides.items():
        assert of_the_controller(render(*switches, **over)) == default, side
        for key in block_keys(block, schema_block()):
            setting = f"policyController.{key}=null"
            if key in SAME_AS_THE_DEFAULT:
                assert of_the_controller(render(*switches, setting, **over)) == default, (side, key)
            elif key in SWITCHED_OFF_BY_A_NULL:
                assert SWITCHED_OFF_BY_A_NULL[key](render(*switches, setting, **over)), (side, key)
            else:
                parent, _, name = f"policyController.{key}".rpartition(".")
                refusal = (
                    f"policyController.{key}: Invalid type"
                    if over
                    else f"{parent}: {name} is required"
                )
                error = render_error(*switches, setting, **over)
                assert refusal in error, f"{side}, {key}: {error}"


def every_key(values: dict, prefix: str = "") -> list[str]:
    """Every key of the block as the values file has it, at every depth."""
    keys = []
    for key, value in values.items():
        keys.append(f"{prefix}{key}")
        if isinstance(value, dict):
            keys += every_key(value, f"{prefix}{key}.")
    return keys


@needs_helm
def test_a_controller_that_is_off_refuses_no_null(chart_with_no_block):
    """With the controller off, every key of the block set to null, one at a
    time and at every depth, over the chart's own values and over values with
    no block, renders what the defaults render, byte for byte. A value is
    required only while the controller reads it: its own type takes a null,
    and the schema's condition holds the type while the controller is on."""
    block = yaml.safe_load(VALUES.read_text())["policyController"]
    default = helm_template(())
    assert default.returncode == 0, default.stderr
    sides = {"the chart's own values": CHART, "values with no block": chart_with_no_block}
    wrong = {}
    for side, chart in sides.items():
        for key in every_key(block):
            rendered = helm_template((f"policyController.{key}=null",), chart=chart)
            if rendered.returncode != 0:
                wrong[side, key] = rendered.stderr.strip().splitlines()[-1]
            elif rendered.stdout != default.stdout:
                wrong[side, key] = "renders other bytes than the defaults"
    assert wrong == {}


@pytest.fixture(scope="module")
def binary() -> str:
    """The controller binary the variable names: skipped without the
    variable, failed when it names no file."""
    path = os.environ.get(BINARY_VARIABLE, "")
    if not path:
        pytest.skip(f"set {BINARY_VARIABLE} to a built smg-policy-controller")
    assert Path(path).is_file(), f"{BINARY_VARIABLE} names no file: {path}"
    return path


def parse(binary: str, args: list[str]) -> subprocess.CompletedProcess:
    """The binary's own parse of ``args``: with ``--help`` last, clap reads
    every flag and value before it and exits without starting anything."""
    return subprocess.run([binary, *args, "--help"], capture_output=True, text=True, timeout=30)


# Each a set of values that changes what the Deployment passes.
ARGUMENT_SETS = {
    "defaults": (),
    "json-logging": ("policyController.logging.json=true", "policyController.logging.level=debug"),
    "a-settle-window": (
        "policyController.shutdownGracePeriodSeconds=60",
        "policyController.shutdownSettleSeconds=20",
    ),
    "no-settle-window": ("policyController.shutdownSettleSeconds=0",),
    "no-thread-count": ("policyController.runtimeWorkerThreads=0",),
    "other-ports": (
        "policyController.ports.gateway=31100",
        "policyController.ports.control=31101",
        "policyController.ports.replication=31102",
        "policyController.ports.admin=31103",
    ),
    "listeners-as-told": ("policyController.identityAsTold={gateway,control,replication}",),
    "extra-arguments": (
        "policyController.extraArgs={--ingestion-listen,0.0.0.0:30104,--load-monitor-interval,2}",
    ),
    "a-spec-document": ("policyController.spec.apiVersion=smg.policy/v1",),
    "an-election": ("policyController.election.enabled=true", "policyController.replicas=2"),
    "an-election-over-a-named-lease": (
        "policyController.election.enabled=true",
        "policyController.election.leaseName=routing",
        "policyController.extraArgs={--lease-duration-ms,2000,--lease-renew-period-ms,500,"
        "--lease-renew-deadline-ms,1250}",
    ),
}


@needs_helm
@pytest.mark.parametrize("name", sorted(ARGUMENT_SETS))
def test_the_binary_parses_every_rendered_argument(binary, name):
    args = controller_args(*ARGUMENT_SETS[name])
    parsed = parse(binary, args)
    assert parsed.returncode == 0, f"the controller refuses {args}: {parsed.stderr}"


@needs_helm
def test_the_binary_refuses_a_rendered_flag_spelled_wrong(binary):
    """The parse above has teeth: one flag of the rendered arguments spelled
    wrong, or one value the flag cannot read, and the binary refuses."""
    args = controller_args()
    assert parse(binary, args).returncode == 0
    for flag in [arg for arg in args if arg.startswith("--")]:
        wrong = [arg + "x" if arg == flag else arg for arg in args]
        refused = parse(binary, wrong)
        assert refused.returncode != 0, f"the controller took {flag}x"
        assert f"{flag}x" in refused.stderr
    unreadable = ["nowhere" if arg == "0.0.0.0:30103" else arg for arg in args]
    assert parse(binary, unreadable).returncode != 0


# The form of a workflow path filter's entry that `starts_the_job` reads: an
# optional leading ``!``, then names with ``*`` and a ``**`` that no slash
# follows. The platform reads more: ``?``, ``+``, a bracket class, and a
# ``**/`` that also matches no directory at all.
READ_ENTRY = re.compile(r"!?(?!.*\*\*/)[A-Za-z0-9_./*-]+")


def starts_the_job(path: str, entries: list[str]) -> bool:
    """A workflow path filter's answer for one path, as the platform gives
    it: the entries are read in order and the last one that matches decides,
    so an entry led by ``!`` takes the path back out. ``**`` crosses
    directories, ``*`` does not. An entry in a form this helper does not
    read fails the case that asked: it is never read by its letters."""
    started = False
    for entry in entries:
        assert READ_ENTRY.fullmatch(entry), f"this helper does not read the filter entry {entry!r}"
        pattern = re.escape(entry.removeprefix("!")).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        if re.fullmatch(pattern, path):
            started = not entry.startswith("!")
    return started


def test_a_path_filter_is_read_as_the_platform_reads_it():
    """The helper's rule on the platform's own examples: the last entry that
    matches decides, so a negated entry takes a path back out and a later
    entry puts it back in. An entry in a form the helper does not read fails
    by name."""
    notes = ["*.md", "!README.md"]
    assert starts_the_job("hello.md", notes)
    assert not starts_the_job("README.md", notes)
    assert not starts_the_job("docs/hello.md", notes)
    assert starts_the_job("README.md", [*notes, "README*"])
    assert starts_the_job("README.doc", [*notes, "README*"])
    assert not starts_the_job("README.md", ["!README.md"])
    assert starts_the_job("docs/mona/octocat.txt", ["docs/**"])
    assert not starts_the_job("docs/mona/octocat.txt", ["docs/*"])
    for unread in ("docs/**/*.md", "*.jsx?", "v[12].md", "v1+.md", "v1\\*.md"):
        with pytest.raises(AssertionError, match="does not read the filter entry"):
            starts_the_job("docs/a.md", [unread])


def test_the_image_job_starts_on_what_these_cases_read():
    """The job that runs these cases on the image's binary starts when
    something they read changes: this file, what configures its run and
    every file of the chart a render reads. The chart's alert rule tests are
    read by no case, so they start nothing. The chart's README and the two
    publishing workflows are each read by one render case and by no binary
    case: the unit job, which starts on every pull request, runs those. The
    binary the cases run is built by the image's file from the crate's
    manifest, build script, source and contracts under the pinned toolchain,
    and from the engine wire's messages the protocol imports: the job starts
    on each of those too."""
    document = yaml.safe_load(IMAGE_WORKFLOW.read_text())
    # PyYAML reads the bare key `on` as the boolean true.
    entries = document[True]["pull_request"]["paths"]
    starts_nothing = (CHART / "README.md", CHART / "tests")
    tests = Path(__file__).resolve().parent
    crate = REPO / "crates" / "policy_controller"
    read = [
        Path(__file__).resolve(),
        # pytest reads both for this run: the project file holds its settings.
        tests / "conftest.py",
        tests.parent / "pyproject.toml",
        # What the image's build reads: the binary the cases run comes of it.
        REPO / "docker" / "policy-controller.Dockerfile",
        REPO / "rust-toolchain.toml",
        crate / "Cargo.toml",
        crate / "build.rs",
        crate / "src" / "bin" / "smg-policy-controller.rs",
        crate / "contracts" / "smg-policy.wit",
        # The engine wire's messages the controller's protocol imports: the
        # build script reads them beside the contract.
        REPO / "crates" / "grpc_client" / "proto" / "common.proto",
        # The example controller, whose image the job builds beside the
        # stock one: its package and what the package compiles.
        REPO / "examples" / "policy-controller" / "Cargo.toml",
        REPO / "examples" / "policy-controller" / "src" / "main.rs",
        # The cargo configuration the build reads: the allocator of an arm64
        # binary takes its page size from it.
        REPO / ".cargo" / "config.toml",
    ] + [
        path
        for path in sorted(CHART.rglob("*"))
        if path.is_file()
        and not any(path == skip or skip in path.parents for skip in starts_nothing)
    ]
    moved = [str(path.relative_to(REPO)) for path in read if not path.is_file()]
    assert moved == [], f"{moved} is no file of the tree: the image job's filter follows it"
    missed = [
        str(path.relative_to(REPO))
        for path in read
        if not starts_the_job(str(path.relative_to(REPO)), entries)
    ]
    assert missed == [], f"a change of {missed} would not start the image job"
    assert not starts_the_job("deploy/helm/smg/README.md", entries)
    assert not starts_the_job("deploy/helm/smg/tests/rules/run.sh", entries)


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def refusal(binary: str, args: list[str]) -> str:
    """What the controller logs when it refuses to start on ``args``."""
    environment = dict(os.environ, HOSTNAME="smg-policy-controller-0")
    completed = subprocess.run(
        [binary, *args], capture_output=True, text=True, timeout=30, env=environment
    )
    assert completed.returncode != 0, f"the controller started on {args}"
    return completed.stdout + completed.stderr


def test_the_binary_refuses_at_start_what_the_chart_refuses_at_render(binary):
    """The refusals of the chart, as the controller words them: the rule, the
    default window and the listeners' names are the controller's, not the
    chart's guess."""
    port = free_ports(1)[0]
    shared = refusal(
        binary, ["--gateway-listen", f"127.0.0.1:{port}", "--control-listen", f"127.0.0.1:{port}"]
    )
    assert "share the address" in shared
    grace = 2 * DEFAULT_SETTLE_SECONDS - 1
    over = refusal(binary, ["--shutdown-grace-period-secs", str(grace)])
    assert f"settle window of {DEFAULT_SETTLE_SECONDS}s is over half the grace period" in over
    # A name the schema refuses is no listener of the controller's. The
    # controller binds its listeners before it reads the name, so they take
    # ports of the case's own: a host that holds a default port reads the
    # same refusal.
    gateway, control, replication, admin = free_ports(4)
    unknown = refusal(
        binary,
        [
            "--gateway-listen",
            f"127.0.0.1:{gateway}",
            "--control-listen",
            f"127.0.0.1:{control}",
            "--replication-listen",
            f"127.0.0.1:{replication}",
            "--admin-listen",
            f"127.0.0.1:{admin}",
            "--identity-as-told",
            "admin",
        ],
    )
    assert 'no listener is named "admin"' in unknown


@needs_helm
def test_the_binary_starts_on_the_rendered_arguments(binary):
    """The controller started with exactly what the Deployment passes (on
    ports free here, a short grace period, each gRPC listener named to take
    identities as told) answers its readiness probe and stops clean on
    SIGTERM."""
    gateway, control, replication, admin, ingestion = free_ports(5)
    documents = render(
        *ON,
        f"policyController.ports.gateway={gateway}",
        f"policyController.ports.control={control}",
        f"policyController.ports.replication={replication}",
        f"policyController.ports.admin={admin}",
        "policyController.shutdownGracePeriodSeconds=4",
        "policyController.shutdownSettleSeconds=1",
        "policyController.identityAsTold={gateway,control,replication}",
        # A flag in extraArgs adds to the value's set: the ingestion listener
        # is named where it is turned on, and a name said in both places
        # names its listener once more.
        "policyController.extraArgs={--ingestion-listen,"
        f"127.0.0.1:{ingestion},--identity-as-told,ingestion,--identity-as-told,gateway}}",
        "router.metrics.serviceMonitor.enabled=true",
    )
    found = container(documents)
    probe = found["readinessProbe"]["httpGet"]
    ports = {port["name"]: port["containerPort"] for port in found["ports"]}
    url = f"http://127.0.0.1:{ports[probe['port']]}{probe['path']}"
    # The other two requests the chart sets up: the liveness probe's, and the
    # scrape of the ServiceMonitor through the Service's port (Prometheus
    # asks /metrics when the endpoint names no path).
    liveness = found["livenessProbe"]["httpGet"]
    targets = {
        port["name"]: port["targetPort"] for port in one(documents, "Service")["spec"]["ports"]
    }
    (endpoint,) = one(documents, "ServiceMonitor")["spec"]["endpoints"]
    others = [
        f"http://127.0.0.1:{ports[liveness['port']]}{liveness['path']}",
        f"http://127.0.0.1:{ports[targets[endpoint['port']]]}{endpoint.get('path', '/metrics')}",
    ]
    # Under Kubernetes HOSTNAME is the pod's name.
    environment = {k: v for k, v in os.environ.items() if k not in LOG_VARIABLES}
    environment["HOSTNAME"] = "smg-policy-controller-6d5f7c9b8-x2x4z"
    process = subprocess.Popen(
        [binary, *found["args"]],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        body = None
        deadline = time.monotonic() + 20
        while body is None and time.monotonic() < deadline:
            assert process.poll() is None, f"the controller exited: {process.stdout.read()}"
            try:
                with DIRECT.open(url, timeout=2) as response:
                    assert response.status == 200
                    body = json.load(response)
            except (urllib.error.URLError, ConnectionError):
                time.sleep(0.1)
        # One replica with no lease flag leads itself.
        assert body == {"status": "ready", "role": "leader"}
        for other in others:
            with DIRECT.open(other, timeout=2) as response:
                assert response.status == 200, other
        # Every container port the Deployment names is a listener of the
        # started controller.
        for port in ports.values():
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=15)
        assert process.returncode == 0, output
        # The output is a pipe here, as it is under a container runtime: the
        # log is plain text, with no color escape.
        assert "smg_policy_controller" in output, "the controller logged nothing"
        # Each of the chart's three names is a listener of the controller's,
        # and so is the one extraArgs turns on: the controller says so once
        # a naming, the gateways' listener named in both places twice.
        told = [line for line in output.splitlines() if "identities are taken as told" in line]
        assert len(told) == 5, output
        for listener in ("gateway", "control", "replication", "ingestion"):
            assert any(f"listener={listener}" in line for line in told), (listener, output)
        assert "\x1b" not in output, f"an escape byte with no terminal: {output!r}"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


# The elected controller on the chart's own Lease and grant. No cluster: the
# three calls a controller makes are answered here, in this process.

LEASES = "/apis/coordination.k8s.io/v1/namespaces/default/leases"
# What a started controller is never left to find in this environment: a
# cluster's configuration and a proxy for its client.
AMBIENT = (
    "KUBECONFIG",
    "KUBERNETES_SERVICE_HOST",
    "KUBERNETES_SERVICE_PORT",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class LeaseStandIn:
    """One namespace's Leases behind the three calls a controller makes (a
    read, a replace under the version it read, a create), answered as an API
    server answers an account that holds ``rules`` and nothing else: a call
    no rule allows is refused as forbidden. Every call is kept as its verb,
    the Lease's name and whether a rule allowed it."""

    def __init__(self, rules: list[dict]):
        self.rules = rules
        self.objects: dict[str, dict] = {}
        self.calls: list[tuple[str, str, bool]] = []
        self._version = 0
        self._lock = threading.Lock()
        stand_in = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:
                """Nothing of the stand-in's joins the test's output."""

            def do_GET(self) -> None:
                stand_in.answer(self, "get")

            def do_PUT(self) -> None:
                stand_in.answer(self, "update")

            def do_POST(self) -> None:
                stand_in.answer(self, "create")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def apply(self, document: dict) -> None:
        """What applying the release does: the object as the chart renders
        it, under a version of the server's."""
        with self._lock:
            self._store(json.loads(json.dumps(document)))

    def delete(self, name: str) -> None:
        """What a delete by hand does: the object is gone."""
        with self._lock:
            del self.objects[name]

    def _store(self, document: dict) -> dict:
        self._version += 1
        document["metadata"]["namespace"] = "default"
        document["metadata"]["resourceVersion"] = str(self._version)
        self.objects[document["metadata"]["name"]] = document
        return document

    def allows(self, verb: str, name: str) -> bool:
        """The rules as an API server reads them. A rule held to names never
        allows a create: no name stands in a create's address."""
        for rule in self.rules:
            if "coordination.k8s.io" not in rule["apiGroups"] or "leases" not in rule["resources"]:
                continue
            if verb not in rule["verbs"]:
                continue
            names = rule.get("resourceNames")
            if names is None or (verb != "create" and name in names):
                return True
        return False

    def answer(self, request: http.server.BaseHTTPRequestHandler, verb: str) -> None:
        length = int(request.headers.get("Content-Length") or 0)
        sent = json.loads(request.rfile.read(length)) if length else {}
        path = request.path.partition("?")[0]
        name = path[len(LEASES) + 1 :] if path.startswith(f"{LEASES}/") else ""
        name = name or sent.get("metadata", {}).get("name", "")
        with self._lock:
            status, body = self._decide(verb, path, name, sent)
        payload = json.dumps(body).encode()
        request.send_response(status)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(payload)))
        request.end_headers()
        request.wfile.write(payload)

    def _decide(self, verb: str, path: str, name: str, sent: dict) -> tuple[int, dict]:
        def failure(code: int, reason: str, message: str) -> tuple[int, dict]:
            status = {
                "kind": "Status",
                "apiVersion": "v1",
                "metadata": {},
                "status": "Failure",
                "message": message,
                "reason": reason,
                "code": code,
            }
            return code, status

        if path != LEASES and not path.startswith(f"{LEASES}/"):
            return failure(404, "NotFound", f"the stand-in serves no {path}")
        allowed = self.allows(verb, name)
        self.calls.append((verb, name, allowed))
        if not allowed:
            return failure(
                403,
                "Forbidden",
                f'leases.coordination.k8s.io "{name}" is forbidden: the account cannot {verb} '
                'resource "leases" in API group "coordination.k8s.io" in the namespace "default"',
            )
        stored = self.objects.get(name)
        if verb == "create":
            if stored is not None:
                return failure(409, "AlreadyExists", f'leases.coordination.k8s.io "{name}" exists')
            return 201, self._store(sent)
        if stored is None:
            return failure(404, "NotFound", f'leases.coordination.k8s.io "{name}" not found')
        if verb == "get":
            return 200, stored
        if sent.get("metadata", {}).get("resourceVersion") != stored["metadata"]["resourceVersion"]:
            return failure(409, "Conflict", f'the object "{name}" has been modified')
        return 200, self._store(sent)


def client_configuration(directory: Path, server: str) -> Path:
    """What a pod's account gives a controller, as a file: the API server
    and the pod's own namespace."""
    path = directory / "kubeconfig"
    configuration = {
        "apiVersion": "v1",
        "kind": "Config",
        "current-context": "stand-in",
        "clusters": [{"name": "stand-in", "cluster": {"server": server}}],
        "contexts": [
            {
                "name": "stand-in",
                "context": {"cluster": "stand-in", "user": "stand-in", "namespace": "default"},
            }
        ],
        "users": [{"name": "stand-in", "user": {}}],
    }
    path.write_text(yaml.safe_dump(configuration))
    return path


class ElectedPod:
    """The controller as one pod of the elected Deployment: exactly the
    arguments the Deployment passes, on ports free here and a short grace
    period, with the pod's address written where the kubelet writes it, the
    pod's name in HOSTNAME, and the client configuration of its account."""

    def __init__(self, binary: str, configuration: Path, name: str):
        gateway, control, replication, admin = free_ports(4)
        found = container(
            render(
                *ELECTED,
                "policyController.replicas=2",
                f"policyController.ports.gateway={gateway}",
                f"policyController.ports.control={control}",
                f"policyController.ports.replication={replication}",
                f"policyController.ports.admin={admin}",
                "policyController.shutdownGracePeriodSeconds=4",
                "policyController.shutdownSettleSeconds=1",
            )
        )
        # The kubelet expands a variable the container itself names.
        assert "POD_IP" in [variable["name"] for variable in found["env"]]
        arguments = [argument.replace("$(POD_IP)", "127.0.0.1") for argument in found["args"]]
        assert arguments != found["args"], "the arguments name no pod address"
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in LOG_VARIABLES and key not in AMBIENT
        }
        environment["HOSTNAME"] = name
        environment["KUBECONFIG"] = str(configuration)
        self.identity = f"{name}@127.0.0.1:{gateway}"
        self.probe = f"http://127.0.0.1:{admin}/readiness"
        self.process = subprocess.Popen(
            [binary, *arguments],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def readiness(self) -> tuple[int, dict]:
        """The probe's status and body; 0 while nothing listens, and when a
        held host does not answer inside the probe's two seconds."""
        if self.process.poll() is not None:
            output = self.process.stdout.read() if self.process.stdout else ""
            raise AssertionError(f"the controller exited: {output}")
        try:
            with DIRECT.open(self.probe, timeout=2) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as refused:
            return refused.code, json.load(refused)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return 0, {}

    def reads(self, role: str, seconds: float = 20) -> None:
        """Waits until the probe answers ready under ``role``."""
        deadline = time.monotonic() + seconds
        answer = self.readiness()
        while answer != (200, {"status": "ready", "role": role}):
            assert time.monotonic() < deadline, f"not ready as {role} in {seconds} s: {answer}"
            time.sleep(0.1)
            answer = self.readiness()

    def stop(self) -> str:
        """SIGTERM, the exit status checked, and the log."""
        self.process.send_signal(signal.SIGTERM)
        output, _ = self.process.communicate(timeout=15)
        assert self.process.returncode == 0, output
        return output

    def end(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


POD_NAMES = ("smg-policy-controller-6d5f7c9b8-x2x4z", "smg-policy-controller-6d5f7c9b8-k8m2p")


def one_leads_and_one_follows(pods: list[ElectedPod]) -> None:
    """Waits until one pod reads ready as the leader and the other as a
    follower."""
    deadline = time.monotonic() + 20
    roles: list[str] = []
    while sorted(roles) != ["follower", "leader"]:
        assert time.monotonic() < deadline, f"no leader and follower in 20 s: {roles}"
        time.sleep(0.2)
        roles = [pod.readiness()[1].get("role", "") for pod in pods]


@needs_helm
def test_two_elected_pods_lead_follow_and_hand_over_on_the_charts_lease_and_grant(binary, tmp_path):
    """Two controllers started as the elected Deployment's pods, over the
    Lease as the chart renders it, behind the Role as the chart renders it:
    the first reads ready as the leader and is the Lease's holder under its
    pod's address, the second reads ready as a follower, and the first's
    stop hands the Lease over. THE GRANT: every call the two made is one the
    Role allows, and the Role holds no verb they did not call."""
    documents = render(*ELECTED, "policyController.replicas=2")
    (rule,) = one(documents, "Role")["rules"]
    lease = one(documents, "Lease")
    stand_in = LeaseStandIn([rule])
    pods: list[ElectedPod] = []
    try:
        stand_in.apply(lease)
        configuration = client_configuration(tmp_path, stand_in.url)
        pods.append(ElectedPod(binary, configuration, POD_NAMES[0]))
        pods[0].reads("leader")
        pods.append(ElectedPod(binary, configuration, POD_NAMES[1]))
        pods[1].reads("follower")
        assert pods[0].readiness() == (200, {"status": "ready", "role": "leader"})
        stored = stand_in.objects[NAME]
        assert stored["spec"]["holderIdentity"] == pods[0].identity
        # The controllers' writes keep what the chart rendered on the object.
        assert stored["metadata"]["labels"] == lease["metadata"]["labels"]

        output = pods[0].stop()
        assert "the lease is released" in output
        pods[1].reads("leader")
        assert stand_in.objects[NAME]["spec"]["holderIdentity"] == pods[1].identity
        pods[1].stop()

        refused = [call for call in stand_in.calls if not call[2]]
        assert refused == [], f"calls the Role does not allow: {refused}"
        assert {name for _, name, _ in stand_in.calls} == set(rule["resourceNames"])
        assert {verb for verb, _, _ in stand_in.calls} == set(rule["verbs"])
    finally:
        for pod in pods:
            pod.end()
        stand_in.close()


@needs_helm
def test_a_missing_lease_leaves_no_leader_until_the_release_is_applied_again(binary, tmp_path):
    """The controller is granted no create, so with the Lease gone neither
    pod leads: each stays not ready and says the refusal. The create is the
    one call of theirs the Role refuses, by design. The Lease applied again,
    as the release renders it, one leads and the other follows with no
    restart."""
    documents = render(*ELECTED, "policyController.replicas=2")
    (rule,) = one(documents, "Role")["rules"]
    stand_in = LeaseStandIn([rule])
    pods: list[ElectedPod] = []
    try:
        configuration = client_configuration(tmp_path, stand_in.url)
        for name in POD_NAMES:
            pods.append(ElectedPod(binary, configuration, name))
        # Through four renew periods of the controller's own timings. A
        # probe a held host did not answer in time is no sample.
        starting = (503, {"status": "not ready", "reason": "starting"})
        answered = [0, 0]
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            for place, pod in enumerate(pods):
                answer = pod.readiness()
                if answer != (0, {}):
                    assert answer == starting
                    answered[place] += 1
            time.sleep(0.2)
        assert min(answered) >= 5, f"the probes answered {answered} times in the window"
        assert stand_in.objects == {}
        assert {call for call in stand_in.calls if not call[2]} == {("create", NAME, False)}

        stand_in.apply(one(documents, "Lease"))
        one_leads_and_one_follows(pods)
        for pod in pods:
            output = pod.stop()
            assert "an election step failed" in output
            assert f'"{NAME}" is forbidden' in output
    finally:
        for pod in pods:
            pod.end()
        stand_in.close()


@needs_helm
def test_a_lease_deleted_under_running_pods_leaves_them_ready_and_nobody_leading(binary, tmp_path):
    """The Lease deleted under a leader and a follower: the leader's next
    renewal finds no object, so it fences itself inside the renew deadline,
    and the one take tried after it, a create, is refused: nobody leads.
    Neither pod turns unready: a controller that read its Lease once reads
    ready as a follower. The Lease applied again, as the release renders it,
    one leads and the other follows with no restart."""
    documents = render(*ELECTED, "policyController.replicas=2")
    (rule,) = one(documents, "Role")["rules"]
    lease = one(documents, "Lease")
    stand_in = LeaseStandIn([rule])
    pods: list[ElectedPod] = []
    try:
        stand_in.apply(lease)
        configuration = client_configuration(tmp_path, stand_in.url)
        pods.append(ElectedPod(binary, configuration, POD_NAMES[0]))
        pods[0].reads("leader")
        pods.append(ElectedPod(binary, configuration, POD_NAMES[1]))
        pods[1].reads("follower")

        stand_in.delete(NAME)
        pods[0].reads("follower")
        # Through the lost record's duration and three renew periods of the
        # controller's own timings: a replica may wait a lost record out
        # before the take that is refused. A probe a held host did not
        # answer in time is no sample.
        follower = (200, {"status": "ready", "role": "follower"})
        answered = [0, 0]
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            for place, pod in enumerate(pods):
                answer = pod.readiness()
                if answer != (0, {}):
                    assert answer == follower
                    answered[place] += 1
            time.sleep(0.2)
        assert min(answered) >= 5, f"the probes answered {answered} times in the window"
        assert stand_in.objects == {}
        assert {call for call in stand_in.calls if not call[2]} == {("create", NAME, False)}

        stand_in.apply(lease)
        one_leads_and_one_follows(pods)
        for pod in pods:
            pod.stop()
    finally:
        for pod in pods:
            pod.end()
        stand_in.close()


# The controller under a spec document, as the Deployment's one pod. No
# cluster: the document's ConfigMap is laid out here as a kubelet mounts it.

EXAMPLE = EXAMPLES / "with-policy-controller.yaml"
# The spec's series as /metrics renders them: the generation the controller
# runs, and its counts of documents applied and rejected.
SPEC_SERIES = (
    "smg_policy_spec_generation",
    'smg_policy_spec_apply_total{result="applied"}',
    'smg_policy_spec_apply_total{result="rejected"}',
)


def example_values(directory: Path, document: dict, name: str) -> Path:
    """The controller's example values with ``document`` as the spec, whole:
    a values file laid over another would join the two documents key by
    key."""
    values = yaml.safe_load(EXAMPLE.read_text())
    values["policyController"]["spec"] = document
    path = directory / name
    path.write_text(yaml.safe_dump(values))
    return path


class Mount:
    """A directory laid out as a kubelet lays a ConfigMap volume: each key's
    file is a link through ``..data``, itself a link to the directory of the
    version in use. An update writes a new directory, turns ``..data`` to it
    by one rename and takes the old directory away: a reader finds one
    version whole or the other, and a file it holds may go from under it."""

    def __init__(self, directory: Path, config: dict):
        self.directory = directory
        self._versions = 0
        directory.mkdir()
        self.turn(config)
        for key in config["data"]:
            (directory / key).symlink_to(Path("..data") / key)

    def turn(self, config: dict) -> None:
        """The mount's update to ``config``."""
        self._versions += 1
        version = self.directory / f"..version-{self._versions}"
        version.mkdir()
        for key, document in config["data"].items():
            (version / key).write_text(document)
        staged = self.directory / "..data_tmp"
        staged.symlink_to(version.name)
        staged.replace(self.directory / "..data")
        # The kubelet takes the old version away once the link is turned.
        if self._versions > 1:
            shutil.rmtree(self.directory / f"..version-{self._versions - 1}")


class DocumentedPod:
    """The controller as the Deployment's one pod under a spec document:
    exactly the arguments the Deployment passes, on ports free here and a
    short grace period, with the pod's name in HOSTNAME and the document's
    ConfigMap mounted. A directory of the case's stands in for the mount's
    path, and nothing else of the arguments is another."""

    def __init__(self, binary: str, directory: Path, values: Path):
        gateway, control, replication, admin = free_ports(4)
        self._settings = (
            *ON,
            f"policyController.ports.gateway={gateway}",
            f"policyController.ports.control={control}",
            f"policyController.ports.replication={replication}",
            f"policyController.ports.admin={admin}",
            "policyController.shutdownGracePeriodSeconds=4",
            "policyController.shutdownSettleSeconds=1",
        )
        documents = render(*self._settings, files=(values,))
        self._deployment = one(documents, "Deployment")
        found = container(documents)
        (mounted,) = found["volumeMounts"]
        self._mount = Mount(directory / "mount", one(documents, "ConfigMap"))
        self.arguments = [
            argument.replace(mounted["mountPath"], str(self._mount.directory), 1)
            if argument.startswith(mounted["mountPath"] + "/")
            else argument
            for argument in found["args"]
        ]
        assert self.arguments != found["args"], "no argument names a file under the mount"
        self._metrics = f"http://127.0.0.1:{admin}/metrics"
        environment = {key: value for key, value in os.environ.items() if key not in LOG_VARIABLES}
        environment["HOSTNAME"] = POD_NAMES[0]
        self.process = subprocess.Popen(
            [binary, *self.arguments],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def upgrade(self, values: Path) -> None:
        """An upgrade of the release to ``values``. It renders the Deployment
        the pod runs under, so it moves no pod and the pod's arguments stand,
        and the kubelet turns the mount to the upgrade's ConfigMap."""
        documents = render(*self._settings, files=(values,))
        assert one(documents, "Deployment") == self._deployment, "the upgrade moves the pod"
        self._mount.turn(one(documents, "ConfigMap"))

    def spec(self) -> tuple[int, ...] | None:
        """The spec's series as they read now; none while nothing listens."""
        if self.process.poll() is not None:
            output = self.process.stdout.read() if self.process.stdout else ""
            raise AssertionError(f"the controller exited: {output}")
        try:
            with DIRECT.open(self._metrics, timeout=2) as response:
                rendered = response.read().decode()
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return None
        series = dict(re.findall(r"^(smg_policy_spec_\S+) (\d+)$", rendered, flags=re.MULTILINE))
        assert set(SPEC_SERIES) <= set(series), f"/metrics renders {sorted(series)}"
        return tuple(int(series[name]) for name in SPEC_SERIES)

    def runs(self, generation: int, applied: int, rejected: int, seconds: float = 20) -> None:
        """Waits until the spec's series read these three."""
        deadline = time.monotonic() + seconds
        while (read := self.spec()) != (generation, applied, rejected):
            assert time.monotonic() < deadline, f"the spec's series read {read} after {seconds} s"
            time.sleep(0.1)

    def stop(self) -> str:
        """SIGTERM, the exit status checked, and the log."""
        self.process.send_signal(signal.SIGTERM)
        output, _ = self.process.communicate(timeout=15)
        assert self.process.returncode == 0, output
        return output

    def end(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()


@needs_helm
def test_the_binary_runs_the_mounted_document_and_a_changed_one_in_place(binary, tmp_path):
    """The controller started as the Deployment's pod on the chart's example,
    its document mounted as a kubelet mounts a ConfigMap: it runs the
    document as its first generation. An upgrade to another document moves
    no pod. The mount turns, and the running controller applies the new
    document as its next generation and logs each with its digest."""
    pod = DocumentedPod(binary, tmp_path, EXAMPLE)
    try:
        pod.runs(generation=1, applied=1, rejected=0)
        pod.upgrade(example_values(tmp_path, RANDOM, "random.yaml"))
        pod.runs(generation=2, applied=2, rejected=0)
        output = pod.stop()
    finally:
        pod.end()
    applied = [line for line in output.splitlines() if "policy spec applied" in line]
    assert len(applied) == 2, output
    for generation, line in enumerate(applied, start=1):
        assert f"generation={generation} digest=" in line, line
    assert "policy spec rejected" not in output


@needs_helm
def test_a_mounted_document_that_does_not_run_is_rejected_and_the_running_one_stays(
    binary, tmp_path
):
    """A mounted document turned to one that does not run. The upgrade
    renders the Deployment as it was, so the pod's arguments stand and no
    pod is started on the document. The running controller rejects it once,
    in its log and in its count, and runs the generation it ran. A
    controller started on those arguments refuses to start and names the
    document's fault: a pod the upgrade had moved would not come up."""
    pod = DocumentedPod(binary, tmp_path, EXAMPLE)
    try:
        pod.runs(generation=1, applied=1, rejected=0)
        pod.upgrade(example_values(tmp_path, UNKNOWN_FILTER, "unknown.yaml"))
        pod.runs(generation=1, applied=1, rejected=1)
        output = pod.stop()
    finally:
        pod.end()
    (rejected,) = [line for line in output.splitlines() if "policy spec rejected" in line]
    assert "the running generation stays" in rejected and "generation=1" in rejected
    assert "no-such-filter" in rejected
    assert output.count("policy spec applied") == 1
    assert "no-such-filter" in refusal(binary, pod.arguments)


def test_the_image_job_builds_the_example_for_both_platforms_and_pushes_nothing():
    """The example controller's image is built by the image's file with the
    package argument naming the example: once for the runner's platform and
    once for arm64 by the file's own cross build, behind an emulation step,
    since the final stage runs the binary. No step of the job pushes an image
    or logs in to a registry."""
    document = yaml.safe_load(IMAGE_WORKFLOW.read_text())
    steps = document["jobs"]["image"]["steps"]
    package = "--build-arg PACKAGE=policy-controller-example"
    builds = [at for at, step in enumerate(steps) if package in step.get("run", "")]
    assert len(builds) == 2, f"{len(builds)} step(s) build the example's image"
    for at in builds:
        assert "--file docker/policy-controller.Dockerfile" in steps[at]["run"]
    (cross,) = [at for at in builds if "--platform" in steps[at]["run"]]
    assert "--platform linux/arm64" in steps[cross]["run"]
    emulated = [
        at
        for at, step in enumerate(steps)
        if step.get("uses", "").startswith("docker/setup-qemu-action@")
    ]
    assert emulated and emulated[0] < cross, "the arm64 build's final stage runs the binary"
    for step in steps:
        told = step.get("run", "") + " " + step.get("uses", "")
        for word in ("docker push", "--push", "docker login", "login-action"):
            assert word not in told, f"the step {step.get('name')!r} holds {word!r}"
