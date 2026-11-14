"""The Helm chart's policy controller block: off by default, and when on,
every argument its Deployment passes is one the controller binary takes.

The controller is a Rust binary (``smg-policy-controller``), so no Python
parser stands in for it here: the binary cases run the real binary, named
by the ``SMG_POLICY_CONTROLLER_BIN`` environment variable, and are skipped
without it. The render cases need the ``helm`` binary alone and are skipped
without it.

This module renders the chart with its own few lines and not through
``test_helm_chart_args``: that module imports the gateway's launcher, which
the job that builds the controller does not install.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
CHART = REPO / "deploy" / "helm" / "smg"
VALUES = CHART / "values.yaml"
SCHEMA = CHART / "values.schema.json"
EXAMPLES = CHART / "examples"
# The chart needs Kubernetes >= 1.26 and `helm template` checks it against
# its default version, which older helm releases put below that.
KUBE_VERSION = "1.30.0"
BINARY_VARIABLE = "SMG_POLICY_CONTROLLER_BIN"
COMPONENT = "policy-controller"
NAME = "smg-policy-controller"
# The block turned on: no controller image is published, so the chart takes
# no default for the repository and the tag.
ON = (
    "policyController.enabled=true",
    "policyController.image.repository=example/smg-policy-controller",
    "policyController.image.tag=dev",
)
# The settle window the controller takes when the chart passes none.
DEFAULT_SETTLE_SECONDS = 5
# The started controller is read on a loopback port: a proxy named in the
# environment is not asked for it.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))

needs_helm = pytest.mark.skipif(
    shutil.which("helm") is None or not CHART.exists(), reason="helm or the chart not present"
)


def helm_template(settings: tuple[str, ...], files: tuple[Path, ...] = (), chart: Path = CHART):
    command = ["helm", "template", "smg", str(chart), "--kube-version", KUBE_VERSION]
    # Pinned so the render never follows the kube context.
    command += ["--namespace", "default"]
    for file in files:
        command += ["-f", str(file)]
    for setting in settings:
        command += ["--set", setting]
    return subprocess.run(command, capture_output=True, text=True)


def render(*settings: str, files: tuple[Path, ...] = ()) -> list[dict]:
    """The documents the chart renders with ``--set`` pairs."""
    completed = helm_template(settings, files)
    assert completed.returncode == 0, completed.stderr
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def render_error(*settings: str) -> str:
    """Helm's error for ``--set`` pairs the chart must refuse."""
    completed = helm_template(settings)
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


def selects(selector: dict[str, str], labels: dict[str, str]) -> bool:
    return all(labels.get(key) == value for key, value in selector.items())


@needs_helm
def test_the_controller_is_off_by_default():
    assert of_the_controller(render()) == []
    assert of_the_controller(render("policyController.enabled=false")) == []


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
    """With no block, a null block, the switch alone named off, or one value
    and no switch, the chart renders what its defaults render, byte for byte.
    Each again with the router's monitor on: the controller's monitor stands
    behind that switch."""
    cells = {
        "no block": (chart_with_no_block, ()),
        "a null block": (CHART, ("policyController=null",)),
        "the switch alone, off": (chart_with_no_block, ("policyController.enabled=false",)),
        "one value and no switch": (chart_with_no_block, ("policyController.ports.admin=9000",)),
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


def answered_as_the_schema_names(chart: Path) -> tuple[list[set[str]], list[dict]]:
    """The controller turned on over ``chart`` and given, round by round, the
    chart's own default for each value the schema names, until it renders:
    the names of each round, and the documents."""
    defaults = yaml.safe_load(VALUES.read_text())["policyController"]
    # The image ON names: the chart has no default for it.
    defaults["image"].update(s.removeprefix("policyController.image.").split("=") for s in ON[1:])
    lists = schema_block()["then"]["properties"]
    given: dict = {"enabled": True}
    rounds: list[set[str]] = []
    answers = chart.parent / "answers.yaml"
    while len(rounds) < 4:
        answers.write_text(yaml.safe_dump({"policyController": given}))
        completed = helm_template((), files=(answers,), chart=chart)
        if completed.returncode == 0:
            return rounds, [
                document for document in yaml.safe_load_all(completed.stdout) if document
            ]
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
def test_turned_on_over_values_with_no_block_keeps_the_security_context(chart_with_no_block):
    """An upgrade that reuses the values of a release from before the block
    and turns the controller on has none of the block's defaults. Answered as
    the schema names until it renders, the controller has a fresh install's
    security context and lacks four defaults, the four the README names: the
    replica count, the worker threads, the grace margin, the probes' timings."""
    rounds, answered = answered_as_the_schema_names(chart_with_no_block)
    assert rounds[0] == set(schema_block()["then"]["required"])
    fresh = render(*ON)
    found = container(fresh)
    assert container(answered).get("securityContext") == found["securityContext"]
    # A fresh install without the four is what the upgrade renders, object
    # for object: a fifth loss fails here.
    deployment = one(fresh, "Deployment")["spec"]
    deployment["replicas"] = None
    threads = found["args"].index("--runtime-worker-threads")
    del found["args"][threads : threads + 2]
    grace = yaml.safe_load(VALUES.read_text())["policyController"]["shutdownGracePeriodSeconds"]
    deployment["template"]["spec"]["terminationGracePeriodSeconds"] = grace
    for probe in ("livenessProbe", "readinessProbe"):
        found[probe].update(initialDelaySeconds=None, periodSeconds=None)
    assert answered == fresh
    # Over values with no block a null is not removed with a default: it is
    # there as a null, and refused as one.
    null = helm_template(
        (ON[0], "policyController.securityContext=null"), chart=chart_with_no_block
    )
    assert "policyController.securityContext: Invalid type" in null.stderr


@needs_helm
def test_only_its_own_example_turns_the_controller_on():
    examples = sorted(EXAMPLES.glob("*.yaml"))
    assert examples, "the chart has examples"
    on = [example.name for example in examples if of_the_controller(render(files=(example,)))]
    assert on == ["with-policy-controller.yaml"]


@needs_helm
def test_the_controller_needs_an_image_named():
    """No controller image is published: neither the repository nor the tag
    has a default, and a release without either is refused."""
    for settings in (
        ("policyController.enabled=true",),
        ("policyController.enabled=true", "policyController.image.repository=example/controller"),
        ("policyController.enabled=true", "policyController.image.tag=dev"),
    ):
        error = render_error(*settings)
        assert "policyController.image.repository" in error
        assert "policyController.image.tag" in error
    assert container(render(*ON))["image"] == "docker.io/example/smg-policy-controller:dev"
    private = render(*ON, "policyController.image.registry=registry.example.com")
    assert container(private)["image"] == "registry.example.com/example/smg-policy-controller:dev"


@needs_helm
def test_the_block_renders_a_deployment_and_a_service_and_nothing_else():
    """No ConfigMap, Role, RoleBinding or ServiceAccount: each arrives with
    the controller flag that reads it."""
    documents = of_the_controller(render(*ON))
    assert sorted(document["kind"] for document in documents) == ["Deployment", "Service"]
    assert {document["metadata"]["name"] for document in documents} == {NAME}

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
    # null leaves the flag out: the controller then starts one worker thread
    # for every CPU the pod may use.
    unset = controller_args("policyController.runtimeWorkerThreads=null")
    assert "--runtime-worker-threads" not in unset


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

    pair = one(render(*ON, "policyController.replicas=2"), "Deployment")["spec"]
    (term,) = pair["template"]["spec"]["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    assert term["podAffinityTerm"]["topologyKey"] == "kubernetes.io/hostname"
    assert (
        term["podAffinityTerm"]["labelSelector"]["matchLabels"] == pair["selector"]["matchLabels"]
    )

    own = one(
        render(
            *ON,
            "policyController.replicas=2",
            "policyController.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution"
            ".nodeSelectorTerms[0].matchExpressions[0].key=zone",
            "policyController.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution"
            ".nodeSelectorTerms[0].matchExpressions[0].operator=Exists",
        ),
        "Deployment",
    )["spec"]["template"]["spec"]["affinity"]
    assert list(own) == ["nodeAffinity"]


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
    assert "policyController.replicas" in render_error(*ON, "policyController.replicas=0")
    assert "policyController.runtimeWorkerThreads" in render_error(
        *ON, "policyController.runtimeWorkerThreads=4097"
    )
    # An empty path is no path: the kubelet would ask the root.
    for probe in ("livenessProbe", "readinessProbe"):
        assert f"policyController.{probe}.path" in render_error(
            *ON, f"policyController.{probe}.path="
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
# the defaults render.
SAME_AS_THE_DEFAULT = {
    "image.registry",
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
    "shutdownSettleSeconds",
}
# These are legal too, each with the one thing it changes: the cluster or
# the controller then takes its own default.
CHANGED_BY_A_NULL = {
    "enabled": lambda documents: of_the_controller(documents) == [],
    "replicas": lambda documents: one(documents, "Deployment")["spec"]["replicas"] is None,
    "service.type": lambda documents: one(documents, "Service")["spec"]["type"] is None,
    "runtimeWorkerThreads": lambda documents: (
        "--runtime-worker-threads" not in container(documents)["args"]
    ),
    "podDisruptionBudget.enabled": lambda documents: (
        "PodDisruptionBudget" not in [document["kind"] for document in documents]
    ),
    "livenessProbe.initialDelaySeconds": lambda documents: (
        container(documents)["livenessProbe"]["initialDelaySeconds"] is None
    ),
    "livenessProbe.periodSeconds": lambda documents: (
        container(documents)["livenessProbe"]["periodSeconds"] is None
    ),
    "readinessProbe.initialDelaySeconds": lambda documents: (
        container(documents)["readinessProbe"]["initialDelaySeconds"] is None
    ),
    "readinessProbe.periodSeconds": lambda documents: (
        container(documents)["readinessProbe"]["periodSeconds"] is None
    ),
    "terminationGraceMarginSeconds": lambda documents: (
        one(documents, "Deployment")["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]
        == 30
    ),
}


@needs_helm
def test_a_null_is_refused_or_renders_what_this_case_names():
    """Every key of the block set to null, one at a time: the schema refuses
    it, or it is one of the legal nulls above. Without the refusals the chart
    would render a listener on port 0, a flag with no value, a probe with no
    path, a grace period of zero or a budget with no number. A key added to
    the block fails here until it is one or the other."""
    block = yaml.safe_load(VALUES.read_text())["policyController"]
    switches = (*ON, "policyController.podDisruptionBudget.enabled=true")
    default = of_the_controller(render(*switches))
    for key in block_keys(block, schema_block()):
        setting = f"policyController.{key}=null"
        if key in SAME_AS_THE_DEFAULT:
            assert of_the_controller(render(*switches, setting)) == default, key
        elif key in CHANGED_BY_A_NULL:
            assert CHANGED_BY_A_NULL[key](render(*switches, setting)), key
        else:
            parent, _, name = f"policyController.{key}".rpartition(".")
            error = render_error(*switches, setting)
            assert f"{parent}: {name} is required" in error, f"{key}: {error}"


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
    "no-thread-count": ("policyController.runtimeWorkerThreads=null",),
    "other-ports": (
        "policyController.ports.gateway=31100",
        "policyController.ports.control=31101",
        "policyController.ports.replication=31102",
        "policyController.ports.admin=31103",
    ),
    "extra-arguments": (
        "policyController.extraArgs={--ingestion-listen,0.0.0.0:30104,--load-monitor-interval,2}",
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
    """The two refusals of the chart, as the controller words them: the rule
    and the default window are the controller's, not the chart's guess."""
    port = free_ports(1)[0]
    shared = refusal(
        binary, ["--gateway-listen", f"127.0.0.1:{port}", "--control-listen", f"127.0.0.1:{port}"]
    )
    assert "share the address" in shared
    grace = 2 * DEFAULT_SETTLE_SECONDS - 1
    over = refusal(binary, ["--shutdown-grace-period-secs", str(grace)])
    assert f"settle window of {DEFAULT_SETTLE_SECONDS}s is over half the grace period" in over


@needs_helm
def test_the_binary_starts_on_the_rendered_arguments(binary):
    """The controller started with exactly what the Deployment passes (on
    ports free here, a short grace period) answers its readiness probe and
    stops clean on SIGTERM."""
    gateway, control, replication, admin = free_ports(4)
    documents = render(
        *ON,
        f"policyController.ports.gateway={gateway}",
        f"policyController.ports.control={control}",
        f"policyController.ports.replication={replication}",
        f"policyController.ports.admin={admin}",
        "policyController.shutdownGracePeriodSeconds=4",
        "policyController.shutdownSettleSeconds=1",
    )
    found = container(documents)
    probe = found["readinessProbe"]["httpGet"]
    ports = {port["name"]: port["containerPort"] for port in found["ports"]}
    url = f"http://127.0.0.1:{ports[probe['port']]}{probe['path']}"
    # Under Kubernetes HOSTNAME is the pod's name.
    environment = dict(os.environ, HOSTNAME="smg-policy-controller-6d5f7c9b8-x2x4z")
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
        assert body == {"status": "ready"}
        # Every container port the Deployment names is a listener of the
        # started controller.
        for port in ports.values():
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=15) == 0, process.stdout.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
