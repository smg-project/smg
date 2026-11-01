"""The Helm chart's router arguments must be options the launcher accepts,
and no secret may travel in them.

The gateway image's entrypoint is ``python3 -m smg.launch_router``, so every
``--flag`` the chart's ``smg.routerArgs`` template emits has to be an exact
option of the ``RouterArgs`` parser. argparse accepts unambiguous
abbreviations by default, which let a few chart flags through by accident;
this check uses exact names.

The chart hands the gateway its secrets (the shared API key, the history
store's connection URL, the Oracle credentials) through Secrets the launcher
reads as a mounted file or an environment variable, never as an argument:
arguments sit in the Deployment object, the process listing and ``/proc``.

The worker discovery provider (``router.discovery``) renders the launcher's
discovery flags, which it parses back into the provider it was given; the
Role and RoleBinding exist only for what reads pods through the Kubernetes
API, and the combinations the gateway cannot run fail the render.
The render checks need the ``helm`` binary and are skipped without it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from smg import router_args
from smg.router_args import RouterArgs

REPO = Path(__file__).resolve().parents[3]
CHART = REPO / "deploy" / "helm" / "smg"
HELPERS = CHART / "templates" / "_helpers.tpl"
ROUTER_TEMPLATE = CHART / "templates" / "deployment-router.yaml"
# The gateway's Consul provider, which reads Consul's own variables.
CONSUL_PROVIDER = REPO / "model_gateway" / "src" / "service_discovery" / "consul.rs"
# The chart needs Kubernetes >= 1.26 and `helm template` checks it against
# its default version, which older helm releases put below that.
KUBE_VERSION = "1.30.0"

# The launcher's secrets: the flag that takes the value itself, and the
# environment variables that carry it or name a file holding it.
SECRET_SOURCES = (
    router_args._API_KEY_SOURCES,
    router_args._TENANT_API_KEYS_SOURCES,
    router_args._POSTGRES_DB_URL_SOURCES,
    router_args._REDIS_URL_SOURCES,
)
SECRET_FLAGS = {sources.flag for sources in SECRET_SOURCES}
SECRET_VARIABLES = {name for sources in SECRET_SOURCES for name in (sources.env, sources.file_env)}

# Per history store: the flag that used to carry the URL, the variable the
# gateway reads it from, the chart's default Secret key, and a URL to render.
STORES = {
    "postgres": (
        "--postgres-db-url",
        "SMG_POSTGRES_DB_URL",
        "postgres-url",
        "postgresql://smg:hunter2@db.example:5432/smg",
    ),
    "redis": ("--redis-url", "SMG_REDIS_URL", "redis-url", "redis://:hunter2@cache.example:6379/0"),
}

# Per Oracle credential (a value of history.oracle): the flag that used to
# carry it, the variable the launcher reads it from, the chart's default
# Secret key, and a value to render.
ORACLE_CREDENTIALS = {
    "dsn": ("--oracle-connect-descriptor", "ATP_DSN", "oracle-dsn", "tcps://db.example:1522/svc"),
    "user": ("--oracle-username", "ATP_USER", "oracle-user", "dbadmin42"),
    "password": ("--oracle-password", "ATP_PASSWORD", "oracle-password", "hunter2"),
}
ORACLE_FLAGS = {flag for flag, _, _, _ in ORACLE_CREDENTIALS.values()}
ORACLE_VARIABLES = {variable for _, variable, _, _ in ORACLE_CREDENTIALS.values()}
ORACLE_SETTINGS = [
    f"history.oracle.{name}={value}" for name, (_, _, _, value) in ORACLE_CREDENTIALS.items()
]

needs_chart = pytest.mark.skipif(not HELPERS.exists(), reason="chart not present in this checkout")
needs_helm = pytest.mark.skipif(
    shutil.which("helm") is None or not HELPERS.exists(), reason="helm or the chart not present"
)
needs_gateway_source = pytest.mark.skipif(
    not CONSUL_PROVIDER.exists(), reason="the gateway source is not in this checkout"
)


def router_args_flags() -> set[str]:
    text = HELPERS.read_text()
    start = text.index('{{- define "smg.routerArgs" -}}')
    end = text.index("{{- define", start + 1)
    return set(re.findall(r'"(--[a-z0-9-]+)"', text[start:end]))


@needs_chart
def test_chart_router_flags_are_launcher_options():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    RouterArgs.add_cli_args(parser)
    known = set(parser._option_string_actions)
    flags = router_args_flags()
    assert flags, "no flags found in smg.routerArgs"
    unknown = sorted(flags - known)
    assert not unknown, f"the chart passes flags the launcher does not accept: {unknown}"


@needs_chart
def test_chart_router_args_carry_no_secret():
    """A secret's value flag never appears in the args: the chart passes a
    file (``--api-key-file``) or sets the variable from a Secret instead."""
    leaked = sorted(router_args_flags() & (SECRET_FLAGS | ORACLE_FLAGS))
    assert not leaked, f"the chart passes a secret in the pod arguments: {leaked}"


@needs_chart
def test_chart_secret_variables_are_launcher_names():
    """Every ``SMG_*`` or ``ATP_*`` variable the chart's templates name is one
    the launcher reads."""
    text = HELPERS.read_text() + ROUTER_TEMPLATE.read_text()
    names = set(re.findall(r"\b(?:SMG|ATP)_[A-Z0-9_]+\b", text))
    assert names >= ORACLE_VARIABLES, "the chart does not set every Oracle variable"
    unknown = sorted(names - SECRET_VARIABLES - ORACLE_VARIABLES)
    assert not unknown, f"the chart sets variables the launcher does not read: {unknown}"


def test_oracle_variables_are_read_by_the_launcher(monkeypatch):
    """The three variables the chart wires for Oracle reach the launcher's
    Oracle settings when the flags are absent."""
    for _, variable, _, value in ORACLE_CREDENTIALS.values():
        monkeypatch.setenv(variable, value)
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    args = RouterArgs.from_cli_args(parser.parse_args(["--history-backend", "oracle"]))
    assert (args.oracle_connect_descriptor, args.oracle_username, args.oracle_password) == tuple(
        value for _, _, _, value in ORACLE_CREDENTIALS.values()
    )


def helm_template(settings: tuple[str, ...], files: tuple[Path, ...] = ()):
    command = ["helm", "template", "smg", str(CHART), "--kube-version", KUBE_VERSION]
    for file in files:
        command += ["-f", str(file)]
    for setting in settings:
        command += ["--set", setting]
    return subprocess.run(command, capture_output=True, text=True)


def render(*settings: str, files: tuple[Path, ...] = ()) -> tuple[str, list[dict]]:
    """Renders the chart with ``--set`` pairs: the text and its documents."""
    completed = helm_template(settings, files)
    assert completed.returncode == 0, completed.stderr
    documents = [document for document in yaml.safe_load_all(completed.stdout) if document]
    return completed.stdout, documents


def render_error(*settings: str) -> str:
    """Renders the chart with ``--set`` pairs that must fail: helm's error."""
    completed = helm_template(settings)
    assert completed.returncode != 0, "the chart rendered values it should refuse"
    return completed.stderr


def router_pod(documents: list[dict]) -> dict:
    """The router's pod template (of its Deployment, or StatefulSet in mesh mode)."""
    (router,) = [
        document
        for document in documents
        if document["kind"] in ("Deployment", "StatefulSet")
        and document["metadata"]["name"] == "smg-router"
    ]
    return router["spec"]["template"]


def router_env(pod: dict) -> dict[str, dict]:
    return {entry["name"]: entry for entry in pod["spec"]["containers"][0]["env"]}


def secret_names(documents: list[dict]) -> set[str]:
    return {document["metadata"]["name"] for document in documents if document["kind"] == "Secret"}


@needs_helm
@pytest.mark.parametrize("store", sorted(STORES))
def test_store_url_value_renders_into_a_chart_secret(store):
    flag, variable, key, url = STORES[store]
    rendered, documents = render(f"history.backend={store}", f"history.{store}.url={url}")
    pod = router_pod(documents)
    args = pod["spec"]["containers"][0]["args"]
    assert flag not in args
    assert url not in " ".join(args)
    assert router_env(pod)[variable]["valueFrom"]["secretKeyRef"] == {
        "name": "smg-history",
        "key": key,
    }
    (secret,) = [
        d for d in documents if d["kind"] == "Secret" and d["metadata"]["name"] == "smg-history"
    ]
    assert secret["stringData"] == {key: url}
    # The URL is in the Secret and nowhere else in the release.
    assert rendered.count(url) == 1
    # A changed URL rolls the pods: the pod template carries the Secret's checksum.
    checksum = pod["metadata"]["annotations"]["checksum/history-secret"]
    assert re.fullmatch(r"[0-9a-f]{64}", checksum)
    _, rotated = render(f"history.backend={store}", f"history.{store}.url={url}-rotated")
    assert router_pod(rotated)["metadata"]["annotations"]["checksum/history-secret"] != checksum


@needs_helm
@pytest.mark.parametrize("store", sorted(STORES))
def test_store_url_from_an_existing_secret(store):
    flag, variable, key, url = STORES[store]
    rendered, documents = render(
        f"history.backend={store}",
        f"history.{store}.existingSecret=db-creds",
        f"history.{store}.secretKey=dsn",
        # A url left in the values is written nowhere: the existing Secret wins.
        f"history.{store}.url={url}",
    )
    assert url not in rendered
    assert "smg-history" not in secret_names(documents)
    pod = router_pod(documents)
    assert flag not in pod["spec"]["containers"][0]["args"]
    assert router_env(pod)[variable]["valueFrom"]["secretKeyRef"] == {
        "name": "db-creds",
        "key": "dsn",
    }
    assert "checksum/history-secret" not in (pod["metadata"]["annotations"] or {})


@needs_helm
@pytest.mark.parametrize(
    "settings",
    [(), ("history.backend=postgres",), ("history.backend=redis",), ("history.backend=oracle",)],
)
def test_store_without_a_url_gets_no_variable(settings):
    _, documents = render(*settings)
    assert not secret_names(documents) & {"smg-history", "smg-secrets"}
    pod = router_pod(documents)
    assert not [name for name in router_env(pod) if name.startswith(("SMG_", "ATP_"))]
    annotations = pod["metadata"]["annotations"] or {}
    assert not [name for name in annotations if name.startswith("checksum/")]
    if settings:
        assert "--history-backend" in pod["spec"]["containers"][0]["args"]


@needs_helm
def test_store_url_variable_reaches_the_mesh_statefulset():
    _, documents = render(
        "router.mesh.enabled=true",
        "router.replicas=2",
        "history.backend=redis",
        "history.redis.existingSecret=cache-creds",
    )
    (router,) = [document for document in documents if document["kind"] == "StatefulSet"]
    env = router_env(router["spec"]["template"])
    assert env["SMG_REDIS_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": "cache-creds",
        "key": "redis-url",
    }


def oracle_secret(documents: list[dict]) -> dict:
    (secret,) = [
        d for d in documents if d["kind"] == "Secret" and d["metadata"]["name"] == "smg-secrets"
    ]
    return secret


@needs_helm
def test_oracle_credentials_render_into_a_chart_secret():
    rendered, documents = render("history.backend=oracle", *ORACLE_SETTINGS)
    pod = router_pod(documents)
    args = pod["spec"]["containers"][0]["args"]
    env = router_env(pod)
    assert "--oracle-pool-max" in args
    for flag, variable, key, value in ORACLE_CREDENTIALS.values():
        assert flag not in args
        assert value not in " ".join(args)
        assert env[variable]["valueFrom"]["secretKeyRef"] == {"name": "smg-secrets", "key": key}
        # The credential is in the Secret and nowhere else in the release.
        assert rendered.count(value) == 1
    assert oracle_secret(documents)["stringData"] == {
        key: value for _, _, key, value in ORACLE_CREDENTIALS.values()
    }
    # A changed credential rolls the pods: the pod template carries the Secret's checksum.
    checksum = pod["metadata"]["annotations"]["checksum/oracle-secret"]
    assert re.fullmatch(r"[0-9a-f]{64}", checksum)
    _, rotated = render(
        "history.backend=oracle", *ORACLE_SETTINGS[:-1], "history.oracle.password=rotated"
    )
    assert router_pod(rotated)["metadata"]["annotations"]["checksum/oracle-secret"] != checksum


@needs_helm
def test_oracle_credentials_from_an_existing_secret():
    rendered, documents = render(
        "history.backend=oracle",
        "history.oracle.existingSecret=db-creds",
        "history.oracle.secretKeys.password=pw",
        # A password left in the values is written nowhere: the existing Secret wins.
        "history.oracle.password=hunter2",
    )
    assert "hunter2" not in rendered
    assert "smg-secrets" not in secret_names(documents)
    pod = router_pod(documents)
    assert not ORACLE_FLAGS & set(pod["spec"]["containers"][0]["args"])
    # All three variables are wired: the custom key for the password, the defaults for the rest.
    env = router_env(pod)
    assert env["ATP_DSN"]["valueFrom"]["secretKeyRef"] == {"name": "db-creds", "key": "oracle-dsn"}
    assert env["ATP_USER"]["valueFrom"]["secretKeyRef"] == {
        "name": "db-creds",
        "key": "oracle-user",
    }
    assert env["ATP_PASSWORD"]["valueFrom"]["secretKeyRef"] == {"name": "db-creds", "key": "pw"}
    assert "checksum/oracle-secret" not in (pod["metadata"]["annotations"] or {})


@needs_helm
def test_a_partial_oracle_configuration_wires_what_is_set():
    """The connect descriptor alone (external authentication, say): the
    Secret and the environment carry it and nothing for the user or password."""
    dsn = ORACLE_CREDENTIALS["dsn"][3]
    _, documents = render("history.backend=oracle", f"history.oracle.dsn={dsn}")
    assert oracle_secret(documents)["stringData"] == {"oracle-dsn": dsn}
    env = router_env(router_pod(documents))
    assert env["ATP_DSN"]["valueFrom"]["secretKeyRef"] == {
        "name": "smg-secrets",
        "key": "oracle-dsn",
    }
    assert not {"ATP_USER", "ATP_PASSWORD"} & set(env)


@needs_helm
def test_oracle_credentials_are_ignored_for_another_backend():
    rendered, documents = render("history.backend=postgres", *ORACLE_SETTINGS)
    assert "smg-secrets" not in secret_names(documents)
    assert not ORACLE_VARIABLES & set(router_env(router_pod(documents)))
    assert "hunter2" not in rendered


# Worker discovery (router.discovery).

# Each provider with the least it renders with.
PROVIDERS = {
    "kubernetes": ("router.discovery.provider=kubernetes",),
    "file": ("router.discovery.provider=file", "router.discovery.file.configMap.name=smg-workers"),
    "slurm": (
        "router.discovery.provider=slurm",
        "router.discovery.slurm.port=8000",
        "router.discovery.slurm.jobNames={serve}",
    ),
    "consul": ("router.discovery.provider=consul", "router.discovery.consul.service=llm"),
}
OTHER_PROVIDERS = sorted(set(PROVIDERS) - {"kubernetes"})
WORKER = (
    "workers[0].name=llama",
    "workers[0].engine=vllm",
    "workers[0].model=meta-llama/Llama-3.1-8B-Instruct",
    "workers[0].image.tag=1.13.0-vllm-v0.31.0",
)
MESH = ("router.mesh.enabled=true", "router.replicas=2")
MESH_PEERS = (*MESH, "router.mesh.peerUrls[0]=10.0.0.1:39527")
ROLE = {"Role", "RoleBinding"}
CLUSTER_ROLE = {"ClusterRole", "ClusterRoleBinding"}
CONSUL_VARIABLES = {"CONSUL_HTTP_TOKEN", "CONSUL_CACERT", "CONSUL_CLIENT_CERT", "CONSUL_CLIENT_KEY"}
EXAMPLES = sorted((CHART / "examples").glob("*.yaml"))


def router_container(documents: list[dict]) -> dict:
    return router_pod(documents)["spec"]["containers"][0]


def launcher_args(args: list[str]) -> RouterArgs:
    """The rendered arguments as the image's launcher parses them."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    RouterArgs.add_cli_args(parser)
    return RouterArgs.from_cli_args(parser.parse_args(args))


def rbac_kinds(documents: list[dict]) -> set[str]:
    return {document["kind"] for document in documents} & (ROLE | CLUSTER_ROLE)


def rbac_namespaces(documents: list[dict]) -> set[str | None]:
    """The namespace each RBAC document names, None for the release's."""
    return {
        document["metadata"].get("namespace")
        for document in documents
        if document["kind"] in ROLE | CLUSTER_ROLE
    }


def named(entries: list[dict], name: str) -> dict:
    (entry,) = [entry for entry in entries if entry["name"] == name]
    return entry


def has_run(args: list[str], run: list[str]) -> bool:
    """Whether ``run`` appears in ``args`` as consecutive items."""
    return any(args[i : i + len(run)] == run for i in range(len(args) - len(run) + 1))


@needs_helm
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_each_provider_reaches_the_launcher_as_that_provider(provider):
    _, documents = render(*PROVIDERS[provider])
    args = router_container(documents)["args"]
    assert has_run(args, ["--discovery-provider", provider])
    # --service-discovery is the other spelling, which the launcher refuses beside it.
    assert "--service-discovery" not in args
    assert launcher_args(args).selected_discovery_provider() == provider


@needs_helm
def test_legacy_service_discovery_renders_the_arguments_it_always_has():
    _, documents = render(
        "router.serviceDiscovery.enabled=true",
        "router.serviceDiscovery.selector=app=sglang",
        "router.serviceDiscovery.port=8000",
        "router.serviceDiscovery.modelIdFrom=namespace",
    )
    args = router_container(documents)["args"]
    assert has_run(
        args,
        [
            "--service-discovery",
            "--selector",
            "app=sglang",
            "--service-discovery-port",
            "8000",
            "--service-discovery-namespace",
            "default",
            "--model-id-from",
            "namespace",
        ],
    )
    assert "--discovery-provider" not in args
    assert launcher_args(args).selected_discovery_provider() == "kubernetes"
    assert rbac_kinds(documents) == ROLE


@needs_helm
def test_kubernetes_provider_takes_the_service_discovery_settings():
    _, documents = render(
        *PROVIDERS["kubernetes"],
        "router.discovery.kubernetes.selector=app=sglang",
        "router.discovery.kubernetes.port=8000",
        "router.discovery.kubernetes.namespace=team-a",
        "router.discovery.kubernetes.modelIdFrom=label:model",
    )
    parsed = launcher_args(router_container(documents)["args"])
    assert parsed.selector == {"app": "sglang"}
    assert parsed.service_discovery_port == 8000
    assert parsed.service_discovery_namespace == "team-a"
    assert parsed.model_id_from == "label:model"
    assert rbac_kinds(documents) == ROLE
    assert rbac_namespaces(documents) == {"team-a"}


@needs_helm
@pytest.mark.parametrize(
    ("settings", "namespace"),
    [
        (("rbac.create=true",), None),
        (PROVIDERS["kubernetes"], None),
        ((*PROVIDERS["kubernetes"], "router.discovery.kubernetes.namespace=default"), None),
        ((*PROVIDERS["kubernetes"], "router.discovery.kubernetes.namespace=team-a"), "team-a"),
        (("router.serviceDiscovery.enabled=true",), None),
        (
            ("router.serviceDiscovery.enabled=true", "router.serviceDiscovery.namespace=team-a"),
            "team-a",
        ),
        (MESH, None),
    ],
    ids=[
        "rbac.create",
        "kubernetes",
        "kubernetes, release namespace",
        "kubernetes, team-a",
        "legacy",
        "legacy, team-a",
        "mesh",
    ],
)
def test_the_role_is_in_the_namespace_discovery_reads_pods_from(settings, namespace):
    """A Role grants nothing outside its own namespace, so it follows the
    namespace discovery watches; the binding's subject stays the release's
    service account."""
    _, documents = render(*settings)
    assert rbac_kinds(documents) == ROLE
    assert rbac_namespaces(documents) == {namespace}
    (binding,) = [d for d in documents if d["kind"] == "RoleBinding"]
    assert binding["subjects"][0]["namespace"] == "default"


@needs_helm
@pytest.mark.parametrize(
    "settings",
    [
        ("router.serviceDiscovery.enabled=true", "router.serviceDiscovery.clusterWide=true"),
        (*PROVIDERS["kubernetes"], "router.discovery.kubernetes.clusterWide=true"),
    ],
    ids=["legacy", "provider"],
)
def test_cluster_wide_kubernetes_discovery_watches_every_namespace_under_a_cluster_role(settings):
    _, documents = render(*settings)
    assert "--service-discovery-namespace" not in router_container(documents)["args"]
    assert rbac_kinds(documents) == CLUSTER_ROLE


@needs_helm
def test_file_provider_reads_its_manifest_from_the_mounted_config_map():
    _, documents = render(*PROVIDERS["file"], "router.discovery.file.configMap.key=fleet.json")
    pod = router_pod(documents)
    container = pod["spec"]["containers"][0]
    assert has_run(container["args"], ["--discovery-file", "/etc/smg/discovery/fleet.json"])
    # The whole ConfigMap, whose updates the kubelet projects (a subPath gets
    # none), and optional: the gateway waits for a manifest not there yet.
    assert named(pod["spec"]["volumes"], "discovery-manifest")["configMap"] == {
        "name": "smg-workers",
        "optional": True,
    }
    assert named(container["volumeMounts"], "discovery-manifest") == {
        "name": "discovery-manifest",
        "mountPath": "/etc/smg/discovery",
        "readOnly": True,
    }
    assert not rbac_kinds(documents)


@needs_helm
def test_file_provider_reads_a_manifest_path_without_a_config_map():
    _, documents = render(
        "router.discovery.provider=file", "router.discovery.file.path=/opt/smg/workers.json"
    )
    pod = router_pod(documents)
    assert has_run(
        pod["spec"]["containers"][0]["args"], ["--discovery-file", "/opt/smg/workers.json"]
    )
    assert "discovery-manifest" not in {volume["name"] for volume in pod["spec"]["volumes"]}


@needs_helm
def test_slurm_provider_passes_the_job_names_of_each_role_and_its_settings():
    _, documents = render(
        "router.discovery.provider=slurm",
        "router.discovery.slurm.port=8000",
        "router.discovery.slurm.prefillJobNames={prefill-a,prefill-b}",
        "router.discovery.slurm.decodeJobNames={decode}",
        "router.discovery.slurm.bootstrapPort=8998",
        "router.discovery.slurm.user=svc-llm",
        "router.discovery.slurm.account=ml",
        "router.discovery.slurm.partition=gpu",
    )
    parsed = launcher_args(router_container(documents)["args"])
    assert parsed.slurm_prefill_job_name == ["prefill-a", "prefill-b"]
    assert parsed.slurm_decode_job_name == ["decode"]
    assert parsed.slurm_job_name == parsed.slurm_encode_job_name == []
    assert (parsed.slurm_port, parsed.slurm_bootstrap_port) == (8000, 8998)
    assert (parsed.slurm_user, parsed.slurm_account, parsed.slurm_partition) == (
        "svc-llm",
        "ml",
        "gpu",
    )
    assert not rbac_kinds(documents)


@needs_helm
def test_consul_provider_passes_its_settings_and_no_credential():
    _, documents = render(
        *PROVIDERS["consul"],
        "router.discovery.consul.address=https://consul.consul:8501",
        "router.discovery.consul.datacenter=dc1",
        "router.discovery.consul.token.existingSecret=consul-acl",
        "router.discovery.consul.tls.existingSecret=consul-tls",
    )
    args = router_container(documents)["args"]
    parsed = launcher_args(args)
    assert (parsed.consul_service, parsed.consul_address, parsed.consul_datacenter) == (
        "llm",
        "https://consul.consul:8501",
        "dc1",
    )
    assert not [arg for arg in args if "consul-acl" in arg or "consul-tls" in arg]
    assert not rbac_kinds(documents)


@needs_helm
def test_consul_token_and_tls_files_come_from_existing_secrets():
    _, documents = render(
        *PROVIDERS["consul"],
        "router.discovery.consul.token.existingSecret=consul-acl",
        "router.discovery.consul.token.secretKey=acl-token",
        "router.discovery.consul.tls.existingSecret=consul-tls",
        "router.discovery.consul.tls.secretKeys.clientCert=tls.crt",
        "router.discovery.consul.tls.secretKeys.clientKey=tls.key",
    )
    pod = router_pod(documents)
    container = pod["spec"]["containers"][0]
    env = router_env(pod)
    assert env["CONSUL_HTTP_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "consul-acl",
        "key": "acl-token",
    }
    assert {name: env[name]["value"] for name in CONSUL_VARIABLES - {"CONSUL_HTTP_TOKEN"}} == {
        "CONSUL_CACERT": "/etc/smg/consul-tls/ca.crt",
        "CONSUL_CLIENT_CERT": "/etc/smg/consul-tls/client.crt",
        "CONSUL_CLIENT_KEY": "/etc/smg/consul-tls/client.key",
    }
    assert named(pod["spec"]["volumes"], "consul-tls")["secret"] == {
        "secretName": "consul-tls",
        "items": [
            {"key": "ca.crt", "path": "ca.crt"},
            {"key": "tls.crt", "path": "client.crt"},
            {"key": "tls.key", "path": "client.key"},
        ],
    }
    assert named(container["volumeMounts"], "consul-tls") == {
        "name": "consul-tls",
        "mountPath": "/etc/smg/consul-tls",
        "readOnly": True,
    }


@needs_helm
def test_consul_tls_with_a_ca_alone_names_no_client_certificate():
    _, documents = render(*PROVIDERS["consul"], "router.discovery.consul.tls.existingSecret=ca")
    env = router_env(router_pod(documents))
    assert env["CONSUL_CACERT"]["value"] == "/etc/smg/consul-tls/ca.crt"
    assert not {"CONSUL_CLIENT_CERT", "CONSUL_CLIENT_KEY", "CONSUL_HTTP_TOKEN"} & set(env)


@needs_helm
@pytest.mark.parametrize("provider", ["", "kubernetes", "file"])
def test_consul_secrets_are_wired_only_for_the_consul_provider(provider):
    settings = PROVIDERS.get(provider, ())
    _, documents = render(
        *settings,
        "router.discovery.consul.token.existingSecret=consul-acl",
        "router.discovery.consul.tls.existingSecret=consul-tls",
    )
    pod = router_pod(documents)
    assert not [name for name in router_env(pod) if name.startswith("CONSUL_")]
    assert "consul-tls" not in {volume["name"] for volume in pod["spec"]["volumes"]}


@needs_chart
@needs_gateway_source
def test_chart_consul_variables_are_the_ones_the_gateway_reads():
    text = HELPERS.read_text() + ROUTER_TEMPLATE.read_text()
    assert set(re.findall(r"\bCONSUL_[A-Z_]+\b", text)) == CONSUL_VARIABLES
    read = set(re.findall(r'\b(?:env|read)\("(CONSUL_[A-Z_]+)"\)', CONSUL_PROVIDER.read_text()))
    assert read >= CONSUL_VARIABLES, f"the gateway reads none of {sorted(CONSUL_VARIABLES - read)}"


@needs_helm
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_check_interval_reaches_the_selected_provider(provider):
    _, documents = render(*PROVIDERS[provider], "router.discovery.checkIntervalSecs=15")
    assert launcher_args(router_container(documents)["args"]).discovery_check_interval_secs == 15


@needs_helm
def test_check_interval_reaches_the_legacy_kubernetes_spelling():
    _, documents = render(
        "router.serviceDiscovery.enabled=true", "router.discovery.checkIntervalSecs=15"
    )
    parsed = launcher_args(router_container(documents)["args"])
    assert parsed.service_discovery
    assert parsed.discovery_check_interval_secs == 15


@needs_helm
def test_check_interval_without_a_provider_is_not_passed():
    """The gateway refuses the interval without a provider."""
    _, documents = render("router.discovery.checkIntervalSecs=15")
    assert "--discovery-check-interval-secs" not in router_container(documents)["args"]


RBAC_CASES = {
    "nothing": ((), set()),
    "rbac.create": (("rbac.create=true",), ROLE),
    "legacy kubernetes": (("router.serviceDiscovery.enabled=true",), ROLE),
    "kubernetes": (PROVIDERS["kubernetes"], ROLE),
    "file": (PROVIDERS["file"], set()),
    "slurm": (PROVIDERS["slurm"], set()),
    "consul": (PROVIDERS["consul"], set()),
    "mesh without peers": (MESH, ROLE),
    "mesh with peers": (MESH_PEERS, set()),
    "mesh with peers, consul": ((*MESH_PEERS, *PROVIDERS["consul"]), set()),
    "mesh with peers, file, rbac.create": (
        (*MESH_PEERS, *PROVIDERS["file"], "rbac.create=true"),
        ROLE,
    ),
}


@needs_helm
@pytest.mark.parametrize("case", list(RBAC_CASES))
def test_rbac_exists_for_what_reads_pods_through_the_kubernetes_api(case):
    settings, kinds = RBAC_CASES[case]
    _, documents = render(*settings)
    assert rbac_kinds(documents) == kinds


@needs_helm
def test_mesh_without_peer_urls_keeps_finding_its_peers_through_kubernetes():
    _, documents = render(*MESH)
    args = router_container(documents)["args"]
    assert has_run(
        args,
        [
            "--service-discovery",
            "--selector",
            "app.kubernetes.io/instance=smg",
            "app.kubernetes.io/component=worker",
            "--router-selector",
            "app.kubernetes.io/component=router",
            "--service-discovery-namespace",
            "default",
        ],
    )
    assert "--discovery-provider" not in args


@needs_helm
def test_mesh_under_the_kubernetes_provider_adds_its_router_selector_to_it():
    _, documents = render(
        *MESH, *PROVIDERS["kubernetes"], "router.discovery.kubernetes.selector=app=sglang"
    )
    args = router_container(documents)["args"]
    assert "--service-discovery" not in args
    parsed = launcher_args(args)
    assert parsed.selected_discovery_provider() == "kubernetes"
    # The release's own routers only, not another release's in the namespace.
    assert parsed.router_selector == {
        "app.kubernetes.io/component": "router",
        "app.kubernetes.io/instance": "smg",
    }
    assert parsed.selector == {"app": "sglang"}
    assert parsed.service_discovery_namespace == "default"
    assert rbac_kinds(documents) == ROLE


@needs_helm
@pytest.mark.parametrize(
    "setting",
    [
        "router.discovery.kubernetes.clusterWide=true",
        "router.discovery.kubernetes.namespace=team-b",
    ],
)
def test_mesh_under_the_kubernetes_provider_must_watch_the_release_namespace(setting):
    error = render_error(*MESH, *PROVIDERS["kubernetes"], setting)
    assert "must watch the release namespace default" in error
    assert "set router.mesh.peerUrls" in error


@needs_helm
@pytest.mark.parametrize("provider", OTHER_PROVIDERS)
def test_mesh_without_peer_urls_under_another_provider_is_refused(provider):
    """The gateway finds mesh peers only inside Kubernetes worker discovery."""
    error = render_error(*MESH, *PROVIDERS[provider])
    assert f"with router.discovery.provider={provider}, set router.mesh.peerUrls" in error


@needs_helm
@pytest.mark.parametrize("provider", OTHER_PROVIDERS)
def test_mesh_with_peer_urls_runs_under_any_provider(provider):
    _, documents = render(*MESH_PEERS, *PROVIDERS[provider])
    args = router_container(documents)["args"]
    assert has_run(args, ["--mesh-peer-urls", "10.0.0.1:39527"])
    assert "--router-selector" not in args
    assert launcher_args(args).selected_discovery_provider() == provider


@needs_helm
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_legacy_service_discovery_beside_a_provider_is_refused(provider):
    error = render_error("router.serviceDiscovery.enabled=true", *PROVIDERS[provider])
    assert "router.serviceDiscovery.enabled is the legacy spelling" in error
    assert f"router.discovery.provider={provider}" in error


@needs_helm
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (("router.discovery.provider=file",), "router.discovery.file.configMap.name"),
        (
            ("router.discovery.provider=slurm", "router.discovery.slurm.jobNames={serve}"),
            "needs router.discovery.slurm.port",
        ),
        (
            ("router.discovery.provider=slurm", "router.discovery.slurm.port=8000"),
            "needs at least one job name",
        ),
        (("router.discovery.provider=consul",), "needs router.discovery.consul.service"),
        (
            (
                *PROVIDERS["consul"],
                "router.discovery.consul.tls.existingSecret=consul-tls",
                "router.discovery.consul.tls.secretKeys.clientCert=tls.crt",
            ),
            "set both or neither",
        ),
        (("router.discovery.provider=etcd",), "must be one of the following"),
    ],
    ids=["file", "slurm-port", "slurm-job", "consul-service", "consul-client-pair", "unknown"],
)
def test_a_provider_without_what_it_needs_is_refused(settings, message):
    assert message in render_error(*settings)


@needs_helm
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_a_provider_replaces_the_chart_workers_urls(provider):
    _, documents = render(*WORKER, *PROVIDERS[provider])
    assert "--worker-urls" not in router_container(documents)["args"]


@needs_helm
def test_chart_workers_without_discovery_are_passed_as_worker_urls():
    _, documents = render(*WORKER)
    assert has_run(
        router_container(documents)["args"], ["--worker-urls", "http://smg-worker-llama:8000"]
    )


@needs_helm
@pytest.mark.parametrize("example", EXAMPLES, ids=[example.name for example in EXAMPLES])
def test_every_example_renders_arguments_the_launcher_parses(example):
    _, documents = render(files=(example,))
    launcher_args(router_container(documents)["args"])


@needs_helm
@pytest.mark.parametrize("settle", [None, 0, 5, 30])
def test_shutdown_settle_window(settle):
    """Omit the optional window by default and preserve explicit values, including zero."""
    settings = () if settle is None else (f"router.shutdownSettleSeconds={settle}",)
    _, documents = render(*settings)
    args = router_pod(documents)["spec"]["containers"][0]["args"]
    flag = "--shutdown-settle-secs"
    if settle is None:
        assert flag not in args
    else:
        assert args[args.index(flag) + 1] == str(settle)
