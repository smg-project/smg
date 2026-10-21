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

CHART = Path(__file__).resolve().parents[3] / "deploy" / "helm" / "smg"
HELPERS = CHART / "templates" / "_helpers.tpl"
ROUTER_TEMPLATE = CHART / "templates" / "deployment-router.yaml"

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


def render(*settings: str) -> tuple[str, list[dict]]:
    """Renders the chart with ``--set`` pairs: the text and its documents."""
    command = ["helm", "template", "smg", str(CHART)]
    for setting in settings:
        command += ["--set", setting]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    documents = [document for document in yaml.safe_load_all(completed.stdout) if document]
    return completed.stdout, documents


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
