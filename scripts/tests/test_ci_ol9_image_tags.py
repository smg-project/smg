"""Exercise OL9 publication tags, release gates, mirrors, and operator summaries."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DIGEST = "sha256:" + "f" * 64
# The interpreter's own library path, for the docker stand-in below: this interpreter runs
# it inside a step whose environment is built from scratch. A shared-libpython build finds
# libpython through the loader variable its setup exported, and without it the stand-in
# exits 127 ("error while loading shared libraries") before its first line of Python.
INTERPRETER_ENV = {
    name: os.environ[name]
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
    if name in os.environ
}


def workflow(name):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def steps(name):
    return workflow(name)["jobs"]["build-and-push"]["steps"]


def outputs(path):
    found = {}
    lines = path.read_text().splitlines() if path.exists() else []
    while lines:
        line = lines.pop(0)
        if "<<" in line:
            name, marker = line.split("<<")
            end = lines.index(marker)
            found[name], lines = lines[:end], lines[end + 1 :]
        else:
            name, _, value = line.partition("=")
            found[name] = value
    return found


def run_step(step, tmp_path, **environment):
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **INTERPRETER_ENV,
            "PATH": os.environ["PATH"],
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
            "GITHUB_REPOSITORY_OWNER": "example",
            **environment,
        },
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("version", ["1.16.0", "1.16.0-rc.1", "1.17.0-dev"])
def test_release_tags_pair_with_the_router_and_candidates_keep_latest(tmp_path, version):
    step = next(s for s in steps("release-docker.yml") if s.get("id") == "tags")
    result = run_step(step, tmp_path, VERSION=version, OWNER="example")
    assert result.returncode == 0, result.stderr
    found = outputs(tmp_path / "output")
    assert found["ol9_tags"] == [tag + "-ol9" for tag in found["tags"]]
    assert f"lightseekorg/smg:{version}-ol9" in found["ol9_tags"]
    assert ("lightseekorg/smg:latest-ol9" in found["ol9_tags"]) == ("-" not in version)


@pytest.mark.parametrize("express", ["true", "false"])
def test_nightly_pairs_ol9_tags_and_express_does_not_move_the_floating_tag(tmp_path, express):
    step = next(s for s in steps("nightly-docker.yml") if s.get("id") == "tags")
    result = run_step(
        step, tmp_path, OWNER="example", EXPRESS=express, GITHUB_SHA="0123456789abcdef"
    )
    assert result.returncode == 0, result.stderr
    found = outputs(tmp_path / "output")
    assert found["ol9_list"] == [tag + "-ol9" for tag in found["list"]]
    assert ("ghcr.io/example/smg:nightly-ol9" in found["ol9_list"]) == (express == "false")


@pytest.mark.parametrize("event,expected", [("push", "false"), ("workflow_dispatch", "true")])
def test_only_a_manual_release_dispatch_can_push(tmp_path, event, expected):
    release = steps("release-docker.yml")
    gate = next(s for s in release if s.get("id") == "should-push")
    assert run_step(gate, tmp_path, EVENT_NAME=event).returncode == 0
    assert outputs(tmp_path / "output")["push"] == expected
    ol9 = next(s for s in release if s.get("id") == "build-ol9")
    assert ol9["with"]["push"] == "${{ steps.should-push.outputs.push }}"


@pytest.mark.parametrize(
    "name,tag_output", [("release-docker.yml", "ol9_tags"), ("nightly-docker.yml", "ol9_list")]
)
def test_ol9_build_uses_its_dockerfile_platform_tags_and_separate_cache(name, tag_output):
    job = workflow(name)["jobs"]["build-and-push"]
    assert job["if"] == "github.repository == 'smg-project/smg'"
    ol9 = next(s for s in job["steps"] if s.get("with", {}).get("file") == "docker/ol9.Dockerfile")
    assert ol9["if"] == "${{ !cancelled() && steps.build.outcome == 'success' }}"
    config = ol9["with"]
    assert config["context"] == "." and config["target"] == "router-image"
    assert config["tags"] == "${{ steps.tags.outputs." + tag_output + " }}"
    assert config["platforms"] == "linux/amd64"
    assert config["cache-from"] == "type=gha,scope=smg-ol9-amd64"
    assert config["cache-to"] == "type=gha,scope=smg-ol9-amd64,mode=max"


def test_recipe_and_workflow_changes_trigger_release_build_validation():
    document = workflow("release-docker.yml")
    events = document.get("on", document.get(True))
    paths = events["push"]["paths"]
    assert "docker/ol9.Dockerfile" in paths
    assert ".github/workflows/release-docker.yml" in paths


@pytest.mark.parametrize("version,latest", [("1.16.0", "true"), ("1.16.0-rc.1", "false")])
@pytest.mark.parametrize("missing,fail_copy", [(False, False), (True, False), (False, True)])
def test_mirror_copies_ol9_reports_missing_images_and_propagates_copy_failures(
    tmp_path, version, latest, missing, fail_copy
):
    package = tmp_path / "bindings" / "python"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(f'version = "{version}"\n')
    binary = tmp_path / "docker"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['DOCKER_LOG'], 'a') as log: log.write(json.dumps(args) + '\\n')\n"
        "if (args[2] == 'inspect' and args[3].startswith('lightseekorg/')\n"
        "    and args[3].endswith('-ol9') and os.environ['MISSING_OL9'] == 'true'): sys.exit(1)\n"
        "if args[2] == 'create' and args[-1].endswith('-ol9') and os.environ['FAIL_OL9_COPY'] == 'true': sys.exit(7)\n"
        f"if args[2] == 'inspect': print({DIGEST!r})\n"
    )
    binary.chmod(0o755)
    step = next(s for s in steps("release-docker.yml") if s.get("id") == "mirror")
    result = run_step(
        step,
        tmp_path,
        VERSION=version,
        LATEST=latest,
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        DOCKER_LOG=str(tmp_path / "calls"),
        MISSING_OL9=str(missing).lower(),
        FAIL_OL9_COPY=str(fail_copy).lower(),
    )
    assert result.returncode == (7 if fail_copy else 0), result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    creates = [call for call in calls if call[2] == "create" and call[-1].endswith("-ol9")]
    found = outputs(tmp_path / "output")
    assert found["digest"] == DIGEST
    if fail_copy:
        assert len(creates) == 1 and "digest_ol9" not in found
    elif missing:
        assert not creates and "digest_ol9" not in found
        assert "-ol9 variant was not mirrored" in result.stdout
    else:
        assert found["digest_ol9"] == DIGEST
        assert len(creates) == 1
        assert creates[0][-1] == f"lightseekorg/smg:{version}-ol9"
        assert f"ghcr.io/example/smg:{version}-ol9" in creates[0]
        assert ("ghcr.io/example/smg:latest-ol9" in creates[0]) == (latest == "true")


@pytest.mark.parametrize(
    "push,outcome,digest",
    [
        ("true", "success", DIGEST),
        ("true", "success", ""),
        ("true", "failure", ""),
        ("true", "skipped", ""),
        ("false", "success", ""),
    ],
)
@pytest.mark.parametrize("latest", ["true", "false"])
def test_summary_claims_publication_only_after_a_push_with_a_digest(
    tmp_path, push, outcome, digest, latest
):
    step = next(s for s in steps("release-docker.yml") if s.get("name") == "Oracle Linux 9 summary")
    result = run_step(
        step,
        tmp_path,
        PUSH=push,
        OL9_OUTCOME=outcome,
        OL9_DIGEST=digest,
        VERSION="1.16.0",
        LATEST=latest,
    )
    assert result.returncode == 0, result.stderr
    text = (tmp_path / "summary").read_text()
    published = push == "true" and outcome == "success" and bool(digest)
    assert ("| Docker Hub |" in text) == published
    assert ("| GHCR |" in text) == published
    assert ("latest-ol9" in text) == (published and latest == "true")
    if push == "false" and outcome == "success":
        assert "built and validated" in text
    elif not published:
        assert "**not** published" in text
