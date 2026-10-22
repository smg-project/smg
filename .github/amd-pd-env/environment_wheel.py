"""Reuse a target-built native wheel only with its exact successful CI evidence."""

import hashlib
import io
import json
import os
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import reuse_foundation as reuse
from reuse_foundation import require

OUT = Path("diagnostic-results")
IMAGE = "vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f"
SOURCE_SHA = "3e69fff74fe64f5ba2dfa31f67c6bdf26413342574047437d2ac520d35e9a489"


def junit(data, count, skipped):
    cases = list(ET.fromstring(data).iter("testcase"))
    require(
        len(cases) == count and sum(c.find("skipped") is not None for c in cases) == skipped,
        "environment suite case/skip count",
    )
    require(
        not any(
            c.find(tag) is not None
            for c in cases
            for tag in ["failure", "error", "rerunFailure", "rerunError"]
        ),
        "environment suite contains failures/reruns",
    )


def validate(files, environment):
    artifact = environment["artifact"]
    prefix = artifact["evidence_prefix"]
    require(prefix in {"", "diagnostic-results/"}, "unknown environment evidence prefix")

    def get(name):
        return files[prefix + name]

    require(
        environment["image"] == os.environ["SMG_QA_IMAGE"] == IMAGE, "environment image mismatch"
    )
    for phase in ["image-before", "installed", "final"]:
        record = json.loads(get(phase + ".json"))
        require(
            record["source_sha"] == record["checkout_sha"] == environment["producer_sha"]
            and record["run_id"] == str(environment["run_id"])
            and record["run_attempt"] == str(environment["run_attempt"])
            and record["configured_image"] == IMAGE
            and record["engine"] == "vllm",
            "environment metadata identity",
        )
    before = json.loads(get("origins-before.json"))
    for phase in ["installed", "final"]:
        after = json.loads(get("origins-" + phase + ".json"))
        require(
            before["records"] == after["records"] and before["active"] == after["active"],
            "producer engine/PyGObject metadata changed",
        )
    for name in ["pipcheck-before.txt", "pipcheck-after.txt"]:
        lines = get(name).decode().splitlines()
        cache_warning = (
            "WARNING: The directory '/github/home/.cache/pip' or its parent directory is not owned "
            "or is not writable by the current user. The cache has been disabled. Check the permissions "
            "and owner of that directory. If executing pip with sudo, you should use sudo's -H flag."
        )
        require(
            lines
            in [
                ["No broken requirements found."],
                [cache_warning, "No broken requirements found."],
            ],
            "environment pipcheck failed",
        )
    for suite, count, skipped in [("completions", 19, 2), ("embeddings", 24, 0), ("router", 10, 2)]:
        junit(get(suite + "/result.xml"), count, skipped)
    preflight = json.loads(get("pycairo/preflight.json"))
    versions = {
        version
        for result in preflight["apt"]
        for version in re.findall(r"(?m)^Setting up libcairo2:amd64 \(([^)]+)\)", result["stdout"])
    }
    require(
        len(versions) == 1
        and environment["cairo_runtime_package"] == "libcairo2=" + versions.pop(),
        "producer Cairo runtime version not bound",
    )
    proof = json.loads(get("pycairo/native-proof.json"))
    require(
        proof["pycairo"] == "1.29.2" and proof["source_sha256"] == SOURCE_SHA,
        "unknown native wheel source/version",
    )
    wheel_path = artifact["wheel_path"]
    require(
        wheel_path == prefix + "pycairo/wheel/" + Path(proof["wheel"]).name, "wheel evidence path"
    )
    require(
        re.fullmatch(
            r"pycairo-1\.29\.2-cp312-cp312-(?:linux_x86_64|manylinux[^/]*_x86_64)\.whl",
            Path(wheel_path).name,
        )
        and proof["python"].startswith("3.12."),
        "unknown target wheel ABI",
    )
    wheel = files[wheel_path]
    require(hashlib.sha256(wheel).hexdigest() == proof["wheel_sha256"], "native wheel hash")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        extensions = [name for name in archive.namelist() if name.endswith(".so")]
        require(
            len(extensions) == 1
            and hashlib.sha256(archive.read(extensions[0])).hexdigest()
            == proof["extension_sha256"],
            "producer native extension differs from wheel",
        )
    png = get("pycairo/native-2x2.png")
    require(
        png[:8] == b"\x89PNG\r\n\x1a\n"
        and struct.unpack(">II", png[16:24]) == (2, 2)
        and hashlib.sha256(png).hexdigest() == proof["png_sha256"],
        "producer PNG proof",
    )
    require(
        get("pycairo/installed-RECORD.txt")
        and proof["loaded_libcairo_sha256"]
        and proof["loaded_elf"]["returncode"] == 0
        and proof["loaded_elf_versions"]["returncode"] == 0,
        "missing producer native ABI evidence",
    )
    return before, proof, wheel_path


def main():
    reuse.load_budget()
    inputs = json.loads(reuse.INPUTS.read_text())
    require(inputs.get("ready") is True, "PD consumer not admitted")
    environment = inputs["environment_prerequisite"]
    verified_job = reuse.verify_environment_job(inputs)
    artifact = environment["artifact"]
    require(
        artifact["name"] == f"amd-vllm-repair-{environment['run_id']}", "environment artifact name"
    )
    require(
        artifact["destination"] == "diagnostic-results/environment-producer"
        and 0 < artifact["bytes"] <= 100 * 1024 * 1024,
        "environment archive destination/size",
    )
    with reuse.api_get(f"actions/artifacts/{artifact['id']}") as response:
        reuse.check_metadata(json.load(response), artifact, environment)
    files = reuse.unpack(reuse.archive_bytes(artifact), artifact)
    before, proof, wheel_path = validate(files, environment)
    destination = OUT / "environment-producer"
    require(
        not destination.exists() and not destination.is_symlink(), "existing environment output"
    )
    destination.mkdir(parents=True)
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    prefix = artifact["evidence_prefix"]
    subprocess.run(
        [
            sys.executable,
            ".github/amd-qa/closure_audit.py",
            "require-vllm",
            str(destination / prefix / "servicer-engine.xml"),
        ],
        check=True,
    )
    wheel_dir = OUT / "pycairo/wheel"
    wheel_dir.mkdir(parents=True)
    (wheel_dir / Path(wheel_path).name).write_bytes(files[wheel_path])
    (OUT / "environment-binding.json").write_text(
        json.dumps(
            {
                "environment": verified_job,
                "artifact_id": artifact["id"],
                "artifact_sha256": artifact["sha256"],
                "image": IMAGE,
                "producer_origins": before,
                "producer_native": proof,
                "wheel_sha256": proof["wheel_sha256"],
                "host_wheel_used": False,
                "cairo_runtime_package": environment["cairo_runtime_package"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
