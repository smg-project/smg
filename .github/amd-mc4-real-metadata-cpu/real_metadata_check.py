"""Real stock-image editable metadata proof; source-only until root admission.

Run once inside a scheduled CPU-only stock749f job container as root. No engine
imports, mocks, package installs, network, shared ownership/configuration writes.
Only the newly created fixture trees are chowned/removed. Raw evidence survives.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time

HEAD = "a03e3dba902a0be3caedd2b65f50af7a2804d3e6"
HELPER = ".github/amd-nixl-environment/nixl_environment/runtime_inputs.py"
HELPER_SHA = "67924e120f74b58b9cf11139caca779ee726a7623d5933799de96c4c666dc15a"
START = time.monotonic()
WORK_SECONDS = 155  # NEW180 = 155 work + 5 kill grace + 20 cleanup, outer-owned.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--carrier-head", required=True)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    require(os.geteuid() == 0 and os.getegid() == 0, "root build caller required")
    require(len(args.nonce) == 32 and all(c in "0123456789abcdef" for c in args.nonce), "nonce32")
    require(len(args.carrier_head) == 40 and all(c in "0123456789abcdef" for c in args.carrier_head), "exact carrier HEAD")
    source = args.source
    require(source.is_absolute() and source.resolve(strict=True) == source and source.is_dir(), "canonical source checkout")
    evidence = args.evidence
    require(evidence.is_absolute() and not os.path.lexists(evidence), "new evidence path required")
    evidence.mkdir(mode=0o700)
    serial = 0
    base = dict(os.environ, PIP_CONFIG_FILE="/dev/null", PIP_NO_INDEX="1",
                PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    # Git root's sudo exception must not silently trust uid995-owned fixtures.
    for name in list(base):
        if name.startswith(("GIT_", "SUDO_", "SETUPTOOLS_SCM_", "VCS_VERSIONING_")):
            del base[name]
    require(base.get("HOME") == os.environ.get("HOME"), "HOME unchanged")

    def run(label, argv, *, env=base, cwd=Path("/tmp"), timeout=25, private=False):
        nonlocal serial
        serial += 1
        remaining = WORK_SECONDS - (time.monotonic() - START)
        require(remaining > 0, "work deadline")
        stem = evidence / (f"{serial:02d}-" + label)
        record = {"argv": list(map(str, argv)), "cwd": str(cwd), "timeout": min(timeout, remaining),
                  "uid": os.geteuid(), "gid": os.getegid(), "PATH": env.get("PATH"),
                  "HOME": env.get("HOME"), "SUDO_UID": env.get("SUDO_UID")}
        with (tempfile.TemporaryFile() if private else stem.with_suffix(".stdout").open("xb")) as out, (tempfile.TemporaryFile() if private else stem.with_suffix(".stderr").open("xb")) as err:
            try:
                child = subprocess.run(list(map(str, argv)), cwd=cwd, env=env,
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       timeout=min(timeout, remaining), check=False)
                record["returncode"] = child.returncode
                out.flush()
                err.flush()
                if private:
                    out.seek(0)
                    err.seek(0)
                    raw_out, raw_err = out.read(2 * 1024**2 + 1), err.read(2 * 1024**2 + 1)
                    require(max(len(raw_out), len(raw_err)) <= 2 * 1024**2, "private config output bound")
                    record["private_stdout_sha256"] = hashlib.sha256(raw_out).hexdigest()
                    record["private_stderr_sha256"] = hashlib.sha256(raw_err).hexdigest()
            finally:
                record["elapsed_total"] = time.monotonic() - START
                stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        return (child.returncode, raw_out, raw_err) if private else (child.returncode, stem.with_suffix(".stdout").read_bytes(), stem.with_suffix(".stderr").read_bytes())

    def git(label, tail, *, cwd=source, env=base, accepted=True):
        result = run(label, ["/usr/bin/git", "-c", "safe.directory=", "-c", "safe.directory=" + str(cwd), *tail], cwd=cwd, env=env)
        require(not accepted or result[0] == 0, label + " Git failed")
        return result

    def config_snapshot(label):
        # Values can contain checkout credentials. Read privately, preserve only
        # hashes and file metadata, and never publish the config content.
        rc, raw, _ = run(label, ["/usr/bin/git", "config", "--null", "--show-origin", "--list"], cwd=Path("/tmp"), private=True)
        require(rc == 0, "read shared config")
        fields = raw.split(b"\0")
        require(fields[-1] == b"" and len(fields[:-1]) % 2 == 0, "config origin encoding")
        paths = {Path("/etc/gitconfig"), Path(base["HOME"]) / ".gitconfig",
                 Path(base.get("XDG_CONFIG_HOME", str(Path(base["HOME"]) / ".config"))) / "git/config"}
        for flag in ("--absolute-git-dir", "--git-common-dir"):
            raw_dir = git(label + "-" + flag[2:], ["rev-parse", "--path-format=absolute", flag])[1].decode().strip()
            directory = Path(raw_dir)
            require(directory.is_absolute(), "absolute local Git config directory")
            paths.update((directory / "config", directory / "config.worktree"))
        for origin in fields[:-1:2]:
            require(origin.startswith(b"file:"), "unexpected shared config origin")
            paths.add(Path(os.fsdecode(origin[5:])))
        result = {"HOME": base.get("HOME"), "XDG_CONFIG_HOME": base.get("XDG_CONFIG_HOME"), "raw_sha256": hashlib.sha256(raw).hexdigest(), "files": {}}
        for path in sorted(paths):
            if not os.path.lexists(path):
                result["files"][str(path)] = None
            else:
                st = path.lstat()
                require(path.is_file(), "config is not a regular resolved file")
                result["files"][str(path)] = {"mode": stat.S_IMODE(st.st_mode), "uid": st.st_uid, "gid": st.st_gid,
                    "link": os.readlink(path) if path.is_symlink() else None, "bytes": path.stat().st_size, "sha256": digest(path)}
        (evidence / (label + ".snapshot.json")).write_text(json.dumps(result, indent=2) + "\n")
        return result

    require(git("source-head", ["rev-parse", "HEAD"])[1].decode().strip() == args.carrier_head, "exact reviewed carrier head")
    git("source-product-ancestor", ["merge-base", "--is-ancestor", HEAD, args.carrier_head])
    require(git("source-clean", ["status", "--porcelain=v1", "--untracked-files=all"])[1] == b"", "clean staged source")
    require(digest(source / HELPER) == HELPER_SHA, "helper bytes")
    require(Path("/usr/bin/git").is_file() and os.access("/usr/bin/git", os.X_OK), "real Git")
    versions = {name: metadata.version(name) for name in ("setuptools", "setuptools-scm", "vcs-versioning", "pip")}
    require(versions["setuptools"] == "79.0.1" and versions["setuptools-scm"] == "10.3.4" and versions["vcs-versioning"] == "2.5.0", "exact stock SCM build chain")
    scm_sources = {}
    for package, distribution in (("setuptools_scm", "setuptools-scm"), ("vcs_versioning", "vcs-versioning")):
        dist = metadata.distribution(distribution)
        record = dist.read_text("RECORD")
        require(record, "stock SCM RECORD missing")
        rows = list(csv.reader(io.StringIO(record)))
        verified = {}
        for name, encoded, size in rows:
            if not name.startswith(package + "/") or not name.endswith(".py"):
                continue
            path = Path(dist.locate_file(name))
            require(path.is_file() and not path.is_symlink() and encoded.startswith("sha256=") and size.isdecimal(), "SCM source RECORD row")
            raw = path.read_bytes()
            require(len(raw) == int(size) and base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=") == encoded[7:], "SCM source RECORD mismatch")
            verified[name] = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        require(verified, "no actual SCM source")
        scm_sources.update(verified)
    for name, expected in {"vcs_versioning/_run_cmd.py": "ceb7e319a8130155a685675687ac8f17b5db95440b0379134750e8be598f567f",
                           "vcs_versioning/_file_finders/_git.py": "a8f374604fbcd92548f09f3e42205f59f61df7f0bd4478c6ee95edcbe66d6853",
                           "vcs_versioning/_backends/_git.py": "a4be1842455c4fe2fd916e87d14c58b6596f36d7516a52a3b6d47ad8ee77dc4b"}.items():
        require(scm_sources[name]["sha256"] == expected, "stock exported SCM chain changed")
    (evidence / "actual-SCM-sources-RECORD.json").write_text(json.dumps(scm_sources, indent=2) + "\n")
    (evidence / "stock-identity.json").write_text(json.dumps({"versions": versions, "python": sys.version,
        "git_sha256": digest(Path("/usr/bin/git")), "git_realpath": str(Path("/usr/bin/git").resolve()),
        "product_HEAD": HEAD, "carrier_HEAD": args.carrier_head, "source_helper_sha256": HELPER_SHA, "caller_uid_gid": [os.geteuid(), os.getegid()],
        "removed_inherited_identity_overrides": sorted(name for name in os.environ if name not in base)}, indent=2) + "\n")
    before = config_snapshot("config-before")
    fixture = Path("/tmp/smg-mc4-scm-proof-" + args.nonce)
    runtime = Path("/tmp/smg-mooncake4-runtime-" + str(int(args.nonce, 16)) + "-1")
    require(not os.path.lexists(fixture) and not os.path.lexists(runtime), "fixture/runtime already exists")
    fixture.mkdir(mode=0o700)
    created_runtime = False
    try:
        checkout, neighbor = fixture / "checkout", fixture / "neighbor outside exact ROOT"
        for label, path in (("checkout", checkout), ("neighbor", neighbor)):
            # Local shared objects are read-only inputs for this bounded fixture;
            # no source/host ownership or configuration is changed.
            git("clone-" + label, ["clone", "--shared", "--no-hardlinks", "--no-checkout", str(source), str(path)])
            git("detach-" + label, ["checkout", "--detach", HEAD], cwd=path)
            for directory, names, files in os.walk(path):
                os.chown(directory, 995, 985)
                for name in names + files:
                    os.chown(Path(directory) / name, 995, 985, follow_symlinks=False)
            require(path.stat().st_uid == 995 and (path / ".git").stat().st_uid == 995 and os.geteuid() != 995, "cross-owner fixture")
            require(git("fixture-head-" + label, ["rev-parse", "HEAD"], cwd=path)[1].decode().strip() == HEAD, "fixture exact HEAD")
        os.environ["GITHUB_WORKSPACE"] = str(checkout)
        os.environ["GITHUB_RUN_ID"] = str(int(args.nonce, 16))
        os.environ["GITHUB_RUN_ATTEMPT"] = "1"
        os.environ["SMG_CI_RUNTIME_STATE_DIR"] = str(runtime)
        runtime.mkdir(mode=0o700)
        created_runtime = True
        spec = importlib.util.spec_from_file_location("actual_mc4_runtime_inputs", source / HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = dict(base, GITHUB_WORKSPACE=str(checkout))
        old = module.checkout_git_environment(env)
        require(run("old-env-effective-before-SCM", ["/usr/bin/git", "rev-parse", "HEAD"], cwd=checkout, env=old)[0] == 0, "old Git environment itself is ineffective")
        # Actual old code's env is correct, then actual stock SCM deletes GIT_*.
        rc, _, err = run("old-git-dubious", ["/usr/bin/git", "rev-parse", "HEAD"], cwd=checkout, env=base)
        require(rc != 0 and b"dubious ownership" in err, "fixture must reproduce Git dubious ownership")
        pip = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--no-deps", "--no-build-isolation", "--no-index", "--no-cache-dir", "--no-compile", "-v"]
        rc, out, err = run("old-editable-metadata", [*pip, "--report", str(evidence / "old-report.json"), "-e", str(checkout / "grpc_servicer")], cwd=checkout, env=old, timeout=40)
        require(rc != 0 and b"dubious ownership" in out + err and b"git introspection failed" in out + err, "real old editable must fail at SCM dubious ownership")
        # Prove tracked bytes remain exact. No product/source file is rewritten
        # between old and new; each real pip command runs its metadata hook.
        git("old-tracked-clean", ["diff", "--exit-code", "HEAD", "--"], cwd=checkout)
        original_env = dict(env)
        with module.checkout_git_path(env) as new:
            wrapper = Path(new["PATH"].split(os.pathsep)[0]) / "git"
            wrapper_raw = wrapper.read_bytes()
            (evidence / "actual-private-git-wrapper").write_bytes(wrapper_raw)
            require(wrapper.parent.stat().st_mode & 0o777 == 0o700 and wrapper.stat().st_mode & 0o777 == 0o700, "private wrapper modes")
            require(new["HOME"] == base["HOME"] and env == original_env, "caller env/HOME changed")
            from vcs_versioning._run_cmd import no_git_env
            scrubbed = no_git_env(new)
            require("GIT_CONFIG_COUNT" not in scrubbed and scrubbed["PATH"] == new["PATH"], "actual stock SCM strips config and preserves PATH")
            rc, real_version, _ = run("real-git-version", ["/usr/bin/git", "--version"])
            require(rc == 0, "real Git version")
            rc, wrapped_version, _ = run("wrapped-git-version", ["git", "--version"], env=new)
            require(rc == 0 and wrapped_version == real_version, "version argument pass-through")
            probe = "space and 'quote' ; $ literal"
            rc, raw, _ = run("wrapped-original-arguments", ["git", "-c", "scm.probe=" + probe, "config", "--get", "scm.probe"], env=scrubbed)
            require(rc == 0 and raw.decode().rstrip("\n") == probe, "original argv preserved")
            rc, safe, _ = run("wrapped-exact-safe-directory", ["git", "config", "--get-all", "safe.directory"], env=scrubbed)
            require(rc == 0 and safe.decode().splitlines()[-2:] == ["", str(checkout)], "last trust entries reset + exact ROOT")
            rc, out, err = run("new-editable-metadata", [*pip, "--report", str(evidence / "new-report.json"), "-e", str(checkout / "grpc_servicer")], cwd=checkout, env=new, timeout=40)
            require(rc == 0 and b"Preparing editable metadata" in out and b"done" in out, "real new editable metadata must pass")
            report = json.loads((evidence / "new-report.json").read_text())
            require(len(report["install"]) == 1 and report["install"][0]["metadata"]["name"] == "smg-grpc-servicer"
                    and report["install"][0]["metadata"]["version"] == "0.14.1"
                    and report["install"][0]["download_info"] == {"url": (checkout / "grpc_servicer").as_uri(), "dir_info": {"editable": True}}, "real editable report source identity")
            rc, out, err = run("neighbor-editable-metadata", [*pip, "--report", str(evidence / "neighbor-report.json"), "-e", str(neighbor / "grpc_servicer")], cwd=neighbor, env=new, timeout=40)
            require(rc != 0 and b"dubious ownership" in out + err and b"git introspection failed" in out + err, "neighbor outside exact ROOT must still fail")
            require(wrapper.read_bytes() == wrapper_raw, "private wrapper changed")
        require(not os.path.lexists(wrapper.parent), "private PATH wrapper removed")
        require(env == original_env, "caller environment changed after context")
        git("new-tracked-clean", ["diff", "--exit-code", "HEAD", "--"], cwd=checkout)
        git("neighbor-tracked-clean", ["diff", "--exit-code", "HEAD", "--"], cwd=neighbor)
        require(config_snapshot("config-after") == before, "shared HOME/config bytes changed")
        require(digest(Path("/usr/bin/git")) == json.loads((evidence / "stock-identity.json").read_text())["git_sha256"], "real Git executable changed")
        require(not any(name.split(".")[0] in {"torch", "vllm", "sglang", "mooncake"} for name in sys.modules), "engine imported")
        result = {"result": "PASS_REAL_STOCK_EDITABLE_METADATA_ONLY", "old_editable": "FAIL_DUBIOUS_OWNER", "new_editable": "PASS", "neighbor_editable": "FAIL_DUBIOUS_OWNER", "full_CPU": False, "GPU": 0, "ABI": False}
    finally:
        shutil.rmtree(fixture)
        if created_runtime:
            runtime.rmdir()
        (evidence / "cleanup.json").write_text(json.dumps({"fixture_absent": not os.path.lexists(fixture), "runtime_absent": not os.path.lexists(runtime), "elapsed": time.monotonic() - START}, indent=2) + "\n")
    result["elapsed"] = time.monotonic() - START
    result["fixture_absent"] = not os.path.lexists(fixture)
    result["runtime_absent"] = not os.path.lexists(runtime)
    (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
