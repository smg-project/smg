"""Central pip transaction for an explicitly prepared private AMD cohort.

This does not create provider inputs or qualify a backend. The full profile
caller must prepare the owned SDK and interpreter before invoking this API.
"""

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ci_verify_sglang_rocm_cohort import CONSTRAINT, CONSTRAINT_SHA, IMAGE

DESTINATION_OPTIONS = {"--prefix", "--target", "--root", "--user", "--python", "-t"}
TAIL_SWITCHES = {
    "--no-cache-dir",
    "--no-build-isolation",
    "--upgrade",
    "--no-deps",
    "--force-reinstall",
}
TAIL_VALUES = {
    "-e",
    "--editable",
    "--no-binary",
    "--only-binary",
    "--find-links",
    "--index-url",
    "--extra-index-url",
    "--constraint",
    "-c",
}
DESTINATION_ENV = {
    "PIP_PREFIX",
    "PIP_TARGET",
    "PIP_ROOT",
    "PIP_USER",
    "PIP_IGNORE_INSTALLED",
    "PIP_FORCE_REINSTALL",
    "PIP_CONFIG_FILE",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONPYCACHEPREFIX",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def directory(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_dir():
        raise ValueError("Existing unaliased owned directory required")
    return path


def install_command(manifest_path, arguments, *, cwd=None):
    """Validate the declared transaction without importing or executing pip."""
    manifest_path = Path(manifest_path)
    spec = json.loads(manifest_path.read_text())
    if spec.get("image") != IMAGE or not spec.get("isolated_children"):
        raise ValueError("Explicit fixed-image isolated cohort required")
    prefix = directory(spec["interpreter_prefix"])
    target = directory(spec["install_prefix"])
    if (
        target.is_relative_to(prefix)
        or not target.is_relative_to(prefix.parent)
        or prefix.is_relative_to(Path("/opt/venv"))
        or target.is_relative_to(Path("/opt/venv"))
    ):
        raise ValueError("Installation must target separate owned cohort data")
    version = spec["python_version"]
    if (
        not isinstance(version, list)
        or len(version) != 2
        or any(type(part) is not int or part < 0 for part in version)
    ):
        raise ValueError("Explicit cohort Python version required")
    site = directory(target / f"lib/python{version[0]}.{version[1]}/site-packages")
    if site.is_relative_to(prefix):
        raise ValueError("Install site must be outside the interpreter prefix")
    if str(site) not in spec["import_roots"]:
        raise ValueError("Install scheme is absent from the cohort import contract")
    python = prefix / "bin/python"
    bootstrap = manifest_path.with_name("ci_sglang_cohort.py")
    if (
        not manifest_path.resolve().is_relative_to(prefix)
        or digest(bootstrap) != spec["bootstrap_sha256"]
        or digest(python) != spec["launchers"][str(python)]
        or digest(spec["interpreter_core"]) != spec["interpreter_core_sha256"]
    ):
        raise ValueError("Private launcher/bootstrap/core Source differs")
    if (
        not isinstance(arguments, (list, tuple))
        or not arguments
        or any(not isinstance(a, str) or not a for a in arguments)
    ):
        raise ValueError("Complete nonempty pip arguments required")
    if any(a.split("=", 1)[0] in DESTINATION_OPTIONS for a in arguments):
        raise ValueError("Caller cannot override the private install destination")

    def editable(value):
        project = directory((Path.cwd() if cwd is None else Path(cwd)) / value.split("[", 1)[0])
        roots = spec.get("editable_projects", {}).get(str(project))
        if not roots or any(str(directory(root)) not in spec["import_roots"] for root in roots):
            raise ValueError("Editable project requires explicit cohort import mapping")

    needs_value = False
    option = None
    for argument in arguments:
        if needs_value:
            if argument.startswith("-"):
                raise ValueError("Missing explicit pip option value")
            if option in ("-e", "--editable"):
                editable(argument)
            needs_value = False
            continue
        if argument.startswith("-"):
            name, equal, value = argument.partition("=")
            if name in TAIL_VALUES:
                if equal and not value:
                    raise ValueError("Empty pip option value")
                if name in ("-e", "--editable") and equal:
                    editable(value)
                option = name
                needs_value = not equal
            elif name not in TAIL_SWITCHES or equal:
                raise ValueError("Undeclared pip option or destination abbreviation")
    if needs_value:
        raise ValueError("Missing explicit pip option value")
    if CONSTRAINT.is_symlink() or digest(CONSTRAINT) != CONSTRAINT_SHA:
        raise ValueError("Original Torch constraint Source differs")
    constraints = os.environ.get("PIP_CONSTRAINT", "")
    if str(CONSTRAINT) not in constraints.split():
        raise ValueError("Original Torch constraint input is absent")
    # pip accepts every long option through PIP_* as well as argv. Keep only
    # the original registry/TLS inputs; all behavior is explicit in arguments.
    registry = {
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_TRUSTED_HOST",
        "PIP_CERT",
        "PIP_CLIENT_CERT",
    }
    environment = {
        k: v
        for k, v in os.environ.items()
        if k not in DESTINATION_ENV and (not k.startswith("PIP_") or k in registry)
    }
    environment["PIP_CONSTRAINT"] = constraints
    environment["PIP_CONFIG_FILE"] = os.devnull
    # Actual captured pip metadata/__init__.py30-60 permits this override.
    environment["_PIP_USE_IMPORTLIB_METADATA"] = "1"
    for key, name in (
        ("HOME", "home"),
        ("TMPDIR", "tmp"),
        ("PIP_CACHE_DIR", "cache"),
        ("PIP_BUILD_TRACKER", "build-tracker"),
    ):
        value = directory(prefix.parent / name)
        environment[key] = str(value)
    environment["PATH"] = str(prefix / "bin") + os.pathsep + os.environ.get("PATH", "")
    # Keep normal dependency resolution. Borrowed distributions retain their
    # real outside-prefix locations, so pip's original local-uninstall guard
    # remains applicable; actual backend write effects need separate approval.
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-user",
        "--prefix",
        str(target),
        "--no-compile",
        "--no-build-isolation",
        *arguments,
    ]
    return command, environment


def installed_records(manifest_path):
    """Observe only the owned target; never import or uninstall a distribution."""
    spec = json.loads(Path(manifest_path).read_bytes())
    target = directory(spec["install_prefix"])
    major, minor = spec["python_version"]
    site = directory(target / f"lib/python{major}.{minor}/site-packages")
    found = {}
    for folder in site.glob("*.dist-info"):
        directory(folder)
        metadata, record = folder / "METADATA", folder / "RECORD"
        if any(path.is_symlink() or not path.is_file() for path in (metadata, record)):
            raise ValueError("Owned installed metadata/RECORD must be regular files")
        name = Parser().parsestr(metadata.read_text())["Name"]
        if not name:
            raise ValueError("Owned distribution name is missing")
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in found:
            raise ValueError("Owned package replacement produced duplicate metadata: " + name)
        paths = set()
        for row in csv.reader(record.read_text().splitlines()):
            if len(row) != 3 or not row[0] or Path(row[0]).is_absolute():
                raise ValueError("Invalid owned installed RECORD")
            path = site / row[0]
            if path.is_symlink() or not path.resolve().is_relative_to(target):
                raise ValueError("Owned installed RECORD escapes its target")
            paths.add(path.resolve())
        found[name] = paths
    return found


def owned_metadata_records(manifest_path):
    """Read the complete declared owned metadata view without importing it.

    Other roots need not use the target's RECORD scheme. In particular, source
    projects may expose egg-info while the target contains their PEP660 metadata.
    Borrowed provider sites are not added to this owned inventory.
    """
    spec = json.loads(Path(manifest_path).read_bytes())
    roots = {directory(p) for p in spec["import_roots"]}
    borrowed = [directory(p) for p in spec.get("borrowed_sites", [])]
    for package in spec.get("packages", []):
        path = Path(package["metadata_path"]).absolute()
        if any(path.is_relative_to(root) for root in borrowed) and not any(
            path.is_relative_to(root) for root in roots
        ):
            continue
        if path.name in ("METADATA", "PKG-INFO"):
            roots.add(directory(path.parent.parent))
        elif path.name.endswith(".egg-info"):
            roots.add(directory(path.parent))
        else:
            raise ValueError("Unsupported declared owned metadata layout")
    target = directory(spec["install_prefix"])
    major, minor = spec["python_version"]
    site = directory(target / f"lib/python{major}.{minor}/site-packages")
    roots.add(site)
    found = {}
    for root in sorted(roots):
        query = metadata.DistributionFinder.Context(path=[str(root)])
        for dist in metadata.MetadataPathFinder.find_distributions(query):
            if not isinstance(dist, metadata.PathDistribution):
                raise ValueError("Owned filesystem metadata origin required")
            origin = Path(dist._path).absolute()
            if origin.is_symlink() or origin != origin.resolve() or not origin.is_relative_to(root):
                raise ValueError("Owned metadata origin is aliased or escapes its root")
            if origin.is_dir():
                candidates = [origin / "METADATA", origin / "PKG-INFO"]
                if any(p.is_symlink() for p in candidates):
                    raise ValueError("Owned metadata body cannot be a symlink")
                body = next((p for p in candidates if p.is_file()), None)
            else:
                body = origin if origin.is_file() else None
            if body is None:
                raise ValueError("Owned metadata body is missing")
            fields = Parser().parsestr(body.read_text())
            if len(fields.get_all("Name", [])) != 1 or len(fields.get_all("Version", [])) != 1:
                raise ValueError("Owned distribution needs one name and version")
            name = re.sub(r"[-_.]+", "-", fields["Name"]).lower()
            version = fields["Version"].strip()
            if not name or not version:
                raise ValueError("Owned distribution name or version is empty")
            found[str(body)] = {
                "name": name,
                "version": version,
                "origin": str(origin),
                "metadata": str(body),
                "sha256": digest(body),
            }

    groups = {}
    for entry in found.values():
        groups.setdefault(entry["name"], []).append(entry)
    for entries in groups.values():
        if len(entries) == 1:
            continue
        targets = [
            e
            for e in entries
            if Path(e["origin"]).parent == site and Path(e["origin"]).name.endswith(".dist-info")
        ]
        if len(targets) != 1:
            raise ValueError("Ambiguous distributions across owned metadata roots")
        primary = targets[0]
        direct = Path(primary["origin"]) / "direct_url.json"
        if direct.is_symlink() or not direct.is_file():
            raise ValueError("Cross-root replacement needs a proven editable origin")
        declared = json.loads(direct.read_text())
        url = urlsplit(declared.get("url", ""))
        if (
            declared.get("dir_info", {}).get("editable") is not True
            or url.scheme != "file"
            or url.netloc not in ("", "localhost")
            or url.query
            or url.fragment
            or not Path(unquote(url.path)).is_absolute()
        ):
            raise ValueError("Cross-root replacement has no declared editable origin")
        project = directory(unquote(url.path))
        mapped = spec.get("editable_projects", {}).get(str(project), [])
        mapped = [directory(p) for p in mapped]
        if not mapped or any(p not in roots for p in mapped):
            raise ValueError("Editable metadata project has no explicit import mapping")
        for entry in entries:
            if entry is primary:
                continue
            origin = Path(entry["origin"])
            if (
                entry["version"] != primary["version"]
                or not origin.name.endswith(".egg-info")
                or not origin.is_relative_to(project)
                or not (origin.parent == project or any(origin.is_relative_to(p) for p in mapped))
            ):
                raise ValueError("Conflicting providers across owned metadata roots")
    return found


def install(manifest_path, arguments, *, cwd, timeout=600):
    """Install with full arguments; refuse ambiguous or stale owned replacements.

    pip's outside-venv uninstall guard also protects this sibling target. A
    replacement that requires old-file cleanup is unsupported: fail the job
    instead of reporting success or deleting unproven files. Retry in a fresh
    owned scope after that failure, never reuse the partial installation.
    """
    cwd = directory(cwd)
    command, environment = install_command(manifest_path, arguments, cwd=cwd)
    spec = json.loads(Path(manifest_path).read_bytes())
    scope = directory(spec["interpreter_prefix"]).parent
    marker = scope / ".private-pip-transaction"
    # Retain a failed/in-flight transaction across ci_retry and process exits.
    # Only a fully successful transaction removes its own exact marker.
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        identity = os.fstat(stream.fileno())
        json.dump({"manifest": str(manifest_path), "sha256": digest(manifest_path)}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    parent_fd = os.open(scope, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    before = installed_records(manifest_path)
    before_metadata = owned_metadata_records(manifest_path)
    subprocess.run(command, env=environment, cwd=cwd, check=True, timeout=timeout)
    after = installed_records(manifest_path)
    after_metadata = owned_metadata_records(manifest_path)
    target = directory(spec["install_prefix"])
    if any(
        not Path(path).is_relative_to(target) and path not in after_metadata
        for path in before_metadata
    ):
        raise ValueError("Owned metadata outside the target disappeared during install")
    claimed = set().union(*after.values()) if after else set()
    if any(name not in after for name in before):
        raise ValueError("Owned distribution disappeared during install")
    if any(
        path.exists() or path.is_symlink() for paths in before.values() for path in paths - claimed
    ):
        raise ValueError("Owned package replacement left stale files; fresh scope required")

    actual = marker.stat()
    if marker.is_symlink() or (actual.st_dev, actual.st_ino) != (identity.st_dev, identity.st_ino):
        raise ValueError("Owned transaction marker changed")
    marker.unlink()
    parent_fd = os.open(scope, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-requires", nargs="+")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.build_requires:
        import tomllib

        requirements = {"wheel"}
        for project in args.build_requires:
            with (Path(project) / "pyproject.toml").open("rb") as stream:
                data = tomllib.load(stream)
            declared = data.get("build-system", {}).get("requires", ["setuptools>=40.8.0"])
            if not isinstance(declared, list) or any(
                not isinstance(r, str) or not r for r in declared
            ):
                parser.error("Explicit complete build-system requirements required")
            requirements.update(declared)
        install(args.manifest, sorted(requirements), cwd=Path.cwd())
        return 0
    arguments = list(args.arguments)
    if arguments[:1] == ["--"]:
        arguments.pop(0)
    options = []
    while arguments[:1] == ["--no-cache-dir"]:
        options.append(arguments.pop(0))
    if arguments[:1] != ["install"]:
        parser.error("Only explicit private install transactions are supported")
    install(args.manifest, options + arguments[1:], cwd=Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
