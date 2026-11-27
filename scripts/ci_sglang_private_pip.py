"""Central pip transaction for an explicitly prepared private AMD cohort.

This does not create provider inputs or qualify a backend. The full profile
caller must prepare the owned SDK and interpreter before invoking this API.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

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


def install_command(manifest_path, arguments):
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
    site = target / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if str(directory(site)) not in spec["import_roots"]:
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
    needs_value = False
    for argument in arguments:
        if needs_value:
            if argument.startswith("-"):
                raise ValueError("Missing explicit pip option value")
            needs_value = False
            continue
        if argument.startswith("-"):
            name, equal, value = argument.partition("=")
            if name in TAIL_VALUES:
                if equal and not value:
                    raise ValueError("Empty pip option value")
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
    environment = {k: v for k, v in os.environ.items() if k not in DESTINATION_ENV}
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


def install(manifest_path, arguments, *, cwd, timeout=600):
    command, environment = install_command(manifest_path, arguments)
    subprocess.run(command, env=environment, cwd=directory(cwd), check=True, timeout=timeout)


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
