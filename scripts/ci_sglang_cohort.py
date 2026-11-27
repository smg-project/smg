#!/usr/bin/env python3
"""Fresh interpreter bootstrap primitive; does not install or qualify an SDK.

The caller supplies a reviewed package-origin manifest. Borrowed paths are added
directly, never through addsitedir, so their .pth files are not evaluated.
The original native/numeric verifier remains a separate required consumer.
"""

import hashlib
import importlib.machinery
import importlib.metadata as metadata
import importlib.util
import json
import os
import shlex
import shutil
import sys
import venv
from pathlib import Path

BOOTSTRAP_PTH = "import ci_sglang_cohort; ci_sglang_cohort.start()\n"
BACKEND_HOOKS = {
    "get_requires_for_build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "build_wheel",
    "build_sdist",
    "build_editable",
    "_supported_features",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(path):
    path = Path(path).absolute()
    if path != path.resolve() or not path.is_dir():
        raise ValueError("Expected an existing directory without symlink aliases")
    return path


def checked_file(path, root, expected):
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
        or digest(path) != expected
    ):
        raise ValueError("Package origin bytes or containment differ: " + str(path))
    return path.resolve()


def normalized(name):
    return name.lower().replace("_", "-").replace(".", "-")


def _check_child_launch(spec):
    if not spec.get("isolated_children"):
        return
    launchers = spec["launchers"]
    executable = Path(sys.executable)
    core = Path(spec["interpreter_core"])
    if (
        str(executable) not in launchers
        or executable.is_symlink()
        or digest(executable) != launchers[str(executable)]
        or core.is_symlink()
        or Path("/proc/self/exe").resolve() != core.resolve()
        or digest(core) != spec["interpreter_core_sha256"]
    ):
        raise ValueError("Owned Python launcher/core identity differs")
    if len(sys.argv) == 3 and sys.argv[1] in BACKEND_HOOKS:
        hook = spec.get("backend_hook")
        if not hook:
            raise ValueError("Declared backend hook Source required")
        script = Path(sys.argv[0])
        control = canonical(sys.argv[2])
        if (
            script.is_symlink()
            or script.resolve() != Path(hook["path"])
            or digest(script) != hook["sha256"]
            or not control.is_relative_to(Path(spec["interpreter_prefix"]).parent)
        ):
            raise ValueError("Backend hook Source or control directory differs")


def _isolated_launchers(prefix):
    binary = prefix / "bin"
    aliases = [p for p in binary.glob("python*") if p.is_file()]
    core = binary / "python-core"
    shutil.copyfile(binary / "python", core)
    core.chmod(0o755)
    launchers = {}
    for path in aliases:
        # Unlink only the newly owned venv aliases, never a borrowed target.
        path.unlink()
        path.write_text(
            "#!/bin/bash -p\nset -euo pipefail\nexec -a "
            + shlex.quote(str(path))
            + " "
            + shlex.quote(str(core))
            + ' -I -B "$@"\n'
        )
        path.chmod(0o755)
        launchers[str(path)] = digest(path)
    return {
        "isolated_children": True,
        "launchers": launchers,
        "interpreter_core": str(core),
        "interpreter_core_sha256": digest(core),
    }


class BorrowedPackages:
    """Read base modules/metadata without exposing its startup path.

    Owned roots take precedence through the normal PathFinder. Replaced base
    distributions are hidden together with their replaced top-level modules.
    This describes a dependency view, not a filesystem write sandbox.
    """

    def __init__(self, sites, replaced_distributions, replaced_modules, metadata_path):
        self.sites = [str(canonical(p)) for p in sites]
        self.replaced_distributions = replaced_distributions
        self.replaced_modules = replaced_modules | {"sitecustomize", "usercustomize"}
        self.metadata_path = metadata_path

    def find_spec(self, fullname, path=None, target=None):
        if path is not None or fullname.split(".")[0] in self.replaced_modules:
            return None
        owned = importlib.machinery.PathFinder.find_spec(fullname, sys.path)
        if owned is not None and owned.loader is not None:
            return owned
        borrowed = importlib.machinery.PathFinder.find_spec(fullname, self.sites)
        if borrowed is not None and borrowed.loader is not None:
            return borrowed
        locations = []
        for found in (owned, borrowed):
            if found is not None:
                for location in found.submodule_search_locations or ():
                    if location not in locations:
                        locations.append(location)
        if not locations:
            return None
        # Keep both declared providers when pytest changes sys.path; ordinary
        # namespace recalculation cannot discover the deliberately hidden base.
        result = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
        result.submodule_search_locations = locations
        return result

    def find_distributions(self, context=None):
        # An explicit path query must retain its ordinary meaning.
        if context is None or self.metadata_path not in context.path:
            return iter(())
        query = metadata.DistributionFinder.Context(name=context.name, path=self.sites)
        return (
            dist
            for dist in metadata.MetadataPathFinder.find_distributions(query)
            if normalized(dist.metadata["Name"]) not in self.replaced_distributions
        )


def activate(manifest_path):
    """Validate origins before importing protected packages in a fresh process."""
    spec = json.loads(Path(manifest_path).read_text())
    site = Path(__file__).resolve().parent
    if Path(sys.prefix).resolve() != Path(spec["interpreter_prefix"]):
        raise ValueError("Wrong cohort interpreter")
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        raise ValueError("Cohort requires the original -I and explicit -B")
    _check_child_launch(spec)
    cache = canonical(spec["pycache_prefix"])
    if not cache.is_relative_to(Path(sys.prefix).resolve()) or any(cache.iterdir()):
        raise ValueError("Expected empty interpreter-owned bytecode prefix")
    # -B prevents writes but still allows cached bytecode reads. Redirect reads
    # to an empty owned prefix, as the original native verifier already does.
    sys.pycache_prefix = str(cache)
    pth = site / "00-smg-cohort.pth"
    if list(site.glob("*.pth")) != [pth] or pth.read_text() != BOOTSTRAP_PTH:
        raise ValueError("Unreviewed cohort startup .pth")
    if digest(__file__) != spec["bootstrap_sha256"]:
        raise ValueError("Bootstrap source differs")
    for package in spec["packages"]:
        name = package["module"]
        if not name.isidentifier():
            raise ValueError("Only explicit top-level package origins are supported")
        if any(k == name or k.startswith(name + ".") for k in sys.modules):
            raise ValueError("Protected module was preloaded: " + name)
    if "pip_shim_sha256" in spec:
        shim = Path(sys.prefix) / "bin/pip"
        if shim.is_symlink() or digest(shim) != spec["pip_shim_sha256"]:
            raise ValueError("Private pip entry Source differs")
    for name, expected in spec.get("console_shim_sha256", {}).items():
        console = Path(sys.prefix) / "bin" / name
        if not name.isidentifier() or console.is_symlink() or digest(console) != expected:
            raise ValueError("Private console entry Source differs")
    roots = [canonical(p) for p in spec["import_roots"]]
    borrowed = [canonical(p) for p in spec.get("borrowed_sites", [])]
    if any(p in roots or str(p) in sys.path for p in borrowed):
        raise ValueError("Borrowed base sites must not be ordinary import roots")
    replaced = {
        normalized(dist.metadata["Name"])
        for dist in metadata.MetadataPathFinder.find_distributions(
            metadata.DistributionFinder.Context(path=[str(p) for p in roots])
        )
    }
    replaced_modules = {
        p["module"]
        for p in spec["packages"]
        if not any(Path(p["module_path"]).is_relative_to(base) for base in borrowed)
    }
    # Plain path insertion deliberately does not process borrowed .pth files.
    sys.path[:0] = [str(p) for p in roots]
    # The finder explicitly checks ordinary owned paths first, then merges
    # namespace providers without exposing the borrowed site as a startup path.
    if borrowed:
        metadata_path = canonical(spec["borrowed_metadata_path"])
        if not metadata_path.is_relative_to(Path(sys.prefix)) or any(metadata_path.iterdir()):
            raise ValueError("Expected empty owned borrowed-metadata path")
        # pip discovers distributions one sys.path location at a time. A
        # dedicated empty marker exposes this view to both that caller and
        # ordinary global metadata queries without exposing the base site.
        sys.path.append(str(metadata_path))
        position = sys.meta_path.index(importlib.machinery.PathFinder)
        sys.meta_path.insert(
            position, BorrowedPackages(borrowed, replaced, replaced_modules, str(metadata_path))
        )
    observed = {}
    for package in spec["packages"]:
        root = canonical(package["package_root"])
        source = checked_file(package["module_path"], root, package["source_sha256"])
        meta = checked_file(package["metadata_path"], root, package["metadata_sha256"])
        found = importlib.util.find_spec(package["module"])
        if found is None or found.origin is None or Path(found.origin).resolve() != source:
            raise ValueError("Resolved module origin differs")
        matches = [
            dist
            for dist in metadata.distributions()
            if normalized(dist.metadata["Name"]) == normalized(package["distribution"])
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous or missing distribution metadata")
        dist = matches[0]
        candidates = [
            Path(dist.locate_file(f)).resolve()
            for f in dist.files or ()
            if f.name == "METADATA" and any(p.endswith(".dist-info") for p in f.parts)
        ]
        if candidates != [meta] or normalized(dist.metadata["Name"]) != normalized(
            package["distribution"]
        ):
            raise ValueError("Resolved distribution metadata origin differs")
        if dist.version != package["version"]:
            raise ValueError("Distribution version differs")
        observed[package["module"]] = {
            "module_path": str(source),
            "metadata_path": str(meta),
            "source_sha256": package["source_sha256"],
            "metadata_sha256": package["metadata_sha256"],
            "version": dist.version,
        }
    return observed


def start():
    """Run our sole trusted .pth before Python looks for sitecustomize."""
    try:
        module = sys.modules[__name__]
        manifest = Path(__file__).with_name("smg-cohort.json")
        # Some Python site implementations process the venv site twice.
        # Recognize only our own unchanged bootstrap, never a foreign module.
        if sys.modules.get("sitecustomize") is module:
            if digest(manifest) != module.SMG_MANIFEST_SHA256:
                raise ValueError("Cohort manifest changed during startup")
            return
        if "sitecustomize" in sys.modules:
            raise ValueError("Foreign sitecustomize was already loaded")
        module.SMG_COHORT = activate(manifest)
        module.SMG_MANIFEST_SHA256 = digest(manifest)
        # Borrowed sitecustomize must not run after plain path insertion. This
        # interpreter's reviewed bootstrap owns the startup module instead.
        sys.modules["sitecustomize"] = module
    except BaseException as error:
        os.write(2, ("Cohort bootstrap refused: " + str(error) + "\n").encode())
        os._exit(78)


def verify_loaded(observed):
    """Check actual loaded top-level objects; metadata alone is insufficient."""
    for name, expected in observed.items():
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != Path(expected["module_path"]):
            raise ValueError("Loaded module origin differs: " + name)
        if digest(origin) != expected["source_sha256"]:
            raise ValueError("Loaded module source changed: " + name)


def create(
    prefix, import_roots, packages, *, borrowed_sites=(), isolated_children=False, backend_hook=None
):
    """Create only an empty no-pip venv plus its reviewed bootstrap files.

    No pip/backend/SDK code is run here. Installation and full native verifier
    path equivalence are intentionally not implemented by this primitive.
    """
    prefix = Path(prefix).absolute()
    canonical(prefix.parent)
    if prefix.exists() or prefix.is_symlink():
        raise ValueError("Fresh exclusive prefix required")
    roots = [str(canonical(p)) for p in import_roots]
    borrowed = [str(canonical(p)) for p in borrowed_sites]
    if set(roots) & set(borrowed):
        raise ValueError("Owned and borrowed data roots must be distinct")
    venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=False).create(prefix)
    sites = list(prefix.glob("lib/python*/site-packages"))
    if len(sites) != 1 or list(sites[0].iterdir()):
        raise ValueError("Unexpected fresh venv site layout")
    site = sites[0]
    target = site / "ci_sglang_cohort.py"
    shutil.copyfile(__file__, target)
    cache = prefix / "source-cache"
    cache.mkdir()
    spec = {
        "interpreter_prefix": str(prefix),
        "bootstrap_sha256": digest(target),
        "import_roots": roots,
        "borrowed_sites": borrowed,
        "pycache_prefix": str(cache),
        "packages": packages,
    }
    if isolated_children:
        spec.update(_isolated_launchers(prefix))
        spec["backend_hook"] = backend_hook
    if borrowed:
        metadata_path = prefix / "borrowed-metadata"
        metadata_path.mkdir()
        spec["borrowed_metadata_path"] = str(metadata_path)
    manifest = site / "smg-cohort.json"
    manifest.write_text(json.dumps(spec, indent=2) + "\n")
    # The interpreter site remains installation-free. All package payloads go
    # to separate data roots. Python catches .pth exceptions by default, so
    # start() must terminate on failure before any consumer can execute.
    (site / "00-smg-cohort.pth").write_text(BOOTSTRAP_PTH)
    return prefix / "bin/python"
