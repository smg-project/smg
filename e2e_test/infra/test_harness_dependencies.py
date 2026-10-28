"""The harness's own dependency list covers what its infra and fixtures packages import.

``pip install ./e2e_test`` is how CI (and a reader of ``e2e_test/pyproject.toml``)
prepares the unit suite, which collects ``e2e_test/infra`` and ``e2e_test/fixtures``.
A third-party module imported unconditionally at the top of a module of either
package and missing from that list breaks the whole suite's collection, unless
another install of the job happens to bring it along.
"""

import ast
import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

E2E_ROOT = Path(__file__).resolve().parents[1]
# The packages the CI command collects, both prepared by ``pip install ./e2e_test``.
SCANNED_PACKAGES = (E2E_ROOT / "infra", E2E_ROOT / "fixtures")
# This package's own names and the modules the ``PYTHONPATH=e2e_test`` layout serves.
OWN_MODULES = {"infra", "fixtures", "e2e_test"}


def _normalized(name: str) -> str:
    """A distribution name as the packaging rules compare it (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    project = tomllib.loads((E2E_ROOT / "pyproject.toml").read_text())["project"]
    return {
        _normalized(re.split(r"[\s<>=!~;\[]", dep, maxsplit=1)[0])
        for dep in project["dependencies"]
    }


def _unconditional_top_level_imports(module: Path) -> set[str]:
    """The first component of every import statement at the module's top level,
    outside any ``try`` (a guarded import is optional by construction) and
    outside functions (a local import is paid for where it is used)."""
    names: set[str] = set()
    for node in ast.parse(module.read_text()).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_module_the_harness_packages_import_at_the_top_is_a_declared_dependency():
    declared = _declared_distributions()
    providers = importlib.metadata.packages_distributions()
    undeclared: dict[str, tuple[str, list[str]]] = {}
    modules = sorted(module for package in SCANNED_PACKAGES for module in package.glob("*.py"))
    for module in modules:
        if module.name.startswith("test_"):
            continue
        for name in _unconditional_top_level_imports(module):
            if name in sys.stdlib_module_names or name in OWN_MODULES or name == "__future__":
                continue
            # A module not installed here cannot be judged (its distribution is
            # unknown); the collection of this suite already needs the rest.
            distributions = sorted(_normalized(dist) for dist in providers.get(name, ()))
            if distributions and not set(distributions) & declared:
                undeclared[name] = (module.relative_to(E2E_ROOT).as_posix(), distributions)
    assert not undeclared, (
        "imported at the top of an infra or fixtures module but missing from e2e_test/pyproject.toml "
        f"dependencies (module: (file, distributions)): {undeclared}"
    )
