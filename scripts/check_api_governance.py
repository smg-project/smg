#!/usr/bin/env python3
"""Validate API inventory, all-crates coverage, and release parity."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "governance/api-surfaces.toml"
RELEASE_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/release-crates.yml"
VERSION_REGISTRY_PATH = REPO_ROOT / "scripts/check_release_versions.sh"
RELEASE_CRATE = re.compile(r"^\s*(?:-\s*)?crate:\s*([A-Za-z0-9_-]+)\s*$")
RELEASE_PATH = re.compile(r"^\s*path:\s*([A-Za-z0-9_./-]+)\s*$")
VERSION_REGISTRY_ENTRY = re.compile(
    r'^\s*"([A-Za-z0-9_-]+)\|([A-Za-z0-9_./-]+)\|([A-Za-z0-9_-]+)"\s*$'
)
CLASSIFICATION_RULES = {
    "published-library": (True, "release-crates"),
    "quality-only": (False, "none"),
    "public-sdk": (True, "release-crates"),
    "external-application": (False, "release-crates"),
    "version-locked-binding": (False, "core-version-sync"),
}


@dataclass(frozen=True)
class PackageRecord:
    name: str
    path: str
    publishable: bool
    has_lib_target: bool


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    has_workspace_lints: bool


@dataclass(frozen=True)
class InventoryEntry:
    name: str
    path: str
    classification: str
    semver: bool
    release: str
    owner: str


def _inventory_entries(data: Mapping[str, Any]) -> list[InventoryEntry]:
    """Build inventory entries from validated TOML data."""
    return [
        InventoryEntry(
            name=package["name"],
            path=package["path"],
            classification=package["classification"],
            semver=package["semver"],
            release=package["release"],
            owner=package["owner"],
        )
        for package in data.get("package", [])
    ]


def packages_from_metadata(metadata: dict[str, Any], repo_root: Path) -> list[PackageRecord]:
    """Normalize Cargo packages whose paths are rooted under ``crates/``."""
    root = repo_root.resolve()
    packages: list[PackageRecord] = []

    for package in metadata["packages"]:
        manifest_parent = Path(package["manifest_path"]).resolve().parent
        try:
            relative_path = manifest_parent.relative_to(root)
        except ValueError:
            continue

        if not relative_path.parts or relative_path.parts[0] != "crates":
            continue

        packages.append(
            PackageRecord(
                name=package["name"],
                path=relative_path.as_posix(),
                publishable=package.get("publish") != [],
                has_lib_target=any(
                    "lib" in target.get("kind", []) for target in package.get("targets", [])
                ),
            )
        )

    return sorted(packages, key=lambda package: package.name)


def crate_manifests(repo_root: Path) -> list[ManifestRecord]:
    """Discover every Cargo manifest under ``crates/`` and its lint inheritance."""
    manifests: list[ManifestRecord] = []
    for manifest in sorted((repo_root / "crates").rglob("Cargo.toml")):
        relative_path = manifest.parent.relative_to(repo_root).as_posix()
        data = tomllib.loads(manifest.read_text())
        manifests.append(
            ManifestRecord(
                relative_path,
                data.get("lints", {}).get("workspace") is True,
            )
        )
    return manifests


def release_crates(workflow_text: str) -> dict[str, str]:
    """Return crate-to-path mappings from release-workflow entries."""
    registry: dict[str, str] = {}
    paths: set[str] = set()
    lines = workflow_text.splitlines()

    for index, line in enumerate(lines):
        crate_match = RELEASE_CRATE.fullmatch(line)
        if crate_match is None:
            continue

        name = crate_match.group(1)
        path: str | None = None
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path_match = RELEASE_PATH.fullmatch(following)
            if path_match is not None:
                path = path_match.group(1)
            break

        if path is None:
            raise ValueError(f"release-workflow crate has no adjacent path: {name}")
        if name in registry:
            raise ValueError(f"duplicate release-workflow crate: {name}")
        if path in paths:
            raise ValueError(f"duplicate release-workflow path: {path}")
        registry[name] = path
        paths.add(path)

    return registry


def version_registry_crates(script_text: str) -> dict[str, str]:
    """Return package-to-path mappings from the shell script's ``CRATES`` array."""
    registry: dict[str, str] = {}
    in_crates = False

    for line_number, line in enumerate(script_text.splitlines(), start=1):
        stripped = line.strip()
        if not in_crates:
            if stripped == "CRATES=(":
                in_crates = True
            continue

        if stripped == ")":
            return registry
        if not stripped or stripped.startswith("#"):
            continue

        match = VERSION_REGISTRY_ENTRY.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid CRATES registry entry on line {line_number}")

        name, path, _workspace_dependency = match.groups()
        if name in registry:
            raise ValueError(f"duplicate CRATES registry package: {name}")
        registry[name] = path

    if in_crates:
        raise ValueError("unterminated CRATES registry array")
    raise ValueError("CRATES registry array not found")


def validate_inventory(
    manifests: Sequence[ManifestRecord],
    packages: Sequence[PackageRecord],
    entries: Sequence[InventoryEntry],
) -> list[str]:
    """Return deterministic metadata-to-inventory validation errors."""
    errors: list[str] = []
    entries_by_name: dict[str, InventoryEntry] = {}
    entries_by_path: dict[str, InventoryEntry] = {}

    for entry in sorted(entries, key=lambda item: (item.name, item.path)):
        if entry.name in entries_by_name:
            errors.append(f"duplicate inventory package name: {entry.name}")
        else:
            entries_by_name[entry.name] = entry

        if entry.path in entries_by_path:
            errors.append(f"duplicate inventory package path: {entry.path}")
        else:
            entries_by_path[entry.path] = entry

    packages_by_name = {package.name: package for package in packages}
    packages_by_path = {package.path: package for package in packages}

    for manifest in sorted(manifests, key=lambda item: item.path):
        if manifest.path not in packages_by_path:
            errors.append(f"crates/ manifest is not a workspace member: {manifest.path}")
        if not manifest.has_workspace_lints:
            errors.append(f"crates/ manifest must use [lints] workspace = true: {manifest.path}")

    for entry in sorted(entries, key=lambda item: item.name):
        rule = CLASSIFICATION_RULES.get(entry.classification)
        if rule is None:
            errors.append(
                f"unsupported inventory classification for {entry.name}: {entry.classification}"
            )
            continue

        expected_semver, expected_release = rule
        if entry.semver != expected_semver:
            errors.append(
                f"{entry.classification} package {entry.name} must set semver = "
                f"{'true' if expected_semver else 'false'}"
            )
        if entry.release != expected_release:
            errors.append(
                f"{entry.classification} package {entry.name} must set release = "
                f'"{expected_release}"'
            )

    for package in sorted(packages, key=lambda item: item.name):
        entry = entries_by_name.get(package.name)
        if entry is None:
            errors.append(f"unclassified crates/ package: {package.name} ({package.path})")
            continue

        if entry.path != package.path:
            errors.append(
                f"inventory path mismatch for {package.name}: "
                f"expected {package.path}, found {entry.path}"
            )

        if package.publishable and entry.classification != "published-library":
            errors.append(f"publishable crate {package.name} must be classified published-library")
        if not package.publishable and entry.classification == "published-library":
            errors.append(f"private crate {package.name} cannot be classified published-library")
        if entry.classification == "published-library" and not package.has_lib_target:
            errors.append(f"published-library crate {package.name} has no lib target")

    for entry in sorted(entries, key=lambda item: item.name):
        if not entry.path.startswith("crates/"):
            continue

        package_by_name = packages_by_name.get(entry.name)
        package_by_path = packages_by_path.get(entry.path)
        if package_by_name is None and package_by_path is None:
            errors.append(
                f"inventory references missing crates/ package: {entry.name} ({entry.path})"
            )

    return errors


def validate_release_coverage(
    entries: Sequence[InventoryEntry], workflow_crates: Mapping[str, str]
) -> list[str]:
    """Return governed release entries missing from or drifting in the workflow."""
    errors: list[str] = []
    entries_by_name = {entry.name: entry for entry in entries}

    for name in sorted(workflow_crates):
        entry = entries_by_name.get(name)
        if entry is None:
            errors.append(
                f"release-crates workflow references package absent from inventory: {name}"
            )
        elif entry.release != "release-crates":
            errors.append(
                f"release-crates workflow references non-release-governed package: {name}"
            )

    for entry in sorted(entries, key=lambda item: item.name):
        if entry.release != "release-crates":
            continue
        workflow_path = workflow_crates.get(entry.name)
        if workflow_path is None:
            errors.append(
                f"release-governed package missing from release-crates workflow: {entry.name}"
            )
        elif workflow_path != entry.path:
            errors.append(
                f"release workflow path mismatch for {entry.name}: "
                f"expected {entry.path}, found {workflow_path}"
            )
    return errors


def validate_version_registry_coverage(
    entries: Sequence[InventoryEntry], registry_crates: Mapping[str, str]
) -> list[str]:
    """Return release-governed entries missing from or drifting in CRATES."""
    errors: list[str] = []
    entries_by_name = {entry.name: entry for entry in entries}

    for name in sorted(registry_crates):
        entry = entries_by_name.get(name)
        if entry is None:
            errors.append(f"version registry references package absent from inventory: {name}")
        elif entry.release != "release-crates":
            errors.append(f"version registry references non-release-governed package: {name}")

    for entry in sorted(entries, key=lambda item: item.name):
        if entry.release != "release-crates":
            continue

        registry_path = registry_crates.get(entry.name)
        if registry_path is None:
            errors.append(f"release-governed package missing from version registry: {entry.name}")
        elif registry_path != entry.path:
            errors.append(
                f"version registry path mismatch for {entry.name}: "
                f"expected {entry.path}, found {registry_path}"
            )
    return errors


def _cargo_metadata() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "cargo",
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--locked",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _inventory_schema_error(data: Mapping[str, Any]) -> str | None:
    if data.get("schema-version") != 1:
        return "unsupported API surface inventory schema-version; expected 1"
    return None


def _run() -> list[str]:
    inventory_data = tomllib.loads(INVENTORY_PATH.read_text())
    schema_error = _inventory_schema_error(inventory_data)
    if schema_error is not None:
        return [schema_error]

    entries = _inventory_entries(inventory_data)
    packages = packages_from_metadata(_cargo_metadata(), REPO_ROOT)
    manifests = crate_manifests(REPO_ROOT)
    errors = validate_inventory(manifests, packages, entries)

    if errors:
        return errors

    workflow_crates = release_crates(RELEASE_WORKFLOW_PATH.read_text())
    errors.extend(validate_release_coverage(entries, workflow_crates))
    registry_crates = version_registry_crates(VERSION_REGISTRY_PATH.read_text())
    errors.extend(validate_version_registry_coverage(entries, registry_crates))
    if errors:
        return errors

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate SMG API surface governance")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)

    try:
        errors = _run()
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        errors = [f"API governance check failed: {exc}"]

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
