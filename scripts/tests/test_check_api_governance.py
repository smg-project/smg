from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_api_governance.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_api_governance", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_api_governance"] = module
    spec.loader.exec_module(module)
    return module


def test_crate_manifests_discovers_nested_manifests_and_workspace_lints(tmp_path: Path) -> None:
    (tmp_path / "crates/known").mkdir(parents=True)
    (tmp_path / "crates/nested/also-known").mkdir(parents=True)
    (tmp_path / "crates/known/Cargo.toml").write_text("[lints]\nworkspace = true\n")
    (tmp_path / "crates/nested/also-known/Cargo.toml").write_text("[package]\nname = 'x'\n")

    module = _load()

    assert module.crate_manifests(tmp_path) == [
        module.ManifestRecord("crates/known", True),
        module.ManifestRecord("crates/nested/also-known", False),
    ]


def test_validate_inventory_requires_manifest_workspace_membership_classification_and_lints() -> (
    None
):
    module = _load()
    manifests = [
        module.ManifestRecord("crates/known", True),
        module.ManifestRecord("crates/not-a-member", False),
    ]
    packages = [module.PackageRecord("known", "crates/known", True, True)]
    entries = [
        module.InventoryEntry(
            "known",
            "crates/known",
            "published-library",
            True,
            "release-crates",
            "CODEOWNERS",
        )
    ]

    assert module.validate_inventory(manifests, packages, entries) == [
        "crates/ manifest is not a workspace member: crates/not-a-member",
        "crates/ manifest must use [lints] workspace = true: crates/not-a-member",
    ]


def test_validate_inventory_rejects_an_unclassified_workspace_crate() -> None:
    module = _load()
    manifests = [module.ManifestRecord("crates/new-crate", True)]
    packages = [module.PackageRecord("new-crate", "crates/new-crate", True, True)]

    assert module.validate_inventory(manifests, packages, []) == [
        "unclassified crates/ package: new-crate (crates/new-crate)"
    ]


def test_release_and_version_registries_must_match_release_governed_inventory() -> None:
    module = _load()
    entries = [
        module.InventoryEntry(
            "known",
            "crates/known",
            "published-library",
            True,
            "release-crates",
            "CODEOWNERS",
        )
    ]

    assert module.validate_release_coverage(entries, {}) == [
        "release-governed package missing from release-crates workflow: known"
    ]
    assert module.validate_version_registry_coverage(entries, {}) == [
        "release-governed package missing from version registry: known"
    ]

    entries.append(
        module.InventoryEntry(
            "private",
            "crates/private",
            "quality-only",
            False,
            "none",
            "CODEOWNERS",
        )
    )
    mappings = {
        "extra": "crates/extra",
        "known": "crates/wrong",
        "private": "crates/private",
    }

    assert module.validate_release_coverage(entries, mappings) == [
        "release-crates workflow references package absent from inventory: extra",
        "release-crates workflow references non-release-governed package: private",
        "release workflow path mismatch for known: expected crates/known, found crates/wrong",
    ]
    assert module.validate_version_registry_coverage(entries, mappings) == [
        "version registry references package absent from inventory: extra",
        "version registry references non-release-governed package: private",
        "version registry path mismatch for known: expected crates/known, found crates/wrong",
    ]


def test_check_accepts_current_repository_inventory() -> None:
    module = _load()

    assert module._run() == []
