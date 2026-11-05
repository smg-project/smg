"""The Rust test shards of PR Test (SMG) agree with the workspace.

scripts/ci_rust_tests.sh holds the shard table and the rust-* path filters of
detect-changes (pr-test-rust.yml) select the shards a pull request runs. These
tests pin what keeps that split a faithful partition of `cargo test` over the
workspace: every workspace package is tested by exactly one shard, a shard's
filter covers every crate its packages depend on (transitively, every
dependency kind) plus the build infrastructure, and the matrix step maps the
matched filters to the shards.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tomllib
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "ci_rust_tests.sh"
WORKFLOW = REPO / ".github" / "workflows" / "pr-test-rust.yml"

# The deliberate splits of one package over several shards.
SPLIT_PACKAGES = {
    "smg": lambda shard: shard.startswith("gateway-"),
    "smg-radix-tree": lambda shard: shard in ("kv", "radix-fuzz"),
}
INFRA_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/config.toml",
    "scripts/ci_rust_tests.sh",
    ".github/workflows/pr-test-rust.yml",
)


def script(*args: str) -> str:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], check=True, capture_output=True, text=True, cwd=REPO
    ).stdout


def shard_table() -> list[dict]:
    rows = []
    for line in script("list").splitlines():
        shard, filt, needs, timeout = line.split()
        rows.append({"shard": shard, "filter": filt, "needs": needs, "test_timeout": int(timeout)})
    return rows


def path_filters() -> dict[str, list[str]]:
    """The detect-changes filters, each flattened to its patterns (YAML anchors included)."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["detect-changes"]["steps"]
    step = next(s for s in steps if str(s.get("uses", "")).startswith("dorny/paths-filter"))
    assert step["with"].get("predicate-quantifier") == "some-with-excludes", (
        "the `!` patterns need some-with-excludes"
    )

    def flatten(item):
        if isinstance(item, list):
            return [p for sub in item for p in flatten(sub)]
        return [item]

    return {
        name: flatten(patterns)
        for name, patterns in yaml.safe_load(step["with"]["filters"]).items()
    }


def glob_matches(pattern: str, path: str) -> bool:
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            regex += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            regex += "[^/]"
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return re.fullmatch(regex, path) is not None


def filter_matches(patterns: list[str], path: str) -> bool:
    """paths-filter under some-with-excludes: some positive pattern matches, no `!` pattern does."""
    positive = [p for p in patterns if not p.startswith("!")]
    negative = [p[1:] for p in patterns if p.startswith("!")]
    return any(glob_matches(p, path) for p in positive) and not any(
        glob_matches(p, path) for p in negative
    )


def workspace() -> dict[str, dict]:
    """Every workspace member: its directory and the member names it depends on through a path."""
    root = tomllib.loads((REPO / "Cargo.toml").read_text())
    members = {}
    by_dir = {}
    for member in root["workspace"]["members"]:
        manifest = tomllib.loads((REPO / member / "Cargo.toml").read_text())
        members[manifest["package"]["name"]] = {"dir": member, "manifest": manifest}
        by_dir[(REPO / member).resolve()] = manifest["package"]["name"]
    for name, info in members.items():
        deps = set()
        for dep_name, spec in table_items(info["manifest"]):
            if not isinstance(spec, dict):
                continue
            if "path" in spec:
                deps.add(by_dir[(REPO / info["dir"] / spec["path"]).resolve()])
            elif spec.get("workspace"):
                # A `workspace = true` dependency on a member goes through [workspace.dependencies].
                ws = root["workspace"]["dependencies"].get(dep_name)
                if isinstance(ws, dict) and "path" in ws:
                    deps.add(by_dir[(REPO / ws["path"]).resolve()])
        info["deps"] = deps - {name}
    return members


def table_items(manifest: dict):
    for k in ("dependencies", "dev-dependencies", "build-dependencies"):
        yield from manifest.get(k, {}).items()
    for target in manifest.get("target", {}).values():
        for k in ("dependencies", "dev-dependencies", "build-dependencies"):
            yield from target.get(k, {}).items()


def closure(members: dict[str, dict], names: list[str]) -> set[str]:
    seen, todo = set(), list(names)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        todo.extend(members[name]["deps"])
    return seen


class ShardTableTests(unittest.TestCase):
    def test_every_package_is_tested_by_exactly_one_shard(self):
        members = workspace()
        owners: dict[str, set[str]] = {}
        for row in shard_table():
            for package in script("packages", row["shard"]).split():
                owners.setdefault(package, set()).add(row["shard"])
        self.assertEqual(
            set(owners),
            set(members),
            "a shard names a package outside the workspace, or a member has no shard",
        )
        for name in members:
            shards = owners[name]
            if name in SPLIT_PACKAGES:
                self.assertTrue(
                    all(SPLIT_PACKAGES[name](s) for s in shards) and len(shards) > 1,
                    f"{name}: {sorted(shards)}",
                )
            else:
                self.assertEqual(len(shards), 1, f"{name} is tested by {sorted(shards)}")

    def test_shard_filters_cover_the_dependency_closures(self):
        members = workspace()
        filters = path_filters()
        for row in shard_table():
            patterns = filters[row["filter"]]
            for name in sorted(closure(members, script("packages", row["shard"]).split())):
                crate = members[name]["dir"]
                probes = ["Cargo.toml", "src/lib.rs", "src/main.rs"]
                # The target directories cargo discovers, where the crate has Rust sources in them.
                probes += [
                    f"{d}/probe.rs"
                    for d in ("tests", "benches", "examples")
                    if any((REPO / crate / d).rglob("*.rs"))
                ]
                probes += ["build.rs"] if (REPO / crate / "build.rs").exists() else []
                for rel in probes:
                    path = f"{crate}/{rel}"
                    self.assertTrue(
                        filter_matches(patterns, path),
                        f"shard {row['shard']}: filter {row['filter']} misses {path} (a dependency of the shard)",
                    )

    def test_shard_filters_cover_the_build_infrastructure(self):
        filters = path_filters()
        for row in shard_table():
            for path in INFRA_FILES:
                self.assertTrue(
                    filter_matches(filters[row["filter"]], path), f"{row['filter']} misses {path}"
                )

    def test_shard_filters_exclude_unrelated_crates(self):
        filters = path_filters()
        self.assertFalse(filter_matches(filters["rust-gateway"], "crates/radix_tree/src/lib.rs"))
        self.assertFalse(filter_matches(filters["rust-tokenizer"], "model_gateway/src/lib.rs"))
        self.assertFalse(filter_matches(filters["rust-multimodal"], "crates/tokenizer/src/lib.rs"))
        self.assertFalse(filter_matches(filters["rust-gateway"], "docs/README.md"))

    def test_the_gateway_filter_covers_the_sources_its_integration_tests_read(self):
        # The metrics catalogue test walks these roots for recorder and family
        # names, and the dashboard test reads the chart: a change there can
        # change their result, so it runs the gateway shards.
        filters = path_filters()
        for path in (
            "bindings/python/src/lib.rs",
            "bindings/golang/src/lib.rs",
            "clients/rust/src/lib.rs",
            "examples/wasm/wasm-guest-storage-hook-passthrough/src/lib.rs",
            "deploy/helm/smg/templates/grafana-dashboard.yaml",
        ):
            self.assertTrue(filter_matches(filters["rust-gateway"], path), path)
        # Only the Rust sources under those roots: a Python client, a Java
        # client or a README there changes no Rust test's result.
        for path in (
            "bindings/python/src/smg/router_args.py",
            "clients/python/smg_client/client.py",
            "clients/java/pom.xml",
            "examples/rl/refit_from_disk.py",
            "examples/wasm/README.md",
        ):
            self.assertFalse(filter_matches(filters["rust-gateway"], path), path)

    def test_every_shard_has_a_filter_and_setup_flags_the_workflow_knows(self):
        filters = path_filters()
        workflow = WORKFLOW.read_text()
        for row in shard_table():
            self.assertIn(row["filter"], filters)
            for need in [] if row["needs"] == "-" else row["needs"].split(","):
                self.assertIn(
                    f"contains(matrix.needs, '{need}')",
                    workflow,
                    f"{row['shard']} needs {need}; no step is conditioned on it",
                )
            self.assertGreater(row["test_timeout"], 0)

    def test_matrix_maps_matched_filters_to_shards(self):
        rows = shard_table()
        every = json.loads(script("matrix", "[]", "true"))
        self.assertEqual([r["shard"] for r in every], [r["shard"] for r in rows])
        self.assertEqual(every[0].keys(), {"shard", "filter", "needs", "test_timeout"})
        self.assertEqual(json.loads(script("matrix", '["docs","common"]', "false")), [])
        tokenizer = json.loads(script("matrix", '["rust-tokenizer","docs"]', "false"))
        self.assertEqual([r["shard"] for r in tokenizer], ["tokenizer"])
        gateway = json.loads(script("matrix", '["rust-gateway","rust-bindings"]', "false"))
        self.assertEqual(
            [r["shard"] for r in gateway],
            ["gateway-unit", "gateway-1", "gateway-2", "gateway-3", "bindings"],
        )
        multimodal = json.loads(script("matrix", '["rust-multimodal"]', "false"))
        self.assertEqual(
            multimodal,
            [
                {
                    "shard": "multimodal",
                    "filter": "rust-multimodal",
                    "needs": "golden,ffmpeg",
                    "test_timeout": 20,
                }
            ],
        )

    def test_commands_only_name_the_packages_of_the_shard(self):
        for row in shard_table():
            if row["shard"].startswith("gateway-") or row["shard"] in ("kv",):
                continue  # these expand cargo metadata; the script's `check` mode covers them
            commands = script("commands", row["shard"]).splitlines()
            named = {m for c in commands for m in re.findall(r"-p (\S+)", c)}
            self.assertEqual(named, set(script("packages", row["shard"]).split()), row["shard"])

    @unittest.skipUnless(shutil.which("cargo"), "needs cargo for the workspace metadata")
    def test_check_mode_accepts_the_table(self):
        out = script("check")
        self.assertIn(": OK", out.strip().splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
