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
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
    "rust-toolchain.toml",
    ".cargo/config.toml",
    ".github/actions/setup-rust/action.yml",
    "scripts/ci_rust_tests.sh",
    "scripts/ci_install_rust.sh",
    "scripts/ci_apt_mirror.sh",
    "scripts/ci_apt_install.sh",
    "scripts/ci_apt_update.sh",
    "scripts/ci_retry.sh",
    "scripts/ci_install_build_deps.sh",
    "scripts/ci_install_protoc.sh",
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
                if crate == "model_gateway":
                    probes.append("src/cli/mod.rs")
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
        # The workflow's rust-ci filter is its own copy of this list (it comes
        # before the rust-infra anchor, so it cannot reuse it) and the only
        # gate for rust-lint and the aarch64 job, so it is read beside the
        # shard filters.
        filters = path_filters()
        for name in sorted({row["filter"] for row in shard_table()} | {"rust-ci"}):
            for path in INFRA_FILES:
                self.assertTrue(filter_matches(filters[name], path), f"{name} misses {path}")
        missing = sorted(set(filters["rust-infra"]) - set(filters["rust-ci"]))
        self.assertEqual(missing, [], f"rust-ci misses rust-infra patterns: {missing}")

    def test_shard_filters_exclude_unrelated_crates(self):
        filters = path_filters()
        self.assertFalse(filter_matches(filters["rust-gateway"], "crates/radix_tree/src/lib.rs"))
        self.assertFalse(filter_matches(filters["rust-tokenizer"], "model_gateway/src/lib.rs"))
        self.assertFalse(filter_matches(filters["rust-multimodal"], "crates/tokenizer/src/lib.rs"))
        self.assertFalse(filter_matches(filters["rust-gateway"], "docs/README.md"))

    def test_the_kv_filter_covers_the_source_the_policy_crates_test_reads(self):
        # The policy controller's binary test holds its allocator settings to
        # the gateway's constant, read from this file by path: a change there
        # can change its result, so it runs the kv shard.
        path = "model_gateway/src/observability/metrics.rs"
        test = REPO / "crates" / "policy_controller" / "tests" / "binary.rs"
        self.assertIn(f'"/../../{path}"', test.read_text())
        filters = path_filters()
        self.assertTrue(filter_matches(filters["rust-kv"], path), path)
        # Only that file: the rest of the gateway changes no test of the shard.
        self.assertFalse(filter_matches(filters["rust-kv"], "model_gateway/src/lib.rs"))

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

    def test_the_kv_shard_builds_the_example_policy_plugin_and_requires_it(self):
        # The policy crate's tests load the example plugin as its own build
        # line wrote it: the shard that runs them builds it first, requires
        # it, and runs on a change to the example or to the tool's installer.
        row = next(r for r in shard_table() if r["shard"] == "kv")
        self.assertEqual(row["needs"], "policy-guest,policy-import")
        workflow = WORKFLOW.read_text()
        # The step: the pinned tool, the line a user runs, then, inside the
        # guest's directory, the repository's formatter (nightly) and the
        # pinned clippy, which the workspace's own runs do not reach.
        step = workflow.split("- name: Build the example policy plugin", 1)[1]
        step = step.split("\n      - name:", 1)[0]
        for line in (
            "bash scripts/ci_install_wasm_tools.sh",
            "bash examples/wasm/wasm-guest-policy/build.sh",
            "cd examples/wasm/wasm-guest-policy",
            "rustup toolchain install nightly --profile minimal --component rustfmt",
            "cargo +nightly fmt -- --check",
            "cargo clippy --locked --target wasm32-unknown-unknown --target-dir target -- -D warnings",
        ):
            self.assertIn(line, step)
        self.assertIn(
            "SMG_REQUIRE_POLICY_GUEST: ${{ contains(matrix.needs, 'policy-guest') && '1' || '' }}",
            workflow,
        )
        filters = path_filters()
        for path in (
            "examples/wasm/wasm-guest-policy/src/lib.rs",
            "examples/wasm/wasm-guest-policy/Cargo.lock",
            "examples/wasm/wasm-guest-policy/build.sh",
            "scripts/ci_install_wasm_tools.sh",
        ):
            self.assertTrue(filter_matches(filters["rust-kv"], path), path)
        # The other guests are the agentic shard's: they do not run this one.
        self.assertFalse(
            filter_matches(filters["rust-kv"], "examples/wasm/wasm-guest-storage-hook/src/lib.rs")
        )

    def test_the_kv_shard_turns_on_every_feature_a_policy_test_target_requires(self):
        # cargo skips a test target whose required features are off, in
        # silence: the shard's line turns each of them on.
        manifest = tomllib.loads((REPO / "crates/policy_controller/Cargo.toml").read_text())
        package = manifest["package"]["name"]
        line = next(
            text
            for text in SCRIPT.read_text().splitlines()
            if f"-p {package} " in text and "--features" in text
        )
        todo = ["default"] + [
            feature.split("/", 1)[1]
            for feature in re.search(r"--features ([^\s\"]+)", line).group(1).split(",")
            if feature.startswith(f"{package}/")
        ]
        features = manifest.get("features", {})
        on = set()
        while todo:
            feature = todo.pop()
            if feature in on:
                continue
            on.add(feature)
            todo.extend(
                name
                for name in features.get(feature, [])
                if not name.startswith("dep:") and "/" not in name
            )
        for target in manifest.get("test", []):
            for need in target.get("required-features", []):
                self.assertIn(need, on, f"the kv shard would skip tests/{target['name']}.rs")

    def test_the_kv_shard_runs_the_policy_crates_library_with_the_in_process_engine(self):
        # The gateway is the one build that turns the policy crate's `in-process` feature on,
        # and its shards run the gateway's own cases. The kv shard runs the crate's library
        # under the features the gateway asks for on every target (its target table adds
        # `client` on Linux and macOS), or no line runs the cases behind `in-process`.
        gateway = tomllib.loads((REPO / "model_gateway" / "Cargo.toml").read_text())
        asked = gateway["dependencies"]["smg-policy-controller"]
        root = tomllib.loads((REPO / "Cargo.toml").read_text())
        self.assertTrue(asked["workspace"])
        self.assertIs(
            root["workspace"]["dependencies"]["smg-policy-controller"]["default-features"], False
        )
        self.assertIn("in-process", asked["features"])
        features = ",".join(sorted(asked["features"]))
        self.assertIn(
            f'echo "-p smg-policy-controller --lib --no-default-features --features {features}"',
            SCRIPT.read_text(),
        )

    def test_the_kv_shard_builds_the_example_controller_outside_the_workspace(self):
        # The example controller is also built where a user stands: its own
        # script writes it outside the workspace with the crate named by
        # git from a snapshot of the checkout, and its cases run there.
        row = next(r for r in shard_table() if r["shard"] == "kv")
        self.assertIn("policy-import", row["needs"].split(","))
        workflow = WORKFLOW.read_text()
        step = workflow.split("- name: Build the example controller outside the workspace", 1)[1]
        step = step.split("\n      - name:", 1)[0]
        for line in (
            "if: contains(matrix.needs, 'policy-import')",
            "bash examples/policy-controller/own-repository.sh "
            '"${RUNNER_TEMP}/own-controller" --snapshot',
            'cd "${RUNNER_TEMP}/own-controller"',
            "cargo test",
        ):
            self.assertIn(line, step)
        filters = path_filters()
        for path in (
            "examples/policy-controller/own-repository.sh",
            "examples/policy-controller/src/main.rs",
            "crates/policy_controller/build.rs",
        ):
            self.assertTrue(filter_matches(filters["rust-kv"], path), path)

    def test_the_example_controllers_script_writes_a_repository_of_its_own(self):
        example = REPO / "examples" / "policy-controller"
        own = example / "own-repository.sh"
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "own"
            subprocess.run(
                ["bash", str(own), str(dest), "file:///a/checkout"], check=True, capture_output=True
            )
            text = (dest / "Cargo.toml").read_text()
            manifest = tomllib.loads(text)
            # The crate by git from the source it is handed, and no key the
            # workspace alone resolves.
            self.assertEqual(
                manifest["dependencies"]["smg-policy-controller"],
                {"git": "file:///a/checkout", "features": ["controller"]},
            )
            self.assertNotIn("workspace", text)
            # The test-only dependencies at the versions the workspace names.
            workspace = tomllib.loads((REPO / "Cargo.toml").read_text())["workspace"][
                "dependencies"
            ]
            dev = manifest["dev-dependencies"]
            self.assertEqual(dev["serde_yaml"], workspace["serde_yaml"])
            # The allocator the example registers, as the workspace names it
            # and under the target rule the example's own manifest names.
            own_manifest = tomllib.loads((example / "Cargo.toml").read_text())
            (rule,) = own_manifest["target"]
            self.assertEqual(
                manifest["target"][rule]["dependencies"]["tikv-jemallocator"],
                workspace["tikv-jemallocator"],
            )
            self.assertEqual(dev["tokio"]["version"], workspace["tokio"]["version"])
            # The door case's dependencies: the mock engine by git from the
            # crate's own source, the client's transport as the workspace
            # names it, and the runtime features the example's manifest asks.
            self.assertEqual(dev["mock-worker"], {"git": "file:///a/checkout"})
            self.assertEqual(dev["tonic"], workspace["tonic"])
            self.assertEqual(
                dev["tokio"]["features"],
                own_manifest["dev-dependencies"]["tokio"]["features"],
            )
            # The decide case's dependency: the engine's client by git from
            # the crate's own source, which holds requests on the mock engine.
            self.assertEqual(dev["smg-grpc-client"], {"git": "file:///a/checkout"})
            # The crate's reference client, which asks the started binary its
            # decisions: a feature the tests alone turn on, in the example's
            # manifest and in the copy's.
            client = {"git": "file:///a/checkout", "features": ["client"]}
            self.assertEqual(dev["smg-policy-controller"], client)
            ours = own_manifest["dev-dependencies"]["smg-policy-controller"]
            self.assertEqual(ours["features"], ["client"])
            # The release profile the stock controller is built with, without
            # its per-package entries: a copy with none builds on cargo's
            # default profile, not the stock build's full LTO in one unit.
            release = tomllib.loads((REPO / "Cargo.toml").read_text())["profile"]["release"]
            release.pop("package", None)
            self.assertEqual(manifest["profile"], {"release": release})
            # The allocator's page setting, as the workspace's cargo
            # configuration holds it: without it an aarch64 build of the
            # copy runs on its build host's page size alone.
            key = "AARCH64_UNKNOWN_LINUX_GNU_JEMALLOC_SYS_WITH_LG_PAGE"
            ours = tomllib.loads((REPO / ".cargo" / "config.toml").read_text())["env"][key]
            written = tomllib.loads((dest / ".cargo" / "config.toml").read_text())
            self.assertEqual(written, {"env": {key: ours}})
            for name in (
                "src/main.rs",
                "tests/serve.rs",
                "policy.yaml",
                "Cargo.lock",
                "rust-toolchain.toml",
            ):
                self.assertTrue((dest / name).is_file(), name)
            # A directory that exists is refused.
            again = subprocess.run(["bash", str(own), str(dest)], capture_output=True, text=True)
            self.assertEqual(again.returncode, 2, again.stderr)
            self.assertIn("exists", again.stderr)
            # With no source named, the line is the guide's own.
            plain = Path(tmp) / "plain"
            subprocess.run(["bash", str(own), str(plain)], check=True, capture_output=True)
            line = next(
                entry
                for entry in (plain / "Cargo.toml").read_text().splitlines()
                if entry.startswith("smg-policy-controller")
            )
            self.assertIn(line, (example / "README.md").read_text())
            # With --snapshot the source is a repository of one revision
            # beside the copy, made from this checkout's tracked files. The
            # revision is the script's own: a host whose configuration signs
            # every revision (here with a program that fails) signs none.
            snap = Path(tmp) / "snap"
            count = int(os.environ.get("GIT_CONFIG_COUNT") or 0)
            signing = {
                **os.environ,
                "GIT_CONFIG_COUNT": str(count + 2),
                f"GIT_CONFIG_KEY_{count}": "commit.gpgsign",
                f"GIT_CONFIG_VALUE_{count}": "true",
                f"GIT_CONFIG_KEY_{count + 1}": "gpg.program",
                f"GIT_CONFIG_VALUE_{count + 1}": "false",
            }
            subprocess.run(
                ["bash", str(own), str(snap), "--snapshot"],
                check=True,
                capture_output=True,
                env=signing,
            )
            source = Path(tmp) / "snap-source"
            named = tomllib.loads((snap / "Cargo.toml").read_text())["dependencies"]
            self.assertEqual(named["smg-policy-controller"]["git"], f"file://{source}")
            for args, expected in (
                (("rev-list", "--count", "HEAD"), "1"),
                (("status", "--short"), ""),
            ):
                read = subprocess.run(
                    ["git", "-C", str(source), *args], check=True, capture_output=True, text=True
                )
                self.assertEqual(read.stdout.strip(), expected, args)
            for name in (
                "crates/policy_controller/Cargo.toml",
                "crates/worker/proto/common.proto",
            ):
                self.assertTrue((source / name).is_file(), name)
        # A directory inside the checkout is refused and never made: the
        # workspace would claim the package.
        inside = example / "own-repository-refused"
        refused = subprocess.run(["bash", str(own), str(inside)], capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2, refused.stderr)
        self.assertIn("inside this checkout", refused.stderr)
        self.assertFalse(inside.exists())

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
                    "needs": "golden,ffmpeg,turbojpeg",
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


class RunModeTests(unittest.TestCase):
    """`run <shard>` runs every invocation of the shard and fails at the end if any did."""

    def run_tokenizer(
        self, failing_calls: set[int]
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        """The tokenizer shard (two invocations, no metadata read) under a cargo stub on PATH that
        records each call's arguments and exits 101 on the calls numbered in ``failing_calls``."""
        with tempfile.TemporaryDirectory() as tmp:
            stubs = Path(tmp) / "bin"
            stubs.mkdir()
            calls = Path(tmp) / "calls"
            calls.write_text("")
            stub = stubs / "cargo"
            stub.write_text(
                "#!/usr/bin/env bash\n"
                'printf "%s\\n" "$*" >> "$CARGO_STUB_CALLS"\n'
                'n=$(wc -l < "$CARGO_STUB_CALLS")\n'
                'case " $CARGO_STUB_FAIL " in *" $n "*) exit 101 ;; esac\n'
                "exit 0\n"
            )
            stub.chmod(0o755)
            done = subprocess.run(
                ["bash", str(SCRIPT), "run", "tokenizer"],
                capture_output=True,
                text=True,
                cwd=REPO,
                env={
                    "PATH": f"{stubs}:/usr/bin:/bin",
                    "CARGO_STUB_CALLS": str(calls),
                    "CARGO_STUB_FAIL": " ".join(str(n) for n in sorted(failing_calls)),
                },
            )
            return done, calls.read_text().splitlines()

    def test_a_red_invocation_does_not_stop_the_ones_after_it_and_the_shard_still_fails(self):
        done, calls = self.run_tokenizer(failing_calls={1})
        self.assertEqual(
            calls,
            [
                "test --no-fail-fast -p llm-tokenizer",
                "test --no-fail-fast -p llm-tokenizer --features tokenizers-v1",
            ],
        )
        self.assertEqual(done.returncode, 101, done.stderr)

    def test_a_shard_whose_invocations_all_pass_exits_0(self):
        done, calls = self.run_tokenizer(failing_calls=set())
        self.assertEqual(len(calls), 2, done.stderr)
        self.assertEqual(done.returncode, 0, done.stderr)


KNOWN = REPO / "model_gateway" / "tests" / "policy_controller_known.tsv"

# A stand-in for cargo: the metadata of a gateway whose test binaries are the scenario's, and
# `cargo test` over them as libtest answers it (--exact, --skip and names; a binary's cases one
# line each on the standard output; cargo's own lines and a case's `stderr` text on the standard
# error; status 101 where a case fails). A case with a `policy` writes the line of a classic
# pick to the file SMG_TEST_CLASSIC_PICKS names, as the gateway's test build does: under the
# case's name, the name of its thread, or under the `thread` the scenario gives.
CARGO_STUB = r"""#!/usr/bin/env python3
import json
import os
import sys

scenario = json.load(open(os.environ["CARGO_STUB_SCENARIO"]))
args = sys.argv[1:]
picks = os.environ.get("SMG_TEST_CLASSIC_PICKS")
with open(os.environ["CARGO_STUB_CALLS"], "a") as calls:
    switch = os.environ.get("SMG_TEST_POLICY_CONTROLLER")
    calls.write(json.dumps({"args": " ".join(args), "switch": switch, "picks": bool(picks)}) + "\n")
if args[0] == "metadata":
    targets = [{"name": name, "kind": ["test"]} for name in scenario]
    print(json.dumps({"packages": [{"name": "smg", "targets": targets}]}))
    sys.exit(0)
split = args.index("--") if "--" in args else len(args)
cargo_args, test_args = args[:split], args[split + 1 :]
binaries = [cargo_args[i + 1] for i, arg in enumerate(cargo_args) if arg == "--test"]
skips = {test_args[i + 1] for i, arg in enumerate(test_args) if arg == "--skip"}
names = [
    arg
    for i, arg in enumerate(test_args)
    if not arg.startswith("--") and (i == 0 or test_args[i - 1] != "--skip")
]
status = 0
for binary in binaries:
    executable = f"{binary}-0123456789abcdef"
    print(f"     Running tests/{binary}.rs (target/debug/deps/{executable})", file=sys.stderr)
    for case, facts in scenario[binary].items():
        if case in skips or (names and case not in names):
            continue
        if "stderr" in facts:
            print(facts["stderr"], file=sys.stderr, flush=True)
        print(f"test {case} ... {facts['result']}", flush=True)
        if facts["result"] == "FAILED":
            status = 101
        if picks and "policy" in facts:
            with open(picks, "a") as file:
                file.write(f"{executable}\t{facts.get('thread', case)}\t{facts['policy']}\n")
sys.exit(status)
"""


class PolicyControllerRunTests(unittest.TestCase):
    """`policy-controller <shard>` runs a gateway shard's tests once more with the policy
    controller on and holds the run to the list of the cases known to differ."""

    # Four test binaries: by name, shard 1 of 3 is dealt alpha_tests and gamma_tests.
    SCENARIO = {
        "alpha_tests": {
            "one::passes": {"result": "ok"},
            "one::refused": {"result": "FAILED"},
            "one::pair": {"result": "ok", "policy": "random"},
        },
        "beta_tests": {"two::other": {"result": "FAILED"}},
        "delta_tests": {"three::plain": {"result": "ok"}},
        "gamma_tests": {"plain": {"result": "ok"}},
    }
    LIST = (
        "# a comment\n"
        "alpha_tests\tone::pair\tclassic\ta pair gateway\n"
        "alpha_tests\tone::refused\tfails\tthe start refuses it\n"
        "beta_tests\ttwo::other\tfails\tanother shard's\n"
    )

    def run_shard(
        self,
        scenario: dict | None = None,
        listed: str | None = None,
        shard: str = "gateway-1",
        hosted: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        with tempfile.TemporaryDirectory() as tmp:
            stubs = Path(tmp) / "bin"
            stubs.mkdir()
            stub = stubs / "cargo"
            stub.write_text(CARGO_STUB)
            stub.chmod(0o755)
            calls = Path(tmp) / "calls"
            calls.write_text("")
            scenario_file = Path(tmp) / "scenario.json"
            scenario_file.write_text(json.dumps(self.SCENARIO if scenario is None else scenario))
            known = Path(tmp) / "known.tsv"
            known.write_text(self.LIST if listed is None else listed)
            done = subprocess.run(
                ["bash", str(SCRIPT), "policy-controller", shard],
                capture_output=True,
                text=True,
                cwd=REPO,
                env={
                    "PATH": f"{stubs}:{Path(sys.executable).parent}:/usr/bin:/bin",
                    "CARGO_STUB_CALLS": str(calls),
                    "CARGO_STUB_SCENARIO": str(scenario_file),
                    "POLICY_CONTROLLER_KNOWN": str(known),
                    **({"GITHUB_ACTIONS": "true"} if hosted else {}),
                    # An interpreter built as a shared library starts only with its library path.
                    **{k: os.environ[k] for k in ("LD_LIBRARY_PATH",) if k in os.environ},
                },
            )
            return done, [json.loads(line) for line in calls.read_text().splitlines()]

    def changed(self, binary: str, case: str, **facts: str) -> dict:
        """The scenario with one case's facts replaced (no facts: the case is gone)."""
        scenario = json.loads(json.dumps(self.SCENARIO))
        if facts:
            scenario[binary][case] = facts
        else:
            del scenario[binary][case]
        return scenario

    def test_a_run_that_matches_the_list_is_green(self):
        done, calls = self.run_shard()
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("policy controller on, gateway-1: OK", done.stdout)
        self.assertEqual(
            calls[1:],
            [
                # The binary with a case listed as failing runs alone, without that case ...
                {
                    "args": "test --no-fail-fast -p smg --test alpha_tests "
                    "-- --exact --skip one::refused",
                    "switch": "on",
                    "picks": True,
                },
                # ... the others run as the shard runs them ...
                {
                    "args": "test --no-fail-fast -p smg --test gamma_tests",
                    "switch": "on",
                    "picks": True,
                },
                # ... and the listed case runs alone after them, its picks not written down.
                {
                    "args": "test --no-fail-fast -p smg --test alpha_tests -- --exact one::refused",
                    "switch": "on",
                    "picks": False,
                },
            ],
        )

    def test_the_run_deals_the_binaries_as_the_shards_do(self):
        binaries = []
        for index in (1, 2, 3):
            _, calls = self.run_shard(shard=f"gateway-{index}")
            dealt = {
                name for call in calls[1:] for name in re.findall(r"--test (\S+)", call["args"])
            }
            binaries.append(sorted(dealt))
        self.assertEqual(
            binaries, [["alpha_tests", "gamma_tests"], ["beta_tests"], ["delta_tests"]]
        )

    def assert_red(self, done: subprocess.CompletedProcess[str], problem: str) -> None:
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"policy controller on, gateway-1: PROBLEM: {problem}", done.stdout)
        self.assertIn("policy controller on, gateway-1: FAILED, 1 problem(s)", done.stdout)

    def test_a_failure_the_list_does_not_name_is_red(self):
        done, _ = self.run_shard(self.changed("gamma_tests", "plain", result="FAILED"))
        self.assert_red(
            done,
            "gamma_tests plain: fails with the policy controller on, and the list does not name it",
        )

    def test_a_listed_failure_that_passes_now_is_red_until_its_line_is_deleted(self):
        done, _ = self.run_shard(self.changed("alpha_tests", "one::refused", result="ok"))
        self.assert_red(
            done,
            "alpha_tests one::refused: passes now with the policy controller on: "
            "delete its line from the list",
        )

    def test_a_listed_failure_its_binary_no_longer_runs_is_red(self):
        done, _ = self.run_shard(self.changed("alpha_tests", "one::refused"))
        self.assert_red(
            done,
            "alpha_tests one::refused: listed as failing, and its binary does not run it "
            "(renamed, removed or ignored): delete or correct its line",
        )

    def test_a_classic_pick_in_a_case_the_list_does_not_name_is_red(self):
        done, _ = self.run_shard(self.changed("gamma_tests", "plain", result="ok", policy="manual"))
        self.assert_red(
            done,
            "gamma_tests plain: a classic policy (manual) places a request with the policy "
            "controller on, and the list does not name the case",
        )

    def test_a_listed_case_no_classic_policy_places_any_more_is_red(self):
        done, _ = self.run_shard(self.changed("alpha_tests", "one::pair", result="ok"))
        self.assert_red(
            done,
            "alpha_tests one::pair: no classic policy places its requests with the policy "
            "controller on any more: delete its line from the list",
        )

    def test_a_pick_on_a_thread_that_is_no_case_says_it_cannot_be_tied_to_a_case(self):
        scenario = self.changed(
            "gamma_tests", "plain", result="ok", policy="manual", thread="tokio-runtime-worker"
        )
        done, _ = self.run_shard(scenario)
        self.assert_red(
            done,
            "gamma_tests: a classic policy (manual) picks on the thread tokio-runtime-worker, "
            "which is no case this binary ran: the pick cannot be tied to its case",
        )

    def test_a_result_is_read_from_the_standard_output_alone(self):
        # What a case writes to the standard error is no result, whatever it looks like.
        noise = "test plain ... FAILED"
        done, _ = self.run_shard(self.changed("gamma_tests", "plain", result="ok", stderr=noise))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(noise, done.stderr)
        self.assertIn("policy controller on, gateway-1: OK", done.stdout)

    def test_a_line_that_names_no_test_binary_or_is_malformed_is_red(self):
        done, _ = self.run_shard(listed=self.LIST + "nowhere_tests\tx\tfails\twhy\n")
        self.assert_red(
            done, "line 5 of the list names nowhere_tests, which is no test binary of the gateway"
        )
        done, _ = self.run_shard(listed=self.LIST + "alpha_tests\tone::passes\tsometimes\twhy\n")
        self.assert_red(
            done,
            "line 5 of the list is not <binary> <case> <fails|classic> <why>, tab separated",
        )

    def test_a_hosted_run_folds_the_expected_failures_and_annotates_a_problem(self):
        done, _ = self.run_shard(hosted=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        folded = done.stdout.split("::group::", 1)[1].split("::endgroup::", 1)[0]
        self.assertIn("test one::refused ... FAILED", folded)
        self.assertNotIn("::error::", done.stdout)
        done, _ = self.run_shard(self.changed("gamma_tests", "plain", result="FAILED"), hosted=True)
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(
            "::error::The tests of gateway-1 with the policy controller on: gamma_tests plain: fails",
            done.stdout,
        )

    def test_only_a_gateway_integration_shard_runs_with_the_policy_controller_on(self):
        for shard in ("gateway-unit", "kv"):
            done, calls = self.run_shard(shard=shard)
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertIn("only a gateway integration shard", done.stderr)
            self.assertEqual([call["args"].split()[0] for call in calls], ["metadata"])

    def test_the_gateway_integration_shards_carry_the_step_and_the_workflow_runs_it(self):
        needs = {row["shard"]: row["needs"] for row in shard_table()}
        for shard in ("gateway-1", "gateway-2", "gateway-3"):
            self.assertEqual(needs[shard], "policy-controller")
        self.assertEqual(needs["gateway-unit"], "ffmpeg")
        workflow = WORKFLOW.read_text()
        step = workflow.split("- name: Run the gateway's tests with the policy controller on", 1)[1]
        step = step.split("\n      - name:", 1)[0]
        for line in (
            "if: contains(matrix.needs, 'policy-controller')",
            "bash scripts/ci_rust_tests.sh policy-controller ${{ matrix.shard }}",
        ):
            self.assertIn(line, step)
        # Behind the shard's own run: the binaries are the ones that run built.
        self.assertLess(
            workflow.index("- name: Run the Rust tests"),
            workflow.index("- name: Run the gateway's tests with the policy controller on"),
        )

    def test_the_list_names_cases_the_gateways_tests_hold(self):
        sources = [path.read_text() for path in (REPO / "model_gateway" / "tests").rglob("*.rs")]
        rows = []
        for line in KNOWN.read_text().splitlines():
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            self.assertEqual(len(fields), 4, line)
            binary, case, kind, why = fields
            self.assertIn(kind, ("fails", "classic"), line)
            self.assertTrue(why, line)
            self.assertTrue((REPO / "model_gateway" / "tests" / f"{binary}.rs").is_file(), line)
            # The case's function is in a test source: a renamed case leaves its line behind.
            name = case.rsplit("::", 1)[-1]
            self.assertTrue(
                any(re.search(rf"\bfn {re.escape(name)}\s*\(", text) for text in sources),
                f"no test source holds `fn {name}`: {line}",
            )
            rows.append((binary, case))
        self.assertEqual(rows, sorted(set(rows)), "the list is sorted and names a case once")


if __name__ == "__main__":
    unittest.main()
