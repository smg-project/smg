#!/usr/bin/env bash
# The Rust test shards of the PR Test (SMG) workflow: one place for what each
# shard builds and runs, so the workflow's rust-tests job, the Rust Cache
# Warmer and a developer's shell run the same commands.
#
#   ci_rust_tests.sh list                  the shard table: shard, path filter, extra setup, test timeout
#   ci_rust_tests.sh matrix <changes> <all> the job matrix (JSON) for the filters in <changes> (a JSON list of
#                                          filter names), or for every shard when <all> is "true"
#   ci_rust_tests.sh packages <shard>      the workspace packages a shard tests
#   ci_rust_tests.sh commands <shard>      the shard's cargo test invocations, one per line
#   ci_rust_tests.sh build <shard>         compile the shard's tests (cargo test --no-run)
#   ci_rust_tests.sh run <shard>           run the shard's tests
#   ci_rust_tests.sh lint <leg>            one lint configuration: all-features, self-hosted, anthropic-only,
#                                          default-build (the default feature set compiles without OpenCV)
#   ci_rust_tests.sh check                 every workspace package, every gateway test binary and every
#                                          radix-tree test binary belongs to exactly one shard
#
# A shard groups crates with a common dependency closure, and the workflow
# runs a shard only when a pull request touches a path that can change what
# it tests (the rust-* filters of detect-changes in pr-test-rust.yml;
# scripts/tests/test_ci_rust_tests.py checks those filters against the
# manifests). The gateway crate's integration test binaries are dealt
# round-robin, by name, over GATEWAY_SHARDS shards; its unit tests, binaries
# and doc tests are a shard of their own. The radix tree's differential fuzz
# is the heaviest test in the workspace by far and runs alone, under its own
# timeout.
#
# The --features of a shard reproduce the feature set the crate's tests had
# under the whole-workspace `cargo test` (where the gateway's dependencies
# unify them): the protocol crate's axum surface, the provider routers, the
# storage hooks of the WASM crate, the mock engine of the ZMQ client.
set -euo pipefail
# Command substitutions keep -e, so a failure inside $(...) (a cargo metadata
# call) fails the caller's assignment instead of shrinking a shard silently.
shopt -s inherit_errexit

GATEWAY_SHARDS=3

# shard | path filter | extra setup (comma-separated: golden, ffmpeg, postgres, wasm-fixtures) | test step timeout (minutes)
SHARDS='
gateway-unit rust-gateway    -                      20
gateway-1    rust-gateway    -                      20
gateway-2    rust-gateway    -                      20
gateway-3    rust-gateway    -                      20
bindings     rust-bindings   -                      20
tokenizer    rust-tokenizer  -                      20
multimodal   rust-multimodal golden,ffmpeg          20
symphony     rust-symphony   -                      20
agentic      rust-agentic    postgres,wasm-fixtures 20
protocols    rust-protocols  -                      20
engine       rust-engine     -                      20
kv           rust-kv         -                      20
radix-fuzz   rust-radix      -                      15
'

shard_table() {
    printf '%s\n' "$SHARDS" | sed '/^[[:space:]]*$/d'
}

shard_row() {
    local shard=$1
    shard_table | awk -v s="$shard" '$1 == s' | grep . || {
        echo "ci_rust_tests.sh: unknown shard '$shard' (see: $0 list)" >&2
        exit 2
    }
}

# The packages a shard tests (static: no cargo needed).
packages_of() {
    case "$1" in
        gateway-unit) echo smg smg-test-harness ;;
        gateway-[0-9]*) echo smg ;;
        bindings)   echo smg-python smg-golang ;;
        tokenizer)  echo llm-tokenizer ;;
        multimodal) echo llm-multimodal ;;
        symphony)   echo smg-symphony tool-parser reasoning-parser smg-response-template ;;
        agentic)    echo data-connector smg-mcp smg-wasm wfaas ;;
        protocols)  echo openai-protocol smg-external-router smg-auth smg-rl smg-client openapi-gen ;;
        engine)     echo smg-grpc-client engine-zmq-client engine-zmq-adapter engine-servicer mock-worker ;;
        kv)         echo kv-index smg-mesh smg-mm-rdma smg-radix-tree ;;
        radix-fuzz) echo smg-radix-tree ;;
        *) shard_row "$1" >/dev/null ;;
    esac
}

# The workspace's package metadata, read once per invocation: the dispatcher
# at the end fills it (load_metadata) before a mode that expands it. The
# callers read it on the left side of a pipeline, a subshell, so a fill made
# there would not outlive the call; the fallback below only covers a caller
# the dispatcher did not foresee.
METADATA=
load_metadata() {
    METADATA=$(cargo metadata --no-deps --format-version 1 --locked)
}
metadata() {
    [ -n "$METADATA" ] || load_metadata
    printf '%s' "$METADATA"
}

# The shards whose commands expand the metadata (test binaries dealt by name).
metadata_shard() {
    case "$1" in
        gateway-[0-9]*|kv) return 0 ;;
        *) return 1 ;;
    esac
}

# The integration test targets of a package, sorted, one per line.
test_targets() {
    metadata | python3 -c '
import json, sys
pkg = sys.argv[1]
for p in json.load(sys.stdin)["packages"]:
    if p["name"] == pkg:
        for name in sorted(t["name"] for t in p["targets"] if "test" in t["kind"]):
            print(name)
' "$1"
}

# `--test a --test b ...` for every n-th target of a package: shard i of n.
# The target list is assigned before it is used, here and at the call sites:
# a command substitution inside another command's arguments loses its exit
# status, and a failed metadata read must fail the shard, not shrink it.
test_selectors() {
    local pkg=$1 index=$2 count=$3 targets
    targets=$(test_targets "$pkg")
    printf '%s\n' "$targets" | awk -v i="$index" -v n="$count" 'NR % n == i % n { printf " --test %s", $1 }'
}

# `--test ...` for every integration test target of a package but the ones named.
test_selectors_except() {
    local pkg=$1; shift
    local target skip targets
    targets=$(test_targets "$pkg")
    for target in $targets; do
        skip=
        for excluded in "$@"; do [ "$target" = "$excluded" ] && skip=1; done
        [ -n "$skip" ] || printf ' --test %s' "$target"
    done
}

# The shard's cargo test invocations: "<cargo test arguments> [:: <test binary arguments>]".
commands_of() {
    local shard=$1
    case "$shard" in
        gateway-unit)
            # The lib's unit tests and the smg binary's. The alias binary amg is the same
            # source (its manifest entry says test = false); an explicit --bins would build
            # and run its harness anyway, so the binary is named.
            # One invocation for both packages: on its own, the harness's `-p` would unify the gateway
            # down to the harness's feature set and build the gateway lib a second time.
            # Exercise opt-in file storage; integration shards retain the default feature set.
            echo "-p smg -p smg-test-harness --features smg/diagnostic-capture --lib --bin smg"
            echo "-p smg --doc" ;;
        gateway-[0-9]*)
            local index=${shard#gateway-} selectors
            [ "$index" -ge 1 ] && [ "$index" -le "$GATEWAY_SHARDS" ] || { echo "no gateway shard $index" >&2; exit 2; }
            selectors=$(test_selectors smg "$index" "$GATEWAY_SHARDS")
            echo "-p smg$selectors" ;;
        bindings)
            echo "-p smg-python -p smg-golang" ;;
        tokenizer)
            echo "-p llm-tokenizer"
            echo "-p llm-tokenizer --features tokenizers-v1" ;;
        multimodal)
            echo "-p llm-multimodal" ;;
        symphony)
            # Nothing else in the shard depends on the template crate, so its feature set here is exactly
            # its symphony feature: a second invocation of it alone would build and run the same units again.
            echo "-p smg-symphony -p tool-parser -p reasoning-parser -p smg-response-template --features smg-response-template/symphony" ;;
        agentic)
            echo "-p data-connector -p smg-mcp -p smg-wasm -p wfaas --features smg-wasm/storage-hooks"
            # Opt-in (#[ignore]) cases against the database the workflow creates for the job.
            echo "-p data-connector --test postgres_integration --test conformance :: --ignored --test-threads=1" ;;
        protocols)
            echo "-p openai-protocol -p smg-external-router -p smg-auth -p smg-rl -p smg-client -p openapi-gen --features openai-protocol/axum,smg-external-router/openai,smg-external-router/anthropic,smg-external-router/gemini" ;;
        engine)
            echo "-p smg-grpc-client -p engine-zmq-client -p engine-zmq-adapter -p engine-servicer -p mock-worker --features engine-zmq-client/mock-engine" ;;
        kv)
            local selectors
            echo "-p kv-index -p smg-mesh -p smg-mm-rdma"
            # The radix tree without its fuzz binary (the radix-fuzz shard); --doc cannot share an invocation with --test.
            selectors=$(test_selectors_except smg-radix-tree fuzz_differential)
            echo "-p smg-radix-tree --lib$selectors"
            echo "-p smg-radix-tree --doc" ;;
        radix-fuzz)
            echo "-p smg-radix-tree --test fuzz_differential" ;;
        *) shard_row "$shard" >/dev/null ;;
    esac
}

run_commands() {
    local shard=$1 mode=$2 line cargo_args test_args commands
    # Assigned first: a process substitution would hide a failure of commands_of.
    commands=$(commands_of "$shard")
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        cargo_args=${line%% :: *}
        test_args=
        [ "$line" != "$cargo_args" ] && test_args=${line#* :: }
        if [ "$mode" = build ]; then
            # Doc tests compile when they run; cargo refuses --doc with --no-run.
            case " $cargo_args " in *" --doc "*) continue ;; esac
            # shellcheck disable=SC2086
            (set -x; cargo test $cargo_args --no-run)
        elif [ -n "$test_args" ]; then
            # shellcheck disable=SC2086
            (set -x; cargo test $cargo_args -- $test_args)
        else
            # shellcheck disable=SC2086
            (set -x; cargo test $cargo_args)
        fi
    done <<< "$commands"
}

lint() {
    case "$1" in
        all-features)
            (set -x; cargo clippy --all-targets --all-features -- -D warnings) ;;
        # Meaningful only because the gateway crate's self dev-dependency
        # disables default features; see model_gateway/Cargo.toml.
        self-hosted)
            (set -x; cargo clippy -p smg -p smg-external-router --no-default-features --features smg/grpc-client,smg/jemalloc-stats --all-targets -- -D warnings) ;;
        anthropic-only)
            (set -x; cargo clippy -p smg -p smg-external-router --no-default-features --features smg/grpc-client,smg/jemalloc-stats,smg/provider-anthropic --all-targets -- -D warnings) ;;
        # The default feature set needs no OpenCV: run where the OpenCV libraries are not installed.
        default-build)
            (set -x; cargo check -p llm-multimodal)
            # Standalone consumers that resolve model files themselves build without `hf-hub`.
            (set -x; cargo check -p llm-multimodal --no-default-features --all-targets)
            (set -x; cargo check --manifest-path bindings/python/Cargo.toml)
            (set -x; cargo check --manifest-path bindings/golang/Cargo.toml) ;;
        *) echo "ci_rust_tests.sh: unknown lint leg '$1' (all-features, self-hosted, anthropic-only, default-build)" >&2; exit 2 ;;
    esac
}

matrix() {
    local changes=$1 all=$2
    shard_table | python3 -c '
import json, sys
changes = set(json.loads(sys.argv[1] or "[]"))
every = sys.argv[2] == "true"
rows = []
for line in sys.stdin:
    shard, filt, needs, timeout = line.split()
    if every or filt in changes:
        rows.append({"shard": shard, "filter": filt, "needs": "" if needs == "-" else needs, "test_timeout": int(timeout)})
print(json.dumps(rows, separators=(",", ":")))
' "$changes" "$all"
}

check() {
    local shard status=0 members gateway radix
    {
        # Assigned before use, so a failed read fails the check instead of emptying a list.
        members=$(metadata | python3 -c 'import json,sys; print(" ".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))')
        gateway=$(test_targets smg | tr '\n' ' ')
        radix=$(test_targets smg-radix-tree | tr '\n' ' ')
        echo "members $members"
        echo "gateway-targets $gateway"
        echo "radix-targets $radix"
        for shard in $(shard_table | awk '{print $1}'); do
            echo "packages $shard $(packages_of "$shard")"
            commands_of "$shard" | sed "s/^/command $shard /"
        done
    } | python3 -c '
import sys
from collections import Counter
members = set(); gateway = []; radix = []; packages = {}; commands = {}
for line in sys.stdin:
    kind, rest = line.rstrip("\n").split(" ", 1)
    if kind == "members": members = set(rest.split())
    elif kind == "gateway-targets": gateway = rest.split()
    elif kind == "radix-targets": radix = rest.split()
    elif kind == "packages":
        shard, _, pkgs = rest.partition(" "); packages[shard] = pkgs.split()
    elif kind == "command":
        shard, _, cmd = rest.partition(" "); commands.setdefault(shard, []).append(cmd)
problems = []
# Every workspace package is tested by exactly one shard; the gateway (unit shard + round-robin
# integration shards) and the radix tree (kv + radix-fuzz) are the two deliberate splits.
split = {"smg": {s for s in packages if s.startswith("gateway-")}, "smg-radix-tree": {"kv", "radix-fuzz"}}
owners = {}
for shard, pkgs in packages.items():
    for p in pkgs: owners.setdefault(p, set()).add(shard)
for p in sorted(members):
    got = owners.get(p, set())
    want = split.get(p)
    if want is not None and got != want: problems.append(f"{p}: tested by {sorted(got)}, expected {sorted(want)}")
    elif want is None and len(got) != 1: problems.append(f"{p}: tested by {sorted(got)}, expected exactly one shard")
for p in owners:
    if p not in members: problems.append(f"{p}: in a shard but not a workspace package")
# Every -p in the commands of a shard is in its package list and the other way round.
import re
for shard, cmds in commands.items():
    used = {m for c in cmds for m in re.findall(r"-p (\S+)", c)}
    if used != set(packages[shard]): problems.append(f"{shard}: commands test {sorted(used)} but the shard lists {sorted(packages[shard])}")
# The gateway integration shards deal every test binary once; the radix shards likewise.
dealt = Counter(t for s, cmds in commands.items() if re.fullmatch(r"gateway-\d+", s) for c in cmds for t in re.findall(r"--test (\S+)", c))
if dealt != Counter(gateway): problems.append(f"gateway test binaries dealt {sorted(dealt.elements())} != {sorted(gateway)}")
if any(n != 1 for n in dealt.values()): problems.append("a gateway test binary is dealt twice")
rdealt = Counter(t for s in ("kv", "radix-fuzz") for c in commands.get(s, []) for t in re.findall(r"--test (\S+)", c) if "smg-radix-tree" in c)
if rdealt != Counter(radix): problems.append(f"radix test binaries dealt {sorted(rdealt.elements())} != {sorted(radix)}")
for p in problems: print("ci_rust_tests.sh check:", p)
gateway_shards = len(split["smg"]) - 1
print(f"ci_rust_tests.sh check: {len(members)} packages, {len(gateway)} gateway test binaries over {gateway_shards} shards, {len(radix)} radix-tree test binaries: " + ("OK" if not problems else "FAILED"))
sys.exit(1 if problems else 0)
' || status=$?
    return $status
}

mode=${1:-}
case "$mode" in
    list)     shard_table ;;
    matrix)   matrix "${2:-[]}" "${3:-false}" ;;
    packages) shard_row "$2" >/dev/null; packages_of "$2" ;;
    commands) shard_row "$2" >/dev/null; ! metadata_shard "$2" || load_metadata; commands_of "$2" ;;
    build)    shard_row "$2" >/dev/null; ! metadata_shard "$2" || load_metadata; run_commands "$2" build ;;
    run)      shard_row "$2" >/dev/null; ! metadata_shard "$2" || load_metadata; run_commands "$2" run ;;
    lint)     lint "${2:-}" ;;
    check)    load_metadata; check ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 2 ;;
esac
