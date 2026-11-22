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
#   ci_rust_tests.sh policy-controller <shard>  run a gateway integration shard's tests once more with the policy
#                                          controller on, held to model_gateway/tests/policy_controller_known.tsv
#   ci_rust_tests.sh lint <leg>            one lint configuration: all-features (clippy with every feature, then
#                                          the workspace's rustdoc under -D warnings), self-hosted, anthropic-only,
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

# shard | path filter | extra steps (comma-separated: golden, ffmpeg, turbojpeg, postgres, wasm-fixtures, policy-guest, policy-import,
#   policy-controller: the shard's tests once more with the policy controller on) | test step timeout (minutes)
SHARDS='
gateway-unit rust-gateway    ffmpeg                 20
gateway-1    rust-gateway    policy-controller      20
gateway-2    rust-gateway    policy-controller      20
gateway-3    rust-gateway    policy-controller      20
bindings     rust-bindings   -                      20
tokenizer    rust-tokenizer  -                      20
multimodal   rust-multimodal golden,ffmpeg,turbojpeg 20
symphony     rust-symphony   -                      20
agentic      rust-agentic    postgres,wasm-fixtures 20
protocols    rust-protocols  -                      20
engine       rust-engine     -                      20
kv           rust-kv         policy-guest,policy-import 20
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
        protocols)  echo openai-protocol smg-worker smg-external-router smg-service-discovery smg-auth smg-rl smg-client openapi-gen ;;
        engine)     echo smg-grpc-client engine-zmq-client engine-zmq-adapter engine-servicer mock-worker gateway-perf ;;
        kv)         echo kv-index smg-mesh smg-mm-rdma smg-policy-controller policy-controller-example smg-radix-tree ;;
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
            # The lib's unit tests. The command line's cases run with the library, and
            # neither executable builds a test harness (their manifest entries say
            # test = false).
            # One invocation for both packages: on its own, the harness's `-p` would unify the gateway
            # down to the harness's feature set and build the gateway lib a second time.
            echo "-p smg -p smg-test-harness --lib"
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
            echo "-p openai-protocol -p smg-worker -p smg-external-router -p smg-service-discovery -p smg-auth -p smg-rl -p smg-client -p openapi-gen --features openai-protocol/axum,smg-worker/runtime,smg-worker/workflow,smg-external-router/openai,smg-external-router/anthropic,smg-external-router/gemini,smg-service-discovery/consul,smg-service-discovery/file,smg-service-discovery/kubernetes,smg-service-discovery/slurm"
            # The discovery crate with no provider built: its shared cases, and the refusal of
            # each provider's settings.
            echo "-p smg-service-discovery" ;;
        engine)
            # `engine-port` is on for the cases of the client crate's engine port: no package of the shard
            # asks for it, so nothing else would turn it on.
            echo "-p smg-grpc-client -p engine-zmq-client -p engine-zmq-adapter -p engine-servicer -p mock-worker -p gateway-perf --features engine-zmq-client/mock-engine,smg-grpc-client/engine-port" ;;
        kv)
            local selectors
            # The example controller rides the policy crate's invocation: it asks for the crate's
            # `controller` feature, which this line turns on already, so nothing is built twice.
            # `worker-host` is on for the cases of the crate's worker objects and for the test targets that
            # run the stock binary, which require it.
            echo "-p kv-index -p smg-mesh -p smg-mm-rdma -p smg-policy-controller -p policy-controller-example --features smg-policy-controller/client,smg-policy-controller/kubernetes,smg-policy-controller/worker-host"
            # The policy crate's library once more, under `discovery`: no package of the shard asks for
            # that feature, so nothing else would run the cases behind it. The line above stays as it
            # is: it runs the case of a build without the feature.
            echo "-p smg-policy-controller --lib --features smg-policy-controller/discovery"
            # The policy crate's case for a host that builds the `operator` feature alone. The shard's
            # first line turns the controller on, so that case would never run as such a host builds it.
            echo "-p smg-policy-controller --no-default-features --features operator --test operator_door"
            # The policy crate's library as a host in process builds it: `worker-host` without the
            # controller. The shard's first line turns the controller on, so that build would never be made.
            echo "-p smg-policy-controller --lib --no-default-features --features worker-host"
            # The policy crate's library under the features the gateway asks for on every target: the engine a
            # gateway keeps in its own process (`in-process`) beside `operator` and `worker-host`, without the
            # controller (on Linux and macOS the gateway also asks for `client`, which turns the controller on).
            # The gateway alone asks for `in-process`, and its shards run the gateway's cases, so no other line
            # runs the cases behind that feature.
            echo "-p smg-policy-controller --lib --no-default-features --features in-process,operator,worker-host"
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
    local shard=$1 mode=$2 line cargo_args test_args commands status=0
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
        # A shard names several binaries per invocation and may run several
        # invocations: without --no-fail-fast cargo stops at the first binary that
        # fails, and under the script's errexit a failed invocation would end this
        # function before the next one, so a red early in the shard would hide the
        # results after it. Every binary of every invocation runs; the status of a
        # failed one is kept and returned at the end, so the shard still fails
        # when any binary did.
        elif [ -n "$test_args" ]; then
            # shellcheck disable=SC2086
            (set -x; cargo test --no-fail-fast $cargo_args -- $test_args) || status=$?
        else
            # shellcheck disable=SC2086
            (set -x; cargo test --no-fail-fast $cargo_args) || status=$?
        fi
    done <<< "$commands"
    return "$status"
}

# The gateway's integration cases known to differ with the policy controller on: one line a case (test
# binary, case, `fails` or `classic`, why), beside the tests it names.
POLICY_CONTROLLER_KNOWN=${POLICY_CONTROLLER_KNOWN:-model_gateway/tests/policy_controller_known.tsv}

# The cases of a test binary that the list names under a kind, one per line.
known_cases() {
    local kind=$1 target=$2
    awk -F '\t' -v k="$kind" -v t="$target" '!/^#/ && $1 == t && $3 == k { print $2 }' "$POLICY_CONTROLLER_KNOWN"
}

# A gateway integration shard's tests once more with the policy controller on in every context the test
# harness builds (SMG_TEST_POLICY_CONTROLLER=on: the harness reads it when a test runs, so these are the
# binaries `build` made). The run is held to the list of the cases known to differ, in three parts, and
# every part runs:
#   1. each binary without its cases listed as `fails`: a failure here is a case the list does not name;
#   2. each binary's cases listed as `fails`, alone: every one still fails, or its line is deleted;
#   3. the picks a classic policy made in part 1 (SMG_TEST_CLASSIC_PICKS: the gateway's test build writes
#      one line a pick where its policy registry holds the policy engine) against the cases listed as
#      `classic`: a pick in a case the list does not name is red, and so is a listed case with no pick.
# The change that cures a case deletes its line: the list only shrinks.
policy_controller_run() {
    local shard=$1 index every targets target names skips work status=0
    case "$shard" in
        gateway-[0-9]*) index=${shard#gateway-} ;;
        *) echo "ci_rust_tests.sh: only a gateway integration shard (gateway-1 to gateway-$GATEWAY_SHARDS) runs with the policy controller on, not '$shard'" >&2; exit 2 ;;
    esac
    [ "$index" -ge 1 ] && [ "$index" -le "$GATEWAY_SHARDS" ] || { echo "no gateway shard $index" >&2; exit 2; }
    [ -f "$POLICY_CONTROLLER_KNOWN" ] || { echo "ci_rust_tests.sh: no list at $POLICY_CONTROLLER_KNOWN" >&2; exit 2; }
    # The shard's binaries, as test_selectors deals them. Assigned before use: a failed metadata read
    # fails the run instead of emptying it.
    every=$(test_targets smg)
    targets=$(printf '%s\n' "$every" | awk -v i="$index" -v n="$GATEWAY_SHARDS" 'NR % n == i % n')
    work=$(mktemp -d)
    : > "$work/picks"
    export SMG_TEST_POLICY_CONTROLLER=on

    # A binary an invocation, in parts 1 and 2: a case is named by its binary and its own name, and the
    # binary's standard output alone is kept for part 3. What cargo and a case write to the standard
    # error stays apart from it. A binary that logs to its raw standard output can still write a line
    # into the line of a result (the name, the result and the line's end are three writes). So part 3
    # reads a failed case from the names the binary lists at its end as well, and holds a pick to the
    # list before it ties it to a result: a result line that is not whole changes no verdict.
    echo "== The policy controller on, part 1 of 3: the tests of $shard, without the cases the list names as failing =="
    for target in $targets; do
        names=$(known_cases fails "$target")
        skips=
        # --skip with --exact leaves out exactly the listed names.
        [ -z "$names" ] || skips=$(printf '%s\n' "$names" | awk 'BEGIN { printf " -- --exact" } { printf " --skip %s", $1 }')
        # shellcheck disable=SC2086
        (set -x; SMG_TEST_CLASSIC_PICKS="$work/picks" cargo test --no-fail-fast -p smg --test "$target"$skips) | tee "$work/part1.$target" || status=$?
    done

    # The failures of this part are the expected ones: a hosted log folds them into one group.
    [ -z "${GITHUB_ACTIONS:-}" ] || echo "::group::The policy controller on, part 2 of 3: expected failures"
    echo "== The policy controller on, part 2 of 3: the cases the list names as failing, alone (each is expected to FAIL) =="
    for target in $targets; do
        names=$(known_cases fails "$target")
        [ -n "$names" ] || continue
        # shellcheck disable=SC2086
        (set -x; cargo test --no-fail-fast -p smg --test "$target" -- --exact $names) | tee "$work/fails.$target" || true
    done
    [ -z "${GITHUB_ACTIONS:-}" ] || echo "::endgroup::"

    echo "== The policy controller on, part 3 of 3: the run of $shard against $POLICY_CONTROLLER_KNOWN =="
    # shellcheck disable=SC2086
    python3 -c '
import os
import re
import sys
from pathlib import Path

known, work, shard, part1_status, every, *targets = sys.argv[1:]
work = Path(work)
every = set(every.split())
problems = []

# The list: (binary, case) -> kind.
listed = {}
for number, line in enumerate(Path(known).read_text().splitlines(), 1):
    if not line.strip() or line.startswith("#"):
        continue
    fields = line.split("\t")
    if len(fields) != 4 or not all(fields) or fields[2] not in ("fails", "classic"):
        problems.append(f"line {number} of the list is not <binary> <case> <fails|classic> <why>, tab separated")
        continue
    binary, case, kind, _why = fields
    if binary not in every:
        problems.append(f"line {number} of the list names {binary}, which is no test binary of the gateway")
    elif (binary, case) in listed:
        problems.append(f"line {number} of the list names {binary} {case} a second time")
    listed[(binary, case)] = kind
mine = {key: kind for key, kind in listed.items() if key[0] in targets}


def results(output):
    """case -> ok, FAILED or ignored, from the standard output one test binary wrote to the file.

    A failed case is read from the names the binary lists at its end too: each of those is one
    write, where a result is three, with room for a line of the raw standard output between them."""
    found = {}
    if not output.exists():
        return found
    lines = output.read_text().splitlines()
    for line in lines:
        test = re.match(r"test (\S+)(?: - should panic)? \.\.\. (ok|FAILED|ignored)\b", line)
        if test:
            found[test.group(1)] = test.group(2)
    lists = [number for number, line in enumerate(lines) if line == "failures:"]
    if lists:
        for line in lines[lists[-1] + 1 :]:
            if line.startswith("test result:"):
                break
            name = re.fullmatch(r"    (\S+)", line)
            if name:
                found[name.group(1)] = "FAILED"
    return found


# Part 1: nothing fails but the cases left out of it.
part1 = {binary: results(work / f"part1.{binary}") for binary in targets}
failed = sorted((binary, case) for binary, cases in part1.items() for case, result in cases.items() if result == "FAILED")
for binary, case in failed:
    problems.append(f"{binary} {case}: fails with the policy controller on, and the list does not name it")
if part1_status != "0" and not failed:
    problems.append("a test command of part 1 failed and its output names no failing case (a build error, a crash, a timeout)")

# Part 2: every case listed as failing still fails.
for (binary, case), kind in sorted(mine.items()):
    if kind != "fails":
        continue
    result = results(work / f"fails.{binary}").get(case)
    if result == "ok":
        problems.append(f"{binary} {case}: passes now with the policy controller on: delete its line from the list")
    elif result != "FAILED":
        problems.append(f"{binary} {case}: listed as failing, and its run shows no failure of it (renamed, removed or ignored): delete or correct its line")

# Part 3: a classic policy picks in the cases listed as classic, and in no other.
picked = {}
for line in (work / "picks").read_text().splitlines():
    fields = line.split("\t")
    if len(fields) != 3:
        problems.append(f"a line of the picks file is not <executable> <thread> <policy>, tab separated: {line}")
        continue
    binary = re.sub(r"-[0-9a-f]{16}$", "", fields[0])
    picked.setdefault((binary, fields[1]), set()).add(fields[2])
for (binary, case), policies in sorted(picked.items()):
    names = ", ".join(sorted(policies))
    # The line names its thread, and a case is known by the thread it runs on. The list comes first:
    # a listed case needs no result line to be known.
    if mine.get((binary, case)) == "classic":
        continue
    if case in part1.get(binary, {}):
        problems.append(f"{binary} {case}: a classic policy ({names}) places a request with the policy controller on, and the list does not name the case")
    else:
        problems.append(f"{binary}: a classic policy ({names}) picks on the thread {case}, and the run shows no result of a case of that name (a thread of no case, or a result line that is not whole): the pick cannot be tied to a listed case")
for (binary, case), kind in sorted(mine.items()):
    if kind == "classic" and (binary, case) not in picked:
        problems.append(f"{binary} {case}: no classic policy places its requests with the policy controller on any more: delete its line from the list")

kinds = list(mine.values())
failing, classic = kinds.count("fails"), kinds.count("classic")
ran = sum(len(cases) for cases in part1.values())
print(f"policy controller on, {shard}: {len(targets)} test binaries, {ran} results read in part 1; the list names {failing} of their cases as failing and {classic} as placed by a classic policy")
for problem in problems:
    print(f"policy controller on, {shard}: PROBLEM: {problem}")
    if os.environ.get("GITHUB_ACTIONS"):
        # An annotation of the hosted run: the problem shows on the pull request.
        print(f"::error::The tests of {shard} with the policy controller on: {problem}")
print(f"policy controller on, {shard}: " + ("OK" if not problems else f"FAILED, {len(problems)} problem(s)"))
sys.exit(1 if problems else 0)
' "$POLICY_CONTROLLER_KNOWN" "$work" "$shard" "$status" "$every" $targets || status=1
    rm -rf "$work"
    return "$status"
}

# Every set of the discovery crate's four provider features but the empty one, a comma list
# a line, and each set that does not hold the file provider once more with the directory
# watch (`file` names `fs-watch`): the 23 distinct builds a host can name.
discovery_feature_sets() {
    local mask features
    for mask in $(seq 0 15); do
        features=""
        [ $((mask & 1)) -eq 0 ] || features="consul"
        [ $((mask & 2)) -eq 0 ] || features="${features:+$features,}file"
        [ $((mask & 4)) -eq 0 ] || features="${features:+$features,}kubernetes"
        [ $((mask & 8)) -eq 0 ] || features="${features:+$features,}slurm"
        [ -z "$features" ] || echo "$features"
        [ $((mask & 2)) -ne 0 ] || echo "${features:+$features,}fs-watch"
    done
}

lint() {
    case "$1" in
        all-features)
            (set -x; cargo clippy --all-targets --all-features -- -D warnings)
            # The workspace's public documentation, every feature on, builds without a warning,
            # so a moved or renamed item cannot leave a dangling or private link behind.
            (set -x; RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --all-features) ;;
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
            # The discovery crate builds each provider under a feature of its own, and a host
            # names the ones it runs: with none, under every set of them, and with the test
            # helpers alone, nothing is left unused and no lint reads one set apart.
            (set -x; cargo clippy -p smg-service-discovery --all-targets -- -D warnings)
            for features in $(discovery_feature_sets) test-util; do
                (set -x; cargo clippy -p smg-service-discovery --features "$features" --all-targets -- -D warnings)
            done
            # The policy crate's `operator` feature is what a host calls in its own process. It
            # builds with no controller behind it, and nothing of it is left unused there.
            (set -x; cargo clippy -p smg-policy-controller --no-default-features --features operator --all-targets -- -D warnings)
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
    policy-controller) shard_row "$2" >/dev/null; load_metadata; policy_controller_run "$2" ;;
    lint)     lint "${2:-}" ;;
    check)    load_metadata; check ;;
    *)
        sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
        exit 2 ;;
esac
