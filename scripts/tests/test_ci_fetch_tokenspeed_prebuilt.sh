#!/bin/bash
# Tests for scripts/ci_fetch_tokenspeed_prebuilt.sh: the per-node cache of the
# extracted TokenSpeed payload on the shared NVMe mount.
#
# Needs GNU coreutils + flock, i.e. Linux. From a Mac run it in a container:
#   docker run --rm -v "$PWD:/repo" -w /repo ubuntu:24.04 \
#       bash scripts/tests/test_ci_fetch_tokenspeed_prebuilt.sh
#
# `docker` is replaced by a fake on PATH that records its calls and serves a
# fake image tree, so no daemon or registry is involved.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci_fetch_tokenspeed_prebuilt.sh"
IMAGE="ghcr.io/smg-project/smg:ci-tokenspeed-aaaa-bbbb"
TAG="ci-tokenspeed-aaaa-bbbb"

assert_eq() {
    local expected="$1" actual="$2" what="${3:-}"
    if [ "$expected" != "$actual" ]; then
        echo "FAIL${what:+ ($what)}: expected '$expected', got '$actual'"
        exit 1
    fi
}

assert_file() { [ -e "$1" ] || { echo "FAIL: expected '$1' to exist"; exit 1; }; }
# -L too: a dangling symlink is invisible to -e but is exactly the leftover we care about.
assert_no_file() { { [ ! -e "$1" ] && [ ! -L "$1" ]; } || { echo "FAIL: expected '$1' to be absent"; exit 1; }; }
assert_link_into() {
    local link="$1" dir="$2" target
    [ -L "$link" ] || { echo "FAIL: expected '$link' to be a symlink"; exit 1; }
    target="$(readlink "$link")"
    [[ "$target" == "$dir"/* ]] || { echo "FAIL: '$link' -> '$target', expected a target under '$dir'"; exit 1; }
}
assert_contains() {
    local haystack="$1" needle="$2"
    [[ "$haystack" == *"$needle"* ]] || { echo "FAIL: expected output to contain '$needle'"; echo "$haystack"; exit 1; }
}

# One sandbox per test: fake image tree, fake docker on PATH, empty cache root,
# and an install root standing in for /opt. The previous sandbox is removed on
# the next setup, the last one on exit.
cleanup_sandbox() { [ -z "${T:-}" ] || rm -rf "$T"; }
trap cleanup_sandbox EXIT
setup() {
    cleanup_sandbox
    T="$(mktemp -d)"
    export FAKE_IMAGE_ROOT="$T/image" FAKE_DOCKER_LOG="$T/docker.log"
    mkdir -p "$FAKE_IMAGE_ROOT/opt/smg-ci/.venv/bin" "$FAKE_IMAGE_ROOT/opt/tokenspeed-src"
    echo "#!/bin/sh" > "$FAKE_IMAGE_ROOT/opt/smg-ci/.venv/bin/python"
    chmod +x "$FAKE_IMAGE_ROOT/opt/smg-ci/.venv/bin/python"
    echo "deadbeef" > "$FAKE_IMAGE_ROOT/opt/smg-ci/tokenspeed.ref"
    echo "src" > "$FAKE_IMAGE_ROOT/opt/tokenspeed-src/README"
    : > "$FAKE_DOCKER_LOG"
    mkdir -p "$T/bin"
    cat > "$T/bin/docker" <<'FAKE'
#!/bin/bash
echo "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
    pull)   [ "${FAKE_PULL_FAIL:-0}" = 1 ] && exit 1; sleep "${FAKE_PULL_SLEEP:-0}"; exit 0 ;;
    login)  exit 0 ;;
    create) echo "cid-fake"; exit 0 ;;
    cp)     [ "${FAKE_CP_FAIL:-0}" = 1 ] && exit 1
            cp -a "${FAKE_IMAGE_ROOT}${2#*:}" "$3" ;;
    rm)     exit 0 ;;
    *)      echo "fake docker: unexpected '$*'" >&2; exit 2 ;;
esac
FAKE
    chmod +x "$T/bin/docker"
    export PATH="$T/bin:$PATH"
    CACHE="$T/cache"
}

# run_fetch <install-root> <run-id> [env assignments...]
run_fetch() {
    local install="$1" run_id="$2"; shift 2
    mkdir -p "$install"
    : > "$install/github.env"
    # Defaults first: a caller's own assignments (later on the env command line) win.
    env TOKENSPEED_PREBUILT_IMAGE="$IMAGE" \
        TOKENSPEED_PREBUILT_CACHE_ROOT="$CACHE" \
        TOKENSPEED_PREBUILT_INSTALL_ROOT="$install" \
        GITHUB_ENV="$install/github.env" GITHUB_RUN_ID="$run_id" GITHUB_JOB="e2e" GITHUB_RUN_ATTEMPT="1" \
        RUNNER_TEMP="$T/runner-temp" \
        "$@" bash "$SCRIPT"
}

pull_count() { grep -c '^pull ' "$FAKE_DOCKER_LOG" || true; }

test_cold_pull_populates_cache_and_links_install_root() {
    setup
    local out; out="$(run_fetch "$T/opt" 100)"
    assert_file "$CACHE/$TAG/.complete"
    assert_file "$CACHE/$TAG/smg-ci/tokenspeed.ref"
    assert_link_into "$T/opt/smg-ci" "$CACHE/jobs"
    assert_link_into "$T/opt/tokenspeed-src" "$CACHE/jobs"
    assert_eq "deadbeef" "$(cat "$T/opt/smg-ci/tokenspeed.ref")" "payload readable through the link"
    assert_eq 1 "$(pull_count)" "one docker pull"
    assert_contains "$(cat "$T/opt/github.env")" "SMG_BAKED_VENV=$T/opt/smg-ci/.venv"
    assert_contains "$(cat "$T/opt/github.env")" "TOKENSPEED_PREBUILT_JOB_DIR=$CACHE/jobs/"
    assert_contains "$out" "Prebuilt payload installed"
}

test_warm_cache_skips_docker_entirely() {
    setup
    run_fetch "$T/opt-a" 100 > /dev/null
    : > "$FAKE_DOCKER_LOG"
    local out; out="$(run_fetch "$T/opt-b" 200)"
    assert_eq "" "$(cat "$FAKE_DOCKER_LOG")" "no docker calls on a warm node"
    assert_link_into "$T/opt-b/smg-ci" "$CACHE/jobs"
    assert_eq "deadbeef" "$(cat "$T/opt-b/smg-ci/tokenspeed.ref")"
    assert_contains "$out" "cache hit"
}

test_job_copies_are_private() {
    setup
    run_fetch "$T/opt-a" 100 > /dev/null
    run_fetch "$T/opt-b" 200 > /dev/null
    [ "$(readlink "$T/opt-a/smg-ci")" != "$(readlink "$T/opt-b/smg-ci")" ] || { echo "FAIL: jobs share a payload dir"; exit 1; }
    echo "pr-glue" > "$T/opt-a/smg-ci/.venv/glue.pth"
    assert_no_file "$CACHE/$TAG/smg-ci/.venv/glue.pth"
    assert_no_file "$T/opt-b/smg-ci/.venv/glue.pth"
}

# Runner pods have their own PID namespaces, so two lanes of one run on the
# same node can share run id, job name, attempt AND shell pid. The job dir
# must still be unique. `exec` keeps the subshell's pid ($BASHPID; `$$` would
# still be the parent's), so the pre-created dir is exactly the one a
# pid-based name would pick.
test_job_dir_is_unique_even_when_pid_and_job_identity_collide() {
    setup
    run_fetch "$T/opt-a" 100 > /dev/null
    mkdir -p "$T/opt-b"; : > "$T/opt-b/github.env"
    (
        collide="$CACHE/jobs/100-e2e-1-$BASHPID"
        mkdir -p "$collide/smg-ci"
        : > "$collide/smg-ci/left-by-another-lane"
        exec env TOKENSPEED_PREBUILT_IMAGE="$IMAGE" TOKENSPEED_PREBUILT_CACHE_ROOT="$CACHE" \
            TOKENSPEED_PREBUILT_INSTALL_ROOT="$T/opt-b" GITHUB_ENV="$T/opt-b/github.env" \
            GITHUB_RUN_ID=100 GITHUB_JOB=e2e GITHUB_RUN_ATTEMPT=1 bash "$SCRIPT"
    ) > /dev/null
    assert_link_into "$T/opt-b/smg-ci" "$CACHE/jobs"
    assert_no_file "$T/opt-b/smg-ci/left-by-another-lane"
}

test_failed_extraction_leaves_no_cache_entry_and_falls_back() {
    setup
    local out; out="$(run_fetch "$T/opt" 100 FAKE_CP_FAIL=1)"
    assert_contains "$out" "lane will build from source"
    assert_no_file "$CACHE/$TAG"
    assert_no_file "$T/opt/smg-ci"
    assert_eq "" "$(grep SMG_BAKED_VENV "$T/opt/github.env" || true)" "no venv advertised"
    # The cache root was set up, but no half-written entry may survive under any name.
    assert_file "$CACHE/jobs"
    assert_eq "" "$(find "$CACHE" -mindepth 1 -maxdepth 1 -type d ! -name jobs ! -name .locks)" "no partial entries"
}

test_unwritable_cache_root_degrades_to_a_job_local_cache() {
    setup
    : > "$T/not-a-dir"
    CACHE="$T/not-a-dir/cache"
    local out; out="$(run_fetch "$T/opt" 100)"
    assert_contains "$out" "job-local cache"
    assert_eq 1 "$(pull_count)" "still pulls the payload"
    assert_link_into "$T/opt/smg-ci" "$T/runner-temp"
    assert_eq "deadbeef" "$(cat "$T/opt/smg-ci/tokenspeed.ref")"
    assert_contains "$(cat "$T/opt/github.env")" "SMG_BAKED_VENV=$T/opt/smg-ci/.venv"
    # One reader, pod-local storage, no reflink: the payload must exist once,
    # in the job dir the workflow's cleanup step removes, not also in the entry.
    assert_no_file "$T/runner-temp/tokenspeed-prebuilt-cache/$TAG/smg-ci"
    assert_no_file "$T/runner-temp/tokenspeed-prebuilt-cache/$TAG/tokenspeed-src"
}

# A populate killed between the rename and the marker leaves an entry dir
# without .complete. The next populate must replace it, not move its temp dir
# inside it and then mark the nested result complete.
test_leftover_entry_without_marker_is_replaced_not_nested() {
    setup
    mkdir -p "$CACHE/$TAG"
    : > "$CACHE/$TAG/leftover"
    run_fetch "$T/opt" 100 > /dev/null
    assert_file "$CACHE/$TAG/.complete"
    assert_file "$CACHE/$TAG/smg-ci/tokenspeed.ref"
    assert_no_file "$CACHE/$TAG/leftover"
    assert_eq "" "$(find "$CACHE/$TAG" -mindepth 1 -maxdepth 1 -name ".*tmp*")" "no nested temp dir"
    assert_eq "deadbeef" "$(cat "$T/opt/smg-ci/tokenspeed.ref")"
}

# The marker alone must not be trusted: an entry missing its payload dirs is
# a miss and gets repopulated instead of poisoning every later lane.
test_marked_entry_missing_payload_is_treated_as_miss() {
    setup
    mkdir -p "$CACHE/$TAG"
    : > "$CACHE/$TAG/.complete"
    local out; out="$(run_fetch "$T/opt" 100)"
    assert_eq 1 "$(pull_count)" "repopulated"
    assert_file "$CACHE/$TAG/smg-ci/tokenspeed.ref"
    assert_contains "$out" "Prebuilt payload installed"
}

# An interrupted `rm -rf` of an entry can leave the marker plus emptied but
# present payload dirs. Directory existence is not proof; the files the venv
# needs must be there, or it is a miss.
test_hollow_entry_with_marker_is_treated_as_miss() {
    setup
    mkdir -p "$CACHE/$TAG/smg-ci" "$CACHE/$TAG/tokenspeed-src"
    : > "$CACHE/$TAG/.complete"
    run_fetch "$T/opt" 100 > /dev/null
    assert_eq 1 "$(pull_count)" "repopulated"
    assert_file "$CACHE/$TAG/smg-ci/.venv/bin/python"
    assert_eq "deadbeef" "$(cat "$T/opt/smg-ci/tokenspeed.ref")"
}

# If the second symlink cannot be created, the first one must not be left
# dangling in the install root, and the job copy must go with it.
test_partial_install_links_are_removed_on_failure() {
    setup
    cat > "$T/bin/ln" <<'FAKE'
#!/bin/bash
for a in "$@"; do case "$a" in *tokenspeed-src) exit 1 ;; esac; done
exec /bin/ln "$@"
FAKE
    chmod +x "$T/bin/ln"
    local out; out="$(run_fetch "$T/opt" 100)"
    assert_contains "$out" "lane will build from source"
    assert_no_file "$T/opt/smg-ci"
    assert_no_file "$T/opt/tokenspeed-src"
    assert_eq "" "$(find "$CACHE/jobs" -mindepth 1 -maxdepth 1)" "job copy removed"
}

# A failed GITHUB_ENV write means later steps never learn about the payload;
# roll the install back rather than report success.
test_unwritable_github_env_rolls_back_the_install() {
    setup
    local out; out="$(run_fetch "$T/opt" 100 GITHUB_ENV="$T/nonexistent/github.env")"
    assert_contains "$out" "lane will build from source"
    assert_no_file "$T/opt/smg-ci"
    assert_no_file "$T/opt/tokenspeed-src"
    assert_eq "" "$(find "$CACHE/jobs" -mindepth 1 -maxdepth 1)" "job copy removed"
}

test_concurrent_callers_pull_once() {
    setup
    run_fetch "$T/opt-a" 100 FAKE_PULL_SLEEP=2 > "$T/a.out" &
    sleep 0.3
    run_fetch "$T/opt-b" 200 FAKE_PULL_SLEEP=2 > "$T/b.out" &
    wait
    assert_eq 1 "$(pull_count)" "second caller waited for the first populate"
    assert_link_into "$T/opt-a/smg-ci" "$CACHE/jobs"
    assert_link_into "$T/opt-b/smg-ci" "$CACHE/jobs"
}

test_stale_job_dirs_are_swept() {
    setup
    mkdir -p "$CACHE/jobs/old-cancelled" "$CACHE/jobs/fresh"
    touch -d '2 days ago' "$CACHE/jobs/old-cancelled"
    run_fetch "$T/opt" 100 > /dev/null
    assert_no_file "$CACHE/jobs/old-cancelled"
    assert_file "$CACHE/jobs/fresh"
}

test_superseded_tag_dirs_are_swept_after_seven_days() {
    setup
    mkdir -p "$CACHE/ci-tokenspeed-old-tag" "$CACHE/ci-tokenspeed-recent-tag"
    touch "$CACHE/ci-tokenspeed-old-tag/.complete" "$CACHE/ci-tokenspeed-recent-tag/.complete"
    touch -d '10 days ago' "$CACHE/ci-tokenspeed-old-tag"
    touch -d '1 day ago' "$CACHE/ci-tokenspeed-recent-tag"
    run_fetch "$T/opt" 100 > /dev/null
    assert_no_file "$CACHE/ci-tokenspeed-old-tag"
    assert_file "$CACHE/ci-tokenspeed-recent-tag"
    assert_file "$CACHE/$TAG/.complete"
}

tests=(
    test_cold_pull_populates_cache_and_links_install_root
    test_warm_cache_skips_docker_entirely
    test_job_copies_are_private
    test_job_dir_is_unique_even_when_pid_and_job_identity_collide
    test_failed_extraction_leaves_no_cache_entry_and_falls_back
    test_unwritable_cache_root_degrades_to_a_job_local_cache
    test_leftover_entry_without_marker_is_replaced_not_nested
    test_marked_entry_missing_payload_is_treated_as_miss
    test_hollow_entry_with_marker_is_treated_as_miss
    test_partial_install_links_are_removed_on_failure
    test_unwritable_github_env_rolls_back_the_install
    test_concurrent_callers_pull_once
    test_stale_job_dirs_are_swept
    test_superseded_tag_dirs_are_swept_after_seven_days
)
# Optional: name one or more tests on the command line to run only those.
[ $# -eq 0 ] || tests=("$@")
for t in "${tests[@]}"; do
    "$t"
    echo "ok - $t"
done
