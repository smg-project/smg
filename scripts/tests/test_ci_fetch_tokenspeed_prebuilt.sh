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
assert_no_file() { [ ! -e "$1" ] || { echo "FAIL: expected '$1' to be absent"; exit 1; }; }
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
# and an install root standing in for /opt.
setup() {
    T="$(mktemp -d)"
    export FAKE_IMAGE_ROOT="$T/image" FAKE_DOCKER_LOG="$T/docker.log"
    mkdir -p "$FAKE_IMAGE_ROOT/opt/smg-ci/.venv/bin" "$FAKE_IMAGE_ROOT/opt/tokenspeed-src"
    echo "#!/bin/sh" > "$FAKE_IMAGE_ROOT/opt/smg-ci/.venv/bin/python"
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
    env "$@" TOKENSPEED_PREBUILT_IMAGE="$IMAGE" \
        TOKENSPEED_PREBUILT_CACHE_ROOT="$CACHE" \
        TOKENSPEED_PREBUILT_INSTALL_ROOT="$install" \
        GITHUB_ENV="$install/github.env" GITHUB_RUN_ID="$run_id" GITHUB_JOB="e2e" GITHUB_RUN_ATTEMPT="1" \
        bash "$SCRIPT"
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

test_unwritable_cache_root_falls_back_to_source_build() {
    setup
    : > "$T/not-a-dir"
    CACHE="$T/not-a-dir/cache"
    local out; out="$(run_fetch "$T/opt" 100)"
    assert_contains "$out" "lane will build from source"
    assert_eq 0 "$(pull_count)" "no pull attempted"
    assert_no_file "$T/opt/smg-ci"
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
    test_unwritable_cache_root_falls_back_to_source_build
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
