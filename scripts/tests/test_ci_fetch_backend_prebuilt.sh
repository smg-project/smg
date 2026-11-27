#!/bin/bash
# Tests for scripts/ci_fetch_backend_prebuilt.sh: the per-node NVMe cache of
# prebuilt vLLM / SGLang / TRT-LLM carrier payloads and shared read-only
# CUDA 13.0 toolkit symlinks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ci_fetch_backend_prebuilt.sh"
TAG="ci-gpu-backend-sglang-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
IMAGE="ghcr.io/smg-project/smg:${TAG}"

assert_eq() {
    local expected="$1" actual="$2" what="${3:-}"
    if [ "$expected" != "$actual" ]; then
        echo "FAIL${what:+ ($what)}: expected '$expected', got '$actual'"
        exit 1
    fi
}

assert_file() { [ -e "$1" ] || { echo "FAIL: expected '$1' to exist"; exit 1; }; }
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

cleanup_sandbox() { [ -z "${T:-}" ] || rm -rf "$T"; }
trap cleanup_sandbox EXIT

setup() {
    cleanup_sandbox
    T="$(mktemp -d)"
    export FAKE_IMAGE_ROOT="$T/image" FAKE_DOCKER_LOG="$T/docker.log"
    for backend in vllm sglang trtllm; do
        mkdir -p "$FAKE_IMAGE_ROOT/opt/smg-ci/$backend/.venv/bin"
        echo "#!/bin/sh" > "$FAKE_IMAGE_ROOT/opt/smg-ci/$backend/.venv/bin/python"
        chmod +x "$FAKE_IMAGE_ROOT/opt/smg-ci/$backend/.venv/bin/python"
        echo "stamp-$backend" > "$FAKE_IMAGE_ROOT/opt/smg-ci/$backend/.venv/.smg-ci-recipe"
    done
    mkdir -p "$FAKE_IMAGE_ROOT/usr/local/cuda-13.0/bin"
    printf '#!/bin/sh\necho "Cuda compilation tools, release 13.0"\n' > "$FAKE_IMAGE_ROOT/usr/local/cuda-13.0/bin/nvcc"
    chmod +x "$FAKE_IMAGE_ROOT/usr/local/cuda-13.0/bin/nvcc"

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
            src="${FAKE_IMAGE_ROOT}${2#*:}"
            [ -e "$src" ] || exit 1
            cp -a "$src" "$3" ;;
    rm)     exit 0 ;;
    *)      echo "fake docker: unexpected '$*'" >&2; exit 2 ;;
esac
FAKE
    chmod +x "$T/bin/docker"
    cat > "$T/bin/sudo" <<'FAKE'
#!/bin/bash
exec "$@"
FAKE
    chmod +x "$T/bin/sudo"
    export PATH="$T/bin:$PATH"
    CACHE="$T/cache"
}

run_fetch() {
    local backend="$1" preinstalled="$2" cuda_root="$3" run_id="$4"; shift 4
    mkdir -p "$preinstalled" "$cuda_root"
    : > "$preinstalled/github.env"
    env SMG_BACKEND_PREBUILT_IMAGE="$IMAGE" \
        SMG_PREINSTALLED_ROOT="$preinstalled" \
        SMG_PREBUILT_CACHE_ROOT="$CACHE/$backend" \
        SMG_PREBUILT_CUDA_ROOT="$cuda_root" \
        GITHUB_ENV="$preinstalled/github.env" GITHUB_RUN_ID="$run_id" GITHUB_JOB="e2e" GITHUB_RUN_ATTEMPT="1" \
        RUNNER_TEMP="$T/runner-temp" \
        "$@" bash "$SCRIPT" "$backend"
}

pull_count() { grep -c '^pull ' "$FAKE_DOCKER_LOG" || true; }

test_cold_pull_populates_backend_and_shared_cuda_link() {
    setup
    local out; out="$(run_fetch sglang "$T/opt-smg" "$T/usr-local" 100)"
    assert_file "$CACHE/sglang/$TAG/.complete"
    assert_file "$CACHE/sglang/$TAG/smg-ci/sglang/.venv/.smg-ci-recipe"
    assert_link_into "$T/opt-smg/sglang" "$CACHE/sglang/jobs"
    assert_link_into "$T/usr-local/cuda-13.0" "$CACHE/sglang/$TAG"
    assert_file "$T/usr-local/cuda/bin/nvcc"
    assert_eq 1 "$(pull_count)" "one docker pull"
    assert_contains "$(cat "$T/opt-smg/github.env")" "SMG_PREBUILT_JOB_DIR=$CACHE/sglang/jobs/"
    assert_contains "$out" "Prebuilt sglang payload installed"
}

test_warm_cache_skips_docker_entirely() {
    setup
    run_fetch sglang "$T/opt-a" "$T/usr-a" 100 > /dev/null
    : > "$FAKE_DOCKER_LOG"
    local out; out="$(run_fetch sglang "$T/opt-b" "$T/usr-b" 200)"
    assert_eq "" "$(cat "$FAKE_DOCKER_LOG")" "no docker calls on warm cache"
    assert_link_into "$T/opt-b/sglang" "$CACHE/sglang/jobs"
    assert_link_into "$T/usr-b/cuda-13.0" "$CACHE/sglang/$TAG"
    assert_contains "$out" "cache hit"
}

test_job_copies_are_private_and_cuda_is_shared_readonly() {
    setup
    run_fetch trtllm "$T/opt-a" "$T/usr-a" 100 > /dev/null
    run_fetch trtllm "$T/opt-b" "$T/usr-b" 200 > /dev/null
    [ "$(readlink "$T/opt-a/trtllm")" != "$(readlink "$T/opt-b/trtllm")" ] || { echo "FAIL: shared job venv"; exit 1; }
    assert_eq "$(readlink "$T/usr-a/cuda-13.0")" "$(readlink "$T/usr-b/cuda-13.0")" "CUDA toolkit shared read-only"
    echo "pr-wheel" > "$T/opt-a/trtllm/.venv/pr.pth"
    assert_no_file "$CACHE/trtllm/$TAG/smg-ci/trtllm/.venv/pr.pth"
    assert_no_file "$T/opt-b/trtllm/.venv/pr.pth"
}

test_self_resolves_content_addressed_tag_when_env_unset() {
    setup
    local out; out="$(run_fetch trtllm "$T/opt-auto" "$T/usr-auto" 100 SMG_BACKEND_PREBUILT_IMAGE="")"
    assert_contains "$out" "Prebuilt image: ghcr.io/smg-project/smg:ci-gpu-backend-trtllm-"
    assert_eq 1 "$(pull_count)" "pulled self-resolved tag"
    assert_link_into "$T/opt-auto/trtllm" "$CACHE/trtllm/jobs"
}

test_failed_pull_falls_back_softly() {
    setup
    local out; out="$(run_fetch vllm "$T/opt-fail" "$T/usr-fail" 100 FAKE_PULL_FAIL=1)"
    assert_contains "$out" "lane will use fresh environment setup"
    assert_no_file "$T/opt-fail/vllm"
    assert_eq "" "$(cat "$T/opt-fail/github.env")" "no job dir exported on failure"
}

test_concurrent_callers_pull_once() {
    setup
    run_fetch sglang "$T/opt-a" "$T/usr-a" 100 FAKE_PULL_SLEEP=2 > "$T/a.out" &
    sleep 0.3
    run_fetch sglang "$T/opt-b" "$T/usr-b" 200 FAKE_PULL_SLEEP=2 > "$T/b.out" &
    wait
    assert_eq 1 "$(pull_count)" "second caller waited on flock"
    assert_link_into "$T/opt-a/sglang" "$CACHE/sglang/jobs"
    assert_link_into "$T/opt-b/sglang" "$CACHE/sglang/jobs"
}

test_stale_jobs_and_superseded_tags_are_swept() {
    setup
    mkdir -p "$CACHE/sglang/jobs/old-job" "$CACHE/sglang/jobs/fresh-job"
    mkdir -p "$CACHE/sglang/ci-gpu-backend-sglang-old" "$CACHE/sglang/ci-gpu-backend-sglang-recent"
    touch -d '2 days ago' "$CACHE/sglang/jobs/old-job"
    touch -d '10 days ago' "$CACHE/sglang/ci-gpu-backend-sglang-old"
    touch -d '1 day ago' "$CACHE/sglang/ci-gpu-backend-sglang-recent"
    run_fetch sglang "$T/opt-sweep" "$T/usr-sweep" 100 > /dev/null
    assert_no_file "$CACHE/sglang/jobs/old-job"
    assert_file "$CACHE/sglang/jobs/fresh-job"
    assert_no_file "$CACHE/sglang/ci-gpu-backend-sglang-old"
    assert_file "$CACHE/sglang/ci-gpu-backend-sglang-recent"
}

tests=(
    test_cold_pull_populates_backend_and_shared_cuda_link
    test_warm_cache_skips_docker_entirely
    test_job_copies_are_private_and_cuda_is_shared_readonly
    test_self_resolves_content_addressed_tag_when_env_unset
    test_failed_pull_falls_back_softly
    test_concurrent_callers_pull_once
    test_stale_jobs_and_superseded_tags_are_swept
)
[ $# -eq 0 ] || tests=("$@")
for t in "${tests[@]}"; do
    "$t"
    echo "ok - $t"
done
