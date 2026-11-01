#!/usr/bin/env bash
#
# E2E Test Runner for SMG
#
# Usage:
#   ./scripts/e2e-test.sh              # Run all e2e tests
#   ./scripts/e2e-test.sh router       # Run router tests only
#   ./scripts/e2e-test.sh chat         # Run chat completions tests
#   ./scripts/e2e-test.sh responses    # Run responses tests
#   ./scripts/e2e-test.sh embeddings   # Run embeddings tests
#   ./scripts/e2e-test.sh benchmarks   # Run benchmarks
#   ./scripts/e2e-test.sh go           # Run Go bindings e2e tests
#
# Environment variables:
#   ROUTER_LOCAL_MODEL_PATH  - Path to local models (default: ~/models)
#   SHOW_WORKER_LOGS         - Show worker logs (default: 0)
#   SHOW_ROUTER_LOGS         - Show router logs (default: 1)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output (only if stdout is a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

# Default environment variables
export SHOW_WORKER_LOGS="${SHOW_WORKER_LOGS:-0}"
export SHOW_ROUTER_LOGS="${SHOW_ROUTER_LOGS:-1}"
export ROUTER_LOCAL_MODEL_PATH="${ROUTER_LOCAL_MODEL_PATH:-$HOME/models}"

# Test directories
E2E_DIR="$ROOT_DIR/e2e_test"

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_dependencies() {
    info "Checking dependencies..."

    if ! command -v python3 &> /dev/null; then
        error "python3 is required but not found"
    fi

    if ! python3 -c "import pytest" &> /dev/null; then
        error "pytest is required. Install with: pip install pytest pytest-rerunfailures"
    fi

    # Check for optional but recommended packages
    if ! python3 -c "import httpx" &> /dev/null; then
        warn "httpx not found. Install with: pip install httpx"
    fi

    if ! python3 -c "import openai" &> /dev/null; then
        warn "openai not found. Install with: pip install openai"
    fi
}

cleanup_processes() {
    info "Cleaning up any stale processes..."
    if [[ -f "$SCRIPT_DIR/ci_killall_sglang.sh" ]]; then
        # the verdict is discarded here: one reading, no wait for a draining card
        CI_GPU_GUARD_WAIT_SECS=0 bash "$SCRIPT_DIR/ci_killall_sglang.sh" "nuke_gpus" 2>/dev/null || true
    fi
}

run_pytest() {
    local test_dirs="$1"
    local extra_args="${2:-}"

    info "Running tests in: $test_dirs"
    info "Model path: $ROUTER_LOCAL_MODEL_PATH"

    # Build pytest command as array to avoid eval and handle paths safely
    local pytest_cmd=(python3 -m pytest --reruns 2 --reruns-delay 5 -s -vv -o log_cli=true --log-cli-level=INFO)

    if [[ -n "$extra_args" ]]; then
        pytest_cmd+=($extra_args)
    fi

    pytest_cmd+=($test_dirs)

    echo ""
    info "Command: ${pytest_cmd[*]}"
    echo ""

    "${pytest_cmd[@]}"
}

run_go_e2e() {
    info "Running Go bindings E2E tests..."

    if ! command -v go &> /dev/null; then
        error "go is required for Go bindings tests"
    fi

    cd "$ROOT_DIR/bindings/golang"

    # Build the FFI library if needed
    if [[ ! -f "target/release/libsmg_go.dylib" ]] && [[ ! -f "target/release/libsmg_go.so" ]]; then
        info "Building Go FFI library..."
        cargo build --release
    fi

    export CGO_LDFLAGS="-L$(pwd)/target/release"
    export LD_LIBRARY_PATH="$(pwd)/target/release:${LD_LIBRARY_PATH:-}"

    cd "$ROOT_DIR"
    run_pytest "$E2E_DIR/bindings_go"
}

print_usage() {
    echo "SMG E2E Test Runner"
    echo ""
    echo "Usage: $0 [test_suite]"
    echo ""
    echo "Test suites:"
    echo "  all         Run all e2e tests (default)"
    echo "  router      Run router tests (e2e_test/router)"
    echo "  chat        Run chat completions tests (e2e_test/chat_completions)"
    echo "  responses   Run responses tests (e2e_test/responses)"
    echo "  embeddings  Run embeddings tests (e2e_test/embeddings)"
    echo "  realtime    Run realtime WebSocket tests (e2e_test/realtime)"
    echo "  benchmarks  Run benchmarks (e2e_test/benchmarks)"
    echo "  go          Run Go bindings e2e tests (e2e_test/bindings_go)"
    echo ""
    echo "Environment variables:"
    echo "  ROUTER_LOCAL_MODEL_PATH  Path to local models (default: ~/models)"
    echo "  SHOW_WORKER_LOGS         Show worker logs (default: 0)"
    echo "  SHOW_ROUTER_LOGS         Show router logs (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 router                             # Run router tests only"
    echo "  ROUTER_LOCAL_MODEL_PATH=/data/models $0 chat  # Custom model path"
}

main() {
    local suite="${1:-all}"

    if [[ "$suite" == "-h" ]] || [[ "$suite" == "--help" ]]; then
        print_usage
        exit 0
    fi

    cd "$ROOT_DIR"

    check_dependencies
    cleanup_processes

    case "$suite" in
        all)
            info "Running all E2E tests..."
            run_pytest "$E2E_DIR/router $E2E_DIR/embeddings $E2E_DIR/chat_completions $E2E_DIR/responses"
            ;;
        router)
            run_pytest "$E2E_DIR/router"
            ;;
        chat|chat_completions|chat-completions)
            run_pytest "$E2E_DIR/chat_completions"
            ;;
        responses)
            run_pytest "$E2E_DIR/responses"
            ;;
        embeddings)
            run_pytest "$E2E_DIR/embeddings"
            ;;
        realtime)
            run_pytest "$E2E_DIR/realtime"
            ;;
        benchmarks)
            run_pytest "$E2E_DIR/benchmarks"
            ;;
        go|golang|go-bindings)
            run_go_e2e
            ;;
        *)
            error "Unknown test suite: $suite. Use --help for usage."
            ;;
    esac

    info "Tests completed!"
}

main "$@"
