#!/usr/bin/env bash
# Launch one "arm" of the BFCL A/B comparison.
#
#   arm A = pure vLLM OpenAI server  (vLLM owns chat template + tokenization +
#           tool/reasoning parsing)
#   arm B = SMG in front of a vLLM gRPC worker (SMG owns chat template +
#           tokenization + tool/reasoning parsing; vLLM runs raw-token)
#
# Both expose an identical OpenAI /v1 endpoint, so the official BFCL harness can
# point at either and the ONLY thing that differs is the frontend — which is
# exactly the variable the A/B isolates.
#
# Everything is parameterised via env vars so the same script works on the H100
# box and in CI. Writes a pidfile + log per process so run_ab.py / the nightly
# can manage lifecycle and tear down cleanly.
#
# Usage:
#   launch_arm.sh a            # start pure-vLLM arm, print its base_url
#   launch_arm.sh b            # start vLLM-gRPC + SMG arm, print its base_url
#   launch_arm.sh stop         # kill anything this script started (via pidfiles)
set -euo pipefail

ARM="${1:?usage: launch_arm.sh <a|b|stop>}"

MODEL="${BFCL_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
# Load source: prefer a pre-staged local copy at $ROUTER_LOCAL_MODEL_PATH/<id>
# (e.g. NVMe /raid/models on the Blackwell node — no download); else the HF repo
# id, which vLLM/HF downloads into HF_HOME. The canonical served name stays
# $MODEL (passed as --served-model-name) so the BFCL handler + requests match.
MODEL_SRC="$MODEL"
if [ -n "${ROUTER_LOCAL_MODEL_PATH:-}" ] && [ -d "$ROUTER_LOCAL_MODEL_PATH/$MODEL" ]; then
  MODEL_SRC="$ROUTER_LOCAL_MODEL_PATH/$MODEL"
fi
GPU="${BFCL_GPU:-0}"                              # CUDA_VISIBLE_DEVICES (e.g. "0" or "0,1")
TP="${BFCL_TP:-1}"                               # tensor-parallel size (match GPU count)
MAX_MODEL_LEN="${BFCL_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${BFCL_GPU_MEM_UTIL:-0.55}"
RUN_DIR="${BFCL_RUN_DIR:-/tmp/bfcl_ab}"

# Pure-vLLM (arm A) tool/reasoning parser flags.
VLLM_TOOL_PARSER="${BFCL_VLLM_TOOL_PARSER:-hermes}"
VLLM_REASONING_PARSER="${BFCL_VLLM_REASONING_PARSER:-}"   # empty = none (non-thinking SKU)
# SMG (arm B) parser flags — SMG registry names, NOT vLLM's.
# `-` not `:-`: an explicit empty value passes through (omits the flag → SMG
# auto-detect, e.g. harmony for gpt-oss); only an unset var defaults to qwen.
SMG_TOOL_PARSER="${BFCL_SMG_TOOL_PARSER-qwen}"
SMG_REASONING_PARSER="${BFCL_SMG_REASONING_PARSER:-}"

# Extra args appended to every vLLM process (both arms). e.g.
# BFCL_VLLM_EXTRA="--enforce-eager" — skips CUDA-graph capture, which has been
# more stable under sustained bfcl load on shared/contended GPUs.
VLLM_EXTRA="${BFCL_VLLM_EXTRA:-}"
# Extra environment for every vLLM process (both arms), as space-separated
# NAME=VALUE words, e.g. BFCL_VLLM_ENV="VLLM_ROCM_USE_AITER=1" — engine knobs
# that vLLM reads only from the environment, set per matrix leg.
VLLM_ENV="${BFCL_VLLM_ENV:-}"

# Ports. Default to an OS-assigned free port (resolved per-arm in the case
# below) so concurrent arms/jobs sharing a host — e.g. bin-packed CI runners
# under hostNetwork — don't collide on a fixed port. Pin via env for a stable
# port on a dev box.
ARM_A_PORT="${BFCL_ARM_A_PORT:-}"          # pure-vLLM OpenAI port
ARM_B_GRPC_PORT="${BFCL_ARM_B_GRPC_PORT:-}" # vLLM gRPC worker port
ARM_B_GW_PORT="${BFCL_ARM_B_GW_PORT:-}"     # SMG OpenAI gateway port
ARM_B_METRICS_PORT="${BFCL_ARM_B_METRICS_PORT:-}" # SMG Prometheus port (defaults to 29000 — collides when arms/legs share a host)

# Executables (override for venv / box paths).
VLLM_BIN="${VLLM_BIN:-vllm}"                      # `vllm serve` console script
VLLM_PYTHON="${VLLM_PYTHON:-python}"             # python that can `-m vllm.entrypoints.grpc_server`
SMG_LAUNCH="${SMG_LAUNCH:-smg launch}"           # SMG launcher (binary subcmd or `python -m smg.launch_router`)

PROBE_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/arm_probe.py"
case "$ARM" in
  a|b)
    # Validate before either engine starts. Scoring keeps RUN_TIMEOUT verbatim;
    # only the observer maps uncapped/long runs to its existing finite bound.
    PROBE_DURATION=$(python3 "$PROBE_SCRIPT" \
      --bounded-duration-for-run-timeout="${RUN_TIMEOUT:-14400}")
    echo "[launch_arm] RUN_TIMEOUT validated; arm B observer bound: ${PROBE_DURATION}s (scoring RUN_TIMEOUT unchanged)" >&2
    ;;
esac

mkdir -p "$RUN_DIR"

# Only this invocation's direct startup children belong to this private owner.
# Coprocess transport FDs are close-on-exec; no engine inherits the control pipe.
if [[ "$ARM" == a || "$ARM" == b ]]; then
  coproc BFCL_OWNER { python3 "$(dirname "${BASH_SOURCE[0]}")/startup_owner.py"; }
  OWNER_PID=$BFCL_OWNER_PID
  OWNER_READ=${BFCL_OWNER[0]}
  OWNER_WRITE=${BFCL_OWNER[1]}
  STARTUP_COMMITTED=0
  STARTED_PIDFILES=()
  STARTED_PIDS=()
  startup_exit() {
    local status="$1" index saved_pid
    trap - EXIT INT TERM
    if [[ "$STARTUP_COMMITTED" != 1 ]]; then
      if owner_request abort >/dev/null 2>/dev/null && [[ "$OWNER_HANDLE" == ok ]]; then
        # These groups were reaped. A later legacy stop must not signal their
        # recycled numeric PIDs. Leave unrelated or replaced pidfiles alone.
        for index in "${!STARTED_PIDFILES[@]}"; do
          if IFS= read -r saved_pid <"${STARTED_PIDFILES[index]}" \
            && [[ "$saved_pid" == "${STARTED_PIDS[index]}" ]]; then
            rm -f -- "${STARTED_PIDFILES[index]}" || true
          fi
        done
      fi
    fi
    exec {OWNER_WRITE}>&- || true
    exec {OWNER_READ}<&- || true
    wait "$OWNER_PID" || true
    exit "$status"
  }
  trap 'startup_exit $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
fi

owner_request() {
  local status=0
  trap '' PIPE
  printf '%s\0' "$@" >&"$OWNER_WRITE" || status=$?
  trap - PIPE
  [[ "$status" == 0 ]] || return "$status"
  IFS=' ' read -r OWNER_HANDLE OWNER_CHILD_PID <&"$OWNER_READ"
}

commit_startup() {
  owner_request commit
  [[ "$OWNER_HANDLE" == ok ]]
  STARTUP_COMMITTED=1
}

# Commands are passed as argv, never through a shell or a persisted pidfile.
start() {
  local name="$1" log="$2"; shift 2
  owner_request start "$log" "$(($# + 1))" env "$@"
  echo "$OWNER_CHILD_PID" >"$RUN_DIR/$name.pid"
  STARTED_PIDFILES+=("$RUN_DIR/$name.pid")
  STARTED_PIDS+=("$OWNER_CHILD_PID")
  echo "[launch_arm] started $name (pid $OWNER_CHILD_PID) -> $log" >&2
}

# Tail a logfile to stderr (streams live in CI; stdout is reserved for the
# base_url). Call directly, not via $(...): LOG_TAIL_HANDLE is kept for stop_log.
stream_log() {
  owner_request start - 5 tail -n +1 -F "$1"
  LOG_TAIL_HANDLE=$OWNER_HANDLE
}

stop_log() {
  owner_request stop "$LOG_TAIL_HANDLE"
  [[ "$OWNER_HANDLE" == ok ]]
}

wait_http() {  # wait_http <url> <timeout_s>
  local url="$1" timeout="${2:-300}" waited=0
  until curl -sf -m 3 "$url" >/dev/null 2>&1; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then echo "[launch_arm] TIMEOUT waiting for $url" >&2; return 1; fi
  done
}

wait_grpc() {  # crude TCP-listen check for the gRPC port
  local port="$1" timeout="${2:-300}" waited=0
  until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then echo "[launch_arm] TIMEOUT waiting for grpc :$port" >&2; return 1; fi
  done
  exec 3>&- 2>/dev/null || true
}

free_port() {  # OS-assigned free TCP port (same idiom as the e2e infra's get_open_port)
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

case "$ARM" in
  a)
    ARM_A_PORT="${ARM_A_PORT:-$(free_port)}"
    # shellcheck disable=SC2206  # intentional word-split of VLLM_ENV
    declare -a cmd=(
      CUDA_VISIBLE_DEVICES="$GPU" $VLLM_ENV "$VLLM_BIN" serve "$MODEL_SRC"
      --served-model-name "$MODEL"
      --enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_PARSER"
      --host 0.0.0.0 --port "$ARM_A_PORT"
      --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
    )
    [ -n "$VLLM_REASONING_PARSER" ] && cmd+=(--reasoning-parser "$VLLM_REASONING_PARSER")
    # shellcheck disable=SC2206  # intentional word-split of optional extra flags
    [ -n "$VLLM_EXTRA" ] && cmd+=($VLLM_EXTRA)
    start arm_a "$RUN_DIR/arm_a_vllm.log" "${cmd[@]}"
    stream_log "$RUN_DIR/arm_a_vllm.log"
    wait_http "http://127.0.0.1:$ARM_A_PORT/health" "${BFCL_STARTUP_TIMEOUT:-420}"
    stop_log
    commit_startup
    echo "http://127.0.0.1:$ARM_A_PORT"
    ;;

  b)
    ARM_B_GRPC_PORT="${ARM_B_GRPC_PORT:-$(free_port)}"
    ARM_B_GW_PORT="${ARM_B_GW_PORT:-$(free_port)}"
    ARM_B_METRICS_PORT="${ARM_B_METRICS_PORT:-$(free_port)}"
    # 1) vLLM gRPC worker (raw-token; SMG will own template+parsing).
    # shellcheck disable=SC2206  # intentional word-split of VLLM_ENV
    declare -a wcmd=(
      CUDA_VISIBLE_DEVICES="$GPU" $VLLM_ENV "$VLLM_PYTHON" -m vllm.entrypoints.grpc_server
      --model "$MODEL_SRC" --served-model-name "$MODEL"
      --host 0.0.0.0 --port "$ARM_B_GRPC_PORT"
      --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
    )
    # shellcheck disable=SC2206  # intentional word-split of optional extra flags
    [ -n "$VLLM_EXTRA" ] && wcmd+=($VLLM_EXTRA)
    start arm_b_worker "$RUN_DIR/arm_b_worker.log" "${wcmd[@]}"
    stream_log "$RUN_DIR/arm_b_worker.log"
    wait_grpc "$ARM_B_GRPC_PORT" "${BFCL_STARTUP_TIMEOUT:-420}"
    stop_log
    # 2) SMG gateway in front, exposing the OpenAI API.
    # shellcheck disable=SC2206  # intentional word-split of SMG_LAUNCH
    declare -a smg_cmd=(
      $SMG_LAUNCH
      --model-path "$MODEL_SRC"
      --worker-urls "grpc://127.0.0.1:$ARM_B_GRPC_PORT"
      --host 0.0.0.0 --port "$ARM_B_GW_PORT"
      # Free port, not the fixed 29000 default — else a second SMG on the same
      # host (concurrent arm/leg) panics with "metrics server bind failed".
      --prometheus-port "$ARM_B_METRICS_PORT"
    )
    # Empty => omit, so SMG auto-detects (e.g. gpt-oss → harmony pipeline).
    [ -n "$SMG_TOOL_PARSER" ] && smg_cmd+=(--tool-call-parser "$SMG_TOOL_PARSER")
    [ -n "$SMG_REASONING_PARSER" ] && smg_cmd+=(--reasoning-parser "$SMG_REASONING_PARSER")
    start arm_b_gateway "$RUN_DIR/arm_b_gateway.log" "${smg_cmd[@]}"
    stream_log "$RUN_DIR/arm_b_gateway.log"
    # /health is process liveness, including while workers/tokenizers are not
    # ready. Require this model's readiness before handing BFCL the base URL.
    # Retain bounded state through inference too: startup success cannot explain
    # a worker that becomes unavailable several minutes into the full suite.
    python3 "$PROBE_SCRIPT" --base-url "http://127.0.0.1:$ARM_B_GW_PORT" \
      --model "$MODEL" --run-dir "$RUN_DIR" \
      --wait-ready "${BFCL_STARTUP_TIMEOUT:-420}"
    stop_log
    start arm_b_probe "$RUN_DIR/arm_b_probe.log" python3 "$PROBE_SCRIPT" \
      --base-url "http://127.0.0.1:$ARM_B_GW_PORT" --model "$MODEL" \
      --run-dir "$RUN_DIR" --duration "$PROBE_DURATION" \
      --metrics-url "http://127.0.0.1:$ARM_B_METRICS_PORT" \
      --worker-port "$ARM_B_GRPC_PORT"
    commit_startup
    echo "http://127.0.0.1:$ARM_B_GW_PORT"
    ;;

  stop)
    # Kill each recorded pid's whole PROCESS GROUP (negative pid). start() uses
    # setsid, so the pid is the group leader; killing only it orphaned vLLM's
    # worker children, leaving ~250 GiB pinned per GPU across jobs.
    declare -a pids=()
    for pf in "$RUN_DIR"/*.pid; do
      [ -e "$pf" ] || continue
      pids+=("$(cat "$pf")")
      rm -f "$pf"
    done
    [ "${#pids[@]}" -eq 0 ] && exit 0
    for pid in "${pids[@]}"; do
      kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5  # let them drain on SIGTERM before the hard kill
    for pid in "${pids[@]}"; do
      kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      echo "[launch_arm] stopped process group $pid" >&2
    done
    ;;

  *)
    echo "usage: launch_arm.sh <a|b|stop>" >&2; exit 2;;
esac
