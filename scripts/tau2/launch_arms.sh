#!/usr/bin/env bash
# Launch one "arm" of the τ²-bench A/B comparison (agent arms only;
# user-sim is GPT-5.2 via the OpenAI API, handled externally).
#
#   arm A = pure vLLM OpenAI server  (vLLM owns chat template + tokenization +
#           tool/reasoning parsing)
#   arm B = SMG in front of a vLLM gRPC worker (SMG owns chat template +
#           tokenization + tool/reasoning parsing; vLLM runs raw-token)
#
# Both expose an identical OpenAI /v1 endpoint, so the τ²-bench harness can
# point at either and the ONLY thing that differs is the frontend — which is
# exactly the variable the A/B isolates.
#
# Everything is parameterised via env vars so the same script works on the H100
# box and in CI. Writes a pidfile + log per process so run_ab.py / the nightly
# can manage lifecycle and tear down cleanly.
#
# Usage:
#   launch_arms.sh a            # start pure-vLLM arm, print its base_url
#   launch_arms.sh b            # start vLLM-gRPC + SMG arm, print its base_url
#   launch_arms.sh stop         # kill anything this script started (via pidfiles)
set -euo pipefail

ARM="${1:?usage: launch_arms.sh <a|b|stop>}"

MODEL="${TAU2_MODEL:-Qwen/Qwen3.8-27B}"
# Load source: prefer a pre-staged local copy at $ROUTER_LOCAL_MODEL_PATH/<id>
# (e.g. NVMe /raid/models on the Blackwell node — no download); else the HF repo
# id, which vLLM/HF downloads into HF_HOME. The canonical served name stays
# $MODEL (passed as --served-model-name) so the τ²-bench handler + requests match.
MODEL_SRC="$MODEL"
if [ -n "${ROUTER_LOCAL_MODEL_PATH:-}" ] && [ -d "$ROUTER_LOCAL_MODEL_PATH/$MODEL" ]; then
  MODEL_SRC="$ROUTER_LOCAL_MODEL_PATH/$MODEL"
fi
GPU="${TAU2_GPU:-0}"                              # CUDA_VISIBLE_DEVICES (e.g. "0" or "0,1")
TP="${TAU2_TP:-1}"                               # tensor-parallel size (match GPU count)
MAX_MODEL_LEN="${TAU2_MAX_MODEL_LEN:-16384}"
GPU_MEM_UTIL="${TAU2_GPU_MEM_UTIL:-0.85}"
RUN_DIR="${TAU2_RUN_DIR:-/tmp/tau2_ab}"

# Pure-vLLM (arm A) tool/reasoning parser flags. `-` (not `:-`) on the reasoning
# parser: an explicit empty value passes through (the flag is omitted → no
# reasoning parse, e.g. gpt-oss/harmony or a non-thinking SKU); only an UNSET var
# defaults to qwen3. (The vLLM tool parser is always non-empty, so `:-` is fine.)
VLLM_TOOL_PARSER="${TAU2_VLLM_TOOL_PARSER:-qwen3_xml}"
VLLM_REASONING_PARSER="${TAU2_VLLM_REASONING_PARSER-qwen3}"
# SMG (arm B) parser flags — SMG registry names, NOT vLLM's. Same `-` rule: an
# explicit empty value passes through (omits the flag → SMG auto-detect, e.g.
# harmony for gpt-oss); only an unset var defaults.
SMG_TOOL_PARSER="${TAU2_SMG_TOOL_PARSER-qwen_xml}"
SMG_REASONING_PARSER="${TAU2_SMG_REASONING_PARSER-qwen3}"

# Extra args appended to every vLLM process (both arms). e.g.
# TAU2_VLLM_EXTRA="--enforce-eager" — skips CUDA-graph capture, which has been
# more stable under sustained tau2 load on shared/contended GPUs.
VLLM_EXTRA="${TAU2_VLLM_EXTRA:-}"

# Ports. Default to an OS-assigned free port (resolved per-arm in the case
# below) so concurrent arms/jobs sharing a host — e.g. bin-packed CI runners
# under hostNetwork — don't collide on a fixed port. Pin via env for a stable
# port on a dev box.
ARM_A_PORT="${TAU2_ARM_A_PORT:-}"          # pure-vLLM OpenAI port
ARM_B_GRPC_PORT="${TAU2_ARM_B_GRPC_PORT:-}" # vLLM gRPC worker port
ARM_B_GW_PORT="${TAU2_ARM_B_GW_PORT:-}"     # SMG OpenAI gateway port
ARM_B_METRICS_PORT="${TAU2_ARM_B_METRICS_PORT:-}" # SMG Prometheus port (defaults to 29000 — collides when arms/legs share a host)

# Executables (override for venv / box paths).
VLLM_BIN="${VLLM_BIN:-vllm}"                      # `vllm serve` console script
VLLM_PYTHON="${VLLM_PYTHON:-python}"             # python that can `-m vllm.entrypoints.grpc_server`
SMG_LAUNCH="${SMG_LAUNCH:-smg launch}"           # SMG launcher (binary subcmd or `python -m smg.launch_router`)

mkdir -p "$RUN_DIR"

# start <name> <logfile> <command...> — detached, pidfile-tracked.
start() {
  local name="$1" log="$2"; shift 2
  setsid env "$@" >"$log" 2>&1 </dev/null &
  echo $! >"$RUN_DIR/$name.pid"
  echo "[launch_arms] started $name (pid $(cat "$RUN_DIR/$name.pid")) -> $log" >&2
}

# Tail a logfile to stderr (streams live in CI; stdout is reserved for the
# base_url). Prints the tail pid so the caller can stop it once healthy.
stream_log() { tail -n +1 -F "$1" >&2 & echo $!; }

wait_http() {  # wait_http <url> <timeout_s>
  local url="$1" timeout="${2:-300}" waited=0
  until curl -sf -m 3 "$url" >/dev/null 2>&1; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then echo "[launch_arms] TIMEOUT waiting for $url" >&2; return 1; fi
  done
}

wait_grpc() {  # crude TCP-listen check for the gRPC port
  local port="$1" timeout="${2:-300}" waited=0
  until (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    sleep 5; waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then echo "[launch_arms] TIMEOUT waiting for grpc :$port" >&2; return 1; fi
  done
  exec 3>&- 2>/dev/null || true
}

free_port() {  # OS-assigned free TCP port (same idiom as the e2e infra's get_open_port)
  python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

case "$ARM" in
  a)
    ARM_A_PORT="${ARM_A_PORT:-$(free_port)}"
    declare -a cmd=(
      CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_BIN" serve "$MODEL_SRC"
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
    log_tail=$(stream_log "$RUN_DIR/arm_a_vllm.log")
    wait_http "http://127.0.0.1:$ARM_A_PORT/health" "${TAU2_STARTUP_TIMEOUT:-420}"
    kill "$log_tail" 2>/dev/null || true
    echo "http://127.0.0.1:$ARM_A_PORT"
    ;;

  b)
    ARM_B_GRPC_PORT="${ARM_B_GRPC_PORT:-$(free_port)}"
    ARM_B_GW_PORT="${ARM_B_GW_PORT:-$(free_port)}"
    ARM_B_METRICS_PORT="${ARM_B_METRICS_PORT:-$(free_port)}"
    # 1) vLLM gRPC worker (raw-token; SMG will own template+parsing).
    declare -a wcmd=(
      CUDA_VISIBLE_DEVICES="$GPU" "$VLLM_PYTHON" -m vllm.entrypoints.grpc_server
      --model "$MODEL_SRC" --served-model-name "$MODEL"
      --host 0.0.0.0 --port "$ARM_B_GRPC_PORT"
      --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
    )
    # shellcheck disable=SC2206  # intentional word-split of optional extra flags
    [ -n "$VLLM_EXTRA" ] && wcmd+=($VLLM_EXTRA)
    start arm_b_worker "$RUN_DIR/arm_b_worker.log" "${wcmd[@]}"
    log_tail=$(stream_log "$RUN_DIR/arm_b_worker.log")
    wait_grpc "$ARM_B_GRPC_PORT" "${TAU2_STARTUP_TIMEOUT:-420}"
    kill "$log_tail" 2>/dev/null || true
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
    log_tail=$(stream_log "$RUN_DIR/arm_b_gateway.log")
    wait_http "http://127.0.0.1:$ARM_B_GW_PORT/health" "${TAU2_STARTUP_TIMEOUT:-420}"
    kill "$log_tail" 2>/dev/null || true
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
      echo "[launch_arms] stopped process group $pid" >&2
    done
    ;;

  *)
    echo "usage: launch_arms.sh <a|b|stop>" >&2; exit 2;;
esac
