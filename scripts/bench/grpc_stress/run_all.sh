#!/usr/bin/env bash
# The cell matrix of the gRPC router stress bed: eight traffic families, each on
# a bed of its own, the cells repeated and interleaved (see README.md).
#
#   run_all.sh [family ...]     families: unary stream idle idle-default long image tenants tools responses
#                               (default: all of them, in that order)
#
# Environment: GS_DIR (run directory) [target/grpc-stress]; SMG_BIN, MOCK_BIN, LOAD_BIN (binaries, as bed.sh and
# cell.sh); TOKENIZER (a chat tokenizer directory or hub id, required); LLAVA_TOKENIZER (a llava-hf model directory
# with config.json and tokenizer.json, for the image family; skipped when unset); GW_CORES, MOCK_CORES, GEN_CORES
# (cpu lists); REPEATS [3]; DURATION [120]; WARMUP [10]; DRAIN [60]; PERF_SECS (a perf record of the gateway on
# the second repeat of every cell, 0 = off) [30]; LOAD_MAX (wait while the host's 1-minute load average is above
# it, 0 = never wait) [0]; HEAP_LG_SAMPLE (when set: one extra 60 s repeat per cell on a bed whose gateway runs
# with jemalloc profiling at that lg_prof_sample, heap profiles at the window's start and end; needs a gateway
# built with --features jemalloc-profiling) [unset]; HEAP_ONLY=1 runs only that pass.
#
# Exits non-zero when a bed did not come up, a cell failed or the report could not be written.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
export GS_DIR=${GS_DIR:-$ROOT/target/grpc-stress}
export SMG_BIN=${SMG_BIN:-$ROOT/target/release/smg} MOCK_BIN=${MOCK_BIN:-$ROOT/target/release/mock-worker}
export LOAD_BIN=${LOAD_BIN:-$ROOT/target/release/stress_load}
: "${TOKENIZER:?set TOKENIZER to a chat tokenizer directory or hub id}"
export TOKENIZER GW_CORES=${GW_CORES:-} MOCK_CORES=${MOCK_CORES:-} GEN_CORES=${GEN_CORES:-}
REPEATS=${REPEATS:-3}; DURATION=${DURATION:-120}; WARMUP=${WARMUP:-10}; DRAIN=${DRAIN:-60}
PERF_ON=${PERF_SECS:-30}; LOAD_MAX=${LOAD_MAX:-0}; HEAP_LG_SAMPLE=${HEAP_LG_SAMPLE:-}
LLAVA_TOKENIZER=${LLAVA_TOKENIZER:-}
families=("$@"); [ ${#families[@]} -gt 0 ] || families=(unary stream idle idle-default long image tenants tools responses)
mkdir -p "$GS_DIR"
failed=0
ts() { date -u +%FT%TZ; }
log() { echo "$(ts) run_all: $*" | tee -a "$GS_DIR/run_all.log"; }
wait_for_load() {
  [ "$LOAD_MAX" = 0 ] && return 0
  while :; do
    load=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN { exit !(l > m) }' || return 0
    log "load average $load above $LOAD_MAX: waiting"
    sleep 30
  done
}
ADMIN=http://127.0.0.1:${ADMIN_PORT:-29609}
common=(--duration "$DURATION" --warmup "$WARMUP" --drain "$DRAIN")

# cell <family> <variant> <repeat> <perf> <heap> -- <stress_load args>
cell() {
  local fam=$1 var=$2 rep=$3 perf=$4 heap=$5; shift 5; [ "${1:-}" = "--" ] && shift
  wait_for_load
  if [ "$fam" = idle ] || [ "$fam" = idle-default ]; then
    # Engines frozen: the streams open and wait; after 20 s of opening and 30 s of holding they all run.
    curl -s -X POST "$ADMIN/admin/fault/all/pause" > /dev/null
    ( sleep 50; curl -s -X POST "$ADMIN/admin/fault/all/resume" > /dev/null ) &
  fi
  PERF_SECS=$perf HEAP=$heap "$HERE/cell.sh" "$fam-$var-r$rep" -- "$@" 2>&1 | tee -a "$GS_DIR/run_all.log" \
    || { log "cell $fam-$var-r$rep failed"; failed=$((failed + 1)); }
  wait
}

# family <name> <bed env...> -- <variant>=<args>...
run_family() {
  local fam=$1; shift
  local -a env_pairs=() variants=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do env_pairs+=("$1"); shift; done
  shift
  variants=("$@")
  if [ "${HEAP_ONLY:-0}" != 1 ]; then
    log "family $fam: bed up (${env_pairs[*]})"
    env "${env_pairs[@]}" "$HERE/bed.sh" up 2>&1 | tee -a "$GS_DIR/run_all.log" || { log "family $fam: bed failed"; "$HERE/bed.sh" down; failed=$((failed + 1)); return 1; }
    for rep in $(seq 1 "$REPEATS"); do
      for v in "${variants[@]}"; do
        local name=${v%%=*} args=${v#*=}
        local perf=0; { [ "$rep" = 2 ] || [ "$REPEATS" = 1 ]; } && perf=$PERF_ON
        # shellcheck disable=SC2086
        cell "$fam" "$name" "$rep" "$perf" 0 -- $args
      done
    done
    "$HERE/bed.sh" down 2>&1 | tee -a "$GS_DIR/run_all.log"
  fi
  if [ -n "$HEAP_LG_SAMPLE" ]; then
    log "family $fam: heap-profile bed up (lg_prof_sample $HEAP_LG_SAMPLE)"
    mkdir -p "$GS_DIR/heap"
    local gw_args=""
    for p in "${env_pairs[@]}"; do case "$p" in GW_ARGS=*) gw_args=${p#GW_ARGS=} ;; esac; done
    env "${env_pairs[@]}" GW_ARGS="$gw_args --jemalloc-prof-dir $GS_DIR/heap" \
      GW_ENV="_RJEM_MALLOC_CONF=prof:true,prof_active:true,prof_accum:true,lg_prof_sample:$HEAP_LG_SAMPLE" \
      "$HERE/bed.sh" up 2>&1 | tee -a "$GS_DIR/run_all.log" \
      || { log "family $fam: heap-profile bed failed"; "$HERE/bed.sh" down; failed=$((failed + 1)); return 1; }
    for v in "${variants[@]}"; do
      local name=${v%%=*} args=${v#*=} heap_args
      # A 60 s cell, except for the idle families: their 20 s opening window is the load the
      # profile is of (10k open streams, dumped at the window's end while the engines are still
      # paused), and a longer window would open more streams than the bed admits.
      case "$fam" in
        idle|idle-default) heap_args=$args ;;
        *) heap_args=$(printf '%s' "$args" | sed -E "s/--duration [0-9]+/--duration 60/") ;;
      esac
      # shellcheck disable=SC2086
      cell "$fam" "$name" heap 0 1 -- $heap_args
    done
    "$HERE/bed.sh" down 2>&1 | tee -a "$GS_DIR/run_all.log"
  fi
}

for fam in "${families[@]}"; do
  case "$fam" in
    unary)
      run_family unary MODEL=mock-model -- \
        "u1k=--mode chat --rate 1000 --prompt-tokens 64 --max-tokens 8 ${common[*]}" \
        "u3k=--mode chat --rate 3000 --prompt-tokens 64 --max-tokens 8 ${common[*]}" \
        "u5k=--mode chat --rate 5000 --prompt-tokens 64 --max-tokens 8 ${common[*]}" ;;
    stream)
      run_family stream MODEL=mock-model -- \
        "s500=--mode chat --stream --rate 500 --prompt-tokens 64 --max-tokens 64 ${common[*]}" \
        "s1500=--mode chat --stream --rate 1500 --prompt-tokens 64 --max-tokens 64 ${common[*]}" ;;
    idle)
      run_family idle MODEL=mock-model "GW_ARGS=--max-concurrent-requests 20000" -- \
        "idle10k=--mode chat --stream --rate 500 --prompt-tokens 32 --max-tokens 16 --warmup 0 --duration 20 --drain 300" ;;
    idle-default)
      run_family idle-default MODEL=mock-model -- \
        "idle10k=--mode chat --stream --rate 500 --prompt-tokens 32 --max-tokens 16 --warmup 0 --duration 20 --drain 300" ;;
    long)
      run_family long MODEL=mock-model -- \
        "l32k=--mode chat --rate 50 --prompt-tokens 32000 --max-tokens 8 --distinct 32 --max-inflight 4000 ${common[*]}" \
        "l128k=--mode chat --rate 50 --prompt-tokens 128000 --max-tokens 8 --distinct 32 --max-inflight 4000 ${common[*]}" ;;
    image)
      if [ -z "$LLAVA_TOKENIZER" ]; then log "image family skipped (LLAVA_TOKENIZER unset)"; continue; fi
      mkdir -p "$GS_DIR/images"
      [ -f "$GS_DIR/images/noise-3.png" ] || python3 "$HERE/gen_images.py" --out "$GS_DIR/images" --count 4 --bytes 4000000 | tee -a "$GS_DIR/run_all.log"
      imgs="--image $GS_DIR/images/noise-0.png --image $GS_DIR/images/noise-1.png --image $GS_DIR/images/noise-2.png --image $GS_DIR/images/noise-3.png"
      for mode in router worker; do
        # The served id carries a vision-processor pattern ("llava-1.5"): the router picks the image
        # processor by substring of the model id (or of config.json's model_type).
        run_family "image-$mode" MODEL=mock-llava-1.5 "TOKENIZER=$LLAVA_TOKENIZER" "GW_ARGS=--mm-processing $mode" -- \
          "img4mb=--mode image $imgs --rate 100 --distinct 16 --prompt-tokens 32 --max-tokens 8 --model mock-llava-1.5 ${common[*]}"
      done ;;
    tenants)
      python3 "$HERE/gen_tenants.py" --out "$GS_DIR/tenants.yaml" --count 200 | tee -a "$GS_DIR/run_all.log"
      run_family tenants MODEL=mock-model "GW_ARGS=--trust-tenant-header --tenant-rate-limit-enabled --tenant-rate-limit-config $GS_DIR/tenants.yaml" -- \
        "t200=--mode chat --rate 1000 --tenants 200 --prompt-tokens 64 --max-tokens 8 ${common[*]}" ;;
    tools)
      run_family tools MODEL=mock-model "GW_ARGS=--parser symphony --tool-call-parser qwen --reasoning-parser qwen3" -- \
        "tools500=--mode tools --rate 500 --turns 10 --tools 8 --max-tokens 8 ${common[*]}" \
        "tools500s=--mode tools --stream --rate 500 --turns 10 --tools 8 --max-tokens 64 ${common[*]}" ;;
    responses)
      run_family responses MODEL=mock-model -- \
        "resp500=--mode responses --rate 500 --chain 10 --prompt-tokens 64 --max-tokens 8 ${common[*]}" ;;
    *) log "unknown family $fam"; exit 2 ;;
  esac
done
log "report"
python3 "$HERE/report.py" "$GS_DIR" --md "$GS_DIR/report.md" --json "$GS_DIR/report.json" | tee -a "$GS_DIR/run_all.log" \
  || { log "report failed"; failed=$((failed + 1)); }
[ "$failed" -eq 0 ] || { log "$failed failure(s), see $GS_DIR/run_all.log"; exit 1; }
