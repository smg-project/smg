#!/usr/bin/env bash
# The CPU-only stress bed for the gateway's gRPC request path: one release
# gateway in front of a fleet of `mock-worker --engine realistic` gRPC engines
# (KV events on). No GPU, no engine: the whole bed is this repository.
#
#   bed.sh up      start the fleet and the gateway, register the workers, wait until a probe answers
#   bed.sh down    stop both (pids from the bed's own pid files only)
#   bed.sh status  print bed.json (pids, ports, the gateway's cgroup)
#
# Environment (defaults in brackets):
#   GS_DIR       run directory for logs, pids and cells [target/grpc-stress]
#   SMG_BIN      gateway binary [target/release/smg]      MOCK_BIN  [target/release/mock-worker]
#   GW_PORT      [29600]  PROM_PORT [29601]  ADMIN_PORT [29609]  MOCK_BASE [29610]  WORKERS [12]
#   MODEL        served model id [mock-model]   TOKENIZER  tokenizer directory or hub id (required)
#   BLOCK        KV block size [16]
#   GW_CORES     cpu list the gateway is pinned to (taskset) [none]   GW_THREADS  tokio worker threads [8]
#   MOCK_CORES   cpu list the fleet is pinned to [none]
#   ENGINE_ARGS  mock engine flags [a realistic engine with fast prefill and 10 ms decode steps]
#   GW_ARGS      extra gateway flags (e.g. "--parser symphony --trust-tenant-header")
#   GW_ENV       extra KEY=VALUE pairs for the gateway's environment (e.g. _RJEM_MALLOC_CONF=...)
#   GW_LOG_LEVEL [warn]
#   SCOPE        auto|1|0: run the gateway and the fleet each in a transient systemd user scope of their own
#                (memory accounting on: the gateway's cgroup memory can be sampled, and neither is throttled by
#                the memory controller of the shell's own cgroup) [auto: when systemd-run works]
set -u
cmd=${1:-}
[ -n "$cmd" ] || { echo "usage: bed.sh up|down|status" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
GS_DIR=${GS_DIR:-$ROOT/target/grpc-stress}
BED=$GS_DIR/bed
SMG_BIN=${SMG_BIN:-$ROOT/target/release/smg}
MOCK_BIN=${MOCK_BIN:-$ROOT/target/release/mock-worker}
GW_PORT=${GW_PORT:-29600}; PROM_PORT=${PROM_PORT:-29601}; ADMIN_PORT=${ADMIN_PORT:-29609}
MOCK_BASE=${MOCK_BASE:-29610}; WORKERS=${WORKERS:-12}
MODEL=${MODEL:-mock-model}; BLOCK=${BLOCK:-16}
GW_CORES=${GW_CORES:-}; MOCK_CORES=${MOCK_CORES:-}; GW_THREADS=${GW_THREADS:-8}
ENGINE_ARGS=${ENGINE_ARGS:---engine realistic --timing linear --prefill-tps 2000000 --decode-base-ms 10 --decode-per-req-ms 0 --max-running 4096 --max-batched-tokens 1000000 --context-length 262144 --kv-tokens 2097152}
GW_ARGS=${GW_ARGS:-}; GW_ENV=${GW_ENV:-}; GW_LOG_LEVEL=${GW_LOG_LEVEL:-warn}
SCOPE=${SCOPE:-auto}
GW=http://127.0.0.1:$GW_PORT
ts() { date -u +%FT%TZ; }
die() { echo "$(ts) bed: $*" >&2; exit 1; }
# The prefix that runs a command in a transient systemd user scope of its own (memory accounting on), so the bed's
# processes are not throttled by the memory controller of the shell's cgroup and the gateway's memory can be read
# from its cgroup. Empty when systemd-run is not usable here (SCOPE=0 forces that).
scope_prefix() {
  local unit=$1
  if [ "$SCOPE" != "0" ] && command -v systemd-run >/dev/null 2>&1 \
     && systemd-run --user --scope --quiet -p MemoryAccounting=yes true 2>/dev/null; then
    echo "systemd-run --user --scope --quiet --unit $unit -p MemoryAccounting=yes"
  elif [ "$SCOPE" = "1" ]; then
    die "SCOPE=1 but systemd-run --user --scope does not work here"
  fi
}

stop_pid_file() {
  local f=$1 pid
  [ -f "$f" ] || return 0
  pid=$(cat "$f")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$f"
}

case "$cmd" in
  down)
    stop_pid_file "$BED/gateway.pid"
    stop_pid_file "$BED/mock.pid"
    echo "$(ts) bed: down"
    ;;
  status)
    cat "$BED/bed.json" 2>/dev/null || die "no bed.json under $BED"
    ;;
  up)
    : "${TOKENIZER:?set TOKENIZER to a tokenizer directory or hub id}"
    [ -x "$SMG_BIN" ] || die "no gateway binary at $SMG_BIN"
    [ -x "$MOCK_BIN" ] || die "no mock-worker binary at $MOCK_BIN"
    mkdir -p "$BED"
    [ -f "$BED/gateway.pid" ] && die "a bed is up already (bed.sh down first)"
    ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    # Every port the bed binds must be free: a worker whose port another process holds exits quietly
    # and the gateway would route a share of the traffic into that stranger.
    for p in "$GW_PORT" "$PROM_PORT" "$ADMIN_PORT" $(seq "$MOCK_BASE" $((MOCK_BASE + WORKERS - 1))); do
      if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then exec 3>&- 3<&-; die "port $p is in use by another process (set GW_PORT/PROM_PORT/ADMIN_PORT/MOCK_BASE to a free range)"; fi
    done

    stamp=$(date +%s)
    # The prefix is taken outside the read's substitution, so SCOPE=1 without a usable systemd-run stops here.
    fleet_prefix=$(scope_prefix "grpc-stress-fleet-$MOCK_BASE-$stamp") || exit 1
    gw_prefix=$(scope_prefix "grpc-stress-gw-$GW_PORT-$stamp") || exit 1
    read -r -a fleet_scope <<< "$fleet_prefix"
    read -r -a gw_scope <<< "$gw_prefix"
    # shellcheck disable=SC2206
    mock_cmd=("${fleet_scope[@]}" ${MOCK_CORES:+taskset -c $MOCK_CORES} "$MOCK_BIN" --grpc-base-port "$MOCK_BASE" --grpc-count "$WORKERS" \
      --model "$MODEL" --tokenizer "$TOKENIZER" --block-size "$BLOCK" --admin-port "$ADMIN_PORT" $ENGINE_ARGS)
    echo "$(ts) bed: fleet: ${mock_cmd[*]}"
    "${mock_cmd[@]}" > "$BED/mock.log" 2>&1 &
    mock_pid=$!
    sleep 1
    if ! grep -q -E '^mock-worker$' "/proc/$mock_pid/comm" 2>/dev/null; then
      child=$(pgrep -P "$mock_pid" | head -1); [ -n "$child" ] && mock_pid=$child
    fi
    echo "$mock_pid" > "$BED/mock.pid"
    for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$ADMIN_PORT/admin/health" >/dev/null 2>&1 && break; sleep 0.5; done
    if grep -q -E 'bind .* failed|Address already in use' "$BED/mock.log"; then
      grep -E 'bind .* failed|Address already in use' "$BED/mock.log" >&2
      stop_pid_file "$BED/mock.pid"; die "a mock worker could not bind its port"
    fi
    fleet=$(curl -s "http://127.0.0.1:$ADMIN_PORT/admin/fleet" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["workers"] if isinstance(d,dict) else d; print(len(w))' 2>/dev/null)
    [ "${fleet:-0}" = "$WORKERS" ] || { stop_pid_file "$BED/mock.pid"; die "the fleet has ${fleet:-0} workers, not $WORKERS (see $BED/mock.log)"; }

    # shellcheck disable=SC2206
    gw_cmd=("${gw_scope[@]}" ${GW_CORES:+taskset -c $GW_CORES} env $GW_ENV "$SMG_BIN" --host 127.0.0.1 --port "$GW_PORT" \
      --enable-igw --policy cache_aware --prometheus-host 127.0.0.1 --prometheus-port "$PROM_PORT" \
      --runtime-worker-threads "$GW_THREADS" --log-level "$GW_LOG_LEVEL" $GW_ARGS)
    echo "$(ts) bed: gateway: ${gw_cmd[*]}"
    "${gw_cmd[@]}" > "$BED/gateway.log" 2>&1 &
    gw_pid=$!
    echo "$gw_pid" > "$BED/gateway.pid"

    for _ in $(seq 1 120); do curl -sf "$GW/health" >/dev/null 2>&1 && break; sleep 0.5; done
    curl -sf "$GW/health" >/dev/null 2>&1 || { tail -20 "$BED/gateway.log" >&2; die "the gateway did not come up"; }
    # systemd-run --scope executes the command itself; confirm the pid is the gateway's.
    if ! grep -q -E '^(smg|amg)$' "/proc/$gw_pid/comm" 2>/dev/null; then
      child=$(pgrep -P "$gw_pid" | head -1)
      [ -n "$child" ] && { gw_pid=$child; echo "$gw_pid" > "$BED/gateway.pid"; }
    fi
    for i in $(seq 0 $((WORKERS - 1))); do
      p=$((MOCK_BASE + i))
      code=$(curl -s -o "$BED/register-$p.json" -w '%{http_code}' --max-time 30 -X POST "$GW/workers" -H 'content-type: application/json' \
        -d "{\"url\":\"grpc://127.0.0.1:$p\",\"connection_mode\":\"grpc\",\"runtime\":\"tokenspeed\",\"models\":[{\"id\":\"$MODEL\"}],\"labels\":{\"tokenizer_path\":\"$TOKENIZER\"},\"kv_block_size\":$BLOCK,\"health\":{\"disable_health_check\":true}}")
      case "$code" in 2*) ;; *) cat "$BED/register-$p.json" >&2; die "registering worker $p: HTTP $code" ;; esac
    done
    # A probe through the whole path (tokenizer loaded, a worker serving) before the bed is declared up.
    ok=0
    for _ in $(seq 1 120); do
      code=$(curl -s -o "$BED/probe.json" -w '%{http_code}' --max-time 30 "$GW/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"probe\"}],\"max_tokens\":2}")
      [ "$code" = "200" ] && { ok=1; break; }
      sleep 1
    done
    [ "$ok" = 1 ] || { cat "$BED/probe.json" >&2; tail -20 "$BED/gateway.log" >&2; die "the probe request did not succeed"; }
    cgroup=$(cut -d: -f3 "/proc/$gw_pid/cgroup" 2>/dev/null | head -1)
    cat > "$BED/bed.json" <<JSON
{"gateway_pid": $gw_pid, "mock_pid": $(cat "$BED/mock.pid"), "gateway_url": "$GW", "metrics_url": "http://127.0.0.1:$PROM_PORT/metrics",
 "admin_url": "http://127.0.0.1:$ADMIN_PORT", "workers": $WORKERS, "model": "$MODEL", "cgroup": "${cgroup:+/sys/fs/cgroup$cgroup}",
 "gw_cores": "$GW_CORES", "mock_cores": "$MOCK_CORES", "smg_bin": "$SMG_BIN", "engine_args": "$ENGINE_ARGS", "gw_args": "$GW_ARGS", "started": "$(ts)"}
JSON
    echo "$(ts) bed: up (gateway pid $gw_pid, $WORKERS workers on $MOCK_BASE.., cgroup ${cgroup:-none})"
    ;;
  *) echo "usage: bed.sh up|down|status" >&2; exit 2 ;;
esac
