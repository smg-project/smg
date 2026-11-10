#!/usr/bin/env bash
# The CPU-only stress bed for the gateway's request path: one release gateway
# in front of a fleet of `mock-worker --engine realistic` gRPC engines (KV
# events on), in front of a prefill fleet and a decode fleet on the gateway's
# prefill-decode path, or in front of the same fleet on the mock's HTTP wire
# behind the gateway's HTTP regular path. No GPU, no engine: the whole bed is
# this repository.
#
#   bed.sh up      start the fleet(s) and the gateway, register the workers, wait until a probe answers
#   bed.sh down    stop them (pids from the bed's own pid files only)
#   bed.sh status  print bed.json (pids, ports, the gateway's cgroup)
#
# Environment (defaults in brackets):
#   GS_DIR       run directory for logs, pids and cells [target/grpc-stress]
#   SMG_BIN      gateway binary [target/release/smg]      MOCK_BIN  [target/release/mock-worker]
#   GW_PORT      [29600]  PROM_PORT [29601]  ADMIN_PORT [29609]  MOCK_BASE [29610]  WORKERS [12], one or more
#   FLEET        grpc | prefill-decode | http: one fleet of regular gRPC engines; a prefill fleet and a decode fleet
#                registered with their roles, so the gateway serves the model on its prefill-decode path; or one
#                fleet of regular HTTP engines, so the gateway serves it on its HTTP regular path [grpc]
#   MODEL        served model id [mock-model]   TOKENIZER  tokenizer directory or hub id (required)
#   BLOCK        KV block size [16]
#   GW_CORES     cpu list the gateway is pinned to (taskset) [none]   GW_THREADS  tokio worker threads [8]
#   MOCK_CORES   cpu list the fleet is pinned to [none]
#   ENGINE_ARGS  mock engine flags [a realistic engine with fast prefill and 10 ms decode steps]
#   GW_POLICY    the gateway's routing policy [cache_aware]
#   GW_ARGS      extra gateway flags (e.g. "--parser symphony --trust-tenant-header")
#   GW_ENV       extra KEY=VALUE pairs for the gateway's environment (e.g. _RJEM_MALLOC_CONF=...)
#   GW_LOG_LEVEL [warn]
#   SCOPE        auto|1|0: run the gateway and the fleet each in a transient systemd user scope of their own
#                (memory accounting on: the gateway's cgroup memory can be sampled, and neither is throttled by
#                the memory controller of the shell's own cgroup) [auto: when systemd-run works]
# The prefill-decode fleet (FLEET=prefill-decode; WORKERS counts the decode engines, on MOCK_BASE.. with ADMIN_PORT):
#   PREFILL_WORKERS      prefill engines, one or more [1]
#   PREFILL_BASE         first port of the prefill engines [MOCK_BASE + WORKERS]
#   BOOTSTRAP_BASE       first bootstrap port of the prefill engines; engine i binds BOOTSTRAP_BASE + i and the gateway
#                        probes it at registration [PREFILL_BASE + PREFILL_WORKERS]
#   PREFILL_ADMIN_PORT   the prefill fleet's admin API [ADMIN_PORT - 1]
#   PREFILL_ENGINE_ARGS  the prefill engines' flags [ENGINE_ARGS with a prompt-side answer of 1 ms: a realistic engine,
#                        so the fleet reports itself and its loads, whose prompt-side admission serves no tokens]
#   DECODE_WINDOW        running requests a decode engine admits: its --max-running and the max_running_requests
#                        label the gateway's decode rooms read [8]
#   PREFILL_WINDOW       the gateway's in-flight bound per prefill engine (--prefill-max-inflight-requests-per-worker,
#                        the prefill admission queue); 0 leaves the bound off [8]
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
FLEET=${FLEET:-grpc}
PREFILL_WORKERS=${PREFILL_WORKERS:-1}
PREFILL_BASE=${PREFILL_BASE:-$((MOCK_BASE + WORKERS))}
BOOTSTRAP_BASE=${BOOTSTRAP_BASE:-$((PREFILL_BASE + PREFILL_WORKERS))}
PREFILL_ADMIN_PORT=${PREFILL_ADMIN_PORT:-$((ADMIN_PORT - 1))}
DECODE_WINDOW=${DECODE_WINDOW:-8}; PREFILL_WINDOW=${PREFILL_WINDOW:-8}
MODEL=${MODEL:-mock-model}; BLOCK=${BLOCK:-16}
GW_CORES=${GW_CORES:-}; MOCK_CORES=${MOCK_CORES:-}; GW_THREADS=${GW_THREADS:-8}
ENGINE_ARGS=${ENGINE_ARGS:---engine realistic --timing linear --prefill-tps 2000000 --decode-base-ms 10 --decode-per-req-ms 0 --max-running 4096 --max-batched-tokens 1000000 --context-length 262144 --kv-tokens 2097152}
# The admin API's fleet count and the load reports come from the realistic engine, so the prefill engines run it too.
PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-$ENGINE_ARGS --gen-ms 1}
GW_POLICY=${GW_POLICY:-cache_aware}
GW_ARGS=${GW_ARGS:-}; GW_ENV=${GW_ENV:-}; GW_LOG_LEVEL=${GW_LOG_LEVEL:-warn}
SCOPE=${SCOPE:-auto}
GW=http://127.0.0.1:$GW_PORT
ts() { date -u +%FT%TZ; }
die() { echo "$(ts) bed: $*" >&2; exit 1; }
case "$FLEET" in
  grpc|http|prefill-decode) ;;
  *) die "FLEET=$FLEET: grpc, prefill-decode or http" ;;
esac
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

# Everything the bed started, by its pid files: the gateway, the decode (or regular) fleet, the prefill fleet.
stop_all() {
  stop_pid_file "$BED/gateway.pid"
  stop_pid_file "$BED/mock.pid"
  stop_pid_file "$BED/prefill.pid"
}

# A failed start leaves no process and no port behind.
bail() { stop_all; die "$@"; }

# start_fleet <name> <wire> <first port> <count> <admin port> <engine flags...>: one mock-worker process of <count>
# engines of the <wire> (grpc or http) on <first port>.. with its admin API on <admin port>, in a scope of its own,
# pinned to MOCK_CORES; its log and pid file are <name>.log and <name>.pid under the bed. Returns 1 when a worker
# could not bind its port (a worker whose port another process holds exits quietly, and the gateway would route a
# share of the traffic into that stranger) or the fleet did not come up whole.
start_fleet() {
  local name=$1 wire=$2 base=$3 count=$4 admin=$5; shift 5
  local prefix pid child fleet scope
  prefix=$(scope_prefix "grpc-stress-$name-$base-$stamp") || return 1
  read -r -a scope <<< "$prefix"
  # shellcheck disable=SC2206
  local mock_cmd=("${scope[@]}" ${MOCK_CORES:+taskset -c $MOCK_CORES} "$MOCK_BIN" "--$wire-base-port" "$base" "--$wire-count" "$count" \
    --model "$MODEL" --tokenizer "$TOKENIZER" --block-size "$BLOCK" --admin-port "$admin" "$@")
  echo "$(ts) bed: $name fleet: ${mock_cmd[*]}"
  "${mock_cmd[@]}" > "$BED/$name.log" 2>&1 &
  pid=$!
  sleep 1
  if ! grep -q -E '^mock-worker$' "/proc/$pid/comm" 2>/dev/null; then
    child=$(pgrep -P "$pid" | head -1); [ -n "$child" ] && pid=$child
  fi
  echo "$pid" > "$BED/$name.pid"
  for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$admin/admin/health" >/dev/null 2>&1 && break; sleep 0.5; done
  if grep -q -E 'bind .* failed|Address already in use' "$BED/$name.log"; then
    grep -E 'bind .* failed|Address already in use' "$BED/$name.log" >&2
    echo "$(ts) bed: a $name worker could not bind its port" >&2
    return 1
  fi
  fleet=$(curl -s "http://127.0.0.1:$admin/admin/fleet" | python3 -c 'import json,sys; d=json.load(sys.stdin); w=d["workers"] if isinstance(d,dict) else d; print(len(w))' 2>/dev/null)
  [ -n "$fleet" ] && [ "$fleet" = "$count" ] || { echo "$(ts) bed: the $name fleet has ${fleet:-no readable list of} workers, not $count (see $BED/$name.log)" >&2; return 1; }
}

# register <wire> <port> <extra labels> <extra fields>: one worker of the model on the wire (grpc or http) at the
# gateway's worker API; the answer is kept as register-<port>.json. The extra labels and fields are JSON fragments,
# each starting with a comma. A gRPC engine names the scheduler wire it speaks; an HTTP engine's runtime is read at
# registration.
register() {
  local wire=$1 p=$2 labels=$3 extra=$4 code runtime=""
  [ "$wire" = grpc ] && runtime=",\"runtime\":\"tokenspeed\""
  code=$(curl -s -o "$BED/register-$p.json" -w '%{http_code}' --max-time 30 -X POST "$GW/workers" -H 'content-type: application/json' \
    -d "{\"url\":\"$wire://127.0.0.1:$p\",\"connection_mode\":\"$wire\"$runtime,\"models\":[{\"id\":\"$MODEL\"}],\"labels\":{\"tokenizer_path\":\"$TOKENIZER\"$labels},\"kv_block_size\":$BLOCK,\"health\":{\"disable_health_check\":true}$extra}")
  case "$code" in 2*) ;; *) cat "$BED/register-$p.json" >&2; bail "registering worker $p: HTTP $code" ;; esac
}

case "$cmd" in
  down)
    stop_all
    echo "$(ts) bed: down"
    ;;
  status)
    cat "$BED/bed.json" 2>/dev/null || die "no bed.json under $BED"
    ;;
  up)
    [ "$WORKERS" -ge 1 ] 2>/dev/null || die "WORKERS=$WORKERS: one engine or more"
    [ "$FLEET" != prefill-decode ] || [ "$PREFILL_WORKERS" -ge 1 ] 2>/dev/null || die "PREFILL_WORKERS=$PREFILL_WORKERS: one prefill engine or more"
    : "${TOKENIZER:?set TOKENIZER to a tokenizer directory or hub id}"
    [ -x "$SMG_BIN" ] || die "no gateway binary at $SMG_BIN"
    [ -x "$MOCK_BIN" ] || die "no mock-worker binary at $MOCK_BIN"
    mkdir -p "$BED"
    [ -f "$BED/gateway.pid" ] && die "a bed is up already (bed.sh down first)"
    ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
    # Every port the bed binds must be free (see start_fleet).
    ports=("$GW_PORT" "$PROM_PORT" "$ADMIN_PORT")
    for i in $(seq 0 $((WORKERS - 1))); do ports+=("$((MOCK_BASE + i))"); done
    if [ "$FLEET" = prefill-decode ]; then
      ports+=("$PREFILL_ADMIN_PORT")
      for i in $(seq 0 $((PREFILL_WORKERS - 1))); do ports+=("$((PREFILL_BASE + i))" "$((BOOTSTRAP_BASE + i))"); done
    fi
    for p in "${ports[@]}"; do
      if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then exec 3>&- 3<&-; die "port $p is in use by another process (set GW_PORT/PROM_PORT/ADMIN_PORT/MOCK_BASE to a free range)"; fi
    done

    stamp=$(date +%s)
    # The prefix is taken outside the read's substitution, so SCOPE=1 without a usable systemd-run stops here.
    gw_prefix=$(scope_prefix "grpc-stress-gw-$GW_PORT-$stamp") || exit 1
    read -r -a gw_scope <<< "$gw_prefix"
    gw_flags=()
    wire=grpc; [ "$FLEET" = http ] && wire=http
    if [ "$FLEET" = prefill-decode ]; then
      # shellcheck disable=SC2086
      start_fleet prefill grpc "$PREFILL_BASE" "$PREFILL_WORKERS" "$PREFILL_ADMIN_PORT" --worker-type prefill --bootstrap-base-port "$BOOTSTRAP_BASE" $PREFILL_ENGINE_ARGS \
        || bail "the prefill fleet did not come up"
      # The decode engines run the regular engine flags with the window the gateway's rooms are told of.
      # shellcheck disable=SC2086
      start_fleet mock grpc "$MOCK_BASE" "$WORKERS" "$ADMIN_PORT" --worker-type decode $ENGINE_ARGS --max-running "$DECODE_WINDOW" \
        || bail "the decode fleet did not come up"
      # The multi-router gateway serves the roles on its prefill-decode router by itself; the mode flag is what
      # admits the prefill in-flight bound (and the per-role policies), with no static worker list.
      gw_flags+=(--pd-disaggregation)
      [ "$PREFILL_WINDOW" -gt 0 ] && gw_flags+=(--prefill-max-inflight-requests-per-worker "$PREFILL_WINDOW")
    else
      # shellcheck disable=SC2086
      start_fleet mock "$wire" "$MOCK_BASE" "$WORKERS" "$ADMIN_PORT" $ENGINE_ARGS || bail "the fleet did not come up"
    fi

    # shellcheck disable=SC2206
    gw_cmd=("${gw_scope[@]}" ${GW_CORES:+taskset -c $GW_CORES} env $GW_ENV "$SMG_BIN" --host 127.0.0.1 --port "$GW_PORT" \
      --enable-igw --policy "$GW_POLICY" --prometheus-host 127.0.0.1 --prometheus-port "$PROM_PORT" \
      --runtime-worker-threads "$GW_THREADS" --log-level "$GW_LOG_LEVEL" "${gw_flags[@]}" $GW_ARGS)
    echo "$(ts) bed: gateway: ${gw_cmd[*]}"
    "${gw_cmd[@]}" > "$BED/gateway.log" 2>&1 &
    gw_pid=$!
    echo "$gw_pid" > "$BED/gateway.pid"

    for _ in $(seq 1 120); do curl -sf "$GW/health" >/dev/null 2>&1 && break; sleep 0.5; done
    curl -sf "$GW/health" >/dev/null 2>&1 || { tail -20 "$BED/gateway.log" >&2; bail "the gateway did not come up"; }
    # systemd-run --scope executes the command itself; confirm the pid is the gateway's.
    if ! grep -q -E '^(smg|amg)$' "/proc/$gw_pid/comm" 2>/dev/null; then
      child=$(pgrep -P "$gw_pid" | head -1)
      [ -n "$child" ] && { gw_pid=$child; echo "$gw_pid" > "$BED/gateway.pid"; }
    fi
    if [ "$FLEET" = prefill-decode ]; then
      # The roles the gateway pairs on: a prefill worker with the bootstrap port its engine bound (the gateway
      # probes it), a decode worker with the running window its rooms are claimed against.
      for i in $(seq 0 $((PREFILL_WORKERS - 1))); do
        register grpc $((PREFILL_BASE + i)) "" ",\"worker_type\":\"prefill\",\"bootstrap_port\":$((BOOTSTRAP_BASE + i))"
      done
      for i in $(seq 0 $((WORKERS - 1))); do
        register grpc $((MOCK_BASE + i)) ",\"max_running_requests\":\"$DECODE_WINDOW\"" ",\"worker_type\":\"decode\""
      done
    else
      for i in $(seq 0 $((WORKERS - 1))); do
        register "$wire" $((MOCK_BASE + i)) "" ""
      done
    fi
    # A probe through the whole path (tokenizer loaded, a worker serving) before the bed is declared up.
    ok=0
    for _ in $(seq 1 120); do
      code=$(curl -s -o "$BED/probe.json" -w '%{http_code}' --max-time 30 "$GW/v1/chat/completions" -H 'content-type: application/json' \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"probe\"}],\"max_tokens\":2}")
      [ "$code" = "200" ] && { ok=1; break; }
      sleep 1
    done
    [ "$ok" = 1 ] || { cat "$BED/probe.json" >&2; tail -20 "$BED/gateway.log" >&2; bail "the probe request did not succeed"; }
    cgroup=$(cut -d: -f3 "/proc/$gw_pid/cgroup" 2>/dev/null | head -1)
    if [ "$FLEET" = prefill-decode ]; then
      prefill_json="\"prefill_pid\": $(cat "$BED/prefill.pid"), \"prefill_workers\": $PREFILL_WORKERS, \"prefill_admin_url\": \"http://127.0.0.1:$PREFILL_ADMIN_PORT\","
      shape="$PREFILL_WORKERS prefill on $PREFILL_BASE.. and $WORKERS decode on $MOCK_BASE.."
    else
      prefill_json=""
      shape="$WORKERS workers on $MOCK_BASE.."
    fi
    cat > "$BED/bed.json" <<JSON
{"gateway_pid": $gw_pid, "mock_pid": $(cat "$BED/mock.pid"), "gateway_url": "$GW", "metrics_url": "http://127.0.0.1:$PROM_PORT/metrics",
 "admin_url": "http://127.0.0.1:$ADMIN_PORT", "workers": $WORKERS, "fleet": "$FLEET", $prefill_json "model": "$MODEL", "cgroup": "${cgroup:+/sys/fs/cgroup$cgroup}",
 "gw_cores": "$GW_CORES", "mock_cores": "$MOCK_CORES", "smg_bin": "$SMG_BIN", "engine_args": "$ENGINE_ARGS", "policy": "$GW_POLICY", "gw_args": "$GW_ARGS", "started": "$(ts)"}
JSON
    echo "$(ts) bed: up (gateway pid $gw_pid, $shape, cgroup ${cgroup:-none})"
    ;;
  *) echo "usage: bed.sh up|down|status" >&2; exit 2 ;;
esac
