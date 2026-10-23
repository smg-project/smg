#!/usr/bin/env bash
# One measurement cell against a bed that is up: the sampler on the gateway,
# the load client pinned away from it, optional perf record and heap profile
# windows inside the sampled window, the metrics before and after.
#
#   cell.sh <cell-name> -- <stress_load arguments...>
#
# Environment: GS_DIR (the bed's), LOAD_BIN [target/release/stress_load], GEN_CORES (cpu list for the
# client) [none], PERF_SECS (a `perf record -g` of the gateway for that many seconds, started 10 s into the
# sampled window; 0 = off) [0], PERF_FREQ [499], HEAP (1: POST /heap_profile at the start and the end of the
# sampled window; needs a gateway built with jemalloc-profiling, started with --jemalloc-prof-dir and
# _RJEM_MALLOC_CONF=prof:true,prof_active:true,prof_accum:true,lg_prof_sample:N) [0], SAMPLE_INTERVAL [1],
# GS_LIVE_DIR (memory-backed directory for the files written during the cell) [/dev/shm].
set -u
name=${1:-}; [ -n "$name" ] || { echo "usage: cell.sh <name> -- <stress_load args>" >&2; exit 2; }
shift; [ "${1:-}" = "--" ] && shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
GS_DIR=${GS_DIR:-$ROOT/target/grpc-stress}
BED=$GS_DIR/bed
LOAD_BIN=${LOAD_BIN:-$ROOT/target/release/stress_load}
GEN_CORES=${GEN_CORES:-}; PERF_SECS=${PERF_SECS:-0}; PERF_FREQ=${PERF_FREQ:-499}; HEAP=${HEAP:-0}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-1}
ts() { date -u +%FT%TZ; }
die() { echo "$(ts) cell $name: $*" >&2; exit 1; }
[ -r "$BED/bed.json" ] || die "no bed under $GS_DIR (bed.sh up first)"
[ -x "$LOAD_BIN" ] || die "no stress_load binary at $LOAD_BIN"
field() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$BED/bed.json" "$1"; }
GW_PID=$(field gateway_pid); GW=$(field gateway_url); METRICS=$(field metrics_url); CG=$(field cgroup)
kill -0 "$GW_PID" 2>/dev/null || die "the gateway (pid $GW_PID) is not running"
CELL=$GS_DIR/cells/$name
mkdir -p "$CELL"
# Files written while the cell runs (the sampler's rows, the client's interim lines, perf's buffer) go to a
# memory-backed directory and are copied into the cell at the end: a blocking write on a busy disk would stall
# the writer for seconds, and a stalled client turns a steady rate into bursts.
LIVE=$(mktemp -d "${GS_LIVE_DIR:-/dev/shm}/grpc-stress-cell.XXXXXX" 2>/dev/null || mktemp -d)
# Ten thousand open streams are ten thousand client sockets.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
# The sampled window from the client's own flags (defaults as stress_load's).
warmup=5; duration=60
argv=("$@")
for ((i = 0; i < ${#argv[@]}; i++)); do
  case "${argv[$i]}" in --warmup) warmup=${argv[$((i + 1))]} ;; --duration) duration=${argv[$((i + 1))]} ;; esac
done
echo "$(ts) cell $name: start (gateway pid $GW_PID, window ${warmup}s+${duration}s)"
curl -s "$METRICS" > "$CELL/before.prom"
python3 "$HERE/sample.py" --pid "$GW_PID" --out "$LIVE/samples.csv" --cgroup "$CG" --metrics "$METRICS" --interval "$SAMPLE_INTERVAL" &
sampler=$!
echo $sampler > "$CELL/sampler.pid"
# Profile windows run on their own clock, inside the sampled window.
(
  if [ "$HEAP" = 1 ]; then
    sleep "$warmup"; curl -s -X POST "$GW/heap_profile" > "$CELL/heap-start.json"
    sleep "$duration"; curl -s -X POST "$GW/heap_profile" > "$CELL/heap-end.json"
  fi
) &
heap_job=$!
(
  if [ "$PERF_SECS" != 0 ]; then
    sleep $((warmup + 10))
    perf record -F "$PERF_FREQ" -g -p "$GW_PID" -o "$LIVE/perf.data" -- sleep "$PERF_SECS" > "$LIVE/perf-record.log" 2>&1
  fi
) &
perf_job=$!
cp /proc/stat "$CELL/stat-start"; cp "$BED/bed.json" "$CELL/stat-start.cores"
t0=$(date +%s.%N)
# The client runs in a transient systemd user scope of its own when that works (as the bed's processes do), so the
# memory controller of the shell's cgroup cannot stall its pacing.
scope=()
if [ "${SCOPE:-auto}" != 0 ] && command -v systemd-run >/dev/null 2>&1 \
   && systemd-run --user --scope --quiet -p MemoryAccounting=yes true 2>/dev/null; then
  scope=(systemd-run --user --scope --quiet --unit "grpc-stress-load-$$-$(date +%s)" -p MemoryAccounting=yes)
fi
# shellcheck disable=SC2206
gen_cmd=("${scope[@]}" ${GEN_CORES:+taskset -c $GEN_CORES} "$LOAD_BIN" --gateway "$GW" "$@" --out "$CELL/summary.json")
echo "$(ts) cell $name: ${gen_cmd[*]}" > "$CELL/gen.cmd"
"${gen_cmd[@]}" > "$LIVE/gen.out" 2> "$LIVE/gen.err"
gen_rc=$?
t1=$(date +%s.%N)
wait $heap_job $perf_job 2>/dev/null
kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null; rm -f "$CELL/sampler.pid"
curl -s "$METRICS" > "$CELL/after.prom"
for f in samples.csv gen.out gen.err perf.data perf-record.log; do [ -f "$LIVE/$f" ] && mv "$LIVE/$f" "$CELL/$f"; done
rmdir "$LIVE" 2>/dev/null
# How busy the gateway's cores and the whole host were over the window: the sampler's rows carry the gateway's
# own CPU; this is the rest of the host next to it (other tenants of the machine), from /proc/stat deltas.
busy=$(python3 - "$CELL/stat-start" <<'PY' 2>/dev/null
import json, sys
def snap(path):
    d = {}
    for l in open(path):
        if l.startswith("cpu") and l[3:4].isdigit():
            p = l.split(); v = list(map(int, p[1:9])); d[int(p[0][3:])] = (sum(v), v[3] + v[4])
    return d
try:
    a = snap(sys.argv[1])
except OSError:
    print("{}"); sys.exit(0)
b = snap("/proc/stat")
busy = {n: 100 * (1 - (b[n][1] - a[n][1]) / max(1, b[n][0] - a[n][0])) for n in a if n in b}
cores = json.load(open(sys.argv[1] + ".cores")).get("gw_cores", "")
sel = set()
for part in cores.split(","):
    if "-" in part:
        lo, hi = part.split("-"); sel.update(range(int(lo), int(hi) + 1))
    elif part.strip():
        sel.add(int(part))
gw = [busy[n] for n in sorted(sel) if n in busy]
print(json.dumps({"host_busy_pct": round(sum(busy.values()) / len(busy), 1), "gw_cores_busy_pct": round(sum(gw) / len(gw), 1) if gw else None}))
PY
)
rm -f "$CELL/stat-start" "$CELL/stat-start.cores"
cat > "$CELL/cell.json" <<JSON
{"name": "$name", "t0": $t0, "t1": $t1, "warmup": $warmup, "duration": $duration, "gen_rc": $gen_rc,
 "gateway_pid": $GW_PID, "perf_secs": $PERF_SECS, "heap": $HEAP, "loadavg_end": "$(cut -d' ' -f1-3 /proc/loadavg)",
 "busy": ${busy:-{\}}, "bed": $(cat "$BED/bed.json")}
JSON
python3 - "$CELL/summary.json" <<'PY' 2>/dev/null
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  done: sampled={s['sampled']} ok={s['ok']} failed={s['failed']} rps={s['achieved_rps']:.0f} ttfb p50/p99={s['ttfb_ms']['p50']:.1f}/{s['ttfb_ms']['p99']:.1f} ms ttft p50/p99={s['ttft_ms']['p50']:.1f}/{s['ttft_ms']['p99']:.1f} ms e2e p99={s['e2e_ms']['p99']:.0f} ms frames/req={s['frames_per_request']:.1f} statuses={s['statuses']} errors={s['errors']}")
PY
echo "$(ts) cell $name: end (rc $gen_rc)"
exit $gen_rc
