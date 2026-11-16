#!/bin/bash
# Local mesh profiling and benchmarking
#
# Spins up mock workers + SMG gateway replicas with mesh enabled.
# Supports interactive profiling and automated benchmark suites.
#
# Prerequisites:
#   cargo build --release -p smg
#   pip install aiohttp  (for load generator)
#
# Usage:
#   ./scripts/mesh-profile-local.sh start [workers]       # Start workers + gateways
#   ./scripts/mesh-profile-local.sh load [rps] [dur] [ps] # Run load (default: 200/60/0)
#   ./scripts/mesh-profile-local.sh metrics                # Show mesh sync metrics
#   ./scripts/mesh-profile-local.sh status                 # Show health
#   ./scripts/mesh-profile-local.sh stop                   # Stop everything
#   ./scripts/mesh-profile-local.sh bench                  # Run full benchmark suite

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${REPO_ROOT}/target/mesh-profile"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "${REPO_ROOT}/target")}"
SMG_BIN="${CARGO_TARGET_DIR}/release/smg"
NUM_WORKERS="${NUM_WORKERS:-20}"
NUM_GATEWAYS="${NUM_GATEWAYS:-2}"
WORKER_BASE_PORT=9000
GATEWAY_BASE_PORT=30000
MESH_BASE_PORT=39500
METRICS_BASE_PORT=29000

mkdir -p "$LOG_DIR"

# ── Core functions ──────────────────────────────────────────────────

start_mock_workers() {
    local count="${1:-$NUM_WORKERS}"
    : > "$LOG_DIR/worker_pids.txt"
    for i in $(seq 0 $((count - 1))); do
        python3 "$SCRIPT_DIR/mock_worker.py" $((WORKER_BASE_PORT + i)) &
        echo $! >> "$LOG_DIR/worker_pids.txt"
    done
    echo "Started $count mock workers (ports $WORKER_BASE_PORT-$((WORKER_BASE_PORT + count - 1)))"
}

start_gateways() {
    if [ ! -f "$SMG_BIN" ]; then
        echo "ERROR: $SMG_BIN not found. Run: cargo build --release -p smg"
        exit 1
    fi
    : > "$LOG_DIR/gateway_pids.txt"
    local log_level="${1:-warn}"
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        local port=$((GATEWAY_BASE_PORT + i))
        local mesh_port=$((MESH_BASE_PORT + i))
        local metrics_port=$((METRICS_BASE_PORT + i))
        local MESH_PEERS=""
        for j in $(seq 0 $((NUM_GATEWAYS - 1))); do
            [ "$i" != "$j" ] && MESH_PEERS="$MESH_PEERS 127.0.0.1:$((MESH_BASE_PORT + j))"
        done
        # Build worker URLs (same as production --worker-urls)
        local WORKER_URLS=""
        for w in $(seq 0 $((NUM_WORKERS - 1))); do
            WORKER_URLS="$WORKER_URLS http://127.0.0.1:$((WORKER_BASE_PORT + w))"
        done
        "$SMG_BIN" --host 127.0.0.1 --port "$port" --policy cache_aware \
            --worker-urls $WORKER_URLS \
            --enable-mesh --mesh-host 127.0.0.1 --mesh-port "$mesh_port" \
            --mesh-server-name "gw-$i" --mesh-peer-urls $MESH_PEERS \
            --prometheus-port "$metrics_port" --prometheus-host 127.0.0.1 \
            --log-level "$log_level" > "$LOG_DIR/gateway-$i.log" 2>&1 &
        echo $! >> "$LOG_DIR/gateway_pids.txt"
    done
    echo "Started $NUM_GATEWAYS gateways (ports $GATEWAY_BASE_PORT-$((GATEWAY_BASE_PORT + NUM_GATEWAYS - 1)))"
}

register_workers() {
    local count="${1:-$NUM_WORKERS}"
    local registered=0
    for i in $(seq 0 $((count - 1))); do
        curl -sf -X POST "http://127.0.0.1:$GATEWAY_BASE_PORT/workers" \
            -H "Content-Type: application/json" \
            -d "{\"url\":\"http://127.0.0.1:$((WORKER_BASE_PORT + i))\"}" >/dev/null 2>&1 \
            && registered=$((registered + 1))
    done
    echo "Registered $registered/$count workers (waiting 20s for activation)"
    sleep 20
}

run_load() {
    local rps="${1:-200}" duration="${2:-60}" prompt_size="${3:-0}"
    local ports="$GATEWAY_BASE_PORT"
    for i in $(seq 1 $((NUM_GATEWAYS - 1))); do ports="$ports,$((GATEWAY_BASE_PORT + i))"; done
    python3 "$SCRIPT_DIR/mesh_load_gen.py" \
        --rps "$rps" --duration "$duration" \
        --gateway-ports "$ports" --prompt-size "$prompt_size"
}

stop_all() {
    for pidfile in "$LOG_DIR/worker_pids.txt" "$LOG_DIR/gateway_pids.txt"; do
        if [ -f "$pidfile" ]; then
            while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$pidfile"
            rm "$pidfile"
        fi
    done
    pkill -f mock_worker.py 2>/dev/null || true
    pkill -f 'smg.*mesh' 2>/dev/null || true
    echo "Stopped."
}

# ── Metrics collection ──────────────────────────────────────────────

# The mesh series (crates/mesh/src/metrics.rs) of every gateway, one line per
# gateway and series. The sender round histogram is written by the end that
# dials a sync stream, the peer gauge and the ack and nack counters by the end
# that accepts it, so one gateway's page shows one side of each pair. A
# gateway that is down or has written no mesh series yet contributes no line
# and does not end the loop (the script runs under `set -e`).
mesh_metrics_lines() {
    local i
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        curl -sf "http://127.0.0.1:$((METRICS_BASE_PORT + i))/metrics" 2>/dev/null \
            | grep -E '^router_mesh_(peer_(connections|ack_total|nack_total|reconnects_total)|sync_round_duration_seconds_(sum|count))\{' \
            | sed "s/^/gw-$i /" || true
    done
}

# Summed over every gateway and peer: open sync streams, successful acks
# (the ack counter carries a status label; a failed ack is not one), nacks,
# reconnects, and the mean sender round.
collect_metrics() {
    local lines
    lines=$(mesh_metrics_lines)
    local PEERS ACKS NACKS RECONNECTS RS RC AR
    PEERS=$(echo "$lines" | awk '$2 ~ /^router_mesh_peer_connections\{/{s+=$3}END{print s+0}')
    ACKS=$(echo "$lines" | awk '$2 ~ /^router_mesh_peer_ack_total\{/ && $2 ~ /status="success"/{s+=$3}END{print s+0}')
    NACKS=$(echo "$lines" | awk '$2 ~ /^router_mesh_peer_nack_total\{/{s+=$3}END{print s+0}')
    RECONNECTS=$(echo "$lines" | awk '$2 ~ /^router_mesh_peer_reconnects_total\{/{s+=$3}END{print s+0}')
    RS=$(echo "$lines" | awk '$2 ~ /^router_mesh_sync_round_duration_seconds_sum\{/{s+=$3}END{print s+0}')
    RC=$(echo "$lines" | awk '$2 ~ /^router_mesh_sync_round_duration_seconds_count\{/{s+=$3}END{print s+0}')
    AR=$(echo "$RS $RC" | awk '{if($2>0)printf "%.2f",$1/$2*1000;else print "0"}')
    echo "${PEERS}|${ACKS}|${NACKS}|${RECONNECTS}|${AR}"
}

show_metrics() {
    echo "=== Mesh peer and sync metrics (every gateway) ==="
    mesh_metrics_lines | sed 's/^/  /'
}

# Monitor fd count, CPU, memory for gateway processes over time
monitor_resources() {
    local interval="${1:-5}"
    local duration="${2:-300}"
    local end=$((SECONDS + duration))
    echo "Monitoring gateway resources every ${interval}s for ${duration}s..."
    echo "Time | Gateway | PID | FDs | RSS_MB | CPU% | Threads"
    echo "-----|---------|-----|-----|--------|------|--------"
    while [ $SECONDS -lt $end ]; do
        if [ -f "$LOG_DIR/gateway_pids.txt" ]; then
            local idx=0
            while read -r pid; do
                if kill -0 "$pid" 2>/dev/null; then
                    local fds rss cpu threads
                    fds=$(ls /proc/"$pid"/fd 2>/dev/null | wc -l || lsof -p "$pid" 2>/dev/null | wc -l || echo "?")
                    rss=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
                    cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
                    threads=$(ls /proc/"$pid"/task 2>/dev/null | wc -l || ps -M -p "$pid" 2>/dev/null | tail -n +2 | wc -l || echo "?")
                    printf "%s | gw-%d | %d | %s | %s | %s | %s\n" \
                        "$(date +%H:%M:%S)" "$idx" "$pid" "$fds" "$rss" "$cpu" "$threads"
                fi
                idx=$((idx + 1))
            done < "$LOG_DIR/gateway_pids.txt"
        fi
        sleep "$interval"
    done
}

show_status() {
    for i in $(seq 0 $((NUM_GATEWAYS - 1))); do
        local port=$((GATEWAY_BASE_PORT + i))
        printf "  Gateway %d (port %d): " "$i" "$port"
        curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && echo "UP" || echo "DOWN"
    done
    local up=0
    for i in $(seq 0 $((NUM_WORKERS - 1))); do
        curl -sf "http://127.0.0.1:$((WORKER_BASE_PORT + i))/health" > /dev/null 2>&1 && up=$((up + 1))
    done
    echo "  Workers: $up/$NUM_WORKERS UP"
}

# ── Benchmark suite ─────────────────────────────────────────────────

run_one_scenario() {
    local workers=$1 rps=$2 duration=$3 prompt_size=${4:-0}
    stop_all >/dev/null 2>&1
    sleep 2

    NUM_WORKERS="$workers"
    start_mock_workers "$workers" >/dev/null
    sleep 1
    start_gateways "warn" >/dev/null
    echo "  Waiting 30s for worker creation workflows..." >&2
    sleep 30

    run_load "$rps" "$duration" "$prompt_size" 2>/dev/null | grep -E 'Done' >&2

    local result
    result=$(collect_metrics)
    local label="${workers}w / ${rps}rps / ${duration}s"
    [ "$prompt_size" -gt 0 ] && label="$label / ${prompt_size}ch"
    IFS='|' read -r peers acks nacks reconnects ar <<< "$result"
    printf "| %-28s | %5s | %7s | %7s | %10s | %9s ms |\n" "$label" "$peers" "$acks" "$nacks" "$reconnects" "$ar"
}

run_bench() {
    echo "# Mesh Benchmark — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "| Scenario                     | Peers | Acks    | Nacks   | Reconnects | Avg Round    |"
    echo "|------------------------------|-------|---------|---------|------------|--------------|"

    # Vary workers
    run_one_scenario 10 100 30
    run_one_scenario 20 100 30
    run_one_scenario 40 100 30

    # Vary RPS
    run_one_scenario 20 50  30
    run_one_scenario 20 200 30
    run_one_scenario 20 500 30

    # Vary prompt size
    run_one_scenario 20 200 30 500
    run_one_scenario 20 200 30 2000

    # Sustained
    run_one_scenario 20 200 120

    stop_all >/dev/null 2>&1
    echo ""
    echo "Done."
}

# ── Main ────────────────────────────────────────────────────────────

case "${1:-help}" in
    start)
        NUM_WORKERS="${2:-$NUM_WORKERS}"
        start_mock_workers "$NUM_WORKERS"
        sleep 1
        # Workers passed via --worker-urls (like production), not API registration
        start_gateways "info"
        echo "Waiting 30s for worker creation workflows..."
        sleep 30
        show_status
        echo ""
        echo "Next steps:"
        echo "  $0 load [rps] [duration] [prompt_size]"
        echo "  $0 monitor [interval_secs] [duration_secs]"
        ;;
    load)
        run_load "${2:-200}" "${3:-60}" "${4:-0}"
        ;;
    monitor)
        monitor_resources "${2:-5}" "${3:-300}"
        ;;
    metrics)
        show_metrics
        ;;
    status)
        show_status
        ;;
    stop)
        stop_all
        ;;
    bench)
        run_bench
        ;;
    *)
        echo "Usage: $0 {start|load|metrics|status|stop|bench|monitor}"
        echo ""
        echo "  start [workers]              Start mock workers + $NUM_GATEWAYS mesh gateways (default: 20 workers)"
        echo "  load [rps] [dur] [psize]     Send load (default: 200 req/s, 60s, 0-char pad)"
        echo "  monitor [interval] [dur]     Track fd count, CPU, memory (default: 5s, 300s)"
        echo "  metrics                      Show mesh sync Prometheus metrics"
        echo "  status                       Show gateway/worker health"
        echo "  stop                         Stop everything"
        echo "  bench                        Run full benchmark suite (9 scenarios)"
        ;;
esac
