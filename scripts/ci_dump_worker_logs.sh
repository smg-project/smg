#!/usr/bin/env bash
# Dump the last worker log snapshot and list all log files.
#
# Usage: ci_dump_worker_logs.sh <log_dir> [artifact_name]
#
# Arguments:
#   log_dir        Directory containing worker log files
#   artifact_name  (optional) Artifact name to display in the file listing

set -euo pipefail

LOG_DIR="${1:?Usage: ci_dump_worker_logs.sh <log_dir> [artifact_name]}"
ARTIFACT="${2:-$LOG_DIR}"

if [ ! -d "$LOG_DIR" ]; then exit 0; fi

SEP="============================================================"
DASH="------------------------------------------------------------"

# Dump last worker log (most recent = last retry of the failed model)
LAST_LOG=$(ls -t "$LOG_DIR"/worker-*.log 2>/dev/null | head -1 || true)
if [ -n "$LAST_LOG" ]; then
    TOTAL=$(wc -l < "$LAST_LOG")
    SHOW=$(( TOTAL < 200 ? TOTAL : 200 ))
    echo ""
    echo "$SEP"
    echo "Last worker log: $(basename "$LAST_LOG") (last $SHOW of $TOTAL lines)"
    echo "$DASH"
    tail -200 "$LAST_LOG"
    echo "$SEP"
fi

# A worker relaunched on its port keeps the previous engine's log as
# worker-<name>.log.<n>; an engine that died before the relaunch left its
# trace there, not in the live file, so show each one's tail.
shopt -s nullglob
ROTATED=("$LOG_DIR"/worker-*.log.[0-9]*)
shopt -u nullglob
for ROTATED_LOG in "${ROTATED[@]}"; do
    TOTAL=$(wc -l < "$ROTATED_LOG")
    SHOW=$(( TOTAL < 60 ? TOTAL : 60 ))
    echo ""
    echo "$SEP"
    echo "Previous engine's log: $(basename "$ROTATED_LOG") (last $SHOW of $TOTAL lines)"
    echo "$DASH"
    tail -60 "$ROTATED_LOG"
    echo "$SEP"
done

# List all log files with sizes, the rotated ones included
shopt -s nullglob
ALL_LOGS=("$LOG_DIR"/*.log "$LOG_DIR"/*.log.[0-9]*)
shopt -u nullglob
echo ""
echo "$SEP"
echo "All worker logs in artifact: $ARTIFACT"
echo "$DASH"
if [ "${#ALL_LOGS[@]}" -gt 0 ]; then ls -lhS "${ALL_LOGS[@]}"; else echo "  (none)"; fi
echo "$SEP"
