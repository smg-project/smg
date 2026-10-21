#!/usr/bin/env bash
# Stops the fork-mode children of a fuzz run that stopped working: in libFuzzer's fork mode a
# child can wedge after its crash handler ran (asleep on a futex, no CPU time accruing) while the
# parent waits for it, and the target's time budget passes with no fuzzing. Every INTERVAL seconds
# (60), for every fork child of the run (a descendant of ROOT running a binary under BINARIES
# without the parent's -fork= flag), the child's CPU ticks are compared with the sample two
# intervals back; a child older than AGE seconds (120) whose ticks grew by fewer than five is
# stopped with SIGKILL, and the stop is logged to LOG with the child's command line, so the
# parent spawns the next job and the run keeps its hour. SIGKILL and not SIGTERM: libFuzzer
# handles TERM itself and exits with its interrupt code, on which the fork parent stops the whole
# target; a killed child counts as an ordinary crash, which -ignore_crashes skips. Parents are
# never touched, and a child that is merely slow under load keeps accruing ticks and is left
# alone. run.sh starts it with its fork-mode runs and stops it with them; the loop also ends by
# itself once ROOT is gone, so an interrupted run leaves no watchdog behind.
#
#   stall-watchdog.sh ROOT BINARIES LOG [INTERVAL [AGE]]
set -u
root=$1
binaries=$2
log=$3
interval=${4:-60}
age_limit=${5:-120}

descendants() {
  local child
  for child in $(pgrep -P "$1"); do
    echo "$child"
    descendants "$child"
  done
}

declare -A before last
while kill -0 "$root" 2>/dev/null && sleep "$interval"; do
  for pid in $(descendants "$root"); do
    cmd=$(tr '\0' ' ' 2>/dev/null < "/proc/$pid/cmdline") || continue
    case $cmd in "$binaries/"*) ;; *) continue ;; esac
    case " $cmd" in *" -fork="*) continue ;; esac
    ticks=$(awk '{print $14 + $15}' 2>/dev/null "/proc/$pid/stat") || continue
    age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "${before[$pid]:-}" ] && [ "${age:-0}" -gt "$age_limit" ] && [ $((ticks - before[$pid])) -lt 5 ]; then
      echo "$(date -u +%FT%TZ) stalled child $pid (${age}s old, $ticks ticks of CPU): ${cmd:0:160}" >> "$log"
      kill -KILL "$pid" 2>/dev/null
      unset "before[$pid]" "last[$pid]"
      continue
    fi
    before[$pid]=${last[$pid]:-$ticks}
    last[$pid]=$ticks
  done
done
