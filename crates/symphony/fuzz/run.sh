#!/usr/bin/env bash
# Runs the fuzz targets for a bounded time each, seeded from the recorded output slices of
# bellwether's hand-written sets (BELLWETHER_FIXTURES names the tree `bellwether unpack` writes)
# and guided by a dictionary of the table's markers, and prints one row per target: executions,
# executions per second, coverage edges, corpus units and crashes. Exits 1 when any target found
# a crash, or never ran; a crash's artifact is under artifacts/<target>/ and the log under
# logs/<target>.log.
#
#   run.sh [--minutes N] [--jobs J] [--sanitizer none|address] [--max-len BYTES] [--keep-going]
#          [--allow NAMES] [--label NAME] [--summary-only] [target ...]
#
# Defaults: every target, 10 minutes each, one at a time, no sanitizer (the crate forbids unsafe
# code; without it a target runs two to three times as many executions), inputs up to 8 KiB.
# A target stops at its first crash; with --keep-going it runs libFuzzer's fork mode, which keeps
# every crash's artifact and goes on, so that one campaign collects every class of breach. The
# summary counts a target's artifacts by the `breach <name>` the harness names when it replays
# them; a fork child that stops working (stall-watchdog.sh) is stopped so the parent spawns the
# next job. --allow NAMES (comma separated) makes the harness look past breaches already reported
# (SYMPHONY_FUZZ_ALLOW); --label NAME keeps a run's logs and artifacts under logs/NAME and
# artifacts/NAME, apart from other runs'; --summary-only prints the table of a run already made.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
minutes=10
jobs=1
sanitizer=none
max_len=8192
keep_going=()
label=run
summary_only=
targets=()
while [ $# -gt 0 ]; do
  case $1 in
    --minutes) minutes=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --sanitizer) sanitizer=$2; shift 2 ;;
    --max-len) max_len=$2; shift 2 ;;
    --keep-going) keep_going=(-fork=1 -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1); shift ;;
    --allow) export SYMPHONY_FUZZ_ALLOW=$2; shift 2 ;;
    --label) label=$2; shift 2 ;;
    --summary-only) summary_only=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "run.sh: unknown option $1" >&2; exit 2 ;;
    *) targets+=("$1"); shift ;;
  esac
done

cd "$here"
if [ ${#targets[@]} -eq 0 ]; then
  mapfile -t targets < <(cargo +nightly fuzz list)
fi
# A target dir of its own when the caller names one, so that the instrumented build never mixes
# with the workspace's artifacts.
target_dir=()
build_dir=$here/target
if [ -n "${CARGO_TARGET_DIR:-}" ]; then
  build_dir=$CARGO_TARGET_DIR/symphony-fuzz
  target_dir=(--target-dir "$build_dir")
fi
triple=$(rustc +nightly -vV | sed -n 's/^host: //p')
logs=logs/$label
artifacts=artifacts/$label
mkdir -p "$logs" "$artifacts" corpus prepared

if [ -z "$summary_only" ]; then
  cargo +nightly run --quiet --release --example prepare -- \
    "${BELLWETHER_FIXTURES:?set it to the tree bellwether unpack writes}" prepared
  # `cargo fuzz build` takes one target or all of them.
  if [ "${#targets[@]}" -eq "$(cargo +nightly fuzz list | wc -l)" ]; then
    cargo +nightly fuzz build --sanitizer "$sanitizer" "${target_dir[@]}"
  else
    for target in "${targets[@]}"; do
      cargo +nightly fuzz build --sanitizer "$sanitizer" "${target_dir[@]}" "$target"
    done
  fi
fi

run_one() {
  local target=$1
  mkdir -p "corpus/$target" "$artifacts/$target"
  # A seed's name is its family; two tables' slices of one family would collide, so name by table.
  for seed in prepared/seeds/*/*; do
    dest="corpus/$target/$(basename "$(dirname "$seed")")-$(basename "$seed")"
    [ -e "$dest" ] || cp "$seed" "$dest"
  done
  cargo +nightly fuzz run --sanitizer "$sanitizer" "${target_dir[@]}" "$target" "corpus/$target" -- \
    -max_total_time=$((minutes * 60)) -max_len="$max_len" -timeout=10 \
    -dict="prepared/dict/$target.dict" -print_final_stats=1 "${keep_going[@]}" \
    -artifact_prefix="$artifacts/$target/" > "$logs/$target.log" 2>&1 || true
}

# The breach each artifact of `target` names when the target's binary replays it, counted; at
# most two hundred artifacts are replayed.
classes_of() {
  local target=$1 binary="$build_dir/$triple/release/$1" class
  local -a found
  mapfile -t found < <(find "$artifacts/$target" -type f | sort)
  for artifact in "${found[@]:0:200}"; do
    class=$("$binary" "$artifact" 2>&1 | grep -oE 'breach [a-z-]+' | head -1 | cut -d' ' -f2 || true)
    echo "${class:-other}"
  done | sort | uniq -c | sort -rn | awk '{printf "%s%s (%d)", (NR>1?", ":""), $2, $1}'
}

if [ -z "$summary_only" ]; then
  # In fork mode the watchdog stops a child that wedged (no CPU time over two minutes), so a
  # wedge costs one job, not the target's budget; it watches this run's own processes alone.
  watchdog=
  if [ ${#keep_going[@]} -gt 0 ]; then
    "$here/stall-watchdog.sh" $$ "$build_dir/$triple/release" "$logs/watchdog.log" &
    watchdog=$!
  fi
  # The runs are waited for by pid, never with a bare `wait`, which would wait on the watchdog too;
  # a run that ended leaves the list, so a later `wait -n` never names a job already reaped.
  runs=()
  running=0
  for target in "${targets[@]}"; do
    run_one "$target" &
    runs+=($!)
    running=$((running + 1))
    if [ "$running" -ge "$jobs" ]; then
      wait -n -p finished "${runs[@]}" || true
      left=()
      for pid in "${runs[@]}"; do [ "$pid" = "${finished:-}" ] || left+=("$pid"); done
      runs=("${left[@]}")
      running=$((running - 1))
    fi
  done
  [ ${#runs[@]} -eq 0 ] || wait "${runs[@]}"
  if [ -n "$watchdog" ]; then
    pkill -P "$watchdog" 2>/dev/null || true
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true
  fi
fi

failed=0
echo
echo "run $label: $minutes min per target, sanitizer $sanitizer, max_len $max_len${keep_going[0]:+, keep going}${SYMPHONY_FUZZ_ALLOW:+, allowed: $SYMPHONY_FUZZ_ALLOW}"
echo
echo "| target | minutes | executions | exec/s | edges | corpus units | crashes | classes |"
echo "|---|---|---|---|---|---|---|---|"
for target in "${targets[@]}"; do
  log="$logs/$target.log"
  # The last status line carries the executions (`#N`, cumulative in fork mode too), the rate
  # (`exec/s`, over the time the target ran: less than its budget when it stopped at a crash) and
  # the edges (`cov:`). No execution count and no crash means the target never ran (a flag or a
  # dictionary libFuzzer refused, a missing binary): that fails the run as a crash does. A crash
  # while the corpus loads comes before the first status line, so the crashes are read first.
  executions=$(grep -oE '^#[0-9]+' "$log" | tail -1 | tr -d '#' || true)
  per_second=$(grep -oE 'exec/s:? *[0-9]+' "$log" | tail -1 | grep -oE '[0-9]+$' || true)
  edges=$(grep -o 'cov: [0-9]*' "$log" | tail -1 | awk '{print $2}' || true)
  # A target the label never ran has no directories yet: zero units, zero crashes, and the
  # table goes on (set -e would otherwise stop at the first missing one under --summary-only).
  units=0
  if [ -d "corpus/$target" ]; then units=$(find "corpus/$target" -type f | wc -l | tr -d ' '); fi
  crashes=0
  if [ -d "$artifacts/$target" ]; then crashes=$(find "$artifacts/$target" -type f | wc -l | tr -d ' '); fi
  classes=none
  if [ "$crashes" -gt 0 ]; then
    failed=1
    classes=$(classes_of "$target")
  elif [ -z "$executions" ]; then
    failed=1
    classes="never ran"
  fi
  echo "| $target | $minutes | ${executions:-?} | ${per_second:-?} | ${edges:-?} | $units | $crashes | $classes |"
done
exit $failed
