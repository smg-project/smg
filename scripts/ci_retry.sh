#!/bin/bash
# Retry a command that can fail transiently (registry pushes, apt mirrors,
# package indexes, git fetches).
#
# Usage: ci_retry.sh <attempts> <delay-seconds> <command> [args...]
#
# The delay doubles after each failed attempt. The exit status is the last
# attempt's, so a deterministic failure still fails the step.

set -uo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 <attempts> <delay-seconds> <command> [args...]" >&2
    exit 2
fi

attempts="$1"
delay="$2"
shift 2

# A non-numeric count would make the loop below run zero times and exit 0
# without ever running the command.
if ! [[ "$attempts" =~ ^[1-9][0-9]*$ ]] || ! [[ "$delay" =~ ^[0-9]+$ ]]; then
    echo "ci_retry: attempts must be a positive integer and delay a non-negative integer (got '${attempts}' '${delay}')" >&2
    exit 2
fi

status=0
for ((attempt = 1; attempt <= attempts; attempt++)); do
    "$@" && exit 0
    status=$?
    if [ "$attempt" -lt "$attempts" ]; then
        echo "ci_retry: attempt ${attempt}/${attempts} of '$*' failed (exit ${status}); retrying in ${delay}s" >&2
        sleep "$delay"
        delay=$((delay * 2))
    fi
done
echo "ci_retry: '$*' failed after ${attempts} attempts (exit ${status})" >&2
exit "$status"
