#!/usr/bin/env bash

set -Eeuo pipefail

readonly DEFAULT_LOCK_PATH=/run/lock/amd-ci-gpu.lock

lock_path=${DEFAULT_LOCK_PATH}
if [[ ${1:-} == --lock-path ]]; then
  [[ $# -ge 3 ]] || {
    echo "ERROR: --lock-path requires a path" >&2
    exit 2
  }
  lock_path=$2
  shift 2
fi

if [[ $# -ne 4 ]]; then
  echo "usage: $0 [--lock-path PATH] OWNER_PID OWNER_STARTTIME READY_PATH WAIT_SECONDS" >&2
  exit 2
fi

readonly OWNER_PID=$1
readonly OWNER_STARTTIME=$2
readonly READY_PATH=$3
readonly WAIT_SECONDS=$4

[[ ${OWNER_PID} =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: owner PID must be numeric" >&2
  exit 2
}
[[ ${OWNER_STARTTIME} =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: owner start time must be numeric" >&2
  exit 2
}
[[ ${WAIT_SECONDS} =~ ^[0-9]+$ ]] || {
  echo "ERROR: wait time must be a non-negative integer" >&2
  exit 2
}

process_matches() {
  local stat_line stat_tail
  local -a fields

  [[ -r /proc/${OWNER_PID}/stat ]] || return 1
  stat_line=$(<"/proc/${OWNER_PID}/stat")
  stat_tail=${stat_line##*) }
  read -r -a fields <<<"${stat_tail}"
  [[ ${#fields[@]} -ge 20 && ${fields[0]} != Z && ${fields[19]} == "${OWNER_STARTTIME}" ]]
}

write_status() {
  local status=$1
  local temporary_directory normalized_ready normalized_lock

  normalized_ready=$(/usr/bin/realpath -m -- "${READY_PATH}" 9>&-) || return 2
  normalized_lock=$(/usr/bin/realpath -m -- "${lock_path}" 9>&-) || return 2
  if [[ ${READY_PATH} == "${lock_path}" || ${READY_PATH} -ef ${lock_path} || ${normalized_ready} == "${normalized_lock}" ]]; then
    printf '%s\n' "ERROR: ready path aliases the lease lock" >&2 || true
    return 2
  fi
  # A fresh private directory prevents following a pre-created status entry.
  # The ready parent directory must still be trusted by this holder's owner.
  temporary_directory=$(mktemp -d -- "${READY_PATH}.tmp.XXXXXXXXXX" 9>&-) || return 1
  if ! printf '%s\n' "${status}" >"${temporary_directory}/status"; then
    rm -f -- "${temporary_directory}/status" 9>&- || true
    rmdir -- "${temporary_directory}" 9>&- || true
    return 1
  fi
  if ! mv -fT -- "${temporary_directory}/status" "${READY_PATH}" 9>&-; then
    rm -f -- "${temporary_directory}/status" 9>&- || true
    rmdir -- "${temporary_directory}" 9>&- || true
    return 1
  fi
  rmdir -- "${temporary_directory}" 9>&-
}

lease_acquired=0
lease_log() (
  trap - EXIT
  # Logging must neither retain the lease nor change the holder's result.
  exec 9>&-
  local event=$1 host=unknown
  # Read the kernel hostname without spawning another logging child.
  IFS= read -r host </proc/sys/kernel/hostname || host=unknown
  if ! /usr/bin/timeout --kill-after=1s 2s logger -t smg-amd-ci-lease -- \
    "event=${event} host=${host} holder_pid=$$ owner_pid=${OWNER_PID} owner_starttime=${OWNER_STARTTIME} lock=${lock_path}"; then
    printf '%s\n' "lease audit unavailable: event=${event} holder_pid=$$ owner_pid=${OWNER_PID}" >&2 || true
  fi
  return 0
)
lease_exit() {
  local result=$?
  if (( lease_acquired )); then
    # Close only this holder's descriptor; preserve the shared lock inode.
    exec 9>&-
    lease_log released
  fi
  return "${result}"
}
trap lease_exit EXIT

process_matches || {
  write_status owner-gone
  exit 1
}

lease_log waiting
exec 9>>"${lock_path}"
if ! flock -w "${WAIT_SECONDS}" 9; then
  lease_log wait-timeout
  write_status busy
  exit 75
fi
lease_acquired=1
lease_log acquired

process_matches || {
  write_status owner-gone
  exit 1
}

write_status acquired

while process_matches; do
  sleep 2 9>&-
done
