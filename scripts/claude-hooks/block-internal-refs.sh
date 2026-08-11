#!/usr/bin/env bash
# Claude Code PreToolUse hook: refuse a Write/Edit whose new text carries
# internal-only references, so the text never reaches disk. `git commit
# --no-verify` cannot get around this one, because there is nothing to commit.
#
# Reads the tool-call JSON on stdin; exit 2 blocks the call and returns stderr
# to the model.

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checker="${repo_root}/scripts/check-no-internal-refs.sh"
[[ -x "${checker}" ]] || exit 0

payload="$(cat)"
text="$(jq -r '[.tool_input.content, .tool_input.new_string] | map(select(. != null)) | join("\n")' <<<"${payload}" 2>/dev/null)"
[[ -z "${text}" ]] && exit 0

scratch="$(mktemp)"
trap 'rm -f "${scratch}"' EXIT
printf '%s\n' "${text}" >"${scratch}"

if ! findings="$("${checker}" "${scratch}" 2>/dev/null)"; then
  target="$(jq -r '.tool_input.file_path // "the edited file"' <<<"${payload}")"
  {
    echo "Blocked: this edit would write internal-only references into ${target}."
    echo "${findings//${scratch}:/line }"
    echo "Rewrite the flagged lines from first principles — describe the behaviour"
    echo "or constraint without naming an internal system, ticket, or host."
  } >&2
  exit 2
fi
