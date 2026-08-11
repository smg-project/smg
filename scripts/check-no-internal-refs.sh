#!/usr/bin/env bash
# Reject content that only makes sense inside a private network from a public
# repository: internal hostnames, internal-tracker identifiers, corporate email
# addresses, and AI-tool attribution.
#
# Codenames for internal systems are deliberately NOT listed here — writing them
# down in a public file is the very leak this guard exists to prevent. Point
# SMG_INTERNAL_REF_PATTERNS at a file of extra extended-regex patterns (one per
# line, `#` comments allowed) to add them from outside the repository.
#
# Usage: check-no-internal-refs.sh <file>...

set -uo pipefail

patterns=(
  # Internal hostnames and short links.
  '\b(fburl|internalfb|intern\.[a-z0-9-]+\.fb)\.com\b'
  '\bphabricator\.[a-z0-9.-]+\b'
  # Corporate email addresses; commit authorship must use the public identity.
  '[A-Za-z0-9._%+-]+@(fb|meta)\.com\b'
  # Internal tracker identifiers: Phabricator diffs, tasks, SEVs.
  '\b(D[0-9]{7,}|T[0-9]{6,}|S[0-9]{6,})\b'
  # AI-tool attribution, in code and in generated text alike.
  '(Generated|Co-[Aa]uthored)[ -][Ww]ith [Cc]laude'
  'noreply@anthropic\.com'
  '🤖 Generated'
)

if [[ -n "${SMG_INTERNAL_REF_PATTERNS:-}" ]]; then
  if [[ ! -r "${SMG_INTERNAL_REF_PATTERNS}" ]]; then
    echo "check-no-internal-refs: cannot read SMG_INTERNAL_REF_PATTERNS=${SMG_INTERNAL_REF_PATTERNS}" >&2
    exit 1
  fi
  while IFS= read -r line; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    patterns+=("${line}")
  done <"${SMG_INTERNAL_REF_PATTERNS}"
fi

# One alternation over all patterns keeps this a single pass per file.
combined="$(
  IFS='|'
  echo "${patterns[*]}"
)"

# Files that state or enforce the policy have to quote the strings they ban.
is_policy_text() {
  case "${1#./}" in
    scripts/check-no-internal-refs.sh | .pre-commit-config.yaml | \
      .github/workflows/pr-naming-check.yml | CONTRIBUTING.md) return 0 ;;
    *) return 1 ;;
  esac
}

status=0
for file in "$@"; do
  is_policy_text "${file}" && continue
  # Binary files have no prose to leak.
  grep -Iq . "${file}" 2>/dev/null || continue
  if matches="$(grep -nEI "${combined}" "${file}")"; then
    while IFS= read -r match; do
      echo "${file}:${match}"
    done <<<"${matches}"
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  cat >&2 <<'EOF'

ERROR: internal-only references found in files bound for a public repository.

Rewrite the flagged lines so they stand on their own: describe the behaviour or
the constraint from first principles instead of naming an internal system,
ticket, or host. Use the GitHub noreply address for authorship.

To extend this guard with codenames that must not be written down here, set
SMG_INTERNAL_REF_PATTERNS to a file of extra regexes outside the repository.
EOF
fi

exit "${status}"
