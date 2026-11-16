#!/usr/bin/env bash
# The one rule for a branch name, read by the pre-push hook (.pre-commit-config.yaml)
# and by the pull-request workflow (.github/workflows/pr-naming-check.yml):
# `<type>/<description>` or `<username>/<description>` in lowercase letters, digits,
# dots, dashes and underscores, with one or more `/`-separated segments after the
# first (a tool's branch, such as dependabot's, nests more). `main` is exempt.
# A `dependabot/` branch keeps the tool's own case after that prefix: an action owner
# is written there as the workflow writes it (`dependabot/github_actions/PyO3/maturin-action-2`).
#
# Usage: branch-name-check.sh [branch]   (without an argument, the checked-out branch)
# Exit status: 0 accepted, 1 refused. Any other status means the check did not run.
set -euo pipefail

branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"
if [ "$branch" = "main" ] \
  || [[ "$branch" =~ ^[a-z0-9]([a-z0-9._-]*[a-z0-9])?(/[a-z0-9._-]+)+$ ]] \
  || [[ "$branch" =~ ^dependabot/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)+$ ]] \
  ; then
  exit 0
fi
echo "Branch \"$branch\" does not follow the naming convention: <type>/<description> or <username>/<description>, lowercase"
exit 1
