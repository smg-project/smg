#!/usr/bin/env bash
# The one rule for a branch name, read by the pre-push hook (.pre-commit-config.yaml)
# and by the pull-request workflow (.github/workflows/pr-naming-check.yml):
# `<type>/<description>` or `<username>/<description>` in lowercase letters, digits,
# dots, dashes and underscores, with one or more `/`-separated segments after the
# first (a tool's branch, such as dependabot's, nests more). `main` is exempt.
#
# Usage: branch-name-check.sh [branch]   (without an argument, the checked-out branch)
set -euo pipefail

branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"
if [ "$branch" = "main" ] || [[ "$branch" =~ ^[a-z0-9]([a-z0-9._-]*[a-z0-9])?(/[a-z0-9._-]+)+$ ]]; then
  exit 0
fi
echo "Branch \"$branch\" does not follow the naming convention: <type>/<description> or <username>/<description>, lowercase"
exit 1
