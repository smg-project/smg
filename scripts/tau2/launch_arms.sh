#!/usr/bin/env bash
# Native frontend A/B; implementation owns only its invocation's sessions.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/native_arms.py" "$@"
