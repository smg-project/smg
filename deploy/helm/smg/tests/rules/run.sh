#!/usr/bin/env bash
# Unit tests of the chart's PrometheusRule.
#
# Renders the rules the chart ships (release `smg` in namespace `smg`, so the
# job selector reads namespace="smg", job="smg-router"), checks them with
# promtool, and replays every `*.test.yaml` beside this script against them
# with `promtool test rules`. Needs `helm` and `promtool` on the PATH.
#
#   tests/rules/run.sh            # this chart
#   tests/rules/run.sh <chart>    # another checkout's chart against these tests
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
chart=${1:-$(cd "$here/../.." && pwd)}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

helm template smg "$chart" --namespace smg --set prometheusRule.enabled=true \
  --show-only templates/prometheusrule.yaml > "$work/prometheusrule.yaml"
# The rule file promtool reads is the PrometheusRule's spec: its groups, de-indented.
sed -n '/^spec:$/,$p' "$work/prometheusrule.yaml" | sed '1d; s/^  //' > "$work/rules.yaml"
grep -q '^groups:$' "$work/rules.yaml" || {
  echo "run.sh: no rule groups rendered from $chart" >&2
  exit 1
}
promtool check rules "$work/rules.yaml"
cp "$here"/*.test.yaml "$work"/
promtool test rules "$work"/*.test.yaml
