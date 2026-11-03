#!/usr/bin/env bash
# Renders each accelerator runner scale set from runner-values-gpu-base.yaml plus its overlay with the
# runner scale-set chart, and with --against <git rev> compares the render, byte for byte, with the one
# of that revision's runner-values-<set>.yaml (the files before the base/overlay split, or any later
# state). The chart comes from CHART (a chart path or archive) or the registry at the pinned version.
#
#   scripts/k8s-runner-resources/check-runner-values.sh [--against <rev>] [<set>...]
#
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
chart=${CHART:-oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set}
version=${CHART_VERSION:-0.14.2}
against=""
if [ "${1:-}" = "--against" ]; then against=$2; shift 2; fi
sets=("$@")
if [ ${#sets[@]} -eq 0 ]; then  # every overlay beside the base: the values files that are not the base, the CPU or the recording set
  for f in "$here"/runner-values-*.yaml; do
    case "$f" in *-gpu-base.yaml|*-cpu.yaml|*-bellwether-record.yaml) continue ;; esac
    f=${f##*/runner-values-}; sets+=("${f%.yaml}")
  done
fi
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
version_flag=(); case "$chart" in oci://*) version_flag=(--version "$version");; esac
render() { helm template "$1" "$chart" ${version_flag[@]+"${version_flag[@]}"} --namespace actions-runner-system "${@:2}"; }
rc=0
for set in "${sets[@]}"; do
  render "$set" -f "$here/runner-values-gpu-base.yaml" -f "$here/runner-values-$set.yaml" > "$tmp/$set.new.yaml"
  if [ -z "$against" ]; then
    echo "$set: rendered from the base and its overlay ($(wc -l < "$tmp/$set.new.yaml") lines)"
    continue
  fi
  git -C "$here" show "$against:scripts/k8s-runner-resources/runner-values-$set.yaml" > "$tmp/$set.rev.yaml"
  if git -C "$here" cat-file -e "$against:scripts/k8s-runner-resources/runner-values-gpu-base.yaml" 2>/dev/null; then
    git -C "$here" show "$against:scripts/k8s-runner-resources/runner-values-gpu-base.yaml" > "$tmp/base.rev.yaml"
    render "$set" -f "$tmp/base.rev.yaml" -f "$tmp/$set.rev.yaml" > "$tmp/$set.old.yaml"
  else
    render "$set" -f "$tmp/$set.rev.yaml" > "$tmp/$set.old.yaml"
  fi
  if cmp -s "$tmp/$set.old.yaml" "$tmp/$set.new.yaml"; then
    echo "$set: renders byte-equal to $against ($(wc -l < "$tmp/$set.new.yaml") lines)"
  else
    echo "$set: RENDER DIFFERS from $against"; diff "$tmp/$set.old.yaml" "$tmp/$set.new.yaml" || true; rc=1
  fi
done
exit $rc
