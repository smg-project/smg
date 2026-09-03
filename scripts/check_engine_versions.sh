#!/bin/bash
# Compare pinned engine versions against the latest upstream releases.
#
# Emits one JSON object per line: {"engine","current","latest","update"}.
# Pins are read from the same files the upgrade PRs edit, so a bump lands
# here automatically. TokenSpeed is a source build pinned to a commit, so
# its "versions" are SHAs and any divergence from upstream main counts as
# an update.
#
# Requires: curl, jq; GH_TOKEN for the TokenSpeed upstream lookup.

set -euo pipefail
cd "$(dirname "$0")/.."

CURL=(curl -fsS --connect-timeout 10 --max-time 60)

require() { # name value
    if [ -z "$2" ]; then
        echo "ERROR: could not determine $1 (pin or upstream format changed?)" >&2
        exit 1
    fi
}

# Newest of two versions under PEP 440 pre-release ordering: rcN sorts
# before its final release (GNU sort -V treats '~' as lowest).
vmax() {
    printf '%s\n%s\n' "$1" "$2" | sed 's/rc/~rc/' | sort -V | tail -1 | sed 's/~rc/rc/'
}

emit() { # engine current latest
    local update=false
    if [ "$1" = "tokenspeed" ]; then
        [ "$2" != "$3" ] && update=true
    else
        [ "$(vmax "$2" "$3")" != "$2" ] && update=true
    fi
    jq -cn --arg e "$1" --arg c "$2" --arg l "$3" --argjson u "$update" \
        '{engine: $e, current: $c, latest: $l, update: $u}'
}

sglang_current=$(sed -n 's/.*"sglang\[all\]==\([^"]*\)".*/\1/p' scripts/ci_install_sglang.sh | head -1)
require "sglang pin" "$sglang_current"
sglang_latest=$("${CURL[@]}" https://pypi.org/pypi/sglang/json | jq -r .info.version)
require "sglang latest" "$sglang_latest"
emit sglang "$sglang_current" "$sglang_latest"

vllm_current=$(sed -n "s/.*default: 'vllm\/vllm-openai:v\([0-9.]*\)'.*/\1/p" .github/workflows/release-vllm-docker.yml | head -1)
require "vllm pin" "$vllm_current"
vllm_latest=$("${CURL[@]}" https://pypi.org/pypi/vllm/json | jq -r .info.version)
require "vllm latest" "$vllm_latest"
emit vllm "$vllm_current" "$vllm_latest"

trtllm_current=$(sed -n 's/^TRTLLM_VERSION="\(.*\)"$/\1/p' scripts/ci_install_trtllm.sh | head -1)
require "tensorrt-llm pin" "$trtllm_current"
trtllm_latest=$("${CURL[@]}" https://pypi.nvidia.com/tensorrt-llm/ \
    | grep -o 'tensorrt_llm-[0-9][^-]*' | sed 's/tensorrt_llm-//;s/rc/~rc/' | sort -uV | tail -1 | sed 's/~rc/rc/')
require "tensorrt-llm latest" "$trtllm_latest"
emit tensorrt-llm "$trtllm_current" "$trtllm_latest"

tokenspeed_current=$(sed -n 's/.*TOKENSPEED_REF:-\([0-9a-f]*\)}.*/\1/p' scripts/ci_install_tokenspeed.sh | head -1)
require "tokenspeed pin" "$tokenspeed_current"
tokenspeed_latest=$(gh api repos/lightseekorg/tokenspeed/commits/main --jq .sha)
require "tokenspeed latest" "$tokenspeed_latest"
emit tokenspeed "$tokenspeed_current" "$tokenspeed_latest"
