#!/bin/bash
# Generate Docker image markdown for release notes.
#
# Queries GHCR for all container tags matching a given SMG version
# and outputs a markdown snippet to paste into release notes.
#
# Usage:
#   ./scripts/release_docker_notes.sh v1.3.1
#   ./scripts/release_docker_notes.sh 1.3.1
#
# Uses the OCI registry API (no auth needed for public packages).
# Falls back to gh CLI if available with read:packages scope.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 <version>" >&2
    echo "Example: $0 v1.3.1" >&2
    exit 1
fi

# Strip leading 'v' for tag matching
VERSION="${VERSION#v}"

PACKAGE="smg"
ORG="smg-project"
REGISTRY="ghcr.io/${ORG}/${PACKAGE}"

# ---------------------------------------------------------------------------
# Fetch tags — try OCI registry API first (works without auth for public pkgs)
# ---------------------------------------------------------------------------
fetch_tags_oci() {
    # Get an anonymous token for the public package
    local token
    token=$(curl -sf "https://ghcr.io/token?scope=repository:${ORG}/${PACKAGE}:pull" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" 2>/dev/null) || return 1

    # List tags from the OCI registry, following rel=next pagination links.
    local url="https://ghcr.io/v2/${ORG}/${PACKAGE}/tags/list?n=1000"
    while [[ -n "$url" ]]; do
        local headers body next
        headers=$(mktemp)
        body=$(curl -sf -D "$headers" -H "Authorization: Bearer ${token}" "$url") || {
            rm -f "$headers"
            return 1
        }
        printf '%s' "$body" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for tag in sorted(data.get('tags', [])):
    print(tag)
" 2>/dev/null || {
            rm -f "$headers"
            return 1
        }
        next=$(python3 - "$headers" <<'PY'
import re
import sys

with open(sys.argv[1], encoding="utf-8") as headers:
    for line in headers:
        if line.lower().startswith("link:"):
            match = re.search(r'<([^>]+)>;\s*rel="next"', line, re.IGNORECASE)
            if match:
                print(match.group(1))
                break
PY
)
        rm -f "$headers"
        if [[ -n "$next" ]]; then
            url="https://ghcr.io${next}"
        else
            url=""
        fi
    done
}

fetch_tags_gh() {
    gh api --paginate \
        "/orgs/${ORG}/packages/container/${PACKAGE}/versions" \
        --jq ".[].metadata.container.tags[]" 2>/dev/null
}

echo "Fetching tags for ${REGISTRY}..." >&2
ALL_TAGS=$(fetch_tags_oci 2>/dev/null || fetch_tags_gh 2>/dev/null || true)

# Filter to tags matching this version
TAGS=$(echo "$ALL_TAGS" | grep "^${VERSION}-" | sort -t'-' -k2,2 -k3,3V || true)

if [[ -z "$TAGS" ]]; then
    echo "No Docker images found for version ${VERSION}" >&2
    echo "" >&2
    echo "Make sure:" >&2
    echo "  1. Docker builds have completed for this release" >&2
    echo "  2. The version tag is correct (tried: ${VERSION}-*)" >&2
    echo "" >&2
    echo "Available tags matching '${VERSION}':" >&2
    echo "$ALL_TAGS" | grep "${VERSION}" | head -10 >&2 || echo "  (none)" >&2
    exit 1
fi

TAG_COUNT=$(echo "$TAGS" | wc -l | tr -d ' ')
echo "Found ${TAG_COUNT} image(s) for v${VERSION}" >&2
echo "" >&2

# ---------------------------------------------------------------------------
# Group tags by engine and output markdown
# Uses python3 to avoid bash 3.x associative array limitations (macOS)
# ---------------------------------------------------------------------------
echo "$TAGS" | python3 -c "
import sys

registry = '${REGISTRY}'
version = '${VERSION}'

tags = [line.strip() for line in sys.stdin if line.strip()]
engines = {'sglang': [], 'vllm': [], 'trtllm': [], 'other': []}

for tag in tags:
    suffix = tag[len(version) + 1:]  # strip 'VERSION-'
    engine = suffix.split('-')[0]
    if engine in engines:
        engines[engine].append(tag)
    else:
        engines['other'].append(tag)

labels = {'sglang': 'SGLang', 'vllm': 'vLLM', 'trtllm': 'TensorRT-LLM', 'other': 'Other'}

print('### Docker Images')
print()
print(f'Pre-built engine images on [GitHub Container Registry](https://github.com/orgs/smg-project/packages/container/package/smg):')
print()

for engine in ['sglang', 'vllm', 'trtllm', 'other']:
    engine_tags = engines[engine]
    if not engine_tags:
        continue
    print(f'**{labels[engine]}:**')
    print('\`\`\`bash')
    for tag in engine_tags:
        print(f'docker pull {registry}:{tag}')
    print('\`\`\`')
    print()

print('<details>')
print(f'<summary>All images for v{version}</summary>')
print()
print('| Engine | Tag | Pull Command |')
print('|--------|-----|--------------|')
for tag in tags:
    suffix = tag[len(version) + 1:]
    engine = suffix.split('-')[0]
    print(f'| {engine} | \`{tag}\` | \`docker pull {registry}:{tag}\` |')
print()
print('</details>')
"
