#!/bin/bash
# Pre-release version check for SMG workspace crates and Python packages.
#
# For each workspace crate, verifies:
#   1. Whether there are code changes since the latest git tag
#   2. Whether the crate version was bumped in its own Cargo.toml
#   3. Whether the workspace root Cargo.toml reflects the new version
#
# For each Python package, verifies:
#   1. Whether there are code changes since the latest git tag
#   2. Whether the package __version__ was bumped
#
# Detects bump level from conventional commits:
#   - feat!: or BREAKING CHANGE → major
#   - feat: → minor
#   - fix:, refactor:, perf:, etc. → patch
#
# After the check, offers to auto-bump any unbumped crates/packages.
#
# Usage: ./check_release_versions.sh [tag]
#        If no tag is given, the latest tag is used.
#
# Exit code 0 = all good, 1 = issues found (user declined fix).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Portable sed in-place: macOS uses `sed -i ''`, GNU uses `sed -i`
sed_inplace() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Escape dots in a version string for use in sed regex
escape_version() {
    echo "${1//./\\.}"
}

# ---------------------------------------------------------------------------
# Workspace crate registry
# Format: "crate_name|directory|workspace_dep_key"
# workspace_dep_key is the key in root Cargo.toml [workspace.dependencies].
# Use "-" for the main gateway crate (no workspace dep entry).
# ---------------------------------------------------------------------------
CRATES=(
    "openai-protocol|crates/protocols|openai-protocol"
    "smg-mm-rdma|crates/mm_rdma|smg-mm-rdma"
    "engine-zmq-client|crates/engine_zmq_client|engine-zmq-client"
    "reasoning-parser|crates/reasoning_parser|reasoning-parser"
    "tool-parser|crates/tool_parser|tool-parser"
    "wfaas|crates/workflow|wfaas"
    "llm-tokenizer|crates/tokenizer|llm-tokenizer"
    "smg-auth|crates/auth|smg-auth"
    "smg-mcp|crates/mcp|smg-mcp"
    "kv-index|crates/kv_index|kv-index"
    "data-connector|crates/data_connector|smg-data-connector"
    "llm-multimodal|crates/multimodal|llm-multimodal"
    "smg-wasm|crates/wasm|smg-wasm"
    "smg-mesh|crates/mesh|smg-mesh"
    "smg-grpc-client|crates/grpc_client|smg-grpc-client"
    "smg-client|clients/rust|-"
    "openapi-gen|clients/openapi-gen|-"
    "smg|model_gateway|-"
)

# ---------------------------------------------------------------------------
# Python package registry (versioned independently from Rust crates)
# Format: "package_name|directory|version_file"
# version_file is relative to REPO_ROOT.
# ---------------------------------------------------------------------------
PYTHON_PACKAGES=(
    "smg-grpc-proto|crates/grpc_client/python|crates/grpc_client/python/pyproject.toml"
    "smg-grpc-servicer|grpc_servicer|grpc_servicer/pyproject.toml"
)

# ---------------------------------------------------------------------------
# SMG version sync: files that must mirror the smg (model_gateway) version.
# Format: "label|file|type"
#   type: "cargo"  → first `version = "X.Y.Z"` line in a Cargo.toml
#         "pyproject" → first `version = "X.Y.Z"` line in a pyproject.toml
#         "workflow"  → `default: 'vX.Y.Z'` for smg_commit in a workflow file
# ---------------------------------------------------------------------------
SMG_VERSION_SYNC=(
    "smg-python|bindings/python/Cargo.toml|cargo"
    "smg-golang|bindings/golang/Cargo.toml|cargo"
    "smg pyproject|bindings/python/pyproject.toml|pyproject"
    "helm chart|deploy/helm/smg/Chart.yaml|helm"
    "sglang-docker|.github/workflows/release-sglang-docker.yml|workflow"
    "vllm-docker|.github/workflows/release-vllm-docker.yml|workflow"
    "trtllm-docker|.github/workflows/release-trtllm-docker.yml|workflow"
)

# ---------------------------------------------------------------------------
# Resolve tag
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    TAG="$1"
else
    TAG=$(git tag --sort=-creatordate 2>/dev/null | head -1)
    if [[ -z "$TAG" ]]; then
        echo -e "${RED}No tags found in repository. Pass a tag explicitly.${NC}"
        echo "Usage: $0 [tag]"
        exit 1
    fi
fi

if ! git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Tag '$TAG' does not exist.${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# VERSION_OVERRIDE: skip conventional-commit detection and use this version
# for all bumps. Set via: VERSION_OVERRIDE=1.3.1 ./check_release_versions.sh
# ---------------------------------------------------------------------------
VERSION_OVERRIDE="${VERSION_OVERRIDE:-}"
if [[ -n "$VERSION_OVERRIDE" && ! "$VERSION_OVERRIDE" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}ERROR: VERSION_OVERRIDE must be in X.Y.Z format, got: '$VERSION_OVERRIDE'${NC}" >&2
    exit 1
fi

echo -e "${BOLD}Checking workspace versions against tag: ${BLUE}$TAG${NC}"
if [[ -n "$VERSION_OVERRIDE" ]]; then
    echo -e "${BOLD}Version override: ${CYAN}$VERSION_OVERRIDE${NC} (ignoring conventional commit detection)"
fi
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Extract version from a Cargo.toml (first version = line in [package])
get_crate_version() {
    local file="$1"
    if grep -qE 'version\.workspace\s*=\s*true|version\s*=\s*\{\s*workspace\s*=\s*true' "$file"; then
        echo -e "${RED}ERROR: $file uses workspace versioning; this script expects explicit version strings.${NC}" >&2
        exit 1
    fi
    grep -m1 '^version' "$file" | sed 's/.*"\(.*\)".*/\1/'
}

# Extract version at a specific git ref (returns empty string if crate missing)
# Falls back to pre-crates-move path (e.g., crates/auth → auth) for older tags.
get_crate_version_at_ref() {
    local path="$1"
    local ref="$2"
    local content
    content=$(git show "$ref:$path/Cargo.toml" 2>/dev/null) || {
        # Fallback: try legacy path before crates/ directory move
        if [[ "$path" == crates/* ]]; then
            content=$(git show "$ref:${path#crates/}/Cargo.toml" 2>/dev/null) || return 0
        else
            return 0
        fi
    }
    if echo "$content" | grep -qE 'version\.workspace\s*=\s*true|version\s*=\s*\{\s*workspace\s*=\s*true'; then
        echo -e "${RED}ERROR: $path/Cargo.toml at $ref uses workspace versioning; this script expects explicit version strings.${NC}" >&2
        exit 1
    fi
    echo "$content" | grep -m1 '^version' | sed 's/.*"\(.*\)".*/\1/'
}

# Extract workspace dep version from root Cargo.toml
get_workspace_dep_version() {
    local dep_key="$1"
    local root_toml="$REPO_ROOT/Cargo.toml"
    grep "^${dep_key} " "$root_toml" 2>/dev/null \
        | grep -o 'version = "[^"]*"' \
        | sed 's/version = "\(.*\)"/\1/'
}

# Detect semver bump level from conventional commits touching a directory.
# Scans commit subjects and bodies for:
#   - "feat!:", "fix!:", "<type>!:" or "BREAKING CHANGE" → major
#   - "feat:" or "feat(<scope>):" → minor
#   - everything else → patch
# Returns: "major", "minor", or "patch"
detect_bump_level() {
    local path="$1"
    local level="patch"

    # Get commit hashes that touch this path (and legacy pre-move path)
    local log_paths=("$path/")
    if [[ "$path" == crates/* ]]; then
        log_paths+=("${path#crates/}/")
    fi
    local commits
    commits=$(git log "$TAG"..HEAD --format='%H' --no-merges -- "${log_paths[@]}")
    if [[ -z "$commits" ]]; then
        echo "patch"
        return
    fi

    while IFS= read -r hash; do
        local subject body
        subject=$(git log -1 --format='%s' "$hash")
        body=$(git log -1 --format='%b' "$hash")

        # Check for breaking change indicators
        # 1. Type with ! suffix: feat!:, fix!:, refactor!:, etc.
        if echo "$subject" | grep -qE '^[a-z]+(\([^)]*\))?!:'; then
            echo "major"
            return
        fi
        # 2. BREAKING CHANGE in commit body or footer
        if echo "$body" | grep -q 'BREAKING CHANGE'; then
            echo "major"
            return
        fi

        # Check for feat → minor
        if echo "$subject" | grep -qE '^feat(\([^)]*\))?:'; then
            level="minor"
        fi
    done <<< "$commits"

    echo "$level"
}

# Bump version by level: major, minor, or patch.
# If VERSION_OVERRIDE is set AND this is the smg crate (3rd arg = "smg"),
# returns the override. Independent crates always use normal bump logic.
bump_version() {
    local version="$1"
    local level="$2"
    local crate_name="${3:-}"
    if [[ -n "$VERSION_OVERRIDE" && "$crate_name" == "smg" ]]; then
        echo "$VERSION_OVERRIDE"
        return
    fi
    if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "ERROR: bump_version only supports X.Y.Z format, got: $version" >&2
        return 1
    fi
    local major minor patch
    IFS='.' read -r major minor patch <<< "$version"
    case "$level" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "${major}.$((minor + 1)).0" ;;
        patch) echo "${major}.${minor}.$((patch + 1))" ;;
        *) echo "ERROR: bump_version unknown level: $level" >&2; return 1 ;;
    esac
}

# Pretty label for bump level
bump_label() {
    case "$1" in
        major) echo -e "${RED}major${NC}" ;;
        minor) echo -e "${YELLOW}minor${NC}" ;;
        patch) echo -e "${CYAN}patch${NC}" ;;
    esac
}

# Update version in a Cargo.toml (first version = line only)
set_crate_version() {
    local file="$1"
    local new_version="$2"
    awk -v new="$new_version" '
        !done && /^version = ".*"/ { sub(/^version = ".*"/, "version = \"" new "\""); done=1 }
        { print }
    ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
    if ! grep -q "^version = \"${new_version}\"" "$file"; then
        echo -e "    ${RED}FAILED to update $file${NC}" >&2
        return 1
    fi
}

# Extract version from a Python package (pyproject.toml or __init__.py)
get_python_version() {
    local file="$1"
    if [[ "$file" == *.toml ]]; then
        grep -m1 '^version' "$file" | sed 's/.*"\([^"]*\)".*/\1/'
    else
        grep '__version__' "$file" | sed 's/.*"\([^"]*\)".*/\1/'
    fi
}

# Extract version from a Python package at a specific git ref
# Falls back to pre-crates-move path (e.g., crates/X/... → X/...) for older tags.
get_python_version_at_ref() {
    local file="$1"
    local ref="$2"
    local content
    content=$(git show "$ref:$file" 2>/dev/null) || {
        # Fallback: try legacy path before crates/ directory move
        if [[ "$file" == crates/* ]]; then
            content=$(git show "$ref:${file#crates/}" 2>/dev/null) || return 0
        else
            return 0
        fi
    }
    if [[ "$file" == *.toml ]]; then
        echo "$content" | grep -m1 '^version' | sed 's/.*"\([^"]*\)".*/\1/'
    else
        echo "$content" | grep '__version__' | sed 's/.*"\([^"]*\)".*/\1/'
    fi
}

# Update version in a Python package (pyproject.toml or __init__.py)
set_python_version() {
    local file="$1"
    local new_version="$2"
    if [[ "$file" == *.toml ]]; then
        # Update first version = line in pyproject.toml (same as Cargo.toml)
        awk -v new="$new_version" '
            !done && /^version = ".*"/ { sub(/"[^"]*"/, "\"" new "\""); done=1 }
            { print }
        ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
        if ! grep -q "^version = \"${new_version}\"" "$file"; then
            echo -e "    ${RED}FAILED to update $file${NC}" >&2
            return 1
        fi
    else
        sed_inplace "s/__version__ = \".*\"/__version__ = \"${new_version}\"/" "$file"
        if ! grep -q "__version__ = \"${new_version}\"" "$file"; then
            echo -e "    ${RED}FAILED to update $file${NC}" >&2
            return 1
        fi
    fi
}

# Check if a version file has changes beyond just the version line.
# Uses rename detection to handle crate directory moves correctly —
# a pure rename (R100) produces no content diff lines, so it returns false.
has_non_version_changes() {
    local file="$1"
    local tag="$2"
    # Build file paths (include legacy pre-move path for rename detection)
    local diff_files=("$file")
    local rel="${file#$REPO_ROOT/}"
    if [[ "$rel" == crates/* ]]; then
        diff_files+=("$REPO_ROOT/${rel#crates/}")
    fi
    # Check if the file was even changed
    if ! git diff --name-only -M "$tag"..HEAD -- "${diff_files[@]}" | grep -q .; then
        return 1
    fi
    # Count diff lines that aren't version-related (exclude --- / +++ headers)
    local pattern
    if [[ "$file" == *.toml ]]; then
        pattern='^[+-][[:space:]]*version[[:space:]]*='
    else
        pattern='^[+-].*__version__'
    fi
    local non_ver_lines
    non_ver_lines=$(git diff -M "$tag"..HEAD -- "${diff_files[@]}" \
        | grep '^[+-]' | grep -v '^[+-][+-][+-]' \
        | grep -cv "$pattern" || true)
    [[ "$non_ver_lines" -gt 0 ]]
}

# Extract version from a Helm Chart.yaml (the `version:` field, not appVersion)
get_helm_chart_version() {
    local file="$1"
    awk -F':[[:space:]]*' '/^version:/ {gsub(/"/, "", $2); print $2; exit}' "$file"
}

# Update both version and appVersion in a Helm Chart.yaml
set_helm_chart_version() {
    local file="$1"
    local old_version="$2"
    local new_version="$3"
    local escaped_old
    escaped_old=$(escape_version "$old_version")
    sed_inplace "s/^\(version:[[:space:]]*\)${escaped_old}/\1${new_version}/" "$file"
    # Unconditionally set appVersion to new_version (handles drift)
    sed_inplace "s/^appVersion:[[:space:]]*.*/appVersion: \"${new_version}\"/" "$file"
    if ! grep -q "^version:[[:space:]]*${new_version}" "$file" \
       || ! grep -q "^appVersion: \"${new_version}\"" "$file"; then
        echo -e "    ${RED}FAILED to update $file${NC}" >&2
        return 1
    fi
}

# Extract smg_commit default version from a workflow file (e.g. "default: 'v1.2.0'")
get_workflow_smg_version() {
    local file="$1"
    grep -m1 "default: 'v[0-9]" "$file" | sed "s/.*default: 'v\([^']*\)'.*/\1/"
}

# Update all smg version references in a workflow file.
# Replaces the version in default: values, || fallbacks, and description examples.
set_workflow_smg_version() {
    local file="$1"
    local old_version="$2"
    local new_version="$3"
    local escaped_old
    escaped_old=$(escape_version "$old_version")
    sed_inplace "s/v${escaped_old}/v${new_version}/g" "$file"
    if ! grep -q "default: 'v${new_version}'" "$file"; then
        echo -e "    ${RED}FAILED to update $file${NC}" >&2
        return 1
    fi
}

# Update workspace dep version in root Cargo.toml
set_workspace_dep_version() {
    local dep_key="$1"
    local old_version="$2"
    local new_version="$3"
    local root_toml="$REPO_ROOT/Cargo.toml"
    local escaped_old
    escaped_old=$(escape_version "$old_version")
    sed_inplace "s/^${dep_key} = { version = \"${escaped_old}\"/${dep_key} = { version = \"${new_version}\"/" "$root_toml"
    if ! grep -q "^${dep_key} .* version = \"${new_version}\"" "$root_toml"; then
        echo -e "    ${RED}FAILED to update $dep_key in workspace Cargo.toml to v${new_version}${NC}" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Phase 1: Check all crates, collect unbumped ones
# ---------------------------------------------------------------------------
issues=0
changed=0
clean=0

# Collect crates that need bumping: "name|path|dep_key|current_version|bump_level"
NEEDS_BUMP=()
# Collect crates with workspace Cargo.toml mismatch: "name|dep_key|crate_version|ws_version|path"
NEEDS_WS_SYNC=()
# Collect Python packages that need bumping: "name|path|version_file|current_version|bump_level"
NEEDS_PY_BUMP=()

for entry in "${CRATES[@]}"; do
    IFS='|' read -r name path dep_key <<< "$entry"

    # 1. Check for code changes since tag (exclude version-only changes in Cargo.toml)
    # Include legacy pre-move path and use rename detection to avoid
    # counting crate directory moves (R100) as content changes.
    _diff_paths=("$path/")
    if [[ "$path" == crates/* ]]; then
        _diff_paths+=("${path#crates/}/")
    fi
    diff_count=$(git diff --name-status -M "$TAG"..HEAD -- "${_diff_paths[@]}" \
        | grep -Ev '^R100' \
        | grep -cv 'Cargo\.toml$' || true)
    if has_non_version_changes "$REPO_ROOT/$path/Cargo.toml" "$TAG"; then
        diff_count=$((diff_count + 1))
    fi
    if [[ "$diff_count" -eq 0 ]]; then
        clean=$((clean + 1))
        continue
    fi

    changed=$((changed + 1))
    current_version=$(get_crate_version "$path/Cargo.toml")
    tag_version=$(get_crate_version_at_ref "$path" "$TAG")

    # Handle crate not existing at the tag (new crate)
    if [[ -z "$tag_version" ]]; then
        echo -e "  ${GREEN}✓${NC} ${BOLD}$name${NC} ($path/) — new crate (v$current_version), $diff_count file(s) changed"
        if [[ "$dep_key" != "-" ]]; then
            ws_version=$(get_workspace_dep_version "$dep_key")
            if [[ "$ws_version" != "$current_version" ]]; then
                echo -e "    ${RED}✗ workspace Cargo.toml has $dep_key v$ws_version, expected v$current_version${NC}"
                NEEDS_WS_SYNC+=("$name|$dep_key|$current_version|$ws_version|$path")
                issues=$((issues + 1))
            fi
        fi
        continue
    fi

    # 2. Check if version was bumped
    if [[ "$current_version" == "$tag_version" ]]; then
        level=$(detect_bump_level "$path")
        echo -e "  ${YELLOW}!${NC} ${BOLD}$name${NC} ($path/) — $diff_count file(s) changed but version not bumped (v$current_version) [$(bump_label "$level")]"
        NEEDS_BUMP+=("$name|$path|$dep_key|$current_version|$level")
        issues=$((issues + 1))
        continue
    fi

    echo -e "  ${GREEN}✓${NC} ${BOLD}$name${NC} ($path/) — v$tag_version → v$current_version ($diff_count file(s) changed)"

    # 3. Check workspace root Cargo.toml
    if [[ "$dep_key" != "-" ]]; then
        ws_version=$(get_workspace_dep_version "$dep_key")
        if [[ "$ws_version" != "$current_version" ]]; then
            echo -e "    ${RED}✗ workspace Cargo.toml has $dep_key v$ws_version, expected v$current_version${NC}"
            NEEDS_WS_SYNC+=("$name|$dep_key|$current_version|$ws_version|$path")
            issues=$((issues + 1))
        fi
    fi
done

# ---------------------------------------------------------------------------
# Phase 1b: Check all Python packages
# ---------------------------------------------------------------------------
py_changed=0
py_clean=0

for entry in "${PYTHON_PACKAGES[@]}"; do
    IFS='|' read -r name path version_file <<< "$entry"

    # 1. Check for code changes since tag (exclude version-only changes in the version file)
    # Include legacy pre-move path and use rename detection.
    _diff_paths=("$path/")
    if [[ "$path" == crates/* ]]; then
        _diff_paths+=("${path#crates/}/")
    fi
    diff_count=$(git diff --name-status -M "$TAG"..HEAD -- "${_diff_paths[@]}" \
        | grep -Ev '^R100' \
        | grep -cv "$(basename "$version_file")$" || true)
    if has_non_version_changes "$REPO_ROOT/$version_file" "$TAG"; then
        diff_count=$((diff_count + 1))
    fi
    if [[ "$diff_count" -eq 0 ]]; then
        py_clean=$((py_clean + 1))
        continue
    fi

    py_changed=$((py_changed + 1))
    current_version=$(get_python_version "$version_file")
    tag_version=$(get_python_version_at_ref "$version_file" "$TAG")

    # Handle package not existing at the tag (new package)
    if [[ -z "$tag_version" ]]; then
        echo -e "  ${GREEN}✓${NC} ${BOLD}$name${NC} ($path/) — new package (v$current_version), $diff_count file(s) changed"
        continue
    fi

    # 2. Check if version was bumped
    if [[ "$current_version" == "$tag_version" ]]; then
        level=$(detect_bump_level "$path")
        echo -e "  ${YELLOW}!${NC} ${BOLD}$name${NC} ($path/) — $diff_count file(s) changed but version not bumped (v$current_version) [$(bump_label "$level")]"
        NEEDS_PY_BUMP+=("$name|$path|$version_file|$current_version|$level")
        issues=$((issues + 1))
        continue
    fi

    echo -e "  ${GREEN}✓${NC} ${BOLD}$name${NC} ($path/) — v$tag_version → v$current_version ($diff_count file(s) changed)"
done

# ---------------------------------------------------------------------------
# Phase 1c: Check SMG version sync (files that must mirror model_gateway)
# ---------------------------------------------------------------------------
smg_version=$(get_crate_version "model_gateway/Cargo.toml")

# If smg is being bumped, use the bumped version as the sync target
smg_target_version="$smg_version"
for entry in ${NEEDS_BUMP[@]+"${NEEDS_BUMP[@]}"}; do
    IFS='|' read -r _name _path _dep_key _cur _level <<< "$entry"
    if [[ "$_name" == "smg" ]]; then
        smg_target_version=$(bump_version "$_cur" "$_level" "smg")
        break
    fi
done

# Collect sync mismatches: "label|file|type|current_version"
NEEDS_VERSION_SYNC=()

for entry in "${SMG_VERSION_SYNC[@]}"; do
    IFS='|' read -r label file type <<< "$entry"

    case "$type" in
        cargo)
            file_version=$(get_crate_version "$file")
            ;;
        pyproject)
            file_version=$(grep -m1 '^version' "$file" | sed 's/.*"\([^"]*\)".*/\1/')
            ;;
        helm)
            file_version=$(get_helm_chart_version "$file")
            ;;
        workflow)
            file_version=$(get_workflow_smg_version "$file")
            ;;
    esac

    if [[ "$file_version" != "$smg_target_version" ]]; then
        echo -e "  ${YELLOW}!${NC} ${BOLD}$label${NC} ($file) — v$file_version, expected v$smg_target_version"
        NEEDS_VERSION_SYNC+=("$label|$file|$type|$file_version")
        issues=$((issues + 1))
    else
        echo -e "  ${GREEN}✓${NC} ${BOLD}$label${NC} ($file) — v$file_version"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Summary:${NC} $changed crate(s) with changes, $clean unchanged; $py_changed Python package(s) with changes, $py_clean unchanged"

if [[ "$issues" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All versions consistent.${NC}"
    exit 0
fi

echo -e "${RED}${BOLD}$issues issue(s) found.${NC}"

# ---------------------------------------------------------------------------
# Phase 2: Offer to fix
# ---------------------------------------------------------------------------
n_bump=${#NEEDS_BUMP[@]}
n_ws=${#NEEDS_WS_SYNC[@]}
n_py=${#NEEDS_PY_BUMP[@]:-0}
n_vsync=${#NEEDS_VERSION_SYNC[@]:-0}
total_fixes=$(( n_bump + n_ws + n_py + n_vsync ))
if [[ "$total_fixes" -eq 0 ]]; then
    exit 1
fi

echo ""
echo -e "${BOLD}Proposed fixes:${NC}"

for entry in ${NEEDS_BUMP[@]+"${NEEDS_BUMP[@]}"}; do
    IFS='|' read -r name path dep_key current_version level <<< "$entry"
    new_version=$(bump_version "$current_version" "$level" "$name")
    echo -e "  $(bump_label "$level") $name v$current_version → v$new_version ($path/Cargo.toml)"
    if [[ "$dep_key" != "-" ]]; then
        echo -e "       sync workspace Cargo.toml $dep_key → v$new_version"
    fi
done

if [[ "$n_ws" -gt 0 ]]; then
    for entry in "${NEEDS_WS_SYNC[@]}"; do
        IFS='|' read -r name dep_key crate_version ws_version path <<< "$entry"
        echo -e "  ${BLUE}sync${NC} workspace Cargo.toml $dep_key v$ws_version → v$crate_version"
    done
fi

for entry in ${NEEDS_PY_BUMP[@]+"${NEEDS_PY_BUMP[@]}"}; do
    IFS='|' read -r name path version_file current_version level <<< "$entry"
    new_version=$(bump_version "$current_version" "$level" "$name")
    echo -e "  $(bump_label "$level") $name v$current_version → v$new_version ($version_file)"
done

for entry in ${NEEDS_VERSION_SYNC[@]+"${NEEDS_VERSION_SYNC[@]}"}; do
    IFS='|' read -r label file type file_version <<< "$entry"
    echo -e "  ${BLUE}sync${NC} $label v$file_version → v$smg_target_version ($file)"
done

echo ""
read -rp "Apply fixes? [y/N] " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "No changes made."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 3: Apply fixes
# ---------------------------------------------------------------------------
echo ""
fix_failed=0

for entry in ${NEEDS_BUMP[@]+"${NEEDS_BUMP[@]}"}; do
    IFS='|' read -r name path dep_key current_version level <<< "$entry"
    new_version=$(bump_version "$current_version" "$level" "$name")

    # Bump crate Cargo.toml
    if set_crate_version "$path/Cargo.toml" "$new_version"; then
        echo -e "  ${GREEN}✓${NC} $path/Cargo.toml → v$new_version"
    else
        fix_failed=$((fix_failed + 1))
    fi

    # Sync workspace root Cargo.toml (read actual ws version in case it drifted)
    if [[ "$dep_key" != "-" ]]; then
        ws_old=$(get_workspace_dep_version "$dep_key")
        if set_workspace_dep_version "$dep_key" "$ws_old" "$new_version"; then
            echo -e "  ${GREEN}✓${NC} Cargo.toml $dep_key → v$new_version"
        else
            fix_failed=$((fix_failed + 1))
        fi
    fi
done

if [[ "$n_ws" -gt 0 ]]; then
    for entry in "${NEEDS_WS_SYNC[@]}"; do
        IFS='|' read -r name dep_key crate_version ws_version path <<< "$entry"
        if set_workspace_dep_version "$dep_key" "$ws_version" "$crate_version"; then
            echo -e "  ${GREEN}✓${NC} Cargo.toml $dep_key → v$crate_version"
        else
            fix_failed=$((fix_failed + 1))
        fi
    done
fi

for entry in ${NEEDS_PY_BUMP[@]+"${NEEDS_PY_BUMP[@]}"}; do
    IFS='|' read -r name path version_file current_version level <<< "$entry"
    new_version=$(bump_version "$current_version" "$level" "$name")

    if set_python_version "$version_file" "$new_version"; then
        echo -e "  ${GREEN}✓${NC} $version_file → v$new_version"
    else
        fix_failed=$((fix_failed + 1))
    fi
done

for entry in ${NEEDS_VERSION_SYNC[@]+"${NEEDS_VERSION_SYNC[@]}"}; do
    IFS='|' read -r label file type file_version <<< "$entry"
    case "$type" in
        cargo)
            if set_crate_version "$file" "$smg_target_version"; then
                echo -e "  ${GREEN}✓${NC} $file → v$smg_target_version"
            else
                fix_failed=$((fix_failed + 1))
            fi
            ;;
        pyproject)
            if set_python_version "$file" "$smg_target_version"; then
                echo -e "  ${GREEN}✓${NC} $file → v$smg_target_version"
            else
                fix_failed=$((fix_failed + 1))
            fi
            ;;
        helm)
            if set_helm_chart_version "$file" "$file_version" "$smg_target_version"; then
                echo -e "  ${GREEN}✓${NC} $file → v$smg_target_version"
            else
                fix_failed=$((fix_failed + 1))
            fi
            ;;
        workflow)
            if set_workflow_smg_version "$file" "$file_version" "$smg_target_version"; then
                echo -e "  ${GREEN}✓${NC} $file → v$smg_target_version"
            else
                fix_failed=$((fix_failed + 1))
            fi
            ;;
    esac
done

echo ""
if [[ "$fix_failed" -gt 0 ]]; then
    echo -e "${RED}${BOLD}$fix_failed fix(es) failed. Check output above.${NC}"
    exit 1
fi
echo -e "${GREEN}${BOLD}Done. Re-run to verify.${NC}"
