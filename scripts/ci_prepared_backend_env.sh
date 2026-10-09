#!/bin/bash
# Shared by the runner image build and CI installers. A prepared venv is
# writable in the runner Pod's own image layer; never put it on a shared mount.

SMG_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

smg_backend_recipe() {
    local backend="$1"
    local files=(ci_prepared_backend_env.sh ci_setup_python_venv.sh
        "ci_install_${backend}.sh" ci_ensure_python_headers.sh ci_apt_mirror.sh ci_retry.sh)
    if [ "$backend" = vllm ]; then
        files+=(ci_install_flashinfer_jit_cache.sh)
    fi
    # Relative filenames make the stamp independent of the checkout location.
    (cd "$SMG_ENV_SCRIPT_DIR" || exit; sha256sum "${files[@]}" | sha256sum | cut -d ' ' -f 1)
}

smg_prepared_env_matches() {
    local backend="$1" venv="$2" manifest
    [ -x "$venv/bin/python" ] && [ -f "$venv/.smg-ci-recipe" ] || return 1
    manifest="$(smg_prepared_env_manifest "$backend" "$venv" 2>/dev/null)" || return 1
    [ "$(cat "$venv/.smg-ci-recipe")" = "$manifest" ] || return 1
}

smg_prepared_env_manifest() {
    smg_backend_recipe "$1" || return 1
    # Both transports are baked. Missing or changed engine/native packages
    # invalidate reuse even if a previous recipe stamp was left behind.
    "$2/bin/python" - "$1" <<'PY'
import importlib.metadata as metadata
import sys

packages = [sys.argv[1], "torch", "flashinfer-python", "flashinfer-jit-cache",
            "nixl", "mooncake-transfer-engine-cuda13"]
packages += sorted(name for d in metadata.distributions()
                   if (name := d.metadata.get("Name", "")).lower().replace("_", "-").startswith("nixl-cu"))
for name in packages:
    print(name, metadata.version(name))
PY
}

smg_mark_prepared_env() {
    # CUDA PyPI wheels need not have a +cu suffix in their package version.
    "${VIRTUAL_ENV:?}/bin/python" -c \
        'import torch; assert torch.version.cuda is not None, "prepared environments require CUDA-enabled PyTorch"' || return 1
    smg_prepared_env_manifest "$1" "${VIRTUAL_ENV:?}" > "$VIRTUAL_ENV/.smg-ci-recipe"
}
