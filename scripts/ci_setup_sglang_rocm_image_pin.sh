#!/bin/bash
# Read the authoritative verifier pin as data, with Bash builtins only.
read_sglang_rocm_image_pin() {
    local verifier_source=${1:?verifier source required}
    local image_declaration='^[[:space:]]*IMAGE([[:space:]:=]|$)'
    local image_double='^IMAGE[[:space:]]*=[[:space:]]*"([^"]+)"[[:space:]]*$'
    local image_single="^IMAGE[[:space:]]*=[[:space:]]*'([^']+)'[[:space:]]*$"
    local image_reference='^[[:alnum:]][[:alnum:]_.:/-]*:[[:alnum:]_.-]+@sha256:[a-f0-9]{64}$'
    local image_count=0 image_pin= image_line
    if [ ! -f "$verifier_source" ] || [ ! -r "$verifier_source" ]; then
        echo "invalid verifier IMAGE pin: missing or unreadable source" >&2
        return 1
    fi
    while IFS= read -r image_line || [ -n "$image_line" ]; do
        if [[ "$image_line" =~ $image_declaration ]]; then
            image_count=$((image_count + 1))
            if [[ "$image_line" =~ $image_double ]] || [[ "$image_line" =~ $image_single ]]; then
                image_pin="${BASH_REMATCH[1]}"
            else
                echo "invalid verifier IMAGE pin: expected one quoted immutable literal" >&2
                return 1
            fi
        fi
    done < "$verifier_source"
    if [ "$image_count" -ne 1 ] || [[ ! "$image_pin" =~ $image_reference ]]; then
        echo "invalid verifier IMAGE pin: expected one quoted immutable literal" >&2
        return 1
    fi
    printf '%s\n' "$image_pin"
}
