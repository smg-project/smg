#!/bin/bash
# Reject partial/provider-incomplete configurations before package installation.
set -euo pipefail
if [ "${CHECK_AMD_RESPONSES:-}" != true ] || \
   [ "${CHECK_RESPONSES_EVIDENCE:-}" != true ] || \
   [ "${CHECK_FINAL_ENV:-}" != true ]; then
    echo "AMD Responses requires amd_responses, responses_evidence and final_environment_check" >&2
    exit 1
fi
if [ "${CHECK_GPU_VENDOR:-}" != amd ] || [ "${CHECK_ENGINE:-}" != sglang ] || \
   [ "${CHECK_GPU_TIER:-}" != 1 ] || [ "${CHECK_SETUP_AGENTIC:-}" != true ]; then
    echo "AMD Responses requires the SGLang/tier1/AMD/provider-service profile" >&2
    exit 1
fi
case "${CHECK_TEST_DIRS:-}" in
    e2e_test/responses|e2e_test/responses/) ;;
    *) echo "AMD Responses requires the complete Responses directory" >&2; exit 1 ;;
esac
if [ -n "${CHECK_TEST_FILTER:-}" ] || [ -n "${CHECK_WIRE:-}" ]; then
    echo "AMD Responses requires the original complete selection and default wire" >&2
    exit 1
fi
case "${CHECK_MIN_SELECTED:-}" in
    ''|*[!0-9]*) echo "AMD Responses requires an integer min_selected >=128" >&2; exit 1 ;;
esac
if ! [ "$CHECK_MIN_SELECTED" -ge 128 ] 2>/dev/null; then
    echo "AMD Responses requires min_selected >=128" >&2
    exit 1
fi
