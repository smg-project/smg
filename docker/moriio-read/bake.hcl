# Overlay for the accompanying exact upstream bake file; run from the f9 source export.
variable "MORIIO_RECIPE_DIR" { default = "." }
target "test" {
  dockerfile = "${MORIIO_RECIPE_DIR}/Dockerfile.rocm"
  contexts = {
    "runtime" = "docker-image://rocm/vllm-ci:runtime-rocm-c67ca115e7d64f7d@sha256:1e16e61342d903a2e509ed05c19a65d4e8176db2dcf4dc0e03cb2ee9d3ba335b",
    "stack" = "docker-image://rocm/vllm-ci:stack-rocm-f0ff2bd31804e4af@sha256:482e36b7f45062dbd065095c59b78f4e2f3a8e0f0c7a971f52a492618b969e0d",
    "test-deps" = "docker-image://rocm/vllm-ci:test-deps-rocm-c67ca115e7d64f7d@sha256:4933c01636bc52d817c53a08d9b95b491c156ae45908f20a226b7c028fdc0ebd"
}
  args = {
    "AITER_NIGHTLY" = "",
    "AITER_ROCM_ARCH" = "gfx942",
    "AITER_SPEC" = "",
    "ARCH" = "gfx942",
    "CI_ARCH" = "gfx942",
    "PYTORCH_ROCM_ARCH" = "gfx942",
    "SCCACHE_BUCKET_NAME" = "",
    "SOURCE_DATE_EPOCH" = "0",
    "STACK" = "rocm",
    "VLLM_VERSION" = ""
}
}
target "test-smoke" {
  dockerfile = "${MORIIO_RECIPE_DIR}/Dockerfile.rocm"
  contexts = {
    "runtime" = "docker-image://rocm/vllm-ci:runtime-rocm-c67ca115e7d64f7d@sha256:1e16e61342d903a2e509ed05c19a65d4e8176db2dcf4dc0e03cb2ee9d3ba335b",
    "stack" = "docker-image://rocm/vllm-ci:stack-rocm-f0ff2bd31804e4af@sha256:482e36b7f45062dbd065095c59b78f4e2f3a8e0f0c7a971f52a492618b969e0d",
    "test-deps" = "docker-image://rocm/vllm-ci:test-deps-rocm-c67ca115e7d64f7d@sha256:4933c01636bc52d817c53a08d9b95b491c156ae45908f20a226b7c028fdc0ebd"
}
  args = {
    "AITER_NIGHTLY" = "",
    "AITER_ROCM_ARCH" = "gfx942",
    "AITER_SPEC" = "",
    "ARCH" = "gfx942",
    "CI_ARCH" = "gfx942",
    "PYTORCH_ROCM_ARCH" = "gfx942",
    "SCCACHE_BUCKET_NAME" = "",
    "SOURCE_DATE_EPOCH" = "0",
    "STACK" = "rocm",
    "VLLM_VERSION" = ""
}
}
