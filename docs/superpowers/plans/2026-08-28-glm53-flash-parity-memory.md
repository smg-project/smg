# GLM-5.3-Flash Preprocessing Parity and Memory Plan

> **Required execution skill:** Use `executing-plans` to implement this plan task by task. The current session is already on the user-requested `shared/glm-5.3-flash` branch, so no new branch or commit should be created unless the user asks.

**Goal:** Make the GLM-5.3-Flash image/video processor match the upstream Hugging Face geometry and pixel-transform semantics, while bounding video preprocessing memory to the final patch tensor plus one temporal group of RGB frames.

**Architecture:** Move aligned-canvas geometry and RGB temporal patchification into a model-neutral `spatial_merge` helper. Keep GLM-specific defaults, config interpretation, and output metadata in `glm53_flash.rs`. Preserve legacy `preprocessor_config.json` precedence, but fall through to `processor_config.json.image_processor` when the legacy file is unreadable or malformed.

**Tech Stack:** Rust, `image`, `ndarray`, existing multimodal transform helpers, Cargo unit tests; Python `uv` only for the existing tokenizer-bundle regression test.

**Specification:** The parity oracle is the generated Hugging Face GLM5-Next image/video processors: [image processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/image_processing_glm5_next.py), [video processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/video_processing_glm5_next.py), and the published [GLM-5.3-Flash processor config](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/processor_config.json).

## Global Constraints

- Do not put GLM-specific policy into `QwenVLProcessorBase`.
- Do not change registry/model-token behavior outside the bugs listed below.
- Preserve output tensor layout: `[grid_t * grid_h * grid_w, 3 * temporal * patch^2]` with flattened axis order `T, merged-H, merged-W, merge-H, merge-W, channel, temporal, patch-H, patch-W`.
- Preserve valid legacy `preprocessor_config.json` precedence for compatibility.
- Use checked arithmetic for config-derived sizes and fail with `TransformError` instead of panicking.
- Do not install dependencies or create commits without explicit user authorization.

## Task 1: Extract and Correct Aligned-Canvas Geometry

**Files:**

- Create: `crates/multimodal/src/vision/processors/spatial_merge.rs`
- Modify: `crates/multimodal/src/vision/processors/mod.rs`
- Modify: `crates/multimodal/src/vision/processors/glm53_flash.rs`

- [x] Add unit tests that call geometry without allocating media:
  - 512 frames at 3840x2160 with the official video budget produces target `(644, 1120)` and grid `(256, 46, 80)`.
  - A one-frame 1x1 video uses the actual frame count for minimum-budget scaling and produces `(168, 168)`.
  - `patch_expand_factor = 2` participates in both alignment and `pixels_per_token`.
  - Zero patch/merge/temporal/expand values and impossible budgets return errors rather than panic.
- [x] Run `cargo test -p llm-multimodal glm53_flash --lib` and confirm the new upstream-parity cases fail against the current analytic square-root/floor implementation.
- [x] Implement upstream-equivalent banker's temporal rounding and binary-search `fit_within_budget` in the neutral helper.
- [x] Change `Params::from_config` to return `Result`, use checked multiplication, and compute `pixels_per_token = temporal * (patch * merge * expand)^2`.
- [x] Route GLM image/video geometry through the helper without changing output metadata.
- [x] Re-run the focused tests and expect all GLM tests to pass.

## Task 2: Stream RGB Frames Directly Into Patches

**Files:**

- Modify: `crates/multimodal/src/vision/processors/spatial_merge.rs`
- Modify: `crates/multimodal/src/vision/processors/glm53_flash.rs`

- [x] Add a parity test comparing `preprocess_video` and `preprocess_video_rgb` for an odd number of non-uniform RGB frames, including resize/pad and temporal duplication.
- [x] Add a helper-level regression test proving an already-sized `RgbFrameRef` remains borrowed rather than copied.
- [x] Run the focused tests. Treat parity as an output-safety net; use the borrowed-frame test plus final static diff inspection as the memory-path evidence.
- [x] Implement a generic borrowed RGB frame source for `DynamicImage` and `RgbFrameRef`.
- [x] Prepare only one temporal group at a time; duplicate the final frame by reference for an incomplete group.
- [x] Write normalized values directly into the final patch vector using a 256-entry per-channel lookup table. Remove `tensors.to_vec()`, `stack_batch`, the permuted owned ndarray, and the all-frame `DynamicImage` conversion.
- [x] Validate frame dimensions and byte lengths before indexing.
- [x] Re-run the focused tests and verify exact tensor/grid/token parity between the two entry points.

## Task 3: Honor Pixel Transform Configuration

**Files:**

- Modify: `crates/multimodal/src/vision/processors/spatial_merge.rs`
- Modify: `crates/multimodal/src/vision/processors/glm53_flash.rs`

- [x] Add processor tests for:
  - `do_rescale = false` retaining 0..255 pixel values when normalization is disabled.
  - A custom `rescale_factor` being applied before normalization.
  - The GLM default remaining `do_rescale = true` with factor `1/255` when the field is absent.
- [x] Run the focused tests and confirm the current unconditional `/255` path fails at least the first two cases.
- [x] Build the patchifier LUT from `do_rescale`, `rescale_factor`, `do_normalize`, mean, and std in Hugging Face order.
- [x] Reject non-finite scale/mean/std and zero std with `TransformError`.
- [x] Re-run the focused tests and expect all pixel-transform cases to pass.

## Task 4: Fall Back From a Broken Legacy Config

**Files:**

- Modify: `model_gateway/src/routers/grpc/multimodal/config.rs`

- [x] Extend `load_image_preprocessor_config_preserves_legacy_precedence_and_falls_back` with a malformed legacy file while a valid nested image processor exists.
- [x] Run `cargo test -p smg load_image_preprocessor_config --lib`; expect the new assertion to fail before the code change. If the environment still lacks `protoc`, record that as an environment blocker and verify this file through formatting/diff review.
- [x] Return a valid legacy config immediately; otherwise continue to `processor_config.json.image_processor` instead of returning `None` merely because the legacy path exists.
- [x] Re-run the focused gateway test when build prerequisites permit.

## Task 5: Full Verification and Scope Review

**Files:**

- Review all files changed by Tasks 1-4.

- [x] Run `cargo fmt --all -- --check`.
- [x] Run `cargo test -p llm-multimodal --quiet`.
- [x] Run `PYTHONPATH=grpc_servicer uv run --no-project --with pytest python -m pytest -q grpc_servicer/tests/test_tokenizer_bundle.py`.
- [x] Run the repository completion harness if one appears; pre-commit passed except its all-features Clippy hook, which requires unavailable system OpenCV. Targeted default-feature Clippy passed with `-D warnings`.
- [x] Run `git diff --check` and review `git diff --stat origin/main..HEAD` plus the working-tree diff for unrelated changes.
- [x] Confirm no full-video `to_vec()`/`DynamicImage` collection or ndarray-wide patchification copy remains in the GLM RGB video path.
- [x] Run the gateway fallback test with explicit existing `protoc` and temporary `uv`-provided well-known proto includes; the assertion passed.
