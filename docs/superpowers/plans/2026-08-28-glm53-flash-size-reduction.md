# GLM-5.3-Flash Size Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the GLM-5.3-Flash correctness and memory fixes while reducing the code diff against the branch merge-base to 900–1100 added lines, excluding plan documents.

**Architecture:** Remove the model-neutral `spatial_merge` module because GLM is its only consumer. Keep a single GLM-owned streaming RGB pipeline that prepares one temporal group at a time, writes patches directly into the final vector, and shares that path between `DynamicImage` and borrowed `RgbFrameRef` inputs.

**Tech Stack:** Rust, `image`, `ndarray`, Cargo tests and Clippy.

**Spec:** `docs/superpowers/plans/2026-08-28-glm53-flash-parity-memory.md`

## Global Constraints

- Count code with `git diff --numstat $(git merge-base origin/main HEAD)` plus untracked code files; exclude `docs/superpowers/plans/`.
- Finish with 900–1100 added code lines and no more than the existing 23 deleted base-branch lines unless a deletion is required by the refactor.
- Preserve Hugging Face resize parity for long-aspect video, actual-frame minimum scaling, and `patch_expand_factor` budgets.
- Preserve `do_rescale`, `rescale_factor`, `do_normalize`, malformed legacy-config fallback, and borrowed-RGB video behavior.
- Do not change Qwen processor policy or add a new shared abstraction with only one caller.

---

### Task 1: Replace the one-caller shared module

**Files:**
- Modify: `crates/multimodal/src/vision/processors/glm53_flash.rs`
- Modify: `crates/multimodal/src/vision/processors/mod.rs`
- Delete: `crates/multimodal/src/vision/processors/spatial_merge.rs`

**Interfaces:**
- Consumes: `DynamicImage`, `RgbFrameRef`, `PreProcessorConfig`, and existing RGB resize helpers.
- Produces: the unchanged `VisionPreProcessor` implementation for `Glm53FlashProcessor`.

- [x] **Step 1: Keep the current GLM regression tests as the behavior oracle**

Run:

```bash
cargo test -p llm-multimodal glm53_flash --lib
```

Expected: all current GLM tests pass before structural changes.

- [x] **Step 2: Move only the required primitives into the GLM processor**

Implement compact GLM-local equivalents of:

```rust
trait RgbSource {
    fn dimensions(&self) -> (u32, u32);
    fn rgb(&self) -> Result<Cow<'_, [u8]>, TransformError>;
}

fn encode<F: RgbSource>(
    frames: &[F],
    budget_frames: usize,
    params: Params,
    config: &PreProcessorConfig,
) -> Result<Patches, TransformError>;
```

The encoder must resize and pad only the current temporal group, repeat the final frame for an incomplete group, and append directly in merge/channel/temporal/patch order.

- [x] **Step 3: Remove the shared module declaration and file**

Delete `mod spatial_merge;` and `spatial_merge.rs`; verify no references remain:

```bash
rg -n "spatial_merge" crates/multimodal/src
```

Expected: no matches.

### Task 2: Retain compact regression coverage

**Files:**
- Modify: `crates/multimodal/src/vision/processors/glm53_flash.rs`
- Test: `model_gateway/src/routers/grpc/multimodal/config.rs`

**Interfaces:**
- Consumes: the GLM-local encoder from Task 1.
- Produces: regression coverage for geometry, layout, RGB borrowing, rescale semantics, and invalid configuration.

- [x] **Step 1: Consolidate related assertions into table-style tests**

Keep explicit expected values:

```rust
assert_eq!(params.geometry(512, 2160, 3840)?.target, (644, 1120));
assert_eq!(params.geometry(1, 1, 1)?.target, (168, 168));
assert_eq!(expanded.geometry(expanded.temporal, 1, 1)?.target, (224, 224));
```

Keep byte-scale expectations `10.0`, `5.0`, and `10.0 / 255.0`, plus zero-std and NaN-factor rejection.

- [x] **Step 2: Run focused regressions**

Run:

```bash
cargo test -p llm-multimodal glm53_flash --lib
cargo test -p smg load_image_preprocessor_config --lib
```

Expected: all selected tests pass.

### Task 3: Enforce size and quality gates

**Files:**
- Verify all modified Rust files.

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: a measured, formatted, linted implementation within the requested size range.

- [x] **Step 1: Measure code additions**

Run:

```bash
base=$(git merge-base origin/main HEAD)
git diff --numstat "$base"
```

Add untracked Rust files manually and exclude plan documents. Expected: 900–1100 added code lines.

- [x] **Step 2: Run formatting and whitespace checks**

Run:

```bash
cargo +nightly fmt --all -- --check
git diff --check
```

Expected: both commands exit 0.

- [x] **Step 3: Run multimodal tests and targeted Clippy**

Run:

```bash
cargo test -p llm-multimodal --quiet
cargo clippy -p llm-multimodal --all-targets -- -D warnings
```

Expected: tests and Clippy pass without enabling the optional OpenCV feature.

## Verification Evidence

- Code diff against the merge-base: 1095 additions and 23 deletions, excluding plan documents.
- `cargo test -p llm-multimodal --quiet`: passed all unit and integration suites.
- `cargo clippy -p llm-multimodal --all-targets -- -D warnings`: passed.
- Focused gateway fallback and Python tokenizer-bundle regressions: passed.
- The repository-wide all-features Clippy hook still requires system OpenCV, which is unavailable in this local environment.
