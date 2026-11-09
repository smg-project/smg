# Code Review Guidelines

## Skills

Use these skills during review:

1. **`/smg:review-pr`** — SMG subsystem-aware checklist. Map changed files to sections, check each applicable section.
2. **`pr-review-toolkit:silent-failure-hunter`** — Spawn as agent on changed files. Finds swallowed errors, inappropriate fallbacks, missing error propagation.
3. **`pr-review-toolkit:pr-test-analyzer`** — Spawn as agent. Checks if tests adequately cover new/changed functionality.
4. **`pr-review-toolkit:type-design-analyzer`** — Spawn as agent only if new types are introduced. Reviews invariants and encapsulation.

## Severity

Prefix EVERY inline comment with one of these markers:

| Marker | Severity | Meaning |
|--------|----------|---------|
| 🔴 | **Important** | A bug that should be fixed before merging |
| 🟡 | **Nit** | A minor issue, worth fixing but not blocking |
| 🟣 | **Pre-existing** | A bug in the codebase NOT introduced by this PR |

Example:

> 🔴 **Important**: Off-by-one in loop bound. `i < len` should be `i < len - 1` because the last element is the sentinel. This will cause an index-out-of-bounds panic when the input array is non-empty.

After posting all inline comments, write a brief summary with a count per severity level.

## Bots

Two reviewers run on every non-draft pull request that a user with write access opens from a
branch of this repository: Claude (`.github/workflows/claude-code-review.yml`) and Codex
(`.github/workflows/codex-code-review.yml`, prompts in `.github/codex/`). A pull request from a
fork never gets either, and the commands below are ignored on one and from anyone without
write access. Both post inline comments with the markers above, and each approves when it finds
nothing 🔴 Important. The workflows run from `main`, and the API keys never enter a job: they
stay in the cluster's secrets and are served by a sidecar in the runner pod that the reviewer's
own process cannot read, so nothing in a pull request can reach them.

- `@claude review` or `@codex review` in a PR comment runs a full review on demand.
- A human reply in a thread Codex opened, or a comment that mentions `@codex`, gets an answer
  from Codex in that thread.
- Each bot grades the other's root findings in-thread with 🤝 Agree, ⚠️ Disagree or ❓ Unsure,
  once per finding, so a disputed comment shows both views.

## Focus on

- Logic errors and bugs that would break production
- Security vulnerabilities (injection, auth bypass, data leaks)
- Missing error handling (silent failures, swallowed errors)
- Broken cross-references to other code or docs
- Incorrect defaults or config values
- `clone()` in gRPC streaming hot paths (per-token response processing)
- `serde(rename)` that doesn't match OpenAI/Anthropic API spec field names
- Worker registry mutations without proper locking (DashMap vs bare HashMap)
- Silent fallbacks to `None`/default when config validation should fail loudly

## Domain knowledge

- Config changes are the #1 source of bugs — they touch CLI args (main.rs), types.rs, cli/configure.rs (the two conversion paths), Python bindings, and Go SDK
- The gRPC pipeline has 10 constructors in pipeline.rs — changes to shared stages must work for all of them
- HTTP and gRPC routers are two separate code paths that must handle the same API contract
- PD disaggregation adds dual-dispatch complexity on top of regular routing

## Skip

- Formatting-only changes
- Documentation-only PRs (`docs/**`)
- Dependency version bumps with no code changes
