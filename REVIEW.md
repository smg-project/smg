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

Two reviewers run on every pull request, drafts included, that a user with write access opens from a
branch of this repository: Claude (`.github/workflows/claude-code-review.yml`) and Codex
(`.github/workflows/codex-code-review.yml`, prompts in `.github/codex/`). A pull request from a fork
never gets either, and the commands below are ignored on one and from anyone without write access.
Both post inline comments with the markers above; each requests changes when it finds something 🔴
Important and approves when it finds nothing. A later push that settles the Important findings lifts
the request: Codex approves, Claude dismisses its own request on an incremental review and approves
on a full one. Each announces itself when it starts, Claude with its tracking comment and Codex with
a status note it updates with the outcome. Claude posts as `claude[bot]`, Codex as `smg-codex-review[bot]`, a GitHub App the
organization owns. The API keys and the App's key never enter a job: they stay in the cluster's
secrets, and a sidecar in the runner pod serves the API calls and mints Codex's one-hour posting
token, so nothing in a pull request can reach the keys. A pull request older than a change to these
workflows keeps the old copy for its thread replies and verdicts until it takes `main`. A pull
request that edits one of the two workflow files gets no Claude run from the edited workflow: the
action's token exchange requires the workflow file to match the copy on `main`, and skips with a
warning otherwise.

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
