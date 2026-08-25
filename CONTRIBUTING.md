# Contributing to SMG

Thank you for your interest in contributing to Shepherd Model Gateway. This
document is the front door. The detailed guides live in the
[smg-docs](https://github.com/smg-project/smg-docs) repository under
[`src/lib/content/contributing/`](https://github.com/smg-project/smg-docs/tree/main/src/lib/content/contributing).

- **Code of Conduct**: [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) — applies to
  every interaction in this repo and in community spaces.
- **Governance**: For how the project is governed — roles, decision making, and how maintainers are added — see [GOVERNANCE.md](GOVERNANCE.md).
- **How to contribute code**: [Contributing guide](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/index.md)
- **Development environment**: [Development setup](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/development.md)
- **Code style**: [Code style](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/code-style.md)
- **Review guidelines**: [`REVIEW.md`](./REVIEW.md)
- **PR template**: [`.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md)

---

## Quick start

```bash
# 1. Fork on GitHub, then clone your fork
git clone git@github.com:<your-user>/smg.git
cd smg

# 2. Install toolchain
rustup toolchain install nightly
rustup component add rustfmt --toolchain nightly
rustup component add clippy rustfmt

# 3. Install pre-commit hooks (enforces rustfmt, clippy, DCO, no-AI-attribution, branch naming)
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# 4. Create a branch (must match <type>/<desc> or <username>/<desc>)
git checkout -b feat/my-change

# 5. Build and test
cargo build
cargo test
```

Full setup details are in the [development setup guide](https://github.com/smg-project/smg-docs/blob/main/src/lib/content/contributing/development.md).

---

## The pre-PR gate

Every PR must pass these five checks **locally** before requesting review:

| # | Command | Expectation |
|---|---------|-------------|
| 1 | `cargo +nightly fmt --all` | No output (silent success) |
| 2 | `cargo clippy --all-targets --all-features -- -D warnings` | Zero warnings, zero errors |
| 3 | `cargo test` | `test result: ok` with 0 failures |
| 4 | `make python-dev` *(if `config/types.rs`, `protocols/`, or `bindings/` changed)* | Successful compilation |
| 5 | Commit format | Conventional commit, DCO sign-off present, no AI attribution |

"Probably passes" is not passing. Paste the output or re-run.

Check 2's `--all-features` enables the `opencv-video` feature, which links
system OpenCV. Install it once with `bash scripts/install_opencv.sh`, or lint
without the feature using `cargo clippy --workspace --all-targets -- -D warnings`.

---

## Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body explaining why>

Signed-off-by: Your Name <your.email@example.com>
```

- **Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`
- **Scope**: the crate or subsystem (`mesh`, `grpc_client`, `worker`, `protocols`, …)
- **One logical change per commit.** Prefer many small commits to one mega-commit.
- **Every commit must be DCO-signed.** Use `git commit -s`. The
  [DCO](https://developercertificate.org/) certifies that you wrote the code or
  have the right to submit it.
- **No AI attribution.** `Co-Authored-By: Claude` / `noreply@anthropic.com` and
  similar are rejected by the `no-ai-co-author` pre-commit hook.

---

## Pull requests

- **Fill in the [PR template](./.github/PULL_REQUEST_TEMPLATE.md)** — especially
  the `Test Plan` section. "Ran `cargo test`" is not a test plan; name the
  scenarios the reviewer can reproduce.
- **Keep PRs small.** Aim for ≤400 lines changed. Above that, split or coordinate
  with reviewers in advance.
- **One concern per PR.** Refactor *or* feature — not both.
- **Link the issue.** `Closes #1234` or `Refs: #1234`.
- **Respond to review comments with a commit SHA and a one-line reason.**
  Example: `Fixed in abc1234 — capped total_chunks at 1024 before allocation`.
  Silence or "Fixed!" makes reviewers re-hunt your work.

---

## Using code agents (Claude Code, Cursor, Copilot, etc.)

Agents are welcome and useful. Three ground rules:

1. **You own the PR, not the agent.** Read every line before opening.
2. **Show the gate output.** The agent must paste real `cargo fmt` / `clippy` /
   `test` output, not "I have run the tests."
3. **No AI attribution in commits, PR bodies, or review replies.** The hook
   will reject it; so will the reviewer.

---

## Reviewing

- Use the severity markers from [`REVIEW.md`](./REVIEW.md): 🔴 Important, 🟡 Nit,
  🟣 Pre-existing.
- Cite `file:line` in every substantive comment.
- Run `/smg:review-pr` to map changed files to subsystem checklists before you
  start.
- Approve small clean PRs fast; the faster turnaround, the fewer giant PRs
  reviewers face later.

---

## Running CI on your own runners

Every self-hosted `runs-on` label in `.github/workflows/` reads from a repository
variable with the upstream label as its fallback, e.g.

```yaml
runs-on: ${{ vars.SMG_RUNNER_CPU || 'k8s-runner-cpu' }}
```

Upstream sets none of these variables, so CI keeps using the labels above with no
change. A fork that runs CI on its own runners only has to set the variables it
needs (**Settings → Secrets and variables → Actions → Variables**) — no workflow
edits, so upstream syncs stay conflict-free.

| Variable | Upstream fallback | Used for |
| --- | --- | --- |
| `SMG_RUNNER_CPU` | `k8s-runner-cpu` | lint, build, unit tests, summaries |
| `SMG_RUNNER_DOCKER` | `cpu-e5` | docker / engine image build and push |
| `SMG_RUNNER_GPU` | `k8s-runner-gpu` | GPU jobs with no fixed GPU count |
| `SMG_RUNNER_GPU_1` | `1-gpu-h100` | 1-GPU e2e jobs |
| `SMG_RUNNER_GPU_2` | `2-gpu-h100` | 2-GPU e2e jobs |
| `SMG_RUNNER_GPU_4` | `4-gpu-h100` | 4-GPU e2e and benchmark jobs |
| `SMG_RUNNER_GPU_8` | `8-gpu-h200` | 8-GPU benchmark jobs |

GitHub-hosted labels (`ubuntu-latest`, `macos-latest`) are left as-is — every
fork already has those.

### Container-mode GPU runners

The GPU jobs declare their container image through a variable:

```yaml
container: ${{ vars.SMG_CI_GPU_CONTAINER_IMAGE }}
```

Unset, the expression is the empty string and the job runs directly on the runner
— what upstream does today. Forks whose GPU runners are container-only (an
Actions Runner Controller scale set with `containerMode: kubernetes` refuses jobs
without a `container:`) set it to a CUDA-capable image and the same jobs run
unchanged.

Only the GPU jobs carry it: they are the ones that need a CUDA toolchain, and
keeping the seam narrow keeps the blast radius small. It can be extended to the
CPU jobs if someone needs it.

**What the image has to provide.** When the variable is set, the GPU job's steps
run *inside* the image, so it is not enough for it to be CUDA-capable:

| requirement | needed by |
| --- | --- |
| `bash`, `pip`, `pytest` | all four GPU jobs |
| `python3` | `go-bindings-benchmark` |
| a usable `docker` client with daemon access | `benchmarks`, `go-bindings-benchmark` |
| CUDA runtime matching the engine under test | all four |

A slim CUDA base image will fail during setup rather than at test time. Pin the
image **by digest** (`repo/image@sha256:...`) rather than by tag: the value is
read at job start, so a mutable tag silently changes the CI environment with no
workflow change to review.

### Benchmark workflows

The `benchmark-*` workflows are gated so a fork does not silently burn its
runners on them:

```yaml
if: github.repository == 'smg-project/smg' || vars.SMG_RUN_BENCHMARKS == 'true'
```

They stay off in a fork until you set `SMG_RUN_BENCHMARKS` to `true`. Only
`benchmark-radix-tree` runs on the self-hosted CPU pool, so it also needs
`SMG_RUNNER_CPU`; the other four are on `ubuntu-latest` and need nothing else.

The `release-*`, `nightly-*` and `stale` workflows are deliberately **not**
opt-in — a fork should not publish artifacts or manage upstream's issues.

---

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities. Contact the
maintainers privately — see [`CODEOWNERS`](./.github/CODEOWNERS) for the current
maintainer list, or reach out in the `#security` channel of the
[Lightseek Slack](https://slack.lightseek.org).

---

## Getting help

- **Questions**: [GitHub Discussions](https://github.com/smg-project/smg/discussions)
- **Bugs**: [GitHub Issues](https://github.com/smg-project/smg/issues/new)
- **Chat**: [Slack](https://slack.lightseek.org) · [Discord](https://discord.gg/wkQ73CVTvR)

---

## License

By contributing to SMG, you agree that your contributions will be licensed under
the [Apache License 2.0](./LICENSE).
