# The release process

A release is cut from `main` onto its own branch and moves there by pull requests, so `main`
never pauses for it. Nothing publishes by itself: a push to a release branch runs CI, and the
artifacts come from the release workflows a maintainer dispatches against the release tag.

The tooling is `scripts/release/release.sh`, reached through `make` (run the targets from a
checkout that has this directory, `main` for instance; `cut` leaves you where you were with the
release branch behind you, `bump` and `pick` leave you on the topic branch they created):

| step | command | what it does | who |
|---|---|---|---|
| plan | `make release-plan [TAG=v1.13.0] [VERSION=1.14.0-rc.1]` | before a cut: the versions the cut's bump writes against the previous final, quietly, nothing changed: the crates and packages by their conventional commits, the gateway and its mirrors to `VERSION` (default: the first candidate of `main`'s development version, `1.14.0-rc.1` when `main` reads `1.14.0-dev`). On `main` every crate changed since the last release reads as unbumped by design: the cut bumps them, so `make check-versions` there lists them all as issues | anyone |
| cut | `make release-cut VERSION=1.14.0-rc.1 [FROM=<sha>] [PUSH=1]` | creates `release/1.14` from `FROM` (default `origin/main`), commits the version bump to `1.14.0-rc.1` on it, prints the push command; then the pull request that moves `main` above the branch, to `1.15.0-dev` (pushed and opened with `PUSH=1`, printed otherwise); before anything is made, the cut checks that the move's topic is free on the remote and, with `PUSH=1`, that `gh` is there and no open pull request carries it | a maintainer with push rights to `release/**` |
| gate | `make release-check VERSION=1.14.0-rc.1` | the release gate: what the repository can read (the versions at the branch head, its CI, the tag, main's drift) and the lines the release owner fills (the four model-provider verifiers) | the release owner |
| pick | `make release-pick RELEASE=1.14 SHA=<sha>[,<sha>]` | `git cherry-pick -x -s` of commits of `main` onto a topic branch off `release/1.14`, pushed, with its pull request against the release branch; a conflict stops with the resolution steps, `make release-pick-pr RELEASE=1.14 PICK=pick/1.14-<sha>` finishes it from `main` (the pick branch may lack these targets) | anyone with push rights; the pull request merges like any other |
| status | `make release-status RELEASE=1.14` | what is on `main` and not on the branch (same patch or picked with `-x` excluded), the branch's own commits, its tags, the open pull requests against it | anyone |
| candidate or final | `make release-bump VERSION=1.14.0-rc.2` (or `VERSION=1.14.0`) | the version bump on a topic branch off `release/1.14` and its pull request against the release branch; the body carries the release gate | the release owner |
| tag | `make release-tag VERSION=1.14.0-rc.1 [PUSH=1]` | checks the versions at the branch head against the previous tag, creates the annotated tag `v1.14.0-rc.1` with a `Signed-off-by` line, prints the push command | a maintainer with tag rights (the tag ruleset) |
| publish | `make release-publish-plan VERSION=1.14.0-rc.1` | prints the workflow dispatches to run against the tag and the artifact names they produce; runs nothing | the release owner dispatches them |
| notes | `make release-notes VERSION=1.14.0 [SINCE=v1.13.0] > notes.md` | the release notes draft: `scripts/release/NOTES-TEMPLATE.md` with the merged pull requests since the previous final seeding Highlights, Notable fixes and Upgrade notes from their "## Release note" sections, the packages and images at the tag under Packages, and the list by type under What's Changed; the narrative stays as marked placeholders (see [Writing the notes](#writing-the-notes)); `scripts/release/release.sh notes --check notes.md` refuses a draft that still carries one | the release owner |
| main follows | `make release-bump-main VERSION=1.14.0` | after a final: a pull request that moves `main`'s crates and packages to at least the release's; the gateway keeps its development version (it moves to the next one only when `main` still reads below the release) | the release owner |
| ship | `make release-ship VERSION=1.14.1 [DRY_RUN=1] [FROM=<step>]` | the candidate, final or patch in one run on an existing release branch: the gate's CI rows for the commit the bump builds on, the bump (pushed to the branch when the rules allow the caller a push, else its pull request, waited for), the tag, the notes, the GitHub release and the publish dispatches; a rerun skips what is there ([Ship](#ship-the-release-in-one-run)) | the release owner, with push or bypass rights for the direct bump |

`PUSH=1` is needed to push a release branch (`cut`, with the pull request that moves `main`) or a
tag (`tag`); `PUSH=0` keeps a topic branch local (`bump`, `pick`, `bump-main` otherwise push it and
open the pull request with `gh`); the ship pushes and opens on its own.
Every commit, pick and tag carries the caller's git identity and sign-off. Before the cut is
pushed, the read-only commands (`check`, `status`, `publish-plan`, `notes`) read the local
`release/<x.y>` and say so, which rehearses a release end to end on a clone; `bump`, `pick`,
`pick-pr`, `tag` and `bump-main` (before the tag) need the branch on `origin`.

## Versions

- A version is `X.Y.Z` or `X.Y.Z-rc.N`; its release branch is `release/X.Y`. Between cuts `main`
  reads `X.Y.Z-dev`, the next development version (the trunk is next): above every release
  before it, below the `X.Y.Z` line it leads. Cargo and Helm read it as a semver pre-release,
  Python tooling as `X.Y.Z.dev0` (as it reads `1.14.0-rc.1` as `1.14.0rc1`), so `smg --version`,
  the nightly wheels (`X.Y.Z.devYYYYMMDD`) and every build of `main` identify above the release
  branch they are a superset of.
- Every version file carries the one string as written: `model_gateway/Cargo.toml`, the two
  binding manifests, `bindings/python/pyproject.toml`, `bindings/python/src/smg/version.py`,
  `deploy/helm/smg/Chart.yaml` and the `SMG_RELEASE_TAG` line of the engine-image workflow.
  Python tooling normalizes a candidate on its own: `1.14.0-rc.1` is published as `1.14.0rc1`
  (PEP 440; `pip` installs it only with `--pre`), Cargo, Helm and the image tags keep it as is.
- A bump (`cut`, `bump`) is `make bump-version` for the gateway's files, then
  `scripts/check_release_versions.py --fix <previous tag>` for the crates and packages changed
  since that tag (their version follows the conventional commits since it) and the mirrors,
  then `cargo update --workspace` for `Cargo.lock`. The previous tag is the branch's own newest
  tag when it has one (the last candidate, for the next one), else the highest final below the
  version (the previous release's, on its own branch: no release tag is reachable from `main`
  under this process, and the checker diffs `tag..HEAD`, so none has to be), else the newest
  tag reachable from the branch (a release made before this process). Two live release lines
  share the crate versions, so the checker reads every release tag, not the comparison tag
  alone: a crate whose version a release tag carries with other code is unbumped (the
  comparison tag's version with changes since it; a version another line published, above or
  below it, that `release-bump-main` moved to `main` or that a cut found there), and a
  proposed version a release tag carries steps to the next free patch (crates.io and PyPI take
  a version once). A version a tag carries with this very code needs nothing. A version
  published without a release tag is not seen by the tags: `make release-check` reads crates.io
  and PyPI for every crate and package at the branch head and fails on a version they hold
  already with changes after it (the commits since the registry's own publish time of that
  version, or since its bump when the registry gives none; crates.io is read through the sparse
  index cargo reads, one component at a time), and `make release-publish-plan` marks such versions
  (`REGISTRY=0` skips the reads).
- `main` moves the moment a release branch is cut: `make release-cut VERSION=1.14.0-rc.1` opens
  the pull request that moves the gateway and its mirrors to `1.15.0-dev` and every crate and
  package to at least the version the cut gave the branch (the same numbers, so the next cut's
  bump computes from what the registries will hold; crates.io and PyPI never see `main`'s
  versions, since `main`'s artifacts are not published). After a final, `make release-bump-main`
  opens the pull request that moves the crates and packages the release raised since the cut
  (its candidates' and final's bumps) to at least the release's; the gateway keeps its
  development version, and moves to the one after the release (`1.15.0-dev` for v1.14.0) only
  when `main` still reads below it (a line cut before this rule, a move not merged). The nightly
  and dev-wheel builds read a `-dev` version as the number to build (`1.15.0.devYYYYMMDD`) and a
  plain `X.Y.Z` as the patch above it. A cut from a `main` that reads below the previous final's
  version is refused: merge the pull request that moves `main` first, or cut from a commit after
  it. `main` never goes back: after a patch of an older line (v1.14.1 once `main` reads
  `1.16.0-dev`), `make release-bump-main` keeps the gateway's version and moves only the crates
  and packages the patch raised.

## What publishes, and when

Nothing, on its own. The `release-pypi`, `release-helm`, `release-crates`, `release-grpc`,
`release-go` (the Go module tag `crates/grpc_client/go/v<version>`) and `release-engine-docker`
workflows run on `workflow_dispatch` only; `release-grpc` and `release-go` also refuse a dispatch
whose ref is not a tag (`release-go`: not the version's tag), since a branch can move between a
dispatch and its retry; `release-docker` builds the image on a push to `main` or `release/**`
that touches the Dockerfile and pushes it on a dispatch only. The release owner dispatches each
against the tag once the gate is green; `make release-publish-plan VERSION=<x.y.z>` prints the
exact commands and the artifact names (the gateway image's `latest` moves only for a final; a
crate publishes when its exact version is not on crates.io yet; a PyPI version that is complete,
wheel and sdist, is skipped, and `-f packages=proto|servicer|both` on `release-grpc` names a
package outright for a retry). The Go bindings module, `github.com/smg-project/smg/bindings/golang`,
is published by the tag `bindings/golang/v<version>` at the release commit: the plan prints the
`git tag` and `git push` for it, run by a maintainer with tag rights beside `v<version>` until
`release-go` pushes it too (`make release-ship` dispatches workflows only, so it is run by hand
either way). The plan prints that item only where `bindings/golang/go.mod` at the ref declares that
path: a tag at an older release would name a `go.mod` that `go get` refuses, so a plan for an older
line leaves it out and says why; the first such tag is 1.16.0's. A release candidate is never published to PyPI or crates.io:
`release-pypi`, `release-crates` and `release-grpc` refuse a ref whose name carries `-rc.` (the
job is skipped, and its first step says why when it is reached anyway),
`make release-publish-plan VERSION=1.14.0-rc.1` leaves the three out and says so,
`make release-check` reports the rule, and the candidate ships as its images, its chart, its two Go
module tags and its GitHub pre-release. The nightly prerelease and the CI images are unchanged by
this process.

## The gate

A candidate or a final is tagged when, on the branch head:

1. CI is green on the head commit itself: a successful run of the `PR Test (SMG)` matrix for
   that commit (it runs on every push to `release/**` and every pull request against it, as for
   `main`; a documentation-only merge runs none, so `make release-check` names the dispatch,
   `gh workflow run pr-test-rust.yml --ref release/<x.y>`). A bump commit changes nothing but
   version strings (`scripts/ci_version_only.py` is the judge), so the workflow lets it inherit
   the GPU lanes, the card tier, from the commit it was made on when that commit's push or
   dispatch run passed (a pull request's run can pass with the card lanes skipped, so it stands
   for nothing), and its own run covers the host rows alone; `make release-check` then reads the
   parent's run for the card tier while the head's own checks are pending, provided the parent is
   green on every check of its own (none red, none pending) besides that run, and says so;
2. the versions are consistent (`make release-check` reads them; `make release-tag` refuses
   otherwise), and no crate or package sits at a version crates.io or PyPI hold already with
   changes after it (`make release-check` reads the registries; such a version is skipped by
   its publish dispatch and the release would ship none of those changes: a bump pull request
   against the release branch moves it);
3. the four model-provider verifiers (DeepSeek, GLM, Kimi, MiniMax) ran against the candidate
   build: the release owner records one link each. `make release-check` prints the four lines
   to fill; the final's bump pull request carries them in its "Release gate" section.

## The steps of a release

Steps 2, 3, 5, 6 and 8 below are one command each with `make release-ship VERSION=<version>`
([Ship](#ship-the-release-in-one-run)); the steps spell out what it does.

1. `make release-cut VERSION=1.14.0-rc.1 FROM=<sha>`: the branch and its bump commit, and the
   topic branch `chore/main-versions-1.15.0-dev` that moves `main` above it, both locally; you
   stay on the branch you ran it from. Read the commits (`git show --stat refs/heads/release/1.14`,
   `git show --stat refs/heads/chore/main-versions-1.15.0-dev`), then
   `git push origin refs/heads/release/1.14:refs/heads/release/1.14` and the printed push and
   `gh pr create` lines of the topic (or `PUSH=1` for all of it). CI runs on the branch; nothing
   is published. `main` moves by its pull request alone.
2. `make release-check VERSION=1.14.0-rc.1`, then `make release-tag VERSION=1.14.0-rc.1` and
   its push (`PUSH=1`, by a maintainer with tag rights).
3. `make release-publish-plan VERSION=1.14.0-rc.1` lists the dispatches for the candidate's
   artifacts; `make release-notes VERSION=1.14.0-rc.1 > notes.md` starts the notes draft
   ([Writing the notes](#writing-the-notes)), and the printed `gh release create ... --prerelease`
   line publishes it once `release.sh notes --check notes.md` passes.
4. Fixes land on `main` first, then `make release-pick RELEASE=1.14 SHA=<sha>` brings them over
   through a pull request against the branch; `make release-status RELEASE=1.14` shows what is
   still on `main` only.
5. The next candidate: `make release-bump VERSION=1.14.0-rc.2`, merge, step 2 and 3 again.
6. The final: `make release-bump VERSION=1.14.0` (its pull request body carries the gate with
   the four verifier records), merge, `make release-tag VERSION=1.14.0`, the dispatches of
   `make release-publish-plan VERSION=1.14.0`, the notes drafted during the cycle, refreshed
   against `make release-notes VERSION=1.14.0`, reviewed, and checked before `gh release create`.
7. `make release-bump-main VERSION=1.14.0`: the pull request that moves `main`'s crates and
   packages to at least the release's (the gateway keeps `1.15.0-dev`).
8. A patch release later: `make release-bump VERSION=1.14.1` on the same branch, picks, step 2
   onwards; a candidate first (`1.14.1-rc.1`) when it needs one.

## Ship: the release in one run

`make release-ship VERSION=1.14.1` (a candidate: `VERSION=1.14.1-rc.1`) takes an existing release
branch from the bump to the dispatches in one run. Every step reads the repository first and
skips what is there, so the same command is run again after an interruption, and `FROM=<step>`
(`release.sh ship <version> --from <step>`) starts at a step by name:

1. `preflight`: the gate's CI rows for the commit the bump builds on (the branch head, or the
   bump's parent when the branch reads the version already). A red row (a failed or cancelled
   run, a read that failed, a matrix that ended without a success or never ran on that commit)
   stops the ship; pending rows are printed and do not hold it: the bump changes version files
   only and ships on its parent's run. `--from bump` skips the preflight once the state was read
   by hand. The four verifier lines are printed for the release owner. The ship needs `gh`
   logged in from its first step: it reads the rules and the runs and opens the release with it.
2. `bump`: the bump commit (as `release-bump` makes it) on `chore/release-<version>`, the
   versions and the registries checked at it before anything leaves the clone; then the branch
   moves to it. By a push when the repository's rules allow the caller one: no branch protection
   of the classic kind, and no pull-request or status-check rule on the branch, or the caller on
   the always-bypass list of every ruleset that carries one. Otherwise by the bump pull request,
   which the ship opens and waits for (`MERGE_WAIT` minutes, 120 by default, a read every
   `MERGE_POLL` seconds, 30). A push the remote refuses all the same falls back to the pull
   request. `BUMP_PUSH=1` pushes without reading the rules (a bypass by team or role, which the
   ship does not read); `BUMP_PUSH=0` always opens the pull request. A rerun finds an open bump
   pull request and waits for it.
3. `tag`: `v<version>` at the branch head, as `release-tag` makes it, pushed. A tag on the
   repository at the head is kept; one anywhere else stops the ship (tags never move).
4. `notes`: `release-notes` rendered once into `.git/release/notes-v<version>.md` (kept across
   reruns, so it can be edited), or the file `NOTES=<path>` names. Notes that are not ready (an
   empty file, a first line that is not the template's for the release, a placeholder
   `release.sh notes --check` still finds) stop the ship before anything is published. A fresh
   render carries the template's author placeholders, so the first run stops here: write the
   notes (see [Writing the notes](#writing-the-notes)), then rerun the ship (`--from notes`).
   Once the release exists (a rerun after an interruption in a later step), the step stands
   down: the published notes stand, with or without `NOTES`. The engine-image list fills in once
   those images exist: `scripts/release_docker_notes.sh v<version>` and `gh release edit` after
   the dispatches.
5. `release`: `gh release create --verify-tag` with the notes, `--prerelease` for a candidate and
   `--latest` for a final; an existing release is kept.
6. `publish`: the dispatches of `release-publish-plan` against the tag, each one once: a
   dispatch whose run exists on the tag and did not fail is skipped, a failed or cancelled run
   is dispatched again, and the runs on the tag are listed at the end.

`DRY_RUN=1` (`--dry-run`) prints each step with the values it reads (the CI rows, the rules, the
tag, the release, the plan's dispatch lines and their runs) and writes nothing, in the clone or
on the repository; `REGISTRY=0` skips the registry reads as for `check`.

The bump reaches the branch without a review round only when the rules let the caller push to
it: a release manager on the bypass list of the ruleset that protects the release branches, or a
branch that carries no such rule. Without that, the bump costs the review round `release-bump`
costs today, and the ship waits for it. The first candidate of a line is cut by `release-cut`
(it also moves `main`); from then on the ship is the one command for every candidate, the final
and the patches.

## Writing the notes

The published notes are a narrative, not a list. `scripts/release/NOTES-TEMPLATE.md` is their
skeleton (the title and opening, Before you upgrade, Highlights with one `###` narrative each and
how the release was validated, Notable fixes by area, Upgrade notes by kind (a Storage subsection
opens the list only when a pull request declares a storage change), Packages, Known limits, then
What's Changed), as the published notes are laid out, and `make release-notes` renders it:

- every merged pull request's release note (one or two sentences on what a user gets, under a
  `## Release note` heading or a `Release note:` line in the pull request body) seeds the
  candidates: features and performance changes under Highlights, fixes under Notable fixes, each
  under the area of its subject's scope (gateway, servicers and engines, Symphony, multimodal,
  tokenizer, metrics and observability, bindings and crates, Helm and deploy); a pull request
  whose note says `none` seeds nothing, and one without a note seeds its subject, marked for the
  author;
- a pull request that declares a breaking change (`feat(x)!:` or a `BREAKING CHANGE` line), a
  removed or renamed flag or setting, a changed default or a configuration that now refuses to
  start seeds Upgrade notes under the subsection its declaration suggests, with the declaration
  named;
- Packages reads the publish plan's crates and packages at the tag against the
  previous final (new crates, moved versions), the engine images' bases from the workflow's table
  and the published image list; What's Changed is the list by type, as before;
- the opening paragraphs, Before you upgrade, every narrative, How this release was validated and
  Known limits are `<!-- AUTHOR: ... -->` comments that say what goes there.

The pull request bodies are read through `gh` in batches and kept under `NOTES_CACHE`
(`~/.cache/smg-release/pull-requests`, one file per number); `NOTES_REFRESH=1` reads every body
again (a note added to a pull request after the first draft), as deleting the directory does;
without `gh`, or offline, every candidate falls back to its subject, marked, and the draft still
renders.

The flow: draft at the first candidate (`make release-notes VERSION=1.14.0-rc.1 > notes.md` reads
the tag, or the branch head before it), write the narratives during the cycle as the candidates
land, and before the final render again (`make release-notes VERSION=1.14.0 > fresh.md`) and diff
its candidate lists and What's Changed against the draft for what landed since. The release
manager reviews the draft before the tag. `scripts/release/release.sh notes --check notes.md`
refuses a file that still carries an author comment or a template token, line by line, so neither
a raw list nor a half-written draft reaches `gh release create`: run it before the release, and in
any step that publishes the notes; `release.sh ship` runs it at its `notes` step and stops on what
it finds.

## Branch protection a maintainer sets

- The `release/**` branches take the same ruleset as `main`: pull requests only, two approving
  reviews with a code owner, the required checks (`finish`, `DCO`, `PR Title & Commit
  Messages`), squash merges. Branch creation stays allowed, so a maintainer can push the cut.
- Tags `v*` are created by the maintainers the tag ruleset lets through.
- The workflows that gate pull requests against `main` run against `release/**` too
  (`PR Test (SMG)`, the fixture parity, the MLX tests, the benchmark gates, the engine-image
  dry run); `CI Tooling CPU Tests` and `PR Validation` run on every pull request already.

## When something goes wrong

- A pick's pull request squash-merges under its title alone (the repository's setting), so the
  picked commits' `(cherry picked from commit …)` trailers never reach the branch: the tool
  records each picked original as a file under `scripts/release/picks/` on the pick branch
  (named by its sha, holding its subject), one commit the squash keeps, and `status`, `check`
  and `notes` read the branch's picks from them (the notes name the originals under their
  types). Files of different names never conflict, so several pick pull requests can be open
  against one branch. The directory is the release branch's; the cut's main-move pull request and
  `bump-main` move version files only.
- A pick conflicts: the tool leaves you on the pick branch and prints the three steps
  (`git add`, `git cherry-pick --continue`, then from `main`
  `make release-pick-pr RELEASE=<x.y> PICK=<pick branch>`, which checks the branch out itself: a
  line cut before a change to these targets does not carry them); `git cherry-pick --abort`
  gives it up.
- A candidate is bad: the next candidate is a bump pull request after the fixes are picked;
  tags are never moved.
- A dispatched workflow fails: dispatch it again against the same tag; every publish step skips
  what exists already (PyPI with `--skip-existing`, the crate action by version, the image tags
  are overwritten with the same build); a PyPI version left with one of its two files is selected
  again and `--skip-existing` uploads the missing one.
- `make release-tag` refuses: the branch head does not read the version (merge the bump first)
  or `scripts/check_release_versions.py --check <previous tag>` finds a crate changed and not
  bumped (a bump pull request fixes it).
