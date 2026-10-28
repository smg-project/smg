# The release process

A release is cut from `main` onto its own branch and moves there by pull requests, so `main`
never pauses for it. Nothing publishes by itself: a push to a release branch runs CI, and the
artifacts come from the release workflows a maintainer dispatches against the release tag.

The tooling is `scripts/release/release.sh`, reached through `make` (run the targets from a
checkout that has this directory, `main` for instance; `cut`, `bump` and `pick` leave you on the
branch they created):

| step | command | what it does | who |
|---|---|---|---|
| cut | `make release-cut VERSION=1.14.0-rc.1 [FROM=<sha>] [PUSH=1]` | creates `release/1.14` from `FROM` (default `origin/main`), commits the version bump to `1.14.0-rc.1` on it, prints the push command | a maintainer with push rights to `release/**` |
| gate | `make release-check VERSION=1.14.0-rc.1` | the release gate: what the repository can read (the versions at the branch head, its CI, the tag, main's drift) and the lines the release owner fills (the four model-provider verifiers) | the release owner |
| pick | `make release-pick RELEASE=1.14 SHA=<sha>[,<sha>]` | `git cherry-pick -x -s` of commits of `main` onto a topic branch off `release/1.14`, pushed, with its pull request against the release branch; a conflict stops with the resolution steps, `make release-pick-pr RELEASE=1.14` finishes it | anyone with push rights; the pull request merges like any other |
| status | `make release-status RELEASE=1.14` | what is on `main` and not on the branch (same patch or picked with `-x` excluded), the branch's own commits, its tags, the open pull requests against it | anyone |
| candidate or final | `make release-bump VERSION=1.14.0-rc.2` (or `VERSION=1.14.0`) | the version bump on a topic branch off `release/1.14` and its pull request against the release branch; the body carries the release gate | the release owner |
| tag | `make release-tag VERSION=1.14.0-rc.1 [PUSH=1]` | checks the versions at the branch head against the previous tag, creates the annotated tag `v1.14.0-rc.1` with a `Signed-off-by` line, prints the push command | a maintainer with tag rights (the tag ruleset) |
| publish | `make release-publish-plan VERSION=1.14.0-rc.1` | prints the workflow dispatches to run against the tag and the artifact names they produce; runs nothing | the release owner dispatches them |
| notes | `make release-notes VERSION=1.14.0 [SINCE=v1.13.0] > notes.md` | the release notes skeleton: the merged pull requests since the previous final, by type, and the Docker image list; prints the `gh release create` line | the release owner |
| main follows | `make release-bump-main VERSION=1.14.0` | after a final: a pull request that moves `main`'s version files to the release's | the release owner |

`PUSH=1` is needed to push a release branch (`cut`) or a tag (`tag`); `PUSH=0` keeps a topic
branch local (`bump`, `pick`, `bump-main` otherwise push it and open the pull request with `gh`).
Every commit, pick and tag carries the caller's git identity and sign-off.

## Versions

- A version is `X.Y.Z` or `X.Y.Z-rc.N`; its release branch is `release/X.Y`.
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
  alone: a changed crate whose version any release tag carries (the comparison tag's, or one
  another line published, above or below it) is unbumped, and a proposed version a release tag
  carries steps to the next free patch (crates.io and PyPI take a version once). A version
  published without a release tag is not seen: the publish workflows skip a version the
  registry holds, and the release owner reads their output while two lines are live.
- `main` keeps the last final release's versions. After a final, `make release-bump-main` opens
  the pull request that moves them: the gateway and its mirrors to the release's version, every
  other crate or package to the release's version where that is higher than main's (crates.io
  and PyPI never take a version twice, so the next cut starts above what was published). The
  nightly and dev builds derive their versions from main's plain `X.Y.Z`, which is why main
  carries no suffix. A cut from a `main` that reads below the previous final's version is refused:
  merge that pull request first, or cut from a commit after it. `main` never goes back: after a
  patch of an older line (v1.14.1 once v1.15.0 is out), `make release-bump-main` keeps the
  gateway's higher version and moves only the crates and packages the patch raised.

## What publishes, and when

Nothing, on its own. The `release-pypi`, `release-helm`, `release-crates`, `release-grpc` and
`release-engine-docker` workflows run on `workflow_dispatch` only; `release-docker` builds the
image on a push to `main` or `release/**` that touches the Dockerfile and pushes it on a
dispatch only. The release owner dispatches each against the tag once the gate is green;
`make release-publish-plan VERSION=<x.y.z>` prints the exact commands and the artifact names
(the gateway image's `latest` moves only for a final; a crate publishes when its exact version
is not on crates.io yet; a PyPI version that is complete, wheel and sdist, is skipped, and
`-f packages=proto|servicer|both` on `release-grpc` names a package outright for a retry). The nightly prerelease and the
CI images are unchanged by this process.

## The gate

A candidate or a final is tagged when, on the branch head:

1. CI is green on the head commit itself: a successful run of the `PR Test (SMG)` matrix for
   that commit (it runs on every push to `release/**` and every pull request against it, as for
   `main`; a documentation-only merge runs none, so `make release-check` names the dispatch,
   `gh workflow run pr-test-rust.yml --ref release/<x.y>`);
2. the versions are consistent (`make release-check` reads them; `make release-tag` refuses
   otherwise);
3. the four model-provider verifiers (DeepSeek, GLM, Kimi, MiniMax) ran against the candidate
   build: the release owner records one link each. `make release-check` prints the four lines
   to fill; the final's bump pull request carries them in its "Release gate" section.

## The steps of a release

1. `make release-cut VERSION=1.14.0-rc.1 FROM=<sha>`: the branch and its bump commit, locally.
   Read the commit, then `git push origin refs/heads/release/1.14:refs/heads/release/1.14`
   (or `PUSH=1`). CI runs on the branch; nothing is published. `main` is not touched.
2. `make release-check VERSION=1.14.0-rc.1`, then `make release-tag VERSION=1.14.0-rc.1` and
   its push (`PUSH=1`, by a maintainer with tag rights).
3. `make release-publish-plan VERSION=1.14.0-rc.1` lists the dispatches for the candidate's
   artifacts; `make release-notes VERSION=1.14.0-rc.1 > notes.md` and the printed
   `gh release create ... --prerelease` line for the GitHub release.
4. Fixes land on `main` first, then `make release-pick RELEASE=1.14 SHA=<sha>` brings them over
   through a pull request against the branch; `make release-status RELEASE=1.14` shows what is
   still on `main` only.
5. The next candidate: `make release-bump VERSION=1.14.0-rc.2`, merge, step 2 and 3 again.
6. The final: `make release-bump VERSION=1.14.0` (its pull request body carries the gate with
   the four verifier records), merge, `make release-tag VERSION=1.14.0`, the dispatches of
   `make release-publish-plan VERSION=1.14.0`, `make release-notes VERSION=1.14.0`.
7. `make release-bump-main VERSION=1.14.0`: the pull request that moves `main`'s versions.
8. A patch release later: `make release-bump VERSION=1.14.1` on the same branch, picks, step 2
   onwards; a candidate first (`1.14.1-rc.1`) when it needs one.

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
  against one branch. The directory is the release branch's; `bump-main` moves version files only.
- A pick conflicts: the tool leaves you on the pick branch and prints the three steps
  (`git add`, `git cherry-pick --continue`, `make release-pick-pr RELEASE=<x.y>`);
  `git cherry-pick --abort` gives it up.
- A candidate is bad: the next candidate is a bump pull request after the fixes are picked;
  tags are never moved.
- A dispatched workflow fails: dispatch it again against the same tag; every publish step skips
  what exists already (PyPI with `--skip-existing`, the crate action by version, the image tags
  are overwritten with the same build); a PyPI version left with one of its two files is selected
  again and `--skip-existing` uploads the missing one.
- `make release-tag` refuses: the branch head does not read the version (merge the bump first)
  or `scripts/check_release_versions.py --check <previous tag>` finds a crate changed and not
  bumped (a bump pull request fixes it).
