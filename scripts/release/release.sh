#!/usr/bin/env bash
# The release process: a release branch per minor version, cut from main and moved by
# pull requests, so main never pauses for a release.
#
#   release.sh cut <version> [<from>]      cut release/<x.y> from <from> (default <remote>/main) and commit the
#                                          version bump to <version> on it; then the pull request that moves main
#                                          above the branch, to the next development version (1.15.0-dev for a
#                                          1.14 cut: the trunk is next); nothing is pushed unless PUSH=1
#   release.sh bump <version>              the next candidate or the final on an existing release/<x.y>: a bump
#                                          commit on a topic branch and its pull request against the release branch
#   release.sh pick <x.y> <sha>[,<sha>..]  cherry-pick commits of main (-x -s) onto a topic branch off
#                                          release/<x.y> and open the pull request; a conflict stops with the steps
#   release.sh pick-pr <x.y> [<branch>]    after a resolved conflict: push the pick branch (the current one, or the
#                                          named one, checked out first) and open its pull request
#   release.sh status <x.y>                what is on main and not on release/<x.y>, the branch's own commits,
#                                          its tags, the open pull requests against it
#   release.sh tag <version>               check the versions at release/<x.y>'s head and tag it v<version>
#                                          (annotated, signed off; pushed only with PUSH=1)
#   release.sh notes <version> [<since>]   the release notes draft on stdout: scripts/release/NOTES-TEMPLATE.md with the
#                                          merged pull requests since the previous final (or <since>) seeding Highlights,
#                                          Notable fixes and Upgrade notes from their "## Release note" sections, and
#                                          listed by type under What's Changed; the narrative paragraphs stay as marked
#                                          placeholders for the author
#   release.sh notes --check <file>        refuse notes that still carry a placeholder (before gh release create)
#   release.sh bump-main <version>         after a final: a pull request that moves main's crates and packages to
#                                          at least the release's; the gateway keeps its development version, or
#                                          moves to the one after <version> when main reads below the release
#   release.sh publish-plan <version>      the publishing steps for v<version>: the workflow dispatches a maintainer
#                                          runs against the tag and the artifact names they produce (prints, runs nothing)
#   release.sh check <version>             the release gate for v<version>: what the repository can read (the versions
#                                          at the branch head, its CI, the tag, main's drift) and the lines the release
#                                          owner fills (the four model-provider verifiers)
#   release.sh ship <version> [--dry-run] [--from <step>]
#                                          the release in one run, on an existing release/<x.y>: the preflight (the CI of
#                                          the commit the bump builds on, read and reported: a red row stops the ship, a
#                                          pending one does not), the bump commit (pushed to the branch when no rule the
#                                          caller cannot bypass requires a pull request, else its pull request, waited
#                                          for), the tag, the notes, the GitHub release (a prerelease for a candidate,
#                                          latest for a final) and the publish plan's dispatches, with the runs listed;
#                                          every step reads the remote first and skips what is there, so an interrupted
#                                          run is rerun as it stands; --from <step> starts at that step (preflight, bump,
#                                          tag, notes, release, publish); --dry-run prints each step with the values it
#                                          reads and writes nothing
#
# No push publishes anything: a bump on a release branch runs CI; the artifacts (wheels, crates, images, the chart)
# come from the release-* workflows dispatched by hand against the release tag, as publish-plan lists them and the ship
# dispatches them. A candidate (X.Y.Z-rc.N) is never published to PyPI or crates.io: publish-plan leaves those dispatches
# out, and release-pypi, release-crates and release-grpc refuse a ref named -rc.
#
# <version> is X.Y.Z or X.Y.Z-rc.N; the release branch is release/X.Y. Every version file carries the one
# string as written (Python tooling normalizes 1.14.0-rc.1 to 1.14.0rc1 on its own). Between cuts main reads
# X.Y.Z-dev, the next development version: above every release before it, below the X.Y.Z line it leads
# (a semver pre-release for cargo and Helm, X.Y.Z.dev0 once Python tooling normalizes it).
#
# Environment: REMOTE (origin); PUSH (cut and tag push nothing unless PUSH=1, the cut's main-move pull request
# included; bump, pick and bump-main push their topic branch and open the pull request unless PUSH=0; the ship pushes);
# REGISTRY=0 (check, publish-plan and the ship do not read crates.io and PyPI); NOTES_CACHE (where notes keeps the pull request
# bodies it read, ~/.cache/smg-release/pull-requests by default), NOTES_REFRESH=1 (notes reads every body again);
# GH, CARGO, CURL, PYTHON, MAKE (the tools).
# The ship's: NOTES (an authored notes file, instead of the one it renders); BUMP_PUSH=1 (push the bump to the release
# branch without reading the rules) or 0 (always its pull request); MERGE_WAIT (minutes it waits for the bump pull
# request's merge, 120) and MERGE_POLL (seconds between two reads of it, 30).
# The commits, the picks and the tag carry the caller's git identity and sign-off.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 # the version checker is imported from the tree; no scripts/__pycache__ left behind

REMOTE=${REMOTE:-origin}
GH=${GH:-gh}
CARGO=${CARGO:-cargo}
CURL=${CURL:-curl}
PYTHON=${PYTHON:-python3}
MAKE=${MAKE:-make}
# The pull request bodies notes reads, one file per number: a second render of the same release reads no body twice.
NOTES_CACHE=${NOTES_CACHE:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/smg-release/pull-requests}
VERSION_RE='^([0-9]+)\.([0-9]+)\.([0-9]+)(-rc\.([0-9]+))?$'
RELEASE_RE='^[0-9]+\.[0-9]+$'
GATEWAY_MANIFEST=model_gateway/Cargo.toml
# The originals picked into a release branch, one file per original named by its sha and holding its subject, written by
# the pick's pull request: the repository squash-merges a pull request under its title alone, so the picked commits'
# `(cherry picked from commit …)` trailers never reach the branch; these files do, and files of different names never
# conflict when several pick pull requests are open against one branch.
PICKS_DIR=scripts/release/picks

say() { printf '%s\n' "$*"; }
note() { printf 'release: %s\n' "$*" >&2; }
die() { printf 'release: %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }

root=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a git repository"
cd "$root"

# --- versions, branches, tags ---------------------------------------------------

version_ok() { [[ $1 =~ $VERSION_RE ]]; }
release_of() { # 1.14.0-rc.1 -> 1.14
  [[ $1 =~ $VERSION_RE ]] || die "not a version: '$1' (X.Y.Z or X.Y.Z-rc.N)"
  printf '%s.%s' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
}
release_ok() { [[ $1 =~ $RELEASE_RE ]] || die "not a release: '$1' (X.Y, the minor version of release/X.Y)"; }
is_final() { [[ $1 != *-* ]]; }
is_candidate() { [[ $1 == *-rc.* ]]; } # a version or a tag, X.Y.Z-rc.N with or without the v: never published to PyPI or crates.io
# Under sort -V a `~` sorts before the end of the string, so a pre-release spelled with it sorts below its final,
# as cargo and PEP 440 order them: 1.14.0 < 1.15.0-dev < 1.15.0-rc.1 < 1.15.0 (main reads X.Y.Z-dev between cuts).
sortable() { sed 's/-rc\./~rc./; s/-dev$/~dev/'; }
unsortable() { sed 's/~rc\./-rc./; s/~dev$/-dev/'; }
version_above() { # <a> <b>: a reads above b
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sortable | sort -V | tail -n 1 | unsortable)" = "$1" ]
}
next_dev_version() { # <version>: main's development version once that version's line is cut: X.(Y+1).0-dev
  [[ $1 =~ $VERSION_RE ]] || die "not a version: '$1' (X.Y.Z or X.Y.Z-rc.N)"
  printf '%s.%s.0-dev' "${BASH_REMATCH[1]}" "$((BASH_REMATCH[2] + 1))"
}

fetch() { git fetch --quiet --tags "$REMOTE" || die "cannot fetch $REMOTE"; }
has_ref() { git rev-parse --verify --quiet "$1^{commit}" >/dev/null 2>&1; }
remote_has_branch() { git ls-remote --exit-code --heads "$REMOTE" "$1" >/dev/null 2>&1; }
remote_has_tag() { git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$1" >/dev/null 2>&1; }
repo_slug() { # owner/name of the GitHub repository $REMOTE points at, from its URL, else as gh names it; empty when neither says
  local url slug=""
  url=$(git remote get-url "$REMOTE" 2>/dev/null || true)
  case $url in
    https://github.com/*) slug=${url#https://github.com/} ;;
    git@github.com:*) slug=${url#git@github.com:} ;;
    ssh://git@github.com/*) slug=${url#ssh://git@github.com/} ;;
  esac
  slug=${slug%/}; slug=${slug%.git}
  [ -n "$slug" ] || slug=$("$GH" repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || true)
  printf '%s' "$slug"
}
release_base() { # the remote-tracking release branch, fetched: what a bump, a pick or a tag builds on
  local branch="release/$1"
  has_ref "$REMOTE/$branch" || die "no $branch on $REMOTE: cut it first (release.sh cut <version> [<from>])"
  printf '%s/%s' "$REMOTE" "$branch"
}
release_ref() { # the release branch a read-only command reads: the remote's, else a local cut not pushed yet (a rehearsal), said so
  local branch="release/$1"
  if has_ref "$REMOTE/$branch"; then printf '%s/%s' "$REMOTE" "$branch"; return 0; fi
  # No remote-tracking ref is not the same as no branch on the remote (a single-branch clone, a narrowed fetch refspec):
  # the remote is asked, once, before a local branch is read as a rehearsal; a stale local cut must never pass for the remote's.
  # The branch joins the clone's fetch refspec (a one-off fetch would leave the remote-tracking ref behind the next merge).
  remote_has_branch "$branch" && die "$branch is on $REMOTE but not fetched here: git remote set-branches --add $REMOTE $branch && git fetch $REMOTE"
  has_ref "refs/heads/$branch" || die "no $branch on $REMOTE or here: cut it first (release.sh cut <version> [<from>])"
  note "$branch is not on $REMOTE yet: reading the local branch (the push: git push $REMOTE refs/heads/$branch:refs/heads/$branch)"
  printf 'refs/heads/%s' "$branch"
}
previous_tag() { # the newest release tag reachable from $1
  git describe --tags --abbrev=0 --match 'v[0-9]*' "$1" 2>/dev/null \
    || die "no release tag (v<x.y.z>) is reachable from $1"
}
previous_final_tag() { # the highest final tag below v<version>; empty when there is none
  { git tag --list 'v[0-9]*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; printf 'v%s\n' "$1"; } \
    | sort -uV | grep -B1 -x "v$1" | head -1 | grep -v -x "v$1" || true
}
own_tags() { # <ref>: the release tags made on the branch (reachable from <ref>, not from main), oldest version first
  # A candidate sorts below its final: a version sort puts `~` before the end of the string.
  comm -23 <(git tag --list 'v[0-9]*' --merged "$1" | sort) <(git tag --list 'v[0-9]*' --merged "$REMOTE/main" | sort) \
    | sortable | sort -V | unsortable
}
since_tag() { # <ref> <version>: the tag a bump or a check compares the versions against
  # The branch's own newest tag when it has one (made on the release branch: the last candidate);
  # else the highest final below <version>, which sits on the previous release's branch from the
  # second release on (no release tag is reachable from main under this process; the checker
  # diffs tag..HEAD, so the tag need not be an ancestor); else the newest tag reachable from
  # <ref>, a release made before this process.
  local ref=$1 version=$2 own final
  own=$(own_tags "$ref" | tail -n 1)
  if [ -n "$own" ]; then printf '%s' "$own"; return 0; fi
  final=$(previous_final_tag "$version")
  if [ -n "$final" ]; then printf '%s' "$final"; return 0; fi
  previous_tag "$ref"
}
manifest_version() { # the version a ref's gateway manifest carries (empty when it is not there)
  git show "$1:$GATEWAY_MANIFEST" 2>/dev/null | grep -m1 '^version = ' | sed 's/version = "\(.*\)"/\1/' || true
}
picked_into() { # <ref>: the originals picked into <ref>, space-bounded: the picks directory at <ref> and the trailers reachable from it
  # A pick's pull request records its originals under $PICKS_DIR on the branch (the squash merge keeps the title alone);
  # a pick landed as its own commits (a fast-forward, the older process) carries the `(cherry picked from commit …)`
  # trailers in its history. Both are read.
  printf ' %s ' "$({ git ls-tree --name-only "$1" -- "$PICKS_DIR/" 2>/dev/null | sed 's|.*/||'; git log --format=%b "$1" | grep -o 'cherry picked from commit [0-9a-f]*' | sed 's/.* //'; } | sort -u | tr '\n' ' ' || true)"
}
picked_in() { # <base>..HEAD: the originals named by the cherry-pick trailers of a pick branch, oldest first
  git log --reverse --format=%b "$1..HEAD" | grep -o 'cherry picked from commit [0-9a-f]*' | sed 's/.* //' || true
}
record_picks() { # <x.y> <base>: the originals of the picks in <base>..HEAD recorded under the picks directory, in one commit
  # The directory alone is staged and committed: a tracked file left modified by a conflict resolution stays out.
  local release=$1 base=$2 full subject added=0 word
  while read -r full; do
    [ -n "$full" ] || continue
    [ -f "$root/$PICKS_DIR/$full" ] && continue
    subject=$(git log -1 --format=%s "$full" 2>/dev/null || true)
    mkdir -p "$root/$PICKS_DIR"
    printf '%s\n' "${subject:-(not fetched)}" > "$root/$PICKS_DIR/$full"
    added=$((added + 1))
  done < <(picked_in "$base")
  [ "$added" -gt 0 ] || return 0
  [ "$added" -eq 1 ] && word=pick || word=picks
  git add -- "$root/$PICKS_DIR"
  git commit --quiet --signoff --message "chore(release): record $added $word into release/$release" \
    --message "The originals of this pull request's cherry-picks, one file each under $PICKS_DIR: the squash merge keeps the title alone, and status, check and notes read the branch's picks from them."
}

require_identity() {
  [ -n "${GIT_COMMITTER_EMAIL:-}" ] || git config --get user.email >/dev/null \
    || die "git user.name and user.email are not set: the commits and the tag carry your sign-off"
}
require_clean() { # [<advice>]: the bump's by default (its commit must carry only the bump)
  [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "the working tree has uncommitted changes: ${1:-commit them first (the bump commit must carry only the bump)}"
}
back_to() { # <start branch> <start commit>: the checkout the command started from (detached there when it was)
  if [ -n "$1" ]; then
    git switch --quiet "$1" || die "cannot switch back to $1"
  else
    git switch --quiet --detach "$2" || die "cannot switch back to $(git rev-parse --short "$2")"
  fi
}
drop_hint() { # <branch> <rerun>: the second way out of a command that stopped with version edits on <branch>: drop them and the
  # branch (whose existence refuses a rerun), then <rerun>
  printf '%s' "drop them (git restore --staged --worktree .), switch back to where you started and delete $1 (git branch -D $1): $2"
}
sign_off() { # "Name <email>", as git commit -s writes it
  local ident
  ident=$(git var GIT_COMMITTER_IDENT)
  ident=${ident% *}
  printf '%s' "${ident% *}"
}

# --- the bump ---------------------------------------------------------------------

bump_tree() { # <version> <since tag>: the gateway's files, then every crate, package and mirror, then Cargo.lock
  local version=$1 since=$2
  note "bump: make bump-version VERSION=$version"
  "$MAKE" --no-print-directory -C "$root" bump-version "VERSION=$version" >/dev/null
  note "bump: scripts/check_release_versions.py --fix $since (the crates and packages changed since $since, the mirrors)"
  "$PYTHON" "$root/scripts/check_release_versions.py" --fix "$since"
  "$PYTHON" "$root/scripts/check_release_versions.py" --check "$since" >/dev/null \
    || die "the versions are still inconsistent: scripts/check_release_versions.py --check $since"
  refresh_lock
}
refresh_lock() { # [<way out>]: Cargo.lock follows the workspace crates' versions; a failure names <way out> after its cause
  [ -f "$root/Cargo.lock" ] || return 0
  command -v "$CARGO" >/dev/null 2>&1 \
    || die "Cargo.lock needs a refresh and '$CARGO' is not on PATH (set CARGO=<path to cargo>)${1:+; $1}"
  note "bump: cargo update --workspace (Cargo.lock follows the crate versions)"
  "$CARGO" update --workspace --offline >/dev/null 2>&1 || "$CARGO" update --workspace >/dev/null \
    || die "cargo update --workspace failed: Cargo.lock does not follow the versions${1:+; $1}"
}
commit_all() { # <subject> <body> <finish> [<rerun>]: the tree's changes in one signed-off commit. A failure stops the command in
  # any context and names both ways out: <finish> (the lines that finish by hand, the commit among them once, before the push:
  # every caller's finish carries its commit_command line; the move hands its whole recipe, whose first lines are no-ops on
  # the topic as it stands), or dropping the edits and the branch (whose existence refuses a rerun) and then <rerun>.
  local branch rerun=${4:-a rerun is refused while it exists}
  branch=$(git branch --show-current)
  git add --update || die "git add failed: the version edits sit on $branch. Once the cause is fixed, finish by hand:
$3
or $(drop_hint "$branch" "$rerun")"
  git diff --cached --quiet && die "nothing to commit: the tree already carries these versions"
  git commit --quiet --signoff --message "$1" --message "$2" \
    || die "the commit '$1' failed: the version edits are staged on $branch. Once the cause is fixed, finish by hand:
$3
or $(drop_hint "$branch" "$rerun")"
}

# --- pushes and pull requests -------------------------------------------------------

push_release_ref() { # <refspec> <what the push triggers>: a release branch or a tag, pushed only on PUSH=1
  local refspec=$1 effect=$2
  if [ "${PUSH:-}" = 1 ]; then
    git push "$REMOTE" "$refspec"
    note "pushed $refspec to $REMOTE: $effect"
  else
    say ""
    say "Not pushed. When ready, a maintainer runs:"
    say "  git push $REMOTE $refspec"
    say "which $effect"
  fi
}
# Pushes name both sides of the refspec and topic branches never track their start point: under
# push.default=upstream, `git push origin topic` of a branch that tracks origin/main pushes to main.
commit_command() { # <subject>: the commit of the tree's changes, as the line to run by hand (in every commit_all caller's finish
  # lines once, before the push: first for the cut and the bump, after the sync and the lock in the move's recipe; so that a
  # failure's message never asks for it twice)
  say "  git add --update && git commit --signoff --message $(printf '%q' "$1")"
}
pr_commands() { # <topic> <base> <title> <body file>: the push of a topic and its pull request, as lines to run by hand
  say "  git push -u $REMOTE refs/heads/$1:refs/heads/$1"
  say "  gh pr create --base $2 --head $1 --title $(printf '%q' "$3") --body-file $(printf '%q' "$4")"
}
move_commands() { # <topic> <source ref> <gateway version> <subject> <body file>: the lines that finish main's move by hand once
  # its topic exists: the sync (a rerun completes what it started), the lock, the commit, then the push and the pull request
  say "  $PYTHON scripts/release/sync_versions.py $2 $3"
  if [ -f "$root/Cargo.lock" ]; then say "  $CARGO update --workspace"; fi
  commit_command "$4"
  pr_commands "$1" main "$4" "$5"
}
open_pr() { # <topic> <base> <title> <body file>: a failure names the lines that finish by hand (the body file among them)
  local topic=$1 base=$2 title=$3 body=$4
  if [ "${PUSH:-}" = 0 ]; then
    say ""
    say "Not pushed (PUSH=0). To open the pull request:"
    pr_commands "$topic" "$base" "$title" "$body"
    return 0
  fi
  command -v "$GH" >/dev/null 2>&1 || die "'$GH' is not installed: push $topic and open the pull request against $base by hand:
$(pr_commands "$topic" "$base" "$title" "$body")"
  git push --quiet -u "$REMOTE" "refs/heads/$topic:refs/heads/$topic" || die "the push of $topic to $REMOTE failed; nothing opened. Once the cause is fixed, finish by hand:
$(pr_commands "$topic" "$base" "$title" "$body")"
  "$GH" pr create --base "$base" --head "$topic" --title "$title" --body-file "$body" || die "gh pr create failed for $topic; the branch is pushed. Once the cause is fixed, open the pull request by hand:
$(pr_commands "$topic" "$base" "$title" "$body")"
}
require_pr_tools() { # <topic> <base>: what open_pr will need under PUSH=1, checked before anything is made or pushed
  local topic=$1 base=$2 open
  command -v "$GH" >/dev/null 2>&1 \
    || die "'$GH' is not installed and PUSH=1 opens the pull request of $topic against $base: install it, or run without PUSH=1 and push by hand"
  open=$("$GH" pr list --head "$topic" --state open --json number --jq '.[].number' | sed 's/^/#/' | paste -sd ' ' -) \
    || die "gh pr list failed (its message is above): cannot read whether an open pull request carries $topic"
  [ -z "$open" ] || die "an open pull request already carries $topic ($open): finish or close it first"
}

# --- the commands -------------------------------------------------------------------

cmd_cut() {
  local version=${1:-} from=${2:-}
  [ -n "$version" ] || die "usage: release.sh cut <version> [<from>]"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release branch from_sha since start start_branch next topic push_move
  release=$(release_of "$version"); branch="release/$release"
  next=$(next_dev_version "$version"); topic="chore/main-versions-$next"
  require_identity; require_clean; fetch
  from=${from:-$REMOTE/main}
  from_sha=$(git rev-parse --verify --quiet "$from^{commit}") || die "unknown commit: $from"
  git merge-base --is-ancestor "$from_sha" "$REMOTE/main" \
    || die "$from is not on $REMOTE/main: a release is cut from main's history"
  has_ref "refs/heads/$branch" && die "$branch exists locally: a new candidate on it is 'release.sh bump $version'"
  remote_has_branch "$branch" && die "$branch exists on $REMOTE: a new candidate on it is 'release.sh bump $version'"
  has_ref "refs/heads/$topic" && die "the branch $topic exists (the pull request that moves main after a cut): delete it or finish it"
  remote_has_branch "$topic" && die "the branch $topic exists on $REMOTE (the pull request that moves main after a cut): finish or delete it"
  # With PUSH=1 the release branch is pushed before main's move: what the move's pull request needs is checked now,
  # not by open_pr once the branch is live.
  [ "${PUSH:-}" != 1 ] || require_pr_tools "$topic" main
  since=$(since_tag "$from_sha" "$version")
  # The previous release's final is the right tag only once main reads above that release: the cut of its line moved
  # main there (bump-main after its final otherwise). A cut from a main below it (that pull request not merged, an
  # older commit) would start from versions below the release that is out (the checker steps a changed crate past
  # the exact versions release tags carry, not above them; the gateway's files would still read below it).
  if ! git merge-base --is-ancestor "$since" "$from_sha" && version_above "${since#v}" "$(manifest_version "$from_sha")"; then
    die "$from reads $(manifest_version "$from_sha"), below ${since#v}: merge the pull request that moves main's versions first (release.sh bump-main ${since#v} opens it), or cut from a commit after it"
  fi
  note "cut: $branch from $(git rev-parse --short "$from_sha") ($from); the versions move from $since to $version"
  # The start is read before the branch is made (`git switch -` and `@{-1}` read the HEAD reflog, which a clone can be
  # without; `git branch --show-current` is the plain branch name even when a tag shares it, and empty when detached):
  # the cut comes back to it, since the next commands run from a checkout that has scripts/release/ (a line cut from a
  # commit before a tooling change on main has none), and the push below names the branch, not HEAD.
  start_branch=$(git branch --show-current); start=${start_branch:-$(git rev-parse HEAD)}
  git switch --quiet --no-track -c "$branch" "$from_sha"
  bump_tree "$version" "$since"
  commit_all "chore(release): bump versions for v$version" \
    "The release branch $branch, cut from $from_sha ($from). Versions since $since." \
    "$(commit_command "chore(release): bump versions for v$version")
  git push $REMOTE refs/heads/$branch:refs/heads/$branch (then main's move, the cut's second half, which only a rerun makes: the shorter way is to drop the edits, delete $branch and rerun the cut)"
  back_to "$start_branch" "$start"
  say ""
  # refs/heads/: a tag named like the branch would answer for a bare name
  say "$branch is at $(git rev-parse --short "refs/heads/$branch"): $(git log -1 --format=%s "refs/heads/$branch")"
  git show --stat --oneline "refs/heads/$branch" | sed -n '2,$p' | sed 's/^/  /'
  say "(You are back on ${start_branch:-$(git rev-parse --short "$start")}; the release commands run from a checkout that has scripts/release/, main for instance.)"
  push_release_ref "refs/heads/$branch:refs/heads/$branch" "runs the test matrix on $branch (nothing is published); then 'release.sh tag $version'"
  # The trunk is next: the moment the branch exists, main moves above everything it will ever release, by a pull
  # request: the gateway and its mirrors to the next development version, every crate and package to at least
  # the version the cut gave the branch (so the next cut bumps from the numbers the registries will hold). The
  # push follows the cut's switch: PUSH=1 pushes the topic and opens the pull request, anything else prints both.
  if [ "${PUSH:-}" = 1 ]; then push_move=1; else push_move=0; fi
  say ""
  note "cut: main moves above $branch: the gateway and its mirrors to $next, every crate and package to at least the branch's"
  if ! PUSH=$push_move move_main "$topic" "$next" "refs/heads/$branch" "chore(release): main reads $next, above release/$release" \
    "\`$branch\` is cut at $(git rev-parse --short "refs/heads/$branch") and reads $version; main moves above it the same moment (the trunk is next): the gateway and its mirrors to \`$next\`, the next development version (a semver pre-release below ${next%-dev} for cargo and Helm, ${next%-dev}.dev0 once Python tooling normalizes it), and every crate and package to at least the version the cut gave the branch, so the next cut's bump starts from the numbers the registries will hold. Main's artifacts are not published; the branch's are." \
    "a rerun of the cut is refused while $branch exists (it stays, pushed or not): to make the move later, recreate the topic (git switch --no-track -c $topic $REMOTE/main) and run the lines above"; then
    say "main carries $next and the branch's crate and package versions already: nothing to move."
  fi
  back_to "$start_branch" "$start" # the move's topic is behind you too, like the release branch
}

cmd_bump() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh bump <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  require_identity; require_clean; fetch
  make_bump "$version"
  open_pr "$bump_topic" "release/$(release_of "$version")" "$bump_title" "$bump_body"
}

# The bump commit the bump and the ship share: its topic, the title and the body file of its pull request.
bump_topic="" bump_title="" bump_body=""
make_bump() { # <version>: the bump commit on its topic branch off the release branch (the clone is left on the topic);
  # bump_topic, bump_title and bump_body name it and its pull request for the caller to open, or to push past
  local version=$1 release branch base since kind
  release=$(release_of "$version"); branch="release/$release"
  base=$(release_base "$release")
  [ "$(manifest_version "$base")" != "$version" ] || die "$branch already reads $version"
  since=$(since_tag "$base" "$version")
  bump_topic="chore/release-$version"
  bump_title="chore(release): bump versions for v$version"
  has_ref "refs/heads/$bump_topic" && die "the branch $bump_topic exists: delete it or finish it"
  if is_final "$version"; then kind="the final"; else kind="the next candidate"; fi
  note "bump: $kind on $branch, versions since $since"
  git switch --quiet --no-track -c "$bump_topic" "$base"
  bump_tree "$version" "$since"
  bump_body=$(mktemp)
  {
    say "The version bump for v$version on \`$branch\` ($kind). Versions since $since:"
    say ""
    say '```'
    git diff --stat HEAD
    say '```'
    say ""
    say "Merging it makes \`$branch\` read v$version (nothing is published by the merge); then \`make release-tag VERSION=$version\` and \`make release-publish-plan VERSION=$version\`."
    say ""
    gate_section "$version" "$base"
  } > "$bump_body"
  commit_all "$bump_title" "$kind on $branch. Versions since $since." \
    "$(commit_command "$bump_title"; pr_commands "$bump_topic" "$branch" "$bump_title" "$bump_body")"
}

cmd_pick() {
  local release=${1:-} shas=${2:-}
  { [ -n "$release" ] && [ -n "$shas" ]; } || die "usage: release.sh pick <x.y> <sha>[,<sha>...]"
  release_ok "$release"
  local base topic sha full names=() wanted=() ordered=() n=0
  require_identity; require_clean; fetch
  base=$(release_base "$release")
  IFS=, read -r -a names <<< "$shas"
  for sha in "${names[@]}"; do
    full=$(git rev-parse --verify --quiet "$sha^{commit}") || die "unknown commit: $sha"
    git merge-base --is-ancestor "$full" "$REMOTE/main" || die "$sha is not on $REMOTE/main: picks come from main"
    [ "$(git rev-list --parents -n 1 "$full" | wc -w)" -le 2 ] || die "$sha is a merge commit: pick its parents' commits"
    git merge-base --is-ancestor "$full" "$base" && die "$sha is already on $base"
    wanted+=("$full")
  done
  while IFS= read -r full; do ordered+=("$full"); done < <(git rev-list --no-walk=sorted --reverse "${wanted[@]}")
  topic="pick/$release"
  for full in "${ordered[@]}"; do
    n=$((n + 1))
    if [ "$n" -le 3 ]; then topic="$topic-$(git rev-parse --short "$full")"; fi
  done
  [ "$n" -le 3 ] || topic="$topic-and-$((n - 3))-more"
  has_ref "refs/heads/$topic" && die "the branch $topic exists: delete it or finish it (make release-pick-pr RELEASE=$release PICK=$topic)"
  note "pick: ${#ordered[@]} commit(s) onto $topic (from $base)"
  git switch --quiet --no-track -c "$topic" "$base"
  if ! git cherry-pick -x --signoff "${ordered[@]}"; then
    local stopped
    stopped=$(git rev-parse --verify --quiet CHERRY_PICK_HEAD || true)
    say ""
    say "release: the pick of ${stopped:0:7} ($(git log -1 --format=%s "${stopped:-HEAD}")) conflicts on:"
    git diff --name-only --diff-filter=U | sed 's/^/    /'
    say "  Resolve it on this branch ($topic):"
    say "    1. edit the files above, then: git add <file>..."
    say "    2. git cherry-pick --continue   (keeps the '(cherry picked from commit ...)' line and your sign-off;"
    say "                                    git cherry-pick --abort gives the pick up)"
    say "    3. make release-pick-pr RELEASE=$release PICK=$topic   (pushes $topic and opens the pull request against release/$release;"
    say "                                    run it from main, as the other release commands)"
    exit 1
  fi
  pick_pr "$release" "$topic"
}

cmd_pick_pr() {
  local release=${1:-} named=${2:-}
  [ -n "$release" ] || die "usage: release.sh pick-pr <x.y> [<pick branch>]"
  release_ok "$release"
  local topic
  topic=$(git rev-parse --abbrev-ref HEAD)
  # The pick branch may lack the release targets (a line cut before a tooling change on main): the command runs from
  # main with the branch named, and checks it out itself, after every refusal: a wrong name or a dirty tree never
  # moves the clone.
  if [ -n "$named" ] && [ "$named" != "$topic" ]; then
    case "$named" in pick/"$release"-*) ;; *) die "$named is not a pick/$release-* branch (the pick branches are pick/$release-<sha>...)" ;; esac
    has_ref "refs/heads/$named" || die "no branch $named here (the pick branches are pick/$release-<sha>...)"
    require_clean "commit or discard them before switching to $named"
    git switch --quiet "$named"
    topic=$named
  fi
  case "$topic" in pick/"$release"-*) ;; *) die "not on a pick/$release-* branch (on $topic): name it, release.sh pick-pr $release <pick branch>" ;; esac
  has_ref CHERRY_PICK_HEAD && die "a cherry-pick is still in progress: git cherry-pick --continue (or --abort) first"
  [ ! -d "$(git rev-parse --git-path sequencer)" ] || die "a cherry-pick sequence is still in progress: git cherry-pick --continue (or --abort) first"
  # A stray change (a lock file a build rewrote after the conflict resolution) would ride into the squash unnamed.
  require_clean "discard them (git restore <file>) or, if the pick needs them, commit them on $topic and say so in the pull request"
  fetch
  pick_pr "$release" "$topic"
}

pick_pr() { # <x.y> <topic>: the pull request of a pick branch
  # The picks are the commits that carry a cherry-pick trailer: a record commit of an earlier run (a pull request
  # creation that failed) changes neither the count, the title nor the body.
  local release=$1 topic=$2 base count title body
  base=$(release_base "$release")
  count=$(picked_in "$base" | grep -c . || true)
  [ "$count" -gt 0 ] || die "$topic has no cherry-picks beyond $base"
  if [ "$count" -eq 1 ]; then
    title=$(git log -1 --format=%s "$(picked_in "$base")" 2>/dev/null || git log --format=%s --grep='cherry picked from commit' -1)
  else
    title="chore(release): pick $count commits from main into release/$release"
  fi
  body=$(mktemp)
  {
    say "Cherry-picks from main into \`release/$release\` (\`git cherry-pick -x -s\`):"
    say ""
    git log --reverse --format='- %h %s' --grep='cherry picked from commit' "$base..HEAD"
    say ""
    say "Picked from:"
    say ""
    git log --format=%b "$base..HEAD" | grep -o '(cherry picked from commit [0-9a-f]*)' | sed 's/^/- /' || true
  } > "$body"
  record_picks "$release" "$base"
  open_pr "$topic" "release/$release" "$title" "$body"
}

cmd_status() {
  local release=${1:-}
  [ -n "$release" ] || die "usage: release.sh status <x.y>"
  release_ok "$release"
  local base picked full short subject n=0
  fetch
  base=$(release_ref "$release")
  picked=$(picked_into "$base")
  say "== release/$release is at $(git rev-parse --short "$base"): $(git log -1 --format=%s "$base")"
  say "== versions: $(manifest_version "$base") ($GATEWAY_MANIFEST)"
  say "== on $REMOTE/main and not on release/$release (same patch or picked with -x excluded):"
  while read -r full short subject; do
    case "$picked" in *" $full "*) continue ;; esac
    n=$((n + 1))
    say "  $short $subject"
  done < <(git log --cherry-pick --right-only --no-merges --format='%H %h %s' "$base...$REMOTE/main")
  say "  ($n commit(s))"
  say "== on release/$release and not on main (the branch's own commits):"
  git log --cherry-pick --left-only --no-merges --format='  %h %s' "$base...$REMOTE/main"
  # The branch's own tags (its candidates, its final, its patches), not every release tag in main's history below it.
  say "== tags made on release/$release (the versions are compared against $(since_tag "$base" "$(manifest_version "$base")")):"
  own_tags "$base" | sed 's/^/  /' | grep . || say "  (none yet)"
  say "== open pull requests against release/$release:"
  if command -v "$GH" >/dev/null 2>&1; then
    "$GH" pr list --base "release/$release" --state open --limit 100 2>/dev/null | sed 's/^/  /' || say "  (gh could not list them)"
  else
    say "  (gh is not installed)"
  fi
}

cmd_tag() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh tag <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release branch base head tagname carried since scratch
  release=$(release_of "$version"); branch="release/$release"; tagname="v$version"
  require_identity; fetch
  base=$(release_base "$release"); head=$(git rev-parse "$base")
  has_ref "refs/tags/$tagname" && die "the tag $tagname exists locally (at $(git rev-parse --short "$tagname^{commit}"))"
  remote_has_tag "$tagname" && die "the tag $tagname exists on $REMOTE"
  carried=$(manifest_version "$head")
  [ "$carried" = "$version" ] || die "$branch reads $carried at $(git rev-parse --short "$head"), not $version: merge the bump first (release.sh bump $version)"
  since=$(since_tag "$head" "$version")
  note "tag: checking the versions at $(git rev-parse --short "$head") against $since"
  scratch=$(mktemp -d)
  git worktree add --quiet --detach "$scratch" "$head"
  if ! (cd "$scratch" && "$PYTHON" scripts/check_release_versions.py --check "$since"); then
    git worktree remove --force "$scratch"
    die "the versions at $branch's head are inconsistent with $since: fix them with a bump pull request first"
  fi
  git worktree remove --force "$scratch"
  git tag --annotate "$tagname" --message "SMG $tagname" --message "Signed-off-by: $(sign_off)" "$head"
  say "tagged $tagname at $(git rev-parse --short "$head") ($branch): $(git log -1 --format=%s "$head")"
  push_release_ref "refs/tags/$tagname" "makes the tag public (a maintainer with tag rights); then 'release.sh publish-plan $version' lists the publishing dispatches and 'release.sh notes $version' drafts the notes"
}

cmd_notes() {
  if [ "${1:-}" = --check ]; then
    [ -n "${2:-}" ] || die "usage: release.sh notes --check <file>"
    [ -f "$2" ] || die "no such file: $2"
    "$PYTHON" "$root/scripts/release/notes.py" check "$2"
    return
  fi
  local version=${1:-} since=${2:-}
  [ -n "$version" ] || die "usage: release.sh notes <version> [<since>], or notes --check <file>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release tagname ref subject count picked full original unknown line path seen=' ' scratch slug compare chart rust contributors
  local lines=() subjects=() refresh=()
  [ "${NOTES_REFRESH:-0}" = 0 ] || refresh=(--refresh)
  release=$(release_of "$version"); tagname="v$version"
  fetch
  if has_ref "refs/tags/$tagname"; then
    ref=$tagname; compare=$tagname
  else
    ref=$(release_ref "$release"); compare="release/$release"
    note "notes: $tagname is not tagged yet; reading release/$release's head"
  fi
  if [ -z "$since" ]; then
    since=$(previous_final_tag "$version")
    [ -n "$since" ] || die "no final release tag below $tagname: pass the tag to start from (release.sh notes $version <tag>)"
  fi
  has_ref "$since" || die "unknown tag or commit: $since"
  # The symmetric range with --cherry-pick: from the second release on, $since is a final on the
  # previous release's branch, and the originals of the fixes picked into it are not news here.
  # A pick lands as one squash commit whose patch matches none of the originals, so --cherry-pick alone would list
  # them again: the commits named in the trailers reachable from $since are left out too (picked_into).
  # On this side a pick's squash commit stands for the originals it recorded under the picks directory (the files it added;
  # "chore(release): pick 2 commits …" says nothing of them): each is listed under its own type, once, left out when
  # $since carries it; a pick landed as its own commits reads the same through its trailer, and a trailer whose
  # original the fetch did not bring keeps the commit's own subject.
  picked=$(picked_into "$since")
  scratch=$(mktemp -d)
  : > "$scratch/subjects"
  count=0
  while read -r full subject; do
    case "$picked" in *" $full "*) continue ;; esac
    mapfile -t lines < <({ for path in $(git show --format= --name-only --diff-filter=A "$full" -- "$PICKS_DIR/"); do printf '%s %s\n' "${path##*/}" "$(git show "$full:$path" | head -n 1)"; done; git log -1 --format=%b "$full" | grep -o 'cherry picked from commit [0-9a-f]*' | sed 's/.* //'; } 2>/dev/null || true)
    subjects=(); unknown=0
    for line in ${lines[@]+"${lines[@]}"}; do
      original=${line%% *}
      case "$picked$seen" in *" $original "*) continue ;; esac
      seen="$seen$original "
      if [ "$line" != "$original" ]; then subjects+=("${line#* }")
      elif has_ref "$original"; then subjects+=("$(git log -1 --format=%s "$original")")
      else unknown=1; fi
    done
    if [ ${#lines[@]} = 0 ] || [ "$unknown" = 1 ]; then subjects+=("$subject"); fi
    for subject in ${subjects[@]+"${subjects[@]}"}; do
      count=$((count + 1))
      printf '%s\n' "$subject" >> "$scratch/subjects"
    done
  done < <(git log --cherry-pick --right-only --first-parent --no-merges --format='%H %s' "$since...$ref")
  # The subjects seed the template (scripts/release/NOTES-TEMPLATE.md) through notes.py, with the rest of the draft's
  # inputs: who wrote the commits; the crates and packages at both ends (the publish plan's list; the previous tag's
  # may predate a component and read shorter, or not at all); the engine-image table, the chart and the toolchain at
  # the release; the published images. The pull request bodies are read through gh and kept under $NOTES_CACHE.
  contributors=$(git shortlog -s --no-merges "$since..$ref" | grep -c . || true)
  component_list "$ref" "$scratch/components-ref" \
    || { note "notes: Packages names the gateway alone"; : > "$scratch/components-ref"; }
  component_list "$since" "$scratch/components-since" 2>/dev/null || : > "$scratch/components-since"
  git show "$ref:.github/workflows/release-engine-docker.yml" > "$scratch/engine-workflow" 2>/dev/null || : > "$scratch/engine-workflow"
  chart=$(git show "$ref:deploy/helm/smg/Chart.yaml" 2>/dev/null | sed -n 's/^version: *//p' | head -n 1)
  rust=$(git show "$ref:Cargo.toml" 2>/dev/null | sed -n 's/^rust-version *= *"\(.*\)"/\1/p' | head -n 1)
  if ! "$root/scripts/release_docker_notes.sh" "$tagname" > "$scratch/docker" 2>/dev/null; then
    printf '### Docker images\n\n(no engine images for %s yet: run %s once they are published)\n' \
      "$tagname" "\`scripts/release_docker_notes.sh $tagname\`" > "$scratch/docker"
  fi
  slug=$(repo_slug)
  "$PYTHON" "$root/scripts/release/notes.py" render --version "$version" --since "$since" --release "$release" \
    --template "$root/scripts/release/NOTES-TEMPLATE.md" --subjects "$scratch/subjects" --count "$count" \
    --contributors "$contributors" --components-since "$scratch/components-since" --components-ref "$scratch/components-ref" \
    --engine-workflow "$scratch/engine-workflow" --docker "$scratch/docker" --chart "$chart" --rust-version "$rust" \
    --repo "$slug" --gh "$GH" --cache "$NOTES_CACHE" --compare-url "${slug:+https://github.com/$slug/compare/$since...$compare}" \
    ${refresh[@]+"${refresh[@]}"} \
    || die "notes: the draft was not rendered (the cause is above); its inputs stay under $scratch"
  rm -rf -- "$scratch"
  note "a draft: replace every <!-- AUTHOR: --> comment, then 'release.sh notes --check <file>' before the GitHub release:"
  if is_final "$version"; then
    note "  gh release create $tagname --verify-tag --title '$tagname' --notes-file <file>"
  else
    note "  gh release create $tagname --verify-tag --prerelease --title '$tagname' --notes-file <file>"
  fi
}

move_main() { # <topic> <gateway version> <source ref> <subject> <intro> <rerun>: main's version files move on <topic>, by a pull
  # request; <rerun> is what the caller offers once a failed move's topic is dropped (the cut: the topic recreated and
  # the lines above run, its branch stays)
  # The gateway and its mirrors to <gateway version> when main reads below it; main never goes back (after a patch of
  # an older line, or a cut while main already reads above, they keep main's). Every crate and package to at least
  # <source ref>'s, as sync_versions.py moves them. The commit, the push and the pull request as open_pr makes them.
  # Returns 1, with nothing made, when main carries the versions already. Every other failure stops the command
  # (die): both callers run this under `if` or `||`, where set -e is off and a failed step would read as "nothing
  # to move" with the version edits left behind.
  local topic=$1 gateway=$2 source=$3 subject=$4 intro=$5 rerun=$6 current moved start start_branch body shown finish
  shown=${source#refs/heads/} # a local release branch is read by its full ref (a tag may share its name), named plainly
  start_branch=$(git branch --show-current); start=${start_branch:-$(git rev-parse HEAD)}
  git switch --quiet --no-track -c "$topic" "$REMOTE/main" || die "cannot make $topic from $REMOTE/main"
  current=$(manifest_version "$REMOTE/main")
  if [ "$current" = "$gateway" ] || version_above "$current" "$gateway"; then
    note "main reads $current, not below $gateway: the gateway and its mirrors keep it; the crates and packages $shown raised move"
    moved="The gateway keeps $current on main (not below $gateway)"
  else
    note "main reads $current, below $gateway: the gateway and its mirrors move to $gateway"
    moved="The gateway reads $gateway on main"
  fi
  # The pull request's body is made before the first edit, so that every failure from here names the lines that finish
  # the move by hand, the pull request's with its body among them; the stat of the edits joins the intro once they exist.
  body=$(mktemp) || die "cannot make a temporary file for the pull request body of $topic (nothing is edited yet). Once the cause is fixed, finish by hand, with a body file of your own (body.md below):
$(move_commands "$topic" "$source" "$gateway" "$subject" body.md)
or switch back to where you started and delete $topic (git branch -D $topic): $rerun"
  say "$intro" > "$body"
  finish=$(move_commands "$topic" "$source" "$gateway" "$subject" "$body")
  note "scripts/release/sync_versions.py $source $gateway (every crate and package at least $shown's)"
  "$PYTHON" "$root/scripts/release/sync_versions.py" "$source" "$gateway" \
    || die "sync_versions.py failed: the version files of $topic are moved in part at most. Once the cause is fixed, finish by hand (a rerun of the sync completes what it started):
$finish
or $(drop_hint "$topic" "$rerun")"
  refresh_lock "the version edits sit on $topic. Once the cause is fixed, finish by hand:
$finish
or $(drop_hint "$topic" "$rerun")"
  if git diff --quiet && git diff --cached --quiet; then
    rm -f "$body"
    back_to "$start_branch" "$start"
    git branch --quiet -D "$topic" || die "cannot delete $topic (nothing to move)"
    return 1
  fi
  {
    say ""
    say '```'
    git diff --stat HEAD
    say '```'
  } >> "$body"
  commit_all "$subject" \
    "$moved and every crate and package at least the version $shown carries (crates.io and PyPI never take a version twice). Main's artifacts are not published; the release branches' are." \
    "$finish" "$rerun"
  open_pr "$topic" main "$subject" "$body"
}

cmd_bump_main() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh bump-main <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  is_final "$version" || die "main follows a final release (X.Y.Z), not a candidate"
  local release source topic next
  release=$(release_of "$version")
  require_identity; require_clean; fetch
  if has_ref "refs/tags/v$version"; then
    source="v$version"
  else
    source=$(release_base "$release")
    note "bump-main: v$version is not tagged yet; reading release/$release's head"
  fi
  topic="chore/main-versions-$version"
  has_ref "refs/heads/$topic" && die "the branch $topic exists: delete it or finish it"
  # The cut of this line moved main to its next development version already (the trunk is next): the gateway keeps
  # it, and the crates and packages the release raised since the cut (its candidates' and final's bumps) move to at
  # least the release's. A main that still reads below the release (a line cut before the rule, a move not merged)
  # takes that development version here.
  next=$(next_dev_version "$version")
  move_main "$topic" "$next" "$source" "chore(release): main's versions follow v$version" \
    "After the v$version release: main's crates and packages follow the release (\`$source\`), so the next cut starts above what crates.io and PyPI already carry; the gateway keeps its development version (the trunk is next), or moves to \`$next\` when main read below the release." \
    "then 'release.sh bump-main $version' again" \
    || die "bump-main: main carries v$version's versions already (the gateway at $(manifest_version "$REMOTE/main"), every crate and package at least the release's): nothing to move"
}

pep440() { # 1.14.0-rc.1 -> 1.14.0rc1, the spelling PyPI shows
  printf '%s' "$1" | sed 's/-rc\./rc/'
}
ref_version() { # <ref> <file>: the version line a file carries at a ref (empty when the file is not there)
  git show "$1:$2" 2>/dev/null | grep -m1 '^version = ' | sed 's/version = "\(.*\)"/\1/' || true
}
publishable() { # <ref>: "kind name version manifest path" of every crate and package the version checker lists, at a ref
  # A manifest whose [package] says `publish = false` or `publish = []` names a crate cargo never publishes (and
  # release-crates.yml has no tier for): left out. The key is read from the parsed manifest, not matched as a line.
  "$PYTHON" - "$1" <<'PYEOF'
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_release_versions", Path("scripts/check_release_versions.py"))
crv = importlib.util.module_from_spec(spec)
sys.modules["check_release_versions"] = crv  # its dataclasses resolve their annotations through sys.modules
spec.loader.exec_module(crv)
ref = sys.argv[1]
for component in (*crv.CRATES, *crv.PACKAGES):
    shown = subprocess.run(["git", "show", f"{ref}:{component.version_file}"], capture_output=True, text=True)
    if shown.returncode != 0:
        continue
    if component.kind == "crate":
        if tomllib.loads(shown.stdout).get("package", {}).get("publish", True) in (False, []):
            continue
        kind = "cargo"
    else:
        kind = crv.version_kind(component.version_file)
    version = crv.read_version(kind, shown.stdout, component.version_file)
    print(component.kind, component.name, version, component.version_file, component.path)
PYEOF
}
component_list() { # <ref> <file>: publishable's lines into <file>, or a failure naming the error (a manifest the checker refuses, an import
  # failure): read outside a pipe or a process substitution, where the exit status would be lost and a short list pass for the whole
  local err
  err=$(mktemp)
  if ! publishable "$1" > "$2" 2> "$err"; then
    say "the crate and package list at $(git rev-parse --short "$1") could not be read ($(tail -n 1 "$err"))" >&2
    return 1
  fi
}

# --- the registries ---------------------------------------------------------------------
# crates.io and PyPI take a version once: a component whose version at the branch head is published already is skipped
# by its publish dispatch, and when its code changed after that version was published the release ships none of those
# changes while every repository-side check reads green. The commits a published version misses are counted from the
# registry's own publish time of that version (the sparse index line's pubtime, PyPI's upload time): a version
# published on main the older way, at its bump, counts every commit since, tag or no tag; a version a tag's dispatch
# published counts only what landed after that dispatch. A held version whose line gives no time counts from its
# bump. The reads run one at a time: crates.io through the sparse index cargo itself reads (one static file per
# crate, no API call: the API asks its clients for one request a second), PyPI's JSON API at most once a second; 20 s
# each. REGISTRY=0 skips them.

USER_AGENT='smg-release-tooling (https://github.com/smg-project/smg)'
index_path() { # <crate name>: the crate's file in the sparse index, as cargo spells it (1/x, 2/xy, 3/x/xyz, xy/zw/xyzw...)
  local name
  name=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  case ${#name} in
    1) printf '1/%s' "$name" ;;
    2) printf '2/%s' "$name" ;;
    3) printf '3/%s/%s' "${name:0:1}" "$name" ;;
    *) printf '%s/%s/%s' "${name:0:2}" "${name:2:2}" "$name" ;;
  esac
}
registry_versions() { # <crate|package> <name>: "version [publish time]" per version the registry holds (yanked ones too: a name takes a version once; the time in UTC, whole seconds, empty when the registry gives none); "unread" when it did not answer; nothing for a name it has not
  local url out code body
  case $1 in crate) url="https://index.crates.io/$(index_path "$2")" ;; *) url="https://pypi.org/pypi/$2/json" ;; esac
  out=$("$CURL" -sS -L -m 20 -A "$USER_AGENT" -w '\n%{http_code}' "$url" 2>/dev/null) || { say unread; return 0; }
  code=${out##*$'\n'}; body=${out%$'\n'*}
  case $code in
    200) "$PYTHON" -c 'import json, sys
from datetime import datetime, timezone


def utc(text):
    if not text:
        return ""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if sys.argv[1] == "crate":
    for line in sys.stdin:
        if line.strip():
            entry = json.loads(line)
            print(entry["vers"], utc(entry.get("pubtime", "")))
else:
    for version, files in json.load(sys.stdin)["releases"].items():
        times = sorted(f["upload_time_iso_8601"] for f in files if f.get("upload_time_iso_8601"))
        print(version, utc(times[0]) if times else "")' "$1" <<< "$body" 2>/dev/null || say unread ;;
    404) ;;
    *) say unread ;;
  esac
}
after_text() { # <count> <bump|publish time>: the commits a published version misses, and from where they are counted
  if [ "$2" = bump ]; then printf '%s commit(s) after its bump; the registry gave no publish time' "$1"; else printf '%s commit(s) after it was published (%s)' "$1" "$2"; fi
}
registry_report() { # <ref> <list>: "name version held|free|unread <commits after> <bump|publish time>" per line of <list> (publishable's output), one registry read at a time
  local ref=$1 list=$2 kind name version manifest path moved published after state line when pypi=0
  while read -r kind name version manifest path; do
    state=free; published=""
    if [ "${REGISTRY:-1}" = 0 ]; then
      state=unread
    else
      if [ "$kind" = package ]; then [ "$pypi" -eq 0 ] || sleep 1; pypi=$((pypi + 1)); fi
      while read -r line when; do
        [ "$line" = unread ] && state=unread
        [ "$line" = "$version" ] && { state=held; published=$when; }
      done < <(registry_versions "$kind" "$name")
    fi
    if [ -n "$published" ]; then
      # the commits that landed after the publish (committer dates: a pick merged later counts, a pick merged before does not)
      after=$(git rev-list --count --after="$published" "$ref" -- "$path")
    else
      moved=$(git log -1 --format=%H -G'^version = ' "$ref" -- "$manifest")
      after=$(git rev-list --count "${moved:-$ref}..$ref" -- "$path")
    fi
    printf '%s %s %s %s %s\n' "$name" "$version" "$state" "$after" "${published:-bump}"
  done < "$list" | sort
}

engine_dispatches() { # <ref> <tag>: one dispatch per (engine, release base), the table read from the workflow at <ref>
  "$PYTHON" - "$1" "$2" <<'PYEOF'
import json
import re
import subprocess
import sys

ref, tag = sys.argv[1:3]
shown = subprocess.run(
    ["git", "show", f"{ref}:.github/workflows/release-engine-docker.yml"], capture_output=True, text=True
)
match = re.search(r"engines='(\{.*?\})'", shown.stdout, re.S) if shown.returncode == 0 else None
if match is None:
    print(f"(no release-base table is readable at {tag}: one dispatch per engine, its first base)")
    for engine in ("vllm", "sglang", "tokenspeed", "trtllm"):
        print(f"gh workflow run release-engine-docker.yml --ref {tag} -f engine={engine}")
    sys.exit(0)
for engine, entry in json.loads(match.group(1)).items():
    for position, base in enumerate(entry.get("release_bases") or [""]):
        with_base = f" -f base_image_ref={base}" if position and base else ""
        print(
            f"gh workflow run release-engine-docker.yml --ref {tag} -f engine={engine}{with_base}"
            f"   # {base or 'the latest nightly carrier'}"
        )
PYEOF
}

registry_mark() { # <report file> <name> <registry>: the mark a plan line carries when the registry holds that version already
  awk -v n="$2" -v r="$3" '$1 == n && $3 == "held" { printf " (on %s already: the dispatch skips it%s)", r, ($4 > 0 ? "; " $4 " commit(s) after " ($5 == "bump" ? "its bump" : "it was published, " $5) : "") }' "$1"
}

cmd_publish_plan() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh publish-plan <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release tagname ref
  release=$(release_of "$version"); tagname="v$version"
  fetch
  if has_ref "refs/tags/$tagname"; then
    ref=$tagname
  else
    ref=$(release_ref "$release")
    note "publish-plan: $tagname is not tagged yet; the plan reads release/$release's head and names the tag as the ref to dispatch on"
  fi
  [ "$(manifest_version "$ref")" = "$version" ] || die "$ref reads $(manifest_version "$ref"), not $version"
  publish_plan "$version" "$ref"
}

publish_plan() { # <version> <ref>: the plan for v<version> as read at <ref> (the tag; the branch head before it exists; for the
  # ship's dry run the head before the bump, whose dispatch lines are the tag's already)
  local version=$1 ref=$2 tagname latest proto servicer list marks crate_name crate_version n=0
  tagname="v$version"
  if is_final "$version"; then latest="and latest"; else latest="(a candidate: latest does not move)"; fi
  proto=$(ref_version "$ref" crates/grpc_client/python/pyproject.toml)
  servicer=$(ref_version "$ref" grpc_servicer/pyproject.toml)
  say "Publishing v$version: nothing below runs by itself. A maintainer dispatches each workflow against the tag"
  say "$tagname (pushed by release.sh tag $version); every run reads its versions from that ref."
  say ""
  # A candidate is never published to PyPI or crates.io: its plan leaves the three registry dispatches out and says so once;
  # the workflows refuse such a ref on their own. The items count themselves, so neither plan has a gap.
  if is_candidate "$version"; then
    say "A candidate is never published to PyPI or crates.io: the Python package, the Rust crates and the gRPC Python packages wait for the final (their workflows, release-pypi.yml, release-crates.yml and release-grpc.yml, refuse a ref named -rc.; this plan leaves their dispatches out)."
    say ""
  else
    say "$((n += 1)). Python package (PyPI: smg==$(pep440 "$version"), wheels + sdist)"
    say "   gh workflow run release-pypi.yml --ref $tagname"
    say "$((n += 1)). Rust crates (crates.io: every workspace crate whose version at $tagname is not published yet; smg $version)"
    say "   gh workflow run release-crates.yml --ref $tagname"
    say "   crate versions at $tagname (a crate whose manifest says publish = false or publish = [] is never published and not listed; a version crates.io holds already is skipped by the dispatch and marked):"
    list=$(mktemp); marks=$(mktemp)
    component_list "$ref" "$list" || die "publish-plan: the crates and packages at $ref are unknown; nothing listed"
    registry_report "$ref" "$list" > "$marks"
    while read -r crate_name crate_version; do
      say "     $crate_name $crate_version$(registry_mark "$marks" "$crate_name" crates.io)"
    done < <(awk '$1 == "crate" { print $2, $3 }' "$list")
    [ "${REGISTRY:-1}" != 0 ] || say "     (the registries were not read: REGISTRY=0)"
  fi
  say "$((n += 1)). Gateway image (Docker Hub lightseekorg/smg:$version and ghcr.io/smg-project/smg:$version $latest; the same tags with -slurm appended = the variant with the Slurm clients)"
  say "   gh workflow run release-docker.yml --ref $tagname"
  say "$((n += 1)). Helm chart (oci://ghcr.io/smg-project/charts/smg, version $version)"
  say "   gh workflow run release-helm.yml --ref $tagname"
  say "$((n += 1)). Engine images (ghcr.io/smg-project/smg:$version-<engine>-<engine version>, built from SMG_RELEASE_TAG = $tagname at that ref;"
  say "   one dispatch per engine and release base, the table of .github/workflows/release-engine-docker.yml at $tagname: a dispatch"
  say "   without base_image_ref builds the engine's first base only, so every other base is a dispatch of its own)"
  # explicit, so that a table the Python cannot read stops the plan on every call path (under `if` and `||`, where set -e is off)
  engine_dispatches "$ref" "$tagname" | sed 's/^/   /' \
    || die "publish-plan: the engine table of .github/workflows/release-engine-docker.yml at $ref could not be read (its message is above); no engine dispatch listed"
  if ! is_candidate "$version"; then
    say "$((n += 1)). gRPC Python packages (PyPI: smg-grpc-proto==${proto:-?}$(registry_mark "$marks" smg-grpc-proto PyPI), smg-grpc-servicer==${servicer:-?}$(registry_mark "$marks" smg-grpc-servicer PyPI); versions already on PyPI are skipped)"
    say "   gh workflow run release-grpc.yml --ref $tagname"
  fi
  say "$((n += 1)). Go module tag (crates/grpc_client/go/v$version)"
  say "   gh workflow run release-go.yml --ref $tagname -f version=$version"
  # The bindings module resolves through its tag only where its go.mod declares the path it lives at (the releases from that
  # change on); at an older ref the tag would name a go.mod that go get refuses, so the plan prints the item only where it holds.
  if [ "$(git show "$ref:bindings/golang/go.mod" 2>/dev/null | sed -n 's/^module //p')" = github.com/smg-project/smg/bindings/golang ]; then
    say "$((n += 1)). Go bindings module tag (bindings/golang/v$version at the release commit; by hand until release-go.yml pushes it)"
    say "   git tag bindings/golang/v$version $tagname^{commit} && git push $REMOTE refs/tags/bindings/golang/v$version"
  else
    say "   (no Go bindings module tag: bindings/golang/go.mod at $ref does not declare github.com/smg-project/smg/bindings/golang, so a tag there would not resolve)"
  fi
  say "$((n += 1)). The GitHub release: release.sh notes $version > notes.md, then the gh release create line it prints."
  say ""
  say "Watch a dispatch: gh run list --workflow <file> --branch $tagname"
}

api_message() { # <gh api output>: the "message" of the error body gh leaves on stdout when the API refused (empty otherwise)
  sed -n 's/.*"message": *"\([^"]*\)".*/\1/p' <<< "$1" | head -n 1
}

VERIFIER_FAMILIES="DeepSeek GLM Kimi MiniMax"

gate_section() { # <version> <ref>: the release gate as a pull request body section
  local version=$1 ref=$2 family
  say "## Release gate"
  say ""
  say "- [ ] CI green on \`$(git rev-parse --abbrev-ref "$ref" | sed "s#^$REMOTE/##")\` at $(git rev-parse --short "$ref") (the branch's own run of the test matrix)"
  if is_final "$version"; then
    say "- [ ] the four model-provider verifiers ran against this release line's candidate build; one record link each:"
  else
    say "- [ ] the four model-provider verifiers run against this candidate build (their records go into the final's bump pull request):"
  fi
  for family in $VERIFIER_FAMILIES; do
    say "  - [ ] $family provider verifier: <record link>"
  done
}

versions_row() { # <head> <version>: the gate's versions row for <head>; 0 when it is green
  local head=$1 version=$2 carried since scratch
  carried=$(manifest_version "$head")
  if [ "$carried" != "$version" ]; then
    say "[FAIL] versions: the branch reads $carried, not $version (release.sh bump $version)"
    return 1
  fi
  since=$(since_tag "$head" "$version")
  scratch=$(mktemp -d)
  git worktree add --quiet --detach "$scratch" "$head"
  if (cd "$scratch" && "$PYTHON" scripts/check_release_versions.py --check "$since" >/dev/null 2>&1); then
    git worktree remove --force "$scratch"
    say "[ok]   versions: the branch reads $version and every crate, package and mirror is consistent with $since"
    return 0
  fi
  git worktree remove --force "$scratch"
  say "[FAIL] versions: the branch reads $version but scripts/check_release_versions.py --check $since finds issues (a bump pull request fixes them)"
  return 1
}

registry_rows() { # <ref> <branch>: the gate's registry rows for the crates and packages at <ref>: a component at a version
  # crates.io or PyPI hold already is skipped by its publish dispatch; with changes after that version was published, the
  # release ships none of them. 0 when the row is green, 1 when it is red, 2 when the registries were not read (REGISTRY=0,
  # an answer missing). The component list is read first, with its exit status: a list the checker could not produce is a
  # failure of this item, never a shorter list that passes.
  local ref=$1 branch=$2 list listed name carried state after from held="" quiet="" unread=0 read=0 failed=0
  list=$(mktemp)
  if ! listed=$(component_list "$ref" "$list" 2>&1); then
    say "[FAIL] registry: $listed; nothing was compared with crates.io or PyPI"
    return 1
  fi
  while read -r name carried state after from; do
    read=$((read + 1))
    case $state in
      held) if [ "$after" -gt 0 ]; then held="$held, $name $carried ($(after_text "$after" "$from"))"; else quiet="$quiet, $name $carried"; fi ;;
      unread) unread=$((unread + 1)) ;;
    esac
  done < <(registry_report "$ref" "$list")
  if [ "$read" -ne "$(grep -c . "$list")" ]; then
    say "[FAIL] registry: $read of $(grep -c . "$list") crates and packages compared with crates.io / PyPI (the report is short)"
    failed=1
  elif [ "${REGISTRY:-1}" = 0 ]; then
    say "[....] registry: crates.io and PyPI not read (REGISTRY=0): a version they hold already is skipped by its publish dispatch"
    failed=2
  elif [ -n "$held" ]; then
    say "[FAIL] registry: at a version crates.io or PyPI hold already, with changes after it (the publish dispatch skips them and ships none of those changes; a bump pull request against $branch moves them): ${held#, }"
    failed=1
  elif [ "$unread" -gt 0 ]; then
    say "[....] registry: $unread of $read read(s) of crates.io / PyPI did not answer; read again before the tag"
    failed=2
  else
    say "[ok]   registry: no crate or package sits at a version crates.io or PyPI hold already with changes after it ($read of $read read)"
  fi
  [ -z "$quiet" ] || say "[note] registry: at a published version with nothing changed after it (the dispatch skips them, nothing is lost): ${quiet#, }"
  return "$failed"
}

inherited_run() { # <repository slug> <head>: the run that stands for <head>'s card tier when <head> changes nothing but version
  # strings on its parent (scripts/ci_version_only.py, the rule the PR Test workflow applies): the parent's successful PR Test (SMG)
  # run on a push or a dispatch (a pull request's run can pass with the card lanes skipped, the docs-only shortcut; those two
  # kinds keep every lane, or inherit it by this very rule), as "<parent short sha> <run url>". The head's own checks come from
  # every workflow on its push, so the parent must be green on every check of its own (none red, none pending), not on that
  # run alone: fails otherwise, as it does when the head is more than such a change or the parent has no such run
  local parent checks runs url
  parent=$(git rev-parse --verify --quiet "$2^") || return 1
  "$PYTHON" scripts/ci_version_only.py "$parent" "$2" >/dev/null 2>&1 || return 1
  checks=$("$GH" api --paginate "repos/$1/commits/$parent/check-runs?per_page=100" --jq '.check_runs[] | "\(.conclusion // .status) \(.name)"' 2>/dev/null) || return 1
  [ -n "$checks" ] || return 1
  ! grep -qvE '^(success|skipped|neutral) ' <<< "$checks" || return 1
  runs=$("$GH" api "repos/$1/actions/workflows/pr-test-rust.yml/runs?head_sha=$parent&per_page=20" --jq '.workflow_runs[] | select(.event != "pull_request") | "\(.conclusion // .status) \(.html_url)"' 2>/dev/null) || return 1
  url=$(grep -m1 '^success ' <<< "$runs" | cut -d' ' -f2)
  [ -n "$url" ] && printf '%s %s\n' "$(git rev-parse --short "$parent")" "$url"
}

ci_rows() { # <head> <branch>: the gate's CI rows for <head>: its check runs, and the test matrix on this very commit (a
  # documentation-only merge runs none, the push trigger's paths-ignore, and the pull request's run was for the merge ref,
  # not the branch head, so a run is required by head sha). 0 when green; 1 when a row is red (a failed or cancelled run,
  # no run of the matrix at all, a read that failed); 2 when a row is pending or the head is not on the repository gh reads.
  # inherited: "<parent> <run url>" once read (inherited_run), for a head that changes nothing but version strings and whose
  # parent is green on every check and whose run passed; read only where a pending or missing run of the head would otherwise fail the row.
  local head=$1 branch=$2 slug checks runs ok=0 bad=0 pending=0 line read_rc=0 error state=0 inherited=""
  if command -v "$GH" >/dev/null 2>&1 && slug=$("$GH" repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null) && [ -n "$slug" ]; then
    # On an API error gh leaves the error body on stdout ({"message": ...}) and exits non-zero: the read is reported
    # as the failure it is, never counted as a check run (a head the remote does not have yet answers 422).
    checks=$("$GH" api --paginate "repos/$slug/commits/$head/check-runs?per_page=100" --jq '.check_runs[] | "\(.conclusion // .status) \(.name)"' 2>/dev/null) || read_rc=$?
    if [ "$read_rc" -ne 0 ]; then
      error=$(api_message "$checks")
      if [[ $error == "No commit found for SHA"* ]]; then
        # Two causes, both outside this read: the head is not pushed (a cut rehearsed locally, a stale or hand-made
        # remote-tracking ref), or gh reads another repository than $REMOTE (a fork as the remote, GH_REPO, gh repo set-default).
        say "[....] CI: $(git rev-parse --short "$head") is not on $slug, the repository gh reads ($REMOTE is $(git remote get-url "$REMOTE")): either the head is not pushed yet, or gh is pointed at another repository than $REMOTE (GH_REPO, gh repo set-default, a fork as the remote); its checks and its PR Test (SMG) run are read here once they agree"
        state=2
      else
        say "[FAIL] CI: the check-runs read failed (${error:-gh exited $read_rc}); the checks and the PR Test (SMG) run of $(git rev-parse --short "$head") are at https://github.com/$slug/commits/$head"
        state=1
      fi
    elif [ -z "$checks" ]; then
      say "[....] CI: no check run on $(git rev-parse --short "$head") yet (https://github.com/$slug/commits/$head)"
      state=2
    else
      while IFS= read -r line; do
        case "$line" in
          success\ *|skipped\ *|neutral\ *) ok=$((ok + 1)) ;;
          failure\ *|cancelled\ *|timed_out\ *|action_required\ *|stale\ *|startup_failure\ *) bad=$((bad + 1)); say "       red: ${line#* }" ;;
          *) pending=$((pending + 1)); say "       pending: ${line#* } (${line%% *})" ;;
        esac
      done <<< "$checks"
      if [ "$bad" -eq 0 ] && [ "$pending" -eq 0 ]; then
        say "[ok]   CI: $ok check(s) green on $(git rev-parse --short "$head")"
      elif [ "$bad" -eq 0 ] && inherited=$(inherited_run "$slug" "$head"); then
        say "[ok]   CI: $ok green, $pending pending on $(git rev-parse --short "$head"): the head changes nothing but version strings on ${inherited%% *}, green on every check, whose PR Test (SMG) run passed (${inherited#* }); the card tier is that run's, the head's own pending checks are expected to end as the parent's did"
      else
        say "[FAIL] CI: $ok green, $bad red, $pending pending on $(git rev-parse --short "$head") (https://github.com/$slug/commits/$head)"
        if [ "$bad" -gt 0 ]; then state=1; else state=2; fi
      fi
    fi
    # A head the remote lacks has no run, and a read that failed above fails here the same way: one line says so.
    if [ "$read_rc" -eq 0 ]; then
      runs=$("$GH" api "repos/$slug/actions/workflows/pr-test-rust.yml/runs?head_sha=$head&per_page=20" --jq '.workflow_runs[] | "\(.conclusion // .status) \(.html_url)"' 2>/dev/null) || read_rc=$?
      if [ "$read_rc" -ne 0 ]; then
        error=$(api_message "$runs")
        say "[FAIL] PR Test (SMG): the workflow-runs read failed (${error:-gh exited $read_rc}); the runs of $(git rev-parse --short "$head") are at https://github.com/$slug/actions/workflows/pr-test-rust.yml"
        state=1
      elif [ -z "$runs" ] && ! { [ -n "$inherited" ] || inherited=$(inherited_run "$slug" "$head"); }; then
        # an absent matrix is not a queued one: nothing arrives by waiting
        say "[FAIL] PR Test (SMG): no run on $(git rev-parse --short "$head") (a documentation-only merge runs none): gh workflow run pr-test-rust.yml --ref $branch"
        state=1
      elif grep -q '^success ' <<< "$runs"; then
        say "[ok]   PR Test (SMG): a successful run on $(git rev-parse --short "$head") ($(grep -m1 '^success ' <<< "$runs" | cut -d' ' -f2))"
      elif ! grep -qE '^(failure|cancelled|timed_out|action_required|stale|startup_failure) ' <<< "$runs" && { [ -n "$inherited" ] || inherited=$(inherited_run "$slug" "$head"); }; then
        say "[ok]   PR Test (SMG): the card tier of $(git rev-parse --short "$head") is ${inherited%% *}'s successful run (${inherited#* }): the head changes nothing but version strings on it; the head's own run ($(paste -sd ';' <<< "${runs:-not started yet}")) is expected to end as the parent's did"
      else
        say "[FAIL] PR Test (SMG): no successful run on $(git rev-parse --short "$head") yet: $(tr '\n' ';' <<< "$runs")"
        # a run still going is waited for; runs that all ended without a success are red
        if grep -Eq '^(queued|in_progress|waiting|requested|pending) ' <<< "$runs"; then [ "$state" -eq 1 ] || state=2; else state=1; fi
      fi
    fi
  else
    say "[....] CI: not readable here (gh is not installed or not logged in): read the checks of $(git rev-parse --short "$head") on the branch page"
  fi
  return "$state"
}

verifier_rows() { # the provider verifiers: the release owner's lines, as the gate prints them
  local family
  say "[....] model-provider verifiers against this candidate build (the release owner fills the record links; they go into the final's bump pull request):"
  for family in $VERIFIER_FAMILIES; do
    say "       [ ] $family provider verifier: <record link>"
  done
}

publish_rule() { # <version>: the gate's line on what the registries take: a candidate never, the final by its dispatches
  if is_candidate "$1"; then
    say "[note] publish: a candidate is never published to PyPI or crates.io (release-pypi, release-crates and release-grpc refuse a ref named -rc.; release.sh publish-plan $1 leaves them out): its artifacts are the images, the chart, the two Go module tags and the GitHub pre-release"
  else
    say "[note] publish: the final publishes to PyPI and crates.io by its dispatches (release.sh publish-plan $1 lists them; a candidate never does)"
  fi
}

cmd_check() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh check <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release branch base head tagname drift=0 failed=0 full picked
  release=$(release_of "$version"); branch="release/$release"; tagname="v$version"
  fetch
  base=$(release_ref "$release"); head=$(git rev-parse "$base")
  say "Release gate for v$version ($branch at $(git rev-parse --short "$head"): $(git log -1 --format=%s "$head"))"
  say ""
  # 1. the versions at the head
  versions_row "$head" "$version" || failed=1
  # 1b. the registries
  registry_rows "$head" "$branch" || failed=1
  publish_rule "$version"
  # 2. CI on the head
  ci_rows "$head" "$branch" || failed=1
  # 3. the tag
  if has_ref "refs/tags/$tagname" || remote_has_tag "$tagname"; then
    if [ "$(git rev-parse "$tagname^{commit}" 2>/dev/null)" = "$head" ]; then
      say "[ok]   tag: $tagname is at the branch head"
    else
      say "[note] tag: $tagname exists but not at the branch head ($(git rev-parse --short "$tagname^{commit}" 2>/dev/null || echo remote)); the head has moved since"
    fi
  else
    say "[....] tag: $tagname not created yet (release.sh tag $version, after the gate)"
  fi
  # 4. main's drift
  picked=$(picked_into "$base")
  while read -r full; do
    case "$picked" in *" $full "*) ;; *) drift=$((drift + 1)) ;; esac
  done < <(git log --cherry-pick --right-only --no-merges --format=%H "$base...$REMOTE/main")
  say "[note] main: $drift commit(s) on $REMOTE/main are not on $branch (release.sh status $release lists them; release.sh pick $release <sha> brings one over)"
  # 5. the provider verifiers: the release owner's lines
  verifier_rows
  say ""
  if [ "$failed" -eq 0 ]; then
    say "The repository-readable items pass; the verifier lines are the release owner's."
  else
    say "A repository-readable item is not green yet."
  fi
  return "$failed"
}

# --- the ship --------------------------------------------------------------------------------
# One run takes a release from the bump to the dispatches. Every step reads the remote first and skips what is there
# already (the branch reads the version, the tag, the GitHub release, a dispatch's run), so an interrupted run is rerun
# as it stands, or from a step with --from; nothing outward happens before the rows the bumped tree answers are green.
# A dry run reads and prints each step with the values it would use; it writes nothing, in the clone or on the remote.

SHIP_STEPS="preflight bump tag notes release publish"
ship_dry=0 ship_version="" ship_release="" ship_branch="" ship_tag="" ship_head="" ship_notes_file="" ship_new=0 ship_had=0
push_block="" dispatched_run=""

ship_say() { say "ship: $*"; }
ship_would() { say "ship: would $*"; }

cmd_ship() {
  local version="" from=preflight step started=0 kind base
  while [ $# -gt 0 ]; do
    case $1 in
      --dry-run) ship_dry=1 ;;
      --from) shift; from=${1:-} ;;
      --from=*) from=${1#--from=} ;;
      -*) die "usage: release.sh ship <version> [--dry-run] [--from <step>] (unknown option $1)" ;;
      *) [ -z "$version" ] || die "usage: release.sh ship <version> [--dry-run] [--from <step>]"; version=$1 ;;
    esac
    shift
  done
  [ -n "$version" ] || die "usage: release.sh ship <version> [--dry-run] [--from <step>]"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  case " $SHIP_STEPS " in *" $from "*) ;; *) die "unknown step '$from': --from takes one of: $SHIP_STEPS" ;; esac
  command -v "$GH" >/dev/null 2>&1 || die "'$GH' is not installed: the ship reads the repository's rules and runs, opens the release and dispatches the workflows through it"
  # the account the ship acts as, read as the rules probe reads it (one host, one token): a bare `gh auth status` judges every host and account gh knows
  "$GH" api user --jq .id >/dev/null 2>&1 || die "'$GH' is not logged in (gh auth login, or GH_TOKEN): the ship reads the repository's rules and runs, opens the release and dispatches the workflows through it"
  ship_version=$version; ship_release=$(release_of "$version"); ship_branch="release/$ship_release"; ship_tag="v$version"
  [ "$ship_dry" -eq 1 ] || require_identity
  fetch
  base=$(release_base "$ship_release"); ship_head=$(git rev-parse "$base")
  PUSH=1 # the ship is the push: the tag's and the bump pull request's helpers push and open
  if is_final "$version"; then kind="the final"; else kind="a candidate"; fi
  ship_say "v$version, $kind on $ship_branch at $(git rev-parse --short "$ship_head") ($(git log -1 --format=%s "$ship_head"))"
  [ "$ship_dry" -eq 0 ] || ship_say "dry run: each step prints what it would do with what it reads now; nothing is written"
  for step in $SHIP_STEPS; do
    [ "$step" != "$from" ] || started=1
    if [ "$started" -eq 0 ]; then ship_say "$step: skipped (--from $from)"; continue; fi
    "ship_$step"
  done
  if [ "$ship_dry" -eq 1 ]; then
    ship_say "dry run of v$version done: nothing was committed, pushed, tagged, created or dispatched"
  else
    ship_say "v$version is out: $ship_branch at $(git rev-parse --short "$ship_head"), the tag $ship_tag, its GitHub release, $ship_new new dispatch(es) ($ship_had were there already); a run that fails is dispatched again by 'release.sh ship $version --from publish'"
  fi
}

ship_preflight() { # the CI of the commit the bump builds on: the bump moves version files only and ships on its parent's run
  local parent=$ship_head ci=0
  if [ "$(manifest_version "$ship_head")" = "$ship_version" ]; then
    parent=$(git rev-parse "$ship_head^")
    ship_say "preflight: $ship_branch reads $ship_version already; the CI of the bump's parent $(git rev-parse --short "$parent") is read"
  else
    ship_say "preflight: the CI of $ship_branch's head $(git rev-parse --short "$parent"), the commit the bump builds on"
  fi
  ci_rows "$parent" "$ship_branch" || ci=$?
  case $ci in
    1) die "a CI row of $(git rev-parse --short "$parent") is red (above): fix it, or run the matrix (gh workflow run pr-test-rust.yml --ref $ship_branch), then rerun the ship; --from bump skips this preflight once the state is read by hand" ;;
    2) ship_say "preflight: the pending rows above do not hold the ship (the bump changes version files only and ships on its parent's run)" ;;
  esac
  verifier_rows
}

ship_rows() { # <ref> <what it is>: the gate rows a bumped tree answers (its versions, the registries) at <ref>; a row that is not
  # green stops the ship before anything leaves the clone (REGISTRY=0 skips the registry reads and says so)
  local ref=$1 what=$2 state=0 registry=0
  versions_row "$ref" "$ship_version" || state=1
  registry_rows "$ref" "$ship_branch" || registry=$?
  if [ "$registry" -eq 2 ] && [ "${REGISTRY:-1}" = 0 ]; then
    ship_say "the registries were not read (REGISTRY=0): the ship goes on"
  elif [ "$registry" -ne 0 ]; then
    state=1
  fi
  [ "$state" -eq 0 ] || die "a row of $what is not green (above): fix it, then rerun the ship (a bump pull request moves a held version; REGISTRY=0 skips the registry reads)"
}

release_push_allowed() { # <branch>: 0 when the repository's rules let the caller move <branch> by a push: no branch protection of
  # the classic kind, and no pull-request or status-check rule, or the caller on the always-bypass list of every ruleset that
  # carries one. push_block says what stands in the way otherwise. A bypass by team or role is not read here: BUMP_PUSH=1 names it.
  local branch=$1 slug protected rules type id me actors
  push_block=""
  if ! slug=$("$GH" repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null) || [ -z "$slug" ]; then
    push_block="gh does not read the repository (not logged in?)"; return 1
  fi
  protected=$("$GH" api "repos/$slug/branches/${branch//\//%2F}" --jq .protected 2>/dev/null) \
    || { push_block="the branch page of $branch could not be read"; return 1; }
  [ "$protected" = false ] || { push_block="$branch carries branch protection (the classic kind)"; return 1; }
  rules=$("$GH" api "repos/$slug/rules/branches/${branch//\//%2F}" \
    --jq '.[] | select(.type == "pull_request" or .type == "required_status_checks") | "\(.type) \(.ruleset_id)"' 2>/dev/null) \
    || { push_block="the rules of $branch could not be read"; return 1; }
  [ -n "$rules" ] || return 0
  me=$("$GH" api user --jq .id 2>/dev/null) || { push_block="the caller's account could not be read"; return 1; }
  while read -r type id; do
    actors=$("$GH" api "repos/$slug/rulesets/$id" \
      --jq '.bypass_actors[]? | select(.actor_type == "User" and .bypass_mode == "always") | .actor_id' 2>/dev/null) || actors=""
    grep -qx "$me" <<< "$actors" \
      || { push_block="the rule '$type' of ruleset $id applies to $branch and the caller is not on its bypass list"; return 1; }
  done <<< "$rules"
  return 0
}

wait_for_merge() { # <pull request number>: returns once it is merged; MERGE_WAIT minutes at most, read every MERGE_POLL seconds
  local number=$1 state deadline
  deadline=$(( $(date +%s) + ${MERGE_WAIT:-120} * 60 ))
  while :; do
    state=$("$GH" pr view "$number" --json state --jq .state 2>/dev/null) \
      || die "gh pr view $number failed: the bump pull request's state is unknown; rerun the ship once gh answers (an open pull request is waited for)"
    case $state in
      MERGED) return 0 ;;
      CLOSED) die "the bump pull request #$number was closed without a merge: reopen it, or delete its topic branch and rerun the ship" ;;
    esac
    [ "$(date +%s)" -lt "$deadline" ] \
      || die "the bump pull request #$number is not merged after ${MERGE_WAIT:-120} min: merge it, then rerun the ship (it reads the branch bumped and goes on from the tag)"
    sleep "${MERGE_POLL:-30}"
  done
}

bump_landed() { # <topic>: the branch after the bump landed: the remote's head reads the version; a topic the branch carries is dropped
  local base
  fetch
  base=$(release_base "$ship_release"); ship_head=$(git rev-parse "$base")
  [ "$(manifest_version "$ship_head")" = "$ship_version" ] \
    || die "$ship_branch reads $(manifest_version "$ship_head") at $(git rev-parse --short "$ship_head") after the bump landed, not $ship_version: the branch moved under the ship; read it (release.sh status $ship_release) before a rerun"
  ship_say "bump: $ship_branch reads $ship_version at $(git rev-parse --short "$ship_head")"
  if has_ref "refs/heads/$1" && git merge-base --is-ancestor "refs/heads/$1" "$ship_head"; then
    git branch --quiet -D "$1"
  fi
}

ship_bump() { # the bump commit reaches the branch: by a push when the rules allow the caller one, else by its pull request
  local topic="chore/release-$ship_version" number direct reason out url start start_branch
  if [ "$(manifest_version "$ship_head")" = "$ship_version" ]; then
    ship_say "bump: done, $ship_branch reads $ship_version at $(git rev-parse --short "$ship_head")"
    ship_rows "$ship_head" "the branch head"
    return 0
  fi
  number=$("$GH" pr list --head "$topic" --base "$ship_branch" --state open --json number --jq '.[0].number // empty' 2>/dev/null || true)
  if [ -n "$number" ]; then
    if [ "$ship_dry" -eq 1 ]; then ship_would "wait for the merge of the open bump pull request #$number ($topic), ${MERGE_WAIT:-120} min at most"; return 0; fi
    ship_say "bump: the pull request #$number of $topic is open (an earlier run): waiting for its merge (${MERGE_WAIT:-120} min at most, read every ${MERGE_POLL:-30} s)"
    wait_for_merge "$number"
    bump_landed "$topic"
    return 0
  fi
  case "${BUMP_PUSH:-}" in
    1) direct=1; reason="directly (BUMP_PUSH=1)" ;;
    0) direct=0; reason="by a pull request (BUMP_PUSH=0)" ;;
    *) if release_push_allowed "$ship_branch"; then direct=1; reason="directly: no rule the caller cannot bypass requires a pull request on $ship_branch"
       else direct=0; reason="by a pull request: $push_block"; fi ;;
  esac
  if [ "$ship_dry" -eq 1 ]; then
    ship_would "commit 'chore(release): bump versions for v$ship_version' on $topic off $ship_branch (make bump-version, check_release_versions.py --fix, cargo update --workspace), check the versions and the registries at it, and move $ship_branch to it $reason"
    return 0
  fi
  has_ref "refs/heads/$topic" && die "the branch $topic exists (a bump that did not land): delete it (git branch -D $topic; the ship remakes the commit) and rerun"
  require_clean
  start_branch=$(git branch --show-current); start=${start_branch:-$(git rev-parse HEAD)}
  make_bump "$ship_version"
  back_to "$start_branch" "$start"
  ship_rows "refs/heads/$topic" "the bump commit (nothing is pushed; it sits on $topic)"
  if [ "$direct" -eq 1 ]; then
    ship_say "bump: pushing $topic to $ship_branch $reason"
    if out=$(git push "$REMOTE" "refs/heads/$topic:refs/heads/$ship_branch" 2>&1); then
      bump_landed "$topic"
      return 0
    fi
    grep -q 'remote rejected' <<< "$out" || die "the push of $topic to $ship_branch failed:
$out"
    ship_say "bump: $REMOTE refused the push ($(grep -m1 'remote rejected' <<< "$out" | sed 's/^ *//')): opening the bump pull request instead"
  else
    ship_say "bump: opening the pull request of $topic against $ship_branch ($reason)"
  fi
  url=$(open_pr "$topic" "$ship_branch" "$bump_title" "$bump_body") \
    || die "the bump pull request of $topic was not opened (its cause is above): rerun the ship once it is fixed; an open pull request is waited for"
  url=$(tail -n 1 <<< "$url"); number=${url##*/}
  ship_say "bump: pull request #$number opened ($url): waiting for its merge (${MERGE_WAIT:-120} min at most, read every ${MERGE_POLL:-30} s)"
  wait_for_merge "$number"
  bump_landed "$topic"
}

ship_tag() { # the tag at the branch head, pushed; a tag that exists elsewhere stops the ship (tags never move)
  local at
  if remote_has_tag "$ship_tag"; then
    at=$(git ls-remote --tags "$REMOTE" "refs/tags/$ship_tag" "refs/tags/$ship_tag^{}" | tail -n 1 | cut -f1)
    [ "$at" = "$ship_head" ] \
      || die "$ship_tag is on $REMOTE at ${at:0:7}, not at $ship_branch's head $(git rev-parse --short "$ship_head"): tags never move; the ship stops"
    ship_say "tag: done, $ship_tag is on $REMOTE at $(git rev-parse --short "$ship_head")"
    return 0
  fi
  if has_ref "refs/tags/$ship_tag"; then
    at=$(git rev-parse "$ship_tag^{commit}")
    [ "$at" = "$ship_head" ] \
      || die "the tag $ship_tag exists here at ${at:0:7}, not at $ship_branch's head $(git rev-parse --short "$ship_head"): delete it (git tag -d $ship_tag) and rerun"
    if [ "$ship_dry" -eq 1 ]; then ship_would "push the tag $ship_tag (made by an earlier run, at the branch head) to $REMOTE"; return 0; fi
    ship_say "tag: $ship_tag exists here at the branch head (an earlier run): pushing it"
    push_release_ref "refs/tags/$ship_tag" "makes the tag public"
    return 0
  fi
  if [ "$ship_dry" -eq 1 ]; then ship_would "tag $ship_tag at $ship_branch's head once it reads $ship_version (the versions checked against the previous tag) and push it to $REMOTE"; return 0; fi
  cmd_tag "$ship_version"
}

notes_unfilled() { # <file>: what keeps a notes file from publication, one line each (nothing when it is ready): an empty file, a
  # first line that is not the template's for this release, a placeholder the template's check still finds (a fresh render
  # carries the author's, so the ship stops here until the notes are written; the check's lines go to stderr for them)
  local first left
  [ -s "$1" ] || { say "the file is empty"; return 0; }
  first=$(head -n 1 "$root/scripts/release/NOTES-TEMPLATE.md" | sed "s/{{version}}/$ship_version/g")
  head -n 1 "$1" | grep -qxF -- "$first" || say "the first line is not '$first'"
  if ! left=$("$PYTHON" "$root/scripts/release/notes.py" check "$1" 2>&1 >/dev/null); then
    printf '%s\n' "$left" >&2
    left=${left%%$'\n'*} left=${left#notes: }
    say "${left#"$1" } (listed above)"
  fi
}

release_existing() { # <tag>: "url (prerelease: bool)" of the GitHub release on <tag>; fails when there is none
  "$GH" release view "$1" --json url,isPrerelease --jq '"\(.url) (prerelease: \(.isPrerelease))"' 2>/dev/null
}

ship_notes() { # the notes file the GitHub release carries: nothing once the release exists (its notes are out); else NOTES
  # when given, else rendered once (an existing render is kept: it may be edited)
  local unfilled rendered existing
  if existing=$(release_existing "$ship_tag"); then
    ship_say "notes: the GitHub release $ship_tag exists ($existing): its notes are out, nothing is read or rendered"
    ship_notes_file=published # ship_release reads the release itself and never opens this
    return 0
  fi
  rendered=$(git rev-parse --git-path release) # under the git directory, out of the tree; relative to the root unless elsewhere
  case $rendered in /*) ;; *) rendered=$root/$rendered ;; esac
  ship_notes_file=${NOTES:-$rendered/notes-$ship_tag.md}
  if [ -n "${NOTES:-}" ]; then
    [ -s "$NOTES" ] || die "NOTES=$NOTES is missing or empty"
    ship_say "notes: $NOTES (NOTES)"
  elif [ -s "$ship_notes_file" ]; then
    ship_say "notes: $ship_notes_file, rendered by an earlier run (edit it, or delete it for a fresh render)"
  elif [ "$ship_dry" -eq 1 ]; then
    ship_would "render the notes (release.sh notes $ship_version) into $ship_notes_file"
    return 0
  else
    mkdir -p "$(dirname "$ship_notes_file")"
    # rendered beside its place and moved there once whole: a render that stops halfway leaves no file a rerun would keep
    cmd_notes "$ship_version" > "$ship_notes_file.partial"
    mv "$ship_notes_file.partial" "$ship_notes_file"
    ship_say "notes: rendered into $ship_notes_file ($(grep -c . "$ship_notes_file") lines)"
  fi
  unfilled=$(notes_unfilled "$ship_notes_file")
  [ -z "$unfilled" ] || die "the notes at $ship_notes_file are not ready for publication: ${unfilled//$'\n'/; } (fill them, then rerun the ship: --from notes)"
}

ship_release() { # the GitHub release on the tag: a prerelease for a candidate, latest for a final; one that exists is kept
  local flags=() existing url
  [ -n "$ship_notes_file" ] || ship_notes # --from release: the notes step ran in an earlier run, or runs now
  if is_final "$ship_version"; then flags=(--latest); else flags=(--prerelease); fi
  if existing=$(release_existing "$ship_tag"); then
    ship_say "release: done, $existing"
    return 0
  fi
  if [ "$ship_dry" -eq 1 ]; then ship_would "create the GitHub release: gh release create $ship_tag --verify-tag ${flags[*]} --title '$ship_tag' --notes-file $ship_notes_file"; return 0; fi
  url=$("$GH" release create "$ship_tag" --verify-tag "${flags[@]}" --title "$ship_tag" --notes-file "$ship_notes_file") \
    || die "gh release create $ship_tag failed (its message is above): rerun the ship once the cause is fixed (--from release)"
  ship_say "release: created $url (${flags[*]#--})"
}

run_rows() { # <tag> <workflow file>: the workflow's runs on <tag>, "id<TAB>title<TAB>status<TAB>conclusion" per line, with gh's
  # status: a read that failed is not an empty list (an empty list would dispatch the whole plan again)
  "$GH" run list --workflow "$2" --branch "$1" --limit 100 --json databaseId,displayTitle,status,conclusion \
    --jq '.[] | "\(.databaseId)\t\(.displayTitle)\t\(.status)\t\(.conclusion // "")"'
}
dispatched() { # <dispatch line> <its workflow's runs>: 0 when a run of this dispatch that did not fail is among them (dispatched_run
  # names it). An engine-image dispatch is told by its run name, which carries the engine and the base image.
  local line=$1 rows=$2 engine="" base="engine default" id title status conclusion
  case "$line" in *" -f engine="*) engine=${line#* -f engine=}; engine=${engine%% *} ;; esac
  case "$line" in *" -f base_image_ref="*) base=${line#* -f base_image_ref=}; base=${base%% *} ;; esac
  while IFS=$'\t' read -r id title status conclusion; do
    [ -n "$id" ] || continue
    if [ -n "$engine" ]; then case "$title" in *"SMG+$engine | base=$base |"*) ;; *) continue ;; esac; fi
    case "$conclusion" in failure | cancelled | timed_out | startup_failure | action_required | stale) continue ;; esac
    dispatched_run="$id (${conclusion:-$status})"
    return 0
  done < "$rows"
  return 1
}

ship_publish() { # the publish plan's dispatches against the tag, each once; then the runs on the tag
  local ref=$ship_tag plan cache line words file failed=0 id workflow status conclusion title table
  local lines=()
  if ! has_ref "refs/tags/$ship_tag"; then
    [ "$ship_dry" -eq 1 ] || die "the tag $ship_tag does not exist: the dispatches run against it (release.sh ship $ship_version --from tag)"
    ref=$ship_head
    ship_say "publish: the plan is read at $(git rev-parse --short "$ref"), before the bump: its dispatch lines are the tag's, its crate versions the bump's once it exists"
  fi
  # The plan is read whole, with its exit status, before the first dispatch: a plan that fails after its first lines
  # (the component list, a registry) dispatches nothing. Its dispatch lines are read from the file, without the comments.
  plan=$(mktemp)
  if ! (publish_plan "$ship_version" "$ref") > "$plan"; then
    die "the publish plan of v$ship_version could not be read (its message is above): nothing is dispatched"
  fi
  mapfile -t lines < <(sed -n 's/^ *\(gh workflow run .*\)$/\1/p' "$plan" | sed 's/ *#.*$//')
  [ ${#lines[@]} -gt 0 ] || die "the publish plan of v$ship_version names no dispatch"
  cache=$(mktemp -d)
  for line in ${lines[@]+"${lines[@]}"}; do
    read -r -a words <<< "$line" # gh workflow run <file> --ref <tag> [-f <input>=<value>]...
    file=${words[3]}
    if [ ! -f "$cache/$file" ] && ! run_rows "$ship_tag" "$file" > "$cache/$file"; then
      die "the runs of $file on $ship_tag could not be read (gh run list --workflow $file --branch $ship_tag; its message is above): a run that is there must not be dispatched twice, so nothing more is dispatched; rerun the ship (--from publish) once gh answers (what was dispatched is skipped then)"
    fi
    if dispatched "$line" "$cache/$file"; then
      ship_say "publish: done, $line: run $dispatched_run"
      ship_had=$((ship_had + 1))
    elif [ "$ship_dry" -eq 1 ]; then
      ship_would "dispatch: $line"
    elif "$GH" "${words[@]:1}" >/dev/null; then
      ship_say "publish: dispatched $line"
      ship_new=$((ship_new + 1))
    else
      ship_say "publish: FAILED $line (gh's message is above)"
      failed=$((failed + 1))
    fi
  done
  [ "$ship_dry" -eq 0 ] || return 0
  ship_say "publish: $ship_new dispatched, $ship_had there already, $failed failed"
  [ "$ship_new" -eq 0 ] || sleep 5 # a dispatch's run is listed a moment after gh returns
  say "the runs on $ship_tag (gh run list --branch $ship_tag):"
  if table=$("$GH" run list --branch "$ship_tag" --limit 100 --json databaseId,workflowName,status,conclusion,displayTitle \
    --jq '.[] | "\(.databaseId)\t\(.workflowName)\t\(.status)\t\(.conclusion // "")\t\(.displayTitle)"'); then
    while IFS=$'\t' read -r id workflow status conclusion title; do
      [ -n "$id" ] || continue
      printf '  %-12s %-36s %-20s %s\n' "$id" "$workflow" "$status${conclusion:+ $conclusion}" "$title"
    done <<< "$table"
  else
    say "  (the run list could not be read; the dispatches above stand)"
  fi
  [ "$failed" -eq 0 ] || die "$failed dispatch(es) failed: rerun the ship (--from publish) once the cause is fixed; the ones that ran are skipped"
}

command=${1:-help}
shift || true
case "$command" in
  cut) cmd_cut "$@" ;;
  bump) cmd_bump "$@" ;;
  pick) cmd_pick "$@" ;;
  pick-pr) cmd_pick_pr "$@" ;;
  status) cmd_status "$@" ;;
  tag) cmd_tag "$@" ;;
  notes) cmd_notes "$@" ;;
  bump-main) cmd_bump_main "$@" ;;
  publish-plan) cmd_publish_plan "$@" ;;
  check) cmd_check "$@" ;;
  ship) cmd_ship "$@" ;;
  help | -h | --help) usage ;;
  *) usage >&2; die "unknown command: $command" ;;
esac
