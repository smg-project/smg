#!/usr/bin/env bash
# The release process: a release branch per minor version, cut from main and moved by
# pull requests, so main never pauses for a release.
#
#   release.sh cut <version> [<from>]      cut release/<x.y> from <from> (default <remote>/main) and commit the
#                                          version bump to <version> on it (nothing is pushed unless PUSH=1)
#   release.sh bump <version>              the next candidate or the final on an existing release/<x.y>: a bump
#                                          commit on a topic branch and its pull request against the release branch
#   release.sh pick <x.y> <sha>[,<sha>..]  cherry-pick commits of main (-x -s) onto a topic branch off
#                                          release/<x.y> and open the pull request; a conflict stops with the steps
#   release.sh pick-pr <x.y>               after a resolved conflict: push the pick branch and open its pull request
#   release.sh status <x.y>                what is on main and not on release/<x.y>, the branch's own commits,
#                                          its tags, the open pull requests against it
#   release.sh tag <version>               check the versions at release/<x.y>'s head and tag it v<version>
#                                          (annotated, signed off; pushed only with PUSH=1)
#   release.sh notes <version> [<since>]   the release notes skeleton on stdout: the merged pull requests since the
#                                          previous final (or <since>) by type, the Docker image list
#   release.sh bump-main <version>         after a final: a pull request that moves main's version files to the
#                                          release's (the gateway to <version>, every crate at least the published)
#   release.sh publish-plan <version>      the publishing steps for v<version>: the workflow dispatches a maintainer
#                                          runs against the tag and the artifact names they produce (prints, runs nothing)
#   release.sh check <version>             the release gate for v<version>: what the repository can read (the versions
#                                          at the branch head, its CI, the tag, main's drift) and the lines the release
#                                          owner fills (the four model-provider verifiers)
#
# No push publishes anything: a bump on a release branch runs CI; the artifacts (wheels, crates, images, the chart)
# come from the release-* workflows dispatched by hand against the release tag, as publish-plan lists them.
#
# <version> is X.Y.Z or X.Y.Z-rc.N; the release branch is release/X.Y. Every version file carries the one
# string as written (Python tooling normalizes 1.14.0-rc.1 to 1.14.0rc1 on its own).
#
# Environment: REMOTE (origin); PUSH (cut and tag push nothing unless PUSH=1; bump, pick and bump-main push
# their topic branch and open the pull request unless PUSH=0); GH, CARGO, PYTHON, MAKE (the tools).
# The commits, the picks and the tag carry the caller's git identity and sign-off.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 # the version checker is imported from the tree; no scripts/__pycache__ left behind

REMOTE=${REMOTE:-origin}
GH=${GH:-gh}
CARGO=${CARGO:-cargo}
PYTHON=${PYTHON:-python3}
MAKE=${MAKE:-make}
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
version_above() { # <a> <b>: a reads above b (a candidate sorts below its final: 1.14.0-rc.1 < 1.14.0)
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sed 's/-rc\./~rc./' | sort -V | tail -n 1 | sed 's/~rc\./-rc./')" = "$1" ]
}

fetch() { git fetch --quiet --tags "$REMOTE" || die "cannot fetch $REMOTE"; }
has_ref() { git rev-parse --verify --quiet "$1^{commit}" >/dev/null 2>&1; }
remote_has_branch() { git ls-remote --exit-code --heads "$REMOTE" "$1" >/dev/null 2>&1; }
remote_has_tag() { git ls-remote --exit-code --tags "$REMOTE" "refs/tags/$1" >/dev/null 2>&1; }
release_base() { # the remote-tracking release branch, fetched
  local branch="release/$1"
  has_ref "$REMOTE/$branch" || die "no $branch on $REMOTE: cut it first (release.sh cut <version> [<from>])"
  printf '%s/%s' "$REMOTE" "$branch"
}
previous_tag() { # the newest release tag reachable from $1
  git describe --tags --abbrev=0 --match 'v[0-9]*' "$1" 2>/dev/null \
    || die "no release tag (v<x.y.z>) is reachable from $1"
}
previous_final_tag() { # the highest final tag below v<version>; empty when there is none
  { git tag --list 'v[0-9]*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; printf 'v%s\n' "$1"; } \
    | sort -uV | grep -B1 -x "v$1" | head -1 | grep -v -x "v$1" || true
}
since_tag() { # <ref> <version>: the tag a bump or a check compares the versions against
  # The branch's own newest tag when it has one (made on the release branch: the last candidate);
  # else the highest final below <version>, which sits on the previous release's branch from the
  # second release on (no release tag is reachable from main under this process; the checker
  # diffs tag..HEAD, so the tag need not be an ancestor); else the newest tag reachable from
  # <ref>, a release made before this process.
  # A candidate sorts below its final: a version sort puts `~` before the end of the string.
  local ref=$1 version=$2 own final
  own=$(comm -23 <(git tag --list 'v[0-9]*' --merged "$ref" | sort) <(git tag --list 'v[0-9]*' --merged "$REMOTE/main" | sort) \
    | sed 's/-rc\./~rc./' | sort -V | tail -n 1 | sed 's/~rc\./-rc./')
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
require_clean() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] \
    || die "the working tree has uncommitted changes: commit them first (the bump commit must carry only the bump)"
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
refresh_lock() { # Cargo.lock follows the workspace crates' versions
  [ -f "$root/Cargo.lock" ] || return 0
  command -v "$CARGO" >/dev/null 2>&1 \
    || die "Cargo.lock needs a refresh and '$CARGO' is not on PATH (set CARGO=<path to cargo>)"
  note "bump: cargo update --workspace (Cargo.lock follows the crate versions)"
  "$CARGO" update --workspace --offline >/dev/null 2>&1 || "$CARGO" update --workspace >/dev/null
}
commit_all() { # <subject> <body>
  git add --update
  git diff --cached --quiet && die "nothing to commit: the tree already carries these versions"
  git commit --quiet --signoff --message "$1" --message "$2"
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
open_pr() { # <topic> <base> <title> <body file>
  local topic=$1 base=$2 title=$3 body=$4
  if [ "${PUSH:-}" = 0 ]; then
    say ""
    say "Not pushed (PUSH=0). To open the pull request:"
    say "  git push -u $REMOTE refs/heads/$topic:refs/heads/$topic"
    say "  gh pr create --base $base --head $topic --title $(printf '%q' "$title") --body-file $(printf '%q' "$body")"
    return 0
  fi
  command -v "$GH" >/dev/null 2>&1 || die "'$GH' is not installed: push $topic and open the pull request against $base by hand"
  git push --quiet -u "$REMOTE" "refs/heads/$topic:refs/heads/$topic"
  "$GH" pr create --base "$base" --head "$topic" --title "$title" --body-file "$body"
}

# --- the commands -------------------------------------------------------------------

cmd_cut() {
  local version=${1:-} from=${2:-}
  [ -n "$version" ] || die "usage: release.sh cut <version> [<from>]"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release branch from_sha since
  release=$(release_of "$version"); branch="release/$release"
  require_identity; require_clean; fetch
  from=${from:-$REMOTE/main}
  from_sha=$(git rev-parse --verify --quiet "$from^{commit}") || die "unknown commit: $from"
  git merge-base --is-ancestor "$from_sha" "$REMOTE/main" \
    || die "$from is not on $REMOTE/main: a release is cut from main's history"
  has_ref "refs/heads/$branch" && die "$branch exists locally: a new candidate on it is 'release.sh bump $version'"
  remote_has_branch "$branch" && die "$branch exists on $REMOTE: a new candidate on it is 'release.sh bump $version'"
  since=$(since_tag "$from_sha" "$version")
  # The previous release's final is the right tag only once main carries that release's versions: a cut before the
  # release-bump-main pull request merged (or from an older commit) would read a crate bumped for that release as
  # bumped already, leave main's lower version, and ship nothing for it while every check passes.
  if ! git merge-base --is-ancestor "$since" "$from_sha" && version_above "${since#v}" "$(manifest_version "$from_sha")"; then
    die "$from reads $(manifest_version "$from_sha"), below ${since#v}: merge the release-bump-main pull request for $since first (release.sh bump-main ${since#v}), or cut from a commit after it"
  fi
  note "cut: $branch from $(git rev-parse --short "$from_sha") ($from); the versions move from $since to $version"
  git switch --quiet --no-track -c "$branch" "$from_sha"
  bump_tree "$version" "$since"
  commit_all "chore(release): bump versions for v$version" \
    "The release branch $branch, cut from $from_sha ($from). Versions since $since."
  say ""
  say "$branch is at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"
  git show --stat --oneline HEAD | sed -n '2,$p' | sed 's/^/  /'
  say "(You are on $branch now; the release commands run from a checkout that has scripts/release/, main for instance.)"
  push_release_ref "refs/heads/$branch:refs/heads/$branch" "runs the test matrix on $branch (nothing is published); then 'release.sh tag $version'"
}

cmd_bump() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh bump <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release branch base since topic body kind
  release=$(release_of "$version"); branch="release/$release"
  require_identity; require_clean; fetch
  base=$(release_base "$release")
  [ "$(manifest_version "$base")" != "$version" ] || die "$branch already reads $version"
  since=$(since_tag "$base" "$version")
  topic="chore/release-$version"
  has_ref "refs/heads/$topic" && die "the branch $topic exists: delete it or finish it"
  if is_final "$version"; then kind="the final"; else kind="the next candidate"; fi
  note "bump: $kind on $branch, versions since $since"
  git switch --quiet --no-track -c "$topic" "$base"
  bump_tree "$version" "$since"
  commit_all "chore(release): bump versions for v$version" "$kind on $branch. Versions since $since."
  body=$(mktemp)
  {
    say "The version bump for v$version on \`$branch\` ($kind). Versions since $since:"
    say ""
    say '```'
    git show --stat --oneline HEAD | sed -n '2,$p'
    say '```'
    say ""
    say "Merging it makes \`$branch\` read v$version (nothing is published by the merge); then \`make release-tag VERSION=$version\` and \`make release-publish-plan VERSION=$version\`."
    say ""
    gate_section "$version" "$base"
  } > "$body"
  open_pr "$topic" "$branch" "chore(release): bump versions for v$version" "$body"
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
  has_ref "refs/heads/$topic" && die "the branch $topic exists: delete it or finish it (release.sh pick-pr $release)"
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
    say "    3. make release-pick-pr RELEASE=$release   (pushes $topic and opens the pull request against release/$release)"
    exit 1
  fi
  pick_pr "$release" "$topic"
}

cmd_pick_pr() {
  local release=${1:-}
  [ -n "$release" ] || die "usage: release.sh pick-pr <x.y>"
  release_ok "$release"
  local topic
  topic=$(git rev-parse --abbrev-ref HEAD)
  case "$topic" in pick/"$release"-*) ;; *) die "not on a pick/$release-* branch (on $topic)" ;; esac
  has_ref CHERRY_PICK_HEAD && die "a cherry-pick is still in progress: git cherry-pick --continue (or --abort) first"
  [ ! -d "$(git rev-parse --git-path sequencer)" ] || die "a cherry-pick sequence is still in progress: git cherry-pick --continue (or --abort) first"
  require_clean
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
  base=$(release_base "$release")
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
  say "== tags on release/$release:"
  git tag --list 'v[0-9]*' --merged "$base" | sort -V | sed 's/^/  /'
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
  local version=${1:-} since=${2:-}
  [ -n "$version" ] || die "usage: release.sh notes <version> [<since>]"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release tagname ref subject type count picked full original unknown line path seen=' '
  local lines=() subjects=()
  local feat=() fix=() perf=() docs=() maint=() other=()
  release=$(release_of "$version"); tagname="v$version"
  fetch
  if has_ref "refs/tags/$tagname"; then
    ref=$tagname
  else
    ref=$(release_base "$release")
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
      case "$subject" in 'chore(release): bump versions'*) continue ;; esac
      type=${subject%%[(:!]*}
      case "$type" in
        feat) feat+=("$subject") ;;
        fix) fix+=("$subject") ;;
        perf) perf+=("$subject") ;;
        docs) docs+=("$subject") ;;
        ci | chore | build | test | refactor | style) maint+=("$subject") ;;
        *) other+=("$subject") ;;
      esac
    done
  done < <(git log --cherry-pick --right-only --first-parent --no-merges --format='%H %s' "$since...$ref")
  say "## SMG $tagname"
  say ""
  say "$count commits since $since on \`release/$release\` (the merged pull requests, by type):"
  section() { # <title> <subjects...>
    local title=$1; shift
    [ $# -gt 0 ] || return 0
    say ""
    say "### $title"
    say ""
    printf -- '- %s\n' "$@"
  }
  section "Features" ${feat[@]+"${feat[@]}"}
  section "Fixes" ${fix[@]+"${fix[@]}"}
  section "Performance" ${perf[@]+"${perf[@]}"}
  section "Documentation" ${docs[@]+"${docs[@]}"}
  section "Maintenance" ${maint[@]+"${maint[@]}"}
  section "Other" ${other[@]+"${other[@]}"}
  say ""
  if ! "$root/scripts/release_docker_notes.sh" "$tagname" 2>/dev/null; then
    say "### Docker images"
    say ""
    say "(no engine images for $tagname yet: run \`scripts/release_docker_notes.sh $tagname\` once they are published)"
  fi
  say ""
  if is_final "$version"; then
    note "the GitHub release: gh release create $tagname --verify-tag --title 'SMG $tagname' --notes-file <this output>"
  else
    note "the GitHub release: gh release create $tagname --verify-tag --prerelease --title 'SMG $tagname' --notes-file <this output>"
  fi
}

cmd_bump_main() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh bump-main <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  is_final "$version" || die "main follows a final release (X.Y.Z), not a candidate"
  local release source topic body current moved
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
  git switch --quiet --no-track -c "$topic" "$REMOTE/main"
  current=$(manifest_version "$REMOTE/main")
  # Main never goes back: after a patch of an older line (v1.14.1 once v1.15.0 is out), the gateway and its mirrors
  # keep main's higher version (the nightly and the next cut derive from it) and only the crates and packages the
  # patch raised move, as sync_versions.py moves them: to at least the release's, never lower.
  if version_above "$current" "$version"; then
    note "bump-main: main reads $current, above $version: the gateway keeps it; the crates and packages v$version raised move"
    moved="The gateway keeps $current on main (above v$version)"
  else
    note "bump-main: make bump-version VERSION=$version"
    "$MAKE" --no-print-directory -C "$root" bump-version "VERSION=$version" >/dev/null
    moved="The gateway reads $version on main"
  fi
  note "bump-main: scripts/release/sync_versions.py $source $version (every crate and package at least the release's)"
  "$PYTHON" "$root/scripts/release/sync_versions.py" "$source" "$version"
  refresh_lock
  if git diff --quiet && git diff --cached --quiet; then
    git switch --quiet - 2>/dev/null || git switch --quiet --detach '@{-1}'
    git branch --quiet -D "$topic"
    die "bump-main: main carries v$version's versions already (the gateway at $current, every crate and package at least the release's): nothing to move"
  fi
  commit_all "chore(release): main's versions follow v$version" \
    "$moved and every crate and package at least the version released from release/$release (crates.io and PyPI never take a version twice). Main's artifacts are not published; the release branches' are."
  body=$(mktemp)
  {
    say "After the v$version release: main's version files follow the release (\`$source\`), so the next cut starts above what crates.io and PyPI already carry."
    say ""
    say '```'
    git show --stat --oneline HEAD | sed -n '2,$p'
    say '```'
  } > "$body"
  open_pr "$topic" main "chore(release): main's versions follow v$version" "$body"
}

pep440() { # 1.14.0-rc.1 -> 1.14.0rc1, the spelling PyPI shows
  printf '%s' "$1" | sed 's/-rc\./rc/'
}
ref_version() { # <ref> <file>: the version line a file carries at a ref (empty when the file is not there)
  git show "$1:$2" 2>/dev/null | grep -m1 '^version = ' | sed 's/version = "\(.*\)"/\1/' || true
}
crate_versions() { # <ref>: "name version" of every workspace crate the version checker lists, at a ref
  "$PYTHON" - "$1" <<'PYEOF'
import importlib.util
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_release_versions", Path("scripts/check_release_versions.py"))
crv = importlib.util.module_from_spec(spec)
sys.modules["check_release_versions"] = crv  # its dataclasses resolve their annotations through sys.modules
spec.loader.exec_module(crv)
ref = sys.argv[1]
for crate in crv.CRATES:
    shown = subprocess.run(["git", "show", f"{ref}:{crate.version_file}"], capture_output=True, text=True)
    if shown.returncode == 0:
        print(crate.name, crv.read_version("cargo", shown.stdout, crate.version_file))
PYEOF
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

cmd_publish_plan() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh publish-plan <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release tagname ref latest proto servicer
  release=$(release_of "$version"); tagname="v$version"
  fetch
  if has_ref "refs/tags/$tagname"; then
    ref=$tagname
  else
    ref=$(release_base "$release")
    note "publish-plan: $tagname is not tagged yet; the plan reads release/$release's head and names the tag as the ref to dispatch on"
  fi
  [ "$(manifest_version "$ref")" = "$version" ] || die "$ref reads $(manifest_version "$ref"), not $version"
  if is_final "$version"; then latest="and latest"; else latest="(a candidate: latest does not move)"; fi
  proto=$(ref_version "$ref" crates/grpc_client/python/pyproject.toml)
  servicer=$(ref_version "$ref" grpc_servicer/pyproject.toml)
  say "Publishing v$version: nothing below runs by itself. A maintainer dispatches each workflow against the tag"
  say "$tagname (pushed by release.sh tag $version); every run reads its versions from that ref."
  say ""
  say "1. Python package (PyPI: smg==$(pep440 "$version"), wheels + sdist)"
  say "   gh workflow run release-pypi.yml --ref $tagname"
  say "2. Rust crates (crates.io: every workspace crate whose version at $tagname is not published yet; smg $version)"
  say "   gh workflow run release-crates.yml --ref $tagname"
  say "   crate versions at $tagname:"
  crate_versions "$ref" | sed 's/^/     /'
  say "3. Gateway image (Docker Hub lightseekorg/smg:$version and ghcr.io/smg-project/smg:$version $latest)"
  say "   gh workflow run release-docker.yml --ref $tagname"
  say "4. Helm chart (oci://ghcr.io/smg-project/charts/smg, version $version)"
  say "   gh workflow run release-helm.yml --ref $tagname"
  say "5. Engine images (ghcr.io/smg-project/smg:$version-<engine>-<engine version>, built from SMG_RELEASE_TAG = $tagname at that ref;"
  say "   one dispatch per engine and release base, the table of .github/workflows/release-engine-docker.yml at $tagname: a dispatch"
  say "   without base_image_ref builds the engine's first base only, so every other base is a dispatch of its own)"
  engine_dispatches "$ref" "$tagname" | sed 's/^/   /'
  say "6. gRPC Python packages (PyPI: smg-grpc-proto==${proto:-?}, smg-grpc-servicer==${servicer:-?}; versions already on PyPI are skipped)"
  say "   gh workflow run release-grpc.yml --ref $tagname"
  say "7. Go module tag (crates/grpc_client/go/v$version)"
  say "   gh workflow run release-go.yml --ref $tagname -f version=$version"
  say "8. The GitHub release: release.sh notes $version > notes.md, then the gh release create line it prints."
  say ""
  say "Watch a dispatch: gh run list --workflow <file> --branch $tagname"
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

cmd_check() {
  local version=${1:-}
  [ -n "$version" ] || die "usage: release.sh check <version>"
  version_ok "$version" || die "not a version: '$version' (X.Y.Z or X.Y.Z-rc.N)"
  local release branch base head tagname carried since scratch slug checks runs drift=0 family failed=0 full picked
  release=$(release_of "$version"); branch="release/$release"; tagname="v$version"
  fetch
  base=$(release_base "$release"); head=$(git rev-parse "$base")
  say "Release gate for v$version ($branch at $(git rev-parse --short "$head"): $(git log -1 --format=%s "$head"))"
  say ""
  # 1. the versions at the head
  carried=$(manifest_version "$head")
  if [ "$carried" = "$version" ]; then
    since=$(since_tag "$head" "$version")
    scratch=$(mktemp -d)
    git worktree add --quiet --detach "$scratch" "$head"
    if (cd "$scratch" && "$PYTHON" scripts/check_release_versions.py --check "$since" >/dev/null 2>&1); then
      say "[ok]   versions: the branch reads $version and every crate, package and mirror is consistent with $since"
    else
      say "[FAIL] versions: the branch reads $version but scripts/check_release_versions.py --check $since finds issues (a bump pull request fixes them)"
      failed=1
    fi
    git worktree remove --force "$scratch"
  else
    say "[FAIL] versions: the branch reads $carried, not $version (release.sh bump $version)"
    failed=1
  fi
  # 2. CI on the head
  if command -v "$GH" >/dev/null 2>&1 && slug=$("$GH" repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null) && [ -n "$slug" ]; then
    checks=$("$GH" api --paginate "repos/$slug/commits/$head/check-runs?per_page=100" --jq '.check_runs[] | "\(.conclusion // .status) \(.name)"' 2>/dev/null || true)
    if [ -z "$checks" ]; then
      say "[....] CI: no check run on $(git rev-parse --short "$head") yet (https://github.com/$slug/commits/$head)"
      failed=1
    else
      local ok=0 bad=0 pending=0 line
      while IFS= read -r line; do
        case "$line" in
          success\ *|skipped\ *|neutral\ *) ok=$((ok + 1)) ;;
          failure\ *|cancelled\ *|timed_out\ *|action_required\ *|stale\ *) bad=$((bad + 1)); say "       red: ${line#* }" ;;
          *) pending=$((pending + 1)) ;;
        esac
      done <<< "$checks"
      if [ "$bad" -eq 0 ] && [ "$pending" -eq 0 ]; then
        say "[ok]   CI: $ok check(s) green on $(git rev-parse --short "$head")"
      else
        say "[FAIL] CI: $ok green, $bad red, $pending pending on $(git rev-parse --short "$head") (https://github.com/$slug/commits/$head)"
        failed=1
      fi
    fi
    # The test matrix on this very commit: a documentation-only merge runs none (the push trigger's paths-ignore)
    # and the pull request's run was for the merge ref, not the branch head, so a run is required by head sha.
    runs=$("$GH" api "repos/$slug/actions/workflows/pr-test-rust.yml/runs?head_sha=$head&per_page=20" --jq '.workflow_runs[] | "\(.conclusion // .status) \(.html_url)"' 2>/dev/null || true)
    if [ -z "$runs" ]; then
      say "[FAIL] PR Test (SMG): no run on $(git rev-parse --short "$head") (a documentation-only merge runs none): gh workflow run pr-test-rust.yml --ref $branch"
      failed=1
    elif grep -q '^success ' <<< "$runs"; then
      say "[ok]   PR Test (SMG): a successful run on $(git rev-parse --short "$head") ($(grep -m1 '^success ' <<< "$runs" | cut -d' ' -f2))"
    else
      say "[FAIL] PR Test (SMG): no successful run on $(git rev-parse --short "$head") yet: $(tr '\n' ';' <<< "$runs")"
      failed=1
    fi
  else
    say "[....] CI: not readable here (gh is not installed or not logged in): read the checks of $(git rev-parse --short "$head") on the branch page"
  fi
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
  say "[....] model-provider verifiers against this candidate build (the release owner fills the record links; they go into the final's bump pull request):"
  for family in $VERIFIER_FAMILIES; do
    say "       [ ] $family provider verifier: <record link>"
  done
  say ""
  if [ "$failed" -eq 0 ]; then
    say "The repository-readable items pass; the verifier lines are the release owner's."
  else
    say "A repository-readable item is not green yet."
  fi
  return "$failed"
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
  help | -h | --help) usage ;;
  *) usage >&2; die "unknown command: $command" ;;
esac
