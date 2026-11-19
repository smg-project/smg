#!/bin/bash
# Of the engines on stdin, print those no open engine-watch issue reports.
#
# stdin: a JSON array of the version check's rows ({"engine","current","latest",...},
# scripts/check_engine_versions.sh). stdout: the same rows, one compact JSON array,
# for the engines whose latest version no open issue labelled engine-watch names
# in its title ("[engine-watch] <engine> <latest> available ...").
#
# The titles are listed from the repository and matched here, not searched for.
# GitHub's in:title search normalizes dots and hyphens, so "sglang 0.5.16" can
# match unrelated issues, or miss; and the search index, which
# `gh issue list --label` asks as well, can trail an issue filed seconds ago,
# which is when the watch asks a second time. The engine's name and the version
# after it are whole words ("1.4.0rc3" does not report "1.4.0"); a commit pin
# (TokenSpeed's) is matched by prefix, because the titles carry its SHA shortened.
#
# Requires: gh, jq; GH_TOKEN and GITHUB_REPOSITORY.

set -euo pipefail

rows=$(cat)
# The loop reads the rows through a process substitution, whose failure the
# shell does not see: without this, input that is no array would print [].
[ "$(jq -r type <<<"$rows")" = array ] ||
    { echo "engine_watch_pending: stdin is not a JSON array" >&2; exit 1; }
titles=$(gh api --paginate \
    "repos/$GITHUB_REPOSITORY/issues?state=open&labels=engine-watch&per_page=100" \
    --jq '.[] | select(has("pull_request") | not) | .title')

pending='[]'
while IFS= read -r row; do
    engine=$(jq -r .engine <<<"$row")
    latest=$(jq -r .latest <<<"$row")
    reported=
    while IFS= read -r title; do
        named=${title#*" $engine "}
        [ "$named" != "$title" ] || continue
        named=${named%% *}
        if [ "$named" = "$latest" ] ||
            [[ $latest =~ ^[0-9a-f]{40}$ && $named =~ ^[0-9a-f]{7,40}$ &&
                $latest == "$named"* ]]; then
            reported=1
            break
        fi
    done <<<"$titles"
    if [ -n "$reported" ]; then
        echo "$engine $latest: an open issue reports it" >&2
    else
        pending=$(jq -c --argjson r "$row" '. + [$r]' <<<"$pending")
    fi
done < <(jq -c '.[]' <<<"$rows")
echo "$pending"
