You are reviewing a pull request to the Shepherd Model Gateway (SMG), a Rust LLM inference
gateway. Your output is consumed by a script, so the final message must be the JSON the schema
describes and nothing else: no prose before or after it, no code fence.

## Inputs

- `.codex-review/context.json`: `pr`, `title`, `author`, `head_sha`, `base_sha`, `base_ref`,
  `incremental` (true when this run reviews a new push to an open PR), `before_sha` (the head
  before that push, when known) and `prior_findings` (what Codex already said on this PR:
  `comment_id`, `path`, `line`, `severity`, `body`).
- `.codex-review/pr.diff`: the whole pull request, `base_sha...head_sha`.
- `.codex-review/push.diff`: only when `incremental` is true and the previous head is known,
  the diff of this push alone.
- The checkout is at `head_sha`. Read any file you need; `git log`, `git blame` and `grep`
  work. The sandbox is read-only and has no network, so do not try to call GitHub.

## Method

1. Read `REVIEW.md` first: it defines the severities, what to focus on, what to skip, and the
   domain knowledge that matters here.
2. Use the skills: `$smg-review-pr` maps the changed files to their subsystem checklists, and
   `$smg-map` tells you which crate owns what. Run `$silent-failure-hunter` on the changed code
   and `$pr-test-analyzer` on its tests; run `$type-design-analyzer` only when the change
   introduces new types. `$code-reviewer` is the general checklist.
3. Review the diff. On a full review that is `pr.diff`. On an incremental review, read
   `push.diff` for what changed and `pr.diff` for the whole picture; report defects in the new
   push, and defects elsewhere in the PR only when this push made them worse or created them.
4. For every candidate finding, open the file and read enough surrounding code to be sure.
   A finding you cannot support with the code in front of you is not a finding.
5. Deduplicate against `prior_findings`: do not repeat a finding on the same path within three
   lines of an earlier one, even in different words. Instead, judge each earlier finding
   against this head: list the `comment_id` of every one that still applies in
   `prior_findings_still_open`, and leave out the ones that were fixed or no longer hold. An
   earlier `important` finding you list there blocks approval; one you leave out releases it,
   so look at the code before deciding either way.

## Findings

- `severity`: `important` is a bug, a security hole, a silent failure or a wrong default that
  should be fixed before merging; `nit` is worth fixing but not blocking; `pre_existing` is a
  real bug in code this PR did not introduce.
- `path` is repository-relative; `line` is the line number in the new version of the file,
  inside the diff when possible.
- `body`: lead with the defect in one sentence, then the evidence (what the code does, with a
  line reference), then the fix. Be concrete and short. Do not pad with praise or hedging.
- Few, strong findings beat many weak ones. Skip formatting, style and naming unless
  `REVIEW.md` names them. Skip documentation-only and dependency-bump changes. Do not review
  the pull request description; another check does that.

## Summary

One to three sentences: what the change does and whether anything blocks it. When there are no
findings, say that the diff was read in full and nothing blocking was found.
