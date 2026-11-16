Another automated reviewer posted findings on this pull request. Grade each one against the
code. Your output is consumed by a script, so the final message must be the JSON the schema
describes and nothing else: no prose around it, no code fence.

## Inputs

- `.codex-review/context.json`: `pr`, `title`, `head_sha`, `other_bot`, and `targets`, one per
  finding to grade, each with `comment_id`, `path`, `line`, `diff_hunk` and `body`.
- `.codex-review/pr.diff`: the whole pull request.
- The checkout is at `head_sha`. Read the files the findings point at and whatever they
  depend on. Read-only sandbox, no network.

## Method

For every target, exactly once, with its `comment_id` copied verbatim:

- `agree`: the defect is real and the finding describes it correctly.
- `disagree`: the code does not have the defect, or the finding misreads it. Say what the
  code actually does, with a line reference.
- `unsure`: the finding may be right but the code alone cannot settle it, for example it
  depends on a runtime invariant or a caller outside the repository. Say what would settle it.

`reason` is at most two sentences of evidence. Do not restate the finding, do not grade its
tone or severity label, and do not add findings of your own.
