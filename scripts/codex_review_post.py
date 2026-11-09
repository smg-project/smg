#!/usr/bin/env python3
"""The Codex code review's two GitHub halves: the context Codex reviews from, and the posting.

Run by .github/workflows/codex-code-review.yml:

    codex_review_post.py context --out-dir DIR
        Reads the workflow event (GITHUB_EVENT_NAME, GITHUB_EVENT_PATH), resolves the pull
        request, decides the run's mode (review, reply, verify) and whether it should run at
        all, and writes DIR/context.json for the prompt. Needs a read token.

    codex_review_post.py post --dir DIR
        Reads DIR/context.json and DIR/output.json (Codex's final message, shaped by the
        mode's JSON schema) and posts: one pull-request review with inline comments, a reply
        in a thread, or one verdict per finding of the other bot. Needs pull-requests: write.

    codex_review_post.py gate
        The same admission rules for another workflow (the Claude review): writes allowed=,
        reason=, user=, pr= and head_sha= to GITHUB_OUTPUT and always exits 0.

Admission, on every trigger: the pull request's head must live in this repository (a fork is
never reviewed and its commands are ignored), and the human who caused the run (the pull
request's author, or the commenter) must have write access to the repository. The other
bot's own review activity is admitted on non-fork pull requests only.

Modes:
    review  a pull request opened, reopened or pushed (incremental), drafts included, or
            `@codex review` in a comment: findings become one review, APPROVE when nothing
            Important was found
    reply   a human replied in a thread Codex opened, or mentioned @codex in any thread:
            one reply in that thread
    verify  the other bot submitted review activity: every root finding of that bot without a
            Codex verdict gets one, in its thread

Environment:
    GITHUB_TOKEN        the token to post with: for `post`, the Codex App's installation
                        token the runner pod's sidecar mints; for the rest, the job's
    GITHUB_REPOSITORY   owner/name
    GITHUB_EVENT_NAME, GITHUB_EVENT_PATH   the workflow event (context)
    GITHUB_API_URL      default https://api.github.com
    GITHUB_OUTPUT       when set, `context` appends mode=, skip= and pr= lines
    CODEX_OTHER_BOT     the login whose findings `verify` grades (default claude[bot])
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

MARKER = "<!-- codex-review -->"
VERDICT_MARKER = "<!-- codex-verdict -->"
BOT = "smg-codex-review[bot]"  # the GitHub App whose token the post job posts with
LEGACY_BOT = "github-actions[bot]"  # what Codex posted as before the App; still ours
OWN_BOTS = frozenset({BOT, LEGACY_BOT})
STATUS_MARKER = "<!-- codex-status -->"  # the one status note per pull request
OTHER_BOT = "claude[bot]"
SEVERITY_INLINE = {
    "important": "🔴 **Important**",
    "nit": "🟡 **Nit**",
    "pre_existing": "🟣 **Pre-existing**",
}
SEVERITY_LABEL = {"important": "🔴 Important", "nit": "🟡 Nit", "pre_existing": "🟣 Pre-existing"}
VERDICTS = {"agree": "🤝 Agree", "disagree": "⚠️ Disagree", "unsure": "❓ Unsure"}
WRITE_PERMISSIONS = {"admin", "write"}
COMMAND_RE = re.compile(r"@codex\s+review\b", re.IGNORECASE)
MENTION_RE = re.compile(r"@codex\b", re.IGNORECASE)
PAGE = 100

Api = Callable[..., Any]


class FindingsError(ValueError):
    """Codex's output does not have the shape its schema promised."""


class ApiError(RuntimeError):
    """A GitHub API call failed."""


# --- GitHub -------------------------------------------------------------------------------------


def github_api(token: str, base: str = "https://api.github.com") -> Api:
    """A callable `api(method, path, body=None)` returning the decoded JSON response."""

    def call(method: str, path: str, body: dict | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{base}/{path}",
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "smg-codex-review",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise ApiError(f"{method} {path}: HTTP {error.code}: {detail}") from error
        return json.loads(raw) if raw else {}

    return call


def list_all(api: Api, path: str) -> list[dict]:
    """Every item of a paginated list endpoint."""
    items: list[dict] = []
    page = 1
    while True:
        separator = "&" if "?" in path else "?"
        suffix = f"{separator}per_page={PAGE}" + (f"&page={page}" if page > 1 else "")
        chunk = api("GET", f"{path}{suffix}")
        items.extend(chunk)
        if len(chunk) < PAGE:
            return items
        page += 1


def is_bot(user: dict) -> bool:
    return user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")


def has_write(api: Api, repo: str, login: str) -> bool:
    """Whether `login` may push to `repo`: GitHub's collaborator permission is admin or write
    (maintain reports as write, triage and read as read; a stranger is a 404)."""
    try:
        answer = api("GET", f"repos/{repo}/collaborators/{login}/permission")
    except ApiError:
        return False
    return isinstance(answer, dict) and answer.get("permission") in WRITE_PERMISSIONS


def is_fork(pr: dict, repo: str) -> bool:
    """Whether the pull request's head lives outside `repo` (a deleted head repo counts)."""
    head_repo = (pr.get("head") or {}).get("repo") or {}
    return str(head_repo.get("full_name", "")).lower() != repo.lower()


# --- diff ---------------------------------------------------------------------------------------

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def diff_lines_by_path(diff_text: str) -> dict[str, set[int]]:
    """The right-side line numbers a unified diff covers, per new path: added and context lines,
    the lines a review comment may sit on. Deleted files have no right side and are skipped."""
    lines: dict[str, set[int]] = {}
    path: str | None = None
    new_line = 0
    in_hunk = False
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            path = None
            in_hunk = False
        elif raw.startswith("+++ "):
            target = raw[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
        elif raw.startswith("@@ ") and path is not None:
            match = HUNK_RE.match(raw)
            if match:
                new_line = int(match.group(1))
                in_hunk = True
        elif in_hunk and path is not None:
            if raw.startswith("\\"):
                continue
            if raw.startswith("-"):
                continue
            lines.setdefault(path, set()).add(new_line)
            new_line += 1
    return lines


# --- findings → review --------------------------------------------------------------------------


def load_findings(path: Path) -> dict:
    """Codex's review output, checked against the shape the review schema promises."""
    return validate_findings(load_output(path))


def load_output(path: Path) -> Any:
    text = path.read_text().strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise FindingsError(f"{path}: not JSON: {error}") from error


def validate_findings(document: Any) -> dict:
    if not isinstance(document, dict) or not isinstance(document.get("summary"), str):
        raise FindingsError("output needs a string `summary`")
    findings = document.get("findings")
    if not isinstance(findings, list):
        raise FindingsError("output needs a `findings` list")
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise FindingsError(f"finding {index} is not an object")
        if not isinstance(finding.get("path"), str) or not finding["path"]:
            raise FindingsError(f"finding {index} needs a `path`")
        if not isinstance(finding.get("line"), int) or finding["line"] < 1:
            raise FindingsError(f"finding {index} needs a positive integer `line`")
        if finding.get("severity") not in SEVERITY_INLINE:
            raise FindingsError(f"finding {index} needs a severity in {sorted(SEVERITY_INLINE)}")
        if not isinstance(finding.get("body"), str) or not finding["body"].strip():
            raise FindingsError(f"finding {index} needs a `body`")
    return document


SEVERITY_OF_ICON = {"🔴": "important", "🟡": "nit", "🟣": "pre_existing"}


def prior_findings(comments: list[dict]) -> list[dict]:
    """Codex's own root findings on the pull request, with the severity their marker carries."""
    found = []
    for c in comments:
        body = c.get("body") or ""
        if (
            c.get("in_reply_to_id") is not None
            or c["user"]["login"] not in OWN_BOTS
            or MARKER not in body
        ):
            continue
        text = body.replace(MARKER, "").strip()
        severity = next(
            (sev for icon, sev in SEVERITY_OF_ICON.items() if text.startswith(icon)), None
        )
        found.append(
            {
                "comment_id": c["id"],
                "path": c.get("path"),
                "line": c.get("line") or c.get("original_line"),
                "severity": severity,
                "body": text,
            }
        )
    return found


def strip_severity_label(body: str) -> str:
    """The model sometimes opens the body with the label the renderer adds; keep one."""
    text = body.strip()
    for label in (*SEVERITY_INLINE.values(), *SEVERITY_LABEL.values()):
        if text.startswith(label):
            text = text[len(label) :].lstrip(" :").strip()
    return text


def render_finding(finding: dict) -> str:
    return (
        f"{MARKER}\n{SEVERITY_INLINE[finding['severity']]}: {strip_severity_label(finding['body'])}"
    )


def render_summary(
    findings: list[dict], summary: str, outside: list[dict], open_prior: int = 0
) -> str:
    counts = {sev: sum(1 for f in findings if f["severity"] == sev) for sev in SEVERITY_LABEL}
    tally = " · ".join(f"{counts[sev]} {label}" for sev, label in SEVERITY_LABEL.items())
    if open_prior:
        plural = "s" if open_prior > 1 else ""
        tally += f" · {open_prior} earlier 🔴 Important finding{plural} still open"
    lines = [MARKER, "### Codex review", "", summary.strip(), "", tally]
    if outside:
        lines += ["", "Findings on lines outside this diff:", ""]
        lines += [
            f"- `{f['path']}:{f['line']}` {SEVERITY_LABEL[f['severity']]}: "
            f"{strip_severity_label(f['body'])}"
            for f in outside
        ]
    return "\n".join(lines) + "\n"


def build_review(
    findings: list[dict],
    summary: str,
    diff_lines: dict[str, set[int]],
    head_sha: str,
    open_prior: int = 0,
) -> dict:
    """One pull-request review: inline comments on the findings the diff can carry, the rest
    named in the body. The event is the caller's."""
    comments = []
    outside = []
    for finding in findings:
        if finding["line"] in diff_lines.get(finding["path"], set()):
            comments.append(
                {
                    "path": finding["path"],
                    "line": finding["line"],
                    "side": "RIGHT",
                    "body": render_finding(finding),
                }
            )
        else:
            outside.append(finding)
    return {
        "commit_id": head_sha,
        "body": render_summary(findings, summary, outside, open_prior),
        "comments": comments,
    }


def has_important(findings: list[dict]) -> bool:
    return any(f["severity"] == "important" for f in findings)


def review_event(findings: list[dict], live_approval: bool, open_prior: int = 0) -> str | None:
    """REQUEST_CHANGES when something Important was found or an earlier Important finding is
    still open; APPROVE when nothing is and no Codex approval is live (the approval is what lifts
    an earlier request); COMMENT when there is something to say otherwise; None when a clean
    rerun has nothing to add."""
    if has_important(findings) or open_prior:
        return "REQUEST_CHANGES"
    if not live_approval:
        return "APPROVE"
    return "COMMENT" if findings else None


def live_approvals(reviews: list[dict]) -> list[int]:
    """The ids of Codex's own approvals that still stand."""
    return [
        review["id"]
        for review in reviews
        if review.get("state") == "APPROVED"
        and review.get("user", {}).get("login") in OWN_BOTS
        and MARKER in (review.get("body") or "")
    ]


def post_review(
    api: Api,
    repo: str,
    pr: int,
    head_sha: str,
    document: dict,
    diff_text: str,
    prior_important: set[int] = frozenset(),
) -> dict | None:
    """Post the run's findings as one review. `prior_important` holds the comment ids of Codex's
    earlier Important findings; the ones the run reports as still open block approval."""
    findings = document["findings"]
    still_open = {
        int(i) for i in document.get("prior_findings_still_open", []) if isinstance(i, int)
    }
    open_prior = len(still_open & set(prior_important))
    live = live_approvals(list_all(api, f"repos/{repo}/pulls/{pr}/reviews"))
    event = review_event(findings, bool(live), open_prior)
    if event is None:
        return None
    review = build_review(
        findings, document["summary"], diff_lines_by_path(diff_text), head_sha, open_prior
    )
    review["event"] = event
    created = api("POST", f"repos/{repo}/pulls/{pr}/reviews", review)
    if isinstance(created, dict):
        created = {**created, "event": event}
    if has_important(findings) or open_prior:
        for review_id in live:
            api(
                "PUT",
                f"repos/{repo}/pulls/{pr}/reviews/{review_id}/dismissals",
                {
                    "message": "Codex found an Important issue in a later push; this approval "
                    "no longer reflects the current head.",
                    "event": "DISMISS",
                },
            )
    return created


# --- the status note ------------------------------------------------------------------------------


VERDICT_WORDS = {
    "REQUEST_CHANGES": "changes requested",
    "APPROVE": "approved",
    "COMMENT": "commented",
}


def render_status(state: str, info: dict) -> str:
    """The one status note per pull request: what Codex is doing, then what it found."""
    sha = info["head_sha"][:7]
    run = info.get("run_url") or ""
    lines = [STATUS_MARKER]
    if state == "reviewing":
        scope = "the new push" if info.get("incremental") else "the whole pull request"
        lines += [
            "> [!NOTE]",
            f"> ⏳ **Codex is reviewing** commit `{sha}` ({scope}).",
            "> Findings arrive as inline comments and the verdict as a review; this note "
            "updates when it is done.",
            ">",
            f"> [Run]({run})",
        ]
        files = list(info.get("files") or [])
        if files:
            plural = "s" if len(files) != 1 else ""
            lines += [
                "",
                "<details>",
                f"<summary>{len(files)} file{plural} in the review</summary>",
                "",
            ]
            lines += [f"- `{name}`" for name in files[:200]]
            if len(files) > 200:
                lines.append(f"- and {len(files) - 200} more")
            lines += ["", "</details>"]
    elif state == "done":
        counts = info.get("counts") or {}
        tally = " · ".join(f"{counts.get(sev, 0)} {label}" for sev, label in SEVERITY_LABEL.items())
        verdict = VERDICT_WORDS.get(info.get("verdict") or "", "nothing new to say")
        lines += [
            "> [!NOTE]",
            f"> ✅ **Codex reviewed** commit `{sha}`: {tally}. Verdict: {verdict}. [Run]({run})",
        ]
    elif state == "failed":
        lines += [
            "> [!CAUTION]",
            f"> ❌ **Codex could not review** commit `{sha}`. See the [run]({run}); a new push "
            "or `@codex review` tries again.",
        ]
    else:
        raise ValueError(f"unknown status state {state!r}")
    return "\n".join(lines) + "\n"


def status_comment(comments: list[dict]) -> dict | None:
    """Codex's status note among a pull request's comments, if it has one."""
    return next(
        (
            c
            for c in comments
            if c.get("user", {}).get("login") in OWN_BOTS and STATUS_MARKER in (c.get("body") or "")
        ),
        None,
    )


def upsert_status(
    api: Api,
    repo: str,
    pr: int,
    body: str,
    *,
    head_sha: str | None = None,
    only_if_absent_for_head: bool = False,
) -> dict:
    """Post the status note or update it in place. With `only_if_absent_for_head`, a note that
    already names the head is left alone, so a late announcement never undoes the outcome."""
    existing = status_comment(list_all(api, f"repos/{repo}/issues/{pr}/comments"))
    if existing and only_if_absent_for_head and head_sha:
        if f"`{head_sha[:7]}`" in (existing.get("body") or ""):
            return existing
    if existing:
        return api("PATCH", f"repos/{repo}/issues/comments/{existing['id']}", {"body": body})
    return api("POST", f"repos/{repo}/issues/{pr}/comments", {"body": body})


def run_announce(api: Api, repo: str, context: dict, state: str, info: dict) -> dict | None:
    """The status note for a review run: `reviewing` when it starts, `failed` when the review
    job dies. Other modes and skipped runs get none; the post job writes the `done` state."""
    if context.get("skip") or context.get("mode") != "review":
        return None
    pr, head = context["pr"], context["head_sha"]
    details = {"head_sha": head, "incremental": bool(context.get("incremental")), **info}
    if state == "reviewing":
        files = list_all(api, f"repos/{repo}/pulls/{pr}/files")
        details["files"] = [f["filename"] for f in files]
        body = render_status("reviewing", details)
        return upsert_status(api, repo, pr, body, head_sha=head, only_if_absent_for_head=True)
    return upsert_status(api, repo, pr, render_status(state, details))


def run_url_from_env() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        return ""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/{run_id}"


# --- threads ------------------------------------------------------------------------------------


def thread_of(comments: list[dict], root_id: int) -> list[dict]:
    root = next((c for c in comments if c["id"] == root_id), None)
    if root is None:
        return []
    replies = sorted(
        (c for c in comments if c.get("in_reply_to_id") == root_id), key=lambda c: c["id"]
    )
    return [root, *replies]


def thread_awaits_reply(comments: list[dict], root_id: int, mention: str | None = None) -> bool:
    """A thread Codex opened, or one where @codex was mentioned, whose last word is a human's."""
    thread = thread_of(comments, root_id)
    if not thread:
        return False
    root = thread[0]
    ours = root["user"]["login"] in OWN_BOTS and MARKER in (root.get("body") or "")
    mentioned = bool(mention and MENTION_RE.search(mention))
    if not (ours or mentioned):
        return False
    return not is_bot(thread[-1]["user"])


def post_reply(api: Api, repo: str, pr: int, comment_id: int, body: str) -> dict:
    return api("POST", f"repos/{repo}/pulls/{pr}/comments/{comment_id}/replies", {"body": body})


def render_reply(output: dict) -> str:
    reply = str(output.get("reply", "")).strip()
    if not reply:
        raise FindingsError("reply output needs a non-empty `reply`")
    prefix = "✅ Withdrawn. " if output.get("withdraw") else ""
    return f"{MARKER}\n{prefix}{reply}"


# --- verdicts -----------------------------------------------------------------------------------


def pending_verdicts(comments: list[dict], other: str) -> list[dict]:
    """The other bot's root findings that carry no Codex verdict yet."""
    graded = {c.get("in_reply_to_id") for c in comments if VERDICT_MARKER in (c.get("body") or "")}
    return [
        c
        for c in comments
        if c.get("in_reply_to_id") is None and c["user"]["login"] == other and c["id"] not in graded
    ]


def render_verdict(verdict: dict) -> str:
    kind = verdict.get("verdict")
    if kind not in VERDICTS:
        raise FindingsError(f"verdict needs one of {sorted(VERDICTS)}")
    return f"{VERDICT_MARKER}\n{VERDICTS[kind]}: {str(verdict.get('reason', '')).strip()}"


def post_verdicts(api: Api, repo: str, pr: int, verdicts: list[dict]) -> list[dict]:
    return [
        api(
            "POST",
            f"repos/{repo}/pulls/{pr}/comments/{verdict['comment_id']}/replies",
            {"body": render_verdict(verdict)},
        )
        for verdict in verdicts
    ]


# --- context ------------------------------------------------------------------------------------


def _pr_fields(pr: dict) -> dict:
    return {
        "pr": pr["number"],
        "head_sha": pr["head"]["sha"],
        "base_sha": pr["base"]["sha"],
        "base_ref": pr["base"]["ref"],
        "title": pr.get("title", ""),
        "author": pr.get("user", {}).get("login", ""),
        "html_url": pr.get("html_url", ""),
    }


def _skip(reason: str) -> dict:
    return {"mode": None, "skip": True, "skip_reason": reason}


def resolve_context(
    event_name: str, event: dict, api: Api, repo: str, other_bot: str = OTHER_BOT
) -> dict:
    """What this run does, from the event that started it."""
    if event_name in ("pull_request", "pull_request_target"):
        pr = event["pull_request"]
        if is_fork(pr, repo):
            return _skip("pull request from a fork")
        if not has_write(api, repo, pr["user"]["login"]):
            return _skip(f"author {pr['user']['login']} has no write access")
        incremental = event.get("action") == "synchronize"
        context = {
            "mode": "review",
            "skip": False,
            "incremental": incremental,
            "before_sha": event.get("before") if incremental else None,
            **_pr_fields(pr),
        }
    elif event_name == "issue_comment":
        issue, comment = event["issue"], event["comment"]
        if "pull_request" not in issue:
            return _skip("not a pull request")
        if is_bot(comment["user"]):
            return _skip("comment by a bot")
        if not COMMAND_RE.search(comment.get("body") or ""):
            return _skip("no @codex review command")
        if not has_write(api, repo, comment["user"]["login"]):
            return _skip(f"commenter {comment['user']['login']} has no write access")
        pr = api("GET", f"repos/{repo}/pulls/{issue['number']}")
        if is_fork(pr, repo):
            return _skip("pull request from a fork")
        context = {
            "mode": "review",
            "skip": False,
            "incremental": False,
            "before_sha": None,
            "requested_by": comment["user"]["login"],
            **_pr_fields(pr),
        }
    elif event_name == "pull_request_review_comment":
        pr, comment = event["pull_request"], event["comment"]
        if is_fork(pr, repo):
            return _skip("pull request from a fork")
        if is_bot(comment["user"]):
            return _skip("comment by a bot")
        if not has_write(api, repo, comment["user"]["login"]):
            return _skip(f"commenter {comment['user']['login']} has no write access")
        root_id = comment.get("in_reply_to_id") or comment["id"]
        comments = list_all(api, f"repos/{repo}/pulls/{pr['number']}/comments")
        if not thread_awaits_reply(comments, root_id, mention=comment.get("body")):
            return _skip("thread is not Codex's, or Codex spoke last")
        thread = thread_of(comments, root_id)
        root = thread[0]
        return {
            "mode": "reply",
            "skip": False,
            "root_comment_id": root_id,
            "path": root.get("path"),
            "line": root.get("line") or root.get("original_line"),
            "diff_hunk": root.get("diff_hunk", ""),
            "thread": [{"author": c["user"]["login"], "body": c.get("body", "")} for c in thread],
            **_pr_fields(pr),
        }
    elif event_name == "pull_request_review":
        review, pr = event["review"], event["pull_request"]
        if is_fork(pr, repo):
            return _skip("pull request from a fork")
        if review["user"]["login"] != other_bot:
            return _skip(f"review by {review['user']['login']}, not {other_bot}")
        comments = list_all(api, f"repos/{repo}/pulls/{pr['number']}/comments")
        targets = pending_verdicts(comments, other_bot)
        if not targets:
            return _skip(f"no {other_bot} finding awaits a verdict")
        return {
            "mode": "verify",
            "skip": False,
            "other_bot": other_bot,
            "targets": [
                {
                    "comment_id": c["id"],
                    "path": c.get("path"),
                    "line": c.get("line") or c.get("original_line"),
                    "diff_hunk": c.get("diff_hunk", ""),
                    "body": c.get("body", ""),
                }
                for c in targets
            ],
            **_pr_fields(pr),
        }
    else:
        return _skip(f"unsupported event {event_name}")

    comments = list_all(api, f"repos/{repo}/pulls/{context['pr']}/comments")
    context["prior_findings"] = prior_findings(comments)
    return context


# --- admission for another workflow -------------------------------------------------------------


def gate(event_name: str, event: dict, api: Api, repo: str) -> dict:
    """Whether the human behind this event may have a review run on this pull request."""
    verdict = {"allowed": False, "reason": "", "user": "", "pr": None, "head_sha": ""}
    if event_name in ("pull_request", "pull_request_target"):
        pr = event["pull_request"]
        verdict["user"] = pr["user"]["login"]
    elif event_name == "issue_comment":
        issue, comment = event["issue"], event["comment"]
        verdict["user"] = comment["user"]["login"]
        if "pull_request" not in issue:
            verdict["reason"] = "not a pull request"
            return verdict
        pr = api("GET", f"repos/{repo}/pulls/{issue['number']}")
    elif event_name == "pull_request_review_comment":
        pr, comment = event["pull_request"], event["comment"]
        verdict["user"] = comment["user"]["login"]
    elif event_name == "pull_request_review":
        pr, review = event["pull_request"], event["review"]
        verdict["user"] = review["user"]["login"]
    else:
        verdict["reason"] = f"unsupported event {event_name}"
        return verdict
    verdict["pr"] = pr["number"]
    verdict["head_sha"] = pr["head"]["sha"]
    if is_fork(pr, repo):
        verdict["reason"] = "pull request from a fork"
        return verdict
    if event_name == "pull_request_review":
        verdict["allowed"] = True  # the other bot's activity; the bot's own gate admitted it
        return verdict
    user = verdict["user"]
    if event_name != "pull_request_target" and event_name != "pull_request":
        comment_user = event["comment"]["user"] if "comment" in event else {}
        if is_bot(comment_user):
            verdict["reason"] = "comment by a bot"
            return verdict
    if not has_write(api, repo, user):
        verdict["reason"] = f"{user} has no write access"
        return verdict
    verdict["allowed"] = True
    return verdict


# --- posting ------------------------------------------------------------------------------------


def run_post(api: Api, repo: str, context: dict, output: Any, diff_text: str) -> Any:
    mode = context["mode"]
    pr = context["pr"]
    if mode == "review":
        prior_important = {
            f["comment_id"]
            for f in context.get("prior_findings", [])
            if f.get("severity") == "important"
        }
        document = validate_findings(output)
        created = post_review(
            api, repo, pr, context["head_sha"], document, diff_text, prior_important
        )
        counts = {
            sev: sum(1 for f in document["findings"] if f["severity"] == sev)
            for sev in SEVERITY_LABEL
        }
        outcome = {
            "head_sha": context["head_sha"],
            "run_url": context.get("run_url", ""),
            "counts": counts,
            "verdict": created["event"] if created else None,
        }
        upsert_status(api, repo, pr, render_status("done", outcome))
        return created
    if mode == "reply":
        return post_reply(api, repo, pr, context["root_comment_id"], render_reply(output))
    if mode == "verify":
        allowed = {t["comment_id"] for t in context["targets"]}
        verdicts = output.get("verdicts") if isinstance(output, dict) else None
        if not isinstance(verdicts, list):
            raise FindingsError("verify output needs a `verdicts` list")
        return post_verdicts(api, repo, pr, [v for v in verdicts if v.get("comment_id") in allowed])
    raise FindingsError(f"unknown mode {mode!r}")


def write_outputs(context: dict) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"mode={context.get('mode') or ''}\n")
        handle.write(f"skip={'true' if context.get('skip') else 'false'}\n")
        handle.write(f"pr={context.get('pr', '')}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    context_cmd = commands.add_parser("context", help="resolve the run's mode and inputs")
    context_cmd.add_argument("--out-dir", required=True, type=Path)
    post_cmd = commands.add_parser("post", help="post Codex's output to the pull request")
    post_cmd.add_argument("--dir", required=True, type=Path)
    commands.add_parser("gate", help="admission verdict for another workflow, to GITHUB_OUTPUT")
    announce_cmd = commands.add_parser("announce", help="post or update the status note")
    announce_cmd.add_argument("--state", choices=("reviewing", "failed"), required=True)
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    repo = os.environ["GITHUB_REPOSITORY"]
    api = github_api(token, os.environ.get("GITHUB_API_URL", "https://api.github.com"))

    if args.command == "gate":
        event_name = os.environ["GITHUB_EVENT_NAME"]
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        verdict = gate(event_name, event, api, repo)
        output_path = os.environ.get("GITHUB_OUTPUT")
        if output_path:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(f"allowed={'true' if verdict['allowed'] else 'false'}\n")
                handle.write(f"reason={verdict['reason']}\n")
                handle.write(f"user={verdict['user']}\n")
                handle.write(f"pr={verdict['pr'] or ''}\n")
                handle.write(f"head_sha={verdict['head_sha']}\n")
        print("admitted" if verdict["allowed"] else f"refused: {verdict['reason']}")
        return 0

    if args.command == "announce":
        event_name = os.environ["GITHUB_EVENT_NAME"]
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        context = resolve_context(
            event_name, event, api, repo, os.environ.get("CODEX_OTHER_BOT", OTHER_BOT)
        )
        info = {"run_url": run_url_from_env()}
        note = run_announce(api, repo, context, args.state, info)
        why = context.get("skip_reason") or context.get("mode")
        print("status note written" if note else f"no status note: {why}")
        return 0

    if args.command == "context":
        event_name = os.environ["GITHUB_EVENT_NAME"]
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        context = resolve_context(
            event_name, event, api, repo, os.environ.get("CODEX_OTHER_BOT", OTHER_BOT)
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "context.json").write_text(json.dumps(context, indent=2) + "\n")
        write_outputs(context)
        if context["skip"]:
            print(f"skip: {context['skip_reason']}")
        else:
            print(f"mode={context['mode']} pr={context['pr']} head={context['head_sha']}")
        return 0

    context = json.loads((args.dir / "context.json").read_text())
    if context.get("skip"):
        print(f"nothing to post: {context.get('skip_reason')}")
        return 0
    output_path = args.dir / "output.json"
    if not output_path.is_file():
        print(f"{output_path} is missing: Codex produced no final message", file=sys.stderr)
        return 1
    diff_path = args.dir / "pr.diff"
    diff_text = diff_path.read_text() if diff_path.is_file() else ""
    context["run_url"] = run_url_from_env()
    result = run_post(api, repo, context, load_output(output_path), diff_text)
    print(json.dumps({"mode": context["mode"], "posted": result is not None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
