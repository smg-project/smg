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

Modes:
    review  a pull request opened, reopened or pushed (incremental), or `@codex review` in a
            comment: findings become one review, APPROVE when nothing Important was found
    reply   a human replied in a thread Codex opened, or mentioned @codex in any thread:
            one reply in that thread
    verify  the other bot submitted review activity: every root finding of that bot without a
            Codex verdict gets one, in its thread

Environment:
    GITHUB_TOKEN        the job's token
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
BOT = "github-actions[bot]"
OTHER_BOT = "claude[bot]"
SEVERITY_INLINE = {
    "important": "🔴 **Important**",
    "nit": "🟡 **Nit**",
    "pre_existing": "🟣 **Pre-existing**",
}
SEVERITY_LABEL = {"important": "🔴 Important", "nit": "🟡 Nit", "pre_existing": "🟣 Pre-existing"}
VERDICTS = {"agree": "🤝 Agree", "disagree": "⚠️ Disagree", "unsure": "❓ Unsure"}
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


def render_finding(finding: dict) -> str:
    return f"{MARKER}\n{SEVERITY_INLINE[finding['severity']]}: {finding['body'].strip()}"


def render_summary(findings: list[dict], summary: str, outside: list[dict]) -> str:
    counts = {sev: sum(1 for f in findings if f["severity"] == sev) for sev in SEVERITY_LABEL}
    tally = " · ".join(f"{counts[sev]} {label}" for sev, label in SEVERITY_LABEL.items())
    lines = [MARKER, "### Codex review", "", summary.strip(), "", tally]
    if outside:
        lines += ["", "Findings on lines outside this diff:", ""]
        lines += [
            f"- `{f['path']}:{f['line']}` {SEVERITY_LABEL[f['severity']]}: {f['body'].strip()}"
            for f in outside
        ]
    return "\n".join(lines) + "\n"


def build_review(
    findings: list[dict], summary: str, diff_lines: dict[str, set[int]], head_sha: str
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
        "body": render_summary(findings, summary, outside),
        "comments": comments,
    }


def has_important(findings: list[dict]) -> bool:
    return any(f["severity"] == "important" for f in findings)


def review_event(findings: list[dict], live_approval: bool) -> str | None:
    """APPROVE when nothing Important was found and no Codex approval is live; COMMENT when
    there is something to say otherwise; None when a clean rerun has nothing to add."""
    if has_important(findings):
        return "COMMENT"
    if not live_approval:
        return "APPROVE"
    return "COMMENT" if findings else None


def live_approvals(reviews: list[dict]) -> list[int]:
    """The ids of Codex's own approvals that still stand."""
    return [
        review["id"]
        for review in reviews
        if review.get("state") == "APPROVED"
        and review.get("user", {}).get("login") == BOT
        and MARKER in (review.get("body") or "")
    ]


def post_review(
    api: Api, repo: str, pr: int, head_sha: str, document: dict, diff_text: str
) -> dict | None:
    findings = document["findings"]
    live = live_approvals(list_all(api, f"repos/{repo}/pulls/{pr}/reviews"))
    event = review_event(findings, bool(live))
    if event is None:
        return None
    review = build_review(findings, document["summary"], diff_lines_by_path(diff_text), head_sha)
    review["event"] = event
    created = api("POST", f"repos/{repo}/pulls/{pr}/reviews", review)
    if has_important(findings):
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
    ours = root["user"]["login"] == BOT and MARKER in (root.get("body") or "")
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
    if event_name == "pull_request":
        pr = event["pull_request"]
        if pr.get("draft"):
            return _skip("draft pull request")
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
        pr = api("GET", f"repos/{repo}/pulls/{issue['number']}")
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
        if is_bot(comment["user"]):
            return _skip("comment by a bot")
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
    context["prior_findings"] = [
        {"path": c.get("path"), "line": c.get("line") or c.get("original_line"), "body": c["body"]}
        for c in comments
        if c.get("in_reply_to_id") is None
        and c["user"]["login"] == BOT
        and MARKER in (c.get("body") or "")
    ]
    return context


# --- posting ------------------------------------------------------------------------------------


def run_post(api: Api, repo: str, context: dict, output: Any, diff_text: str) -> Any:
    mode = context["mode"]
    pr = context["pr"]
    if mode == "review":
        return post_review(api, repo, pr, context["head_sha"], validate_findings(output), diff_text)
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
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    repo = os.environ["GITHUB_REPOSITORY"]
    api = github_api(token, os.environ.get("GITHUB_API_URL", "https://api.github.com"))

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
    result = run_post(api, repo, context, load_output(output_path), diff_text)
    print(json.dumps({"mode": context["mode"], "posted": result is not None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
