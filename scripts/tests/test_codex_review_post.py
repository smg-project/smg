"""The Codex review poster: findings in, one pull-request review out."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex_review_post as post  # noqa: E402

DIFF = """\
diff --git a/src/a.rs b/src/a.rs
index 1111111..2222222 100644
--- a/src/a.rs
+++ b/src/a.rs
@@ -10,4 +10,5 @@ fn f() {
 let a = 1;
-let b = 2;
+let b = 3;
+let c = 4;
 use_it(a, b);
 done();
diff --git a/src/gone.rs b/src/gone.rs
deleted file mode 100644
index 3333333..0000000
--- a/src/gone.rs
+++ /dev/null
@@ -1,2 +0,0 @@
-fn old() {}
-fn older() {}
"""


def finding(path="src/a.rs", line=11, severity="important", body="Off by one."):
    return {"path": path, "line": line, "severity": severity, "body": body}


class FakeApi:
    """Records every call; answers GETs from a canned table."""

    def __init__(self, gets=None):
        self.gets = gets or {}
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == "GET":
            return self.gets.get(path, [])
        return {"id": 99}


def test_diff_lines_cover_added_and_context_lines_of_the_new_file():
    lines = post.diff_lines_by_path(DIFF)
    assert lines["src/a.rs"] == {10, 11, 12, 13, 14}


def test_diff_lines_skip_deleted_files():
    assert "src/gone.rs" not in post.diff_lines_by_path(DIFF)


def test_a_finding_on_a_diff_line_becomes_an_inline_comment():
    review = post.build_review([finding()], "fine", post.diff_lines_by_path(DIFF), "abc")
    assert review["comments"] == [
        {
            "path": "src/a.rs",
            "line": 11,
            "side": "RIGHT",
            "body": f"{post.MARKER}\n🔴 **Important**: Off by one.",
        }
    ]
    assert review["commit_id"] == "abc"


def test_a_finding_outside_the_diff_moves_into_the_body():
    review = post.build_review([finding(line=300)], "fine", post.diff_lines_by_path(DIFF), "abc")
    assert review["comments"] == []
    assert "`src/a.rs:300`" in review["body"]
    assert "Off by one." in review["body"]


def test_body_counts_findings_per_severity_and_carries_the_marker():
    findings = [
        finding(),
        finding(line=12, severity="nit", body="Name."),
        finding(line=13, severity="pre_existing", body="Old."),
    ]
    body = post.render_summary(findings, "Looks solid.", [])
    assert body.startswith(post.MARKER)
    assert "Looks solid." in body
    assert "1 🔴 Important" in body and "1 🟡 Nit" in body and "1 🟣 Pre-existing" in body


def test_no_important_finding_and_no_live_approval_approves():
    assert post.review_event([finding(severity="nit")], live_approval=False) == "APPROVE"


def test_an_important_finding_comments():
    assert post.review_event([finding()], live_approval=True) == "COMMENT"


def test_a_clean_rerun_on_an_approved_pr_posts_nothing():
    assert post.review_event([], live_approval=True) is None


def test_a_rerun_with_only_nits_on_an_approved_pr_comments():
    assert post.review_event([finding(severity="nit")], live_approval=True) == "COMMENT"


def test_live_approvals_are_the_undismissed_marked_approvals_of_the_bot():
    reviews = [
        {"id": 1, "state": "APPROVED", "user": {"login": post.BOT}, "body": f"{post.MARKER} ok"},
        {"id": 2, "state": "DISMISSED", "user": {"login": post.BOT}, "body": f"{post.MARKER} ok"},
        {"id": 3, "state": "APPROVED", "user": {"login": "claude[bot]"}, "body": "lgtm"},
        {"id": 4, "state": "APPROVED", "user": {"login": post.BOT}, "body": "unrelated approval"},
    ]
    assert post.live_approvals(reviews) == [1]


def test_posting_an_important_finding_dismisses_the_earlier_approval():
    api = FakeApi(
        gets={
            "repos/o/r/pulls/7/reviews?per_page=100": [
                {"id": 5, "state": "APPROVED", "user": {"login": post.BOT}, "body": post.MARKER}
            ]
        }
    )
    post.post_review(api, "o/r", 7, "abc", {"summary": "s", "findings": [finding()]}, DIFF)
    methods = [(m, p) for m, p, _ in api.calls if m != "GET"]
    assert ("PUT", "repos/o/r/pulls/7/reviews/5/dismissals") in methods
    created = next(b for m, p, b in api.calls if m == "POST" and p == "repos/o/r/pulls/7/reviews")
    assert created["event"] == "COMMENT"


def test_posting_a_clean_review_approves_once():
    api = FakeApi()
    post.post_review(api, "o/r", 7, "abc", {"summary": "s", "findings": []}, DIFF)
    created = [b for m, p, b in api.calls if m == "POST" and p == "repos/o/r/pulls/7/reviews"]
    assert [c["event"] for c in created] == ["APPROVE"]


def test_a_thread_awaits_a_reply_when_a_human_spoke_last_under_our_root():
    comments = [
        {"id": 1, "in_reply_to_id": None, "user": {"login": post.BOT}, "body": f"{post.MARKER} x"},
        {"id": 2, "in_reply_to_id": 1, "user": {"login": "simo"}, "body": "why?"},
    ]
    assert post.thread_awaits_reply(comments, root_id=1) is True


def test_a_thread_we_answered_last_does_not_await_a_reply():
    comments = [
        {"id": 1, "in_reply_to_id": None, "user": {"login": post.BOT}, "body": f"{post.MARKER} x"},
        {"id": 2, "in_reply_to_id": 1, "user": {"login": "simo"}, "body": "why?"},
        {
            "id": 3,
            "in_reply_to_id": 1,
            "user": {"login": post.BOT},
            "body": f"{post.MARKER} because",
        },
    ]
    assert post.thread_awaits_reply(comments, root_id=1) is False


def test_a_thread_rooted_by_someone_else_is_not_ours_unless_mentioned():
    comments = [
        {"id": 1, "in_reply_to_id": None, "user": {"login": "claude[bot]"}, "body": "x"},
        {"id": 2, "in_reply_to_id": 1, "user": {"login": "simo"}, "body": "hm"},
    ]
    assert post.thread_awaits_reply(comments, root_id=1) is False
    assert post.thread_awaits_reply(comments, root_id=1, mention="@codex hm") is True


def test_pending_verdicts_are_the_other_bots_root_comments_without_our_verdict():
    comments = [
        {
            "id": 1,
            "in_reply_to_id": None,
            "user": {"login": "claude[bot]"},
            "body": "🔴 a",
            "path": "p",
            "line": 1,
        },
        {
            "id": 2,
            "in_reply_to_id": None,
            "user": {"login": "claude[bot]"},
            "body": "🟡 b",
            "path": "p",
            "line": 2,
        },
        {
            "id": 3,
            "in_reply_to_id": 2,
            "user": {"login": post.BOT},
            "body": f"{post.VERDICT_MARKER} 🤝 Agree",
        },
        {"id": 4, "in_reply_to_id": None, "user": {"login": "simo"}, "body": "human root"},
        {"id": 5, "in_reply_to_id": 1, "user": {"login": "claude[bot]"}, "body": "reply, not root"},
    ]
    assert [c["id"] for c in post.pending_verdicts(comments, other="claude[bot]")] == [1]


def test_verdict_bodies_carry_the_marker_and_the_icon():
    body = post.render_verdict({"verdict": "disagree", "reason": "The bound is exclusive."})
    assert body.startswith(post.VERDICT_MARKER)
    assert "⚠️ Disagree" in body and "The bound is exclusive." in body


def test_post_verdicts_replies_under_each_target():
    api = FakeApi()
    post.post_verdicts(api, "o/r", 7, [{"comment_id": 1, "verdict": "agree", "reason": "Yes."}])
    body = post.render_verdict({"verdict": "agree", "reason": "Yes."})
    assert api.calls == [("POST", "repos/o/r/pulls/7/comments/1/replies", {"body": body})]


def test_findings_file_is_validated_before_posting(tmp_path):
    bad = tmp_path / "findings.json"
    bad.write_text(json.dumps({"summary": "s", "findings": [{"path": "p"}]}))
    with pytest.raises(post.FindingsError):
        post.load_findings(bad)


# --- who may trigger a review -------------------------------------------------------------------


def pr_payload(author="simo", head_repo="o/r"):
    return {
        "number": 7,
        "title": "t",
        "draft": False,
        "user": {"login": author},
        "head": {"sha": "h" * 40, "repo": {"full_name": head_repo}},
        "base": {"sha": "b" * 40, "ref": "main", "repo": {"full_name": "o/r"}},
    }


class GateApi(FakeApi):
    """Permission lookups: simo writes, rando reads; every list is empty."""

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path.startswith("repos/o/r/collaborators/"):
            login = path.split("/")[4]
            return {"permission": "write" if login in ("simo", "alex") else "read"}
        if method == "GET":
            return self.gets.get(path, [])
        return {"id": 99}


def test_write_access_is_admin_or_write_only():
    api = GateApi()
    assert post.has_write(api, "o/r", "simo") is True
    assert post.has_write(api, "o/r", "rando") is False


def test_a_fork_pull_request_is_never_reviewed():
    event = {"action": "opened", "pull_request": pr_payload(head_repo="rando/r")}
    context = post.resolve_context("pull_request_target", event, GateApi(), "o/r")
    assert context["skip"] is True and "fork" in context["skip_reason"]


def test_a_pull_request_by_a_user_without_write_access_is_not_reviewed():
    event = {"action": "opened", "pull_request": pr_payload(author="rando")}
    context = post.resolve_context("pull_request_target", event, GateApi(), "o/r")
    assert context["skip"] is True and "write access" in context["skip_reason"]


def test_a_pull_request_by_a_writer_from_this_repository_is_reviewed():
    event = {"action": "opened", "pull_request": pr_payload()}
    context = post.resolve_context("pull_request_target", event, GateApi(), "o/r")
    assert context["skip"] is False and context["mode"] == "review"


def test_a_command_from_a_user_without_write_access_is_ignored():
    event = {
        "issue": {"number": 7, "pull_request": {"url": "u"}},
        "comment": {"body": "@codex review", "user": {"login": "rando", "type": "User"}},
    }
    api = GateApi(gets={"repos/o/r/pulls/7": pr_payload()})
    context = post.resolve_context("issue_comment", event, api, "o/r")
    assert context["skip"] is True and "write access" in context["skip_reason"]


def test_a_command_on_a_fork_pull_request_is_ignored_even_from_a_writer():
    event = {
        "issue": {"number": 7, "pull_request": {"url": "u"}},
        "comment": {"body": "@codex review", "user": {"login": "simo", "type": "User"}},
    }
    api = GateApi(gets={"repos/o/r/pulls/7": pr_payload(head_repo="rando/r")})
    context = post.resolve_context("issue_comment", event, api, "o/r")
    assert context["skip"] is True and "fork" in context["skip_reason"]


def test_a_thread_reply_from_a_user_without_write_access_is_ignored():
    root = {
        "id": 1,
        "in_reply_to_id": None,
        "user": {"login": post.BOT},
        "body": f"{post.MARKER} x",
    }
    reply = {
        "id": 2,
        "in_reply_to_id": 1,
        "user": {"login": "rando", "type": "User"},
        "body": "why?",
    }
    event = {"pull_request": pr_payload(), "comment": reply}
    api = GateApi(gets={"repos/o/r/pulls/7/comments?per_page=100": [root, reply]})
    context = post.resolve_context("pull_request_review_comment", event, api, "o/r")
    assert context["skip"] is True and "write access" in context["skip_reason"]


def test_the_other_bots_review_on_a_fork_pull_request_is_not_verified():
    event = {
        "review": {"user": {"login": "claude[bot]"}},
        "pull_request": pr_payload(head_repo="x/r"),
    }
    context = post.resolve_context("pull_request_review", event, GateApi(), "o/r")
    assert context["skip"] is True and "fork" in context["skip_reason"]


def test_the_gate_admits_a_writer_on_a_pull_request_of_this_repository():
    event = {"action": "opened", "pull_request": pr_payload()}
    verdict = post.gate("pull_request_target", event, GateApi(), "o/r")
    assert verdict == {"allowed": True, "reason": "", "user": "simo", "pr": 7, "head_sha": "h" * 40}


def test_the_gate_refuses_a_commenter_without_write_access():
    event = {
        "issue": {"number": 7, "pull_request": {"url": "u"}},
        "comment": {"body": "@claude review", "user": {"login": "rando", "type": "User"}},
    }
    api = GateApi(gets={"repos/o/r/pulls/7": pr_payload()})
    verdict = post.gate("issue_comment", event, api, "o/r")
    assert verdict["allowed"] is False and "write access" in verdict["reason"]
    assert verdict["user"] == "rando" and verdict["pr"] == 7


def test_the_gate_refuses_a_fork_whoever_asks():
    event = {
        "pull_request": pr_payload(head_repo="rando/r"),
        "comment": {"body": "@claude", "user": {"login": "simo", "type": "User"}, "id": 3},
    }
    verdict = post.gate("pull_request_review_comment", event, GateApi(), "o/r")
    assert verdict["allowed"] is False and "fork" in verdict["reason"]


# --- an earlier Important finding that is still open blocks approval ----------------------------


def test_prior_findings_carry_their_comment_id_and_severity():
    comments = [
        {
            "id": 11,
            "in_reply_to_id": None,
            "user": {"login": post.BOT},
            "path": "p",
            "line": 3,
            "body": f"{post.MARKER}\n🔴 **Important**: Off by one.",
        },
        {
            "id": 12,
            "in_reply_to_id": None,
            "user": {"login": post.BOT},
            "path": "p",
            "line": 9,
            "body": f"{post.MARKER}\n🟡 **Nit**: Name.",
        },
        {"id": 13, "in_reply_to_id": 11, "user": {"login": "simo"}, "body": "fixed"},
    ]
    assert post.prior_findings(comments) == [
        {
            "comment_id": 11,
            "path": "p",
            "line": 3,
            "severity": "important",
            "body": "🔴 **Important**: Off by one.",
        },
        {"comment_id": 12, "path": "p", "line": 9, "severity": "nit", "body": "🟡 **Nit**: Name."},
    ]


def test_an_earlier_important_finding_still_open_blocks_approval():
    assert post.review_event([], live_approval=False, open_prior=1) == "COMMENT"
    assert post.review_event([], live_approval=False, open_prior=0) == "APPROVE"


def test_posting_with_an_open_prior_important_dismisses_the_live_approval_and_says_so():
    api = FakeApi(
        gets={
            "repos/o/r/pulls/7/reviews?per_page=100": [
                {"id": 5, "state": "APPROVED", "user": {"login": post.BOT}, "body": post.MARKER}
            ]
        }
    )
    document = {"summary": "s", "findings": [], "prior_findings_still_open": [11]}
    post.post_review(api, "o/r", 7, "abc", document, DIFF, prior_important={11, 12})
    created = next(b for m, p, b in api.calls if m == "POST" and p == "repos/o/r/pulls/7/reviews")
    assert created["event"] == "COMMENT"
    assert "1 earlier 🔴 Important finding still open" in created["body"]
    assert ("PUT", "repos/o/r/pulls/7/reviews/5/dismissals") in [(m, p) for m, p, _ in api.calls]


def test_an_unknown_id_in_still_open_is_ignored():
    api = FakeApi()
    document = {"summary": "s", "findings": [], "prior_findings_still_open": [999]}
    post.post_review(api, "o/r", 7, "abc", document, DIFF, prior_important={11})
    created = next(b for m, p, b in api.calls if m == "POST" and p == "repos/o/r/pulls/7/reviews")
    assert created["event"] == "APPROVE"
