"""The Codex review run assembles its skills from the guide repo and the review agents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import codex_review_skills as skills  # noqa: E402

AGENT = """\
---
name: silent-failure-hunter
description: Use this agent when reviewing code changes for silent failures.
model: inherit
color: yellow
---

You are an elite error handling auditor.

## Core Principles
"""


def test_an_agent_becomes_a_skill_with_only_name_and_description_in_front():
    text = skills.agent_to_skill(AGENT)
    assert text.startswith(
        "---\nname: silent-failure-hunter\n"
        "description: Use this agent when reviewing code changes for silent failures.\n---\n"
    )
    assert "model: inherit" not in text and "color:" not in text
    assert "You are an elite error handling auditor." in text and "## Core Principles" in text


def test_assemble_links_guide_skills_and_writes_agent_skills(tmp_path):
    guide = tmp_path / "guide"
    (guide / "skills" / "review-pr").mkdir(parents=True)
    (guide / "skills" / "review-pr" / "SKILL.md").write_text(
        "---\nname: review-pr\ndescription: d\n---\n"
    )
    (guide / ".agents" / "skills").mkdir(parents=True)
    (guide / ".agents" / "skills" / "smg-review-pr").symlink_to("../../skills/review-pr")
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "silent-failure-hunter.md").write_text(AGENT)
    target = tmp_path / "home" / ".agents" / "skills"

    installed = skills.assemble(target, guide, agents)

    assert (target / "smg-review-pr" / "SKILL.md").read_text().startswith("---\nname: review-pr")
    assert (
        (target / "silent-failure-hunter" / "SKILL.md")
        .read_text()
        .startswith("---\nname: silent-failure-hunter\n")
    )
    assert installed == ["silent-failure-hunter", "smg-review-pr"]
