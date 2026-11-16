#!/usr/bin/env python3
"""Assemble the skills a Codex review run loads, the way the Claude review loads its plugins.

    codex_review_skills.py --target ~/.agents/skills \
        --guide <checkout of smg-project/smg-dev-guide> \
        --agents <checkout>/plugins/pr-review-toolkit/agents

The guide repo ships Codex-format skills under `.agents/skills` (links into `skills/`); each is
copied under its link name (`smg-review-pr`, `smg-map`, ...). The review toolkit ships Claude
agents, Markdown with a frontmatter of `name`, `description`, `model`, `color`; Codex skills
need `name` and `description` alone, so each agent becomes `<name>/SKILL.md` with the other
keys dropped and the body kept. Codex reads `$HOME/.agents/skills` on every run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

FENCE = "---"


def split_frontmatter(text: str) -> tuple[list[str], str]:
    """The frontmatter lines (without the fences) and the body after them."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != FENCE:
        raise ValueError("no frontmatter")
    for index in range(1, len(lines)):
        if lines[index].strip() == FENCE:
            return lines[1:index], "".join(lines[index + 1 :])
    raise ValueError("unterminated frontmatter")


def frontmatter_fields(lines: list[str]) -> dict[str, str]:
    """`key: value` pairs; an indented line continues the previous value."""
    fields: dict[str, str] = {}
    key = None
    for raw in lines:
        line = raw.rstrip("\n")
        if line[:1] in (" ", "\t") and key:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip()
    return fields


def agent_to_skill(text: str) -> str:
    """A Claude agent definition as a Codex skill: name and description in front, body as is."""
    front, body = split_frontmatter(text)
    fields = frontmatter_fields(front)
    name = fields.get("name")
    description = fields.get("description")
    if not name or not description:
        raise ValueError("agent frontmatter needs `name` and `description`")
    return f"{FENCE}\nname: {name}\ndescription: {description}\n{FENCE}\n{body}"


def assemble(target: Path, guide: Path | None, agents: Path | None) -> list[str]:
    """Install the guide's skills and the converted agents under `target`; the names installed."""
    target.mkdir(parents=True, exist_ok=True)
    installed: list[str] = []
    if guide is not None:
        for entry in sorted((guide / ".agents" / "skills").iterdir()):
            source = entry.resolve()
            if not (source / "SKILL.md").is_file():
                continue
            destination = target / entry.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            installed.append(entry.name)
    if agents is not None:
        for agent in sorted(agents.glob("*.md")):
            skill = agent_to_skill(agent.read_text())
            name = frontmatter_fields(split_frontmatter(skill)[0])["name"]
            destination = target / name
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "SKILL.md").write_text(skill)
            installed.append(name)
    return sorted(installed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target", required=True, type=Path, help="the skills directory to fill")
    parser.add_argument("--guide", type=Path, help="a checkout of smg-project/smg-dev-guide")
    parser.add_argument("--agents", type=Path, help="a directory of Claude agent .md files")
    args = parser.parse_args(argv)
    for name in assemble(args.target, args.guide, args.agents):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
