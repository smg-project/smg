"""Every entry of a workflow's path filter names a path on the tree.

A `paths` or `paths-ignore` entry that names a moved file or directory matches
nothing: the workflow it starts (a benchmark gate, a fixture job) stops running
on the changes it was written for, and the entry reads as coverage. The root of
an entry is its leading path components up to the first glob component; a root
that does not exist on the tree is a stale entry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = sorted(
    path
    for suffix in ("yml", "yaml")
    for path in (ROOT / ".github" / "workflows").glob(f"*.{suffix}")
)
GLOB = re.compile(r"[*?\[]")


def filter_root(entry: str) -> str:
    """The leading path components of a filter entry with no glob in them;
    the whole entry when it holds none. A negation (`!`) is an entry too. An
    entry whose first component is a glob (`**/*.md`) has an empty root and
    the tree check skips it."""
    components = []
    for component in entry.lstrip("!").split("/"):
        if GLOB.search(component):
            break
        components.append(component)
    return "/".join(components)


def filter_entries(workflow: Path):
    """Every `(event, key, entry)` of the workflow's `paths` and `paths-ignore`."""
    document = yaml.safe_load(workflow.read_text())
    # PyYAML reads the bare key `on` as the boolean true.
    triggers = document.get("on", document.get(True)) or {}
    if not isinstance(triggers, dict):
        return
    for event, spec in triggers.items():
        if not isinstance(spec, dict):
            continue
        for key in ("paths", "paths-ignore"):
            for entry in spec.get(key) or []:
                yield event, key, entry


@pytest.mark.parametrize(
    "entry,root",
    [
        ("Cargo.toml", "Cargo.toml"),
        ("model_gateway/src/**", "model_gateway/src"),
        ("!crates/**/tests/**", "crates"),
        ("model_gateway/tests/bellwether_*.rs", "model_gateway/tests"),
        ("**/*.md", ""),
    ],
)
def test_the_root_of_an_entry_is_its_glob_free_prefix(entry, root):
    assert filter_root(entry) == root


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_every_path_filter_entry_has_a_root_on_the_tree(workflow):
    stale = [
        f"{event}.{key}: {entry}"
        for event, key, entry in filter_entries(workflow)
        if filter_root(entry) and not (ROOT / filter_root(entry)).exists()
    ]
    assert not stale, f"{workflow.name} starts on paths that do not exist: {stale}"
