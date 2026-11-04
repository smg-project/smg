#!/usr/bin/env python3
"""The release notes draft, and the check that none of it is published unfilled.

``release.sh notes`` reads the merged pull requests of a release (the squash subjects and the
picks, its own git reading), the crates and packages at the previous tag and at the release, the
engine-image table and the published image list, and hands them to ``render``, which fills
``scripts/release/NOTES-TEMPLATE.md``: each pull request's "## Release note" sentence, read from
its body in GraphQL batches through ``gh`` and cached by number, seeds Highlights (features and
performance), Notable fixes (fixes) and Upgrade notes (a declared breaking change, a removed or
renamed flag or setting, a changed default, a configuration that now refuses to start), under the
area of the subject's scope; What's Changed lists every subject by type. The narrative stays as
``<!-- AUTHOR: ... -->`` comments for the author to replace; ``check`` refuses a file that still
carries one, or a ``{{token}}`` the render left.

    notes.py render --version 1.14.0 --since v1.13.0 --release 1.14 ...
    notes.py check notes.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<title>.*)$")
NUMBER = re.compile(r"\(#(\d+)\)\s*$")
HEADING = re.compile(r"^#{1,6}\s+(?P<text>.*?)\s*#*\s*$")
NOTE_HEADING = re.compile(r"^release[ -]notes?\b(?P<rest>.*)$", re.IGNORECASE)
NOTE_LINE = re.compile(r"^\**release[ -]notes?\**\s*:\**\s*(?P<text>.*)$", re.IGNORECASE)
LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
# The spellings of a release note that declares none, one set read in two places: the renderer drops the section
# on it, the check reports a candidate line that still reads as one. Accepted: none, n/a, nothing, no release note,
# no (user-visible | user-facing | visible | behaviour | functional) change(s), not user-visible, internal (only);
# alone, or followed by a reason after a separator (: ; , . ( ) or a dash of any width) or "because". A sentence
# that goes on in its own words ("None of the defaults change", "Nothing breaks when", "No change as long as") is a
# release note; one that reads as none still declares for Upgrade notes through its text (Pull.take).
NONE_NOTE = re.compile(
    r"^\W*(none|n/?a|nothing|no release note|no (user[- ](?:visible|facing) |visible |behaviour |behavior "
    r"|functional )?changes?|not user[- ](?:visible|facing)|internal( only)?)\b"
    r"\s*(?:(?:[:;,.)(\-\u2013\u2014]|because\b).*)?$",
    re.IGNORECASE,
)
NONE_LINE = re.compile(
    r"^- (?P<text>.+?) \(#\d+\)$", re.MULTILINE
)  # a candidate line as the renderer seeds it
DECLARING_HEADING = re.compile(r"breaking|upgrade|compatib|migrat|deprecat", re.IGNORECASE)
DETAILS = re.compile(r"^\s*</?(details|summary)\b", re.IGNORECASE)
CHECKBOX = re.compile(r"^\[[ xX]\]\s*")
BREAKING = re.compile(r"\bBREAKING[ -]CHANGES?\b")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
CODE_SPAN = re.compile(r"`[^`\n]+`")
TOKEN = re.compile(r"\{\{([a-z_]+)\}\}")
BATCH = 100  # pull requests per GraphQL query

# What a pull request declares, read in its subject, its release-note sentence, the body sections whose
# heading speaks of breaking, upgrading, compatibility, migration or deprecation, and any BREAKING CHANGE
# line. Each is a candidate for Upgrade notes, under the subsection its kind suggests.
KINDS = (
    (
        "a configuration that now refuses to start",
        re.compile(
            r"(?<!no longer )(?<!never )(refus\w*|fails?)\s+to\s+start|start-?up (error|failure)"
            r"|(is|are) (now )?(a )?start-?up errors?|stops? (the gateway|start-?up) before"
            r"|exits? (with (code )?)?\d+ (before|at) start",
            re.IGNORECASE,
        ),
    ),
    (
        "a changed default",
        re.compile(
            r"now defaults? to|defaults? (is|are) now|is (now )?the default|new default"
            r"|changed default|default (flip|change|moves|moved|becomes|became|changed|changes)"
            r"|(on|off|enabled|disabled) by default|by default (now|from this release)",
            re.IGNORECASE,
        ),
    ),
    (
        "a removed or renamed flag or setting",
        re.compile(
            r"\b(remove[sd]?|removing|rename[sd]?|renaming|drop(s|ped|ping)?|retire[sd]?|retiring)\b"
            r"[^.\n]{0,80}?\b(flag|option|setting|key|variable|metric|label|endpoint|route|header"
            r"|argument|parameter|field|command|crate|module|--[a-z])"
            r"|\b(flag|option|metric|setting|endpoint|header|label|field|variable)s?\b[^.\n]{0,60}?"
            r"\b(is|was|are|were|gets|got) (removed|renamed|gone|dropped|retired|deleted)"
            r"|no longer (accepted|recogni[sz]ed|exists?|emitted|exported|read|honou?red)",
            re.IGNORECASE,
        ),
    ),
)

# The area of a subject's scope: the heading its candidate sits under. A scope of the excluded set (documentation,
# CI, tests, benchmarks, the tooling) seeds nothing; an unknown scope sits under "Other areas".
AREAS = (
    (
        "Gateway",
        "gateway router routers routing policies policy http api openai anthropic messages responses chat"
        " completions protocols protocol config auth admission health liveness readiness discovery mesh core"
        " storage history grpc streaming sse rl tenancy pd epd kv cache mcp embeddings rerank classify cli server",
    ),
    (
        "Servicers and engines",
        "servicer servicers engine engines engine_zmq_adapter engine-zmq-adapter engine_zmq_client"
        " engine-zmq-client engine_servicer engine-servicer zmq vllm sglang tokenspeed trtllm tensorrt"
        " tensorrt-llm grpc_servicer grpc-servicer scheduler kv-events kv_events launchers",
    ),
    (
        "Symphony",
        "symphony parser parsers tool_parser tool-parser reasoning harmony grammar template templates"
        " response_template response-template chat_template chat-template",
    ),
    (
        "Multimodal",
        "multimodal mm media image images video audio vision llm-multimodal llm_multimodal processor processors",
    ),
    ("Tokenizer", "tokenizer tokenizers tokenization bpe llm-tokenizer llm_tokenizer"),
    (
        "Metrics and observability",
        "metrics observability tracing traces logging logs log otel prometheus telemetry",
    ),
    (
        "Bindings and crates",
        "python bindings binding golang go crate crates openai-protocol openai_protocol grpc_client"
        " grpc-client client clients sdk pyo3 launcher proto grpc_proto grpc-proto",
    ),
    ("Helm and deploy", "helm deploy docker chart k8s kubernetes compose dockerfile container"),
)
OTHER_AREA = "Other areas"
EXCLUDED_SCOPES = frozenset(
    "docs doc readme ci chore build test tests e2e bench benches benchmark benchmarks tau2 bfcl"
    " mock-worker mock_worker mock scripts release deps dependencies workflow workflows".split()
)
STORAGE_SCOPES = frozenset(
    "storage history postgres redis oracle data_connector data-connector conversations conversation store".split()
)
HIGHLIGHT_TYPES = frozenset(("feat", "perf"))
# The types whose wording is read for Upgrade notes; a breaking marker (`!`, BREAKING CHANGE) counts on every type.
DECLARING_TYPES = frozenset(("feat", "fix", "perf", "refactor"))
# What's Changed, by type, as the plain list always read
TYPE_SECTIONS = (
    ("Features", frozenset(("feat",))),
    ("Fixes", frozenset(("fix",))),
    ("Performance", frozenset(("perf",))),
    ("Documentation", frozenset(("docs",))),
    ("Maintenance", frozenset(("ci", "chore", "build", "test", "refactor", "style"))),
)
ENGINE_LABELS = {
    "vllm": "vLLM",
    "sglang": "SGLang",
    "tokenspeed": "TokenSpeed",
    "trtllm": "TensorRT-LLM",
}


def warn(text: str) -> None:
    print(f"notes: {text}", file=sys.stderr)


def outside_code(text: str, change: Callable[[str], str]) -> str:
    """The text with `change` applied to what sits outside backtick spans, which stay as written: a span that
    quotes a comment keeps it, a comment that holds a span goes with it."""
    spans: list[str] = []

    def mask(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"\0{len(spans) - 1}\0"

    return re.sub(
        r"\0(\d+)\0", lambda found: spans[int(found.group(1))], change(CODE_SPAN.sub(mask, text))
    )


def without_comments(text: str) -> str:
    return outside_code(text, lambda part: COMMENT.sub("", part))


@dataclass
class Pull:
    """A merged pull request as its squash subject names it, and what its body says."""

    subject: str
    number: int | None
    type: str
    scope: str
    bang: bool
    read: bool = False  # the body was read (a pull request the repository has)
    body: str = ""
    section: bool = False  # the body has a release-note section, with a sentence or left empty
    text: str | None = None  # the section's sentence as written, kept when it reads as none
    # the release-note sentence; "" when the section declares none; None without a section
    note: str | None = None
    kinds: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, subject: str) -> Pull:
        found = NUMBER.search(subject)
        number = int(found.group(1)) if found else None
        parsed = SUBJECT.match(subject)
        if parsed is None:
            return cls(subject, number, "", "", False)
        scope = (parsed.group("scope") or "").strip().lower()
        return cls(subject, number, parsed.group("type"), scope, parsed.group("bang") == "!")

    @property
    def first_scope(self) -> str:
        """The first scope of the subject (``feat(gateway, python)`` names two)."""
        return re.split(r"[,/ ]+", self.scope)[0] if self.scope else ""

    @property
    def area(self) -> str | None:
        """The area heading of the subject's scope; None for a subject that seeds nothing."""
        if self.type == "docs" or self.first_scope in EXCLUDED_SCOPES:
            return None
        for area, scopes in AREAS:
            if self.first_scope in scopes.split():
                return area
        return OTHER_AREA

    @property
    def seeds(self) -> bool:
        """Whether the pull request is a candidate line: a subject with an area, no note declaring none."""
        return self.area is not None and self.note != ""

    def take(self, body: str) -> None:
        """The body as read: its release-note section, the sentence as written, and the note it is ("" when
        the sentence reads as none; what such a sentence goes on to declare still reaches Upgrade notes)."""
        self.body = body
        self.section, self.text = section_text(body)
        self.note = "" if self.text is not None and NONE_NOTE.match(self.text) else self.text

    def declarations(self) -> str:
        """The text read for Upgrade notes: the subject, the sentence as written, the declaring sections, the
        footers."""
        parts = [self.subject, self.text or ""]
        declaring = False
        for line in without_comments(self.body).split("\n"):
            heading = HEADING.match(line)
            if heading:
                declaring = DECLARING_HEADING.search(heading.group("text")) is not None
            elif declaring or BREAKING.search(line):
                parts.append(line)
        return "\n".join(parts)

    def declare(self) -> None:
        """What the pull request declares for Upgrade notes; nothing for a subject that seeds nothing."""
        self.kinds = []
        if self.area is None:
            return
        if self.type in DECLARING_TYPES:
            text = self.declarations()
            self.kinds = [kind for kind, pattern in KINDS if pattern.search(text)]
        if self.bang or BREAKING.search(self.body):
            self.kinds.insert(0, "a breaking change")

    @property
    def line(self) -> str:
        """The candidate line: the sentence, or the subject marked for the author with what is missing."""
        if self.note:
            return f"- {self.note} (#{self.number})"
        if self.number is None:
            missing = "no pull request number in the subject"
        elif self.read and self.section:
            missing = "the release-note section is empty"
        else:
            missing = "no release note in the pull request"
        return f"- {self.subject} <!-- AUTHOR: {missing}: write the sentence or drop the line -->"

    @property
    def upgrade_line(self) -> str:
        """The Upgrade notes candidate: the release note, or the subject when there is none (a sentence that reads
        as none rides in the comment, so the line still says what the declaration refers to)."""
        if self.note:
            text, said = f"{self.note} (#{self.number})", ""
        else:
            text = self.subject
            quoted = (self.text or "").replace("-->", "--&gt;").replace("<!--", "&lt;!--")
            said = f'; the release note says "{quoted}"' if quoted else ""
        declared = f"declared: {', '.join(self.kinds)}{said}; keep it here, move it or drop it"
        return f"- {text} <!-- AUTHOR: {declared} -->"

    def upgrade_slot(self) -> str:
        """The Upgrade notes subsection a declaring pull request sits under: its kind first, then its area."""
        if "a configuration that now refuses to start" in self.kinds:
            return "refuses_to_start"
        if self.first_scope in STORAGE_SCOPES:
            return "storage"
        if "a changed default" in self.kinds:
            return "defaults"
        return {
            "Metrics and observability": "metrics",
            "Servicers and engines": "engine",
            "Bindings and crates": "bindings",
            "Gateway": "api",
            "Symphony": "api",
            "Multimodal": "api",
            "Tokenizer": "api",
        }.get(self.area or "", "unsorted")


# --- the pull request bodies ------------------------------------------------------------------------


def section_text(body: str) -> tuple[bool, str | None]:
    """Whether the body has a release-note section, and its sentence as written: None when the section is absent
    or was left empty (nothing, or a template's placeholder comment alone).

    The section is a heading (any level) that starts with "Release note", read until the next heading, or a
    "Release note:" line, read until the blank line after it; list markers go, the lines join into one. Comments
    are stripped, and a comment opener or closer left in the sentence is written out as text so it cannot open a
    comment in the draft; what sits in backtick spans stays as written either way.
    """
    mode = None
    collected: list[str] = []
    for line in without_comments(body.replace("\r\n", "\n")).split("\n"):
        heading = HEADING.match(line)
        if heading:
            if mode:
                break
            titled = NOTE_HEADING.match(heading.group("text").strip("*_` "))
            if titled:
                mode = "heading"
                rest = titled.group("rest").strip(" :*_`")
                collected = [rest] if rest else []
            continue
        if mode == "inline" and not line.strip():
            break
        if mode and DETAILS.match(line):
            break  # the pull request template's checklist follows the last section without a heading
        if mode:
            collected.append(line)
            continue
        inline = NOTE_LINE.match(line.strip())
        if inline:
            mode = "inline"
            collected = [inline.group("text")]
    if mode is None:
        return False, None
    parts = (CHECKBOX.sub("", LIST_MARKER.sub("", part.strip())) for part in collected)
    text = re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()
    if not text:
        return True, None
    return True, outside_code(
        text, lambda part: part.replace("<!--", "&lt;!--").replace("-->", "--&gt;")
    )


def note_section(body: str) -> tuple[bool, str | None]:
    """Whether the body has a release-note section, and the note it carries: the sentence, "" when it reads as
    none (NONE_NOTE), None when the section is absent or was left empty."""
    section, text = section_text(body)
    return section, ("" if text is not None and NONE_NOTE.match(text) else text)


def release_note(body: str) -> str | None:
    """The release-note sentence of a body: None without one (no section, or an empty one), "" for a declared none."""
    return note_section(body)[1]


def read_bodies(pulls: list[Pull], gh: str, repo: str, cache: Path, refresh: bool = False) -> None:
    """Each numbered pull request's body from the cache, the rest fetched in batches and cached; every one fetched
    again under refresh (a note added to a pull request after the first draft). A fetch that fails leaves the
    earlier read standing where there is one, and says so."""
    numbered: dict[
        int, list[Pull]
    ] = {}  # a number can be listed twice (a revert, a pick's original)
    for pull in pulls:
        if pull.number is not None:
            numbered.setdefault(pull.number, []).append(pull)
    if not numbered:
        return
    if not repo:
        warn(
            "the repository is unknown (no github.com remote, and gh did not name it): the bodies are not read"
        )
        return
    store = cache / repo.replace("/", "-")
    store.mkdir(parents=True, exist_ok=True)
    records = {} if refresh else {number: cached(store, number) for number in numbered}
    missing = sorted(number for number in numbered if records.get(number) is None)
    if missing:
        fetched = fetch(missing, gh, repo, store)
        for number in missing:
            if number in fetched:
                records[number] = cached(store, number)
        unreached = [number for number in missing if number not in fetched]
        earlier = [number for number in unreached if cached(store, number) is not None]
        for number in earlier:  # a refresh that failed: the earlier read stands, and stderr says so
            records[number] = cached(store, number)
        if earlier:
            warn(
                f"{len(earlier)} of them render from the earlier read; the refresh did not reach them"
            )
        if len(unreached) > len(earlier):
            warn(f"{len(unreached) - len(earlier)} of them fall back to their subjects")
    for number, listed_pulls in numbered.items():
        record = records.get(number)
        if record is None or record.get("missing"):
            continue
        for pull in listed_pulls:
            pull.read = True
            pull.body = record.get("body") or ""


def cached(store: Path, number: int) -> dict | None:
    """The cached record of a pull request; None when there is none, or when the file does not parse (a write cut
    short): read again."""
    try:
        return json.loads((store / f"{number}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def fetch(numbers: list[int], gh: str, repo: str, store: Path) -> set[int]:
    """The bodies of these pull requests through gh, BATCH per GraphQL query, one file each under store; returns
    the numbers written (every number of a batch that answered; a missing record for one that is no pull request)."""
    written: set[int] = set()
    if shutil.which(gh) is None:
        warn(f"'{gh}' is not installed: the bodies of {len(numbers)} pull requests are not read")
        return written
    owner, name = repo.split("/", 1)
    for start in range(0, len(numbers), BATCH):
        batch = numbers[start : start + BATCH]
        fields = " ".join(
            f"p{number}: pullRequest(number: {number}) {{ number title body author {{ login }} }}"
            for number in batch
        )
        query = f'query {{ repository(owner: "{owner}", name: "{name}") {{ {fields} }} }}'
        done = subprocess.run(
            [gh, "api", "graphql", "-f", f"query={query}"], capture_output=True, text=True
        )
        # gh exits 1 when any alias errors (a number that is an issue, not a pull request) and still prints the
        # data; a repository the token cannot see answers "repository": null and no data at all
        try:
            data = json.loads(done.stdout)["data"]["repository"]
            if not isinstance(data, dict):
                raise TypeError("no repository in the answer")
        except (ValueError, KeyError, TypeError) as error:
            first = done.stderr.strip().split("\n")[0] if done.stderr.strip() else ""
            cause = first or str(error) or f"gh exited {done.returncode}"
            warn(f"the read of {len(numbers) - start} pull request bodies failed ({cause})")
            return written
        for number in batch:
            record = data.get(f"p{number}") or {"number": number, "missing": True}
            if "author" in record:
                record["author"] = (record.get("author") or {}).get("login", "")
            # written whole: a run cut during the write leaves no half file for every later render to refuse
            partial = store / f"{number}.json.tmp"
            partial.write_text(json.dumps(record), encoding="utf-8")
            os.replace(partial, store / f"{number}.json")
        written.update(batch)
    return written


# --- the sections --------------------------------------------------------------------------------------


def by_area(pulls: list[Pull], empty: str) -> str:
    blocks = []
    for area in (*(area for area, _ in AREAS), OTHER_AREA):
        lines = [pull.line for pull in pulls if pull.area == area]
        if lines:
            blocks.append(f"**{area}**\n\n" + "\n".join(lines))
    return "\n\n".join(blocks) if blocks else f"<!-- AUTHOR: {empty} -->"


def upgrade_sections(pulls: list[Pull]) -> dict[str, str]:
    slots: dict[str, list[str]] = {
        name: []
        for name in (
            "storage",
            "refuses_to_start",
            "defaults",
            "api",
            "metrics",
            "engine",
            "bindings",
            "unsorted",
        )
    }
    for pull in pulls:
        if pull.kinds:
            slots[pull.upgrade_slot()].append(pull.upgrade_line)
    empty = "<!-- AUTHOR: nothing declared by the pull requests under this heading: write it or delete the heading. -->"
    sections = {
        f"upgrade_{name}": "\n".join(lines) or empty
        for name, lines in slots.items()
        if name not in ("storage", "unsorted")
    }
    # Storage opens the list only when a release carries a storage change (a schema migration): the heading comes
    # with its candidates, or not at all
    sections["upgrade_storage"] = (
        "### Storage\n\n" + "\n".join(slots["storage"]) if slots["storage"] else ""
    )
    unsorted = slots["unsorted"]
    sections["upgrade_unsorted"] = (
        "<!-- AUTHOR: candidates with no obvious subsection: place each under one above, or drop it. -->\n\n"
        + "\n".join(unsorted)
        if unsorted
        else ""
    )
    return sections


def whats_changed(pulls: list[Pull], count: int, since: str, release: str) -> str:
    lines = [
        f"{count} commits since {since} on `release/{release}` (the merged pull requests, by type):"
    ]
    known = frozenset().union(*(types for _, types in TYPE_SECTIONS))
    sections = [
        (title, [pull.subject for pull in pulls if pull.type in types])
        for title, types in TYPE_SECTIONS
    ]
    sections.append(("Other", [pull.subject for pull in pulls if pull.type not in known]))
    for title, subjects in sections:
        if subjects:
            lines.append(f"\n### {title}\n")
            lines.extend(f"- {subject}" for subject in subjects)
    return "\n".join(lines)


def listed(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def components(path: Path) -> dict[tuple[str, str], str]:
    """{(kind, name): version} from a component list (``kind name version manifest path`` per line)."""
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if len(words) >= 3:
            versions[(words[0], words[1])] = words[2]
    return versions


def engine_bases(workflow: str) -> str | None:
    """The engine images' base images from the workflow's table, as one clause; None without a readable table."""
    match = re.search(r"engines='(\{.*?\})'", workflow, re.S)
    if match is None:
        return None
    try:
        table = json.loads(match.group(1))
    except ValueError:
        return None
    clauses = []
    for engine, entry in table.items():
        bases = [base for base in (entry.get("release_bases") or []) if base] or [
            "the latest nightly carrier"
        ]
        clauses.append(f"{ENGINE_LABELS.get(engine, engine)} ({listed(bases)})")
    return listed(clauses) if clauses else None


def packages(args: argparse.Namespace) -> str:
    """The Packages section: the publish plan's components at the release, against the previous tag."""
    before = components(Path(args.components_since)) if args.components_since else {}
    after = components(Path(args.components_ref)) if args.components_ref else {}
    bases = (
        engine_bases(Path(args.engine_workflow).read_text(encoding="utf-8"))
        if args.engine_workflow
        else None
    )
    chart = args.chart or args.version
    gateway = f"- `smg` {args.version} on PyPI and crates.io, with container images and Helm chart {chart}."
    if bases:
        gateway += f" Release images build on {bases}."
    else:
        gateway += (
            " <!-- AUTHOR: the engine images' bases"
            " (the engine-image workflow's table is not readable at this ref) -->"
        )
    lines = [gateway]
    others = [
        f"`{name}` {version}"
        for (kind, name), version in sorted(after.items())
        if kind == "package" and name != "smg"
    ]
    if others:
        lines.append(
            f"- {listed(others)}."
            " <!-- AUTHOR: the upgrade order between gateways and engine pods when the servicer changed. -->"
        )
    crates_after = {
        name: version
        for (kind, name), version in after.items()
        if kind == "crate" and name != "smg"
    }
    crates_before = {name: version for (kind, name), version in before.items() if kind == "crate"}
    new = [
        f"`{name}` {version}"
        for name, version in sorted(crates_after.items())
        if name not in crates_before
    ]
    moved = [
        f"`{name}` {version} (from {crates_before[name]})"
        for name, version in sorted(crates_after.items())
        if name in crates_before and crates_before[name] != version
    ]
    crates = []
    if new:
        crates.append(f"New crates: {', '.join(new)}.")
    if moved:
        crates.append(f"The library crates moved to new versions: {', '.join(moved)}.")
    if crates:
        lines.append(
            f"- {' '.join(crates)}"
            " <!-- AUTHOR: name the ones whose public types changed (see Bindings and Rust crates). -->"
        )
    if args.rust_version:
        lines.append(f"- Building from source needs Rust {args.rust_version}.")
    docker = Path(args.docker).read_text(encoding="utf-8").strip() if args.docker else ""
    return "\n".join(lines) + (f"\n\n{docker}" if docker else "")


# --- render and check --------------------------------------------------------------------------------


def render(args: argparse.Namespace) -> int:
    template = Path(args.template).read_text(encoding="utf-8")
    subjects = [
        line
        for line in Path(args.subjects).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pulls = [
        Pull.of(subject)
        for subject in subjects
        if not subject.startswith("chore(release): bump versions")
    ]
    read_bodies(pulls, args.gh, args.repo, Path(args.cache), args.refresh)
    for pull in pulls:
        if pull.read:
            pull.take(pull.body)
        pull.declare()
    with_note = sum(1 for pull in pulls if pull.note)
    none = sum(1 for pull in pulls if pull.note == "")
    empty = sum(1 for pull in pulls if pull.read and pull.section and pull.note is None)
    no_section = sum(1 for pull in pulls if pull.read and not pull.section)
    unread = sum(1 for pull in pulls if pull.number is not None and not pull.read)
    no_number = sum(1 for pull in pulls if pull.number is None)
    warn(
        f"{len(pulls)} pull requests: {with_note} carry a release note, {none} declare none, {empty} leave the"
        f" section empty, {no_section} have no release-note section, {unread} could not be read, {no_number}"
        " subjects name no pull request (the last four fall back to their subject, marked for the author)"
    )
    values = {
        "version": args.version,
        "previous": args.since.removeprefix("v"),
        "commits": str(args.count),
        "contributors": str(args.contributors),
        "highlights": by_area(
            [pull for pull in pulls if pull.type in HIGHLIGHT_TYPES and pull.seeds],
            "no feature or performance pull request carried a release note: write the highlights by hand.",
        ),
        "notable_fixes": by_area(
            [pull for pull in pulls if pull.type == "fix" and pull.seeds],
            "no fix pull request carried a release note: write the notable fixes by hand.",
        ),
        **upgrade_sections(pulls),
        "packages": packages(args),
        "full_changelog": f"[Full changelog]({args.compare_url})"
        if args.compare_url
        else "<!-- AUTHOR: the compare link of the previous final and this tag -->",
        "whats_changed": whats_changed(pulls, args.count, args.since, args.release),
    }
    unknown = sorted({name for name in TOKEN.findall(template) if name not in values})
    if unknown:
        warn(
            f"the template names {', '.join('{{' + name + '}}' for name in unknown)}, which the render does not fill"
        )
        return 2
    text = TOKEN.sub(lambda found: values[found.group(1)], template)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n") + "\n"
    sys.stdout.write(text)
    return 0


def template_tokens(path: Path) -> set[str] | None:
    """The token names the template defines; None when it cannot be read (every token-shaped text counts then)."""
    try:
        return set(TOKEN.findall(path.read_text(encoding="utf-8")))
    except OSError:
        return None


def placeholders(text: str, names: set[str] | None = None) -> list[tuple[int, str]]:
    """(line, excerpt) of every author comment, every template token of the given names, and every comment left
    open in the text.

    A sentence may quote a template expression of its own (`{{ bos_token }}`): only the names the notes template
    defines are placeholders, every token-shaped text when the names are unknown. An author comment counts wherever
    "AUTHOR:" sits in it (a stray opener before it would swallow it), and an opener without a closer is reported
    too: published, it hides everything after it, or up to the closer of the next comment, which absorbs it (a
    second opener inside one matched comment). Backtick spans are skipped: a comment quoted in code hides nothing.
    A candidate line whose sentence reads as a declared none (NONE_NOTE) is reported as well: the renderer seeds
    none, so one in a draft came from an older render or a hand, and it is not a release note.
    """
    found = []
    scanned = CODE_SPAN.sub(
        lambda span: " " * len(span.group(0)), text
    )  # the same offsets, the spans blanked
    closed_to = 0
    for match in COMMENT.finditer(scanned):
        closed_to = match.end()
        # the last opener in the match is the comment's; one before it had no closer and hides the prose between
        opener = match.group(0).rfind("<!--")
        if opener:
            found.append(
                (
                    text.count("\n", 0, match.start()) + 1,
                    "<!-- (an opener without a closer: it hides everything up to the next closer)",
                )
            )
        start = match.start() + opener
        if "AUTHOR:" in scanned[start : match.end()].upper():
            excerpt = text[start : match.end()].split("\n")[0]
            found.append((text.count("\n", 0, start) + 1, excerpt))
    stray = scanned.find("<!--", closed_to)
    if stray != -1:
        found.append(
            (
                text.count("\n", 0, stray) + 1,
                "<!-- (an opener without a closer: it hides everything after it)",
            )
        )
    for match in TOKEN.finditer(text):
        if names is None or match.group(1) in names:
            found.append((text.count("\n", 0, match.start()) + 1, match.group(0)))
    for match in NONE_LINE.finditer(text):
        if NONE_NOTE.match(match.group("text")):
            found.append((text.count("\n", 0, match.start()) + 1, match.group(0)))
    return sorted(found)


def check(args: argparse.Namespace) -> int:
    path = Path(args.file)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        warn(f"{path} is empty")
        return 1
    left = placeholders(text, template_tokens(Path(args.template)))
    if not left:
        warn(f"{path} carries no placeholder")
        return 0
    comments = sum(1 for _, excerpt in left if excerpt.startswith("<!--"))
    tokens = sum(1 for _, excerpt in left if excerpt.startswith("{{"))
    nones = len(left) - comments - tokens
    warn(
        f"{path} still carries {len(left)} placeholder(s): {comments} author comment(s), {tokens} template token(s),"
        f" {nones} line(s) that say none"
    )
    for line, excerpt in left:
        shown = excerpt if len(excerpt) <= 100 else excerpt[:97] + "..."
        print(f"  line {line}: {shown}", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("render", help="the notes draft on stdout")
    for name in ("version", "since", "release", "template", "subjects"):
        draft.add_argument(f"--{name}", required=True)
    draft.add_argument("--count", type=int, required=True, help="the commits What's Changed counts")
    draft.add_argument("--contributors", type=int, default=0)
    draft.add_argument(
        "--components-since", default="", help="publishable's lines at the previous tag"
    )
    draft.add_argument("--components-ref", default="", help="publishable's lines at the release")
    draft.add_argument(
        "--engine-workflow", default="", help="the engine-image workflow at the release"
    )
    draft.add_argument(
        "--docker",
        default="",
        help="the published image list, as release_docker_notes.sh prints it",
    )
    draft.add_argument("--chart", default="", help="the Helm chart version at the release")
    draft.add_argument("--rust-version", default="")
    draft.add_argument(
        "--repo", default="", help="owner/name of the repository the pull requests are read from"
    )
    draft.add_argument("--gh", default="gh")
    draft.add_argument(
        "--cache", required=True, help="where the pull request bodies are kept, by number"
    )
    draft.add_argument("--compare-url", default="")
    draft.add_argument(
        "--refresh", action="store_true", help="read every body again, whatever the cache holds"
    )
    draft.set_defaults(run=render)
    verify = commands.add_parser("check", help="refuse notes that still carry a placeholder")
    verify.add_argument("file")
    verify.add_argument("--template", default=str(Path(__file__).with_name("NOTES-TEMPLATE.md")))
    verify.set_defaults(run=check)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
