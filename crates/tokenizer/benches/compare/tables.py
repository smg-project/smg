#!/usr/bin/env python3
"""Write the tables of a run from its csv files, in the form of the crate README.

    tables.py                     RESULTS.md beside this file, from the three csv files beside it
    tables.py --readme PATH       also replace the tables of that README, from its heading
                                  "#### Encode, cold" to the next "## " heading

Every figure is the median over a cell's rounds of the p50 of one process. A margin is the
reference over SMG, minus one, for a time, and SMG over the reference, minus one, for a
throughput: positive means SMG is ahead. A change is this release over the previous one, minus
one. Standard library only.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = json.loads((HERE / "models.json").read_text())
CORPUS = json.loads((HERE / "corpus.json").read_text())
SIZES = ("256", "2k", "32k", "128k", "500k", "1m")
# Two builds of one source differ by up to about this much on a cell from code placement alone.
BAND = 0.20
# A change between two releases of this crate counts as faster or slower beyond this much.
LEVEL = 0.05
# The step and first-delta cells time one call between two reads of a clock that moves in steps of
# about this many nanoseconds: a change of one step or less is below what those cells resolve.
TICK = 10.0


def read(name):
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator - 1


def percent(value):
    return "-" if value is None else f"{100 * value:+.0f}"


class Run:
    """The figures of a run: the median over the rounds, by family, cell and column."""

    def __init__(self, new, old):
        self.new, self.old = new, old
        self.encode, self.misc, self.memory, self.backend = {}, {}, {}, {}
        # Lines without a figure, by family, cell and column: the process failed, or it ended at
        # status 0 and its cell found its own stream text wrong (the status `incorrect`).
        self.failed, self.wrong = {}, {}
        for row in read("encode-rounds.csv"):
            if row["backend"]:
                self.backend.setdefault((row["family"], row["arm"]), row["backend"])
            if row["rc"] == "0" and row["ns_per_token_p50"]:
                key = (row["family"], row["class"], row["mode"], row["arm"])
                self.encode.setdefault(key, []).append(float(row["ns_per_token_p50"]))
        for row in read("misc-rounds.csv"):
            if row["backend"]:
                # The decode lines say which path a build took where the encode file has no line of it.
                self.backend.setdefault((row["family"], row["arm"]), row["backend"])
            if row["rc"] != "0":
                counts = self.wrong if row["rc"] == "incorrect" else self.failed
                key = (row["family"], row["cell"], row["arm"])
                counts[key] = counts.get(key, 0) + 1
                continue
            for name, text in row.items():
                if name.endswith(("_p50", "_p99")) and text:
                    key = (row["family"], row["cell"], name, row["arm"])
                    self.misc.setdefault(key, []).append(float(text))
        for row in read("load-memory.csv"):
            for name, text in row.items():
                if name not in ("family", "arm", "backend") and text not in ("", "-"):
                    self.memory[(row["family"], name, row["arm"])] = float(text)

    def v1_encodes(self, family):
        """Whether the v1 build took the v1 encoder for the family (else its next encode path)."""
        return "hf:v1" in self.backend.get((family, "v1"), "")

    def v1_path(self, family):
        """The encode path the v1 build took on a family whose file the v1 pipeline declines."""
        backend = self.backend.get((family, "v1"), "")
        if "hf:tokenizers" in backend:
            return "the `tokenizers` crate's"
        if "hf:native" in backend or "tiktoken" in backend:
            return "SMG's own"
        return f"`{backend}`" if backend else "one the csv files do not name"

    def enc(self, family, cls, mode, arm):
        if arm == "v1" and not self.v1_encodes(family):
            return None
        return median(self.encode.get((family, cls, mode, arm), []))

    def other(self, family, cell, name, arm):
        return median(self.misc.get((family, cell, name, arm), []))

    def batch(self, family, arm):
        """Tokens per second of the batch call where the column has one, else of the plain loop."""
        if arm == "v1" and not self.v1_encodes(family):
            return None
        api = self.other(family, "batch1000", "batch_api_tok_per_s_p50", arm)
        if api is not None:
            return api
        return self.other(family, "batch1000", "batch_loop_tok_per_s_p50", arm)


def encode_table(run, mode):
    title = f"#### Encode, {mode} (ns per token; in brackets the margin vs fastokens / vs v1, in %)"
    lines = [title, "", "| family | vocabulary | " + " | ".join(SIZES) + " |", "|---" * 8 + "|"]
    for model in MODELS:
        cells = []
        for cls in SIZES:
            ours = run.enc(model["name"], cls, mode, run.new)
            if ours is None:
                cells.append("-")
                continue
            margins = [
                percent(ratio(run.enc(model["name"], cls, mode, arm), ours))
                for arm in ("fastokens", "v1")
            ]
            cells.append(f"{ours:.1f} ({' / '.join(margins)})")
        lines.append(f"| {model['family']} | {model['vocabulary']} | " + " | ".join(cells) + " |")
    return lines


def changes(run, model, mode):
    values = []
    for cls in SIZES:
        new = run.enc(model["name"], cls, mode, run.new)
        values.append(ratio(new, run.enc(model["name"], cls, mode, run.old)))
    return [value for value in values if value is not None]


def release_table(run):
    def cell(values):
        if not values:
            return "-"
        low, high = 100 * min(values), 100 * max(values)
        return f"{100 * statistics.median(values):+.1f} % ({low:+.1f} % .. {high:+.1f} %)"

    old, new = run.old.removeprefix("smg-"), run.new.removeprefix("smg-")
    lines = [
        f"#### v{old} to {new} (change in ns per token; negative = {new} faster)",
        "",
        "| family | cold: median change (range over the six sizes) | warm: median change (range) |",
        "|---|---|---|",
    ]
    for model in MODELS:
        cold, warm = cell(changes(run, model, "cold")), cell(changes(run, model, "warm"))
        lines.append(f"| {model['family']} | {cold} | {warm} |")
    return lines


# title, cell, csv column, the unit of a figure
DECODE = (
    ("full decode, 2,048 ids", "dec2048", "full_ns_per_token_p50", "ns per token"),
    ("incremental decode, 2,048 ids", "dec2048", "inc_ns_per_token_p50", "ns per token"),
    ("first delta of one stream", "dec2048", "first_delta_ns_p50", "ns"),
    ("64 streams: step", "stream64", "stream_step_ns_p50", "ns"),
    ("64 streams: first delta", "stream64", "stream_first_delta_wall_ns_p50", "ns"),
)
BATCH = "batch of 1,000 texts, one thread"


def decode_table(run):
    lines = [
        "#### Decode and streaming (ns per token or ns per step; in brackets the margin vs fastokens,"
        " in %; the batch column, in Mtok/s, carries the margin vs fastokens / vs v1)",
        "",
        "| family | " + " | ".join(entry[0] for entry in DECODE) + f" | {BATCH} (Mtok/s) |",
        "|---" * 7 + "|",
    ]
    for model in MODELS:
        family = model["name"]
        cells = []
        for _, cell, name, _ in DECODE:
            ours = run.other(family, cell, name, run.new)
            theirs = run.other(family, cell, name, "fastokens")
            cells.append("-" if ours is None else f"{ours:.1f} ({percent(ratio(theirs, ours))})")
        ours = run.batch(family, run.new)
        if ours is None:
            cells.append("-")
        else:
            margins = [percent(ratio(ours, run.batch(family, arm))) for arm in ("fastokens", "v1")]
            cells.append(f"{ours / 1e6:.1f} ({' / '.join(margins)})")
        lines.append(f"| {model['family']} | " + " | ".join(cells) + " |")
    return lines


def moves(run, family):
    """Each decode, streaming and batch cell of a family against the previous release.

    One tuple per column of the table: the title, this release's figure, the previous one's,
    the change, whether more is better (a throughput), and the scale and unit of a figure.
    """
    found = []
    for title, cell, name, unit in DECODE:
        now, before = (run.other(family, cell, name, arm) for arm in (run.new, run.old))
        found.append((title, now, before, ratio(now, before), False, 1.0, unit))
    now, before = run.batch(family, run.new), run.batch(family, run.old)
    found.append((BATCH, now, before, ratio(now, before), True, 1e6, "Mtok/s"))
    return found


def decode_release_table(run):
    old, new = run.old.removeprefix("smg-"), run.new.removeprefix("smg-")
    lines = [
        f"#### Decode, streaming and batch: v{old} to {new} (change in time; negative = {new}"
        " faster; the batch column is the change in tokens per second, where positive ="
        f" {new} faster)",
        "",
        "| family | " + " | ".join(entry[0] for entry in DECODE) + f" | {BATCH} |",
        "|---" * 7 + "|",
    ]
    for model in MODELS:
        cells = [
            "-" if entry[3] is None else f"{100 * entry[3]:+.1f} %"
            for entry in moves(run, model["name"])
        ]
        lines.append(f"| {model['family']} | " + " | ".join(cells) + " |")
    return lines


# title, csv column, divisor, format, the words of the reading list
MEMORY = (
    ("load, ms", "load_ms", 1.0, ".1f", "the load time"),
    (
        "resident memory after the load, MB",
        "rss_kb_after_load",
        1024.0,
        ".1f",
        "the resident memory after the load",
    ),
    (
        "high-water mark after a 1m-token encode, MB",
        "hwm_kb_after_1m",
        1024.0,
        ".0f",
        "the high-water mark after a 1m-token encode",
    ),
    (
        "growth per thread at 64 threads, one shared tokenizer, kB",
        "growth_kb_per_thread_shared",
        1.0,
        ".0f",
        "the growth per thread on one shared tokenizer",
    ),
    (
        "resident level per thread, a tokenizer per thread, MB",
        "level_kb_per_thread_own",
        1024.0,
        ".1f",
        "the resident level per thread with a tokenizer per thread",
    ),
)


def memory_table(run):
    old, new = run.old.removeprefix("smg-"), run.new.removeprefix("smg-")
    lines = [
        f"#### Load and memory (each cell: SMG {new} / SMG v{old} / fastokens)",
        "",
        "| family | " + " | ".join(entry[0] for entry in MEMORY) + " |",
        "|---" * 6 + "|",
    ]
    for model in MODELS:
        cells = []
        for _, name, divisor, form, _ in MEMORY:
            values = [
                run.memory.get((model["name"], name, arm))
                for arm in (run.new, run.old, "fastokens")
            ]
            cells.append(
                " / ".join("-" if v is None else format(v / divisor, form) for v in values)
            )
        lines.append(f"| {model['family']} | " + " | ".join(cells) + " |")
    return lines


def sides(margins):
    """How many margins stand beyond the placement band on each side, and inside it."""
    ahead = sum(1 for margin in margins if margin > BAND)
    behind = sum(1 for margin in margins if margin < -BAND)
    return ahead, behind, len(margins) - ahead - behind


def against(title, arm, margins, note=""):
    """One line of the reading list: a column of the decode table against a reference."""
    ahead, behind, inside = sides(margins)
    return (
        f"- {title[0].upper()}{title[1:]}, against {arm}: SMG is ahead beyond the band on {ahead}"
        f" of {len(margins)} families, behind beyond it on {behind}, inside it on {inside} (the"
        f" margins run from {percent(min(margins))} % to {percent(max(margins))} %).{note}"
    )


def slower(named):
    """The sentence that names every slower cell with both figures; empty without one."""
    return " Slower: " + "; ".join(named) + "." if named else ""


def fastest(run, column):
    """The references of the csv files, and by mode in how many encode cells the column reads
    a lower time than every one of them (of how many cells)."""
    references = sorted({arm for _, _, _, arm in run.encode} - {run.new, run.old})
    found = {}
    for mode in ("cold", "warm"):
        cells = ahead = 0
        for model in MODELS:
            for cls in SIZES:
                ours = run.enc(model["name"], cls, mode, column)
                others = [run.enc(model["name"], cls, mode, arm) for arm in references]
                others = [value for value in others if value is not None]
                if ours is not None and others:
                    cells += 1
                    ahead += all(ours < value for value in others)
        found[mode] = (ahead, cells)
    return references, found


def reading(run):
    old, new = run.old.removeprefix("smg-"), run.new.removeprefix("smg-")
    lines = ["#### Reading the tables", ""]
    references, ours = fastest(run, run.new)
    _, theirs = fastest(run, run.old)
    if ours["cold"][1] + ours["warm"][1]:
        lines.append(
            f"- Of {ours['cold'][1] + ours['warm'][1]} encode cells SMG {new} reads a lower time"
            f" than every reference in the csv files ({', '.join(references)}) on"
            f" {ours['cold'][0] + ours['warm'][0]}: {ours['cold'][0]} of {ours['cold'][1]} cold and"
            f" {ours['warm'][0]} of {ours['warm'][1]} warm; SMG v{old} in the same rounds reads"
            f" {theirs['cold'][0] + theirs['warm'][0]}."
        )
    for mode in ("cold", "warm"):
        moved = []
        for model in MODELS:
            for cls in SIZES:
                now = run.enc(model["name"], cls, mode, run.new)
                before = run.enc(model["name"], cls, mode, run.old)
                change = ratio(now, before)
                if change is not None:
                    moved.append((change, f"{model['family']} at {cls}", now, before))
        if not moved:
            continue
        faster = sum(1 for entry in moved if entry[0] < -LEVEL)
        slow = sorted((entry for entry in moved if entry[0] > LEVEL), reverse=True)
        named = [
            f"{name}, {now:.1f} against {before:.1f} ns per token ({100 * change:+.1f} %)"
            for change, name, now, before in slow
        ]
        lines.append(
            f"- Against v{old}, {mode}: of {len(moved)} encode cells {faster} read faster and"
            f" {len(slow)} slower by more than {100 * LEVEL:.0f} %,"
            f" {len(moved) - faster - len(slow)} within it; the median change is"
            f" {100 * statistics.median(entry[0] for entry in moved):+.1f} %.{slower(named)}"
        )
    moved = [
        (model["family"], *entry)
        for model in MODELS
        for entry in moves(run, model["name"])
        if entry[3] is not None
    ]
    if moved:

        def worse(entry):
            return -entry[4] if entry[5] else entry[4]

        def one_step(entry):
            return entry[7] == "ns" and abs(entry[2] - entry[3]) <= 1.1 * TICK

        fast = [entry for entry in moved if worse(entry) < -LEVEL]
        slow = sorted((entry for entry in moved if worse(entry) > LEVEL), key=worse, reverse=True)
        named = [
            f"{entry[0]}, {entry[1]}, {entry[2] / entry[6]:.1f} against {entry[3] / entry[6]:.1f}"
            f" {entry[7]} ({100 * entry[4]:+.1f} %{', one step of the clock' if one_step(entry) else ''})"
            for entry in slow
        ]
        steps = sum(1 for entry in fast if one_step(entry))
        clock = (
            f" The step and first-delta cells read a clock that moves in steps of about {TICK:.0f}"
            f" ns; {steps} of the faster cells differ by one step or less."
            if steps or any(one_step(entry) for entry in slow)
            else ""
        )
        lines.append(
            f"- Against v{old}, decode, streaming and batch: of {len(moved)} cells {len(fast)} read"
            f" faster and {len(slow)} slower by more than {100 * LEVEL:.0f} %,"
            f" {len(moved) - len(fast) - len(slow)} within it.{slower(named)}{clock}"
        )
    for arm in ("fastokens", "v1"):
        for mode in ("cold", "warm"):
            margins = []
            for model in MODELS:
                for cls in SIZES:
                    ours = run.enc(model["name"], cls, mode, run.new)
                    margin = ratio(run.enc(model["name"], cls, mode, arm), ours)
                    if margin is not None:
                        margins.append(margin)
            if not margins:
                continue
            ahead, behind, inside = sides(margins)
            lines.append(
                f"- Against {arm}, {mode}: of {len(margins)} encode cells SMG is ahead beyond the"
                f" placement band ({100 * BAND:.0f} %) on {ahead} and behind beyond it on {behind};"
                f" {inside} read inside the band (the margins run from {percent(min(margins))} %"
                f" to {percent(max(margins))} %)."
            )
    for title, cell, name, _ in DECODE:
        margins, missing, failed, wrong = [], [], 0, 0
        for model in MODELS:
            ours = run.other(model["name"], cell, name, run.new)
            margin = ratio(run.other(model["name"], cell, name, "fastokens"), ours)
            if margin is not None:
                margins.append(margin)
            elif ours is not None:
                missing.append(model["family"])
                key = (model["name"], cell, "fastokens")
                failed += run.failed.get(key, 0) > 0
                wrong += run.wrong.get(key, 0) > 0
        note = ""
        if missing:
            why = ""
            if failed == len(missing) and not wrong:
                why = ", whose process failed in every round"
            elif wrong == len(missing) and not failed:
                why = ", whose stream text was wrong in every round"
            elif failed or wrong:
                why = ", whose process failed or gave a wrong stream text"
            note = f" The fastokens column has no figure for {', '.join(missing)}{why}."
        if margins:
            lines.append(against(title, "fastokens", margins, note))
    for arm in ("fastokens", "v1"):
        margins = [
            ratio(run.batch(model["name"], run.new), run.batch(model["name"], arm))
            for model in MODELS
        ]
        margins = [margin for margin in margins if margin is not None]
        if margins:
            lines.append(against(BATCH, arm, margins))
    for label, arm in ((f"v{old}", run.old), ("fastokens", "fastokens")):
        parts = []
        for _, name, _, _, words in MEMORY:
            lower, higher, count = 0, [], 0
            for model in MODELS:
                change = ratio(
                    run.memory.get((model["name"], name, run.new)),
                    run.memory.get((model["name"], name, arm)),
                )
                if change is None:
                    continue
                count += 1
                if change < -LEVEL:
                    lower += 1
                elif change > LEVEL:
                    higher.append(model["family"])
            if count:
                more = f" (higher on {', '.join(higher)})" if higher else ""
                parts.append(f"{lower} of {count} families in {words}{more}")
        if parts:
            lines.append(
                f"- Load and memory, against {label}: SMG {new} reads lower by more than"
                f" {100 * LEVEL:.0f} % on " + "; on ".join(parts) + "."
            )
    declined = [model for model in MODELS if not run.v1_encodes(model["name"])]
    if declined:
        # Which path that build took on each of them, as its backend text says.
        took = {}
        for model in declined:
            took.setdefault(run.v1_path(model["name"]), []).append(model["family"])
        paths = "; ".join(f"{path} for {', '.join(names)}" for path, names in took.items())
        lines.append(
            f"- The v1 column reads `-` for {', '.join(model['family'] for model in declined)}:"
            " the v1 pipeline declines their files, and that build takes its next encode path"
            f" there ({paths})."
        )
    spread = []
    for model in MODELS:
        for _, cell, name, _ in DECODE:
            ours = run.other(model["name"], cell, name, run.new)
            spread.append(ratio(run.other(model["name"], cell, name, "v1"), ours))
    spread = [value for value in spread if value is not None]
    if spread:
        lines.append(
            "- The placement band: the v1 build shares SMG's decode and streaming code, and between the"
            f" two builds those {len(spread)} cells read {percent(min(spread))} % to"
            f" {percent(max(spread))} %."
        )
    shares = CORPUS["non_ascii_bytes_percent"].values()
    low, high = min(shares), max(shares)
    lines.append(
        f"- The corpus: {low:.0f} to {high:.0f} % of the bytes of a size class are outside ASCII"
        " (`corpus.json`)."
    )
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--new", default="smg-1.16.0", help="the column of this release")
    parser.add_argument("--old", default="smg-1.15.0", help="the column of the previous release")
    parser.add_argument("--readme", type=Path, help="a README whose tables to replace")
    args = parser.parse_args()
    run = Run(args.new, args.old)
    blocks = [
        encode_table(run, "cold"),
        encode_table(run, "warm"),
        release_table(run),
        decode_table(run),
        decode_release_table(run),
        memory_table(run),
        reading(run),
    ]
    tables = "\n\n".join("\n".join(block) for block in blocks) + "\n"
    head = (
        "# Comparative tokenizer benchmark: results\n\n"
        "Written by `tables.py` from `encode-rounds.csv`, `misc-rounds.csv` and `load-memory.csv`;"
        " [README.md](README.md) says what a cell measures and how to run it.\n\n"
    )
    (HERE / "RESULTS.md").write_text(head + tables.replace("#### ", "## "), encoding="utf-8")
    if args.readme:
        text = args.readme.read_text(encoding="utf-8")
        start = text.index("#### Encode, cold")
        end = text.index("\n## ", start) + 1
        args.readme.write_text(text[:start] + tables + "\n" + text[end:], encoding="utf-8")


if __name__ == "__main__":
    main()
