"""The verdicts written by the comparative tokenizer benchmark tables."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tokenizer_compare_tables", ROOT / "crates/tokenizer/benches/compare/tables.py"
)
assert SPEC and SPEC.loader
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


MODELS = [{"name": "fixture", "family": "fixture"}]
# Three rounds a release whose medians differ by eight percent while the rounds overlap, and
# three whose every round of the new release stands below every round of the old one.
OVERLAPPING = ([100.0, 96.0, 104.0], [92.0, 101.0, 90.0])
APART = ([100.0, 99.0, 101.0], [90.0, 89.0, 91.0])


def batch_run(old, new, column="batch_api_tok_per_s_p50"):
    run = object.__new__(tables.Run)
    run.new, run.old = "smg-new", "smg-old"
    run.encode, run.memory, run.backend = {}, {}, {}
    run.failed, run.wrong = {}, {}
    run.misc = {
        ("fixture", "batch1000", column, run.old): list(old),
        ("fixture", "batch1000", column, run.new): list(new),
    }
    return run


def reading(run):
    with patch.object(tables, "MODELS", MODELS):
        return "\n".join(tables.reading(run))


def batch_cell(run):
    """The batch cell of the fixture row in the release table: its last column."""
    with patch.object(tables, "MODELS", MODELS):
        row = tables.decode_release_table(run)[-1]
    return row.rstrip(" |").rsplit("|", 1)[-1].strip()


class TableTests(unittest.TestCase):
    def test_three_noisy_batch_rounds_are_not_reported_as_moved(self):
        old, new = OVERLAPPING
        noisy = reading(batch_run(old, new))
        settled = reading(batch_run(old * 4, new * 4))

        self.assertNotIn("decode, streaming and batch", noisy)
        self.assertIn("decode, streaming and batch: of 1 cells", settled)
        self.assertIn("1 slower", settled)

    def test_three_batch_rounds_wholly_apart_are_reported_as_moved(self):
        apart = reading(batch_run(*APART))

        self.assertIn("decode, streaming and batch: of 1 cells", apart)
        self.assertIn("1 slower", apart)

    def test_release_table_shows_a_dash_for_an_unresolved_batch_cell(self):
        self.assertEqual(batch_cell(batch_run(*OVERLAPPING)), "-")
        self.assertEqual(batch_cell(batch_run(*APART)), "-10.0 %")

    def test_batch_rounds_are_counted_on_the_loop_column_without_a_batch_call(self):
        old, new = OVERLAPPING
        column = "batch_loop_tok_per_s_p50"
        few = batch_run(old, new, column)
        ten = batch_run(old * 4, new * 4, column)

        self.assertEqual(few.batch("fixture", few.old), 100.0)
        self.assertEqual(batch_cell(few), "-")
        self.assertEqual(batch_cell(ten), "-8.0 %")


if __name__ == "__main__":
    unittest.main()
