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


def batch_run(rounds):
    run = object.__new__(tables.Run)
    run.new, run.old = "smg-new", "smg-old"
    run.encode, run.memory, run.backend = {}, {}, {}
    run.failed, run.wrong = {}, {}
    run.misc = {
        ("fixture", "batch1000", "batch_api_tok_per_s_p50", run.old): [100.0] * rounds,
        ("fixture", "batch1000", "batch_api_tok_per_s_p50", run.new): [90.0] * rounds,
    }
    return run


class TableTests(unittest.TestCase):
    def test_three_noisy_batch_rounds_are_not_reported_as_moved(self):
        models = [{"name": "fixture", "family": "fixture"}]
        with patch.object(tables, "MODELS", models):
            noisy = "\n".join(tables.reading(batch_run(3)))
            settled = "\n".join(tables.reading(batch_run(10)))

        self.assertNotIn("decode, streaming and batch", noisy)
        self.assertIn("decode, streaming and batch: of 1 cells", settled)
        self.assertIn("1 slower", settled)


if __name__ == "__main__":
    unittest.main()
