"""The runner of the comparative tokenizer benchmark (crates/tokenizer/benches/compare/run.py):
what the summary of `ids` says of a parity file that is not whole, and what the record of a cell
that ran into the timeout keeps. Shell scripts stand in for the binaries of the columns."""

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tokenizer_compare_run", ROOT / "crates/tokenizer/benches/compare/run.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
FAMILY = runner.MODELS[0]["name"]


def record(index, ids):
    """The line the parity cell writes for one string."""
    return json.dumps({"i": index, "ids": ids}) + "\n"


WHOLE = record(0, [1, 2]) + record(1, [3]) + record(2, [4, 5])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "parity.jsonl").write_text(
            "".join(json.dumps({"i": index, "text": f"text {index}"}) + "\n" for index in range(3))
        )
        self.args = types.SimpleNamespace(
            corpus=corpus,
            models=root / "models",
            output=root / "output",
            columns=[],
            families=[FAMILY],
        )
        (self.args.output / "bin").mkdir(parents=True)

    def column(self, name, script):
        """A stand-in for the binary of a column: a shell script."""
        path = self.args.output / "bin" / name
        path.write_text("#!/bin/sh\n" + script)
        path.chmod(0o755)

    def parity(self, name, text, status=0):
        """A column whose parity cell leaves `text` in its file and ends at `status`."""
        source = self.args.output / f"{name}.parity"
        source.write_text(text)
        # The cell is started as: parity <tokenizer files> <corpus> <the file to write> --no-inc
        self.column(name, f'cp "{source}" "$4"\nexit {status}\n')

    def summary(self):
        """The line of the one family in the summary that `ids` writes."""
        runner.ids(self.args)
        return (self.args.output / "ids-summary.txt").read_text().splitlines()[0]

    def test_whole_files_carry_no_note(self):
        self.parity("smg", WHOLE)
        self.parity("fastokens", WHOLE)
        line = self.summary()
        self.assertIn("3 strings, smg failed on 0; fastokens 3 of 3", line)
        self.assertNotIn("incomplete output", line)
        self.assertNotIn("exit status", line)

    def test_a_file_cut_off_inside_a_line_is_named_at_exit_status_0(self):
        # A write that failed (a full disk) leaves such a file, and the cell still ends at 0.
        self.parity("smg", WHOLE[:-5])
        self.parity("fastokens", WHOLE)
        line = self.summary()
        self.assertIn("2 strings, smg failed on 0; fastokens 2 of 2", line)
        self.assertIn("incomplete output: smg 2 of 3", line)
        self.assertNotIn("exit status", line)

    def test_a_file_that_ends_before_the_last_string_is_named_at_exit_status_0(self):
        self.parity("smg", WHOLE)
        self.parity("fastokens", record(0, [1, 2]))
        line = self.summary()
        self.assertIn("incomplete output: fastokens 1 of 3", line)
        self.assertNotIn("exit status", line)

    def test_an_empty_file_is_named_at_exit_status_0(self):
        # Every write failed, the first one too: the column loaded, and its file holds no string.
        self.parity("smg", WHOLE)
        self.parity("fastokens", "")
        line = self.summary()
        self.assertIn("smg failed on 0; fastokens 0 of 3", line)
        self.assertIn("incomplete output: fastokens 0 of 3", line)
        self.assertNotIn("did not load", line)
        self.assertNotIn("exit status", line)
        # The strings of a column that loaded stand in the total of the run.
        total = (self.args.output / "ids-summary.txt").read_text().splitlines()[-1]
        self.assertEqual(total, "TOTAL SAME: fastokens 0 of 3")

    def test_a_file_cut_off_inside_its_first_line_is_named_at_exit_status_0(self):
        self.parity("smg", WHOLE)
        self.parity("fastokens", WHOLE[:9])
        line = self.summary()
        self.assertIn("smg failed on 0; fastokens 0 of 3", line)
        self.assertIn("incomplete output: fastokens 0 of 3", line)
        self.assertNotIn("did not load", line)
        self.assertNotIn("exit status", line)

    def test_a_non_zero_exit_status_is_named_beside_the_short_file(self):
        self.parity("smg", WHOLE)
        self.parity("fastokens", record(0, [1, 2]) + '{"i": 1, "ids": [3', status=7)
        line = self.summary()
        self.assertIn("exit status not 0: fastokens 7", line)
        self.assertIn("incomplete output: fastokens 1 of 3", line)

    def test_a_non_zero_exit_status_is_named_beside_the_empty_file(self):
        # A process that died before its first write had loaded: its file is there.
        self.parity("smg", WHOLE)
        self.parity("fastokens", "", status=7)
        line = self.summary()
        self.assertIn("exit status not 0: fastokens 7", line)
        self.assertIn("incomplete output: fastokens 0 of 3", line)
        self.assertNotIn("did not load", line)

    def test_a_column_that_did_not_load_is_not_named_as_incomplete(self):
        self.parity("smg", WHOLE)
        self.column("fastokens", "exit 1\n")
        line = self.summary()
        self.assertIn("fastokens did not load", line)
        self.assertIn("exit status not 0: fastokens 1", line)
        self.assertNotIn("incomplete output", line)

    def test_a_column_without_a_file_reads_did_not_load_at_exit_status_0(self):
        # The cell makes its file once the column has loaded: without one, no string is counted.
        self.parity("smg", WHOLE)
        self.column("fastokens", "exit 0\n")
        line = self.summary()
        self.assertIn("fastokens did not load", line)
        self.assertNotIn("incomplete output", line)
        self.assertNotIn("exit status", line)

    def test_a_cell_that_ran_into_the_timeout_keeps_its_stderr(self):
        # A thread that panicked before a barrier leaves its cell waiting until the timeout.
        self.column("smg", 'echo "thread panicked at cells.rs" >&2\nexec sleep 30\n')
        with patch.object(runner, "CELL_TIMEOUT", 2):
            ctx = runner.one(self.args, "smg", FAMILY, None, ["mem"])["ctx"]
        self.assertEqual(ctx["rc"], 124)
        self.assertEqual(ctx["stderr"], "timeout; thread panicked at cells.rs\n")


if __name__ == "__main__":
    unittest.main()
