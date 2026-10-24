"""Stdlib controls for owned log identity; no engine or transfer is simulated."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from nixl_environment import mooncake_transfer


class TransferLogPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.cases = [
            "router/test_pd_mooncake.py::TestPDMooncakeKvTransfer::" + name + "[pd_grpc]"
            for name in ("test_minted_params_injected", "test_workers_transferred_kv")
        ]
        self.recipe = {"cases": self.cases}
        self.events = []

    def logs(self, positive=True, relative=True):
        for role in ("prefill", "decode"):
            path = self.root / "e2e-logs" / (role + ".log")
            self.events.append(
                {
                    "nodeid": self.cases[0],
                    "kind": "worker",
                    "identity": {
                        "role": role,
                        "log": str(path),
                        "kv_config": {"kv_connector": "MooncakeConnector"},
                    },
                }
            )
            raw = b"Num successful transfers=2\n" if positive else b"Num successful transfers=0\n"
            self.increment(str(path.relative_to(self.root)) if relative else str(path), raw)
        path = str(self.root / "gateway.log")
        self.events.append(
            {
                "nodeid": self.cases[0],
                "kind": "gateway",
                "identity": {"pid": 1},
                "log_sources": [path],
            }
        )
        self.increment(path, b"vLLM PD (Mooncake): injecting minted kv_transfer_params\n")

    def increment(self, source, raw, nodeid=None):
        artifact = str(len(self.events)) + ".log"
        (self.root / artifact).write_bytes(raw)
        self.events.append(
            {
                "nodeid": nodeid or self.cases[1],
                "kind": "log",
                "source": source,
                "artifact": artifact,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    def verify(self):
        mooncake_transfer.verify(self.recipe, self.events, self.root)
        return json.loads((self.root / "mooncake-transfer.json").read_text())

    def test_relative_capture_matches_absolute_owned_identity(self):
        self.logs()
        self.assertEqual(self.verify()["positive_transfer_counts"], [2, 2])

    def test_absolute_capture_stays_valid(self):
        self.logs(relative=False)
        self.assertEqual(self.verify()["owned_roles"], ["decode", "prefill"])

    def test_foreign_positive_cannot_satisfy_owned_zero(self):
        self.logs(positive=False)
        self.increment("e2e-logs/foreign.log", b"Num successful transfers=99\n")
        with self.assertRaisesRegex(ValueError, "did not report successful KV transfer"):
            self.verify()

    def test_other_case_positive_cannot_satisfy_owned_zero(self):
        self.logs(positive=False)
        self.increment("e2e-logs/decode.log", b"Num successful transfers=99\n", "other-case")
        with self.assertRaisesRegex(ValueError, "did not report successful KV transfer"):
            self.verify()

    def test_tampered_increment_stays_red(self):
        self.logs()
        next(row for row in self.events if row["kind"] == "log")["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "log increment differs"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
