"""The file-discovery e2e's own pieces: the manifest writer, and the flags
that point a gateway at it. The rest is in e2e_test/discovery/common.py."""

from __future__ import annotations

import json
from pathlib import Path

from ..common import Gateway

# A periodic reread no test outlives: under it, only the startup read and
# the watcher read the manifest.
NO_REREAD_SECS = 3600


class Manifest:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def document(workers: list[dict]) -> str:
        return json.dumps({"version": 1, "workers": workers})

    def write(self, workers: list[dict]) -> None:
        """Replace the manifest as writers should: a temporary file beside
        it, renamed over it."""
        staged = self.path.with_suffix(".staged")
        staged.write_text(self.document(workers))
        staged.replace(self.path)

    def rewrite_in_place(self, workers: list[dict]) -> None:
        """Overwrite the manifest itself, as an editor does."""
        with open(self.path, "w") as manifest:
            manifest.write(self.document(workers))


def file_gateway(manifest: Manifest, log_path: Path) -> Gateway:
    """A gateway, not yet started, discovering the workers `manifest` lists."""
    return Gateway(
        lambda check_interval_secs: [
            "--discovery-provider",
            "file",
            "--discovery-file",
            str(manifest.path),
            "--discovery-check-interval-secs",
            str(check_interval_secs),
        ],
        log_path,
    )
