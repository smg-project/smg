"""The Python proto package gathers its proto files through links, one per file.

``crates/grpc_client/python/smg_grpc_proto/proto`` holds a link to every proto file
of the gRPC client's directory and to the worker crate's ``common.proto``. A build
from the checkout, the nightly package and the release version check read the
package's files through those links, so a proto file without its link would be
missing from all three. The links are read from the index: a workflow step may
already have replaced them with copies in the working tree.
"""

from __future__ import annotations

import posixpath
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE_PROTO = "crates/grpc_client/python/smg_grpc_proto/proto"
SOURCES = ("crates/grpc_client/proto", "crates/worker/proto")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True
    ).stdout


def test_the_proto_package_links_every_proto_file():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout: the links are read from its index")
    links = {}
    for entry in git("ls-files", "--stage", "--", PACKAGE_PROTO).splitlines():
        meta, path = entry.split("\t", 1)
        mode, blob, _stage = meta.split()
        assert mode == "120000", f"{path} is not a link in the tree"
        pointee = git("cat-file", "blob", blob)
        links[path] = posixpath.normpath(posixpath.join(posixpath.dirname(path), pointee))
    sources = sorted(
        path for path in git("ls-files", "--", *SOURCES).splitlines() if path.endswith(".proto")
    )
    assert sorted(links.values()) == sources, "one link for each proto file of the two directories"
    renamed = {
        link: target for link, target in links.items() if Path(link).name != Path(target).name
    }
    assert not renamed, f"a link carries another name than its file: {renamed}"
