"""The Python proto package gathers its proto files through links, one per file.

``crates/grpc_client/python/smg_grpc_proto/proto`` holds a link to every proto file
of the gRPC client's directory and to the worker crate's ``common.proto``. A build
from the checkout, the nightly package and the release version check read the
package's files through those links, so a proto file without its link would be
missing from all three. The links are read from the index: a workflow step may
already have replaced them with copies in the working tree.

The script that generates the servicer tests' stubs reads the same two
directories, named in its own text: the second case holds the names equal.
"""

from __future__ import annotations

import ast
import posixpath
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE_PROTO = "crates/grpc_client/python/smg_grpc_proto/proto"
SOURCES = ("crates/grpc_client/proto", "crates/worker/proto")
STUB_SCRIPT = "grpc_servicer/scripts/gen_proto_stubs.py"


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


def path_parts(node: ast.expr) -> list[str]:
    """The quoted parts of a ``repo / "a" / "b"`` path expression, in order."""
    if isinstance(node, ast.BinOp):
        return path_parts(node.left) + path_parts(node.right)
    return [node.value] if isinstance(node, ast.Constant) else []


def test_the_stub_script_reads_the_linked_directories():
    script = ast.parse((REPO / STUB_SCRIPT).read_text())
    named = [
        node.value
        for node in ast.walk(script)
        if isinstance(node, ast.Assign)
        and [getattr(target, "id", None) for target in node.targets] == ["proto_dirs"]
    ]
    assert len(named) == 1, "the script names its proto directories in one list"
    assert tuple("/".join(path_parts(path)) for path in named[0].elts) == SOURCES
