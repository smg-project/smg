#!/usr/bin/env python3
"""Generate ``smg_grpc_proto`` stubs from this checkout's proto files, for tests.

The released package builds its stubs at install time (``crates/grpc_client/
python/setup.py``); while a proto change is in flight the tests need stubs of
the local ``.proto`` files instead. This writes an importable
``smg_grpc_proto`` package (the same layout as the release) into a directory,
without committing generated code:

    python3 grpc_servicer/scripts/gen_proto_stubs.py /tmp/smg-proto-gen
    SMG_GRPC_PROTO_PATH=/tmp/smg-proto-gen pytest -q grpc_servicer/tests

Requires ``grpcio-tools`` (the release caps it below 1.82 for protobuf 6).
"""

from __future__ import annotations

import pathlib
import shutil
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = pathlib.Path(argv[1]).resolve()
    repo = pathlib.Path(__file__).resolve().parents[2]
    proto_dir = repo / "crates" / "grpc_client" / "proto"
    package_src = repo / "crates" / "grpc_client" / "python" / "smg_grpc_proto"
    protos = sorted(proto_dir.glob("*.proto"))
    if not protos:
        print(f"no .proto files under {proto_dir}", file=sys.stderr)
        return 1

    import grpc_tools
    from grpc_tools import protoc

    package = target / "smg_grpc_proto"
    generated = package / "generated"
    if package.exists():
        shutil.rmtree(package)
    generated.mkdir(parents=True)
    init = (package_src / "__init__.py").read_text()
    init = init.replace(
        '__version__ = version("smg-grpc-proto")',
        'try:\n    __version__ = version("smg-grpc-proto")\n'
        "except Exception:  # locally generated stubs carry no distribution metadata\n"
        '    __version__ = "0.0.0+local"',
    )
    (package / "__init__.py").write_text(init)
    (generated / "__init__.py").write_text('"""Auto-generated protobuf stubs. Do not edit."""\n')

    well_known = pathlib.Path(grpc_tools.__file__).parent / "_proto"
    args = [
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--proto_path={well_known}",
        f"--python_out={generated}",
        f"--grpc_python_out={generated}",
        f"--pyi_out={generated}",
        *map(str, protos),
    ]
    if protoc.main(args) != 0:
        print("protoc failed", file=sys.stderr)
        return 1
    # grpcio-tools emits absolute imports between the generated modules.
    for module in generated.glob("*_pb2*.py"):
        text = module.read_text()
        for proto in protos:
            name = proto.stem + "_pb2"
            text = text.replace(f"import {name}", f"from . import {name}")
        module.write_text("# mypy: ignore-errors\n" + text)
    print(f"generated {len(protos)} protos into {package}")
    print(f"export SMG_GRPC_PROTO_PATH={target}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
