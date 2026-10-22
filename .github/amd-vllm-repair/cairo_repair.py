"""Target-container Cairo prerequisites and native proof; no model imports."""

import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

from reuse import require

OUT = Path("diagnostic-results/pycairo")
SOURCE = "https://files.pythonhosted.org/packages/bb/5e/19ab2572981a996ce96b853a189d2dcced40506b67d9555a7568457abcf2/pycairo-1.29.2.tar.gz"
SOURCE_SHA = "3e69fff74fe64f5ba2dfa31f67c6bdf26413342574047437d2ac520d35e9a489"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def prerequisites():
    include = Path(sysconfig.get_path("include"))
    header = include / "Python.h"
    patch = include / "patchlevel.h"
    text = patch.read_text() if patch.is_file() else ""
    headers_ok = header.is_file() and all(
        re.search(rf"#define\s+PY_{name}_VERSION\s+{value}\b", text)
        for name, value in [("MAJOR", sys.version_info.major), ("MINOR", sys.version_info.minor)]
    )
    tools = {name: shutil.which(name) for name in ["cc", "pkg-config", "readelf", "ldconfig"]}
    checks = []
    cairo_ok = False
    if tools["pkg-config"]:
        checks = [command(["pkg-config", "--exists", "cairo >= 1.15.10"])]
        cairo_ok = checks[0]["returncode"] == 0
        checks += [
            command(["pkg-config", arg, "cairo"])
            for arg in ["--modversion", "--cflags", "--variable=pcfiledir", "--variable=includedir"]
        ]
    packages = []
    if not headers_ok:
        packages.append(f"python{sys.version_info.major}.{sys.version_info.minor}-dev")
    if not tools["cc"]:
        packages.append("build-essential")
    if not tools["readelf"]:
        packages.append("binutils")
    if not cairo_ok:
        packages += ["pkg-config", "libcairo2-dev"]
    require(tools["ldconfig"], "missing libc-bin ldconfig")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "include": str(include),
        "platinclude": sysconfig.get_path("platinclude"),
        "Python.h": str(header),
        "headers_ok": headers_ok,
        "tools": tools,
        "cairo_checks": checks,
        "needed_packages": packages,
        "libcairo_candidates": command(["ldconfig", "-p"]),
    }


def preflight():
    OUT.mkdir(parents=True)
    record = {"source_url": SOURCE, "source_sha256": SOURCE_SHA, "apt": []}
    try:
        require(sys.version_info[:2] == (3, 12), "unexpected active Python ABI")
        record["before"] = prerequisites()
        packages = record["before"]["needed_packages"]
        if packages:
            require(
                os.environ.get("GITHUB_ACTIONS") == "true"
                and Path("/.dockerenv").is_file()
                and os.geteuid() == 0,
                "apt repair requires the disposable Actions container",
            )
            for argv in [
                ["apt-get", "update"],
                ["apt-get", "install", "-y", "--no-install-recommends", *packages],
            ]:
                result = command(argv)
                record["apt"].append(result)
                require(result["returncode"] == 0, "container prerequisite installation failed")
        record["after"] = prerequisites()
        require(
            not record["after"]["needed_packages"], "active Python/Cairo prerequisites unresolved"
        )
        pc_dir = record["after"]["cairo_checks"][3]["stdout"].strip()
        pc = Path(pc_dir) / "cairo.pc"
        require(pc.is_file(), "missing actual cairo.pc")
        record["cairo_pc"] = {"path": str(pc.resolve()), "sha256": sha(pc), "text": pc.read_text()}
    finally:
        (OUT / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")


def verify():
    import cairo

    require(metadata.version("pycairo") == "1.29.2", "unexpected pycairo version")
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    context = cairo.Context(surface)
    context.set_source_rgb(0.2, 0.4, 0.6)
    context.paint()
    output = io.BytesIO()
    surface.write_to_png(output)
    png = output.getvalue()
    require(
        png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (2, 2),
        "native PNG proof failed",
    )
    (OUT / "native-2x2.png").write_bytes(png)
    extension = Path(cairo._cairo.__file__).resolve()
    wheels = list((OUT / "wheel").glob("pycairo-1.29.2-*.whl"))
    require(len(wheels) == 1, "missing/ambiguous target-built pycairo wheel")
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = [name for name in wheel.namelist() if name.endswith(".so")]
        require(
            len(members) == 1
            and hashlib.sha256(wheel.read(members[0])).hexdigest() == sha(extension),
            "installed extension differs from target-built wheel",
        )
    loaded = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and re.fullmatch(r"libcairo\.so(?:\.\d+)*", Path(fields[5]).name):
            loaded.add(Path(fields[5]).resolve())
    require(len(loaded) == 1, "missing/ambiguous actually loaded libcairo")
    library = loaded.pop()
    elf = command(["readelf", "-h", str(library)])
    require(elf["returncode"] == 0, "loaded Cairo ELF inspection failed")
    versions = command(["readelf", "-V", str(library)])
    require(versions["returncode"] == 0, "loaded Cairo ELF version inspection failed")
    record_text = metadata.distribution("pycairo").read_text("RECORD")
    require(record_text, "missing actual installed wheel RECORD")
    (OUT / "installed-RECORD.txt").write_text(record_text)
    record = {
        "pycairo": metadata.version("pycairo"),
        "cairo": cairo.cairo_version_string(),
        "python": sys.version,
        "executable": sys.executable,
        "wheel": str(wheels[0]),
        "wheel_sha256": sha(wheels[0]),
        "extension": str(extension),
        "extension_sha256": sha(extension),
        "loaded_libcairo": str(library),
        "loaded_libcairo_sha256": sha(library),
        "loaded_elf": elf,
        "loaded_elf_versions": versions,
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "source_url": SOURCE,
        "source_sha256": SOURCE_SHA,
    }
    (OUT / "native-proof.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "verify"])
    args = parser.parse_args()
    (preflight if args.mode == "preflight" else verify)()
