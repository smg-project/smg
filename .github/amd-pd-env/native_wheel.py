"""Reproduce native CPU proof for the exact CI-built wheel; never builds a wheel."""

import hashlib
import importlib.metadata as metadata
import io
import json
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

from reuse_foundation import require

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
    binding = json.loads((OUT.parent / "environment-binding.json").read_text())
    original = binding["producer_native"]
    record["same_loaded_libcairo"] = (
        record["loaded_libcairo_sha256"] == original["loaded_libcairo_sha256"]
    )
    record["producer_wheel_sha256"] = binding["wheel_sha256"]
    (OUT / "native-proof.json").write_text(json.dumps(record, indent=2) + "\n")
    require(
        record["same_loaded_libcairo"]
        and record["cairo"] == original["cairo"]
        and record["extension_sha256"] == original["extension_sha256"]
        and record["wheel_sha256"] == binding["wheel_sha256"],
        "consumer native ABI/bytes differ from producer proof",
    )


if __name__ == "__main__":
    verify()
