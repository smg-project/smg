"""Install only the two byte-pinned official repairs under unchanged engine constraints."""

import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from reuse import digest, require

HERE = Path(__file__).parent


def fetch(wheel, directory):
    url = urllib.parse.urlsplit(wheel["url"])
    require(url.scheme == "https" and url.hostname == "files.pythonhosted.org", "wheel host")
    require(Path(wheel["filename"]).name == wheel["filename"], "wheel filename")
    target = directory / wheel["filename"]
    require(not target.exists(), "refuse wheel overwrite")
    with urllib.request.urlopen(wheel["url"], timeout=60) as response:
        data = response.read(wheel["archive_bytes"] + 1)
    require(len(data) == wheel["archive_bytes"], "wheel size")
    require(digest(data) == wheel["archive_sha256"], "wheel digest")
    target.write_bytes(data)
    return target


def main():
    wheels = json.loads((HERE / "wheels.json").read_text())
    require(len(wheels) == 2, "two repair wheels required")
    directory = Path("diagnostic-results/repair-wheels")
    directory.mkdir()
    paths = [fetch(wheel, directory) for wheel in wheels]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--retries", "0", *map(str, paths)],
        check=True,
    )
    (directory / "verified.json").write_text(json.dumps(wheels, indent=2) + "\n")


if __name__ == "__main__":
    main()
