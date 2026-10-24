"""Check all wheel RECORD entries and pinned native bytes without importing them."""

import base64
import csv
import io
import stat
import zipfile
from pathlib import PurePosixPath

from artifact_io import digest, require


def verify(payload, native=None):
    with zipfile.ZipFile(io.BytesIO(payload)) as wheel:
        infos = wheel.infolist()
        require(len(infos) <= 20000, "wheel member count")
        require(sum(i.file_size for i in infos) <= 1024**3, "wheel expanded size")
        names = [i.filename for i in infos]
        require(len(names) == len(set(names)), "duplicate wheel member")
        files = {}
        for info in infos:
            path, mode = PurePosixPath(info.filename), info.external_attr >> 16
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in info.filename
                and not info.flag_bits & 1
                and not stat.S_ISLNK(mode),
                "unsafe wheel member",
            )
            if not info.is_dir():
                files[info.filename] = wheel.read(info)
        records = [name for name in files if name.endswith(".dist-info/RECORD")]
        require(len(records) == 1, "wheel RECORD identity")
        checked = {}
        for row in csv.reader(io.StringIO(files[records[0]].decode())):
            require(len(row) == 3, "malformed RECORD row")
            name, encoded_hash, size = row
            require(name in files and name not in checked, "unknown/duplicate RECORD entry")
            if name == records[0]:
                require(not encoded_hash and not size, "RECORD self-entry")
            else:
                expected = "sha256=" + base64.urlsafe_b64encode(
                    bytes.fromhex(digest(files[name]))
                ).decode().rstrip("=")
                require(encoded_hash == expected, f"wheel RECORD hash: {name}")
                require(size == str(len(files[name])), f"wheel RECORD size: {name}")
            checked[name] = {"bytes": len(files[name]), "sha256": digest(files[name])}
        require(set(checked) == set(files), "RECORD does not cover wheel members")
        if native is not None:
            require(native["name"] in checked, "native member missing")
            require(
                checked[native["name"]] == {"bytes": native["bytes"], "sha256": native["sha256"]},
                "native member identity",
            )
        return {"record": records[0], "members": checked, "imported_or_installed": False}
