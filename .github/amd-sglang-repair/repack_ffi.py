"""Repack the audited upstream wheel with only the read-only class-field fix."""

import base64
import csv
import hashlib
import io
import json
import zipfile

VERSION = "0.1.12+amdreadonlydict2"
ORIGINAL_SHA = "817af52916ca9987e019ae9c811406835c7f26c590b2a7bcfa9db0e3809f4228"
CANDIDATE_SHA = "552d381104af049dd3f7bb5bd310ff66203f46a86e9371a561412a3996b0d19b"
OLD_INFO = "apache_tvm_ffi-0.1.12.dist-info/"
NEW_INFO = "apache_tvm_ffi-" + VERSION + ".dist-info/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def requirements(data):
    return [
        line
        for line in data.splitlines()
        if line.startswith((b"Requires-Dist:", b"Requires-Python:"))
    ]


def repack(original, directory):
    require(sha(original.read_bytes()) == ORIGINAL_SHA, "official wheel digest")
    target = directory / original.name.replace("-0.1.12-", "-" + VERSION + "-")
    require(not target.exists(), "refuse candidate overwrite")
    with zipfile.ZipFile(original) as archive:
        upstream = {i.filename: archive.read(i) for i in archive.infolist() if not i.is_dir()}
    source = upstream["tvm_ffi/registry.py"].decode()
    require(
        sha(source.encode()) == "f0378d38c7738281c21b7100d9cafa037ed421c118ede8a1484214a414d5c395",
        "original registry source",
    )
    needle = "        if name not in type_cls.__dict__:  # skip attributes defined directly on this class\n"
    require(source.count(needle) == 1, "exact registry patch context")
    fixed = source.replace(
        needle,
        "        # type.__dict__ is read-only even when absent from the class body.\n"
        '        if name == "__dict__":\n            continue\n' + needle,
    )
    files = {
        name.replace(OLD_INFO, NEW_INFO): data
        for name, data in upstream.items()
        if not name.endswith(".dist-info/RECORD")
    }
    files["tvm_ffi/registry.py"] = fixed.encode()
    version = files["tvm_ffi/_version.py"].decode()
    version = version.replace("version = '0.1.12'", "version = '" + VERSION + "'")
    version = version.replace(
        "version_tuple = (0, 1, 12)", "version_tuple = (0, 1, 12, 'amdreadonlydict2')"
    )
    files["tvm_ffi/_version.py"] = version.encode()
    metadata = files[NEW_INFO + "METADATA"]
    require(metadata.count(b"Version: 0.1.12\n") == 1, "original package version")
    files[NEW_INFO + "METADATA"] = metadata.replace(
        b"Version: 0.1.12\n", ("Version: " + VERSION + "\n").encode()
    )
    require(
        requirements(metadata) == requirements(files[NEW_INFO + "METADATA"]), "requirements changed"
    )
    native = {}
    changes = []
    for old_name, data in upstream.items():
        if old_name.endswith(".dist-info/RECORD"):
            continue
        name = old_name.replace(OLD_INFO, NEW_INFO)
        if files[name] != data:
            changes.append(name)
        if old_name.endswith((".so", ".h", ".hpp", ".cc", ".cpp", ".c", ".pyx", ".pxi")):
            require(files[name] == data, "native/header bytes changed")
            native[name] = sha(data)
    require(
        set(changes) == {"tvm_ffi/registry.py", "tvm_ffi/_version.py", NEW_INFO + "METADATA"},
        "unexpected changes",
    )
    record_path = NEW_INFO + "RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        writer.writerow(
            [
                name,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("="),
                len(data),
            ]
        )
    writer.writerow([record_path, "", ""])
    files[record_path] = record.getvalue().encode()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 22, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)
    require(
        target.stat().st_size == 8908001 and sha(target.read_bytes()) == CANDIDATE_SHA,
        "candidate wheel identity",
    )
    (directory / "local-ffi-repack.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "original_sha256": ORIGINAL_SHA,
                "candidate_sha256": CANDIDATE_SHA,
                "candidate_bytes": target.stat().st_size,
                "native_and_header_sha256_unchanged": native,
                "requirements_unchanged": [line.decode() for line in requirements(metadata)],
                "changed_members": changes,
                "native_binary_rebuilt": False,
            },
            indent=2,
        )
        + "\n"
    )
    return target
