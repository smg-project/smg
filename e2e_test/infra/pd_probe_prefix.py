"""Bounded startup/admission prefixes; never cancellation ACK or trace completeness."""

import base64
import hashlib
import json
import math
import os
import re
import stat

BYTE_LIMIT, RECORD_LIMIT = 256 * 1024, 256


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def read_prefix(directory):
    paths = sorted(directory.iterdir())
    if len(paths) != 2:
        raise ValueError("expected exactly two scheduler files")
    files, count, size = {}, 0, 0
    for path in paths:
        if re.fullmatch(r"scheduler-[1-9][0-9]*-[0-9a-f]{32}\.jsonl", path.name) is None:
            raise ValueError("unknown trace filename")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("nonregular trace")
        with open(
            path, "rb", opener=lambda p, f: os.open(p, f | os.O_NOFOLLOW | os.O_NONBLOCK)
        ) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("nonregular trace")
            raw = stream.read(BYTE_LIMIT - size + 1)
        size += len(raw)
        if size > BYTE_LIMIT or not raw or not raw.endswith(b"\n"):
            raise ValueError("oversized, empty or truncated trace")
        records = [
            json.loads(
                line,
                object_pairs_hook=_object,
                parse_float=_number,
                parse_constant=_number,
            )
            for line in raw.decode("utf-8").splitlines()
        ]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("trace record must be an object")
        count += len(records)
        if count > RECORD_LIMIT:
            raise ValueError("record limit exceeded")
        files[path.name] = (raw, records)
    return files


def export_prefix(metadata, files):
    if (
        len(files) > 2
        or sum(len(raw) for raw, _ in files.values()) > BYTE_LIMIT
        or sum(len(records) for _, records in files.values()) > RECORD_LIMIT
    ):
        raise ValueError("export limits exceeded")
    payload = dict(metadata, accepted=False, observation="partial", files={})
    payload.update(
        file_count=len(files),
        byte_count=sum(len(raw) for raw, _ in files.values()),
        record_count=sum(len(rows) for _, rows in files.values()),
    )
    chunks = []
    for name, (raw, records) in files.items():
        if re.fullmatch(r"scheduler-[1-9][0-9]*-[0-9a-f]{32}\.jsonl", name) is None:
            raise ValueError("unknown export filename")
        encoded = base64.b64encode(raw).decode("ascii")
        parts = [encoded[i : i + 3072] for i in range(0, len(encoded), 3072)]
        payload["files"][name] = dict(
            bytes=len(raw),
            records=len(records),
            chunks=len(parts),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        chunks.extend(f"SMG_PD_PROBE_CHUNK {name} {i} {part}" for i, part in enumerate(parts))
    print(
        "SMG_PD_PROBE_BEGIN",
        json.dumps(payload, allow_nan=False),
        *chunks,
        "SMG_PD_PROBE_END",
        sep="\n",
        flush=True,
    )
