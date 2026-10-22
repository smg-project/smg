"""CPU raw-prefix controls; these fixtures never represent a backend pass."""

import base64
import hashlib
import json
import os

import pytest
from infra import pd_probe_prefix as prefix


@pytest.fixture
def sample(tmp_path):
    for index in (1, 2):
        (tmp_path / f"scheduler-{index}-{'a' * 31}{index}.jsonl").write_bytes(
            b'{"schema":1,"seq":1,"padding":"' + b"x" * 4000 + b'"}\n'
        )
    return tmp_path


def test_original_bytes_are_fully_recoverable(sample, capsys):
    files = prefix.read_prefix(sample)
    prefix.export_prefix({"sha": "a" * 40, "run": "run"}, files)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "SMG_PD_PROBE_BEGIN" and lines[-1] == "SMG_PD_PROBE_END"
    header, chunks = json.loads(lines[1]), {}
    assert header["accepted"] is False and header["observation"] == "partial"
    for line in lines[2:-1]:
        marker, name, index, data = line.split()
        assert marker == "SMG_PD_PROBE_CHUNK" and len(line) < 4096
        assert int(index) == len(chunks.setdefault(name, []))
        chunks[name].append(data)
    for name, (raw, rows) in files.items():
        assert raw == (sample / name).read_bytes()
        assert base64.b64decode("".join(chunks[name]), validate=True) == raw
        assert header["files"][name] == dict(
            bytes=len(raw),
            records=len(rows),
            chunks=len(chunks[name]),
            sha256=hashlib.sha256(raw).hexdigest(),
        )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "symlink",
        "fifo",
        "filename",
        "torn",
        "empty",
        "malformed",
        "unicode",
        "duplicate",
        "nan",
        "overflow",
        "object",
        "bytes",
        "records",
    ],
)
def test_raw_prefix_is_bounded_and_strict(sample, fault):
    path = next(sample.iterdir())
    if fault == "missing":
        path.unlink()
    elif fault == "extra":
        (sample / "extra").touch()
    elif fault == "filename":
        path.rename(sample / "foreign.jsonl")
    elif fault in ("symlink", "fifo"):
        path.unlink()
        if fault == "symlink":
            path.symlink_to(next(sample.iterdir()))
        else:
            os.mkfifo(path)
    else:
        values = {
            "torn": b"{}",
            "empty": b"",
            "malformed": b"{\n",
            "unicode": b"\xff\n",
            "duplicate": b'{"a":1,"a":2}\n',
            "nan": b'{"a":NaN}\n',
            "overflow": b'{"a":1e999}\n',
            "object": b"[]\n",
            "bytes": b"x" * (prefix.BYTE_LIMIT + 1),
            "records": b"{}\n" * 257,
        }
        path.write_bytes(values[fault])
    with pytest.raises((ValueError, OSError)):
        prefix.read_prefix(sample)


@pytest.mark.parametrize("limit", ["files", "bytes", "records"])
def test_export_refuses_oversized_input(limit):
    files = {"scheduler-1-" + "a" * 32 + ".jsonl": (b"{}\n", [{}])}
    if limit == "files":
        files.update({str(i): (b"{}\n", [{}]) for i in (1, 2)})
    elif limit == "bytes":
        files[next(iter(files))] = (b"x" * (prefix.BYTE_LIMIT + 1), [])
    else:
        files[next(iter(files))] = (b"{}\n", [{}] * 257)
    with pytest.raises(ValueError, match="limits"):
        prefix.export_prefix({}, files)


def test_export_rejects_filename_injection():
    with pytest.raises(ValueError, match="filename"):
        prefix.export_prefix({}, {"foreign\nSMG_PD_PROBE_END": (b"{}\n", [{}])})


def test_failed_setup_can_export_empty_diagnostic(capsys):
    prefix.export_prefix({"error": "startup failed"}, {})
    header = json.loads(capsys.readouterr().out.splitlines()[1])
    assert header["error"] == "startup failed" and header["file_count"] == 0
    assert header["accepted"] is False
