"""Offline artifact rejection controls; no API requests or binary execution."""

import hashlib
import io
import json
import zipfile
from copy import deepcopy

import pytest
import reuse_foundation as reuse


def artifact():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("payload", b"known")
    data = stream.getvalue()
    return data, {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "files": {"payload": {"bytes": 5, "sha256": hashlib.sha256(b"known").hexdigest()}},
    }


@pytest.mark.parametrize("kind", ["archive", "payload", "unknown-member"])
def test_artifact_corruption_rejected(kind):
    data, expected = artifact()
    if kind == "archive":
        data = data[:-1] + bytes([data[-1] ^ 1])
    elif kind == "payload":
        expected["files"]["payload"]["sha256"] = "0" * 64
    else:
        expected["files"] = {"different": expected["files"]["payload"]}
    with pytest.raises(ValueError):
        reuse.unpack(data, expected)


def test_valid_artifact_and_expired_or_wrong_identity():
    data, expected = artifact()
    assert reuse.unpack(data, expected) == {"payload": b"known"}
    inputs = json.loads(reuse.INPUTS.read_text())
    expected = inputs["artifacts"][0]
    meta = {
        "id": expected["id"],
        "name": expected["name"],
        "expired": False,
        "size_in_bytes": expected["bytes"],
        "digest": "sha256:" + expected["sha256"],
        "workflow_run": {"id": inputs["run_id"], "head_sha": inputs["producer_sha"]},
    }
    reuse.check_metadata(meta, expected, inputs)
    for field, value in [("expired", True), ("digest", "sha256:" + "0" * 64), ("id", 0)]:
        bad = deepcopy(meta)
        bad[field] = value
        with pytest.raises(ValueError):
            reuse.check_metadata(bad, expected, inputs)
    meta["workflow_run"]["head_sha"] = "0" * 40
    with pytest.raises(ValueError):
        reuse.check_metadata(meta, expected, inputs)
