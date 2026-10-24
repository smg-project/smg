"""Apply one exact job-private SG2 startup fail-fast patch, without engine import."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def digest(data):
    return hashlib.sha256(data).hexdigest()


def spec():
    return json.loads(Path(__file__).with_name("registration-overlay.json").read_text())


def apply(directory):
    marker = os.environ.get("E2E_JOB_MARKER", "")
    if not re.fullmatch(r"e2e-[0-9]+-[0-9]+-amd-mi300x-[12]", marker):
        raise RuntimeError("SG2 overlay requires the actual same-job marker")
    contract = spec()
    target = Path(contract["path"])
    lock = Path(directory) / "sglang-registration-overlay.lock"
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        status = target.lstat()
        if not stat.S_ISREG(status.st_mode) or status.st_size > 256 * 1024:
            raise RuntimeError("SG2 overlay target is not the bounded regular actual source")
        data = target.read_bytes()
        actual = digest(data)
        if actual == contract["before_sha256"]:
            old, new = contract["before"].encode(), contract["after"].encode()
            if data.count(old) != 1:
                raise RuntimeError("SG2 overlay source anchor differs")
            data = data.replace(old, new)
            if digest(data) != contract["after_sha256"]:
                raise RuntimeError("SG2 overlay patched source differs")
            temporary = target.with_name(target.name + f".sg2-{os.getpid()}")
            created = False
            try:
                with temporary.open("xb") as stream:
                    created = True
                    os.fchmod(stream.fileno(), stat.S_IMODE(status.st_mode))
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
        elif actual != contract["after_sha256"]:
            raise RuntimeError("SG2 overlay refuses unknown actual source")
        if digest(target.read_bytes()) != contract["after_sha256"]:
            raise RuntimeError("SG2 overlay post-write identity differs")
    witness = {"job_marker": marker, "path": str(target), "before_sha256": contract["before_sha256"],
               "after_sha256": contract["after_sha256"], "observed_before_sha256": actual,
               "applier_module": __file__, "applier_sha256": digest(Path(__file__).read_bytes()),
               "runtime_origin_verified": False, "RDMA_or_transfer_accepted": False}
    (Path(directory) / "sglang-registration-overlay-applied.json").write_text(json.dumps(witness, indent=2) + "\n")


def verify(directory):
    contract = spec()
    target = Path(contract["path"])
    if (not target.is_file() or target.is_symlink() or target.stat().st_size > 256 * 1024
            or digest(target.read_bytes()) != contract["after_sha256"]):
        raise RuntimeError("SG2 registration source was replaced after patch")
    (Path(directory) / "sglang-registration-overlay-final.json").write_text(
        json.dumps({"path": str(target), "sha256": contract["after_sha256"],
                    "runtime_origin_verified": False, "RDMA_or_transfer_accepted": False}, indent=2) + "\n")
