"""Reviewed e4e2 artifact IO, used by the CPU relay without importing artifacts."""

import hashlib
import io
import os
import stat
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
INPUTS = Path(__file__).with_name("inputs.json")
API = "https://api.github.com/repos/smg-project/smg/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git_output(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT}", *args], cwd=ROOT, text=True
    )


def check_metadata(meta, artifact, inputs):
    producer = meta["workflow_run"]
    require(meta["id"] == artifact["id"] and meta["name"] == artifact["name"], "artifact identity")
    require(not meta["expired"], "artifact expired")
    require(meta["size_in_bytes"] == artifact["bytes"], "artifact archive size")
    require(meta["digest"] == "sha256:" + artifact["sha256"], "artifact archive digest")
    require(producer["id"] == inputs["run_id"], "producer run")
    require(producer["head_sha"] == inputs["producer_sha"], "producer head")


def unpack(data, artifact):
    require(
        len(data) == artifact["bytes"] and digest(data) == artifact["sha256"], "archive bytes/hash"
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len(members) == len(artifact["files"]), "unexpected archive members")
        actual = {}
        for info in members:
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in info.filename
                and not info.is_dir()
                and not stat.S_ISLNK(mode)
                and (stat.S_IFMT(mode) in (0, stat.S_IFREG))
                and not info.flag_bits & 1,
                "unsafe archive member",
            )
            require(
                info.filename not in actual and info.filename in artifact["files"],
                "unknown/duplicate file",
            )
            expected = artifact["files"][info.filename]
            require(info.file_size == expected["bytes"], "payload size")
            payload = archive.read(info)
            require(digest(payload) == expected["sha256"], "payload digest")
            actual[info.filename] = payload
        return actual


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_get(path):
    request = urllib.request.Request(
        API + path,
        headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "Accept": "application/vnd.github+json",
            "User-Agent": "smg-shared-product-relay",
        },
    )
    opener = urllib.request.build_opener(NoRedirect())
    # Retry only opening this read-only API request, never a response-body read.
    # The caller's existing overall deadline still bounds all relay requests.
    for attempt in range(2):
        try:
            return opener.open(request, timeout=30)
        except urllib.error.HTTPError as exc:
            if attempt == 1 or exc.code not in (502, 503, 504):
                raise
            exc.close()
        except urllib.error.URLError as exc:
            if attempt == 1 or not isinstance(exc.reason, TimeoutError):
                raise
        except TimeoutError:
            if attempt == 1:
                raise


def archive_bytes(artifact):
    try:
        with api_get(f"actions/artifacts/{artifact['id']}/zip") as response:
            return response.read(artifact["bytes"] + 1)
    except urllib.error.HTTPError as exc:
        if exc.code != 302:
            raise
        location = exc.headers["Location"]
    url = urllib.parse.urlsplit(location)
    require(
        url.scheme == "https" and url.hostname and not url.username and not url.password,
        "archive redirect",
    )
    # The GitHub bearer token is never forwarded to the signed blob URL.
    with urllib.request.urlopen(location, timeout=60) as response:
        return response.read(artifact["bytes"] + 1)
