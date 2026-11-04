"""Exercise the real Oracle download transaction with local archives and fake I/O.

Only the actual cache/download block is extracted from cmd_setup_oracle_client;
apt, shared services and the user's home are never touched. Real unzip validates
and extracts local ZIP fixtures. wget/timeout/sleep are bounded fault seams.
"""

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ci_agentic_svc_deps.sh"
CLIENT = "instantclient_23_26"
ARCHIVE = "instantclient-basic-linux.x64-23.26.1.0.0.zip"
URL = f"https://download.oracle.com/otn_software/linux/instantclient/2326100/{ARCHIVE}"
LIBRARIES = ("libclntsh.so.23.1", "libclntshcore.so.23.1", "libociei.so", "libnnz.so")
ELF = b"\x7fELFcomplete local archive fixture"

FAKE = r"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
log = Path(os.environ["CI_ORACLE_FIXTURE_LOG"])
previous = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
with log.open("a") as stream:
    stream.write(json.dumps({"tool": name, "args": args}) + "\n")
if name == "wget":
    attempt = sum(item["tool"] == "wget" for item in previous)
    exits = json.loads(os.environ["CI_ORACLE_FIXTURE_EXITS"])
    status = exits[min(attempt, len(exits) - 1)]
    target = Path(args[args.index("-O") + 1]) if "-O" in args else Path(args[-1].rsplit("/", 1)[-1])
    if status:
        target.write_bytes(b"partial ZIP after failed TLS connection")
    else:
        shutil.copyfile(os.environ["CI_ORACLE_FIXTURE_ARCHIVE"], target)
    raise SystemExit(status)
if name == "sleep":
    raise SystemExit(0)
if name == "timeout":
    if os.environ.get("CI_ORACLE_FIXTURE_TIMEOUT") == "1":
        raise SystemExit(124)
    assert args[:2] == ["--kill-after=5s", "120s"], args
    os.execvp(args[2], args[2:])
if name == "unzip":
    validates = any(arg.startswith("-t") for arg in args)
    if os.environ.get("CI_ORACLE_FIXTURE_PARTIAL_EXTRACT") == "1" and not validates:
        destination = Path(args[args.index("-d") + 1]) if "-d" in args else Path.cwd()
        partial = destination / "instantclient_23_26"
        partial.mkdir(parents=True)
        (partial / "incomplete").write_text("injected extraction failure")
        raise SystemExit(31)
    status = subprocess.call([os.environ["CI_ORACLE_FIXTURE_REAL_UNZIP"], *args])
    if status == 0 and not validates and os.environ.get("CI_ORACLE_FIXTURE_PUBLISH_RACE") == "1":
        destination = Path(args[args.index("-d") + 1]) if "-d" in args else Path.cwd()
        extracted = destination / "instantclient_23_26"
        winner = Path(os.environ["CI_ORACLE_FIXTURE_CACHE"]) / "instantclient_23_26"
        if extracted.resolve() != winner.resolve():
            shutil.copytree(extracted, winner, symlinks=True)
        (winner / "winner").write_text("another completed installer")
    raise SystemExit(status)
raise AssertionError(name)
"""


@pytest.fixture
def download(tmp_path):
    setup = SCRIPT.read_text().split("cmd_setup_oracle_client() {", 1)[1]
    start = setup.index('    if [ ! -d "$INSTANT_CLIENT_DIR/instantclient_23_26" ]; then')
    end = setup.index('\n    if [ -n "$LIBAIO_PATH" ]; then', start)
    actual_block = setup[start:end]
    zip_assignments = [
        line.strip()
        for line in setup[:start].splitlines()
        if line.strip().startswith("INSTANT_CLIENT_ZIP=")
    ]
    assert len(zip_assignments) == 1, "Expected one production INSTANT_CLIENT_ZIP assignment"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/ci_retry.sh", scripts / "ci_retry.sh")
    driver = scripts / "download.sh"
    driver.write_text(
        '#!/bin/bash\nset -euo pipefail\nINSTANT_CLIENT_DIR="$1"\n'
        f"CI_AGENTIC_SVC_DEPS_LIB_ONLY=1 source {shlex.quote(str(SCRIPT))}\n"
        f"{zip_assignments[0]}\n"
        "download_fixture() {\n" + actual_block + "\n}\ndownload_fixture\n"
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("wget", "timeout", "sleep", "unzip"):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n" + FAKE)
        path.chmod(0o755)
    payload = tmp_path / "payload.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        for library in LIBRARIES:
            archive.writestr(f"{CLIENT}/{library}", ELF)
        link = zipfile.ZipInfo(f"{CLIENT}/libclntsh.so")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "libclntsh.so.23.1")
    calls = tmp_path / "calls.jsonl"
    cache = tmp_path / "cache"

    def execute(
        exits=(0,),
        *,
        corrupt=False,
        partial_extract=False,
        forced_timeout=False,
        wrong_folder=False,
        publish_race=False,
    ):
        if corrupt:
            payload.write_bytes(b"not a valid ZIP")
        if wrong_folder:
            with zipfile.ZipFile(payload, "w") as archive:
                archive.writestr("wrong-client/file", "unexpected directory")
        result = subprocess.run(
            ["bash", str(driver), str(cache)],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": f"{tools}:{os.environ['PATH']}",
                "CI_ORACLE_FIXTURE_LOG": str(calls),
                "CI_ORACLE_FIXTURE_ARCHIVE": str(payload),
                "CI_ORACLE_FIXTURE_EXITS": json.dumps(exits),
                "CI_ORACLE_FIXTURE_REAL_UNZIP": shutil.which("unzip"),
                "CI_ORACLE_FIXTURE_PARTIAL_EXTRACT": "1" if partial_extract else "0",
                "CI_ORACLE_FIXTURE_TIMEOUT": "1" if forced_timeout else "0",
                "CI_ORACLE_FIXTURE_PUBLISH_RACE": "1" if publish_race else "0",
                "CI_ORACLE_FIXTURE_CACHE": str(cache),
            },
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        records = (
            [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        )
        return result, records, cache

    execute.cache = cache
    return execute


def selected(records, tool):
    return [call["args"] for call in records if call["tool"] == tool]


def safe_download_arguments(records):
    for args in selected(records, "wget"):
        assert args[-1] == URL
        assert not any(arg.startswith("--no-check") for arg in args)
        assert "--tries=1" in args and "--timeout=30" in args
        assert "-O" in args
    for args in selected(records, "timeout"):
        assert args[:2] == ["--kill-after=5s", "120s"]


def empty_cache(cache):
    assert not (cache / CLIENT).exists(), "failed transaction published a client cache"
    assert not cache.exists() or list(cache.iterdir()) == [], "failed download left temporary files"


def test_tls_failure_then_success_retries_and_overwrites_partial_archive(download):
    result, calls, cache = download((4, 0))
    assert result.returncode == 0, result.stdout + result.stderr
    gets = selected(calls, "wget")
    assert len(gets) == 2
    assert gets[0][gets[0].index("-O") + 1] == gets[1][gets[1].index("-O") + 1]
    assert selected(calls, "sleep") == [["5"]]
    assert (cache / CLIENT / "libclntsh.so").read_bytes() == ELF
    assert list(cache.iterdir()) == [cache / CLIENT]
    safe_download_arguments(calls)


def test_exhausted_network_failure_preserves_failure_and_cleans_stage(download):
    result, calls, cache = download((4, 4, 4))
    assert result.returncode == 4
    assert len(selected(calls, "wget")) == 3
    assert selected(calls, "sleep") == [["5"], ["10"]]
    empty_cache(cache)
    safe_download_arguments(calls)


def test_invalid_archive_is_not_extracted_or_published(download):
    result, calls, cache = download(corrupt=True)
    assert result.returncode != 0
    assert len(selected(calls, "wget")) == 1
    unzip_calls = selected(calls, "unzip")
    assert len(unzip_calls) == 1
    assert any(arg.startswith("-t") for arg in unzip_calls[0])
    empty_cache(cache)


def test_partial_extraction_failure_never_becomes_cached_client(download):
    result, _, cache = download(partial_extract=True)
    assert result.returncode == 31
    empty_cache(cache)


def test_complete_archive_is_published_once_without_leftover_zip(download):
    result, calls, cache = download()
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(selected(calls, "wget")) == 1
    assert selected(calls, "sleep") == []
    assert list(cache.iterdir()) == [cache / CLIENT]
    assert (cache / CLIENT / "libclntsh.so").read_bytes() == ELF
    safe_download_arguments(calls)


def test_existing_client_cache_is_preserved_without_download(download):
    existing = download.cache / CLIENT
    existing.mkdir(parents=True)
    for library in (*LIBRARIES, "libclntsh.so"):
        (existing / library).write_bytes(ELF)
    marker = existing / "existing"
    marker.write_text("preserve existing cache")
    inode = marker.stat().st_ino
    result, calls, cache = download()
    assert result.returncode == 0
    assert calls == []
    assert (cache / CLIENT / "existing").stat().st_ino == inode


def test_certificate_failure_remains_failure_without_disabling_verification(download):
    result, calls, cache = download((5, 5, 5))
    assert result.returncode == 5
    assert len(selected(calls, "wget")) == 3
    empty_cache(cache)
    safe_download_arguments(calls)


def test_attempts_have_finite_timeout_and_exhausted_timeout_stays_failure(download):
    result, calls, cache = download(forced_timeout=True)
    assert result.returncode == 124
    assert len(selected(calls, "timeout")) == 3
    assert selected(calls, "sleep") == [["5"], ["10"]]
    assert selected(calls, "wget") == []
    empty_cache(cache)
    safe_download_arguments(calls)


def test_valid_zip_with_wrong_client_directory_is_not_published(download):
    result, _, cache = download(wrong_folder=True)
    assert result.returncode != 0
    assert f"Oracle archive {ARCHIVE} has no {CLIENT}/ root" in result.stderr
    empty_cache(cache)


def test_failed_transaction_cleans_only_its_own_stage(download):
    other_stage = download.cache / ".oracle-client.another-live-installer"
    other_stage.mkdir(parents=True)
    marker = other_stage / "owned"
    marker.write_text("another transaction must survive")
    result, _, cache = download((4, 4, 4))
    assert result.returncode == 4
    assert list(cache.iterdir()) == [other_stage]
    assert marker.read_text() == "another transaction must survive"


def test_nonempty_cache_published_by_another_installer_is_not_overwritten(download):
    result, _, cache = download(publish_race=True)
    assert result.returncode != 0
    assert list(cache.iterdir()) == [cache / CLIENT]
    assert (cache / CLIENT / "winner").read_text() == "another completed installer"
    assert (cache / CLIENT / "libclntsh.so").read_bytes() == ELF
    assert not (cache / CLIENT / CLIENT).exists()
