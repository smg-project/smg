"""The extension runs jemalloc with the gateway's options.

The router launched from Python (``smg launch``, ``python -m smg``) runs the
Rust gateway inside ``smg.smg_rs``, whose jemalloc must take the same
``malloc_conf`` as the ``smg`` executable: a background purging thread, dirty
pages returned after ten seconds, muzzy pages at once. Otherwise a traffic
burst's freed pages stay resident for the life of the process. Asked to
through ``_RJEM_MALLOC_CONF``, jemalloc prints its statistics when the process
exits, and the options in effect are their ``opt`` section.
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("smg.smg_rs")

SERVER_MALLOC_CONF = (
    '"background_thread":true',
    '"dirty_decay_ms":10000',
    '"muzzy_decay_ms":0',
)


def jemalloc_options(malloc_conf: str) -> str:
    """The ``opt`` section of jemalloc's JSON statistics for a process that
    loads the extension under ``_RJEM_MALLOC_CONF=<malloc_conf>``."""
    env = dict(os.environ, _RJEM_MALLOC_CONF=malloc_conf)
    completed = subprocess.run(
        [sys.executable, "-c", "import smg.smg_rs"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    stats = completed.stderr
    start = stats.find('"opt":{')
    assert start >= 0, f"no jemalloc statistics from the extension: {stats[:400]!r}"
    return stats[start : stats.index("}", start)]


@pytest.mark.unit
def test_extension_applies_the_server_malloc_conf():
    options = jemalloc_options("stats_print:true,stats_print_opts:J")
    for expected in SERVER_MALLOC_CONF:
        assert expected in options, f"{expected} not in effect: {options}"


@pytest.mark.unit
def test_environment_overrides_the_server_malloc_conf_entry_by_entry():
    options = jemalloc_options(
        "stats_print:true,stats_print_opts:J,background_thread:false,dirty_decay_ms:2000"
    )
    for expected in ('"background_thread":false', '"dirty_decay_ms":2000', '"muzzy_decay_ms":0'):
        assert expected in options, f"{expected} not in effect: {options}"
