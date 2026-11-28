"""A unit test of the harness runs with no declared startup profile.

CI runs the unit tests of the harness with ``--noconftest``, so nothing records
the pytest selection a declared startup profile needs, and worker setup refuses
to start under one. A profile left in the shell by a real run would then fail
every unit test that drives the setup. A test file that drives it imports the
fixture below: an imported autouse fixture binds the tests of the importing file.
"""

from __future__ import annotations

import pytest
from infra.constants import ENV_STARTUP_PROFILE


@pytest.fixture(autouse=True)
def no_startup_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take a startup profile declared in the shell out of the test's environment."""
    monkeypatch.delenv(ENV_STARTUP_PROFILE, raising=False)
