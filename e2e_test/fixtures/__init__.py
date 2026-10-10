"""Fixtures for E2E tests.

This package contains modular pytest fixtures split by responsibility:
- hooks.py: Pytest collection hooks and marker registration
- setup_backend.py: Backend setup fixtures (class/function-scoped)
- markers.py: Helper utilities for marker extraction

"""

# Pytest hooks (imported by conftest.py via pytest_plugins)
from .hooks import (
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_makereport,
    pytest_runtest_setup,
    pytest_sessionfinish,
)

# Marker helpers
from .markers import get_marker_kwargs, get_marker_value

# Fixtures (imported by conftest.py)
from .setup_backend import backend_router, setup_backend

__all__ = [
    # Hooks
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_runtest_makereport",
    "pytest_runtest_setup",
    "pytest_sessionfinish",
    # Backend fixtures
    "setup_backend",
    "backend_router",
    # Marker helpers
    "get_marker_value",
    "get_marker_kwargs",
]
