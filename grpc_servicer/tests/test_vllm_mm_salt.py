"""Unit tests for the multimodal identity cache salt (engine-free, no vLLM required).

Run with: pytest grpc_servicer/tests/test_vllm_mm_salt.py
"""

import importlib.util
from pathlib import Path

# Import the module directly to avoid pulling vllm via the package __init__
_MODULE_PATH = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "mm_salt.py"
_spec = importlib.util.spec_from_file_location("mm_salt", _MODULE_PATH)
mm_salt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm_salt)


def test_empty_hashes_produce_no_salt():
    assert mm_salt.mm_identity_cache_salt([]) is None


def test_salt_is_deterministic_per_content():
    salt = mm_salt.mm_identity_cache_salt(["h1", "h2"])
    assert salt == "mm:h1,h2"
    assert salt == mm_salt.mm_identity_cache_salt(["h1", "h2"])


def test_different_images_get_different_salts():
    assert mm_salt.mm_identity_cache_salt(["dog"]) != mm_salt.mm_identity_cache_salt(["passport"])


def test_salt_is_order_sensitive():
    # Same images in a different order occupy different placeholder positions.
    assert mm_salt.mm_identity_cache_salt(["h1", "h2"]) != mm_salt.mm_identity_cache_salt(
        ["h2", "h1"]
    )
