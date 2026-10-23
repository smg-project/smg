"""Additional source pins for the opt-in local allocator observer."""

import os

SOURCE_SHA256 = {
    "v1/core/kv_cache_manager.py": "bd9bf46a2101f9368d4941867415a239d18d95e0b956f82f885ce25e10e69f3b",
    "v1/core/block_pool.py": "05c73052934a9d4664ccd5df562ce71011278d9a511932da102ef2efff00e934",
    "v1/core/kv_cache_utils.py": "1b20c836aece4982dd87f7e06e14f5ae07a1fc2636101074cfad11584b1a0dcf",
    "v1/core/kv_cache_coordinator.py": "6a1e00227b981a7bb6108a42db0e446c716f789274f40c63eb9a6ce4d699ddcb",
    "v1/core/single_type_kv_cache_manager.py": "3a3cb20bd6aa1e0ad46471fc9e5e01ee5797ab4bb759e00ef5fb309d2830db13",
}


def enabled():
    value = os.environ.get("SMG_PD_TRACE_LOCAL_KV", "0")
    if value not in ("0", "1"):
        raise ValueError("unknown local KV observation opt-in")
    return value == "1"
