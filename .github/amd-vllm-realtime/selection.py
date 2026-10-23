"""Bind all local realtime IDs and unchanged markers before fixtures execute."""

import hashlib
import json

from reuse import require


def signature(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()


def records_for(items):
    records = []
    for item in items:
        require(item.module.__name__.endswith("test_realtime_local"), "cloud realtime selected")
        require(item.module.MODEL == "Qwen/Qwen3-ASR-1.7B", "original ASR model changed")
        records.append(
            {
                "nodeid": item.nodeid,
                "parameters": getattr(getattr(item, "callspec", None), "params", {}),
                "module_model": item.module.MODEL,
                "markers": [
                    {"name": mark.name, "args": mark.args, "kwargs": mark.kwargs}
                    for mark in item.iter_markers()
                ],
            }
        )
    return sorted(records, key=lambda record: record["nodeid"])


def require_selection(records, recipe):
    require(len(records) == recipe["selected_count"], "complete realtime selection changed")
    require({r["nodeid"] for r in records} == set(recipe["cases"]), "realtime IDs changed")
    require(signature(records) == recipe["selection_metadata_sha256"], "realtime metadata changed")


def capture(items, recipe, output):
    records = records_for(items)
    output.write_text(json.dumps(records, indent=2, default=str) + "\n")
    require_selection(records, recipe)
