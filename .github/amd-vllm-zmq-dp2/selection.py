"""Bind the complete original collection before any inference fixture executes."""

import hashlib
import json

from reuse import require


def signature(records):
    normalized = []
    for record in records:
        item = {**record, "declared_model_spec": dict(record["declared_model_spec"])}
        # Cache location can differ; the independent cache gate checks the real resolved path.
        item["declared_model_spec"]["model"] = item["explicit_model_marker"]
        normalized.append(item)
    normalized.sort(key=lambda item: item["nodeid"])
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode()).hexdigest()


def require_selection(records, recipe):
    require(len(records) == recipe["selected_count"], "complete chat selection count changed")
    require({r["nodeid"] for r in records} == set(recipe["cases"]), "original chat IDs changed")
    require(
        signature(records) == recipe["selection_metadata_sha256"],
        "case params/markers/model changed",
    )


def capture(items, recipe, output):
    from fixtures.markers import model_id_for_engine, resolve_class_marker
    from infra.model_specs import get_model_spec

    records = []
    for item in items:
        model = model_id_for_engine(resolve_class_marker(item, "model"), "vllm")
        records.append(
            {
                "nodeid": item.nodeid,
                "parameters": getattr(getattr(item, "callspec", None), "params", {}),
                "explicit_model_marker": model,
                "declared_model_spec": get_model_spec(model),
                "markers": [
                    {"name": mark.name, "args": mark.args, "kwargs": mark.kwargs}
                    for mark in item.iter_markers()
                ],
            }
        )
    output.write_text(json.dumps(records, indent=2, default=str) + "\n")
    require_selection(records, recipe)
