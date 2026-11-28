"""Immutable LoRA adapters preloaded by SGLang's resolved ServerArgs.

The tokenized scheduler wire takes internal IDs, while the public API takes
adapter names. No adapter is loaded or unloaded on this request path.
"""

from __future__ import annotations

import copy
from typing import Any


class PreloadedLoras:
    def __init__(self, server_args: Any) -> None:
        # ServerArgs preserves CLI declarations separately from resolved values.
        if callable(getattr(server_args, "resolved_dict", None)):
            server_args.resolve_once()
            resolved = server_args.resolved_dict()
            if resolved.get("lora_paths") and callable(
                getattr(server_args, "check_server_args", None)
            ):
                from sglang.srt.server_args import ServerArgs

                if isinstance(server_args, ServerArgs):
                    from sglang.srt.arg_groups.lora_hook import check_lora_server_args
                    from sglang.srt.arg_groups.overrides import resolution_result
                    from sglang.srt.lora.lora_registry import LoRARef

                    refs = resolution_result(server_args, "lora_paths")
                    if isinstance(refs, dict) or any(not isinstance(ref, LoRARef) for ref in refs):
                        # LoRA normalization happens during engine validation,
                        # after the Rust launcher's facts snapshot. Run the native
                        # hook on a copy: its deterministic IDs match the engine,
                        # which must still validate its own declarations once.
                        snapshot_args = copy.deepcopy(server_args)
                        check_lora_server_args(snapshot_args)
                        resolved = snapshot_args.resolved_dict()
        else:
            resolved = {
                "enable_lora": getattr(server_args, "enable_lora", False),
                "lora_paths": getattr(server_args, "lora_paths", None),
            }
        self.enabled = bool(resolved.get("enable_lora", False))
        self.adapters: list[dict[str, Any]] = []
        self._ids: dict[str, str] = {}
        for ref in resolved.get("lora_paths") or ():
            adapter: dict[str, Any] = {}
            for field in ("lora_id", "lora_name", "lora_path"):
                value = ref.get(field) if isinstance(ref, dict) else getattr(ref, field, None)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Preloaded LoRA {field} must be a non-empty string")
                adapter[field] = value
            pinned = ref.get("pinned") if isinstance(ref, dict) else getattr(ref, "pinned", None)
            if pinned is not None and not isinstance(pinned, bool):
                raise ValueError("Preloaded LoRA pinned must be a boolean")
            adapter["pinned"] = bool(pinned)
            for alias in (adapter["lora_name"], adapter["lora_id"]):
                if alias in self._ids and self._ids[alias] != adapter["lora_id"]:
                    raise ValueError(f"Ambiguous preloaded LoRA name or ID: {alias}")
                self._ids[alias] = adapter["lora_id"]
            self.adapters.append(adapter)
        if self.adapters and not self.enabled:
            raise ValueError("Preloaded LoRA adapters require --enable-lora")

    def resolve(self, selector: str) -> str | None:
        if not selector:
            return None
        if not self.enabled:
            raise ValueError("LoRA is not enabled; launch the worker with --enable-lora")
        if selector not in self._ids:
            raise ValueError(f"Unknown preloaded LoRA adapter: {selector}")
        return self._ids[selector]
