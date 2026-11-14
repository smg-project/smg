"""Immutable LoRA adapters preloaded by SGLang's resolved ServerArgs.

The tokenized scheduler wire takes internal IDs, while the public API takes
adapter names. No adapter is loaded or unloaded on this request path.
"""

from __future__ import annotations

from typing import Any


class PreloadedLoras:
    def __init__(self, server_args: Any) -> None:
        # ServerArgs preserves CLI declarations and keeps parsed LoRARefs in
        # its resolved view. Resolve before the Rust launcher's facts snapshot
        # too, so that snapshot and the scheduler inherit the very same IDs.
        if callable(getattr(server_args, "resolved_dict", None)):
            server_args.resolve_once()
            resolved = server_args.resolved_dict()
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
