"""Adapt SGLang module moves and scheduler layouts at the engine boundary."""

import importlib
from collections.abc import Callable
from typing import Any


def get_tokenizer(*args: Any, **kwargs: Any) -> Any:
    """Load through the engine's tokenizer factory on either module layout."""
    try:
        module = importlib.import_module("sglang.srt.hf_transformers.tokenizer")
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "sglang.srt.hf_transformers",
            "sglang.srt.hf_transformers.tokenizer",
        }:
            raise
        module = importlib.import_module("sglang.srt.utils.hf_transformers_utils")
    return module.get_tokenizer(*args, **kwargs)


def get_disaggregation_types() -> tuple[Any, str]:
    """Return the engine's mode enum and fake host without eager engine imports."""
    try:
        module = importlib.import_module("sglang.srt.disaggregation.mode")
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.srt.disaggregation.mode":
            raise
        module = importlib.import_module("sglang.srt.disaggregation.utils")
    return module.DisaggregationMode, module.FAKE_BOOTSTRAP_HOST


def get_multimodal_types() -> tuple[Any, Any, Any]:
    """Return the engine's modality, data item and processor output types."""
    try:
        module = importlib.import_module("sglang.srt.managers.req_types")
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.srt.managers.req_types":
            raise
        module = importlib.import_module("sglang.srt.managers.schedule_batch")
    return module.Modality, module.MultimodalDataItem, module.MultimodalProcessorOutput


class SchedulerOutputAdapter:
    """Normalize one scheduler socket's outputs without changing the wire schema.

    Select token/usage layouts independently on the first batch: upstream changed
    them in separate commits. Only sparse usage needs retained request state.
    """

    def __init__(self, *, allow_empty_rows: bool = False) -> None:
        """Own one stream; only the Python consumer permits absent legacy rows."""
        self._allow_empty_rows = allow_empty_rows
        self._rows: Callable[[Any], list[list[int]]] | None = None
        self._usage: Callable[[Any], tuple[list[int], list[int]]] | None = None
        self._counts: dict[str, tuple[int, int]] = {}
        self._has_dense_usage = False

    def read(self, out: Any) -> tuple[list[list[int]], list[int], list[int]]:
        """Return token rows and cumulative counts without mutating engine output."""
        if self._rows is None:
            # msgspec structs in SGLang; vars also supports dataclass/test outputs.
            fields = getattr(type(out), "__struct_fields__", None)
            if fields is None:
                fields = vars(out)
            if "output_ids_flat" in fields and "output_ids_offsets" in fields:
                self._rows = self._flat_rows
            elif "output_ids" in fields:
                self._rows = (
                    self._legacy_rows_or_empty if self._allow_empty_rows else self._legacy_rows
                )
            else:
                raise ValueError("unsupported BatchTokenIDOutput token layout")
        rows = self._rows(out)
        prompts, cached = self.read_usage(out)
        return rows, prompts, cached

    def read_usage(self, out: Any) -> tuple[list[int], list[int]]:
        """Read generation or embedding usage, retaining only sparse updates."""
        if self._usage is None:
            fields = getattr(type(out), "__struct_fields__", None)
            if fields is None:
                fields = vars(out)
            self._has_dense_usage = "prompt_tokens" in fields and "cached_tokens" in fields
            self._usage = self._sparse_usage if "request_stats" in fields else self._dense_usage
        return self._usage(out)

    def forget(self, rid: str) -> None:
        """Release retained counts on cancellation or terminal output."""
        self._counts.pop(rid, None)

    @staticmethod
    def _legacy_rows(out: Any) -> list[list[int]]:
        """Reuse list rows and box legacy array rows once for MessagePack."""
        rows = out.output_ids
        if rows is None or len(rows) != len(out.rids):
            raise ValueError("output_ids row count does not match rids")
        # Older engines use array('q') rows; lists need no second copy.
        return [row if isinstance(row, list) else list(row) for row in rows]

    @staticmethod
    def _legacy_rows_or_empty(out: Any) -> list[list[int]]:
        """Preserve the Python consumer's empty-token fallback on older engines."""
        if not out.output_ids:
            return [[] for _ in out.rids]
        return SchedulerOutputAdapter._legacy_rows(out)

    @staticmethod
    def _flat_rows(out: Any) -> list[list[int]]:
        """Validate offsets and box the flat array once before slicing rows."""
        flat, offsets = out.output_ids_flat, out.output_ids_offsets
        if (
            flat is None
            or offsets is None
            or len(offsets) != len(out.rids) + 1
            or offsets[0] != 0
            or offsets[-1] != len(flat)
        ):
            raise ValueError("invalid BatchTokenIDOutput flat token offsets")
        # Box the flat array once, avoiding an intermediate array per row.
        tokens = flat if isinstance(flat, list) else list(flat)
        rows = []
        for start, end in zip(offsets, offsets[1:]):
            if start < 0 or end < start:
                raise ValueError("invalid BatchTokenIDOutput flat token offsets")
            rows.append(tokens[start:end])
        return rows

    @staticmethod
    def _dense_usage(out: Any) -> tuple[list[int], list[int]]:
        """Reuse complete legacy columns without creating request state."""

        def column(values: list[int] | None) -> list[int]:
            """Pad absent counts to the batch width."""
            if values is not None and len(values) == len(out.rids):
                return values
            values = list(values or [])
            values.extend(0 for _ in range(len(out.rids) - len(values)))
            return values

        return column(out.prompt_tokens), column(out.cached_tokens)

    def _sparse_usage(self, out: Any) -> tuple[list[int], list[int]]:
        """Carry first/retracted counts forward until each request finishes."""
        prompts, cached = [], []
        stats = out.request_stats or []
        # Modern SGLang also retains dense headers for its native Rust sink.
        dense_prompts = (out.prompt_tokens or []) if self._has_dense_usage else []
        dense_cached = (out.cached_tokens or []) if self._has_dense_usage else []
        reasons = out.finished_reasons or []
        for i, rid in enumerate(out.rids):
            current = stats[i] if i < len(stats) else None
            if current is not None:
                counts = (current.prompt_tokens, current.cached_tokens)
                self._counts[rid] = counts
            elif i < len(dense_prompts):
                counts = (dense_prompts[i], dense_cached[i] if i < len(dense_cached) else 0)
                self._counts[rid] = counts
            else:
                counts = self._counts.get(rid, (0, 0))
            prompts.append(counts[0])
            cached.append(counts[1])
            if i < len(reasons) and reasons[i] is not None:
                self.forget(rid)
        return prompts, cached
