"""In-process SMG tokenizer. No gateway, GPU, or HTTP service is required."""

import json

from smg_tokenizer._native import TOKENIZERS_V1
from smg_tokenizer._native import Tokenizer as _Tokenizer

__version__ = "0.1.0"
build_features = "tokenizers-v1" if TOKENIZERS_V1 else "default"


class Tokenizer:
    """Load a local, pinned checkpoint directory using SMG's tokenizer factory."""

    def __init__(self, path, *, chat_template_path=None):
        self._inner = _Tokenizer(str(path), chat_template_path)

    def encode(self, text, *, add_special_tokens=False):
        return self._inner.encode(text, add_special_tokens)

    @property
    def encode_path(self):
        return self._inner.encode_path

    @property
    def raw_tool_call_arguments(self):
        return self._inner.raw_tool_call_arguments

    def render(self, messages, *, tools=None, template_kwargs=None, now=None):
        """Return text, exact token IDs, and optional UTF-8 byte offsets.

        This entry point renders a complete conversation without a generation
        header. It never flattens and re-encodes a native renderer's segments.
        Offsets are absent if the exact encoding cannot be mapped to flat text.
        """
        text, ids, offsets, segmentwise = self._inner.render_json(
            json.dumps(
                {
                    "messages": messages,
                    "tools": tools,
                    "template_kwargs": template_kwargs or {},
                    "now": now,
                },
                ensure_ascii=False,
            )
        )
        return {"text": text, "input_ids": ids, "offsets": offsets, "segmentwise": segmentwise}
