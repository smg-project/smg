"""SMG's tokenizer for Python: the token ids SMG serving produces.

:class:`Tokenizer` loads a checkpoint's tokenizer with the Rust code the
gateway serves with (``llm-tokenizer``), so a training or evaluation pipeline
can encode text and decode ids exactly as SMG does, without a running
gateway.

Parity holds against a gateway built from the same smg commit
(:attr:`Tokenizer.smg_build`; compare the gateway's ``--version-verbose``)
that loads the same files: pass what the gateway loads (its tokenizer path,
else its model path; ``config.json`` and ``generation_config.json`` beside
the tokenizer files pick the stop ids).

Short calls (an encode of less than 16 KiB, a decode of fewer than 4096 ids)
run on the calling thread with the GIL held, which costs less than releasing
it; encode many texts with the batch methods. The batch methods, longer texts
and loading release the GIL and run on a Rust thread pool of
``SMG_TOKENIZER_THREADS`` threads (one when ``TOKENIZERS_PARALLELISM`` is off,
rayon's default otherwise: ``RAYON_NUM_THREADS``, else the CPU count), one per
process: a process forked after the parent used the tokenizer (torch
``DataLoader`` workers, ``datasets.map(num_proc=...)``) starts its own.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from smg.smg_rs import Tokenizer as _Tokenizer

if TYPE_CHECKING:
    from typing import Protocol

    class _ToList(Protocol):
        def tolist(self) -> Any: ...

    # Annotations only (the wheel supports Python 3.9, where `X | Y` does not
    # evaluate): token ids, as a sequence or an array or tensor of them.
    TokenIds = Sequence[int] | _ToList

__all__ = ["Tokenizer"]


def _ids(token_ids: Any) -> Any:
    """Token ids as a list when they come as a numpy array or a torch tensor
    (which the Rust side does not read as a sequence); as given otherwise."""
    tolist = getattr(token_ids, "tolist", None)
    return tolist() if callable(tolist) else token_ids


class Tokenizer:
    """A checkpoint's tokenizer, loaded, encoding and decoding as SMG serving does.

    ``source`` is a checkpoint directory, a tokenizer file, or a Hugging Face
    model id, downloaded from the Hub as the gateway downloads it (honouring
    ``HF_TOKEN``, ``HF_HOME`` and ``HF_ENDPOINT``; the revision is always
    ``main``: for a pinned revision or an offline host, download with
    ``huggingface_hub.snapshot_download`` and pass the directory).
    """

    smg_build: ClassVar[str] = _Tokenizer.smg_build
    """The smg release and commit this binding was built from: the token ids
    are those of a gateway built from the same commit."""

    def __init__(self, source: str | os.PathLike[str]) -> None:
        self._inner = _Tokenizer(os.fspath(source))
        # Read once: they never change, and each read from Rust sorts them anew.
        special: dict[str, Any] = self._inner.special_tokens()
        self._additional_special_tokens: list[str] = special.pop("additional_special_tokens")
        self._special_tokens: dict[str, str | None] = special

    @classmethod
    def from_pretrained(cls, source: str | os.PathLike[str]) -> Tokenizer:
        """Load a tokenizer; the same as ``Tokenizer(source)``."""
        return cls(source)

    @property
    def source(self) -> str:
        """Where the tokenizer was loaded from: the absolute path of a local
        checkpoint or of the Hub snapshot a model id downloaded to, or the
        OpenAI model name ``tiktoken`` loaded."""
        source: str = self._inner.source
        return source

    @property
    def vocab_size(self) -> int:
        """The base vocabulary's size; for a Hugging Face tokenizer, without
        the added tokens, as ``transformers`` reports ``vocab_size``."""
        size: int = self._inner.vocab_size
        return size

    def __len__(self) -> int:
        """The vocabulary's size with the added tokens, as ``len()`` of a
        ``transformers`` tokenizer."""
        return len(self._inner)

    @property
    def special_tokens(self) -> dict[str, Any]:
        """The special tokens by role (``bos_token``, ``eos_token``,
        ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``,
        ``mask_token``; ``None`` when the checkpoint has none) and
        ``additional_special_tokens``: every added token marked special,
        sorted by id."""
        return {
            **self._special_tokens,
            "additional_special_tokens": list(self._additional_special_tokens),
        }

    @property
    def bos_token(self) -> str | None:
        return self._special("bos_token")

    @property
    def eos_token(self) -> str | None:
        return self._special("eos_token")

    @property
    def pad_token(self) -> str | None:
        return self._special("pad_token")

    @property
    def unk_token(self) -> str | None:
        return self._special("unk_token")

    @property
    def bos_token_id(self) -> int | None:
        return self._special_id("bos_token")

    @property
    def eos_token_id(self) -> int | None:
        return self._special_id("eos_token")

    @property
    def pad_token_id(self) -> int | None:
        return self._special_id("pad_token")

    @property
    def unk_token_id(self) -> int | None:
        return self._special_id("unk_token")

    @property
    def eos_token_ids(self) -> list[int]:
        """The ids SMG stops generation at: ``eos_token_id`` of the
        checkpoint's ``config.json`` and ``generation_config.json``, with the
        id of the tokenizer's own ``eos_token``."""
        ids: list[int] = self._inner.eos_token_ids
        return ids

    def _special(self, role: str) -> str | None:
        return self._special_tokens[role]

    def _special_id(self, role: str) -> int | None:
        token = self._special(role)
        return None if token is None else self.token_to_id(token)

    def token_to_id(self, token: str) -> int | None:
        """The id of a token (in the vocabulary's own spelling, such as
        ``Ġthe`` for byte-level vocabularies), or ``None``."""
        token_id: int | None = self._inner.token_to_id(token)
        return token_id

    def id_to_token(self, token_id: int) -> str | None:
        """The vocabulary's spelling of an id, or ``None``."""
        token: str | None = self._inner.id_to_token(token_id)
        return token

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Encode text.

        ``add_special_tokens`` adds what the tokenizer's post-processor adds
        (a BOS for Llama-style checkpoints, nothing for most byte-level ones),
        as ``transformers`` does by default. SMG encodes ``/v1/completions``,
        ``/generate`` and ``/v1/tokenize`` text with ``add_special_tokens=False``
        and embedding input with ``True``. As in ``transformers``,
        special-token text in the input (``<|im_end|>``) encodes to the
        special token.
        """
        ids: list[int] = self._inner.encode(text, add_special_tokens)
        return ids

    def encode_batch(
        self, texts: Sequence[str], *, add_special_tokens: bool = True
    ) -> list[list[int]]:
        """Encode texts in parallel; each as :meth:`encode` encodes it."""
        if isinstance(texts, (str, bytes, bytearray)):
            # One text is a sequence too: of one-character texts.
            raise TypeError(
                "encode_batch() takes a sequence of texts, not a single text: encode() encodes one"
            )
        ids: list[list[int]] = self._inner.encode_batch(list(texts), add_special_tokens)
        return ids

    def decode(self, token_ids: TokenIds, *, skip_special_tokens: bool = False) -> str:
        """Decode ids (a sequence, or a numpy array or torch tensor of them),
        leaving out special tokens when ``skip_special_tokens``.

        The text is SMG's decode, which does not apply ``transformers``'
        ``clean_up_tokenization_spaces``.
        """
        text: str = self._inner.decode(_ids(token_ids), skip_special_tokens)
        return text

    def decode_batch(
        self, sequences: Sequence[TokenIds] | _ToList, *, skip_special_tokens: bool = False
    ) -> list[str]:
        """Decode sequences of ids (or a 2-D array or tensor) in parallel;
        each as :meth:`decode` does."""
        texts: list[str] = self._inner.decode_batch(
            [_ids(token_ids) for token_ids in _ids(sequences)], skip_special_tokens
        )
        return texts

    def __repr__(self) -> str:
        return f"{type(self).__name__}(source={self.source!r}, vocab_size={self.vocab_size})"
