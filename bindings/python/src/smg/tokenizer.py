"""SMG's tokenizer for Python: the token ids and chat prompts SMG serving produces.

:class:`Tokenizer` loads a checkpoint's tokenizer with the Rust code the
gateway serves with (``llm-tokenizer``), so a training or evaluation pipeline
can encode text, decode ids and render chats exactly as SMG does, without a
running gateway. Chats render through the gateway's own request path for
``/v1/chat/completions`` (gRPC workers): the request deserialized, normalized
and validated as the HTTP layer does, refused where the router refuses it,
message content shaped for the template, tool-call ``arguments`` strings
parsed for templates that take objects, ``chat_template_kwargs`` /
``reasoning_effort`` / ``tool_choice`` passed as the gateway passes them, and
the prompt encoded as the gateway's tokenize step does: the rendered text
without special tokens, or, for renderers that encode segment by segment
(Kimi K3, and the DeepSeek V3.2 / V4 / V4.1 encoders when message text spells
a control token), their own encode, which a flat encode of the text does not
reproduce.

Parity holds against a gateway built from the same smg commit
(:attr:`Tokenizer.smg_build`; compare the gateway's ``--version-verbose``)
that loads the same files: pass what the gateway loads (its tokenizer path,
else its model path; ``config.json`` and ``generation_config.json`` beside
the tokenizer files pick the renderer and the stop ids) and the same
``chat_template`` file when it is given one. A chat request renders under
:attr:`Tokenizer.model`, as the router renders it under the served model's
id: that id selects the provider profile (DeepSeek, Kimi, MiniMax, GLM) whose
request rewrites apply. It defaults to the Hub id of a Hub source or of a
Hugging Face cache snapshot; for any other path, pass ``model=`` the id SMG
serves the checkpoint as (or a vendor-named alias of it: the gateway's
``--model-alias`` registry is not read here). Without one, a request renders
under the model it names itself, which selects a profile by that name alone,
and a request that names none is refused.

Short calls (an encode of less than 16 KiB, a decode of fewer than 4096 ids,
a chat whose request and prompt are shorter than 16 KiB) run on the calling
thread with the GIL held, which costs less than releasing it; encode many
texts with the batch methods. The batch methods and longer calls release the
GIL. A decode of 4096 ids or more, the parse, checks and render of a chat
request of 16 KiB or more, and a Hub download stay on the calling thread;
what can reach rayon (the batch methods, building the tokenizer, and any
encode of 16 KiB or more, a chat's prompt among them) runs on a Rust thread
pool of ``SMG_TOKENIZER_THREADS`` threads (one when
``TOKENIZERS_PARALLELISM`` is off, rayon's default otherwise:
``RAYON_NUM_THREADS``, else the CPU count), one per process: a process
forked after the parent used the tokenizer (torch ``DataLoader`` workers,
``datasets.map(num_proc=...)``) starts its own.

Not reproduced: media content (SMG expands each image, video or audio item
into model-specific tokens), gpt-oss models (SMG renders them with Harmony),
and dates from templates that call ``strftime_now``, which writes the local
time at the render, as in serving (the same ``llm-tokenizer`` code). Where
``SOURCE_DATE_EPOCH`` (whole seconds since the Unix epoch) is set when the
process loads its first chat template, it writes that instant instead, also
in local time: ``llm-tokenizer`` reads the variable once, then, so set it in
the environment the process starts with.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

from smg.smg_rs import Tokenizer as _Tokenizer

if TYPE_CHECKING:
    from typing import Protocol

    class _ToList(Protocol):
        def tolist(self) -> Any: ...

    # Annotations only (the wheel supports Python 3.9, where `X | Y` does not
    # evaluate): a chat completion request, as a mapping or as its JSON text;
    # token ids, as a sequence or an array or tensor of them.
    ChatRequest = Mapping[str, Any] | str | bytes
    TokenIds = Sequence[int] | _ToList

__all__ = ["RenderedChat", "Tokenizer"]


class RenderedChat(TypedDict):
    """A chat request rendered as SMG renders it.

    ``text`` is the prompt; ``token_ids`` are the ids SMG feeds the model.
    ``from_text`` says whether those are the flat encode of ``text``
    (``encode(text, add_special_tokens=False)``): false for renderers that
    encode segment by segment, whose ids may differ from any encode of the
    text (never re-encode it). ``generation_prompt`` says whether the
    renderer wrote its generation header after the conversation: as
    ``add_generation_prompt`` asks; whatever it asks, for a renderer with an
    encoder of its own, as the engines' encoders do; and before the final
    message's text, for a renderer that cannot continue a message itself and
    appends the text to continue after the header (inside the reasoning block
    the header opens, if it opens one). ``continued_final_message`` says
    whether the renderer kept the final message open itself, as
    ``continue_final_message`` asks. ``warnings`` lists the rewrites SMG made
    to the request while rendering it, which serving only logs: tool-call
    ``arguments`` that are not a JSON object rendered as ``{}``, message
    extensions of another provider's profile dropped, deprecated fields
    migrated.
    """

    text: str
    token_ids: list[int]
    from_text: bool
    generation_prompt: bool
    continued_final_message: bool
    warnings: list[str]


def _json_value(value: Any) -> Any:
    """What ``json.dumps`` cannot write itself: numpy values (as their Python
    values), pydantic models, such as the OpenAI SDK's messages (their fields
    that are set), and mappings and sequences of other types."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _request_json(request: ChatRequest) -> str:
    """A chat request as the JSON text a client would send, so that the
    gateway's own deserializer reads it (its key order, number parsing and
    unknown-field handling included)."""
    if isinstance(request, str):
        return request
    if isinstance(request, (bytes, bytearray)):
        return bytes(request).decode("utf-8")
    if isinstance(request, Mapping):
        return json.dumps(dict(request), ensure_ascii=False, allow_nan=False, default=_json_value)
    raise TypeError(f"a chat request is a mapping or its JSON text, not {type(request).__name__}")


def _ids(token_ids: Any) -> Any:
    """Token ids as a list when they come as a numpy array or a torch tensor
    (which the Rust side does not read as a sequence); as given otherwise."""
    tolist = getattr(token_ids, "tolist", None)
    return tolist() if callable(tolist) else token_ids


def _rendered(rendered: tuple[str, list[int], bool, bool, bool, list[str]]) -> RenderedChat:
    text, token_ids, from_text, generation_prompt, continued, render_warnings = rendered
    return RenderedChat(
        text=text,
        token_ids=token_ids,
        from_text=from_text,
        generation_prompt=generation_prompt,
        continued_final_message=continued,
        warnings=render_warnings,
    )


class Tokenizer:
    """A checkpoint's tokenizer, loaded, encoding and rendering as SMG serving does.

    ``source`` is a checkpoint directory, a tokenizer file, or a Hugging Face
    model id, downloaded from the Hub as the gateway downloads it (honouring
    ``HF_TOKEN``, ``HF_HOME`` and ``HF_ENDPOINT``; the revision is always
    ``main``: for a pinned revision or an offline host, download with
    ``huggingface_hub.snapshot_download`` and pass the directory).
    ``chat_template`` is a template file to render with instead of the
    checkpoint's own, as the gateway's ``--chat-template``. ``model`` is the
    model id chat requests render under: the id SMG serves the checkpoint as,
    which selects the provider profile (see :attr:`provider_profile`). It
    defaults to the Hub id of a Hub source or of a Hugging Face cache
    snapshot (``.../models--{org}--{name}/snapshots/...``), and to none for
    any other path, where chat rendering then needs it (or a model named in
    the request).
    """

    smg_build: ClassVar[str] = _Tokenizer.smg_build
    """The smg release and commit this binding was built from: the token ids
    are those of a gateway built from the same commit."""

    def __init__(
        self,
        source: str | os.PathLike[str],
        *,
        chat_template: str | os.PathLike[str] | None = None,
        model: str | None = None,
    ) -> None:
        self._inner = _Tokenizer(
            os.fspath(source),
            None if chat_template is None else os.fspath(chat_template),
            model,
        )
        # Read once: they never change, and each read from Rust sorts them anew.
        special: dict[str, Any] = self._inner.special_tokens()
        self._additional_special_tokens: list[str] = special.pop("additional_special_tokens")
        self._special_tokens: dict[str, str | None] = special

    @classmethod
    def from_pretrained(
        cls,
        source: str | os.PathLike[str],
        *,
        chat_template: str | os.PathLike[str] | None = None,
        model: str | None = None,
    ) -> Tokenizer:
        """Load a tokenizer; the same as ``Tokenizer(source, ...)``."""
        return cls(source, chat_template=chat_template, model=model)

    @property
    def source(self) -> str:
        """Where the tokenizer was loaded from: the absolute path of a local
        checkpoint or of the Hub snapshot a model id downloaded to, or the
        OpenAI model name ``tiktoken`` loaded."""
        source: str = self._inner.source
        return source

    @property
    def chat_template(self) -> str | None:
        """The absolute path of the chat template file given at load, if any."""
        path: str | None = self._inner.chat_template
        return path

    @property
    def model(self) -> str | None:
        """The model id chat requests render under, if any: a request that
        names no model takes it, and one that names another (a client's
        alias) is rewritten to it, as the router rewrites a request to the
        served model's id. Without one, a request renders under the model it
        names, and one that names none is refused."""
        model: str | None = self._inner.model
        return model

    @property
    def provider_profile(self) -> str | None:
        """The provider profile :attr:`model` selects: ``"deepseek"``,
        ``"kimi"``, ``"minimax"``, ``"zai"`` (GLM), ``"openai"`` or
        ``"generic"``; ``None`` without a model id."""
        profile: str | None = self._inner.provider_profile
        return profile

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
        and embedding input with ``True``; a chat prompt's special tokens come
        from its template (see :meth:`render_chat`). As in
        ``transformers``, special-token text in the input (``<|im_end|>``)
        encodes to the special token.
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

    def render_chat(self, request: ChatRequest) -> RenderedChat:
        """Render a chat completion request as SMG renders it.

        ``request`` is the body a client would send to ``/v1/chat/completions``
        (a mapping, or its JSON text: a logged request body replays as is):
        ``messages`` and optionally ``tools``, ``tool_choice``,
        ``add_generation_prompt`` (``True`` when absent, as in the gateway),
        ``continue_final_message``, ``chat_template_kwargs``,
        ``reasoning_effort``, ``thinking`` and the other fields the gateway
        reads. It renders under :attr:`model` (see there). A request SMG would
        refuse with 400 raises ``ValueError`` with the gateway's reason.
        """
        return _rendered(self._inner.render_chat(_request_json(request)))

    def render_chat_batch(self, requests: Sequence[ChatRequest]) -> list[RenderedChat]:
        """Render chat completion requests in parallel; each as :meth:`render_chat` does."""
        if isinstance(requests, (Mapping, str, bytes, bytearray)):
            raise TypeError(
                "render_chat_batch() takes a sequence of requests, not a single request: "
                "render_chat() renders one"
            )
        rendered = self._inner.render_chat_batch([_request_json(r) for r in requests])
        return [_rendered(r) for r in rendered]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(source={self.source!r}, model={self.model!r}, "
            f"provider_profile={self.provider_profile!r}, vocab_size={self.vocab_size})"
        )
