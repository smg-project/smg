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
request of 16 KiB or more, a Hub download, and reading and hashing the
loaded files for the fingerprint stay on the calling thread; what can reach
rayon (the batch methods, ``apply_chat_template`` given several
conversations, building the tokenizer, and any encode of 16 KiB or more, a
chat's prompt among them) runs on a Rust thread pool of
``SMG_TOKENIZER_THREADS`` threads (one when ``TOKENIZERS_PARALLELISM`` is
off, rayon's default otherwise: ``RAYON_NUM_THREADS``, else the CPU count),
one per process: a process forked after the parent used the tokenizer (torch
``DataLoader`` workers, ``datasets.map(num_proc=...)``) starts its own, and
``os.fork()`` waits for the calls running in other threads (not for a Hub
download, which holds none of their locks). A tokenizer pickles as the files
it loaded, the smg build and their fingerprint, and reloads from those files
in the receiving process (where the path it loaded is missing, from a
download of the Hub snapshot it was in), refusing files that changed and
another smg build. The fingerprint is also what makes ``datasets.map``
recompute after a template or tokenizer file changes.

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
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast

from smg.smg_rs import Tokenizer as _Tokenizer

if TYPE_CHECKING:
    from typing import Protocol

    class _ToList(Protocol):
        def tolist(self) -> Any: ...

    # Annotations only (the wheel supports Python 3.9, where `X | Y` does not
    # evaluate): a chat message, in the shape a client sends to
    # /v1/chat/completions; a chat completion request, as a mapping or as its
    # JSON text; token ids, as a sequence or an array or tensor of them.
    Message = Mapping[str, Any]
    ChatRequest = Mapping[str, Any] | str | bytes
    TokenIds = Sequence[int] | _ToList

__all__ = ["RenderWarning", "RenderedChat", "Tokenizer"]


class RenderWarning(UserWarning):
    """A rewrite SMG made to a request while rendering it, which serving
    only logs: tool-call ``arguments`` that are not a JSON object rendered as
    ``{}``, message extensions of another provider's profile dropped,
    deprecated fields migrated. :meth:`Tokenizer.render_chat` returns them;
    :meth:`Tokenizer.apply_chat_template` issues them as warnings of this
    category (``warnings.simplefilter("error", RenderWarning)`` makes them
    errors)."""


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
    to the request while rendering it (see :class:`RenderWarning`).
    """

    text: str
    token_ids: list[int]
    from_text: bool
    generation_prompt: bool
    continued_final_message: bool
    warnings: list[str]


# transformers' apply_chat_template arguments (4.x and 5.x) that have no
# equivalent in SMG serving, with why. Given their no-op value (None or False)
# they are accepted and ignored; any other value is refused rather than
# handed to the template as a variable.
_UNSUPPORTED_CHAT_ARGUMENTS = {
    "chat_template": "the chat template is fixed when the tokenizer loads "
    "(Tokenizer(source, chat_template=path)), as the gateway fixes it at startup",
    "documents": "SMG serving does not pass per-request documents to the template",
    "padding": "pad the returned ids yourself",
    "truncation": "truncate the returned ids yourself",
    "max_length": "truncate the returned ids yourself",
    "return_tensors": "the ids come back as lists of ints",
    "return_dict": "render_chat() returns the text, the ids and how they were rendered",
    "return_assistant_tokens_mask": "SMG's renderers record no assistant spans",
    "tokenizer_kwargs": "chat prompts encode without special tokens, as SMG serving encodes them",
    "sanitize_control_tokens": "SMG serving has no such switch: control-token text in a message "
    "encodes as its renderer encodes it",
}

# Template variables SMG's render context fills itself, which a keyword
# argument does not replace: refused, with why. (transformers lets one replace
# strftime_now; given messages, its call to Jinja's render() fails.)
_CONTEXT_VARIABLES = {
    "messages": "the conversation fills it; pass that as the first argument",
    "strftime_now": "the template writes the local time at the render, as in serving; to pin "
    "it, start the process with SOURCE_DATE_EPOCH set (read once, as the process loads its "
    "first chat template)",
}


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


def _rendered(
    rendered: tuple[str, list[int] | None, bool, bool, bool, list[str], bool],
) -> RenderedChat:
    text, token_ids, from_text, generation_prompt, continued, render_warnings, _ = rendered
    return RenderedChat(
        text=text,
        token_ids=token_ids if token_ids is not None else [],
        from_text=from_text,
        generation_prompt=generation_prompt,
        continued_final_message=continued,
        warnings=render_warnings,
    )


def _restore(
    cls: type[Tokenizer],
    source: str,
    chat_template: str | None,
    model: str | None,
    hub_snapshot: tuple[str, str] | None,
    smg_build: str,
    fingerprint: str,
) -> Tokenizer:
    """Unpickle a tokenizer: load it again from the files it was loaded
    from, or, where the path it loaded is missing (another host), from a
    download of the Hub snapshot it was in; and refuse it when what loaded is
    not what was pickled."""
    if smg_build != _Tokenizer.smg_build:
        raise RuntimeError(
            f"this tokenizer was pickled by smg {smg_build} and is unpickled by smg "
            f"{_Tokenizer.smg_build}, whose token ids may differ; load it again: "
            f"Tokenizer({source!r}, ...)"
        )
    if hub_snapshot is not None and not os.path.exists(source):
        # Another host: the snapshot's files from the Hub, and the source's
        # path in them.
        hub_id, inside = hub_snapshot
        downloaded = cls(hub_id, chat_template=chat_template, model=model)
        missing = (
            f"this tokenizer was pickled from {source}, which is not here; {hub_id} "
            f"downloaded from the Hub (revision main) to {downloaded.source}"
        )
        if not inside:
            tokenizer = downloaded
        elif os.path.exists(os.path.join(downloaded.source, inside)):
            location = os.path.join(downloaded.source, inside)
            tokenizer = cls(location, chat_template=chat_template, model=model)
        else:
            raise RuntimeError(
                f"{missing}, without {inside}: make the pickled snapshot available here"
            )
        if tokenizer.fingerprint != fingerprint:
            raise RuntimeError(
                f"{missing}, whose files differ: make the pickled snapshot available here"
            )
        return tokenizer
    tokenizer = cls(source, chat_template=chat_template, model=model)
    if tokenizer.fingerprint != fingerprint:
        if hub_snapshot is not None:
            raise RuntimeError(
                f"the files of {source} (or its chat template) differ from those this "
                "tokenizer was pickled with: changed since, or only partly in this host's "
                f"Hugging Face cache, where Tokenizer({hub_snapshot[0]!r}) downloads the "
                "missing tokenizer and config files (DeepSeek V4.1's encoding/encoding.py "
                "comes from huggingface_hub.hf_hub_download); load it again: "
                f"Tokenizer({source!r}, ...)"
            )
        raise RuntimeError(
            f"the files of {source} (or its chat template) changed since this tokenizer "
            "was pickled; load it again: Tokenizer(...)"
        )
    return tokenizer


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
    def fingerprint(self) -> str:
        """A digest of the smg build, :attr:`model`, the chat template file,
        a file source itself and the checkpoint's tokenizer files (those the
        gateway's Hub download fetches, with ``config.json`` and
        ``generation_config.json``, and DeepSeek V4.1's
        ``encoding/encoding.py``, which that download does not fetch):
        tokenizers with one fingerprint render and encode alike, and a file
        edited in place changes it."""
        fingerprint: str = self._inner.fingerprint
        return fingerprint

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
        from its template (see :meth:`apply_chat_template`). As in
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

    def apply_chat_template(
        self,
        conversation: Sequence[Message] | Sequence[Sequence[Message]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        *,
        add_generation_prompt: bool = False,
        continue_final_message: bool | str = False,
        tokenize: bool = True,
        tool_choice: str | Mapping[str, Any] | None = None,
        reasoning_effort: str | None = None,
        **template_kwargs: Any,
    ) -> str | list[int] | list[str] | list[list[int]]:
        """Render a conversation (or a list of conversations) as SMG renders
        it, shaped like ``transformers``' ``apply_chat_template``.

        Returns the token ids SMG feeds the model (``tokenize=True``) or the
        prompt text; a list of conversations returns a list, rendered in
        parallel. ``tools=None`` declares no tools and ``tools=[]`` an empty
        list, which templates (and the gateway) tell apart. Keyword arguments
        beyond these reach the template as ``chat_template_kwargs`` do
        (``enable_thinking=False``); one that names a variable the render
        fills itself (``messages``, ``strftime_now``, a special token the
        checkpoint defines) raises ``TypeError``, as SMG would not pass it,
        and so does a ``transformers`` argument SMG has no equivalent for
        (``chat_template``, ``documents``, ``padding``, ``truncation``,
        ``max_length``, ``return_tensors``, ``return_dict``,
        ``return_assistant_tokens_mask``, ``tokenizer_kwargs``,
        ``sanitize_control_tokens``) unless it is given its no-op value
        (``None`` or ``False``), which is ignored. ``continue_final_message``
        continues the final message's content: ``True``, or ``"content"``, as
        ``transformers`` 5 names the field to continue (any other field raises
        ``TypeError``). ``tool_choice`` and ``reasoning_effort`` are request
        fields, which the gateway also hands the template. Messages are what a
        client sends to ``/v1/chat/completions``: a tool call carries its
        ``id`` and ``type``, as the gateway requires.

        A rendering that does not do what ``add_generation_prompt`` or
        ``continue_final_message`` ask (a checkpoint whose own encoder writes
        the generation header after a final user or tool message, a final
        message that is not the assistant's to continue) raises
        ``ValueError``; :meth:`render_chat` returns such a rendering as SMG
        makes it. A final assistant message that the renderer cannot continue
        itself is continued as SMG continues it: its text appended after the
        generation header. The rewrites SMG makes while rendering are issued
        as :class:`RenderWarning`. For renderers that encode segment by
        segment, the ids are not the encode of the returned text;
        :meth:`render_chat` says which.
        """
        for name, reason in _UNSUPPORTED_CHAT_ARGUMENTS.items():
            if name not in template_kwargs:
                continue
            if template_kwargs[name] is None or template_kwargs[name] is False:
                del template_kwargs[name]
                continue
            raise TypeError(f"apply_chat_template() does not take {name!r}: {reason}")
        self._check_template_kwargs(template_kwargs)
        if isinstance(continue_final_message, str) and continue_final_message != "content":
            raise TypeError(
                "apply_chat_template() does not take "
                f"continue_final_message={continue_final_message!r}: SMG continues the final "
                "message's content (True)"
            )

        def request(messages: Sequence[Message]) -> str:
            body: dict[str, Any] = {
                "messages": list(messages),
                # Always sent: the gateway renders the generation header
                # when the request leaves it out.
                "add_generation_prompt": add_generation_prompt,
            }
            if continue_final_message:
                body["continue_final_message"] = True
            if tools is not None:
                body["tools"] = list(tools)
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
            if reasoning_effort is not None:
                body["reasoning_effort"] = reasoning_effort
            if template_kwargs:
                body["chat_template_kwargs"] = template_kwargs
            return _request_json(body)

        def checked(
            rendered: tuple[str, list[int] | None, bool, bool, bool, list[str], bool], label: str
        ) -> tuple[str | list[int], list[str]]:
            text, token_ids, _, generation_prompt, _, render_warnings, continues = rendered
            if continue_final_message and not continues:
                raise ValueError(
                    f"{label}SMG did not continue the final message, as continue_final_message "
                    "asks: it continues a final assistant message only, and renders a "
                    "conversation that ends with another role's message closed; render_chat() "
                    "returns this rendering as it is"
                )
            if not continue_final_message and generation_prompt != add_generation_prompt:
                written = "wrote" if generation_prompt else "did not write"
                raise ValueError(
                    f"{label}SMG {written} the generation prompt although add_generation_prompt "
                    f"is {add_generation_prompt} (a renderer with an encoder of its own writes it "
                    "as the engines' encoders do); render_chat() returns this rendering as it is"
                )
            result = (token_ids or []) if tokenize else text
            return result, [f"{label}{message}" for message in render_warnings]

        batched = (
            isinstance(conversation, (list, tuple))
            and len(conversation) > 0
            and isinstance(conversation[0], (list, tuple))
        )
        if batched:
            conversations = cast("Sequence[Sequence[Message]]", conversation)
            rendered = self._inner.render_chat_batch(
                [request(messages) for messages in conversations], tokenize
            )
            results = []
            for index, one in enumerate(rendered):
                result, render_warnings = checked(one, f"requests[{index}]: ")
                for message in render_warnings:
                    warnings.warn(message, RenderWarning, stacklevel=2)
                results.append(result)
            return cast("list[str] | list[list[int]]", results)
        messages = cast("Sequence[Message]", conversation)
        result, render_warnings = checked(self._inner.render_chat(request(messages), tokenize), "")
        for message in render_warnings:
            warnings.warn(message, RenderWarning, stacklevel=2)
        return result

    def _check_template_kwargs(self, template_kwargs: Mapping[str, Any]) -> None:
        """Refuse the keyword arguments SMG's render context does not let
        replace its own variables."""
        for name, reason in _CONTEXT_VARIABLES.items():
            if name in template_kwargs:
                raise TypeError(
                    f"apply_chat_template() does not pass {name!r} to the template: {reason}"
                )
        for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
            token = self._special(name)
            if name in template_kwargs and token is not None:
                raise TypeError(
                    f"apply_chat_template() does not pass {name!r} to the template: the "
                    f"checkpoint's own ({token!r}) fills it, as in serving"
                )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _restore,
            (
                type(self),
                self.source,
                self.chat_template,
                self.model,
                self._inner.hub_snapshot,
                self.smg_build,
                self.fingerprint,
            ),
        )

    def __copy__(self) -> Tokenizer:
        # Immutable: a copy can share it.
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Tokenizer:
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(source={self.source!r}, model={self.model!r}, "
            f"provider_profile={self.provider_profile!r}, vocab_size={self.vocab_size})"
        )
