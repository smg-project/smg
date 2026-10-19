"""
Shared tokenizer bundle utilities for gRPC servicers.

Builds and streams tokenizer artifacts as ZIP bundles over gRPC: every
engine's ``GetTokenizer`` takes the archive from :func:`tokenizer_bundle`
(built in a worker thread the first time, served from memory after that) and
streams it with :func:`tokenizer_chunks`.
"""

import asyncio
import hashlib
import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

from smg_grpc_proto.generated import common_pb2

# Streaming chunk size (aligned with Rust grpc_client limits)
CHUNK_SIZE = 64 * 1024  # 64 KB per gRPC chunk

# Files to include in the tokenizer ZIP bundle.
# Aligned with crates/tokenizer/src/hub.rs:is_tokenizer_file() plus model config files.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",  # SentencePiece
    "tiktoken.model",  # tiktoken
    "chat_template.json",
    "preprocessor_config.json",  # multimodal image preprocessor
    "processor_config.json",  # Transformers v5 nested multimodal processors
    "video_preprocessor_config.json",  # dedicated video preprocessor
]

# Glob patterns for additional tokenizer-related files
TOKENIZER_GLOBS = ["*.tiktoken", "*.jinja", "*.model"]


def build_tokenizer_zip(tokenizer_dir: Path) -> io.BytesIO:
    """Create an in-memory ZIP archive of tokenizer files from a directory."""
    buf = io.BytesIO()
    added: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Exact-name files
        for name in TOKENIZER_FILES:
            filepath = tokenizer_dir / name
            if filepath.is_file():
                zf.write(filepath, name)
                added.add(name)
        # Glob patterns (*.tiktoken, *.jinja, *.model)
        for pattern in TOKENIZER_GLOBS:
            for match in tokenizer_dir.glob(pattern):
                if match.is_file() and match.name not in added:
                    zf.write(match, match.name)
                    added.add(match.name)
    if not added:
        raise FileNotFoundError(f"No tokenizer files found in {tokenizer_dir}")
    buf.seek(0)
    return buf


class TokenizerBundle:
    """One tokenizer directory's archive, built once and kept.

    The first :meth:`archive` call builds it (:func:`build_tokenizer_zip` and
    the SHA-256 of its bytes) in a worker thread: reading and deflating the
    directory is file I/O and CPU measured in hundreds of milliseconds, and on
    the event loop it would stall every stream, health check and abort the
    servicer is serving meanwhile. Callers that arrive during the build await
    the same one, and a caller that goes away does not cancel it for the
    others. Once built, the archive is served from memory: the directory is
    the model's, fixed for the servicer's lifetime. A failed build is reported
    to its callers and not kept, so the next call tries again.
    """

    def __init__(self, tokenizer_dir: Path) -> None:
        self.tokenizer_dir = tokenizer_dir
        self._archive: tuple[bytes, str] | None = None
        self._build: asyncio.Task | None = None

    @staticmethod
    def build(tokenizer_dir: Path) -> tuple[bytes, str]:
        """The archive's bytes and their SHA-256 hex digest, built in one pass."""
        data = build_tokenizer_zip(tokenizer_dir).getvalue()
        return data, hashlib.sha256(data).hexdigest()

    async def archive(self) -> tuple[bytes, str]:
        """The archive's bytes and their SHA-256 hex digest, from memory once built."""
        if self._archive is not None:
            return self._archive
        loop = asyncio.get_running_loop()
        build = self._build
        # A build in flight is shared; one left by a loop that is gone is not.
        if build is None or build.get_loop() is not loop:
            build = self._build = loop.create_task(self._build_once())
        return await asyncio.shield(build)

    async def _build_once(self) -> tuple[bytes, str]:
        try:
            self._archive = await asyncio.to_thread(self.build, self.tokenizer_dir)
            return self._archive
        finally:
            self._build = None


# The bundles this process has been asked for, by directory: a servicer
# serves one model, so this is its one archive for the process's lifetime.
_bundles: dict[Path, TokenizerBundle] = {}


async def tokenizer_bundle(tokenizer_dir: str | os.PathLike[str]) -> tuple[bytes, str]:
    """The archive of ``tokenizer_dir`` and its SHA-256 hex digest, for the
    servicers' ``GetTokenizer``: built off the event loop the first time this
    process is asked for the directory and served from memory after that
    (:class:`TokenizerBundle`), however many requests ask for it.
    """
    key = Path(tokenizer_dir)
    bundle = _bundles.get(key)
    if bundle is None:
        bundle = _bundles[key] = TokenizerBundle(key)
    return await bundle.archive()


def tokenizer_chunks(zip_data, sha256: str) -> Iterator[common_pb2.GetTokenizerChunk]:
    """The archive's bytes as ``GetTokenizerChunk`` messages of at most
    :data:`CHUNK_SIZE` bytes each, in order; the SHA-256 fingerprint rides on
    the last chunk only, every other chunk carries an empty ``sha256``. An
    empty archive streams no chunk at all.

    ``zip_data`` is the archive's bytes (a ``bytes`` or the ``memoryview`` of
    the builder's buffer); ``sha256`` is its hex digest, computed by the caller
    (:func:`tokenizer_bundle` hands both over).
    """
    offset = 0
    total = len(zip_data)
    while offset < total:
        end = min(offset + CHUNK_SIZE, total)
        yield common_pb2.GetTokenizerChunk(
            data=bytes(zip_data[offset:end]),
            sha256=sha256 if end == total else "",
        )
        offset = end
