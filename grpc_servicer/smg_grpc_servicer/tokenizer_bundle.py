"""
Shared tokenizer bundle utilities for gRPC servicers.

Builds and streams tokenizer artifacts as ZIP bundles over gRPC: every
engine's ``GetTokenizer`` builds the archive with :func:`build_tokenizer_zip`
and streams it with :func:`tokenizer_chunks`.
"""

import io
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


def tokenizer_chunks(zip_data, sha256: str) -> Iterator[common_pb2.GetTokenizerChunk]:
    """The archive's bytes as ``GetTokenizerChunk`` messages of at most
    :data:`CHUNK_SIZE` bytes each, in order; the SHA-256 fingerprint rides on
    the last chunk only, every other chunk carries an empty ``sha256``. An
    empty archive streams no chunk at all.

    ``zip_data`` is the archive's bytes (a ``bytes`` or the ``memoryview`` of
    the builder's buffer); ``sha256`` is its hex digest, computed by the caller
    (one servicer computes it off the event loop, with the archive).
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
