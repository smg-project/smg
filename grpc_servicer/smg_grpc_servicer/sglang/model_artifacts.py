"""Local metadata for SGLang models whose weights stay in object storage.

SGLang's RunAI loader keeps the original URI for streaming weights and reads
config/tokenizer files from its persistent metadata cache. The gRPC frontend
needs that same cache before it reads ModelConfig and when it serves GetTokenizer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from smg_grpc_servicer.rust_lifecycle import (
    ServicerConfigError,
    holds_tokenizer,
    resolve_tokenizer_dir,
)
from smg_grpc_servicer.sglang.config_view import EngineConfigView

if TYPE_CHECKING:
    from sglang.srt.utils.runai_utils import ObjectStorageModel

logger = logging.getLogger(__name__)


def _object_storage_model(source: str) -> type[ObjectStorageModel] | None:
    # Local paths and Hub IDs need neither SGLang imports nor a streamer.
    if "://" not in source:
        return None
    from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

    return ObjectStorageModel if is_runai_obj_uri(source) else None


def prepare_model_metadata(server_args: Any) -> None:
    """Fetch metadata, never weights, before the frontend reads ModelConfig.

    Keep the engine's URI arguments intact. The cache belongs to SGLang and
    outlives the downloader's context manager; no TemporaryDirectory is owned
    by this frontend. Download errors must stop startup before the server binds.
    """
    args = EngineConfigView(server_args).server_args
    for source in dict.fromkeys((args.model_path, args.tokenizer_path)):
        if not source:
            continue
        storage = _object_storage_model(source)
        if storage is not None:
            try:
                storage.download_and_get_path(source)
            except Exception as error:
                raise ServicerConfigError(
                    f"Could not prepare object storage metadata for {source!r}: {error}"
                ) from error


def local_metadata_source(source: str) -> str:
    """Use the object storage metadata cache; keep local paths and Hub IDs as given."""
    storage = _object_storage_model(source)
    return storage.get_path(source) if storage is not None else source


def tokenizer_dir_for(server_args: Any) -> str | None:
    """Resolve the selected tokenizer to its local files, including the S3 cache."""
    args = EngineConfigView(server_args).server_args
    source = str(args.tokenizer_path or args.model_path)
    storage = _object_storage_model(source)
    if storage is None:
        return resolve_tokenizer_dir(source, args.revision)

    directory = storage.get_path(source)
    if not holds_tokenizer(directory):
        logger.warning("Object storage metadata for %r holds no tokenizer in %s", source, directory)
        return None
    return directory
