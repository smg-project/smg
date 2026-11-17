"""Object storage metadata reaches both gRPC frontends without changing weight URIs."""

from __future__ import annotations

import asyncio
import hashlib
import io
import sys
import types
import zipfile
from types import SimpleNamespace

import grpc
import pytest
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer.rust_lifecycle import ServicerConfigError
from smg_grpc_servicer.sglang import model_artifacts, rust
from smg_grpc_servicer.tokenizer_bundle import tokenizer_bundle
from test_tokenizer_bundle import _get_tokenizer


@pytest.fixture
def storage(monkeypatch, tmp_path):
    downloads = []

    class ObjectStorageModel:
        @staticmethod
        def get_path(source):
            return str(tmp_path / hashlib.sha256(source.encode()).hexdigest())

        @staticmethod
        def download_and_get_path(source):
            downloads.append(source)
            path = tmp_path / hashlib.sha256(source.encode()).hexdigest()
            path.mkdir(exist_ok=True)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "generation_config.json",
                "preprocessor_config.json",
                "image_preprocessing.json",
                "processor_config.json",
                "video_preprocessor_config.json",
                "chat_template.jinja",
            ):
                (path / name).write_text(name)
            return str(path)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils.runai_utils",
        SimpleNamespace(
            ObjectStorageModel=ObjectStorageModel,
            is_runai_obj_uri=lambda source: source.lower().startswith(("s3://", "gs://", "az://")),
        ),
    )
    return SimpleNamespace(model=ObjectStorageModel, downloads=downloads)


def _args(model_path, tokenizer_path=None):
    return SimpleNamespace(model_path=model_path, tokenizer_path=tokenizer_path, revision="pinned")


@pytest.mark.parametrize("scheme", ["s3", "gs", "az"])
def test_launcher_prepares_metadata_before_facts_and_bundles_the_native_cache(storage, scheme):
    uri = f"{scheme}://bucket/checkpoint"
    args = _args(uri, uri)
    launcher = rust.SglangLauncher()
    launcher.prepare(args)
    assert storage.downloads == [uri]
    # ModelConfig and the streaming weight loader still see the original URI.
    assert args.model_path == args.tokenizer_path == uri
    directory = launcher.tokenizer_dir(args)
    assert directory == storage.model.get_path(uri)

    archive, _ = asyncio.run(tokenizer_bundle(directory))
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert bundle.read("tokenizer.json") == b"tokenizer.json"
        assert bundle.read("config.json") == b"config.json"
        assert bundle.read("generation_config.json") == b"generation_config.json"
        assert bundle.read("chat_template.jinja") == b"chat_template.jinja"
        assert "processor_config.json" in bundle.namelist()
        assert "image_preprocessing.json" in bundle.namelist()
        assert "video_preprocessor_config.json" in bundle.namelist()


def test_separate_tokenizer_source_is_prepared_and_selected(storage):
    args = _args("s3://bucket/weights", "s3://bucket/tokenizer")
    model_artifacts.prepare_model_metadata(args)
    assert storage.downloads == [args.model_path, args.tokenizer_path]
    assert model_artifacts.tokenizer_dir_for(args) == storage.model.get_path(args.tokenizer_path)


def test_local_tokenizer_override_is_kept_when_weights_are_remote(storage, tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}")
    args = _args("s3://bucket/weights", str(tmp_path))
    model_artifacts.prepare_model_metadata(args)
    assert storage.downloads == [args.model_path]
    assert model_artifacts.tokenizer_dir_for(args) == str(tmp_path)


def test_native_media_reads_model_metadata_with_a_separate_tokenizer(storage, monkeypatch):
    from smg_grpc_servicer.sglang import mm_defaults

    monkeypatch.setenv("SMG_SGLANG_MM_PROCESSOR", "smg")
    monkeypatch.setenv("SMG_SGLANG_MM_DTYPE", "bfloat16")
    monkeypatch.setattr(mm_defaults, "configure_threads", lambda: None)
    args = _args("s3://bucket/vision-model", "s3://bucket/tokenizer")
    model_artifacts.prepare_model_metadata(args)
    options = rust.SglangLauncher().server_extras(args, model_artifacts.tokenizer_dir_for(args))
    assert options["smg_media_processor"]["model_dir"] == storage.model.get_path(args.model_path)
    assert options["smg_media_processor"]["model_id"] == args.model_path


@pytest.mark.parametrize("source", ["org/model", "/local/model"])
def test_non_object_storage_sources_keep_the_existing_resolver(monkeypatch, source):
    calls = []
    monkeypatch.setattr(
        model_artifacts,
        "resolve_tokenizer_dir",
        lambda path, revision: calls.append((path, revision)) or "/resolved",
    )
    args = _args(source)
    model_artifacts.prepare_model_metadata(args)
    assert model_artifacts.tokenizer_dir_for(args) == "/resolved"
    assert calls == [(source, "pinned")]


def test_metadata_download_failure_stops_startup(storage, monkeypatch):
    def denied(source):
        raise PermissionError("access denied")

    monkeypatch.setattr(storage.model, "download_and_get_path", denied)
    with pytest.raises(ServicerConfigError, match="Could not prepare object storage metadata"):
        rust.SglangLauncher().prepare(_args("s3://bucket/denied"))


def test_full_launcher_prepares_metadata_before_reading_model_facts(storage, monkeypatch):
    class FactsReached(Exception):
        pass

    class Launcher(rust.SglangLauncher):
        def kv_events_default(self, args):
            return None

        def facts(self, args):
            assert storage.downloads == [args.model_path]
            assert self.tokenizer_dir(args) == storage.model.get_path(args.model_path)
            raise FactsReached

    monkeypatch.setitem(sys.modules, "smg", types.ModuleType("smg"))
    monkeypatch.setitem(
        sys.modules, "smg.servicer", SimpleNamespace(init_servicer_tracing=lambda: None)
    )
    with pytest.raises(FactsReached):
        asyncio.run(Launcher().serve(_args("s3://bucket/checkpoint")))


def test_config_only_cache_is_not_advertised_as_a_tokenizer(storage, tmp_path, caplog):
    uri = "s3://bucket/config-only"
    path = tmp_path / hashlib.sha256(uri.encode()).hexdigest()
    path.mkdir()
    (path / "config.json").write_text("{}")
    assert model_artifacts.tokenizer_dir_for(_args(uri)) is None
    assert "holds no tokenizer" in caplog.text


def test_python_get_tokenizer_streams_the_same_object_storage_files(storage):
    args = _args("s3://bucket/checkpoint")
    model_artifacts.prepare_model_metadata(args)
    rpc = _get_tokenizer("sglang", tokenizer_bundle)

    async def collect():
        return [
            chunk
            async for chunk in rpc(
                SimpleNamespace(server_args=args), common_pb2.GetTokenizerRequest(), None
            )
        ]

    chunks = asyncio.run(collect())
    archive = b"".join(chunk.data for chunk in chunks)
    assert hashlib.sha256(archive).hexdigest() == chunks[-1].sha256
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert bundle.read("tokenizer.json") == b"tokenizer.json"
        assert bundle.read("preprocessor_config.json") == b"preprocessor_config.json"


def test_python_get_tokenizer_reports_missing_metadata_as_failed_precondition(storage):
    class Aborted(Exception):
        pass

    class Context:
        async def abort(self, code, details):
            assert code == grpc.StatusCode.FAILED_PRECONDITION
            raise Aborted(details)

    rpc = _get_tokenizer("sglang", tokenizer_bundle)

    async def collect():
        return [
            chunk
            async for chunk in rpc(
                SimpleNamespace(server_args=_args("s3://bucket/missing")),
                common_pb2.GetTokenizerRequest(),
                Context(),
            )
        ]

    with pytest.raises(Aborted, match="No local tokenizer metadata"):
        asyncio.run(collect())


def test_native_downloader_keeps_metadata_after_close_and_excludes_weights(monkeypatch, tmp_path):
    """Use SGLang's real cache/downloader; only object storage I/O is stood in."""
    native = pytest.importorskip("sglang.srt.utils.runai_utils")
    monkeypatch.setattr(
        native, "envs", SimpleNamespace(SGLANG_CACHE_DIR=SimpleNamespace(get=lambda: str(tmp_path)))
    )
    calls = []
    closed = []

    class Streamer:
        def __init__(self, model_path, dst):
            self.dst = dst

        def pull_files(self, allow_pattern, ignore_pattern):
            from pathlib import Path

            calls.append(ignore_pattern)
            path = Path(self.dst)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.json").write_text("{}")
            (path / "config.json").write_text("{}")

        def __exit__(self, *_args):
            closed.append(self.dst)

    monkeypatch.setitem(
        sys.modules, "runai_model_streamer", SimpleNamespace(ObjectStorageModel=Streamer)
    )
    args = _args("s3://bucket/native-checkpoint")
    model_artifacts.prepare_model_metadata(args)
    directory = model_artifacts.tokenizer_dir_for(args)
    assert directory == native.ObjectStorageModel.get_path(args.model_path)
    assert closed == [directory]
    assert all(pattern in calls[0] for pattern in ("*.pt", "*.safetensors", "*.bin", "*.pth"))
    archive, _ = asyncio.run(tokenizer_bundle(directory))
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert bundle.read("config.json") == b"{}"
        assert bundle.read("tokenizer.json") == b"{}"
