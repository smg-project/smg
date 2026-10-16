import asyncio
import hashlib
import io
import random
import zipfile
from types import SimpleNamespace

import pytest
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer.tokenizer_bundle import CHUNK_SIZE, build_tokenizer_zip, tokenizer_chunks


def test_build_tokenizer_zip_includes_processor_config(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(
        '{"image_processor": {}, "video_processor": {}}', encoding="utf-8"
    )
    # The gateway prefers a dedicated video config over the nested
    # video_processor block; omitting it from the bundle would make bundle
    # and lazy deployments of the same repo resolve different video configs.
    (tmp_path / "video_preprocessor_config.json").write_text("{}", encoding="utf-8")

    bundle = build_tokenizer_zip(tmp_path)

    with zipfile.ZipFile(bundle) as archive:
        assert "processor_config.json" in archive.namelist()
        assert "video_preprocessor_config.json" in archive.namelist()


# ---------------------------------------------------------------------------
# tokenizer_chunks: the one chunk loop every engine's GetTokenizer streams.
# ---------------------------------------------------------------------------


def _reference_frames(zip_data, sha256):
    """The chunk loop the three servicers carried before it was shared, kept
    here as the oracle: the generator's frames must be these, byte for byte."""
    frames = []
    offset = 0
    total = len(zip_data)
    while offset < total:
        end = min(offset + CHUNK_SIZE, total)
        is_last = end == total
        frames.append(
            common_pb2.GetTokenizerChunk(
                data=bytes(zip_data[offset:end]),
                sha256=sha256 if is_last else "",
            ).SerializeToString()
        )
        offset = end
    return frames


def _digest(data) -> str:
    return hashlib.sha256(data).hexdigest()


def _frames(zip_data, sha256):
    return [chunk.SerializeToString() for chunk in tokenizer_chunks(zip_data, sha256)]


def test_an_exact_multiple_of_the_chunk_size_fills_every_chunk():
    data = bytes(range(256)) * (3 * CHUNK_SIZE // 256)
    assert len(data) == 3 * CHUNK_SIZE
    chunks = list(tokenizer_chunks(data, _digest(data)))
    assert [len(chunk.data) for chunk in chunks] == [CHUNK_SIZE] * 3
    assert b"".join(chunk.data for chunk in chunks) == data
    assert [chunk.sha256 for chunk in chunks] == ["", "", _digest(data)]


def test_a_remainder_is_the_last_chunk_and_carries_the_hash():
    data = b"\x5a" * (2 * CHUNK_SIZE + 7)
    chunks = list(tokenizer_chunks(data, _digest(data)))
    assert [len(chunk.data) for chunk in chunks] == [CHUNK_SIZE, CHUNK_SIZE, 7]
    assert b"".join(chunk.data for chunk in chunks) == data
    assert [chunk.sha256 for chunk in chunks] == ["", "", _digest(data)]


def test_an_empty_bundle_streams_no_chunk():
    assert list(tokenizer_chunks(b"", _digest(b""))) == []
    assert list(tokenizer_chunks(memoryview(b""), _digest(b""))) == []


def test_a_bundle_smaller_than_a_chunk_is_one_chunk_with_the_hash():
    data = b"PK\x05\x06" + bytes(18)
    (chunk,) = tokenizer_chunks(data, _digest(data))
    assert chunk.data == data
    assert chunk.sha256 == _digest(data)


@pytest.mark.parametrize(
    "size",
    [0, 1, CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 1, 2 * CHUNK_SIZE, 5 * CHUNK_SIZE + 4321],
    ids=lambda size: f"{size}B",
)
def test_the_frames_are_the_servicers_former_loop_byte_for_byte(size):
    data = bytes((index * 7919 + 13) & 0xFF for index in range(size))
    sha256 = _digest(data)
    expected = _reference_frames(data, sha256)
    # The servicers hand the builder's buffer over as a memoryview; bytes stream the same.
    assert _frames(memoryview(data), sha256) == expected
    assert _frames(data, sha256) == expected
    assert len(expected) == (size + CHUNK_SIZE - 1) // CHUNK_SIZE


def test_the_builders_archive_streams_whole_and_verifies_against_its_last_chunk(tmp_path):
    # Incompressible content, so the deflated archive spans several chunks.
    (tmp_path / "tokenizer.json").write_bytes(random.Random(7).randbytes(3 * CHUNK_SIZE + 511))
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    zip_data = build_tokenizer_zip(tmp_path).getbuffer()
    sha256 = _digest(zip_data)
    chunks = list(tokenizer_chunks(zip_data, sha256))
    joined = b"".join(chunk.data for chunk in chunks)
    assert joined == bytes(zip_data)
    assert _digest(joined) == chunks[-1].sha256
    assert len(chunks) == 4
    assert all(chunk.sha256 == "" for chunk in chunks[:-1])
    with zipfile.ZipFile(io.BytesIO(joined)) as archive:
        assert sorted(archive.namelist()) == ["tokenizer.json", "tokenizer_config.json"]


def test_the_tokenspeed_servicer_streams_the_shared_frames(tmp_path, tokenspeed_servicer_mod):
    """GetTokenizer end to end on one engine's servicer, the engine stubbed:
    the frames on the wire are the shared generator's, the former loop's."""
    (tmp_path / "tokenizer.json").write_bytes(random.Random(11).randbytes(2 * CHUNK_SIZE + 77))
    (tmp_path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    cls = tokenspeed_servicer_mod.TokenSpeedSchedulerServicer
    servicer = cls.__new__(cls)
    servicer._config = SimpleNamespace(server_args=SimpleNamespace(tokenizer=str(tmp_path)))

    async def collect():
        request = common_pb2.GetTokenizerRequest()
        return [chunk.SerializeToString() async for chunk in servicer.GetTokenizer(request, None)]

    frames = asyncio.run(collect())
    zip_data = build_tokenizer_zip(tmp_path).getbuffer()
    assert frames == _reference_frames(zip_data, _digest(zip_data))
    assert len(frames) == 3
