import ast
import asyncio
import hashlib
import io
import logging
import random
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest
from smg_grpc_proto.generated import common_pb2
from smg_grpc_servicer import tokenizer_bundle as bundle_module
from smg_grpc_servicer.tokenizer_bundle import (
    CHUNK_SIZE,
    TokenizerBundle,
    build_tokenizer_zip,
    tokenizer_bundle,
    tokenizer_chunks,
)


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


# ---------------------------------------------------------------------------
# tokenizer_bundle: the archive every engine's GetTokenizer serves, built in a
# worker thread the first time it is asked for and from memory after that.
# ---------------------------------------------------------------------------


def _tokenizer_dir(tmp_path):
    (tmp_path / "tokenizer.json").write_bytes(b'{"id": 12345, "content": "<|token|>"}, ' * 1000)
    (tmp_path / "tokenizer_config.json").write_text('{"model_max_length": 8192}', encoding="utf-8")
    (tmp_path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    return tmp_path


def _builds(monkeypatch, gate=None):
    """The directories the shared module's builder is asked for from now on.

    With a ``gate``, every build first waits for the test to open it from the
    loop; the real archive follows. A build that ran on the loop would hold
    the loop itself, so nobody could open the gate: the wait gives up after a
    few seconds and such a test fails instead of hanging."""
    builds = []
    builder = bundle_module.build_tokenizer_zip

    def building(tokenizer_dir):
        builds.append(tokenizer_dir)
        if gate is not None:
            gate.wait(5.0)
        return builder(tokenizer_dir)

    monkeypatch.setattr(bundle_module, "build_tokenizer_zip", building)
    return builds


async def _ticks_while_held(work, gate, ticks=20):
    """Start ``work`` on the loop, sleep a millisecond at a time on the same
    loop ``ticks`` times while its build is held, then open the gate. Returns
    the work's result and whether it was still pending when the gate opened:
    it is when its build runs in a worker thread, and it is not when the build
    ran on the loop (the ticks could only start after it)."""
    pending = asyncio.ensure_future(work())
    for _ in range(ticks):
        await asyncio.sleep(0.001)
    still_building = not pending.done()
    gate.set()
    return await pending, still_building


def test_the_bundle_is_the_builders_archive_byte_for_byte(tmp_path):
    """What the shared helper hands over is build_tokenizer_zip's archive and
    its digest: nothing on the wire changes with the build moving off the loop."""
    tokenizer_dir = _tokenizer_dir(tmp_path)
    data, sha256 = asyncio.run(tokenizer_bundle(tokenizer_dir))
    expected = build_tokenizer_zip(tokenizer_dir).getvalue()
    assert data == expected
    assert sha256 == _digest(expected)
    assert TokenizerBundle.build(tokenizer_dir) == (expected, sha256)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]


def test_the_event_loop_stays_free_while_the_archive_is_built(tmp_path, monkeypatch):
    """A coroutine sleeping a millisecond at a time on the servicer's loop keeps
    running while the archive is built: the build is in a worker thread, held
    here until that coroutine has made its ticks."""
    tokenizer_dir = _tokenizer_dir(tmp_path)
    gate = threading.Event()
    builds = _builds(monkeypatch, gate)
    (data, sha256), still_building = asyncio.run(
        _ticks_while_held(lambda: tokenizer_bundle(tokenizer_dir), gate)
    )
    assert still_building, "the loop could not tick while the archive was built"
    assert builds == [tokenizer_dir]
    assert data == build_tokenizer_zip(tokenizer_dir).getvalue()
    assert sha256 == _digest(data)


def test_the_archive_is_built_once_and_served_from_memory(tmp_path, monkeypatch):
    tokenizer_dir = _tokenizer_dir(tmp_path)
    builds = _builds(monkeypatch)
    first = asyncio.run(tokenizer_bundle(tokenizer_dir))
    # Another loop in the same process: the same archive, no second build.
    second = asyncio.run(tokenizer_bundle(tokenizer_dir))
    assert second is first
    assert builds == [tokenizer_dir]


def test_concurrent_first_callers_share_one_build(tmp_path, monkeypatch):
    """Several gateways starting together ask one worker at once: one build."""
    tokenizer_dir = _tokenizer_dir(tmp_path)
    gate = threading.Event()
    builds = _builds(monkeypatch, gate)

    async def five_at_once():
        callers = [asyncio.ensure_future(tokenizer_bundle(tokenizer_dir)) for _ in range(5)]
        await asyncio.sleep(0)  # all five wait on the one held build
        gate.set()
        return await asyncio.gather(*callers)

    results = asyncio.run(five_at_once())
    assert builds == [tokenizer_dir]
    assert all(result is results[0] for result in results)
    assert results[0][0] == build_tokenizer_zip(tokenizer_dir).getvalue()


def test_a_caller_that_goes_away_leaves_the_build_to_the_others(tmp_path, monkeypatch):
    tokenizer_dir = _tokenizer_dir(tmp_path)
    gate = threading.Event()
    builds = _builds(monkeypatch, gate)

    async def the_first_leaves():
        first = asyncio.ensure_future(tokenizer_bundle(tokenizer_dir))
        second = asyncio.ensure_future(tokenizer_bundle(tokenizer_dir))
        await asyncio.sleep(0)  # both wait on the one held build
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        gate.set()
        return await second

    data, sha256 = asyncio.run(the_first_leaves())
    assert data == build_tokenizer_zip(tokenizer_dir).getvalue()
    assert sha256 == _digest(data)
    assert builds == [tokenizer_dir]


def test_a_failed_build_is_not_kept(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(tokenizer_bundle(tmp_path))  # nothing to bundle yet
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    data, sha256 = asyncio.run(tokenizer_bundle(tmp_path))
    assert data == build_tokenizer_zip(tmp_path).getvalue()
    assert sha256 == _digest(data)


# ---------------------------------------------------------------------------
# GetTokenizer on the four engines' servicers: the real RPC bodies, lifted
# from the modules (test_vllm_server_info's way, the engines' imports left
# out) and run with the names the bodies use.
# ---------------------------------------------------------------------------

_SERVICERS = Path(__file__).parents[1] / "smg_grpc_servicer"

_SERVICER_CLASSES = {
    "vllm": "VllmEngineServicer",
    "sglang": "SGLangSchedulerServicer",
    "tokenspeed": "TokenSpeedSchedulerServicer",
    "mlx": "MlxEngineServicer",
}


def _servicer_for(engine, tokenizer_dir):
    """A stand-in servicer with the attributes the engine's GetTokenizer reads."""
    path = str(tokenizer_dir)
    if engine == "vllm":
        return SimpleNamespace(engine=SimpleNamespace(model_config=SimpleNamespace(tokenizer=path)))
    if engine == "sglang":
        return SimpleNamespace(server_args=SimpleNamespace(tokenizer_path=path))
    if engine == "tokenspeed":
        return SimpleNamespace(_config=SimpleNamespace(server_args=SimpleNamespace(tokenizer=path)))
    return SimpleNamespace(model_dir=path)


class _CountedBundle:
    """The shared helper, counting the calls a servicer body makes to it."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, tokenizer_dir):
        self.calls += 1
        return await tokenizer_bundle(tokenizer_dir)


def _get_tokenizer(engine, bundle):
    source = _SERVICERS / engine / "servicer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == _SERVICER_CLASSES[engine]
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "GetTokenizer"
    )
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "Path": Path,
        "grpc": grpc,
        "logger": logging.getLogger(f"{__name__}.{engine}"),
        "get_exception_traceback": lambda: "",
        "tokenizer_bundle": bundle,
        "tokenizer_chunks": tokenizer_chunks,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["GetTokenizer"]


class _NoAbort:
    async def abort(self, code, details):
        raise AssertionError(f"GetTokenizer aborted: {code} {details}")


@pytest.mark.parametrize("engine", sorted(_SERVICER_CLASSES))
def test_the_servicer_streams_the_shared_bundle_off_the_loop_and_once(
    engine, tmp_path, monkeypatch
):
    """The servicer's frames are the shared builder's archive byte for byte;
    the loop keeps ticking while the archive is built; a second request
    streams the same archive without a second build."""
    tokenizer_dir = _tokenizer_dir(tmp_path)
    bundle = _CountedBundle()
    get_tokenizer = _get_tokenizer(engine, bundle)
    servicer = _servicer_for(engine, tokenizer_dir)
    gate = threading.Event()
    builds = _builds(monkeypatch, gate)

    async def frames():
        request = common_pb2.GetTokenizerRequest()
        return [c.SerializeToString() async for c in get_tokenizer(servicer, request, _NoAbort())]

    first, still_building = asyncio.run(_ticks_while_held(frames, gate))
    second = asyncio.run(frames())
    zip_data = build_tokenizer_zip(tokenizer_dir).getvalue()
    assert first == _reference_frames(zip_data, _digest(zip_data))
    assert still_building, f"{engine}: the loop could not tick while the archive was built"
    assert second == first
    assert bundle.calls == 2
    assert builds == [tokenizer_dir]
