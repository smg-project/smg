"""Real Python-frame controls; these functions are not a TVM/TileLang substitute."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from host_ir_trace import LIMIT, ParserTrace, copy_sources, read_source  # noqa: E402


def compact_call():
    host_code = 'T.call_packed("device0", A)'
    function_names = ["device0"]
    private_token = "SENSITIVE_SENTINEL"
    return host_code.index(function_names[0]) + 0 * len(private_token)


def verbose_old_lookup():
    host_code = 'tir.Call(op=tir.tvm_call_packed, args=[StringImm(value="device0")])'
    function_names = ["device0"]
    return host_code.index(f'T.call_packed("{function_names[0]}"')


@pytest.mark.parametrize("function, raises", [(compact_call, False), (verbose_old_lookup, True)])
def test_real_frame_saved_on_return_and_exception(tmp_path, function, raises):
    output = tmp_path / "trace"
    before = sys.gettrace()
    if raises:
        with pytest.raises(ValueError, match="substring not found"):
            with ParserTrace(function.__code__, output):
                function()
    else:
        with ParserTrace(function.__code__, output):
            function()
    assert sys.gettrace() is before
    result = json.loads((output / "calls.json").read_text())
    calls = result["calls"]
    assert not result["invalid"]
    assert len(calls) == 1 and calls[0]["function_names"] == ["device0"]
    raw = (output / calls[0]["host_file"]).read_bytes()
    assert calls[0]["host_sha256"] == hashlib.sha256(raw).hexdigest()
    assert calls[0]["exception_types_observed"] == (["ValueError"] if raises else [])
    assert "SENSITIVE_SENTINEL" not in json.dumps(calls) + raw.decode()


def test_no_target_call_and_existing_output_refused(tmp_path):
    with pytest.raises(ValueError, match="was not observed"):
        with ParserTrace(verbose_old_lookup.__code__, tmp_path / "empty"):
            compact_call()  # Unrelated frame is not misidentified as the selected parser.
    with pytest.raises(FileExistsError):
        with ParserTrace(compact_call.__code__, tmp_path / "empty"):
            compact_call()


def test_existing_trace_hook_preserved(tmp_path):
    previous = sys.gettrace()

    def hook(frame, event, argument):
        return None

    sys.settrace(hook)
    try:
        with pytest.raises(ValueError, match="existing Python trace"):
            with ParserTrace(compact_call.__code__, tmp_path / "trace"):
                compact_call()
        assert sys.gettrace() is hook
    finally:
        sys.settrace(previous)


def test_oversized_actual_text_refused_and_hook_restored(tmp_path):
    def too_large():
        host_code = "x" * (LIMIT + 1)
        function_names = ["device0"]
        return len(host_code) + len(function_names)

    previous = sys.gettrace()
    with pytest.raises(ValueError, match="oversized actual parser host_code"):
        with ParserTrace(too_large.__code__, tmp_path / "trace"):
            too_large()
    assert sys.gettrace() is previous


def test_early_original_failure_preserved_without_host_text(tmp_path):
    def early_failure():
        raise RuntimeError("original parse failure")

    output = tmp_path / "trace"
    with pytest.raises(RuntimeError, match="original parse failure"):
        with ParserTrace(early_failure.__code__, output):
            early_failure()
    assert json.loads((output / "calls.json").read_text())["invalid"]


@pytest.mark.parametrize("mutation", [None, "symlink", "size"])
def test_whole_public_source_copy_and_bounds(tmp_path, mutation):
    source = tmp_path / "public.py"
    source.write_bytes(b"public bytes")
    path = source
    if mutation == "symlink":
        path = tmp_path / "escaped.py"
        path.symlink_to(source)
    elif mutation == "size":
        source.write_bytes(b"x" * (LIMIT + 1))
    output = tmp_path / "sources"
    if mutation:
        with pytest.raises(ValueError, match="escapes allowlist|exceeds cap"):
            copy_sources(output, (path,))
    else:
        copy_sources(output, (path,))
        record = json.loads((output / "index.json").read_text())[0]
        assert (output / record["file"]).read_bytes() == source.read_bytes()
        assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("required", [False, True])
def test_missing_public_sources_recorded_and_required_only_rejects(tmp_path, required):
    source = tmp_path / "absent.py"
    output = tmp_path / "sources"
    if required:
        with pytest.raises(FileNotFoundError, match="required actual"):
            copy_sources(output, (source,), (source,))
    else:
        copy_sources(output, (source,), ())
    assert json.loads((output / "index.json").read_text()) == [
        {"path": str(source), "exists": False, "required": required}
    ]


def test_public_source_read_requests_only_bounded_bytes(tmp_path, monkeypatch):
    source = tmp_path / "public.py"
    source.write_bytes(b"x" * (LIMIT + 2))
    original = Path.open
    reads = []

    class LimitedReader:
        def __enter__(self):
            self.stream = original(source, "rb")
            return self

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

        def __exit__(self, *args):
            self.stream.close()

    def limited(path, *args, **kwargs):
        return LimitedReader() if path == source else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", limited)
    with pytest.raises(ValueError, match="exceeds cap"):
        read_source(source)
    assert reads == [LIMIT + 1]
