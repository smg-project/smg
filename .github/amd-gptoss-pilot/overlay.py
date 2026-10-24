"""Apply a fixed, job-local development overlay; never alter package metadata."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def target(root, relative):
    path = root / relative
    require(not path.is_symlink(), f"symlink target: {relative}")
    require(path.resolve().is_relative_to(root.resolve()), f"escaped target: {relative}")
    return path


def apply(root, manifest):
    pending = []
    for entry in manifest["files"]:
        path = target(root, entry["path"])
        if entry["old_sha256"] is None:
            require(not path.exists(), f"new helper already exists: {entry['path']}")
            if entry.get("source"):
                require(entry["source"] == "mm_trace.py", "unexpected observation helper")
                data, mode = (HERE / entry["source"]).read_bytes(), 0o644
            else:
                data, mode = entry["new_content"].encode(), 0o644
        else:
            require(path.is_file(), f"missing original: {entry['path']}")
            data, mode = path.read_bytes(), stat.S_IMODE(path.stat().st_mode)
            require(digest(data) == entry["old_sha256"], f"original hash: {entry['path']}")
            text = data.decode()
            for edit in entry["edits"]:
                require(text.count(edit["old"]) == 1, f"ambiguous hunk: {entry['path']}")
                text = text.replace(edit["old"], edit["new"], 1)
            data = text.encode()
        require(digest(data) == entry["new_sha256"], f"patch hash: {entry['path']}")
        pending.append((path, data, mode))
    # All original states and all resulting bytes are checked before any write.
    for path, data, mode in pending:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return verify(root, manifest)


def verify(root, manifest):
    records = []
    for entry in manifest["files"]:
        path = target(root, entry["path"])
        require(path.is_file(), f"missing overlay: {entry['path']}")
        actual = digest(path.read_bytes())
        require(actual == entry["new_sha256"], f"overlay overwritten: {entry['path']}")
        records.append(dict(path=str(path), sha256=actual))
    return records


def package_record(root, manifest):
    distribution = importlib.metadata.distribution("sglang")
    require(distribution.version == "0.5.21", "SGLang distribution changed")
    record = distribution.read_text("RECORD")
    located = {
        str(Path(distribution.locate_file(p)).resolve()): p.hash for p in distribution.files or ()
    }
    return dict(
        version=distribution.version,
        RECORD_sha256=digest(record.encode()) if record is not None else None,
        source_entries=[
            dict(
                path=e["path"], record_hash=str(located.get(str(target(root, e["path"]).resolve())))
            )
            for e in manifest["files"]
        ],
        interpretation=(
            "Source overlay is intentional. Editable sources may be outside RECORD; "
            "metadata/RECORD are not rewritten or claimed to validate changed sources."
        ),
    )


def origins(root, manifest):
    records = []
    for entry in manifest["files"]:
        if entry.get("defer_origin_to_qwen_trace"):
            extra = {
                "srt/managers/mm_schedule.py": "sglang.srt.managers.mm_schedule",
                "srt/model_executor/forward_batch_info.py": "sglang.srt.model_executor.forward_batch_info",
            }
            require(
                ((entry["path"] == "srt/managers/scheduler.py"
                  and entry["module"] == "sglang.srt.managers.scheduler")
                 or (os.environ.get("SMG_QWEN_MM_TRACE") == "1"
                     and extra.get(entry["path"]) == entry["module"]))
                and entry["defer_origin_to_qwen_trace"] is True,
                "only exact opted-in diagnostic origins may be deferred",
            )
            records.append(
                dict(
                    module=entry["module"],
                    origin=None,
                    status="DEFERRED_NOT_YET_VERIFIED",
                    expected_source_sha256=entry["new_sha256"],
                )
            )
            continue
        module = importlib.import_module(entry["module"])
        origin = Path(module.__file__).resolve()
        require(origin == target(root, entry["path"]).resolve(), "import origin mismatch")
        require(digest(origin.read_bytes()) == entry["new_sha256"], "import source mismatch")
        records.append(dict(module=entry["module"], origin=str(origin), sha256=entry["new_sha256"]))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("apply", "final"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "overlay.json").read_text())
    value = os.environ.get("SMG_QWEN_MM_TRACE", "0")
    require(value in {"0", "1"}, "invalid MM observation opt-in")
    if value == "1":
        extra = json.loads((HERE / "mm_overlay.json").read_text())
        require(os.environ.get("SMG_QWEN_MM_MODEL") == extra["model_id"]
                == "Qwen/Qwen3-VL-8B-Instruct", "MM observation model opt-in")
        require(digest((HERE / "overlay.json").read_bytes()) == extra["base_overlay_sha256"],
                "canonical overlay changed")
        manifest["files"] += extra["files"]
    require(os.environ.get("SGLANG_GPT_OSS_MXFP4_BF16") == "1", "explicit BF16 switch missing")
    require(os.environ.get("SMG_SGLANG_ROCM_IMAGE") == manifest["image"], "image binding")
    spec = importlib.util.find_spec("sglang")  # Top-level lookup, no engine import.
    require(spec and spec.submodule_search_locations, "SGLang package missing")
    root = Path(next(iter(spec.submodule_search_locations))).resolve()
    require(str(root) == manifest["package_root"], "actual image package root changed")
    directory = args.directory
    if args.mode == "apply":
        directory.mkdir(exist_ok=False)
    require(directory.is_dir(), "overlay evidence directory missing")
    result = dict(mode=args.mode, scope="PARTIAL_DEVELOPER_PILOT_NO231_ACCEPTANCE")
    try:
        before = package_record(root, manifest)
        if args.mode == "apply":
            (directory / "record-before.json").write_text(json.dumps(before, indent=2) + "\n")
            result["files"] = apply(root, manifest)
        else:
            result["files"] = verify(root, manifest)
            require(
                before == json.loads((directory / "record-before.json").read_text()),
                "distribution RECORD or entries changed after overlay",
            )
        result["intentional_source_changes"] = [
            dict(
                path=e["path"],
                original_sha256=e["old_sha256"],
                overlay_sha256=e["new_sha256"],
                RECORD_rewritten=False,
            )
            for e in manifest["files"]
        ]
        result["imports"] = origins(root, manifest)
        require(before == package_record(root, manifest), "overlay rewrote metadata")
        result["success"] = True
    except Exception:
        result.update(success=False, error=traceback.format_exc())
        raise
    finally:
        (directory / (args.mode + ".json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
