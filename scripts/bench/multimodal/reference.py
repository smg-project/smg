#!/usr/bin/env python3
"""The reference arm of the multimodal baseline: the serving engine's own media path.

For every class of the resolved manifest this runs what the engine's API server runs
for the same bytes, at the pinned library versions (see ``requirements.txt``): the image
loader (Pillow open, EXIF transpose, transparency composited over white, ``RGB``), the
checkpoint's image processor from ``transformers`` (the fast class the preprocessor
config names), the video loader (the OpenCV decode and the family's frame picks, then
the video processor) and the audio feature extractor on the decoded samples. It reports
the same rows as the crate arm: stage latencies (median and p90), throughput, process
CPU time per item, the output shape and grid, and digests of the decoded RGB bytes and
of the float32 output, so a crate row and a reference row of one class compare field by
field.

One process with one compute thread is the single-thread shape; ``--threads N`` runs N
processes with one compute thread each, the way N API-server processes would, started
behind a barrier. Run from the entry point, or by hand inside the pinned environment::

    python -I reference.py --manifest resolved.json --out reference.jsonl --threads 1 --repeats 10
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import queue as queue_mod
import sys
import time
from pathlib import Path

# One compute thread per process at the library level too, before the libraries load: the
# worker pools OpenBLAS and OpenMP start at import survive a fork as bookkeeping only, and
# the first call in a forked worker that needs them waits forever (the audio feature
# extractor's matmul, on a torch build over OpenBLAS). Set outright: an inherited pool
# size would bring the hang back and belie the table's one compute thread per process.
for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pool] = "1"

import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

# --- the engine's image loader rules (its own helpers, kept to the letter) -----------


def _has_transparency(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA", "PA"):
        return True
    return "transparency" in getattr(image, "info", {})


def _rgba_to_rgb(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    converted = Image.new("RGB", image.size, background)
    converted.paste(image, mask=image.split()[3])
    return converted


def convert_image_mode(image: Image.Image, to_mode: str) -> Image.Image:
    if image.mode == to_mode:
        return image
    if to_mode == "RGB" and _has_transparency(image):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return _rgba_to_rgb(image)
    return image.convert(to_mode)


def load_image(data: bytes) -> Image.Image:
    """Open, decode, orient and convert the way the engine's loader does."""
    image = Image.open(io.BytesIO(data))
    image.load()
    with contextlib.suppress(Exception):
        image = ImageOps.exif_transpose(image)
    return convert_image_mode(image, "RGB")


# --- the engine's video frame picks for the Qwen3-VL family ------------------------


def qwen3_vl_frame_indices(
    total: int, source_fps: float, fps: float = 2.0, min_frames: int = 4, max_frames: int = 768
) -> list[int]:
    num = int(total / source_fps * min(fps, 30))
    num = min(max(num, min_frames), max_frames, total)
    return np.linspace(0, total - 1, num).round().astype(int).tolist()


def load_video(data: bytes, tmp_dir: Path):
    """Decode the picked frames with OpenCV (BGR to RGB), as the engine's loader does."""
    import cv2  # noqa: PLC0415  (imported where the venv has it)

    path = tmp_dir / f"clip-{os.getpid()}.bin"
    path.write_bytes(data)
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        duration = total / source_fps if source_fps else 0.0
        wanted = qwen3_vl_frame_indices(total, source_fps)
        wanted_set = set(wanted)
        frames = []
        for index in range(max(wanted) + 1 if wanted else 0):
            ok = cap.grab()
            if not ok:
                break
            if index in wanted_set:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
        path.unlink(missing_ok=True)
    array = np.stack(frames) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
    # The engine hands the processor this metadata and `do_sample_frames=False`: the
    # frames are already the picks.
    metadata = {
        "total_num_frames": total,
        "fps": source_fps,
        "duration": duration,
        "video_backend": "opencv",
        "frames_indices": wanted[: len(frames)],
    }
    return array, metadata


def load_audio(data: bytes):
    import soundfile as sf  # noqa: PLC0415

    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0], rate


# --- processors -----------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Family:
    """The processors of one checkpoint directory, loaded once per process."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self._image = None
        self._video = None
        self._audio = None

    def image(self):
        if self._image is None:
            from transformers import AutoImageProcessor  # noqa: PLC0415

            self._image = AutoImageProcessor.from_pretrained(self.config_dir)
        return self._image

    def video(self):
        if self._video is None:
            from transformers import AutoVideoProcessor  # noqa: PLC0415

            self._video = AutoVideoProcessor.from_pretrained(self.config_dir)
        return self._video

    def audio(self):
        if self._audio is None:
            from transformers import AutoFeatureExtractor  # noqa: PLC0415

            self._audio = AutoFeatureExtractor.from_pretrained(self.config_dir)
        return self._audio


def tensor_bytes(tensor) -> bytes:
    return tensor.detach().contiguous().to("cpu").numpy().tobytes()


def run_once(cls: dict, family: Family, blobs: list[bytes], tmp_dir: Path) -> dict:
    """One iteration of a class: stage times in ms and the outputs for the digests."""
    kind = cls["kind"]
    if kind in ("image", "batch"):
        t0 = time.perf_counter()
        images = [load_image(blob) for blob in blobs]
        t1 = time.perf_counter()
        out = family.image()(images=images, return_tensors="pt")
        t2 = time.perf_counter()
        pixels = out["pixel_values"]
        grid = out.get("image_grid_thw")
        return {
            "decode_ms": (t1 - t0) * 1000.0,
            "preprocess_ms": (t2 - t1) * 1000.0,
            "decoded_bytes": sum(im.width * im.height * 3 for im in images),
            "output_bytes": pixels.numel() * pixels.element_size(),
            "rgb": images[0],
            "pixels": pixels,
            "grid": grid.flatten().tolist() if grid is not None else [],
            "shape": list(pixels.shape),
            "frames": None,
        }
    if kind == "video":
        t0 = time.perf_counter()
        frames, metadata = load_video(blobs[0], tmp_dir)
        t1 = time.perf_counter()
        out = family.video()(
            videos=[frames], video_metadata=[metadata], return_tensors="pt", do_sample_frames=False
        )
        t2 = time.perf_counter()
        pixels = out["pixel_values_videos"]
        grid = out.get("video_grid_thw")
        return {
            "decode_ms": (t1 - t0) * 1000.0,
            "preprocess_ms": (t2 - t1) * 1000.0,
            "decoded_bytes": int(frames.nbytes),
            "output_bytes": pixels.numel() * pixels.element_size(),
            "rgb": None,
            "pixels": pixels,
            "grid": grid.flatten().tolist() if grid is not None else [],
            "shape": list(pixels.shape),
            "frames": int(frames.shape[0]),
        }
    if kind == "audio":
        t0 = time.perf_counter()
        samples, rate = load_audio(blobs[0])
        t1 = time.perf_counter()
        # The engine's audio processor hands the extractor these arguments: pad to the
        # longest clip of the batch, never truncate to the extractor's 30 s window.
        out = family.audio()(
            samples,
            sampling_rate=rate,
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_attention_mask=True,
        )
        t2 = time.perf_counter()
        features = out["input_features"]
        return {
            "decode_ms": (t1 - t0) * 1000.0,
            "preprocess_ms": (t2 - t1) * 1000.0,
            "decoded_bytes": int(samples.nbytes),
            "output_bytes": features.numel() * features.element_size(),
            "rgb": None,
            "pixels": features,
            "grid": [],
            "shape": list(features.shape),
            "frames": None,
        }
    raise ValueError(f"unknown class kind {kind!r}")


def worker(cls: dict, repeats: int, barrier, queue, tmp_dir: str) -> None:
    try:
        # the setup inside the try too: a failure there aborts the barrier and reports,
        # rather than leaving the parent waiting for a result that never comes
        import torch  # noqa: PLC0415

        torch.set_num_threads(1)
        family = Family(cls["config_dir"])
        blobs = [Path(path).read_bytes() for path in cls["files"]]
        run_once(cls, family, blobs, Path(tmp_dir))  # warm-up, not recorded
        barrier.wait()
        # CLOCK_PROCESS_CPUTIME_ID (user + system, every thread of the process), the
        # clock the crate arm reads; times(2) counts in 10 ms ticks
        before = time.process_time()
        samples = []
        for _ in range(repeats):
            result = run_once(cls, family, blobs, Path(tmp_dir))
            samples.append((result["decode_ms"], result["preprocess_ms"]))
        after = time.process_time()
        queue.put({"samples": samples, "cpu_ms": (after - before) * 1000.0})
    except Exception as error:  # noqa: BLE001  (reported as the class's error)
        with contextlib.suppress(Exception):
            barrier.abort()
        queue.put({"error": f"{type(error).__name__}: {error}"})


def summary(values: list[float]) -> dict:
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]  # noqa: E731
    return {"median_ms": pick(0.5), "p90_ms": pick(0.9), "min_ms": ordered[0]}


def facts(cls: dict, tmp_dir: Path) -> dict:
    """Shape, grid and digests from one untimed pass in this process."""
    import torch  # noqa: PLC0415

    torch.set_num_threads(1)
    family = Family(cls["config_dir"])
    blobs = [Path(path).read_bytes() for path in cls["files"]]
    result = run_once(cls, family, blobs, tmp_dir)
    pixels = result["pixels"]
    f32 = pixels.to(torch.float32) if pixels.dtype != torch.float32 else pixels
    rgb = result["rgb"]
    processor = {
        "image": type(family._image).__name__ if family._image else None,
        "video": type(family._video).__name__ if family._video else None,
        "audio": type(family._audio).__name__ if family._audio else None,
    }
    return {
        "shape": result["shape"],
        "grid": result["grid"],
        "video_frames": result["frames"],
        "decoded_bytes": result["decoded_bytes"],
        "output_bytes": result["output_bytes"],
        "output_dtype": str(pixels.dtype),
        "processor": {k: v for k, v in processor.items() if v},
        "digests": {
            "rgb_sha256": sha256(np.asarray(rgb, dtype=np.uint8).tobytes())
            if rgb is not None
            else None,
            "f32_sha256": sha256(tensor_bytes(f32)),
        },
    }


def run_class(cls: dict, threads: int, repeats: int, tmp_dir: Path) -> dict:
    repeats = min(max(1, cls.get("repeats") or repeats), repeats)
    base = {
        "kind": "row",
        "class": cls["name"],
        "class_kind": cls["kind"],
        "threads": threads,
        "repeats": repeats,
    }
    try:
        known = facts(cls, tmp_dir)
    except Exception as error:  # noqa: BLE001
        return {**base, "error": f"{type(error).__name__}: {error}"}
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(threads + 1)
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(cls, repeats, barrier, queue, str(tmp_dir)))
        for _ in range(threads)
    ]
    for proc in procs:
        proc.start()
    try:
        barrier.wait(timeout=1800)
    except Exception:  # noqa: BLE001  (a worker failed before the barrier)
        pass
    started = time.perf_counter()
    results: list[dict] = []
    while len(results) < len(procs):
        try:
            results.append(queue.get(timeout=5))
        except queue_mod.Empty:
            # a worker that died without reporting (the OOM killer, a native crash)
            # would hold this read forever otherwise
            dead = [proc.exitcode for proc in procs if proc.exitcode not in (None, 0)]
            if dead:
                results.append({"error": f"a worker exited with code {dead[0]} before it reported"})
                break
    wall_ms = (time.perf_counter() - started) * 1000.0
    for proc in procs:
        proc.join()
    errors = [r["error"] for r in results if "error" in r]
    if errors:
        return {**base, "error": errors[0]}
    samples = [s for r in results for s in r["samples"]]
    iterations = len(samples)
    items_per_request = len(cls["files"]) if cls["kind"] in ("image", "batch") else 1
    items = iterations * items_per_request
    cpu_ms = sum(r["cpu_ms"] for r in results)
    decode = [s[0] for s in samples]
    preprocess = [s[1] for s in samples]
    total = [a + b for a, b in samples]
    return {
        **base,
        "items_per_request": items_per_request,
        "iterations": iterations,
        "wall_ms": wall_ms,
        "requests_per_s": iterations / (wall_ms / 1000.0) if wall_ms else None,
        "items_per_s": items / (wall_ms / 1000.0) if wall_ms else None,
        "cpu_ms_per_request": cpu_ms / iterations if iterations else None,
        "cpu_ms_per_item": cpu_ms / items if items else None,
        "cpu_note": (
            "the worker processes' own CPU time over the timed iterations "
            "(the warm-up and the processor load excluded)"
        ),
        "stage_ms": {
            "decode": summary(decode),
            "preprocess": summary(preprocess),
            "total": summary(total),
        },
        "bytes": {"decoded": known["decoded_bytes"], "output": known["output_bytes"]},
        **{
            k: known[k]
            for k in ("shape", "grid", "video_frames", "output_dtype", "processor", "digests")
        },
    }


def meta(threads: int, repeats: int) -> dict:
    import PIL  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    try:
        import cv2  # noqa: PLC0415

        cv2_version = cv2.__version__
    except Exception:  # noqa: BLE001
        cv2_version = None
    return {
        "kind": "meta",
        "arm": "reference",
        "arch": platform.machine(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "opencv": cv2_version,
        "pillow_libjpeg_turbo": bool(
            __import__("PIL.features", fromlist=["features"]).check_feature("libjpeg_turbo")
        ),
        "threads": threads,
        "repeats": repeats,
        "compute_threads_per_process": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--only", default=None, help="comma-separated class names")
    parser.add_argument(
        "--tmp", default=None, help="scratch directory for the video decoder's files"
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    only = set(args.only.split(",")) if args.only else None
    tmp_dir = Path(args.tmp or os.environ.get("TMPDIR", "/tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ok = err = 0
    with open(args.out, "w") as out:
        out.write(json.dumps(meta(args.threads, args.repeats)) + "\n")
        for cls in manifest["classes"]:
            if only and cls["name"] not in only:
                continue
            print(f"reference: {cls['name']} ({args.threads} processes)", file=sys.stderr)
            row = run_class(cls, args.threads, args.repeats, tmp_dir)
            if "error" in row:
                err += 1
                print(f"reference: {cls['name']} failed: {row['error']}", file=sys.stderr)
            else:
                ok += 1
            out.write(json.dumps(row) + "\n")
            out.flush()
    print(f"REFERENCE_DONE classes={ok + err} ok={ok} err={err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
