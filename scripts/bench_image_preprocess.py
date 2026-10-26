#!/usr/bin/env python3
"""Benchmark: HF transformers image preprocessing.

Compare with Rust benchmark:
    cargo bench -p llm-multimodal --bench image_preprocess   # run in smg-project/llm-multimodal

Usage:
    python scripts/bench_image_preprocess.py
    python scripts/bench_image_preprocess.py --mmmu   # include MMMU real images
"""

import argparse
import statistics
import time

import numpy as np
from PIL import Image


def make_test_image(width: int, height: int) -> Image.Image:
    """Create a synthetic RGB image matching the Rust benchmark."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    xs = np.arange(width) % 256
    ys = np.arange(height) % 256
    arr[:, :, 0] = xs[np.newaxis, :]
    arr[:, :, 1] = ys[:, np.newaxis]
    arr[:, :, 2] = (xs[np.newaxis, :] + ys[:, np.newaxis]) % 256
    return Image.fromarray(arr)


def bench_processor(processor, images: list[Image.Image], label: str, n_iter: int = 20):
    """Time the processor over multiple iterations."""
    # Warmup
    processor(images=images, return_tensors="pt")

    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        processor(images=images, return_tensors="pt")
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    print(f"  {label:30s}  {mean:8.2f} ms ± {std:5.2f} ms  (n={n_iter})")
    return mean


def bench_qwen3_vl(args):
    try:
        from transformers import Qwen2VLImageProcessorFast
    except ImportError:
        from transformers import AutoImageProcessor

        Qwen2VLImageProcessorFast = None

    model_path = "/raid/models/Qwen/Qwen3-VL-8B-Instruct"
    try:
        if Qwen2VLImageProcessorFast:
            processor = Qwen2VLImageProcessorFast.from_pretrained(model_path)
        else:
            processor = AutoImageProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"  Skipping Qwen3-VL: {e}")
        return

    print("\n=== Qwen3-VL (HF transformers) ===")

    sizes = [(224, 224), (640, 480), (1024, 768), (1920, 1080), (3840, 2160)]
    for w, h in sizes:
        img = make_test_image(w, h)
        bench_processor(processor, [img], f"single {w}x{h}")

    # Batch of 3
    for w, h in [(640, 480), (1024, 768)]:
        imgs = [make_test_image(w + i * 10, h + i * 10) for i in range(3)]
        bench_processor(processor, imgs, f"batch3 {w}x{h}")

    # MMMU real images
    if args.mmmu:
        bench_mmmu_images(processor, "Qwen3-VL")


def bench_qwen2_vl(args):
    try:
        from transformers import Qwen2VLImageProcessorFast
    except ImportError:
        from transformers import AutoImageProcessor

        Qwen2VLImageProcessorFast = None

    model_path = "/raid/models/Qwen/Qwen2-VL-2B-Instruct"
    try:
        if Qwen2VLImageProcessorFast:
            processor = Qwen2VLImageProcessorFast.from_pretrained(model_path)
        else:
            processor = AutoImageProcessor.from_pretrained(model_path)
    except Exception as e:
        print(f"  Skipping Qwen2-VL: {e}")
        return

    print("\n=== Qwen2-VL (HF transformers) ===")

    sizes = [(224, 224), (640, 480), (1024, 768), (1920, 1080)]
    for w, h in sizes:
        img = make_test_image(w, h)
        bench_processor(processor, [img], f"single {w}x{h}")


def bench_mmmu_images(processor, model_name: str):
    """Benchmark with real MMMU Art category images."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  Skipping MMMU: datasets not installed")
        return

    print(f"\n  --- MMMU Art images ({model_name}) ---")
    ds = load_dataset("MMMU/MMMU", "Art", split="validation")
    images = []
    for row in ds:
        for i in range(1, 8):
            img = row.get(f"image_{i}")
            if img is not None:
                images.append(img.convert("RGB"))

    print(f"  Loaded {len(images)} images from MMMU Art")
    sizes = [f"{img.width}x{img.height}" for img in images]
    print(f"  Sizes: {', '.join(sizes[:5])}{'...' if len(sizes) > 5 else ''}")

    # Benchmark: process all images one by one
    times = []
    for img in images:
        start = time.perf_counter()
        processor(images=[img], return_tensors="pt")
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    mean = statistics.mean(times)
    std = statistics.stdev(times)
    total = sum(times)
    print(f"  per-image:  {mean:8.2f} ms ± {std:5.2f} ms  ({len(images)} images)")
    print(f"  total:      {total:8.2f} ms")

    # Benchmark: process all as a batch
    batch_times = []
    for _ in range(5):
        start = time.perf_counter()
        processor(images=images, return_tensors="pt")
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed * 1000)

    mean_batch = statistics.mean(batch_times)
    print(f"  full batch: {mean_batch:8.2f} ms  (n=5)")


def bench_llama4(args):
    try:
        from transformers import AutoImageProcessor
    except ImportError:
        print("\n  Skipping Llama4: transformers not installed")
        return

    model_path = "/raid/models/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    try:
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"\n  Skipping Llama4: {e}")
        return

    print("\n=== Llama4-Maverick (HF transformers) ===")

    sizes = [(224, 224), (336, 336), (640, 480), (1024, 768), (1920, 1080)]
    for w, h in sizes:
        img = make_test_image(w, h)
        bench_processor(processor, [img], f"single {w}x{h}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark HF image preprocessing")
    parser.add_argument("--mmmu", action="store_true", help="Include MMMU real images")
    args = parser.parse_args()

    print("Image Preprocessing Benchmark (HF transformers)")
    print("=" * 60)
    print("Compare with: cargo bench -p llm-multimodal --bench image_preprocess (in smg-project/llm-multimodal)")

    bench_qwen3_vl(args)
    bench_qwen2_vl(args)
    bench_llama4(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
