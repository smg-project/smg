"""Generate narrow-frame expectations with the official MiniMax video processor.

Requires transformers==4.57.1, torch==2.14.0 and torchvision==0.29.0.
Run from the repository root.
The downloaded reference code is pinned and only its video processor is loaded.
"""

import importlib.util
import json
import tempfile
import urllib.request
from pathlib import Path

import torch

REVISION = "c5454eb03678d8710e54a4e0fc681b9f3b4a3dba"
SOURCE = f"https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8/resolve/{REVISION}/video_processor.py"
COLORS = [[32, 96, 224], [224, 64, 16]]


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "video_processor.py"
        path.write_bytes(urllib.request.urlopen(SOURCE).read())
        spec = importlib.util.spec_from_file_location("minimax_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        processor = module.MiniMaxM3VLVideoProcessor()
        cases = []
        for width, height in [(100, 20), (20, 100), (1, 1), (28, 28), (100, 100)]:
            video = torch.tensor(COLORS, dtype=torch.uint8)[:, :, None, None]
            video = video.expand(2, 3, height, width).contiguous()
            # Call the pixel path directly: frames are already sampled.
            # Transformers expects hashable normalization parameters here.
            output = processor._preprocess(
                videos=[video],
                **{
                    name: (
                        tuple(getattr(processor, name))
                        if name in ("image_mean", "image_std")
                        else getattr(processor, name)
                    )
                    for name in (
                        "do_convert_rgb",
                        "do_resize",
                        "size",
                        "resample",
                        "do_rescale",
                        "rescale_factor",
                        "do_normalize",
                        "image_mean",
                        "image_std",
                        "patch_size",
                        "temporal_patch_size",
                        "merge_size",
                        "min_pixels",
                        "max_pixels",
                    )
                },
                return_tensors="pt",
            )
            pixels = output["pixel_values_videos"]
            # Each frame has constant RGB channels. Verify that every spatial
            # patch is identical before storing one value per channel/frame.
            blocks = pixels.reshape(-1, 3, 2, 14 * 14)
            values = blocks[0, :, :, 0]
            torch.testing.assert_close(blocks, values[None, :, :, None].expand_as(blocks))
            cases.append(
                {
                    "width": width,
                    "height": height,
                    "grid": output["video_grid_thw"][0].tolist(),
                    "shape": list(pixels.shape),
                    "channel_frame_values": values.flatten().tolist(),
                }
            )
        destination = Path("crates/multimodal/tests/fixtures/golden/minimax_narrow_video.json")
        destination.write_text(
            json.dumps({"source": SOURCE, "colors": COLORS, "cases": cases}, indent=2) + "\n"
        )
        print(destination.read_text())


if __name__ == "__main__":
    main()
