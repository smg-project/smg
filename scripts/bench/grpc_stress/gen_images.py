#!/usr/bin/env python3
"""Write N PNG files of about --bytes each (RGB noise, so they do not
compress: the body size a 4 MB photo has, with a decode cost to match).
Standard library only."""

import argparse
import os
import struct
import sys
import zlib


def chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def png(width, height, seed):
    rng = int.from_bytes(os.urandom(8), "little") if seed is None else seed
    rows = bytearray()
    state = rng
    for _ in range(height):
        rows.append(0)  # filter: none
        # xorshift64 per row keeps this fast enough for a few megabytes
        line = bytearray(width * 3)
        for i in range(0, width * 3, 8):
            state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
            state ^= state >> 7
            state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
            line[i : i + 8] = state.to_bytes(8, "little")
        rows += line[: width * 3]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 1))
        + chunk(b"IEND", b"")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--bytes", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    side = int((a.bytes / 3) ** 0.5)
    for k in range(a.count):
        path = os.path.join(a.out, f"noise-{k}.png")
        with open(path, "wb") as f:
            f.write(png(side, side, a.seed + k))
        print(path, os.path.getsize(path), f"{side}x{side}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
