# The multimodal release comparison: 1.16.0 against 1.15.0

380 image cells (20 processor configurations by 19 image classes), time per image on one core: 338 faster by more than 2 %, 37 level (within 2 %), 5 slower by more than 2 %. Medians of 15 interleaved rounds; change = the head's time over the base's, minus one, so a negative figure is a faster head.

1.16.0 is `322350993` built with cargo 1.100.0 (3acc79c0c 2026-11-05); 1.15.0 is `82c0f1fcd` built with cargo 1.98.0 (797e8a9bc 2026-08-05); both sides run the harness source of `322350993`. Measured 2026-11-20. What a cell measures is in `crates/multimodal/benches/release/README.md`; beside this file, `environment.json` holds the identities, `cells.csv` every cell and `rounds.csv` every window.

Cells that read slower by more than 2 % (in how many rounds the head was the slower side, and the head's time over the base's in every round):

- `llama4` BMP: +7 % (28.5 ms against 26.7 ms; slower in 10 of 15 rounds: 0.969 1.059 1.466 1.092 1.283 1.127 0.916 1.125 1.006 0.879 1.068 1.040 1.110 0.829 0.976)
- `llava` JPEG 8192x8192: +6 % (345 ms against 325 ms; slower in 15 of 15 rounds: 1.119 1.069 1.127 1.020 1.066 1.057 1.059 1.046 1.057 1.105 1.058 1.035 1.030 1.063 1.104)
- `llava_next` JPEG 8192x8192: +4 % (377 ms against 362 ms; slower in 14 of 15 rounds: 1.064 1.036 1.056 0.933 1.036 1.035 1.073 1.031 1.008 1.069 1.024 1.058 1.078 1.053 1.074)
- `step3_vl` (model type `step5_vl`) JPEG 256x256: +3 % (3.76 ms against 3.66 ms; slower in 7 of 15 rounds: 0.991 0.942 1.120 1.105 0.865 1.012 0.990 1.456 1.256 0.925 0.893 1.265 0.921 1.063 0.870)
- `llama4` JPEG greyscale: +3 % (26.6 ms against 25.9 ms; slower in 7 of 15 rounds: 0.939 0.918 1.456 1.117 1.228 1.153 0.995 1.007 1.071 0.891 0.986 0.948 1.123 0.841 0.960)

## Every processor configuration, medians over the image classes

| processor | time per image | least .. most | CPU per image | batch of ten | faster / level / slower | same bytes, the standard classes | same bytes, the wider list on two wires, alpha composited and kept | the reference's time over the head's (least .. most) | the reference's bytes: decoded pixels, float32 | the base's float32 against the reference |
|---|---|---|---|---|---|---|---|---|---|---|
| `deepseek_v41` | -15 % | -22 % .. -1 % | -15 % | -44 % | 18 / 1 / 0 | 20 of 20 | 236 of 236 | - (no such class in the library) | - | - |
| `gemma4` | -47 % | -75 % .. -13 % | -47 % | -67 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | 1.89x (1.26x .. 4.91x) | 20 of 20, 20 of 20 | 20 of 20 |
| `glm53_flash` | -51 % | -62 % .. -26 % | -51 % | -45 % | 19 / 0 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 2.78x (1.62x .. 4.79x) | 20 of 20, 20 of 20 | 20 of 20 |
| `kimi_k25` | -35 % | -47 % .. -8 % | -35 % | -62 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | 2.56x (1.48x .. 3.98x) | 20 of 20, 0 of 20 | 0 of 20 |
| `kimi_k3` | -36 % | -61 % .. -12 % | -36 % | -61 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | - (its processor is checkpoint code) | - | - |
| `llama4` | -2 % | -22 % .. +7 % | -2 % | -4 % | 9 / 8 / 2 | 0 of 20 | 0 of 236 | 0.88x (0.49x .. 4.22x) | 20 of 20, 0 of 20 | 0 of 20 |
| `llava_next` | -19 % | -37 % .. +4 % | -19 % | -18 % | 18 / 0 / 1 | 0 of 20 | 0 of 236 | 1.95x (1.28x .. 3.37x) | 20 of 20, 20 of 20 | 0 of 20 |
| `llava` | -17 % | -33 % .. +6 % | -17 % | -9 % | 18 / 0 / 1 | 0 of 20 | 0 of 236 | 1.98x (1.10x .. 3.66x) | 20 of 20, 20 of 20 | 0 of 20 |
| `llava` with the `pixtral` processor | -29 % | -41 % .. -9 % | -29 % | -25 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | 3.31x (1.52x .. 5.47x) | 20 of 20, 20 of 20 | 20 of 20 |
| `minimax_m3` | -36 % | -49 % .. -11 % | -36 % | -47 % | 19 / 0 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 2.25x (1.37x .. 3.78x) | 20 of 20, 20 of 20 | 20 of 20 |
| `muse_spark` | -28 % | -32 % .. -16 % | -28 % | -22 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | - (no such class in the library) | - | - |
| `nemotron_omni` | -34 % | -47 % .. -2 % | -34 % | -40 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | - (its processor is checkpoint code) | - | - |
| `phi3_v` | -48 % | -58 % .. -32 % | -48 % | -44 % | 19 / 0 / 0 | 20 of 20 | 236 of 236 | - (its processor is checkpoint code) | - | - |
| `qwen3_omni` | -13 % | -30 % .. +1 % | -13 % | 0 % | 13 / 6 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 3.76x (1.85x .. 5.52x) | 20 of 20, 20 of 20 | 20 of 20 |
| `qwen3_vl` | -12 % | -31 % .. 0 % | -11 % | +1 % | 12 / 7 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 3.91x (1.83x .. 5.65x) | 20 of 20, 20 of 20 | 20 of 20 |
| `qwen3_vl` (second config) | -13 % | -30 % .. 0 % | -12 % | -3 % | 11 / 8 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 3.73x (1.74x .. 5.69x) | 20 of 20, 20 of 20 | 20 of 20 |
| `qwen_vl` | -41 % | -50 % .. -22 % | -41 % | -36 % | 19 / 0 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 2.98x (1.68x .. 4.08x) | 20 of 20, 20 of 20 | 20 of 20 |
| `qwen_vl` (second config) | -43 % | -52 % .. -23 % | -43 % | -35 % | 19 / 0 / 0 | 20 of 20 | 224 of 224 (12 refused on both) | 3.09x (1.68x .. 4.39x) | 20 of 20, 20 of 20 | 20 of 20 |
| `step3_vl` | -4 % | -16 % .. 0 % | -4 % | -2 % | 16 / 3 / 0 | 20 of 20 | 228 of 228 (8 refused on both) | - (no such class in the library) | - | - |
| `step3_vl` (model type `step5_vl`) | -4 % | -14 % .. +3 % | -4 % | -4 % | 14 / 4 / 1 | 20 of 20 | 228 of 228 (8 refused on both) | - (no such class in the library) | - | - |

## `deepseek_v41`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 1.49 | -22 % | 1.47 | -21 % | yes |
| JPEG 1024x768 | 5.86 | -17 % | 5.81 | -17 % | yes |
| JPEG 1920x1080 | 14.5 | -16 % | 14.4 | -16 % | yes |
| JPEG 4096x2304 | 40.2 | -6 % | 39.9 | -6 % | yes |
| JPEG 8192x8192 | 313 | -1 % | 311 | -1 % | yes |
| JPEG progressive | 8.92 | -8 % | 8.84 | -8 % | yes |
| JPEG CMYK | 10.4 | -9 % | 10.3 | -9 % | yes |
| JPEG greyscale | 5.44 | -13 % | 5.39 | -13 % | yes |
| JPEG EXIF-rotated | 2.96 | -22 % | 2.93 | -22 % | yes |
| PNG 256x256 | 1.45 | -11 % | 1.43 | -11 % | yes |
| PNG 1024x768 | 7.89 | -18 % | 7.83 | -18 % | yes |
| PNG 1920x1080 | 19.7 | -18 % | 19.6 | -18 % | yes |
| PNG 3840x2160 | 60.1 | -15 % | 59.7 | -15 % | yes |
| PNG RGBA (alpha) | 10.5 | -21 % | 10.4 | -21 % | yes |
| WebP lossy | 7.37 | -10 % | 7.29 | -10 % | yes |
| WebP lossless | 18.1 | -4 % | 18.0 | -4 % | yes |
| GIF | 10.3 | -19 % | 10.2 | -19 % | yes |
| BMP | 5.89 | -15 % | 5.83 | -15 % | yes |
| TIFF | 5.28 | -13 % | 5.23 | -13 % | yes |
| batch of ten 1024x768 JPEGs | 5.66 | -44 % | 5.61 | -44 % | yes |

## `gemma4`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 1.60 | -75 % | 1.58 | -75 % | yes | 5.44 | 3.40x | yes |
| JPEG 1024x768 | 4.76 | -59 % | 4.71 | -59 % | yes | 12.6 | 2.64x | yes |
| JPEG 1920x1080 | 9.91 | -46 % | 9.83 | -46 % | yes | 18.7 | 1.89x | yes |
| JPEG 4096x2304 | 34.7 | -31 % | 34.5 | -31 % | yes | 65.5 | 1.89x | yes |
| JPEG 8192x8192 | 309 | -13 % | 306 | -13 % | yes | 483 | 1.56x | yes |
| JPEG progressive | 7.87 | -46 % | 7.81 | -46 % | yes | 13.9 | 1.77x | yes |
| JPEG CMYK | 9.30 | -42 % | 9.24 | -42 % | yes | 16.6 | 1.79x | yes |
| JPEG greyscale | 4.49 | -58 % | 4.44 | -58 % | yes | 9.10 | 2.02x | yes |
| JPEG EXIF-rotated | 3.20 | -66 % | 3.17 | -66 % | yes | 7.43 | 2.32x | yes |
| PNG 256x256 | 1.54 | -74 % | 1.52 | -74 % | yes | 7.57 | 4.91x | yes |
| PNG 1024x768 | 7.03 | -51 % | 6.97 | -50 % | yes | 19.6 | 2.79x | yes |
| PNG 1920x1080 | 15.2 | -42 % | 15.0 | -42 % | yes | 41.9 | 2.76x | yes |
| PNG 3840x2160 | 54.6 | -30 % | 54.2 | -30 % | yes | 149 | 2.73x | yes |
| PNG RGBA (alpha) | 9.49 | -47 % | 9.41 | -47 % | yes | 26.1 | 2.75x | yes |
| WebP lossy | 6.38 | -50 % | 6.33 | -50 % | yes | 11.3 | 1.77x | yes |
| WebP lossless | 17.2 | -28 % | 17.1 | -28 % | yes | 23.6 | 1.37x | yes |
| GIF | 9.45 | -45 % | 9.37 | -45 % | yes | 11.9 | 1.26x | yes |
| BMP | 4.88 | -58 % | 4.82 | -58 % | yes | 8.11 | 1.66x | yes |
| TIFF | 4.27 | -59 % | 4.23 | -60 % | yes | 7.95 | 1.86x | yes |
| batch of ten 1024x768 JPEGs | 4.55 | -67 % | 4.52 | -67 % | yes | 11.9 | 2.62x | yes |

## `glm53_flash`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.62 | -55 % | 0.61 | -55 % | yes | 1.88 | 3.03x | yes |
| JPEG 1024x768 | 4.34 | -57 % | 4.31 | -57 % | yes | 16.2 | 3.74x | yes |
| JPEG 1920x1080 | 11.7 | -56 % | 11.6 | -56 % | yes | 55.9 | 4.79x | yes |
| JPEG 4096x2304 | 97.0 | -50 % | 96.4 | -50 % | yes | 227 | 2.34x | yes |
| JPEG 8192x8192 | 390 | -29 % | 386 | -29 % | yes | 712 | 1.82x | yes |
| JPEG progressive | 7.41 | -45 % | 7.31 | -45 % | yes | 16.2 | 2.18x | yes |
| JPEG CMYK | 8.83 | -40 % | 8.76 | -40 % | yes | 19.6 | 2.22x | yes |
| JPEG greyscale | 3.85 | -60 % | 3.81 | -60 % | yes | 14.6 | 3.80x | yes |
| JPEG EXIF-rotated | 2.28 | -54 % | 2.25 | -55 % | yes | 5.46 | 2.40x | yes |
| PNG 256x256 | 0.68 | -52 % | 0.67 | -53 % | yes | 2.99 | 4.37x | yes |
| PNG 1024x768 | 6.69 | -51 % | 6.63 | -51 % | yes | 24.5 | 3.66x | yes |
| PNG 1920x1080 | 17.1 | -52 % | 17.0 | -51 % | yes | 73.3 | 4.30x | yes |
| PNG 3840x2160 | 117 | -47 % | 116 | -47 % | yes | 316 | 2.71x | yes |
| PNG RGBA (alpha) | 9.04 | -47 % | 8.96 | -47 % | yes | 29.5 | 3.26x | yes |
| WebP lossy | 5.82 | -50 % | 5.77 | -50 % | yes | 16.2 | 2.78x | yes |
| WebP lossless | 16.7 | -26 % | 16.6 | -26 % | yes | 27.0 | 1.62x | yes |
| GIF | 8.66 | -47 % | 8.59 | -47 % | yes | 16.0 | 1.84x | yes |
| BMP | 4.20 | -60 % | 4.17 | -60 % | yes | 11.3 | 2.68x | yes |
| TIFF | 3.67 | -62 % | 3.63 | -62 % | yes | 14.0 | 3.81x | yes |
| batch of ten 1024x768 JPEGs | 8.87 | -45 % | 8.81 | -45 % | yes | 20.0 | 2.26x | yes |

## `kimi_k25`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.58 | -36 % | 0.57 | -37 % | yes | 1.57 | 2.68x | pixels yes, float32 no |
| JPEG 1024x768 | 3.98 | -42 % | 3.94 | -42 % | yes | 10.9 | 2.74x | pixels yes, float32 no |
| JPEG 1920x1080 | 10.5 | -47 % | 10.4 | -47 % | yes | 29.5 | 2.80x | pixels yes, float32 no |
| JPEG 4096x2304 | 45.4 | -37 % | 45.1 | -37 % | yes | 134 | 2.94x | pixels yes, float32 no |
| JPEG 8192x8192 | 318 | -8 % | 316 | -8 % | yes | 579 | 1.82x | pixels yes, float32 no |
| JPEG progressive | 7.12 | -24 % | 7.07 | -24 % | yes | 12.9 | 1.81x | pixels yes, float32 no |
| JPEG CMYK | 8.55 | -20 % | 8.47 | -20 % | yes | 17.3 | 2.02x | pixels yes, float32 no |
| JPEG greyscale | 3.58 | -37 % | 3.55 | -37 % | yes | 8.69 | 2.43x | pixels yes, float32 no |
| JPEG EXIF-rotated | 2.21 | -35 % | 2.19 | -35 % | yes | 4.31 | 1.95x | pixels yes, float32 no |
| PNG 256x256 | 0.65 | -31 % | 0.64 | -30 % | yes | 2.58 | 3.98x | pixels yes, float32 no |
| PNG 1024x768 | 6.11 | -35 % | 6.06 | -35 % | yes | 21.2 | 3.47x | pixels yes, float32 no |
| PNG 1920x1080 | 15.8 | -42 % | 15.6 | -42 % | yes | 54.8 | 3.48x | pixels yes, float32 no |
| PNG 3840x2160 | 64.1 | -36 % | 63.7 | -36 % | yes | 219 | 3.42x | pixels yes, float32 no |
| PNG RGBA (alpha) | 8.72 | -33 % | 8.66 | -33 % | yes | 27.3 | 3.13x | pixels yes, float32 no |
| WebP lossy | 5.54 | -26 % | 5.50 | -26 % | yes | 14.2 | 2.56x | pixels yes, float32 no |
| WebP lossless | 16.4 | -11 % | 16.3 | -11 % | yes | 24.2 | 1.48x | pixels yes, float32 no |
| GIF | 8.47 | -30 % | 8.41 | -30 % | yes | 12.5 | 1.48x | pixels yes, float32 no |
| BMP | 3.94 | -40 % | 3.90 | -40 % | yes | 8.61 | 2.19x | pixels yes, float32 no |
| TIFF | 3.27 | -44 % | 3.24 | -44 % | yes | 8.34 | 2.55x | pixels yes, float32 no |
| batch of ten 1024x768 JPEGs | 3.86 | -62 % | 3.83 | -62 % | yes | 14.4 | 3.73x | pixels yes, float32 no |

## `kimi_k3`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 0.60 | -34 % | 0.59 | -35 % | yes |
| JPEG 1024x768 | 3.95 | -43 % | 3.91 | -43 % | yes |
| JPEG 1920x1080 | 10.5 | -47 % | 10.4 | -47 % | yes |
| JPEG 4096x2304 | 46.2 | -61 % | 45.9 | -61 % | yes |
| JPEG 8192x8192 | 408 | -21 % | 404 | -21 % | yes |
| JPEG progressive | 7.08 | -24 % | 7.03 | -24 % | yes |
| JPEG CMYK | 8.53 | -21 % | 8.47 | -21 % | yes |
| JPEG greyscale | 3.58 | -38 % | 3.55 | -39 % | yes |
| JPEG EXIF-rotated | 2.24 | -36 % | 2.22 | -36 % | yes |
| PNG 256x256 | 0.66 | -30 % | 0.65 | -31 % | yes |
| PNG 1024x768 | 6.13 | -38 % | 6.08 | -38 % | yes |
| PNG 1920x1080 | 15.8 | -41 % | 15.7 | -41 % | yes |
| PNG 3840x2160 | 62.2 | -54 % | 61.5 | -54 % | yes |
| PNG RGBA (alpha) | 8.65 | -34 % | 8.59 | -34 % | yes |
| WebP lossy | 5.54 | -30 % | 5.49 | -30 % | yes |
| WebP lossless | 16.3 | -12 % | 16.2 | -12 % | yes |
| GIF | 8.43 | -30 % | 8.36 | -30 % | yes |
| BMP | 4.03 | -40 % | 3.99 | -40 % | yes |
| TIFF | 3.38 | -43 % | 3.35 | -43 % | yes |
| batch of ten 1024x768 JPEGs | 3.85 | -61 % | 3.81 | -61 % | yes |

## `llama4`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 1.34 | -22 % | 1.32 | -22 % | no | 4.92 | 3.66x | pixels yes, float32 no |
| JPEG 1024x768 | 25.9 | -2 % | 25.6 | -2 % | no | 17.0 | 0.66x | pixels yes, float32 no |
| JPEG 1920x1080 | 39.8 | -8 % | 39.4 | -8 % | no | 35.7 | 0.90x | pixels yes, float32 no |
| JPEG 4096x2304 | 67.1 | -1 % | 66.5 | -1 % | no | 78.5 | 1.17x | pixels yes, float32 no |
| JPEG 8192x8192 | 346 | +1 % | 343 | +1 % | no | 532 | 1.54x | pixels yes, float32 no |
| JPEG progressive | 28.8 | -7 % | 28.6 | -7 % | no | 20.8 | 0.72x | pixels yes, float32 no |
| JPEG CMYK | 31.4 | 0 % | 31.0 | 0 % | no | 22.8 | 0.73x | pixels yes, float32 no |
| JPEG greyscale | 26.6 | +3 % | 26.3 | +3 % | no | 15.2 | 0.57x | pixels yes, float32 no |
| JPEG EXIF-rotated | 10.8 | -6 % | 10.8 | -6 % | no | 9.54 | 0.88x | pixels yes, float32 no |
| PNG 256x256 | 1.44 | -22 % | 1.41 | -22 % | no | 6.06 | 4.22x | pixels yes, float32 no |
| PNG 1024x768 | 29.2 | 0 % | 29.0 | 0 % | no | 25.9 | 0.89x | pixels yes, float32 no |
| PNG 1920x1080 | 45.9 | -7 % | 45.5 | -7 % | no | 54.2 | 1.18x | pixels yes, float32 no |
| PNG 3840x2160 | 86.8 | -10 % | 85.8 | -11 % | no | 165 | 1.90x | pixels yes, float32 no |
| PNG RGBA (alpha) | 31.1 | -7 % | 30.8 | -7 % | no | 32.1 | 1.03x | pixels yes, float32 no |
| WebP lossy | 27.6 | -1 % | 27.4 | -1 % | no | 18.5 | 0.67x | pixels yes, float32 no |
| WebP lossless | 37.7 | -1 % | 37.5 | -1 % | no | 30.1 | 0.80x | pixels yes, float32 no |
| GIF | 31.2 | -5 % | 31.0 | -5 % | no | 19.4 | 0.62x | pixels yes, float32 no |
| BMP | 28.5 | +7 % | 28.2 | +7 % | no | 14.0 | 0.49x | pixels yes, float32 no |
| TIFF | 25.9 | 0 % | 25.6 | 0 % | no | 13.8 | 0.53x | pixels yes, float32 no |
| batch of ten 1024x768 JPEGs | 37.4 | -4 % | 37.0 | -4 % | no | 18.2 | 0.49x | pixels yes, float32 no |

## `llava_next`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 1.96 | -37 % | 1.94 | -37 % | no | 5.67 | 2.89x | yes |
| JPEG 1024x768 | 6.46 | -17 % | 6.40 | -17 % | no | 14.1 | 2.18x | yes |
| JPEG 1920x1080 | 11.8 | -8 % | 11.7 | -8 % | no | 23.1 | 1.95x | yes |
| JPEG 4096x2304 | 40.4 | -7 % | 40.1 | -7 % | no | 70.9 | 1.76x | yes |
| JPEG 8192x8192 | 377 | +4 % | 373 | +4 % | no | 515 | 1.37x | yes |
| JPEG progressive | 9.23 | -15 % | 9.16 | -15 % | no | 14.6 | 1.58x | yes |
| JPEG CMYK | 10.8 | -12 % | 10.7 | -11 % | no | 18.8 | 1.75x | yes |
| JPEG greyscale | 5.76 | -21 % | 5.71 | -21 % | no | 11.4 | 1.97x | yes |
| JPEG EXIF-rotated | 4.43 | -32 % | 4.39 | -32 % | no | 9.66 | 2.18x | yes |
| PNG 256x256 | 1.75 | -35 % | 1.73 | -34 % | no | 5.89 | 3.37x | yes |
| PNG 1024x768 | 8.25 | -24 % | 8.19 | -23 % | no | 22.1 | 2.68x | yes |
| PNG 1920x1080 | 17.1 | -19 % | 17.0 | -19 % | no | 44.9 | 2.63x | yes |
| PNG 3840x2160 | 59.7 | -14 % | 59.1 | -14 % | no | 157 | 2.63x | yes |
| PNG RGBA (alpha) | 10.9 | -24 % | 10.8 | -24 % | no | 28.2 | 2.59x | yes |
| WebP lossy | 7.86 | -13 % | 7.80 | -13 % | no | 13.5 | 1.72x | yes |
| WebP lossless | 18.6 | -7 % | 18.5 | -7 % | no | 26.0 | 1.40x | yes |
| GIF | 10.8 | -20 % | 10.8 | -19 % | no | 13.9 | 1.28x | yes |
| BMP | 6.17 | -21 % | 6.11 | -21 % | no | 9.88 | 1.60x | yes |
| TIFF | 5.63 | -21 % | 5.57 | -21 % | no | 10.3 | 1.83x | yes |
| batch of ten 1024x768 JPEGs | 8.16 | -18 % | 8.10 | -17 % | no | 14.0 | 1.72x | yes |

## `llava`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 1.10 | -29 % | 1.07 | -29 % | no | 2.69 | 2.45x | yes |
| JPEG 1024x768 | 3.72 | -14 % | 3.69 | -14 % | no | 7.38 | 1.98x | yes |
| JPEG 1920x1080 | 8.41 | -12 % | 8.35 | -12 % | no | 17.1 | 2.03x | yes |
| JPEG 4096x2304 | 30.9 | -12 % | 30.7 | -12 % | no | 68.8 | 2.23x | yes |
| JPEG 8192x8192 | 345 | +6 % | 342 | +6 % | no | 466 | 1.35x | yes |
| JPEG progressive | 6.80 | -10 % | 6.74 | -10 % | no | 10.3 | 1.52x | yes |
| JPEG CMYK | 8.27 | -8 % | 8.21 | -8 % | no | 13.7 | 1.66x | yes |
| JPEG greyscale | 3.22 | -19 % | 3.19 | -19 % | no | 6.25 | 1.94x | yes |
| JPEG EXIF-rotated | 2.50 | -33 % | 2.48 | -33 % | no | 5.00 | 2.00x | yes |
| PNG 256x256 | 1.02 | -33 % | 1.01 | -33 % | no | 3.74 | 3.66x | yes |
| PNG 1024x768 | 5.81 | -20 % | 5.76 | -20 % | no | 17.7 | 3.05x | yes |
| PNG 1920x1080 | 13.7 | -18 % | 13.6 | -18 % | no | 39.8 | 2.90x | yes |
| PNG 3840x2160 | 51.1 | -18 % | 50.8 | -18 % | no | 156 | 3.04x | yes |
| PNG RGBA (alpha) | 8.37 | -26 % | 8.30 | -26 % | no | 23.3 | 2.78x | yes |
| WebP lossy | 5.31 | -10 % | 5.26 | -10 % | no | 9.05 | 1.70x | yes |
| WebP lossless | 16.1 | -5 % | 16.0 | -5 % | no | 20.4 | 1.26x | yes |
| GIF | 8.33 | -19 % | 8.26 | -18 % | no | 9.16 | 1.10x | yes |
| BMP | 3.81 | -17 % | 3.77 | -17 % | no | 5.37 | 1.41x | yes |
| TIFF | 3.20 | -17 % | 3.18 | -17 % | no | 5.65 | 1.76x | yes |
| batch of ten 1024x768 JPEGs | 3.67 | -9 % | 3.65 | -9 % | no | 6.44 | 1.75x | yes |

## `llava` with the `pixtral` processor, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.48 | -32 % | 0.47 | -33 % | yes | 1.80 | 3.73x | yes |
| JPEG 1024x768 | 3.12 | -30 % | 3.09 | -30 % | yes | 17.1 | 5.47x | yes |
| JPEG 1920x1080 | 9.62 | -41 % | 9.54 | -41 % | yes | 22.0 | 2.29x | yes |
| JPEG 4096x2304 | 34.1 | -29 % | 33.9 | -29 % | yes | 70.5 | 2.07x | yes |
| JPEG 8192x8192 | 305 | -16 % | 303 | -16 % | yes | 463 | 1.52x | yes |
| JPEG progressive | 6.20 | -20 % | 6.15 | -20 % | yes | 17.0 | 2.74x | yes |
| JPEG CMYK | 7.64 | -17 % | 7.58 | -17 % | yes | 22.2 | 2.91x | yes |
| JPEG greyscale | 2.64 | -36 % | 2.61 | -36 % | yes | 12.3 | 4.67x | yes |
| JPEG EXIF-rotated | 1.83 | -35 % | 1.81 | -35 % | yes | 6.44 | 3.53x | yes |
| PNG 256x256 | 0.57 | -28 % | 0.56 | -28 % | yes | 2.98 | 5.24x | yes |
| PNG 1024x768 | 5.33 | -28 % | 5.29 | -28 % | yes | 23.6 | 4.43x | yes |
| PNG 1920x1080 | 15.0 | -38 % | 14.8 | -38 % | yes | 46.7 | 3.12x | yes |
| PNG 3840x2160 | 54.2 | -28 % | 53.8 | -28 % | yes | 156 | 2.88x | yes |
| PNG RGBA (alpha) | 7.80 | -31 % | 7.75 | -31 % | yes | 29.9 | 3.83x | yes |
| WebP lossy | 4.67 | -24 % | 4.63 | -24 % | yes | 15.5 | 3.31x | yes |
| WebP lossless | 15.5 | -9 % | 15.4 | -9 % | yes | 29.1 | 1.88x | yes |
| GIF | 7.61 | -27 % | 7.54 | -27 % | yes | 15.5 | 2.04x | yes |
| BMP | 3.10 | -35 % | 3.07 | -35 % | yes | 11.7 | 3.78x | yes |
| TIFF | 2.26 | -37 % | 2.23 | -38 % | yes | 11.7 | 5.16x | yes |
| batch of ten 1024x768 JPEGs | 3.46 | -25 % | 3.43 | -25 % | yes | 16.3 | 4.73x | yes |

## `minimax_m3`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.77 | -49 % | 0.76 | -50 % | yes | 2.31 | 2.99x | yes |
| JPEG 1024x768 | 5.08 | -41 % | 5.04 | -41 % | yes | 12.8 | 2.53x | yes |
| JPEG 1920x1080 | 10.5 | -32 % | 10.4 | -33 % | yes | 21.7 | 2.07x | yes |
| JPEG 4096x2304 | 35.0 | -25 % | 34.8 | -25 % | yes | 67.0 | 1.91x | yes |
| JPEG 8192x8192 | 312 | -11 % | 310 | -11 % | yes | 456 | 1.46x | yes |
| JPEG progressive | 8.29 | -29 % | 8.15 | -29 % | yes | 15.4 | 1.86x | yes |
| JPEG CMYK | 9.73 | -27 % | 9.66 | -27 % | yes | 18.4 | 1.89x | yes |
| JPEG greyscale | 4.77 | -42 % | 4.74 | -42 % | yes | 11.3 | 2.37x | yes |
| JPEG EXIF-rotated | 3.04 | -48 % | 3.02 | -47 % | yes | 7.71 | 2.53x | yes |
| PNG 256x256 | 0.83 | -46 % | 0.82 | -46 % | yes | 3.14 | 3.78x | yes |
| PNG 1024x768 | 7.25 | -40 % | 7.19 | -40 % | yes | 22.1 | 3.05x | yes |
| PNG 1920x1080 | 15.8 | -33 % | 15.7 | -32 % | yes | 45.2 | 2.87x | yes |
| PNG 3840x2160 | 54.8 | -27 % | 54.5 | -27 % | yes | 153 | 2.79x | yes |
| PNG RGBA (alpha) | 9.93 | -36 % | 9.85 | -36 % | yes | 28.3 | 2.85x | yes |
| WebP lossy | 6.70 | -36 % | 6.64 | -36 % | yes | 15.0 | 2.25x | yes |
| WebP lossless | 17.5 | -17 % | 17.3 | -17 % | yes | 24.9 | 1.43x | yes |
| GIF | 9.70 | -34 % | 9.62 | -34 % | yes | 13.3 | 1.37x | yes |
| BMP | 4.94 | -46 % | 4.89 | -46 % | yes | 9.63 | 1.95x | yes |
| TIFF | 4.57 | -45 % | 4.50 | -45 % | yes | 9.58 | 2.10x | yes |
| batch of ten 1024x768 JPEGs | 4.54 | -47 % | 4.51 | -47 % | yes | 15.5 | 3.42x | yes |

## `muse_spark`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 1.28 | -31 % | 1.27 | -31 % | yes |
| JPEG 1024x768 | 10.1 | -28 % | 10.0 | -28 % | yes |
| JPEG 1920x1080 | 27.8 | -26 % | 27.5 | -26 % | yes |
| JPEG 4096x2304 | 64.5 | -29 % | 64.0 | -29 % | yes |
| JPEG 8192x8192 | 349 | -20 % | 347 | -20 % | yes |
| JPEG progressive | 13.0 | -25 % | 12.9 | -25 % | yes |
| JPEG CMYK | 14.5 | -24 % | 14.4 | -24 % | yes |
| JPEG greyscale | 9.70 | -29 % | 9.62 | -29 % | yes |
| JPEG EXIF-rotated | 4.96 | -29 % | 4.91 | -29 % | yes |
| PNG 256x256 | 1.35 | -32 % | 1.33 | -32 % | yes |
| PNG 1024x768 | 12.0 | -29 % | 11.9 | -29 % | yes |
| PNG 1920x1080 | 32.7 | -28 % | 32.4 | -28 % | yes |
| PNG 3840x2160 | 85.1 | -27 % | 84.2 | -28 % | yes |
| PNG RGBA (alpha) | 15.0 | -29 % | 14.9 | -29 % | yes |
| WebP lossy | 11.7 | -25 % | 11.5 | -26 % | yes |
| WebP lossless | 22.4 | -16 % | 22.2 | -16 % | yes |
| GIF | 14.6 | -27 % | 14.5 | -27 % | yes |
| BMP | 10.1 | -31 % | 10.0 | -31 % | yes |
| TIFF | 9.87 | -28 % | 9.77 | -28 % | yes |
| batch of ten 1024x768 JPEGs | 13.9 | -22 % | 13.7 | -22 % | yes |

## `nemotron_omni`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 2.77 | -42 % | 2.70 | -43 % | yes |
| JPEG 1024x768 | 12.3 | -33 % | 12.2 | -33 % | yes |
| JPEG 1920x1080 | 32.8 | -34 % | 32.6 | -34 % | yes |
| JPEG 4096x2304 | 152 | -15 % | 151 | -15 % | yes |
| JPEG 8192x8192 | 786 | -2 % | 781 | -2 % | yes |
| JPEG progressive | 14.7 | -31 % | 14.6 | -32 % | yes |
| JPEG CMYK | 16.1 | -30 % | 16.0 | -30 % | yes |
| JPEG greyscale | 11.2 | -38 % | 11.1 | -38 % | yes |
| JPEG EXIF-rotated | 4.90 | -39 % | 4.85 | -39 % | yes |
| PNG 256x256 | 2.50 | -47 % | 2.47 | -47 % | yes |
| PNG 1024x768 | 14.3 | -34 % | 14.2 | -34 % | yes |
| PNG 1920x1080 | 37.8 | -34 % | 37.5 | -34 % | yes |
| PNG 3840x2160 | 167 | -17 % | 165 | -17 % | yes |
| PNG RGBA (alpha) | 16.3 | -36 % | 16.2 | -36 % | yes |
| WebP lossy | 13.4 | -33 % | 13.3 | -33 % | yes |
| WebP lossless | 23.9 | -23 % | 23.8 | -22 % | yes |
| GIF | 16.2 | -34 % | 16.1 | -34 % | yes |
| BMP | 11.9 | -37 % | 11.8 | -37 % | yes |
| TIFF | 11.3 | -38 % | 11.2 | -38 % | yes |
| batch of ten 1024x768 JPEGs | 14.3 | -40 % | 14.2 | -40 % | yes |

## `phi3_v`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 6.01 | -46 % | 5.96 | -45 % | yes |
| JPEG 1024x768 | 7.98 | -54 % | 7.92 | -54 % | yes |
| JPEG 1920x1080 | 12.8 | -43 % | 12.7 | -43 % | yes |
| JPEG 4096x2304 | 35.9 | -35 % | 35.7 | -35 % | yes |
| JPEG 8192x8192 | 298 | -54 % | 296 | -54 % | yes |
| JPEG progressive | 11.1 | -48 % | 10.9 | -48 % | yes |
| JPEG CMYK | 12.4 | -44 % | 12.3 | -44 % | yes |
| JPEG greyscale | 7.37 | -56 % | 7.31 | -56 % | yes |
| JPEG EXIF-rotated | 8.81 | -58 % | 8.74 | -58 % | yes |
| PNG 256x256 | 5.28 | -52 % | 5.23 | -52 % | yes |
| PNG 1024x768 | 10.2 | -50 % | 10.1 | -50 % | yes |
| PNG 1920x1080 | 18.1 | -41 % | 18.0 | -41 % | yes |
| PNG 3840x2160 | 55.8 | -32 % | 55.4 | -32 % | yes |
| PNG RGBA (alpha) | 12.9 | -46 % | 12.8 | -46 % | yes |
| WebP lossy | 9.65 | -50 % | 9.57 | -50 % | yes |
| WebP lossless | 20.0 | -33 % | 19.9 | -33 % | yes |
| GIF | 12.6 | -46 % | 12.5 | -46 % | yes |
| BMP | 7.86 | -55 % | 7.79 | -55 % | yes |
| TIFF | 7.37 | -56 % | 7.30 | -56 % | yes |
| batch of ten 1024x768 JPEGs | 9.73 | -44 % | 9.65 | -44 % | yes |

## `qwen3_omni`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.55 | -30 % | 0.54 | -30 % | yes | 2.07 | 3.76x | yes |
| JPEG 1024x768 | 3.77 | +1 % | 3.73 | +1 % | yes | 20.8 | 5.52x | yes |
| JPEG 1920x1080 | 11.6 | -16 % | 11.5 | -16 % | yes | 57.4 | 4.96x | yes |
| JPEG 4096x2304 | 79.5 | 0 % | 78.8 | 0 % | yes | 303 | 3.81x | yes |
| JPEG 8192x8192 | 439 | -22 % | 436 | -22 % | yes | 893 | 2.03x | yes |
| JPEG progressive | 6.93 | -1 % | 6.87 | 0 % | yes | 19.0 | 2.74x | yes |
| JPEG CMYK | 8.34 | -1 % | 8.28 | -1 % | yes | 22.2 | 2.66x | yes |
| JPEG greyscale | 3.27 | -5 % | 3.24 | -5 % | yes | 15.5 | 4.74x | yes |
| JPEG EXIF-rotated | 2.06 | -27 % | 2.04 | -27 % | yes | 7.46 | 3.62x | yes |
| PNG 256x256 | 0.63 | -25 % | 0.61 | -25 % | yes | 3.08 | 4.92x | yes |
| PNG 1024x768 | 5.92 | -13 % | 5.88 | -13 % | yes | 25.0 | 4.22x | yes |
| PNG 1920x1080 | 16.7 | -22 % | 16.6 | -22 % | yes | 83.4 | 4.99x | yes |
| PNG 3840x2160 | 100 | -13 % | 99.4 | -13 % | yes | 372 | 3.72x | yes |
| PNG RGBA (alpha) | 8.56 | -21 % | 8.49 | -21 % | yes | 32.6 | 3.81x | yes |
| WebP lossy | 5.35 | -2 % | 5.31 | -2 % | yes | 18.0 | 3.36x | yes |
| WebP lossless | 16.3 | 0 % | 16.2 | 0 % | yes | 30.1 | 1.85x | yes |
| GIF | 8.42 | -14 % | 8.31 | -15 % | yes | 18.8 | 2.24x | yes |
| BMP | 3.76 | -7 % | 3.72 | -7 % | yes | 13.4 | 3.56x | yes |
| TIFF | 3.26 | -1 % | 3.22 | -1 % | yes | 16.8 | 5.16x | yes |
| batch of ten 1024x768 JPEGs | 6.67 | 0 % | 6.61 | 0 % | yes | 22.4 | 3.35x | yes |

## `qwen3_vl`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.51 | -31 % | 0.50 | -31 % | yes | 2.42 | 4.73x | yes |
| JPEG 1024x768 | 3.72 | 0 % | 3.69 | +1 % | yes | 19.3 | 5.18x | yes |
| JPEG 1920x1080 | 11.2 | -17 % | 11.1 | -17 % | yes | 63.1 | 5.65x | yes |
| JPEG 4096x2304 | 76.1 | -2 % | 75.6 | -2 % | yes | 319 | 4.20x | yes |
| JPEG 8192x8192 | 451 | -26 % | 448 | -26 % | yes | 1011 | 2.24x | yes |
| JPEG progressive | 6.89 | -1 % | 6.83 | -1 % | yes | 19.2 | 2.78x | yes |
| JPEG CMYK | 8.21 | 0 % | 8.15 | 0 % | yes | 23.1 | 2.81x | yes |
| JPEG greyscale | 3.29 | -2 % | 3.26 | -1 % | yes | 15.3 | 4.65x | yes |
| JPEG EXIF-rotated | 2.03 | -20 % | 2.01 | -21 % | yes | 7.31 | 3.60x | yes |
| PNG 256x256 | 0.61 | -27 % | 0.59 | -27 % | yes | 2.90 | 4.78x | yes |
| PNG 1024x768 | 5.94 | -12 % | 5.89 | -11 % | yes | 25.6 | 4.32x | yes |
| PNG 1920x1080 | 16.2 | -23 % | 16.1 | -23 % | yes | 85.6 | 5.30x | yes |
| PNG 3840x2160 | 98.9 | -14 % | 98.2 | -14 % | yes | 387 | 3.91x | yes |
| PNG RGBA (alpha) | 8.48 | -19 % | 8.37 | -19 % | yes | 32.2 | 3.79x | yes |
| WebP lossy | 5.32 | -1 % | 5.27 | -1 % | yes | 19.3 | 3.63x | yes |
| WebP lossless | 16.1 | 0 % | 16.0 | 0 % | yes | 29.5 | 1.83x | yes |
| GIF | 8.15 | -15 % | 8.09 | -16 % | yes | 17.3 | 2.12x | yes |
| BMP | 3.72 | -7 % | 3.69 | -7 % | yes | 13.1 | 3.51x | yes |
| TIFF | 3.06 | -2 % | 3.03 | -2 % | yes | 15.4 | 5.03x | yes |
| batch of ten 1024x768 JPEGs | 6.54 | +1 % | 6.49 | +1 % | yes | 23.6 | 3.61x | yes |

## `qwen3_vl` (second config), milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.51 | -30 % | 0.50 | -32 % | yes | 1.83 | 3.59x | yes |
| JPEG 1024x768 | 3.69 | -1 % | 3.65 | -1 % | yes | 14.8 | 4.02x | yes |
| JPEG 1920x1080 | 11.0 | -19 % | 10.9 | -19 % | yes | 62.5 | 5.69x | yes |
| JPEG 4096x2304 | 76.5 | -1 % | 76.0 | -1 % | yes | 308 | 4.03x | yes |
| JPEG 8192x8192 | 446 | -26 % | 443 | -26 % | yes | 987 | 2.21x | yes |
| JPEG progressive | 6.91 | 0 % | 6.86 | 0 % | yes | 18.9 | 2.73x | yes |
| JPEG CMYK | 8.24 | -1 % | 8.18 | -1 % | yes | 23.6 | 2.87x | yes |
| JPEG greyscale | 3.29 | -2 % | 3.26 | -2 % | yes | 14.5 | 4.39x | yes |
| JPEG EXIF-rotated | 2.00 | -26 % | 1.98 | -26 % | yes | 7.39 | 3.69x | yes |
| PNG 256x256 | 0.61 | -27 % | 0.59 | -28 % | yes | 3.32 | 5.47x | yes |
| PNG 1024x768 | 5.85 | -13 % | 5.81 | -12 % | yes | 25.3 | 4.32x | yes |
| PNG 1920x1080 | 16.3 | -23 % | 16.1 | -23 % | yes | 78.4 | 4.82x | yes |
| PNG 3840x2160 | 98.9 | -17 % | 97.6 | -17 % | yes | 359 | 3.63x | yes |
| PNG RGBA (alpha) | 8.50 | -20 % | 8.44 | -20 % | yes | 32.7 | 3.85x | yes |
| WebP lossy | 5.30 | -1 % | 5.26 | -1 % | yes | 19.8 | 3.73x | yes |
| WebP lossless | 16.1 | -1 % | 16.0 | -1 % | yes | 28.0 | 1.74x | yes |
| GIF | 8.24 | -15 % | 8.18 | -15 % | yes | 17.3 | 2.11x | yes |
| BMP | 3.83 | -5 % | 3.79 | -5 % | yes | 12.7 | 3.31x | yes |
| TIFF | 3.11 | 0 % | 3.08 | +1 % | yes | 15.3 | 4.93x | yes |
| batch of ten 1024x768 JPEGs | 6.38 | -3 % | 6.33 | -3 % | yes | 21.8 | 3.42x | yes |

## `qwen_vl`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.75 | -44 % | 0.73 | -44 % | yes | 2.23 | 2.98x | yes |
| JPEG 1024x768 | 5.67 | -45 % | 5.61 | -45 % | yes | 17.9 | 3.15x | yes |
| JPEG 1920x1080 | 15.4 | -45 % | 15.3 | -45 % | yes | 60.5 | 3.93x | yes |
| JPEG 4096x2304 | 111 | -33 % | 111 | -33 % | yes | 330 | 2.96x | yes |
| JPEG 8192x8192 | 443 | -23 % | 440 | -23 % | yes | 900 | 2.03x | yes |
| JPEG progressive | 8.77 | -39 % | 8.69 | -39 % | yes | 21.4 | 2.44x | yes |
| JPEG CMYK | 10.4 | -33 % | 10.3 | -33 % | yes | 23.8 | 2.29x | yes |
| JPEG greyscale | 5.39 | -48 % | 5.34 | -48 % | yes | 16.7 | 3.09x | yes |
| JPEG EXIF-rotated | 2.92 | -45 % | 2.89 | -45 % | yes | 8.07 | 2.76x | yes |
| PNG 256x256 | 0.82 | -40 % | 0.81 | -41 % | yes | 3.36 | 4.08x | yes |
| PNG 1024x768 | 7.93 | -43 % | 7.85 | -43 % | yes | 29.1 | 3.67x | yes |
| PNG 1920x1080 | 20.8 | -41 % | 20.6 | -41 % | yes | 84.8 | 4.08x | yes |
| PNG 3840x2160 | 124 | -29 % | 122 | -30 % | yes | 389 | 3.14x | yes |
| PNG RGBA (alpha) | 10.8 | -38 % | 10.7 | -38 % | yes | 34.5 | 3.20x | yes |
| WebP lossy | 7.35 | -41 % | 7.29 | -41 % | yes | 20.1 | 2.73x | yes |
| WebP lossless | 18.2 | -22 % | 18.1 | -22 % | yes | 30.6 | 1.68x | yes |
| GIF | 10.2 | -40 % | 10.1 | -40 % | yes | 19.1 | 1.87x | yes |
| BMP | 5.62 | -50 % | 5.56 | -50 % | yes | 16.1 | 2.86x | yes |
| TIFF | 4.95 | -50 % | 4.91 | -50 % | yes | 16.2 | 3.27x | yes |
| batch of ten 1024x768 JPEGs | 9.41 | -36 % | 9.34 | -36 % | yes | 24.6 | 2.61x | yes |

## `qwen_vl` (second config), milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 | reference ms per image | reference over head | the reference's bytes |
|---|---|---|---|---|---|---|---|---|
| JPEG 256x256 | 0.72 | -45 % | 0.71 | -45 % | yes | 2.58 | 3.57x | yes |
| JPEG 1024x768 | 5.81 | -47 % | 5.76 | -47 % | yes | 18.3 | 3.16x | yes |
| JPEG 1920x1080 | 15.9 | -44 % | 15.8 | -44 % | yes | 64.9 | 4.09x | yes |
| JPEG 4096x2304 | 112 | -32 % | 111 | -33 % | yes | 326 | 2.92x | yes |
| JPEG 8192x8192 | 454 | -23 % | 451 | -23 % | yes | 906 | 2.00x | yes |
| JPEG progressive | 8.70 | -38 % | 8.63 | -38 % | yes | 20.9 | 2.41x | yes |
| JPEG CMYK | 10.3 | -35 % | 10.2 | -34 % | yes | 24.2 | 2.35x | yes |
| JPEG greyscale | 5.30 | -50 % | 5.25 | -50 % | yes | 16.6 | 3.14x | yes |
| JPEG EXIF-rotated | 2.99 | -48 % | 2.96 | -48 % | yes | 8.18 | 2.74x | yes |
| PNG 256x256 | 0.82 | -43 % | 0.80 | -43 % | yes | 3.59 | 4.39x | yes |
| PNG 1024x768 | 7.89 | -43 % | 7.77 | -43 % | yes | 27.6 | 3.50x | yes |
| PNG 1920x1080 | 20.8 | -42 % | 20.6 | -42 % | yes | 83.9 | 4.04x | yes |
| PNG 3840x2160 | 125 | -31 % | 123 | -32 % | yes | 392 | 3.15x | yes |
| PNG RGBA (alpha) | 10.9 | -39 % | 10.9 | -39 % | yes | 33.9 | 3.10x | yes |
| WebP lossy | 7.16 | -43 % | 7.10 | -43 % | yes | 19.7 | 2.75x | yes |
| WebP lossless | 18.1 | -23 % | 17.9 | -23 % | yes | 30.3 | 1.68x | yes |
| GIF | 10.3 | -39 % | 10.2 | -39 % | yes | 19.4 | 1.88x | yes |
| BMP | 5.72 | -49 % | 5.66 | -49 % | yes | 16.0 | 2.80x | yes |
| TIFF | 5.11 | -52 % | 5.07 | -52 % | yes | 15.8 | 3.09x | yes |
| batch of ten 1024x768 JPEGs | 9.28 | -35 % | 9.22 | -35 % | yes | 24.6 | 2.65x | yes |

## `step3_vl`, milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 3.59 | -2 % | 3.55 | -2 % | yes |
| JPEG 1024x768 | 10.9 | -6 % | 10.8 | -6 % | yes |
| JPEG 1920x1080 | 22.9 | -4 % | 22.7 | -4 % | yes |
| JPEG 4096x2304 | 93.3 | -3 % | 92.4 | -3 % | yes |
| JPEG 8192x8192 | 397 | -2 % | 394 | -2 % | yes |
| JPEG progressive | 13.5 | -3 % | 13.4 | -3 % | yes |
| JPEG CMYK | 15.1 | -4 % | 14.9 | -4 % | yes |
| JPEG greyscale | 10.2 | -7 % | 10.1 | -6 % | yes |
| JPEG EXIF-rotated | 5.33 | -12 % | 5.28 | -13 % | yes |
| PNG 256x256 | 3.06 | 0 % | 3.02 | 0 % | yes |
| PNG 1024x768 | 12.8 | -7 % | 12.7 | -7 % | yes |
| PNG 1920x1080 | 27.8 | -12 % | 27.6 | -12 % | yes |
| PNG 3840x2160 | 111 | -12 % | 110 | -12 % | yes |
| PNG RGBA (alpha) | 15.0 | -16 % | 14.9 | -16 % | yes |
| WebP lossy | 12.1 | -4 % | 12.0 | -4 % | yes |
| WebP lossless | 22.9 | -1 % | 22.7 | -1 % | yes |
| GIF | 14.9 | -13 % | 14.8 | -12 % | yes |
| BMP | 10.9 | -3 % | 10.8 | -3 % | yes |
| TIFF | 10.2 | -4 % | 10.1 | -4 % | yes |
| batch of ten 1024x768 JPEGs | 17.4 | -2 % | 17.3 | -2 % | yes |

## `step3_vl` (model type `step5_vl`), milliseconds per image

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per image | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| JPEG 256x256 | 3.76 | +3 % | 3.72 | +3 % | yes |
| JPEG 1024x768 | 10.9 | -5 % | 10.8 | -5 % | yes |
| JPEG 1920x1080 | 23.0 | -2 % | 22.8 | -2 % | yes |
| JPEG 4096x2304 | 91.7 | -3 % | 90.6 | -4 % | yes |
| JPEG 8192x8192 | 397 | -1 % | 394 | -2 % | yes |
| JPEG progressive | 13.8 | -1 % | 13.7 | 0 % | yes |
| JPEG CMYK | 15.3 | -3 % | 15.2 | -3 % | yes |
| JPEG greyscale | 10.5 | -3 % | 10.4 | -3 % | yes |
| JPEG EXIF-rotated | 5.33 | -14 % | 5.27 | -14 % | yes |
| PNG 256x256 | 3.00 | -5 % | 2.96 | -5 % | yes |
| PNG 1024x768 | 12.8 | -6 % | 12.7 | -6 % | yes |
| PNG 1920x1080 | 28.2 | -10 % | 28.0 | -10 % | yes |
| PNG 3840x2160 | 113 | -10 % | 112 | -10 % | yes |
| PNG RGBA (alpha) | 15.3 | -13 % | 15.1 | -13 % | yes |
| WebP lossy | 12.2 | -5 % | 12.1 | -5 % | yes |
| WebP lossless | 23.0 | -2 % | 22.9 | -2 % | yes |
| GIF | 15.3 | -10 % | 15.1 | -10 % | yes |
| BMP | 10.7 | -4 % | 10.6 | -4 % | yes |
| TIFF | 10.5 | 0 % | 10.4 | 0 % | yes |
| batch of ten 1024x768 JPEGs | 16.9 | -4 % | 16.7 | -4 % | yes |
