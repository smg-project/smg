# The multimodal release comparison: 1.16.0 against 1.15.0

38 clip cells (2 processor configurations by 19 clip classes), time per clip on one core: 7 faster by more than 2 %, 25 level (within 2 %), 6 slower by more than 2 %. Medians of 15 interleaved rounds; change = the head's time over the base's, minus one, so a negative figure is a faster head.

1.16.0 is `322350993` built with cargo 1.100.0 (3acc79c0c 2026-11-05); 1.15.0 is `82c0f1fcd` built with cargo 1.98.0 (797e8a9bc 2026-08-05); both sides run the harness source of `322350993`. Measured 2026-11-20. What a cell measures is in `crates/multimodal/benches/release/README.md`; beside this file, `environment.json` holds the identities, `cells.csv` every cell and `rounds.csv` every window.

Cells that read slower by more than 2 % (in how many rounds the head was the slower side, and the head's time over the base's in every round):

- `qwen3_omni` 16-bit 44.1 kHz stereo, 30 s: +3 % (718 ms against 694 ms; slower in 15 of 15 rounds: 1.031 1.030 1.046 1.036 1.033 1.033 1.077 1.098 1.022 1.028 1.029 1.032 1.038 1.032 1.032)
- `qwen3_asr` 16-bit 44.1 kHz mono, 30 s: +3 % (716 ms against 693 ms; slower in 15 of 15 rounds: 1.032 1.031 1.041 1.025 1.033 1.034 1.040 1.031 1.027 1.032 1.099 1.036 1.037 1.034 1.091)
- `qwen3_asr` 16-bit 44.1 kHz stereo, 30 s: +3 % (718 ms against 695 ms; slower in 15 of 15 rounds: 1.031 1.032 1.032 1.034 1.036 1.034 1.034 1.034 1.031 1.029 1.036 1.030 1.037 1.034 1.099)
- `qwen3_omni` 16-bit 44.1 kHz mono, 30 s: +3 % (714 ms against 692 ms; slower in 15 of 15 rounds: 1.032 1.028 1.037 1.030 1.032 1.032 1.091 1.102 1.028 1.029 1.029 1.027 1.036 1.031 1.028)
- `qwen3_asr` 16-bit 22.05 kHz mono, 30 s: +3 % (690 ms against 669 ms; slower in 15 of 15 rounds: 1.026 1.038 1.045 1.027 1.027 1.037 1.027 1.016 1.029 1.025 1.022 1.030 1.034 1.034 1.114)
- `qwen3_omni` 16-bit 22.05 kHz mono, 30 s: +3 % (688 ms against 668 ms; slower in 15 of 15 rounds: 1.031 1.032 1.029 1.032 1.032 1.030 1.031 1.083 1.022 1.029 1.028 1.025 1.031 1.032 1.027)

## Every processor configuration, medians over the clip classes

| processor | time per clip | least .. most | CPU per clip | faster / level / slower | same bytes, the standard classes |
|---|---|---|---|---|---|
| `qwen3_asr` | +1 % | -3 % .. +3 % | 0 % | 4 / 12 / 3 | 19 of 19 |
| `qwen3_omni` | 0 % | -3 % .. +3 % | +1 % | 3 / 13 / 3 | 19 of 19 |

## `qwen3_asr`, milliseconds per clip

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per clip | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| 16-bit 16 kHz mono, 0.1 s | 0.16 | +2 % | 0.15 | +2 % | yes |
| 16-bit 16 kHz mono, 1 s | 0.59 | +1 % | 0.57 | 0 % | yes |
| 16-bit 16 kHz mono, 30 s | 14.9 | -3 % | 14.7 | -3 % | yes |
| 16-bit 16 kHz mono, 600 s | 324 | 0 % | 321 | 0 % | yes |
| 16-bit 16 kHz mono, clipped signal, 30 s | 14.9 | -2 % | 14.7 | -2 % | yes |
| 16-bit 16 kHz mono, constant offset, 30 s | 14.7 | -3 % | 14.6 | -3 % | yes |
| 16-bit 16 kHz mono, silence, 30 s | 14.9 | -1 % | 14.7 | -2 % | yes |
| 16-bit 8 kHz mono, 30 s | 36.8 | +1 % | 36.5 | +1 % | yes |
| 16-bit 22.05 kHz mono, 30 s | 690 | +3 % | 686 | +3 % | yes |
| 16-bit 44.1 kHz mono, 30 s | 716 | +3 % | 711 | +3 % | yes |
| 16-bit 48 kHz mono, 30 s | 80.3 | +2 % | 79.6 | +1 % | yes |
| 16-bit 96 kHz mono, 30 s | 146 | 0 % | 145 | 0 % | yes |
| 16-bit 44.1 kHz stereo, 30 s | 718 | +3 % | 713 | +3 % | yes |
| 16-bit 48 kHz stereo, 30 s | 83.0 | +1 % | 82.4 | +1 % | yes |
| 16-bit 48 kHz six channels, 30 s | 93.7 | +1 % | 92.9 | +1 % | yes |
| 24-bit 16 kHz mono, 30 s | 15.1 | -2 % | 15.0 | -2 % | yes |
| 32-bit 16 kHz mono, 30 s | 15.1 | -1 % | 15.0 | -1 % | yes |
| float 16 kHz mono, 30 s | 14.8 | -3 % | 14.7 | -2 % | yes |
| A-law 8 kHz mono, 30 s | 37.2 | +1 % | 36.8 | +1 % | yes |

## `qwen3_omni`, milliseconds per clip

| class | 1.16.0, one thread | vs 1.15.0 | CPU ms per clip | vs 1.15.0 | same bytes as 1.15.0 |
|---|---|---|---|---|---|
| 16-bit 16 kHz mono, 0.1 s | 0.16 | -1 % | 0.15 | +1 % | yes |
| 16-bit 16 kHz mono, 1 s | 0.60 | +1 % | 0.58 | +1 % | yes |
| 16-bit 16 kHz mono, 30 s | 14.9 | -3 % | 14.8 | -3 % | yes |
| 16-bit 16 kHz mono, 600 s | 324 | 0 % | 322 | 0 % | yes |
| 16-bit 16 kHz mono, clipped signal, 30 s | 14.9 | -2 % | 14.8 | -2 % | yes |
| 16-bit 16 kHz mono, constant offset, 30 s | 15.0 | -2 % | 14.9 | -2 % | yes |
| 16-bit 16 kHz mono, silence, 30 s | 14.8 | -2 % | 14.7 | -2 % | yes |
| 16-bit 8 kHz mono, 30 s | 36.9 | +1 % | 36.6 | +1 % | yes |
| 16-bit 22.05 kHz mono, 30 s | 688 | +3 % | 684 | +3 % | yes |
| 16-bit 44.1 kHz mono, 30 s | 714 | +3 % | 710 | +3 % | yes |
| 16-bit 48 kHz mono, 30 s | 80.4 | +2 % | 79.8 | +2 % | yes |
| 16-bit 96 kHz mono, 30 s | 145 | -1 % | 144 | -1 % | yes |
| 16-bit 44.1 kHz stereo, 30 s | 718 | +3 % | 714 | +3 % | yes |
| 16-bit 48 kHz stereo, 30 s | 83.5 | +2 % | 82.9 | +2 % | yes |
| 16-bit 48 kHz six channels, 30 s | 93.9 | +1 % | 93.3 | +1 % | yes |
| 24-bit 16 kHz mono, 30 s | 15.1 | -3 % | 14.9 | -3 % | yes |
| 32-bit 16 kHz mono, 30 s | 15.0 | -1 % | 14.8 | -2 % | yes |
| float 16 kHz mono, 30 s | 14.9 | -2 % | 14.7 | -3 % | yes |
| A-law 8 kHz mono, 30 s | 37.1 | +1 % | 36.9 | +1 % | yes |
