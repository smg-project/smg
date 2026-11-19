# Comparative tokenizer benchmark: results

Written by `tables.py` from `encode-rounds.csv`, `misc-rounds.csv` and `load-memory.csv`; [README.md](README.md) says what a cell measures and how to run it.

## Encode, cold (ns per token; in brackets the margin vs fastokens / vs v1, in %)

| family | vocabulary | 256 | 2k | 32k | 128k | 500k | 1m |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 | byte-level BPE | 81.5 (-40 / +43) | 81.7 (-39 / +9) | 61.8 (-22 / +65) | 61.9 (-28 / +80) | 52.7 (-34 / +115) | 39.3 (-33 / +179) |
| DeepSeek V3.2 | byte-level BPE | 77.7 (-24 / +56) | 83.6 (-29 / +14) | 67.5 (-20 / +83) | 62.1 (-21 / +84) | 51.6 (-28 / +116) | 36.7 (-14 / +216) |
| Llama 3.2 | byte-level BPE | 73.8 (-40 / +36) | 79.9 (-36 / +17) | 63.7 (-24 / +58) | 52.2 (-26 / +89) | 43.5 (-36 / +137) | 33.8 (-29 / +202) |
| Llama 4 | byte-level BPE | 85.3 (-36 / +35) | 102.7 (-28 / +17) | 89.3 (-27 / +58) | 73.9 (-10 / +96) | 69.0 (-34 / +116) | 48.1 (-17 / +201) |
| gpt-oss | byte-level BPE | 82.2 (-38 / +43) | 87.8 (-26 / +35) | 86.7 (-26 / +65) | 79.0 (-20 / +92) | 62.4 (-29 / +147) | 46.5 (-22 / +224) |
| Qwen 3.6 | byte-level BPE | 73.6 (-18 / +261) | 75.8 (-3 / +238) | 81.3 (-8 / +299) | 70.5 (+0 / +331) | 64.0 (-13 / +386) | 47.4 (-4 / +565) |
| GLM-5.3 | byte-level BPE | 78.1 (-39 / +37) | 74.9 (-43 / +18) | 57.6 (-20 / +84) | 44.3 (-22 / +120) | 41.8 (-35 / +135) | 32.8 (-27 / +198) |
| MiniMax M3 | byte-level BPE | 83.8 (-39 / +33) | 93.8 (-24 / +16) | 79.9 (-23 / +123) | 65.2 (-14 / +160) | 53.1 (-10 / +215) | 42.2 (-1 / +291) |
| MiMo V2.6 | byte-level BPE | 69.0 (-36 / +54) | 76.1 (-23 / +29) | 62.6 (-19 / +165) | 50.8 (-7 / +192) | 46.2 (-12 / +229) | 38.9 (+10 / +298) |
| Mistral (tekken) | byte-level BPE (tekken.json) | 79.4 (+867 / -) | 99.2 (+645 / -) | 74.1 (+920 / -) | 61.5 (+1138 / -) | 52.2 (+1342 / -) | 42.0 (+1647 / -) |
| Gemma 3 | byte-fallback BPE | 166.5 (+160 / +232) | 126.7 (+161 / +287) | 133.2 (+148 / +231) | 99.0 (+177 / +326) | 87.4 (+178 / +410) | 64.8 (+185 / +582) |
| Gemma 4 | byte-fallback BPE | 168.5 (+104 / +231) | 130.9 (+138 / +301) | 104.7 (+171 / +458) | 95.6 (+164 / +674) | 82.8 (+166 / +855) | 60.3 (+175 / +1250) |
| Llama 2 (byte fallback) | byte-fallback BPE | 115.9 (+136 / +168) | 46.0 (+150 / +368) | 28.0 (+232 / +914) | 20.6 (+276 / +1505) | 19.1 (+247 / +1935) | 17.7 (+299 / +2120) |
| Kimi K3 | tiktoken file | 142.9 (+87 / -) | 132.9 (+66 / -) | 109.2 (+131 / -) | 101.3 (+144 / -) | 85.2 (+199 / -) | 81.3 (+206 / -) |
| BERT base | WordPiece | 106.8 (+1080 / -) | 88.3 (+1085 / -) | 96.4 (+1076 / -) | 103.1 (+1078 / -) | 104.6 (+1301 / -) | 101.2 (+1368 / -) |
| T5 small | Unigram | 1563.8 (+9 / -) | 1528.2 (+15 / -) | 1668.3 (+25 / -) | 1559.9 (+23 / -) | 1610.1 (+26 / -) | 1721.2 (+22 / -) |

## Encode, warm (ns per token; in brackets the margin vs fastokens / vs v1, in %)

| family | vocabulary | 256 | 2k | 32k | 128k | 500k | 1m |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 | byte-level BPE | 11.6 (+70 / +79) | 9.4 (+106 / +68) | 12.7 (+56 / +955) | 16.2 (+27 / +735) | 20.0 (+5 / +488) | 21.5 (+1 / +441) |
| DeepSeek V3.2 | byte-level BPE | 11.7 (+79 / +77) | 9.4 (+188 / +69) | 12.7 (+87 / +803) | 16.1 (+53 / +670) | 19.8 (+22 / +455) | 22.5 (+14 / +449) |
| Llama 3.2 | byte-level BPE | 9.2 (+48 / +55) | 7.1 (+146 / +59) | 14.2 (+9 / +630) | 15.2 (+10 / +650) | 19.2 (-2 / +430) | 22.5 (-14 / +366) |
| Llama 4 | byte-level BPE | 14.3 (+47 / +117) | 9.1 (+130 / +91) | 16.4 (+21 / +1054) | 19.6 (+52 / +700) | 23.0 (-4 / +530) | 25.3 (+9 / +489) |
| gpt-oss | byte-level BPE | 12.4 (+48 / +112) | 8.2 (+93 / +90) | 15.6 (+13 / +970) | 18.1 (+53 / +779) | 23.2 (+35 / +516) | 26.9 (+21 / +483) |
| Qwen 3.6 | byte-level BPE | 18.1 (+85 / +691) | 8.5 (+194 / +1164) | 17.0 (+84 / +1881) | 18.9 (+64 / +1593) | 25.8 (+45 / +1160) | 30.2 (+26 / +958) |
| GLM-5.3 | byte-level BPE | 11.6 (+51 / +89) | 7.7 (+151 / +82) | 12.8 (+11 / +653) | 15.6 (+15 / +613) | 19.1 (-5 / +404) | 22.5 (-13 / +348) |
| MiniMax M3 | byte-level BPE | 14.6 (+69 / +229) | 10.8 (+176 / +178) | 17.6 (+50 / +880) | 22.0 (+30 / +730) | 31.7 (+41 / +455) | 28.6 (+44 / +528) |
| MiMo V2.6 | byte-level BPE | 13.2 (+68 / +161) | 7.9 (+196 / +130) | 17.0 (+40 / +861) | 18.1 (+31 / +833) | 24.8 (+29 / +553) | 26.8 (+20 / +524) |
| Mistral (tekken) | byte-level BPE (tekken.json) | 10.7 (+6744 / -) | 7.8 (+6873 / -) | 14.3 (+5284 / -) | 16.8 (+4318 / -) | 20.3 (+3587 / -) | 22.4 (+3179 / -) |
| Gemma 3 | byte-fallback BPE | 10.0 (+499 / +57) | 7.7 (+515 / +88) | 15.4 (+762 / +2660) | 20.4 (+508 / +2138) | 23.9 (+356 / +1697) | 26.5 (+389 / +1538) |
| Gemma 4 | byte-fallback BPE | 9.6 (+396 / +10) | 7.0 (+368 / -2) | 13.4 (+358 / -7) | 16.4 (+520 / +4465) | 22.3 (+321 / +3397) | 22.4 (+400 / +3186) |
| Llama 2 (byte fallback) | byte-fallback BPE | 7.7 (+309 / -40) | 7.3 (+276 / -55) | 13.3 (+183 / -63) | 14.1 (+297 / +1854) | 16.1 (+217 / +2141) | 15.6 (+290 / +2212) |
| Kimi K3 | tiktoken file | 24.1 (+792 / -) | 35.3 (+260 / -) | 42.9 (+546 / -) | 51.9 (+413 / -) | 56.7 (+355 / -) | 60.5 (+339 / -) |
| BERT base | WordPiece | 103.6 (+602 / -) | 69.6 (+1266 / -) | 99.0 (+1120 / -) | 99.5 (+1099 / -) | 101.9 (+1184 / -) | 102.3 (+1288 / -) |
| T5 small | Unigram | 375.1 (+26 / -) | 1012.2 (+22 / -) | 1281.5 (+35 / -) | 1392.2 (+21 / -) | 1592.8 (+30 / -) | 1641.2 (+24 / -) |

## v1.15.0 to 1.16.0 (change in ns per token; negative = 1.16.0 faster)

| family | cold: median change (range over the six sizes) | warm: median change (range) |
|---|---|---|
| DeepSeek V4.1 | -18.2 % (-26.1 % .. -12.9 %) | -35.2 % (-45.5 % .. -20.3 %) |
| DeepSeek V3.2 | -21.7 % (-29.9 % .. -10.6 %) | -34.8 % (-45.4 % .. -16.8 %) |
| Llama 3.2 | -20.1 % (-24.4 % .. -12.8 %) | -26.3 % (-33.6 % .. -14.2 %) |
| Llama 4 | -16.2 % (-20.8 % .. -9.6 %) | -19.4 % (-25.6 % .. -5.7 %) |
| gpt-oss | -11.8 % (-20.3 % .. -5.4 %) | -21.9 % (-27.4 % .. -8.2 %) |
| Qwen 3.6 | -22.9 % (-29.5 % .. -14.6 %) | -33.2 % (-50.7 % .. -12.8 %) |
| GLM-5.3 | -17.8 % (-28.7 % .. -17.1 %) | -24.3 % (-32.4 % .. -14.2 %) |
| MiniMax M3 | -17.8 % (-21.0 % .. +2.0 %) | -22.8 % (-37.8 % .. -12.0 %) |
| MiMo V2.6 | -23.5 % (-26.6 % .. -13.9 %) | -29.9 % (-44.3 % .. -13.8 %) |
| Mistral (tekken) | -11.9 % (-22.8 % .. -0.2 %) | -23.1 % (-25.0 % .. -6.5 %) |
| Gemma 3 | -14.6 % (-19.9 % .. -8.2 %) | -31.3 % (-37.9 % .. -12.1 %) |
| Gemma 4 | -9.6 % (-13.0 % .. -4.4 %) | -13.8 % (-21.1 % .. -10.5 %) |
| Llama 2 (byte fallback) | -10.6 % (-12.2 % .. -5.7 %) | -9.9 % (-16.9 % .. -3.3 %) |
| Kimi K3 | -21.1 % (-24.2 % .. -17.6 %) | -29.8 % (-37.6 % .. -24.3 %) |
| BERT base | -91.4 % (-92.9 % .. -91.0 %) | -91.9 % (-92.8 % .. -85.2 %) |
| T5 small | -11.2 % (-15.4 % .. +1.1 %) | -12.8 % (-14.6 % .. -8.8 %) |

## Decode and streaming (ns per token or ns per step; in brackets the margin vs fastokens, in %; the batch column, in Mtok/s, carries the margin vs fastokens / vs v1)

| family | full decode, 2,048 ids | incremental decode, 2,048 ids | first delta of one stream | 64 streams: step | 64 streams: first delta | batch of 1,000 texts, one thread (Mtok/s) |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | 8.6 (+268) | 90.9 (+270) | 50.0 (+460) | 41.0 (+485) | 110.0 (+619) | 48.1 (+4 / +372) |
| DeepSeek V3.2 | 8.4 (+281) | 89.5 (+283) | 50.0 (+460) | 50.0 (+382) | 111.0 (+821) | 56.0 (+36 / +426) |
| Llama 3.2 | 7.8 (+284) | 89.1 (+267) | 50.0 (+460) | 40.0 (+500) | 120.0 (+251) | 44.3 (-13 / +277) |
| Llama 4 | 8.1 (+303) | 89.2 (+295) | 50.0 (+460) | 50.0 (+400) | 110.0 (+401) | 40.9 (+9 / +424) |
| gpt-oss | 8.4 (+273) | 88.9 (+278) | 50.0 (+462) | 41.0 (+488) | 110.0 (+474) | 36.9 (+7 / +372) |
| Qwen 3.6 | 8.3 (+296) | 88.5 (+292) | 50.0 (+460) | 40.0 (+500) | 121.0 (+298) | 31.2 (+28 / +687) |
| GLM-5.3 | 6.5 (+290) | 87.7 (+235) | 50.0 (+460) | 41.0 (+485) | 110.0 (+274) | 53.6 (-9 / +293) |
| MiniMax M3 | 6.6 (+268) | 94.9 (+210) | 50.0 (+460) | 40.0 (+500) | 121.0 (+305) | 31.2 (+16 / +328) |
| MiMo V2.6 | 7.6 (+274) | 88.6 (+257) | 50.0 (+460) | 41.0 (+488) | 121.0 (+240) | 38.6 (-1 / +268) |
| Mistral (tekken) | 7.6 (-) | 89.1 (-) | 50.0 (-) | 40.0 (+752) | 241.0 (+453) | 37.7 (+2693 / -) |
| Gemma 3 | 15.9 (+562) | 105.6 (+498) | 260.0 (+212) | 51.0 (+1000) | 391.0 (+82) | 36.1 (+389 / +1296) |
| Gemma 4 | 15.7 (+564) | 104.5 (+502) | 250.0 (+216) | 51.0 (+980) | 390.0 (+75) | 44.7 (+440 / +1642) |
| Llama 2 (byte fallback) | 13.5 (+668) | 119.8 (+532) | 270.0 (+263) | 60.0 (+1070) | 401.0 (+130) | 70.5 (+150 / +403) |
| Kimi K3 | 6.9 (+32) | 91.4 (+195) | 50.0 (+340) | 40.0 (+428) | 130.0 (+224) | 23.4 (+386 / -) |
| BERT base | 6.6 (+8399) | 96.2 (+1765) | 211.0 (+464) | 50.0 (+3286) | 320.0 (+367) | 9.9 (+682 / -) |
| T5 small | 6.7 (+1756) | 108.0 (+337) | 210.0 (+72) | 60.0 (+600) | 361.0 (+78) | 0.7 (+13 / -) |

## Decode, streaming and batch: v1.15.0 to 1.16.0 (change in time; negative = 1.16.0 faster; the batch column is the change in tokens per second, where positive = 1.16.0 faster)

| family | full decode, 2,048 ids | incremental decode, 2,048 ids | first delta of one stream | 64 streams: step | 64 streams: first delta | batch of 1,000 texts, one thread |
|---|---|---|---|---|---|---|
| DeepSeek V4.1 | -9.9 % | +1.5 % | +0.0 % | -18.0 % | -0.9 % | +15.6 % |
| DeepSeek V3.2 | -5.3 % | +0.3 % | +0.0 % | +0.0 % | +0.0 % | +28.3 % |
| Llama 3.2 | -8.7 % | +0.2 % | +0.0 % | -20.0 % | +9.1 % | +14.7 % |
| Llama 4 | -12.2 % | -1.3 % | +0.0 % | +0.0 % | -0.9 % | +19.2 % |
| gpt-oss | -7.0 % | -0.3 % | +0.0 % | -18.0 % | -0.9 % | +14.1 % |
| Qwen 3.6 | -7.5 % | -0.3 % | +0.0 % | +0.0 % | +0.0 % | +40.0 % |
| GLM-5.3 | -9.2 % | +0.2 % | +0.0 % | -18.0 % | -15.4 % | +13.9 % |
| MiniMax M3 | -4.9 % | +0.4 % | +0.0 % | -20.0 % | +0.8 % | +29.2 % |
| MiMo V2.6 | -6.8 % | +0.3 % | +0.0 % | -18.0 % | -6.9 % | +32.6 % |
| Mistral (tekken) | -12.9 % | +0.1 % | +0.0 % | -2.4 % | +100.8 % | +10.8 % |
| Gemma 3 | -0.8 % | +0.9 % | +0.0 % | -15.0 % | -7.1 % | +8.9 % |
| Gemma 4 | +2.5 % | +1.6 % | -0.4 % | -15.0 % | -5.1 % | +49.1 % |
| Llama 2 (byte fallback) | +2.0 % | +0.7 % | -3.6 % | +0.0 % | +2.8 % | +6.6 % |
| Kimi K3 | -9.7 % | -0.3 % | +0.0 % | +0.0 % | +0.0 % | +48.6 % |
| BERT base | -99.0 % | -95.7 % | -86.7 % | -97.6 % | -81.5 % | +689.8 % |
| T5 small | -93.6 % | -75.2 % | -46.3 % | -83.4 % | -33.1 % | +0.9 % |

## Load and memory (each cell: SMG 1.16.0 / SMG v1.15.0 / fastokens)

| family | load, ms | resident memory after the load, MB | high-water mark after a 1m-token encode, MB | growth per thread at 64 threads, one shared tokenizer, kB | growth per thread, a tokenizer per thread, MB |
|---|---|---|---|---|---|
| DeepSeek V4.1 | 281.4 / 483.5 / 743.3 | 72.4 / 138.2 / 112.8 | 94 / 145 / 197 | 698 / 971 / 16305 | 69.4 / 123.4 / 123.1 |
| DeepSeek V3.2 | 290.4 / 685.2 / 724.9 | 71.0 / 136.9 / 113.0 | 93 / 143 / 197 | 698 / 971 / 14390 | 72.4 / 118.5 / 118.1 |
| Llama 3.2 | 269.9 / 794.5 / 1169.5 | 93.3 / 184.4 / 284.1 | 116 / 252 / 336 | 707 / 998 / 15531 | 98.0 / 244.0 / 294.2 |
| Llama 4 | 455.5 / 1329.1 / 2280.4 | 116.8 / 284.2 / 439.9 | 132 / 390 / 526 | 707 / 988 / 32629 | 108.5 / 342.9 / 465.4 |
| gpt-oss | 440.9 / 1294.2 / 2105.8 | 114.0 / 287.6 / 442.0 | 133 / 396 / 528 | 716 / 987 / 32827 | 118.3 / 350.0 / 463.4 |
| Qwen 3.6 | 407.7 / 1074.8 / 1481.1 | 112.7 / 244.5 / 186.0 | 134 / 246 / 270 | 655 / 894 / 32854 | 145.9 / 209.0 / 175.5 |
| GLM-5.3 | 324.7 / 972.8 / 1430.4 | 107.4 / 211.4 / 326.0 | 123 / 291 / 377 | 717 / 1001 / 14776 | 109.3 / 265.7 / 334.7 |
| MiniMax M3 | 319.9 / 800.2 / 1170.3 | 87.4 / 170.6 / 158.8 | 111 / 177 / 276 | 724 / 990 / 32791 | 101.6 / 152.8 / 190.9 |
| MiMo V2.6 | 242.5 / 703.5 / 947.5 | 71.9 / 140.3 / 209.5 | 95 / 162 / 270 | 703 / 1000 / 15276 | 79.0 / 171.7 / 228.5 |
| Mistral (tekken) | 345.6 / 393.7 / 193.1 | 81.0 / 99.7 / 71.9 | 112 / 130 / 95 | 692 / 926 / 463 | 71.9 / 90.1 / 90.6 |
| Gemma 3 | 674.6 / 1769.0 / 2435.4 | 157.1 / 288.8 / 473.0 | 177 / 466 / 506 | 604 / 696 / 1883 | 174.2 / 419.7 / 461.2 |
| Gemma 4 | 578.4 / 1755.6 / 2597.3 | 156.9 / 288.5 / 467.8 | 173 / 464 / 499 | 599 / 690 / 1822 | 171.8 / 415.8 / 462.3 |
| Llama 2 (byte fallback) | 43.1 / 115.5 / 142.0 | 26.4 / 40.5 / 42.7 | 52 / 58 / 119 | 631 / 976 / 1841 | 19.9 / 28.9 / 39.9 |
| Kimi K3 | 370.6 / 431.0 / 2328.2 | 83.3 / 101.9 / 72.1 | 113 / 130 / 79 | 711 / 999 / 181 | 75.7 / 93.8 / 66.3 |
| BERT base | 14.6 / 22.2 / 51.8 | 14.5 / 20.2 / 19.5 | 24 / 603 / 596 | 52 / 1882 / 1934 | 7.4 / 12.0 / 13.9 |
| T5 small | 42.8 / 50.0 / 85.5 | 35.5 / 43.1 / 44.6 | 393 / 462 / 455 | 1656 / 1657 / 1672 | 31.4 / 39.0 / 36.0 |

## Reading the tables

- Of 192 encode cells SMG 1.16.0 reads a lower time than every reference in the csv files (fastokens, tiktoken-rs, tokenizers, v1) on 127: 42 of 96 cold and 85 of 96 warm; SMG v1.15.0 in the same rounds reads 83.
- Against v1.15.0, cold: of 96 encode cells 91 read faster and 0 slower by more than 5 %, 5 within it; the median change is -17.3 %.
- Against v1.15.0, warm: of 96 encode cells 95 read faster and 0 slower by more than 5 %, 1 within it; the median change is -23.7 %.
- Against v1.15.0, decode, streaming and batch: of 96 cells 47 read faster and 2 slower by more than 5 %, 47 within it. Slower: Mistral (tekken), 64 streams: first delta, 241.0 against 120.0 ns (+100.8 %); Llama 3.2, 64 streams: first delta, 120.0 against 110.0 ns (+9.1 %).
- Against fastokens, cold: of 96 encode cells SMG is ahead beyond the placement band (20 %) on 40 and behind beyond it on 36; 20 read inside the band (the margins run from -43 % to +1647 %).
- Against fastokens, warm: of 96 encode cells SMG is ahead beyond the placement band (20 %) on 82 and behind beyond it on 0; 14 read inside the band (the margins run from -14 % to +6873 %).
- Against v1, cold: of 72 encode cells SMG is ahead beyond the placement band (20 %) on 66 and behind beyond it on 0; 6 read inside the band (the margins run from +9 % to +2120 %).
- Against v1, warm: of 72 encode cells SMG is ahead beyond the placement band (20 %) on 66 and behind beyond it on 3; 3 read inside the band (the margins run from -63 % to +4465 %).
- Full decode, 2,048 ids, against fastokens: SMG is ahead beyond the band on 15 of 15 families, behind beyond it on 0, inside it on 0 (the margins run from +32 % to +8399 %). The fastokens column has no figure for Mistral (tekken), whose process failed in every round.
- Incremental decode, 2,048 ids, against fastokens: SMG is ahead beyond the band on 15 of 15 families, behind beyond it on 0, inside it on 0 (the margins run from +195 % to +1765 %). The fastokens column has no figure for Mistral (tekken), whose process failed in every round.
- First delta of one stream, against fastokens: SMG is ahead beyond the band on 15 of 15 families, behind beyond it on 0, inside it on 0 (the margins run from +72 % to +464 %). The fastokens column has no figure for Mistral (tekken), whose process failed in every round.
- 64 streams: step, against fastokens: SMG is ahead beyond the band on 16 of 16 families, behind beyond it on 0, inside it on 0 (the margins run from +382 % to +3286 %).
- 64 streams: first delta, against fastokens: SMG is ahead beyond the band on 16 of 16 families, behind beyond it on 0, inside it on 0 (the margins run from +75 % to +821 %).
- Batch of 1,000 texts, one thread, against fastokens: SMG is ahead beyond the band on 8 of 16 families, behind beyond it on 0, inside it on 8 (the margins run from -13 % to +2693 %).
- Batch of 1,000 texts, one thread, against v1: SMG is ahead beyond the band on 12 of 12 families, behind beyond it on 0, inside it on 0 (the margins run from +268 % to +1642 %).
- Load and memory, against v1.15.0: SMG 1.16.0 reads lower by more than 5 % on 16 of 16 families in the load time; on 16 of 16 families in the resident memory after the load; on 16 of 16 families in the high-water mark after a 1m-token encode; on 15 of 16 families in the growth per thread on one shared tokenizer; on 16 of 16 families in the growth per thread with a tokenizer per thread.
- Load and memory, against fastokens: SMG 1.16.0 reads lower by more than 5 % on 15 of 16 families in the load time (higher on Mistral (tekken)); on 14 of 16 families in the resident memory after the load (higher on Mistral (tekken), Kimi K3); on 14 of 16 families in the high-water mark after a 1m-token encode (higher on Mistral (tekken), Kimi K3); on 13 of 16 families in the growth per thread on one shared tokenizer (higher on Mistral (tekken), Kimi K3); on 15 of 16 families in the growth per thread with a tokenizer per thread (higher on Kimi K3).
- The v1 column reads `-` for Mistral (tekken), Kimi K3, BERT base, T5 small: the v1 pipeline declines their files, and that build runs SMG's own encoder there.
- The placement band: the v1 build shares SMG's decode and streaming code, and between the two builds those 80 cells read -18 % to +25 %.
- The corpus: 17 to 39 % of the bytes of a size class are outside ASCII (`corpus.json`).
