# Comparative tokenizer harness

The harness behind "Against other implementations" in the [crate README](../../README.md#against-other-implementations):
one cell module, `harness/cells.rs`, compiled behind a small adapter per implementation, so every
implementation runs the same measurement code on the same texts. Each adapter is a package and a workspace
of its own, with the `Cargo.lock` its published numbers ran with; nothing here is a member of the
repository's workspace or a dependency of the crate.

| column | adapter | implementation | build profile |
|---|---|---|---|
| `smg` | `harness/smg-tok` | this crate, from the checkout the harness lives in | opt-level 3, fat LTO, one codegen unit |
| `v1` | `harness/smg-tok --features tokenizers-v1` | this crate with the `tokenizers` 1.0.0-rc.3 pipeline as its encoder | the same |
| `fastokens` | `harness/fastokens-tok` | the fastokens stack: one open-source Rust tokenizer crate over `fastokens` 0.3.5, `riptoken` 0.3.0 (tiktoken files), `tekken` 0.2.0 (`tekken.json`) and `tokenizers` 0.22.2 (where those decline a file), at the commit in `references.json` | that crate's own workspace profile: thin LTO, panic abort |
| `tokenizers` | `harness/hf-tok` | the `tokenizers` crate 0.23.2 through its default pipeline (`encode_fast`) | opt-level 3, fat LTO, one codegen unit |
| `tiktoken-rs` | `harness/tt-tok` | `tiktoken-rs` 0.12.1 on the two families that have a ranks file or built-in ranks (Kimi K3, gpt-oss) | the same |

A column of an earlier release of this crate is the `smg` adapter built against a checkout of that
release (`--previous` below), by the toolchain of this checkout, so it differs by the crate alone.

## Reproduce

Linux only (the cells read `/proc`, the runner pins with `taskset`). Run from the repository root with
the pinned Rust toolchain; the Python runner uses only the standard library.

```sh
git worktree add ../smg-v1.15.0 v1.15.0
python crates/tokenizer/benches/compare/run.py prepare --corpus <corpus> --previous smg-1.15.0=../smg-v1.15.0
python crates/tokenizer/benches/compare/run.py ids --corpus <corpus>
python crates/tokenizer/benches/compare/run.py run --corpus <corpus> --cores 2,4,6,8
python crates/tokenizer/benches/compare/run.py run --corpus <corpus> --cores 2,4,6,8 --rounds 5 --edge 0.10
python crates/tokenizer/benches/compare/run.py collect --release 1.16.0 --csv crates/tokenizer/benches/compare
python crates/tokenizer/benches/compare/tables.py --readme crates/tokenizer/README.md
```

`prepare` downloads the tokenizer files of the sixteen checkpoints in `models.json` at their pinned
revisions and verifies every SHA-256 (no model weights; `--models PATH` keeps them elsewhere,
`--skip-fetch` takes a complete directory as it is), checks the corpus against `corpus.json`, clones the
fastokens stack at its pinned commit (`--fastokens-src PATH` takes an existing checkout), and builds one
binary per column: a copy of the adapter under `--output` whose path dependency names the checkout,
built `--locked` against the adapter's `Cargo.lock`. A `--previous` column starts from the same lock, and
cargo changes what that release's manifest requires. `--columns` and `--families` take a subset.

`ids` runs first: every column encodes the 7,015 parity strings of every family, and
`ids-summary.txt` counts the strings on which a column's ids equal the `smg` column's. A family's line
also names a column whose process ended at a non-zero exit status, and a column whose file does not hold
a whole record for every string. Timing two implementations that tokenize a text differently compares
different work, so read it before the tables.

`run` writes one JSON line per process under `--output/raw/`: `{"ctx": ..., "out": ...}`, where `ctx` is
the column, the family, the round, the core, the exit status and the load average, and `out` is the
cell's own line. A process that failed has a non-zero `rc`, and one that ran into the timeout keeps what
it wrote to stderr until then. The `v1` column on a family whose file the v1 pipeline declines takes its
next encode path (this crate's own encoder, or the `tokenizers` crate's): its `backend` string then has
no `hf:v1`, and that cell is no reference. The last command above is the edge pass: five more rounds on
the encode cells in which `smg` stands within 10 % of its best reference, where three rounds can leave
the side open.

`collect` writes the three csv files that `tables.py` reads from those raw lines, one line per process:
`encode-rounds.csv` (the p50 of nanoseconds per token), `misc-rounds.csv` (decode, streams and batch) and
`load-memory.csv`. A process that failed keeps its line with an empty figure, and so does one whose cell found
its own stream text wrong (its status reads `incorrect`). `--release` names the `smg`
column (`smg-1.16.0`) and `--csv` the directory, here the one `tables.py` reads.

Run on an otherwise idle machine, each stream on a physical core whose sibling is idle; do not compile
or run other benchmarks at the same time. Two builds of one source differ by up to about 20 % on a cell
from code placement alone, so a margin inside that band is not a difference between implementations.

## What a cell measures

- **`enc`** (encode): one process per column, family, size class and mode, pinned to one core; 3 warm-ups
  and 30 timed encodes (10 for the `tokenizers` column at the two largest classes); the p50 of nanoseconds
  per token. *Cold* takes a different text for every encode (the class's 33 texts in order), *warm* the
  same text every time. The warm text is the first of its class, encoded again and again, so a warm
  figure is the speed of that one text with hot caches, not the speed of mixed traffic. The six warm texts
  hold 1,548, 8,873, 151,173, 599,401, 2,289,914 and 4,613,798 bytes (the 256 class to the 1M class), and
  90.1, 41.2, 50.1, 42.3, 39.4 and 40.9 % of those bytes are outside ASCII, where their classes hold 17.1,
  31.1, 37.6, 38.3, 38.7 and 38.7 % (`corpus.json`; the 1M class holds each 500k text twice). Such a text
  can read slower warm than its class reads cold, as some cells from the 32k class up do in
  `encode-rounds.csv`. A round is one pass over the families, the columns interleaved inside each cell
  in an order that rotates from cell to cell; a cell's figure is the median of its rounds.
- **`dec`** (decode): the first 2,048 ids of a 32k-token text. A full decode (30 repetitions, ns per
  token); the same ids stepped through one stream (ns per token, and the time of a step); the first
  delta, the time a fresh stream takes from its first id to its first text (300 repetitions). The cell
  also runs on the first 32,768 ids of a 128k-token text (10 repetitions).
- **`stream`**: 64 streams of 256 steps interleaved on one thread (3 repetitions): the time of a step,
  and the time to each stream's first text, both as wall time and as the stream's own time. Every
  stream's text is checked against the full decode of its ids.
- **`batch`**: the first 1,000 parity strings, in a plain loop and through the implementation's batch
  call where it has one, on one thread (10 repetitions): tokens per second.
- **`info`** and **`mem`** (load and memory): the load time and the resident set after the load; the
  high-water mark after one encode of a 500k-token and of a 1M-token text; the allocator calls of one
  2k-token encode; the resident set per thread with 64 threads alive, read while every thread still
  holds its tokenizer: the growth on one shared tokenizer, then with a tokenizer per thread the growth
  of that phase and the level counted from before the first phase. The table shows the level: the
  second phase starts from what the first left resident, so its own growth depends on the first.
- **`parity`**: for every string the ids, the decoded text and the streamed text, written to a file.

`cells.rs` holds more cells than the runner starts (`threads`, `fail`, `robust`, `share`, `rss`, `fuzz`:
concurrency, failure handling, a soak and a differential fuzz); an adapter binary run without arguments
lists them. Every binary routes its allocations through a counting allocator (one relaxed atomic add per
call), which is where the allocator counts come from; an implementation that allocates more pays that
add more often.

## Corpus

The texts are test data and are not stored in this repository. `corpus.json` records what the published
numbers ran on, and `prepare` refuses a corpus directory that differs (`--any-corpus` accepts one, which
is a new experiment):

- `perf/<class>/000.txt` to `032.txt`: 33 texts for each of the classes 256, 2k, 32k, 128k and 500k
  tokens, counted with the DeepSeek V4.1 Flash vocabulary, with the SHA-256 of every file. The class 1m
  is derived by the runner: text *i* is the 500k text *i* followed by the 500k text *i + 1*.
- `parity.jsonl`: 7,015 strings, one `{"i", "cls", "text"}` object per line, in 34 classes: news text in
  eight languages whole and cut short, emoji and joiner sequences, combining marks, source code, JSON
  files, tool-call payloads, whitespace and control runs, numbers and punctuation, random valid UTF-8,
  random bytes decoded lossily, and special scalar values.

How it was assembled: news articles of the XL-Sum test split in Arabic, Chinese, English, Hindi,
Japanese, Korean, Russian and Thai (the first 400 of each language's first file, named in `corpus.json`),
and pieces of open-source code and configuration (Rust, Python, Go, shell, C, Jinja templates,
TOML and YAML, protobuf, JSON) with generated tool-call payloads. A seeded assembly draws from those
pools until a text reaches its class's token count.

## Files

- `harness/cells.rs`: the cells. `harness/<adapter>/`: `src/main.rs` (the adapter), `Cargo.toml`,
  `Cargo.lock`.
- `models.json`: the sixteen checkpoints by the name the crate README gives them, with repository,
  revision and the SHA-256 of every file the tokenizers read.
- `references.json`: where the fastokens stack comes from, and the version of every outside
  implementation as its adapter's lock pins it.
- `corpus.json`: the corpus record above.
- `run.py`: the runner.
- `encode-rounds.csv`, `misc-rounds.csv`, `load-memory.csv`: the figures of the 1.16.0 run, one line per process
  (family, cell, column, round), written by `run.py collect` from that run's raw lines; `environment.json` says
  what it ran on.
- `tables.py`: writes [RESULTS.md](RESULTS.md) from those three files and, with `--readme`, the tables and the reading
  list of the crate README. A figure is the median over a cell's rounds; the `v1` column has none where its
  `backend` string has no `hf:v1`.
