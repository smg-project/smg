# Fuzzing Symphony

Coverage-guided fuzz targets for the engine, built with [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz)
on libFuzzer. A target feeds the fuzzer's bytes to an engine as an output cut into deltas and
asserts the crate's contract on every input; libFuzzer mutates the bytes towards new code paths,
so the shapes it composes are not limited to what a recording or a seeded scramble has shown. The
checks are in `src/lib.rs`, whose module doc lists them: no failed `feed` and no panic;
conservation of every byte; what the parser says not depending on the cuts; well-formed call
streams with whole JSON arguments when a call ends (at a terminal's close or at the stream's end)
unless a `Malformed` event came while it was open; and the adapters rendering any sequence, the
Chat Completions stream agreeing with its whole message.

This directory is its own Cargo workspace: it needs a nightly toolchain and is not a member of the
repository's workspace, which stays on the pinned stable release.

## Targets

| target | table | call syntax |
|---|---|---|
| `any_table` | the input's last byte picks one of `Table::ALL` | every syntax |
| `json_calls` | `qwen3` | a JSON object between markers |
| `tagged_calls` | `qwen3_tagged` | `<function=` / `<parameter=` tags |
| `keyed_calls` | `glm` | `<arg_key>` / `<arg_value>` pairs after the name |
| `attributed_calls` | `deepseek_v4.1` | parameters as attributed elements |
| `message_calls` | `kimi_k3` | calls as typed messages in a turn of messages |
| `xml_calls` | `minimax_m3` | XML elements with a separator token |
| `gemma_calls` | `gemma4` | Gemma's `call:name{...}` with quoted strings |
| `pythonic_calls` | `lfm2.5` | a Python call list |

Every target declares the same two tools (a string and an integer; a string and an object), so
that typed assembly, unknown tools and forced shapes are all reachable.

## Running

```sh
rustup toolchain install nightly
cargo install cargo-fuzz --locked

cd crates/symphony/fuzz
./run.sh                              # every target, 10 minutes each, one at a time
./run.sh --minutes 30 --jobs 3 json_calls keyed_calls
```

`run.sh` builds the targets, writes the seeds and dictionaries (below), runs each target for the
bounded time and prints one row per target: executions, executions per second, coverage edges
(`cov:` in libFuzzer's last status line), corpus units, crashes and the breach classes the
artifacts name. It exits 1 when a target crashed or never ran. Logs are under `logs/<label>/`, crash
artifacts under `artifacts/<label>/<target>/` (`--label` names a run; `run` by default), the
growing corpus under `corpus/<target>/`; none of them is tracked. With `CARGO_TARGET_DIR` set,
the instrumented build goes to `$CARGO_TARGET_DIR/symphony-fuzz`.

A target stops at its first crash. `--keep-going` runs libFuzzer's fork mode instead, which keeps
every crash's artifact and continues, so that one campaign collects every class; a class that
recurs on every mutation of a seed still restarts the job each time and holds that target back.
A fork child that stops working (asleep after its crash handler, no CPU time accruing while the
parent waits for it) is stopped by `stall-watchdog.sh`, which `run.sh` runs beside its fork-mode
runs: a child older than two minutes whose CPU ticks did not grow over two samples is killed
(SIGKILL, which the parent counts as a crash it ignores; a TERM would reach libFuzzer's own
handler, whose exit code makes the parent stop the whole target), the stop is logged under
`logs/<label>/watchdog.log`, and the parent spawns the next job; parents are never touched, and
the watchdog ends with the run.
Once a class is reported, `--allow <names>` (the environment variable `SYMPHONY_FUZZ_ALLOW`, comma
separated) makes the harness look past it, so that the next class surfaces. Every check honors it.
An allowance is for a run, never for the default; the names are the ones the breach messages carry.
The whole-JSON promise breaks under three names, so that an allowance is exact: `empty-arguments`
(a call ended with no arguments fragment at all), `tolerated-json` (arguments whose shape closed,
or one scalar token, but which JSON refuses: a value left out, a trailing comma, an escape JSON
lacks, a short `\u`, whitespace outside JSON's, a number or a word cut short) and `whole-json` (the
committed prefix of a value that never closed). `json-prefix` holds until a call's
`Malformed { InvalidArguments }` event of no bytes says where its arguments stopped being JSON;
from there `verbatim-after-break` asks that every fragment be the model's bytes as written.
`pythonic-cut-call-closed` is a pythonic call that ended without its own `)` and had its object
closed for it (one that said nothing and left it open is a breach of the whole-JSON names, so no
event carries two names). A name is a shape, from any reader on every target, not a reader's own.
The nightly workflow allows the shapes of the classes reported and not yet fixed, named in
`symphony-fuzz.yml`; a name leaves the list with its fix, and a new reader's breach of an allowed
shape surfaces then.

One target by hand, with libFuzzer's own options after `--`:

```sh
cargo +nightly fuzz run --sanitizer none json_calls corpus/json_calls -- -max_total_time=600 -dict=prepared/dict/json_calls.dict
```

The sanitizer is off by default: the crate forbids unsafe code, and without instrumentation for
memory errors a target runs two to three times as many executions. `--sanitizer address` turns it
on. The nightly workflow `symphony-fuzz.yml` runs every target for ten minutes a night and on
demand, and uploads the artifacts and logs.

## The input

An input is the text the engine reads, followed by twelve bytes of parameters: eight piece sizes
(one to sixteen characters each, cycled), a flags byte (bit 0: an empty delta between the pieces;
bits 1-2: the engine's finish, stop, length, abort or another; bit 3: read whole), the length of
the prompt's tail at the front of the text, a scale for the pieces (one to eight) and, for
`any_table`, the table. Bytes that are not UTF-8 read as the replacement character, as a decoder
would release them. An input shorter than twelve bytes is an output read whole with a stop.

`examples/prepare.rs` writes a seed per recorded output slice in this layout, so a seed replays as
its table's own output and every mutation of it stays a case, and a dictionary per target: every
terminal of its table (of every table for `any_table`), the declared names and the atoms of the
call syntaxes, which lets libFuzzer compose markers it would otherwise have to discover byte by
byte. A dictionary word is at most 64 bytes, libFuzzer's limit; `prepare` refuses a longer entry
rather than let the driver skip it without a word. The harness's tests decode the seeds as written
and read the dictionaries back.

The slices come from bellwether's recorded sets, not from this repository (`../tests/common/slices.rs`,
shared with the crate's bench): for each table that stands for a call syntax, the hand-written
parse sets (`common`, `turns`) of the checkpoint group it reads, as `<slug>/parse/<set>.jsonl` of
the tree `bellwether unpack` writes. `BELLWETHER_FIXTURES` names that tree; `run.sh` and the
harness's seed tests refuse to run, or skip with the reason, without it. Locally, check out
smg-project/bellwether at the commit in `.github/versions/bellwether.ref` and point the variable at
its `fixtures/` directory: bellwether keeps the hand-written sets plain there, so no Git LFS fetch
is needed and `bellwether unpack` writes the same files for them beside the benchmark sets. Per
table, the seed of a family is the longest recorded output whose message has the family's shape
(content alone; one call with arguments; two or more calls; a thought and a call), placed after
the prompt the group's template leaves; the malformed seed is the one call's output with the last
character of the call's body before its closing marker cut (the `}` of a JSON object, the `>` of
a tag, the `]` of a list), so the call reaches its close malformed. A group the tree has not
recorded is named and left out; a tree with no group recorded fails `prepare`.

## When a target crashes

libFuzzer writes the input under `artifacts/<label>/<target>/crash-<hash>` (or `timeout-`,
`oom-`) and the log ends with the panic message, `breach <name>:` followed by the case (the
table, the prompt's tail, the output as a string literal, the cuts and the finish) and what the
check saw. In fork mode the message is in the job's own log; replaying the artifact prints it.

1. Reproduce: `cargo +nightly fuzz run --sanitizer none <target> artifacts/<label>/<target>/crash-<hash>`.
2. Minimize: `cargo +nightly fuzz tmin --sanitizer none <target> artifacts/<label>/<target>/crash-<hash>`
   writes the smallest input that still crashes next to it as `minimized-from-<hash>`.
3. Explain: `cargo +nightly run --release --example explain -- <target> <file>` prints the decoded
   case and runs the checks on it outside the fuzzer; with `--events` it prints the events of the
   whole replay and of the replay in pieces first, one per line, the streams a reader sees.
4. File it: an issue with the target, the crate's commit, the decoded case from step 3 (the output
   as a Rust string literal and the cuts are enough to replay it; attach the minimized bytes when
   the literal is unwieldy), the invariant that failed and the check's message. Nothing else from
   the machine that ran it. A crash is a parser defect: the fix adds the output to the subject's
   outputs in `tests/contract.rs`, where the contract test then replays it under every chunking.

A case that is no breach but a wrong check is fixed in `src/lib.rs`, with the case as a unit test.
