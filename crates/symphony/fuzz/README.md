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
committed prefix of a value that never closed). Every fragment is judged by the crate's own judge
of a JSON prefix (`json::Prefix`, the one the readers hold their fragments to), which refuses at the
first byte a strict reader refuses and from the backslash of an escape no continuation completes:
a refusal the partial parser shares is `json-prefix`, and one it tolerates (the shapes above, the
short `\u` included) is `tolerated-json` at the fragment that brings it, as it would be at the
call's end, so that an allowance means one thing at a cut and at a close. `json-prefix` holds until
a call's `Malformed { InvalidArguments }` event of no bytes says where its arguments stopped being
JSON; from there `verbatim-after-break` asks that every fragment be the model's bytes as written.
`pythonic-cut-call-closed` is a pythonic call that ended without its own `)` and had its object
closed for it (one that said nothing and left it open is a breach of the whole-JSON names, so no
event carries two names). A name is a shape, from any reader on every target, not a reader's own.
The nightly workflow allows no class; the classes it allowed are
[on record below](#the-classes-the-nightly-allows), and a name enters `symphony-fuzz.yml` with its
report and leaves with its fix, so a new reader's breach of a shape once allowed surfaces at once.

One target by hand, with libFuzzer's own options after `--`:

```sh
cargo +nightly fuzz run --sanitizer none json_calls corpus/json_calls -- -max_total_time=600 -dict=prepared/dict/json_calls.dict
```

The sanitizer is off by default: the crate forbids unsafe code, and without instrumentation for
memory errors a target runs two to three times as many executions. `--sanitizer address` turns it
on. The nightly workflow `symphony-fuzz.yml` runs every target for ten minutes a night and on
demand, and uploads the artifacts and logs.

## The classes the nightly allows

No breach class is allowed: the scheduled run passes no `--allow`, and the dispatch input's default
is empty. This section is the record of the classes that were allowed and left, with their
reproducers and what ended each: `tolerated-json` on 2026-11-08, `empty-arguments` on 2026-11-08
and `made-up-empty-string` before it. A name enters the allowance with its report and leaves it
with its fix, from both lines that write it in `symphony-fuzz.yml`: the scheduled run's list and
the dispatch input's default. A night's red is a new report: a name that left is one too. The
`tolerated-json` inputs below are the harness's own, from its tests in `src/lib.rs`, but for the one
marked as the prefix judge's own, from the table its test
`it_refuses_what_the_partial_parser_tolerates_where_a_strict_reader_does` reads: in
`the_whole_json_promise_names_its_three_breaches`, each one as a Qwen3-table call's whole arguments
followed by the call's end marker (`ended_with`) is the breach it names; the short `\u` escape at a
cut is the fragment case of the test beside it,
`a_fragments_breach_is_named_by_what_the_partial_parser_makes_of_it`; whitespace outside JSON's has
no input there, its shape is `prefix_breach`'s.

- `tolerated-json` (left the allowance on 2026-11-08): arguments that strict JSON refuses but that
  the readers hand on. The name is given by two judges that agree on the shapes: at a fragment,
  where the crate's prefix judge refuses the text and the partial parser takes it whole; at the
  call's end, where the arguments' shape closed but the whole-JSON check fails. A scalar cut short
  (`0e`, `n`, `1.`) is the prefix of a longer value, so the prefix judge accepts it at every
  fragment and the name comes at the call's end alone; the same scalar cut short before a delimiter
  (`{"a": 1.}`, from the `TOLERATED` table in `../src/json/prefix.rs`) is named at the fragment. The
  other shapes are a value left out (`{"city": "Paris", "days": ,}`), a trailing comma (`{"a":
  1,}`), an escape JSON lacks (`"\h"`), a short `\u` escape at a cut (`{"a": ["\u1"]`) and
  whitespace outside JSON's. The fix decides per shape whether the readers refuse it (`Malformed {
  InvalidArguments }` at the fragment that brings it, or at the call's end for a scalar cut short,
  which stays a valid prefix while it arrives) or the judge accepts it as the engines' own JSON
  readers do, and makes the readers and the judge agree. What ended it: the readers that take whole
  JSON arguments (the JSON, JSON-list and shaped syntaxes) say `Malformed { InvalidArguments }` at
  the fragment that brings every shape above (#3790, #5004), so the harness sees the word before the
  end; the readers that type a value by the declared schema (the keyed, attributed, tagged, pythonic
  and Gemma syntaxes) hand the harness a value already spelled as JSON, which the prefix judge
  takes. Runs on main without any allowance found no breach of any class: on 2026-11-08 the nine
  targets ran three minutes each (5.1 million executions) and the four readers that still read
  values through the tolerant parser (`pythonic_calls`, `gemma_calls`, `xml_calls`, `any_table`)
  fifteen minutes each (7.1 million executions), 87 target-minutes with no crash and no class named.
  The typed readers' own reading of a value JSON refuses (a trailing comma dropped, an unreadable
  value kept as a string) is a question of parity with each engine's own parser, on record as its
  own report, not a breach of this promise.
- `empty-arguments` (left the allowance on 2026-11-08): a call that ended with no arguments fragment
  at all and no `Malformed` word inside it other than a stop's (an `OpenAtTheStop` or
  `OpenAtTheMarker` of no bytes), where the contract asks for `Malformed` before the end of a cut
  call, or whole arguments (the harness names it at the call's end, as `ended_with("")` does). It
  was allowed for the pythonic reader's cut call, which a fix of October ended (#3791):
  `[get_weather(` ended by the stream says `Malformed { UnterminatedRegion }` before its end
  (`a_call_either_ending_cuts_right_after_its_parenthesis_says_so_before_its_end` in
  `../src/pythonic.rs`). An allowed breach leaves no trace in the nightly's log, so a dispatch of
  `symphony-fuzz.yml` with `allow` set to `tolerated-json` alone ran the nine targets for the
  nightly's ten minutes each on main `7de38d7c2` (workflow run 42463947758) and found no breach;
  the name then left both lines of the allowance. A night that names it again is a new report: its artifact names the reader and
  the input, which go on record here with the fix.
- `made-up-empty-string` (left the allowance with #3880, the pythonic reader's fix for a parameter
  without a value): a fragment that wrote an empty string with no byte of a value behind it, where
  the model left the value out before a comma or a close. The harness still names it
  (`made_up_empty_string` in `src/lib.rs`), so a new red under the name is a new report.

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
   outputs in `tests/parser/contract.rs`, where the contract test then replays it under every chunking.

A case that is no breach but a wrong check is fixed in `src/lib.rs`, with the case as a unit test.
