# smg-response-template

A Rust port of the response-template parser of Hugging Face transformers
(`transformers.utils.chat_parsing`, release 5.18.0). A checkpoint can describe
in the `response_template` of its `tokenizer_config.json` how its output splits
into message fields (`thinking`, `content`, `tool_calls`, ...). This crate loads
such a template and parses model output with it, streaming or at once, and
returns what transformers returns: the same keys in the same order, the same
strings, ints kept apart from floats, floats bit for bit, and the same region
events for the same chunks.

A template this crate would not parse exactly as transformers does is refused
when it loads (`LoadError::Unsupported`, naming the construct), never parsed
differently. Callers can then fall back to another parser.
`read_response_template` reads every template transformers loads, its patterns
as the `regex` module reads them, also one this crate does not run;
`ResponseTemplate::unsupported` then names what it does not run.

## API

| transformers 5.18.0 | this crate |
|---|---|
| `ResponseParser(template, prefix=..., tools=...)` | `ResponseParser::new(&template, prefix, &tools)` |
| `.initial_events`, `.feed(text)`, `.finalize()` | `initial_events()`, `feed(text)`, `finalize(self)` |
| `parse_response(text, template, prefix=..., tools=...)` | `parse_response(text, &template, prefix, &tools)` |
| `load_response_template(dict)` | `load_response_template(&serde_json::Value)`; `read_response_template` also returns a template this crate does not run (`ResponseTemplate::unsupported`) |
| `ResponseTemplate.truncate_past_last_anchor` | `ResponseTemplate::truncate_past_last_anchor` |
| event dicts / message dict | `Event` (serializes to the same dicts) / `Message` |
| `ValueError` while loading | `LoadError::Invalid` |
| exceptions while parsing | `ParseError` (`Content { kind, .. }` names the Python class; `Unsupported` for a template this crate does not run) |

`ResponseTemplate::last_anchor_end` gives where the last start anchor of a text
ends, or `None` without one, where `truncate_past_last_anchor` keeps the whole
text.

`prefix` is required as in transformers; pass `""` to opt out. `finalize`
consumes the parser, so the `RuntimeError`s of feeding or finalizing twice
cannot happen. Results match transformers' for a template `Value` equal to
what Python's `json.loads` returns; serde_json reads some float literals, and
integers beyond u64, differently. The files follow transformers' layout:
`response_templates.rs`, `response_parser.rs` and `content_parsers.rs` port
the files of the same names, keeping their function names and statement
order. `pyre.rs` stands in for the `regex` module and `py.rs` for the Python
built-ins the parser applies to model output (`str.strip`, `int`, `float`,
`json.loads`). As in transformers, every `feed` searches the text not yet
committed again, so a stream costs time quadratic in how much text stays
uncommitted.

## Regular expressions

transformers compiles patterns with the `regex` module (`regex.compile(pattern,
regex.DOTALL)`, version 0 behaviour), and streaming depends on its
`search(..., partial=True)`, whose results come from the module's matcher
rather than from the pattern's language.

`pyre/parse.rs` ports the module's parser (`_regex_core.py`) and reads a
pattern into a tree of every construct the module reads: literals and every
escape, classes with ranges, class escapes and POSIX classes, assertions,
every kind of group, look-around, backreferences, quantifiers of any item
(greedy, lazy and possessive, `{m,n}` included), inline flags (a verbose
pattern is read as verbose), fuzzy settings, group calls, conditionals and
verbs. It rejects what the module rejects. It does not read Unicode properties
in braces (`\p{...}`), named characters (`\N{...}`), POSIX classes other than
the 14 POSIX names, group names outside ASCII, a group inside a group of the
same name, version 1 behaviour (`(?V1)`), or groups nested more than 100 deep
(the module takes up to 198; a template must not be able to exhaust the
stack): a template with one of these in an open, close or start anchor pattern
is refused when it is read.

`pyre.rs` runs a tree on a backtracking matcher that reproduces the module's
results. It runs:

- literal characters and escapes of them (`\n`, `\t`, `\x41`, `\u00e9`,
  `\012`, ...), also `{`, `}` and `]` where the module reads them as
  literals;
- `.`, `\w`, `\W`, `\s`, `\S`, `\d`, `\D` (with the module's Unicode
  tables), and classes of characters, ranges of them and class escapes, read
  as the module reads them (`[\s-a]` is `\s`, `-` and `a`; `[\b]` is a
  backspace; `[[a]]` is `[` or `a`, then `]`; a class with a class escape
  and its complement is any character, also when negated: `[^\s\S]` matches
  every character);
- `^`, `\A`, `$`, `\Z`, `\z`, `\b`;
- groups, named or not, `(?:...)`, alternation, comments, and inline flags
  that change nothing a match returns (`s`, `u`, `x`, `f`, `V0`);
- `*`, `+`, `?` and `{m,n}` after one character, and `?` after a group;
- `*?`, `+?`, `??` and `{m,n}?` after one character. Open and close patterns
  are searched partially, and after a lazy repeat the module's partial
  matches follow its own rules: before a literal character it reports a
  partial match once at most one character is left, and before a literal
  string once no later copy of the string is left, whatever the repeat could
  take. The matcher reproduces them where they decide the result alone: for
  `.*?` before a literal, for `*?` after another item and before `\b` and a
  literal, and for a lazy repeat without a maximum (`*?`, `+?`, `{m,}?`) after
  another item and before a literal, outside alternatives, repeats and
  lookaheads (as in a call name read with `\S+?`);
- backreferences in start anchors and tag patterns;
- `(?=...)` without groups or backreferences in it, in start anchors and tag
  patterns (they are only used with `finditer`, never searched partially).

The matcher refuses everything else, naming it: lookahead in open and close
patterns, negative lookahead and lookbehind, inline flags that change a match
(`i`, `m`, `w`, `a`, `L`, `-s`, and the global `b`, `e`, `p`, `r`), possessive
quantifiers, repeats of groups and assertions, other lazy repeats in open and
close patterns (where the module's own first sets, or its search for a literal
that follows, decide where a partial match starts), `\B`, `\m`, `\M`, `\G`,
`\K`, `\X`, `\R`, `\h`, Unicode properties and POSIX classes, atomic groups,
branch reset groups, conditionals, group calls, fuzzy settings, verbs, and
backreferences in open and close patterns. So are alternatives two of
which can end in a negated class of one character, also before an empty group
(the module drops empty groups, merges such alternatives into one class and
matches `[^a]|[^b]` and `[^a](?:)|[^b]` as `[^ab]`), and a backreference inside
an optional group (the module does not retry the group where it failed before,
even when the group it refers to has changed since).

## Other templates refused when they load

`read_response_template` reads these too; `ResponseTemplate::unsupported`
names the reason.

- A start anchor or tag pattern that can match the empty string, or a field
  whose open and close patterns can both match it (transformers can loop
  forever there).
- `content_args` that transformers only reads while parsing and that fail
  there or depend on Python typing: not a dict, a `value_parser` that is not a
  dict naming a known parser, a `string_delims` that is not a list of pairs of
  non-empty strings, an empty or non-string `line_sep` or `kv_sep`, a
  non-string `tag_pattern`, a `key` or `value` group a match can leave unset.
- A value transformers stores in two places and then changes in place: a list
  in `defaults` for a repeated field without `join` (transformers appends to
  the template's own list, so it grows from one parse to the next), or a
  transform that uses the parsed content, or an item of it, at two paths where
  one contains the other (casting tool-call arguments changes both).

## Known differences

Both raise an explicit error where transformers returns a value:

- A value JSON cannot hold (NaN, an infinity, an integer outside i64 and u64, a
  string with a lone surrogate): `ParseError::Unrepresentable`, from the call
  whose events or message would carry it. A lone surrogate is parsed into a
  private-use character (from U+F0000 or U+100000 on), so a JSON text that
  also holds one of those characters and gives a key holding it twice raises
  it too, even where transformers then drops the value.
- JSON nested 512 deep: `ParseError::Content` with `PyErrorKind::Recursion`.
  Python raises `RecursionError` only near its C recursion limit, about 10,000
  levels.

## Compiling a template into a symphony table

With the `symphony` feature, `to_symphony::compile(&template, stops)` lowers a
template into a table of smg's `symphony` parser (a `symphony::Format`), which
reads a stream with literal markers and never runs a regular expression. The
table gives the message transformers' whole-text reading gives,
`parse_response(output + stop, template, prefix=prompt, tools=tools)`, at every
chunking of the output, except where a difference below applies. `stops` are
the texts of the tokens the model stops on, which the stop decoder hides from
the stream.

- `start_anchor` (one literal or several) and a `start_anchor_pattern` that is
  a finite set of literals (alternatives, optional literals, at most 32
  spellings) become the table's turn anchors, in the order the pattern tries
  them; the prompt's replay starts after the last one.
- The field with no open is the text outside the other fields: `content` is a
  content state, trimmed with Python's `str.strip` unless `strip: false`; any
  other name, or no such field, is a framing state, whose text is dropped.
  `thinking` or `reasoning_content` is a reasoning state, `content` a content
  state, and a field of another name a framing state.
- A delimiter's marker is a literal: an open's first literal after optional
  whitespace, a close's last literal before it. Whitespace the pattern takes
  beside the marker must be next to a state that is trimmed or dropped, and
  must not be able to take the first character of another marker that moves
  that state. Alternatives are markers of their own, tried in the pattern's
  order.
- A close made of markers (`<|end|>` and then the anchor's text) moves where
  its markers move in turn; text after a marker that leads into a framing state
  is dropped with it. A close of the outside text that is a stop text adds no
  row, and another close of it drops its marker. Every marker that is a stop
  text is named a stop token of the table.
- A `tool_calls` field that repeats, with tagged parameters (`xml-inline`),
  becomes a state whose text is a call, read by symphony's call syntax given as
  data (`CallSyntax::Shaped`). An open's marker enters the state, and what
  follows it in the pattern is the call's header, which captures the tool's
  name when the transform names it by a group of the open (the transform may
  also write the name itself). A close's marker leaves the state, and what comes
  before it is the call's trailer (`</function>\s*` before `</tool_call>`). The
  alternatives of an open share one header, and those of a close one trailer.
  The stream's stop closes a call, as transformers closes a region at the end
  of the text whatever its close: the parameters that ended are the call's
  arguments, also when a close came without its trailer.
- The tag pattern lowers into the tag, from its first literal through its
  value, which must be one way through the pattern, and the ways the value ends
  after it, in the order the pattern tries them: each is the text the match
  takes and then, last, a lookahead, so that a value may end at its closing tag
  or where the next tag begins. Parameters are found as `finditer` finds them,
  each written when its value ends, its text read by the template's
  `value_parser`. In a call with a trailer, a value's end must end with a
  literal the trailer cannot read, since the call's text ends where the trailer
  begins (`EndBeforeTrailer` otherwise).
- Values of calls read with `to_symphony::TRANSFORMERS`, this crate's port of
  `_sub_parse`, `_coerce`, `_schema_types` and `_coerce_tool_calls`, the code
  `parse_response` runs.

Calls whose arguments are JSON, or key-value lines, are not lowered yet
(`Construct::JsonArguments`, `JsonEnvelope`, `KeyValueLines`). Everything the
compile does not lower is refused by name: `Refusals` lists every construct the
template uses, each with a stable `Construct` code, and the compile goes on
after a refused construct to find the rest. `tests/corpus.tsv` pins the outcome
for the public templates of `tests/fixtures/compiled/`, and
`tests/to_symphony.tsv` for every template of `tests/fixtures/hf/` that loads.

Where a compiled table's message differs from transformers', by design, each
difference has a code; `tests/to_symphony_differs.tsv` lists every recorded
session that shows one, and the test checks each listed code by undoing it:

- X1: a field that does not repeat, written more than once, keeps every stay
  in order, each trimmed, where transformers keeps the last. A stream cannot
  take back text it sent.
- X2: text the prompt wrote after its last anchor (a prefill) belongs to the
  prompt and is not streamed again.
- X3: a stop text is not fed to the table; a region it closes closes at the
  stream's stop, and a stop text transformers keeps as text is not in the
  message.
- X5: where transformers raises, the table's events are defined. A call the
  stream cut short (for length, or with any finish but the model's stop) is
  reported cut: its arguments' object is left open, and the text of a parameter
  the cut left unfinished is `Malformed`, where transformers closes the region
  at the end of the text.
- X6: fields other than the thought, the content and the calls are dropped.
- X9: with no anchor in the prompt, reading starts at the output, where
  transformers reads the whole prompt as the message's start.

A session that gives a marker back (an open whose header does not match, a
close without its trailer) is never listed: it reads as transformers does.

### A compiled table's layout

A response template reads more than one spelling of a call, and a structural
tag writes one. `to_symphony::layout::learn(&format, stops, render)` learns the
spelling the model's chat template writes, a `symphony::shape::Layout`, from
the chat template's renderings of a probe conversation: the probe's tools
(`layout::tools()`) and the conversations of each `layout::Case`. The caller
renders each case with the chat template, and `learn` reads every turn back
through the table:

- The turn with the probe's two calls must read back as exactly those calls,
  with no content. A chat template that writes one call in a turn is read with
  one call, and its layout holds one call in a tag. One that renders no turn
  without a thought is read from the turns with the probe's thought.
- Each part of a call is the text the turn wrote there: the text before the
  call, the header with the tool's name in its place, a parameter with its key
  and value in their places, the text before, between and after the
  parameters, the trailer and the close. The layout also holds the text between
  two calls and after the last, how `true` and `null` are spelled, and the text
  around the thought and the answer.
- Every call and parameter must be written alike, the parameters in the order
  of the tool's schema, and the parts must write the turns again byte for byte:
  the turns with one call and with the probe's thought write the calls and the
  answer as the turn without them. A text the layout writes for every tool must
  not name one.
- The structural tags write the markers on the way into a region themselves,
  so the text after a thought, and before the marker that opens the answer's
  region, must hold no marker of the table, and the answer's text must stand
  right after its opener and right before its close.
- The template thinks first where its turns without a thought still write one,
  with its opener, before the calls and before the answer, where it renders no
  turn without one, or where it writes one before the calls and renders no
  answer without one: the structural tags then write the thought first where
  the prompt has not opened it. `Learned::apply` gives the table its layout and
  that mark. A template that writes its own thought before the calls but not
  before the answer, or the other way round, has no layout.

Anything else gives no layout, with a stable code (`NoLayout::code`), and the
table then has no structural tag for its calls. Only calls given as tagged
parameters have a layout.

`scripts/record_chat_template_renders.py` records two public chat templates'
renderings of the probe with transformers 5.18.0 in
`tests/fixtures/chat_templates/`. `tests/layout.rs` learns the layouts of Muse
Glimmer and Qwen 3.5 from them, and `tests/layouts.tsv` pins the outcome for
every pair of a response template and a chat template the recordings name.

## Parity tests

`scripts/generate_hf_fixtures.py` runs transformers itself and records what it
returns in `tests/fixtures/hf/`: every parser session of transformers' own tests
(`tests/utils/test_chat_parsing.py` at tag v5.18.0), the templates built into
`transformers serve`, the templates huggingface/trl writes for gpt-oss and
GLM-4.5 (`trl/chat_template_utils.py`), templates with the shapes of public
checkpoints written with neutral markers, prompts for these templates that end
inside a region, a delimiter or the start anchor (with and without tools), rules
of loading and parsing, seeded random templates and sessions, regex rows for
every pattern of that corpus, for Python-dialect details and for random
patterns, and the Python built-ins and character tables the port reproduces. The
tests replay each session whole, at every two-way split and in recorded
chunkings, and compare values and event traces. `tests/unsupported.tsv` lists
the recorded templates transformers accepts and this crate refuses, with the
reason; the test fails when that changes in either direction. Every template
transformers accepts is read, except those the reader does not read, and every
regex row's pattern is read exactly when the `regex` module compiles it.

The fixtures are regenerated by hand, and CI only replays them in the crate's
tests. They are pinned to transformers 5.18.0 (tag v5.18.0, commit
a906d3c4b65095f2308b6a6a193e934d03b8eb5d), `regex` 2026.9.10 and Python 3.12,
and are next regenerated when transformers 5.19 is released. To regenerate:

```bash
pip install transformers==5.18.0 regex==2026.9.10
git clone --depth 1 --branch v5.18.0 https://github.com/huggingface/transformers /tmp/transformers
python crates/response_template/scripts/generate_hf_fixtures.py --transformers-src /tmp/transformers
cargo test -p smg-response-template --features symphony
```

`--check` fails when the committed fixtures are stale.
`--extra-cases FILE --out DIR` records local cases and `--fuzz N --out DIR`
random ones; run the tests on them with `RESPONSE_TEMPLATE_HF_FIXTURES=DIR`.

`--compiled --sources DIR` records `tests/fixtures/compiled/`: the public
templates a compiled table is read against (transformers' tests, built-ins and
TRL's templates, and the templates of two Hub checkpoints, each read from a
named source at a pinned revision; `--fetch-sources DIR` downloads the pinned
files), sessions of each with transformers' whole-text reading, and how
transformers types a call's values (`values.jsonl`), which
`tests/to_symphony.rs` checks the value reader of a symphony table against.
`tests/to_symphony_parity.rs` replays every session through the table its
template compiles into, at every chunking.

To move to a newer transformers: update the pins in the generator (its
docstring and `PINNED_*`) and here, regenerate, review the fixture diff, adjust
the port, and update `TRANSFORMERS_VERSION` in `tests/hf_parity.rs`.

## License

Apache-2.0. Portions are ported from Hugging Face transformers (Apache-2.0);
the fixtures are recorded from it and include templates and inputs from its
tests.
