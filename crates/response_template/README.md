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

## API

| transformers 5.18.0 | this crate |
|---|---|
| `ResponseParser(template, prefix=..., tools=...)` | `ResponseParser::new(&template, prefix, &tools)` |
| `.initial_events`, `.feed(text)`, `.finalize()` | `initial_events()`, `feed(text)`, `finalize(self)` |
| `parse_response(text, template, prefix=..., tools=...)` | `parse_response(text, &template, prefix, &tools)` |
| `load_response_template(dict)` | `load_response_template(&serde_json::Value)` |
| `ResponseTemplate.truncate_past_last_anchor` | `ResponseTemplate::truncate_past_last_anchor` |
| event dicts / message dict | `Event` (serializes to the same dicts) / `Message` |
| `ValueError` while loading | `LoadError::Invalid` |
| exceptions while parsing | `ParseError` (`Content { kind, .. }` names the Python class) |

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

The `adapter` feature adds the glue smg's reasoning and tool parsers use, which
is not part of transformers: `adapter::check` says whether they can use a
template, and an `adapter::ResponseParserState` holds the parser of one output
for both of them. They read the fields `transformers serve` reads (`thinking`,
here also `reasoning_content`, `content` and `tool_calls`): a stream from the
region chunks and closed tool-call regions after the prompt, a complete output
from the parsed message.

Where smg's use differs from `transformers serve` by design:

- smg does not use a template that holds a float, since it reads the tokenizer
  config with serde_json.
- The parsers see the output without the stop or end-of-sequence text the stop
  decoder removes; `transformers serve` also feeds that text, so whitespace
  before it, or the marker inside a region it ends, can differ.
- The prompt tail is cut from the rendered prompt, not from the decoded prompt
  tokens, which can differ in special tokens the tokenizer adds (such as a
  BOS token, which matters only when the prompt holds no start anchor), in
  `clean_up_tokenization_spaces` and in Unicode normalization.
- Without tools (or with `tool_choice: none`) the tool parser does not run and
  tool calls are dropped; with `separate_reasoning: false` reasoning is
  returned as content; prompt regions are not streamed; a continued assistant
  message returns only the generated part.
- When the template reads content only after an opener, a forced tool
  choice (Chat `required`, a named function or `allowed_tools` in required
  mode, Messages `any` or `tool`) and a JSON `response_format` are refused
  with a 400: smg would constrain them to JSON from the first token, which
  opens no region, so the parsers would read none of it. A template whose
  content has no opener reads that JSON as content.
- Tool-call arguments are cast to the types the request's tools declare, as
  `parse_response(..., tools=...)` does; `transformers serve` passes no tools.
- A tool-call region whose value is a list gives one call per item
  (`transformers serve` fails there), and arguments are serialized as smg's
  other tool parsers do. In a stream, the content of a chunk is sent before
  the tool calls that close in it.
- A region that fails to parse (its content does not parse, it holds a value
  JSON cannot, or its calls cannot be read), also of a field the parsers do not
  read, is content from where it began, as generated, and the rest of the
  output passes through unparsed, where `transformers serve` fails a complete
  response and ends a stream. A complete output is still the parsed message
  wherever `transformers serve` can read that. `take_error` says why, for a
  log. A missing required field is not an error.
- A stream holds back at most 8 KiB of text that could still begin a delimiter,
  where `transformers serve` holds it all and searches it again for every
  chunk; past that, the text goes to the current region, and only a delimiter
  that starts in its last 4 KiB is waited for. A longer delimiter, such as a
  long whitespace run before an opener, can leave part of its text in the
  region.

## Regular expressions

transformers compiles patterns with the `regex` module, and streaming depends on
its `search(..., partial=True)`, whose results come from the module's matcher
rather than from the pattern's language. `pyre.rs` parses the constructs used
by the templates of transformers' tests and of `transformers serve` and by the
public checkpoint shapes the parity fixtures record, plus positive classes,
ranges in classes and `\w` and `\s` in classes (as in tool-name patterns like
`[A-Za-z_][A-Za-z0-9_]*`, `[\w.-]+` or `[^\s<]+`), and runs them on a
backtracking matcher that reproduces those results.
Supported:

- literal characters other than `{`, `}` and `]`, `\` before ASCII
  punctuation, `\n`;
- `.` (transformers sets `DOTALL`), `\w`, `\s` (with the `regex` module's
  Unicode tables), `[...]` and `[^...]` of literal characters, ranges of
  them, `\w` and `\s` (`-` is literal only as the last item; `[`, `]`, `&`,
  `|`, `~` and a `^` other than the first are not taken there);
- `^`, `$`, `\Z`, `\b`;
- `(?:...)`, `(?P<name>...)`, alternation;
- `*`, `+`, `?` after one character, `?` after a group; `*?` after one
  character, in open and close patterns only as `.*?` before a literal, or
  after another item and before `\b` and a literal (other positions change
  what a partial search reports);
- `\1` to `\9` in start anchors and tag patterns.

Anything else is refused, for example look-around, inline flags, `{m,n}` and
`{`, `}` or `]` that Python reads as literal characters, possessive and other
lazy quantifiers, escapes such as `\d` or `\t`, a `-` next to `\w` or `\s` in a
class other than as its last item (the `regex` module reads it as a literal,
Python's `re` rejects it), unnamed groups, backreferences in open and close
patterns, and groups nested more than 100 deep (the `regex` module takes up to
198; a template must not be able to exhaust the stack). So are alternatives two
of which can end in a negated class of one character, also before an empty
group (the `regex` module drops empty groups, merges such alternatives into one
class and matches `[^a]|[^b]` and `[^a](?:)|[^b]` as `[^ab]`), and a
backreference inside an optional group (the module does not retry the group
where it failed before, even when the group it refers to has changed since).

## Other templates refused when they load

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

## Parity tests

`scripts/generate_hf_fixtures.py` runs transformers itself and records what it
returns in `tests/fixtures/hf/`: every parser session of transformers' own tests
(`tests/utils/test_chat_parsing.py` at tag v5.18.0), the templates built into
`transformers serve`, templates with the shapes of public checkpoints written
with neutral markers, prompts for these templates that end inside a region, a
delimiter or the start anchor (with and without tools), rules of loading and
parsing, seeded random templates and sessions, regex rows for every pattern of
that corpus, for Python-dialect details and for random patterns, and the Python
built-ins and character tables the port reproduces. The tests
replay each session whole, at every two-way split and in recorded chunkings,
and compare values and event traces. `tests/unsupported.tsv` lists the
recorded templates transformers accepts and this crate refuses, with the
reason; the test fails when that changes in either direction.

The fixtures are regenerated by hand, and CI only replays them in the crate's
tests. They are pinned to transformers 5.18.0 (tag v5.18.0, commit
a906d3c4b65095f2308b6a6a193e934d03b8eb5d), `regex` 2026.9.10 and Python 3.12,
and are next regenerated when transformers 5.19 is released. To regenerate:

```bash
pip install transformers==5.18.0 regex==2026.9.10
git clone --depth 1 --branch v5.18.0 https://github.com/huggingface/transformers /tmp/transformers
python crates/response_template/scripts/generate_hf_fixtures.py --transformers-src /tmp/transformers
cargo test -p smg-response-template --features adapter
```

`--check` fails when the committed fixtures are stale.
`--extra-cases FILE --out DIR` records local cases and `--fuzz N --out DIR`
random ones; run the tests on them with `RESPONSE_TEMPLATE_HF_FIXTURES=DIR`.

To move to a newer transformers: update the pins in the generator (its
docstring and `PINNED_*`) and here, regenerate, review the fixture diff, adjust
the port, and update `TRANSFORMERS_VERSION` in `tests/hf_parity.rs`.

## License

Apache-2.0. Portions are ported from Hugging Face transformers (Apache-2.0);
the fixtures are recorded from it and include templates and inputs from its
tests.
