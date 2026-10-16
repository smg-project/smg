# Notes on the assemblers

Each call syntax has an assembler that turns a call's bytes into its events. The module docs say
what each one reads and show one call; these notes say what each syntax says and what the assembler
does with it, byte accounting included, and where a rule comes from.

## The JSON assembler (`json::assembler`)

Every byte of the object lands in exactly one event: the head up to the arguments value is the
`source` of `ToolCallStart`, the argument bytes are the `source` of their fragments, the tail after
the value is the `source` of `ToolCallEnd`. When the arguments come before the name they are held
until the name is whole, since a call starts before its arguments.

The assembler keeps the promise `ToolCallArguments` makes, that the fragments so far always form a
valid JSON prefix, as far as the prefix parser can tell: each new run of argument bytes is judged by
`Prefix`, which accepts what `PartialJson` accepts in prefix mode, before it is emitted, and from
the first byte it cannot take, the argument bytes come back as `Malformed` with `InvalidArguments`
instead, nothing already emitted being revised. The prefix parser tolerates what the old crate
tolerated (a literal's prefix, a bracket closed by the wrong kind where the parent can take it), so
the promise is exactly as strong as that parser.

Every byte is read once. The outline is taken by a `Scanner` that keeps its place between pieces,
and the prefix is judged on the bytes that arrived, so a call's cost is linear in its length however
the text is cut: the first version read the whole text again on every piece, and a 259 KB argument
string fed by the character took 800 s against 0.03 s whole.

A started call whose object never closes is closed at `finish` with what arrived, and the bytes
after its arguments value (a comma cut short, a complete member, or bytes that are no member) come
back as `Malformed` with `UnterminatedRegion`.

`feed` takes one piece and returns how many of its bytes the object took: all of them before the
close, and up to the closing brace in the piece that carries it, so the format routes whatever
follows the object (a closing marker, more text) itself. Bytes fed after the close are not taken.

What the assembler does not decide: the call's index and id (the format mints them and passes them
to `Assembler::new`), whether the name is a declared tool, and whether a string-valued arguments
member should be decoded. The format decides those with the events in hand.

## The Qwen tagged assembler (`tagged::assembler`)

Every byte of the call lands in exactly one event. Whitespace between tags is the template's and
goes into the source of the next event: the function tag's into `ToolCallStart`, a parameter tag's
and the value's text into the fragments they produce, the rest into the closing `}`. Anything else
between tags, a tag out of its place, a tag that cuts a name short and a name that is empty come
back as `Malformed` with a reason, so the model's bytes are never read as something they are not and
never vanish into a source. Inside a value only `</parameter>` is a tag; `<function=`, `<parameter=`
and `</function>` there are the value's text (the engine's parser ends a value at the next
`<parameter=` or `</function>` as well, a tolerance to judge with adversarial fixtures rather than
port).

Two endings. `Assembler::close` is for a block the model ended before `</function>` (its own closing
marker came, or the next call began): the call is over as far as the model is concerned, so an open
string is closed, the object is closed, and what was held comes back as `Malformed`.
`Assembler::finish` is for a stream that was cut: nothing is closed, as in `crate::json::Assembler`,
because arguments cut short must not look complete to a client. A value that is written whole and
never closed comes back as `Malformed` in both, since its key was never written.

`feed` takes one piece and returns how many of its bytes the call took: all of them before
`</function>`, and up to that tag in the piece that carries it, so the format routes whatever
follows (the block's closing marker, more text) itself.

## The keyed assembler (`tagged::keyed`)

What the syntax says, and what the assembler does with it:

- **The name is the text before the first `<arg_key>`**, or before the call's end when the call
  has no arguments, less the whitespace around it, which is the template's and goes into the
  next event's source. The name is known only when that tag arrives, so `ToolCallStart` waits
  for it; names are short. A name holds no whitespace, quote, brace, bracket, parenthesis or
  angle bracket: any other text there (the JSON of a `<tool_call>` block quoted inside a code
  fence, prose, a call written as Python) is reported, and no call starts, so a block that names
  no call writes no arguments either.
- **A value's type comes from the request's tools**, as in the Qwen assembler: a declared string
  is its text, streamed as it arrives; every other value is written whole at `</arg_value>`
  through `json`, which reads the JSON the templates write for a number, a boolean, a list,
  an object or null, and infers the rest. Nothing is taken from a value: the templates write it
  directly between its tags.
- **Inside a value only `</arg_value>` is a tag**; the other three there are the value's text.
- **Two endings.** `Assembler::close` is the call's closing marker, or the block's end before
  it: after the last value it names the call if it has not, closes the object and pushes
  `ToolCallEnd` with the marker's bytes; inside a value it closes an open string and the object
  and reports what was held. `Assembler::finish` is a stream that was cut: nothing is closed.
- **Every byte of the call lands in exactly one event**, with the Qwen assembler's accounting:
  the name's bytes in `ToolCallStart`, each tag's and value's bytes in the fragments they
  produce, text where the syntax has tags as `Malformed` run by run with the whitespace before
  it as `Dropped { Wrapper }`.

## The attributed assembler (`tagged::attributed`)

What the syntax says, and what the assembler does with it:

- **The request's tools type the value** (`Typing::Declared`), as in the keyed assembler: a
  declared string is its text, every byte of it, streamed as it arrives; every other value is
  written whole at `</parameter>` through `json`, which reads the JSON the template writes
  for a number, a boolean, a list, an object or null (`tojson`), and infers the rest. A
  parameter tag with anything between the name's closing quote and `>` is no parameter tag the
  syntax has: the whole tag comes back as `Malformed`, and no parameter opens.
- **Or the tag's `string` attribute types the value** (`Typing::StringAttribute`, DSML):
  `string="true"` is the text, streamed like a declared string; `string="false"` is JSON the
  model wrote, written whole at the close through `json`'s inference. The request's tools are
  not consulted: the model says the type. A parameter tag with another attribute, or none, is
  no parameter tag.
- **Or the tag's `type` attribute types the value** (`Typing::TypeAttribute`, XTML), with
  every type: `string` streams, `number`, `boolean`, `null`, `array` and `object` are written
  whole through `json`'s inference; a type that is none of the six is no parameter tag.
- **A tag ends at `>`, or at a marker** (`TagEnd`): XTML's `<|sep|>` is a token of its own
  and may arrive in pieces, so the scanner holds it like the other tags, and inside a value it
  is the value's text. XTML's call tag carries the model's count of the turn's calls,
  ` index="N"`, which the family's tags allow (`Tags::indexed_calls`); the assembler keeps
  its own index, as every assembler does.
- **Inside a value only `</parameter>` is a tag**; `<parameter name="` there is the value's
  text.
- **A value may be a CDATA section** where the family's template writes one (`Tags::cdata`):
  MiniCPM5 wraps a value that holds `<`, `&` or a newline as `<![CDATA[…]]>`. The section's
  opener and closer are the template's wrapping and go into the fragments' sources, not the
  value; the text between them is the value as it is, streamed as it arrives for a declared
  string, read whole at the closing tag for any other type. Text between `]]>` and the closing
  tag, which the template never writes, is reported.
- **A value may stand on its own line** (`Tags::value_on_its_own_line`): dots3's template
  writes a newline after the parameter tag and one before its closing tag. Those two are the
  template's framing and go into the fragments' sources, not the value: the first byte of a
  value is dropped when it is a newline, and a trailing newline is held until the closing tag
  says it was the framing, or more text says it was the value's.
- **The newlines between the tags are the template's** and go into the next event's source;
  text where the syntax has tags is reported.
- **Two endings**, as DSML's: `Assembler::close` is the invoke's closing tag, or the block's
  end before it; `Assembler::finish` is a stream that was cut, and closes nothing.
- **Every byte of the call lands in exactly one event.**

## The XML assembler (`tagged::xml`)

What the syntax says, and what the assembler does with it:

- **An element is a key, and its children say what it holds.** Text makes it a leaf; a child
  tag makes it a container: a list when the tool declares an array there or, failing a
  declaration, when the child is an `<item>`, else an object (a list's elements are `<item>`
  elements, but so is a member named `item`, which is why the declaration decides first); no
  child at all makes it empty. A tag under an element the tool declares a string or an
  integer is the leaf's text (a file's `<template>`, an e-mail's `<p>`), and with no
  declaration a tag opens a child only after the template's separator, since the template
  writes the separator before every tag it writes and never inside a value. A list's items
  and an object's members are elements in turn, so the tree nests as deep as the arguments
  do. Whitespace right after an opening tag is held until the element's next byte says what
  it is: a leaf's text when text or the closing tag follows (a value that is one space is one
  space), the template's when a child tag follows.
- **A value's type comes from the request's tools at every depth** (`Declared::kind_at`:
  below an object a key selects a property, below an array any element selects its items). A
  declared string is its text, streamed as it arrives; every other leaf is written whole at
  its closing tag through `json`, which reads the booleans and numbers the template writes
  as text and infers the rest. An empty element is `[]` or `{}` when the tool declares an
  array or an object there, `null` when nothing is declared and it is a list's item, and `""`
  everywhere else: the template writes `None` inside a list as an empty element, skips a
  mapping's `None` member, and leaves a top-level argument that is `None` out altogether.
  Nothing is taken from a value: the template writes it directly between the tags, with no
  escaping.
- **A top-level null is never written.** The template leaves an argument whose value is `None`
  out (`for k, v in arguments.items() if v is not none`), so no output carries one; the
  parity test counts that as a corpus class.
- **A tag is `<`, a name and `>`**, the name any run of characters without whitespace or angle
  brackets, with a leading `/` for a closing tag. A `<` followed by anything else is
  text, so a comparison in a value stays text; a `<` that does spell a tag is one, inside a
  value too, as every marker parser reads it. Inside a leaf, a tag other than the leaf's own
  closing tag is the leaf's text; a closing tag that matches nothing open is reported.
- **The name ends at its quote**, and the invoke tag at `>`; text between the two is reported,
  as for DSML. An invoke with no name starts no call, and everything in it is reported.
- **The separator keeps its place.** When nothing is held before it, it is `Dropped { Wrapper }`
  at once; when a value's text or an opening tag is held, its bytes go into the next
  fragment's source, so every event's bytes stay in the output's order.
- **Two endings.** `Assembler::close` is the invoke's closing tag (`INVOKE_CLOSE`, the one
  terminal a started call's end carries), or the next invoke's opening or the block's close,
  which end the call with no bytes of their own: every open container is closed, a tag the
  end cut short is reported and an open streamed string gets its quote, a leaf never written
  and an element without a child come back as `Malformed`, and the arguments object is
  closed. An invoke that named no call takes whichever terminal ended it, so that it is
  reported the same way however it ended. `Assembler::finish` is a stream that was
  cut: nothing is closed, what was held comes back as `Malformed { UnterminatedRegion }`, and a
  call that started still ends, with no bytes of its own, as every assembler ends one.
- **Every byte of the invoke lands in exactly one event**, with the tagged assemblers'
  accounting: the name's bytes in `ToolCallStart`, each tag's and value's bytes in the
  fragments they produce, text where the syntax has tags as `Malformed` run by run with the
  whitespace before it as `Dropped { Wrapper }`.

## The value reader (`tagged::value`)

The rules, in the order `json` applies them:

- A parameter declared `string` is its text, every byte of it. Nothing is trimmed and nothing is
  decoded: `"exact phrase"` keeps its quotes, indentation and a trailing newline stay, and
  `&amp;` is five characters. A parameter is declared `string` when its schema admits that
  type at all: `"type": "string"`, a list of types or an `anyOf` that includes it, or, when the
  schema has no `type`, an `enum` of strings, with or without `null` among them, or a `const`
  string, which is an enum of one. An `allOf` admits what all of its members that name a type
  admit, so `{"allOf": [{"type": "string"}, {"maxLength": 8}]}` is a string, and it narrows the
  types named beside it to those, so `{"type": ["string", "null"], "allOf": [{"type": "string"},
  {"maxLength": 8}]}` is a string that is never null. A `number` admits every integer, so an
  `allOf` of an `integer` and a `number` is an integer. OpenAPI's `"nullable": true` admits `null`
  beside whatever else the schema admits. An `enum` beside a `type` adds nothing, as for the
  engine's parser: BFCL declares `{"type": "array", "enum": [...]}` for a list whose members come
  from the enum, and that is an array. When the schema admits `null` as well as `string`, the text
  `null`, and the `None`
  a template writes for a null argument, are null; any other text is the string. A string
  declared alone keeps `null` as its text, as the engine's readers do; bellwether #56 refuses a case
  whose reference holds a null there, since no output carries it.
- A parameter declared `integer` is that integer when its text is a sign and digits, with the
  whitespace around it ignored. It may be spelled `+5` or `007`; it is written in JSON's
  spelling, with the digits the model wrote.
- Everything else is inferred: text that is JSON is that JSON; text that is a Python literal
  (`True`, `None`, `{'size': 'large'}`, `['n1', 'n2']`, a tuple) is the JSON it stands for,
  written with `json.dumps`'s separators; and the rest is a string holding the text exactly.
  That covers a parameter the tool does not declare, a declared integer whose text is not one,
  and every other declared type. Most templates write an array or object with `tojson` and a
  boolean or null as Python's `True`, `False` and `None`; Seed-OSS's writes every value with
  `{{ value }}`, so an object comes as Python's repr. Inference reads both, so declaring a
  `number`, `boolean`, `array` or `object` changes nothing.

A value read as JSON keeps the model's own bytes: `[1, 2.5]` stays spaced as written, and
`9007199254740993` keeps every digit. JSON here means what `serde_json` reads into a value from
inside the arguments object, because that is how the adapters read a call's arguments back. Text it
refuses there, such as a lone surrogate escape or nesting that reaches its depth limit once the
object is around it, is a string like any other text, so one odd value never costs a call its other
arguments.

Ported from `safe_val` and `coerce_value` in `crates/tool_parser/src/parsers/qwen_xml.rs` and from
`coerce_by_schema_type` and `param_types_for_function` in that crate's `helpers.rs`. Three things
differ. The first two are how the templates write values and what bellwether's references record:

- A string keeps its text exactly. The old functions trimmed it, and unquoted a value declared
  `string` that happened to be a JSON string literal, so a search for `"exact phrase"` lost its
  quotes and a file's contents lost their final newline.
- JSON keeps the model's bytes. The old functions parsed the value and wrote it again compactly,
  so `{"a": 1}` reached the client as `{"a":1}`.
- The old functions had a rule each for `number`, `boolean`, `array` and `object`. Each gave
  exactly what inference gives, so they are not carried over.

## The pythonic assembler (`pythonic`)

What the syntax says, and what the assembler does with it:

- **A call is a name, `(`, arguments, `)`.** The name's bytes and the `(` are `ToolCallStart`'s
  source; the `)` is `ToolCallEnd`'s. A list's brackets, the commas between calls and the
  whitespace around them are the template's and come back as `Dropped { Wrapper }`; any other
  text where a call should begin is `Malformed`.
- **Each argument is `key=value`.** The value is a Python expression the template wrote for a
  JSON value: a string in either quote, a number, `True`, `False`, `None`, a list, a tuple, a
  dict, or the JSON a template writes with `tojson` for a nested value. Its extent is known
  when a `,` or the call's `)` arrives outside every string and bracket, so a value is held
  until then and written whole, through the value module's literal reader; a value that is
  not a literal is a string holding its text. Streaming a long string value as it arrives is
  a later step: the templates that write Python write short values, and the extent rule is
  what makes every chunking say the same.
- **Every byte lands in exactly one event**, as in the other assemblers: the key, the `=`, the
  value and the separator after it in the fragment they produce.
- **Two endings.** `Assembler::close` is the region's closing marker (or the next opener):
  a call still open is closed for the client, an open argument's bytes and a name that never
  got its `(` come back as `Malformed`, and the list's wrapping before any call is dropped.
  `Assembler::finish` is a stream that was cut: nothing is closed, what was held comes back
  as `Malformed { UnterminatedRegion }`, and a call that started still ends, with no bytes of
  its own, as every assembler ends one.
- **Python's whitespace is allowed** between a name and its `(`, around the `=`, and around a
  value, as the old parser's regular expression allowed it; a name or a key is the text less
  that whitespace.

Ported in spirit from the old crate's pythonic parser, which compiled a regular expression per call
and read the arguments with a Python-literal evaluator at the end; here the arguments are read as
the stream arrives, at the value's extent.
