# Notes on the assemblers

Each call syntax has an assembler that turns a call's bytes into its events. The module docs say
what each one reads and show one call; these notes say what each syntax says and what the assembler
does with it, byte accounting included, and where a rule comes from.

## The JSON assembler (`json::assembler`)

Every byte of the object lands in exactly one event: the head up to the arguments value is the
`source` of `ToolCallStart`, the argument bytes are the `source` of their fragments, the tail after
the value is the `source` of `ToolCallEnd`. When the arguments come before the name they are held
until the name is whole, since a call starts before its arguments. An object that ends without an
arguments member (its key never came) is a call with none: one fragment, `{}`, of no bytes goes
before the end, so the call reads as one written `"arguments": {}` and as every other syntax's call
with nothing in it, on every path that reads the fragments (a client of the stream, the message, a
server call made from the arguments). A member whose key came and whose value never began is not
one: the call ends with what arrived, and nothing is made up for a value the model botched.

The assembler keeps the promise `ToolCallArguments` makes, that the fragments so far are a prefix
of JSON a strict reader accepts until the parser says otherwise: each new run of argument bytes is
judged by `Prefix` before it is emitted, and at the first byte a strict reader refuses, one
`Malformed { InvalidArguments }` event of no bytes says that the arguments stopped being JSON
there; the bytes from the break to the value's end are fragments still, the model's own as written,
nothing already emitted being revised (`json::fragments`). The call so carries exactly what the
model wrote, and says once that it is no JSON, whole and in pieces alike: a client that reads the
arguments as JSON is told, at the byte where its own reader would refuse them, and one that shows
them sees what the model wrote (`"end_range": 2.0 * Math.PI` where a number was due). Cutting the
fragments at the break and handing the rest back as malformed text, which the readers did before,
gave a client a call cut at the first bad token and the rest of its arguments as the answer's
content, and nothing honest can be done with either. The oracle for the break is the template
reference and the JSON grammar: the template writes a JSON object that the engine's own server
hands to a strict reader (`json.loads`, `serde_json`, a client's library), so a call its terminal
closed is whole only if such a reader takes its arguments. The judge therefore refuses what the
partial parser (`PartialJson`, the old crate's reader) tolerated and JSON lacks, each at its first
byte, whole and in pieces alike:

- a value left out before a comma or a close (`{"days": ,}`, `[,,]`) and a trailing comma before a
  close (`{"days": 3, }`): refused at the comma or the close that stands where a value or a key
  must;
- an escape JSON lacks (`"Pa\yris"`), a `\u` escape with fewer than four hex digits
  (`"\u12G4"`) and a surrogate escape without its other half: refused from the escape's
  backslash, which is held until the escape is decided (a letter JSON knows, four hex digits that
  are a character or a pair), so no fragment ends inside an escape and the fragments end before
  the one the reader refuses, whichever byte of it the reader names;
- a bracket closed by the wrong kind (`{"a": [1}`) and whitespace JSON lacks between tokens (a
  no-break space, an ideographic space): refused at the character;
- a number or a word cut short before a delimiter (`1.}`, `-]`, `nu}`): refused at the delimiter,
  or from the word's first letter, which the readers hold until the word is decided.

A raw control character inside a string is the one thing JSON lacks that is let through, escaped
rather than refused (the next paragraph), since the engines' own servers re-serialize what their
partial reader took and hand out the escape; nothing else is normalized, since no engine's reader
takes it. A value the outline ended and a strict reader does not take whole (a number or a word the
object's brace stops, `"arguments": 12.}`) is said to be none by a `Malformed` event of no bytes
before the call's end, as at the stream's end. The value reader (`json::arguments`) and the Mistral
object reader (`json::named_arguments`) judge and say the same. The contract tests
(`tests/parser/contract.rs`) check both halves of the promise under every chunking: a fragment never takes
a call's arguments past a prefix a strict reader accepts until the break is said, every fragment
after it is the model's bytes as written, and a call that closed without a `Malformed` event has
arguments a strict reader parses whole.

A raw control character inside a string (U+0000..U+001F: a newline in a multi-line value, a tab
in code), which JSON forbids there and no strict reader accepts, is written in the fragment as its
escape (`\n`, `\t`, `\r`, `\b`, `\f` or `\u00XX`; after a backslash the escape's own letter, since
the backslash already stands), while the fragment's `source` keeps the model's bytes: the fragments
stay a valid prefix and the whole is JSON a strict reader accepts, which is what the engines' own
servers hand out for the same bytes once they re-serialize what their partial reader took. The
prefix judge notes where it met one (`Prefix::escaped`), so a fragment with none costs nothing
more. The value reader (`json::arguments`) and the Mistral object reader (`json::named_arguments`)
write their fragments the same way.

Every byte is read once. The outline is taken by a `Scanner` that keeps its place between pieces,
and the prefix is judged on the bytes that arrived, so a call's cost is linear in its length however
the text is cut: the first version read the whole text again on every piece, and a 259 KB argument
string fed by the character took 800 s against 0.03 s whole.

A started call whose object never closes is closed at `finish` with what arrived, and the bytes
after its arguments value (a comma cut short, a complete member, or bytes that are no member) come
back as `Malformed` with `UnterminatedRegion`. An arguments value the close itself cut (a string or
a container left open, a word or a number the bytes end on; the next call's opener, the region's
closing marker and the end of the stream all cut this way) is said to be none: the fragments sent
stand, since a fragment passed on cannot be taken back, nothing is made up for a value the model
never finished, and a `Malformed` event of no bytes comes before the call's end, so that a client
that reads the arguments as JSON is told rather than left to find out; the engine names the reason
for a region a marker closed. The value reader (`json::arguments`) and the Mistral object reader
(`json::named_arguments`) say the same at their end; the Gemma assembler, which holds a member's
key until its value begins, reports the key of a member the end cut after its colon and closes the
object whole without it. Whole arguments that only the object's brace is missing after are a call
with no word.

`feed` takes one piece and returns how many of its bytes the object took: all of them before the
close, and up to the closing brace in the piece that carries it, so the format routes whatever
follows the object (a closing marker, more text) itself. Bytes fed after the close are not taken.

What the assembler does not decide: the call's index and id (the format mints them and passes them
to `Assembler::new`), whether the name is a declared tool, and whether a string-valued arguments
member should be decoded. The format decides those with the events in hand.

## The named reader (`json::named`)

A call written as its tool's name, a marker, then JSON is read by one assembler whose parameter
(`After`) says what the JSON is: the call's object (Inkling's `<|content_invoke_tool_json|>`, after
which the object names the function again) or the arguments object alone (Mistral's `[ARGS]`). The
header is the same either way (`json::header`): the bytes before the marker are held until the
marker's last byte arrives, the search covering only the new bytes and the marker's length before
them, so a name that never meets its marker costs each piece its own length. What differs is said
once, at the marker: with the object after it the header is the template's wrapping, `Dropped`,
and the JSON assembler reads the object and starts the call at its name; with the arguments after
it the header names the call, `ToolCallStart` carries the name with the marker as its source, and
the arguments-object reader (`json::named_arguments`) streams the object's bytes as the fragments,
accounted for through the JSON assembler's judge. Both end the call at the object's brace and
leave the bytes after it to the engine. A name the region ends after with no marker is reported by
the reason the syntax names (`NameWithoutACall`, `NameWithoutArguments`), and as
`UnterminatedRegion` when the stream is cut; an empty header leaves nothing, which is how a thought
or an answer message reads under the Inkling table.

## The list reader (`json::list`)

A list of call objects is read by one assembler around the JSON assembler, each object going to
one of its own with the next free index (spent only by an object that starts a call) and its id
from the engine's `CallIds`. What frames the calls is one value, `Framing`, given at construction
and mapped from the table's call syntax by the engine's factory: `Bare` (xLAM) is the list alone in
the output, where anything but a list is prose; `Framed` (Llama Nemotron) is the list between
markers, where text in place of it is malformed; `ObjectOrList` (the `json` table) also reads a
`{` at the first byte as one call alone and every further object after it, behind a `;` or behind
whitespace alone, as another call, and carries the family's call marker the output may open with
(Llama 3.1's `<|python_tag|>`), dropped as the template's wrapping where it does. The stages
(deciding at the first byte that is not whitespace, inside the marker, between objects, inside one,
after the `]` or the lone object) and what they say of the bytes around the objects (the brackets,
the commas, the `;` and the wrapping whitespace `Dropped`; text between objects `Malformed`, one run
per event, a run ending at whitespace, a comma, a brace or the `]`, the wrapping between two runs
the model's and in the second run's event (`event::release_between`); text after the list or the
lone object `Malformed` with its whitespace, one event per piece once text follows) are the same
under every framing.

The JSON family so reads as one object reader with the framings around it, each stated once:
none, where a table's markers frame one object (`CallSyntax::Json`, the JSON assembler itself); a
header (`json::named`, whose `After` says what the JSON after the marker is); a list (`json::list`,
whose `Framing` says what is around the objects). The lone object between markers is not a framing
of the list reader: the engine, not the reader, reports what follows the object there
(`Engine::wrapping`, per piece), where the list reader holds the whitespace after a lone object
across pieces and reads a `;`, another object or prose after it; folding it in would change those
events, or put a branch on the busiest JSON path for no reader fewer. The forced call
(`json::forced`) reads through the value reader (`json::arguments`), which ends where the value is
whole and holds the strings a caller types, not at an object's brace: a different reader by design,
named by the request rather than framed.

## The Qwen tagged assembler (`tagged::assembler`)

Every byte of the call lands in exactly one event. Whitespace between tags is the template's and
goes into the source of the next event: the function tag's into `ToolCallStart`, a parameter tag's
and the value's text into the fragments they produce, the rest into the closing `}`. Anything else
between tags, a tag out of its place, a tag that cuts a name short and a name that is empty is
refused (`tagged::refuse`): counted by a `Malformed` event of no bytes with the reason, the bytes
dropped as a malformed call's (`Dropped { MalformedCall }`), never content, so the model's bytes are
never read as something they are not, never shown as the answer, and never vanish into a source.
Inside a value only `</parameter>` is a tag; `<function=`, `<parameter=` and `</function>` there
are the value's text (the engine's parser ends a value at the next `<parameter=` or `</function>`
as well, a tolerance to judge with adversarial fixtures rather than port).

Two endings. `Assembler::close` is for a block the model ended before `</function>` (its own closing
marker came, or the next call began): the call is over as far as the model is concerned, so after
the last value the object is closed, and a value held for its close is refused, its member never
written. A streamed string the close finds open, a byte of whose value went out, is
closed first, as the stop closes one (the third ending, below): the model ended the call, so the
value's bytes are the value whole, the start of a tag held is its text since no tag followed, the
newline held as the template's framing is the closing quote's source, and one `Malformed {
OpenAtTheMarker }` event of no bytes stands before the quote (`tagged::left_open`), so
that a call delivered under `tool_calls` reads as JSON; a string no byte of the value followed is
left open, since nothing is made up for a value the model never wrote: the fragments stand, and the
one `Malformed` event before `ToolCallEnd` carries what was held or no bytes and says that the
arguments are no whole value, as at `finish` (`tagged::end_cut_call`); the engine names the reason
for a block a marker closed.
`Assembler::finish` is for a stream that was cut: nothing is closed, as in `crate::json::Assembler`,
because arguments cut short must not look complete to a client, and the call's bytes stay the
call's. A value held for its close that the cut found open leaves as the call's own fragment, its
key and its bytes as the model wrote them (`, "prefs": {"size": "large", "temp`; a section that
closed is written whole), since a client reads the prefix as written under `length`, the way the
JSON assembler sends the word it held, and never the call's bytes as the answer's prose. The bytes
go as written where they are a strict prefix of a JSON value, as the JSON of a Python literal the
cut found complete (`True` as `true`, the close's reading) and as the opening of a JSON string
holding them where they are neither (`"None]`), so that every fragment keeps the arguments a
strict JSON prefix, the crate's contract (`tagged::value::cut`). Before this the held value left
as `Malformed` text, the parameter tag included, and a chat client read
`<parameter name="prefs">{"size": "large", "temp` as content next to a call whose arguments stopped
at the member before. A string whose key came and no byte of it opens, `{"key": "`; a parameter tag
the cut found open and the template's whitespace between tags leave as wrapping (`Dropped`), as the
stop drops a tag the end cut short; the start of the closing tag held behind a value is a tag the
end cut short too, in the families whose tags are text, dropped after the call's end so every
byte keeps its place, and never the value's text (a held start of the opening tag is the value's
text, as the whole tag is inside a value, and so is a held strict prefix of a special token where
the tags are built of them, which the model cannot write and no cut splits): the limit fell while
the model was writing, and the start of
the closing tag at the end of its output is that tag (a cut inside a string value's closing tag
used to deliver the tag's bytes as the value's, `"from_currency": "KRW</｜DSML｜ parameter`; the
value is `"KRW`). At `finish` a started call's arguments are no whole
JSON whatever stage the cut found, since the object closes only at the call's own end, and the call
says so as the JSON assembler does: one `Malformed { UnterminatedRegion }` event of no bytes before
`ToolCallEnd`. The fragments stand, since a fragment passed on cannot be taken back, and nothing is
made up for a value the model never finished; the four assemblers of `tagged` share that end
(`tagged::end_cut_call`), and what a block that named no call held is refused as
`UnterminatedRegion` by every one of them, so that the reader's release of a refused output that
made no call takes the whole slice.

A third ending, the stream's stop. `Assembler::stop` is for the model's own end of output inside
the block (`EngineFinish::Stop`; a cut for length, an abort or any other finish is `finish`): the
call ends as the block's end would end it, and a streamed string still open is closed first, its
quote and then the object's brace, since the model wrote nothing after the value's bytes, so they
are the value whole, and a call delivered under `tool_calls` has to read as JSON. The start of a tag
held at the stop is the value's text, since no tag followed, unless it begins with one of the
family's structural tokens (XTML's `<|close|>argument`, `Tags::structural_tokens`), which the model
cannot have written as text: that tail is a tag the end cut short and is dropped after the call,
as the engine drops its own terminals' held prefixes; a newline held as the template's
framing before the closing tag stays the template's, the closing quote's source. One `Malformed {
OpenAtTheStop }` event of no bytes stands before the fragment that closes what was open: for a
string the stop closed under any table, and for the call itself where the table has no stop row,
since there the template writes the closing marker the model skipped (under the row the stop
stands for that marker, and a call open at it is the expected shape; `docs/engine.md`). A value
held for its close (a number, an object, a string that may still be null) and a string the model
opened without writing a byte of it keep the marker's path: refused, the member never
written, or, where its opening fragment went out with the parameter tag (the Qwen assembler), left
open and said as the block's end leaves it; nothing made up. The two differ by what the client has
seen: a streamed string's fragments already reached it, so closing the string adds the quote alone,
while a held value's bytes reached nobody and its close is a tag the model never wrote, so the stop
writes no member of it; a held scalar that reads whole as its declared kind (`1.5` under `number`)
is refused the same way today. A cut keeps `finish`'s, the prefix as
written under `length`, the held value in the call. The four
assemblers of `tagged` share this ending (`tagged::open_at_the_stop`); the JSON, pythonic and Gemma
assemblers, whose quotes and braces are the model's own, end a call the stop found open as a cut
ends it. The shape that decided it, recorded on GLM-4.5V: a call whose string value the model ends
with its end-of-box marker, a newline and the end of its output, writing neither `</arg_value>` nor
`</tool_call>`; the family's own parser drops the pair, and the prefix as written reached clients
as `tool_calls` with arguments no JSON reader takes. The live stream of that shape carries the
call's closing marker after the open value (the model skips `</arg_value>` alone, then writes
`</tool_call>` and its turn's stop token, which the engine hides), so the marker ends the call
before the stop does: the four assemblers close a streamed string at their closing marker the
same way, under `Malformed { OpenAtTheMarker }`, the one word that tells the two ends apart.

`feed` takes one piece and takes every byte of it: the block's calls follow one another, and the
block's closing marker is the format's to read. The templates write one function block per call
region, and a model answering with several calls at once writes them one after another inside one
region, which the engines' own parsers read as the calls they are; so after `</function>` the
assembler stays in the block. Whitespace there is the template's and goes into the next event's
source, the next `<function=` starts the next call at the next index with the id that index mints,
and anything else is refused under `TextBetweenCalls`. A region that ends after its last call
closed ends nothing: the whitespace after the call is dropped, and a tag the end cut short is
refused, under `UnterminatedRegion` at the stream's cut.

## The keyed assembler (`tagged::keyed`)

What the syntax says, and what the assembler does with it:

- **The name is the text before the first `<arg_key>`**, or before the call's end when the call
  has no arguments, less the whitespace around it, which is the template's and goes into the
  next event's source. The name is known only when that tag arrives, so `ToolCallStart` waits
  for it; names are short. A name holds no whitespace, quote, brace, bracket, parenthesis or
  angle bracket: any other text there (the JSON of a `<tool_call>` block quoted inside a code
  fence, prose, a call written as Python) is refused, and no call starts, so a block that names
  no call writes no arguments either.
- **A value's type comes from the request's tools**, as in the Qwen assembler: a declared string
  is its text, streamed as it arrives; every other value is written whole at `</arg_value>`
  through `json`, which reads the JSON the templates write for a number, a boolean, a list,
  an object or null, and infers the rest. Nothing is taken from a value: the templates write it
  directly between its tags.
- **Inside a value only `</arg_value>` is a tag**; the other three there are the value's text.
- **Two endings.** `Assembler::close` is the call's closing marker, or the block's end before
  it: after the last value it names the call if it has not, closes the object and pushes
  `ToolCallEnd` with the marker's bytes; inside a value held for its close it refuses what was
  held and closes the object without that member; inside a streamed string it closes the string
  first, the start of a tag held as its text, under one `Malformed { OpenAtTheMarker }` event of
  no bytes before the closing quote (`tagged::left_open`), since the model ended the
  call and the value's bytes are complete, then the object. `Assembler::finish` is a stream that was cut, by the token
  limit and not by the model: nothing is closed, and the call's bytes stay the call's, since a
  client reads the prefix as written under `length` and the member is never lost. A value held
  for its close leaves as the call's own fragment, its key and its bytes as the model wrote
  them (`, "new_preferences": {"size": "large", "temp`) where they are a strict prefix of a
  JSON value and as the opening of a string holding them where they are not, so that every
  fragment keeps the arguments a strict JSON prefix; a start of a tag held behind the value is
  its text too where the two together are still such a prefix (`{"q": "a <`) or where the
  value opens as a string anyway (`hel<`), and leaves after the call's end as wrapping where
  the text alone is a JSON prefix or a complete Python literal (`12</arg_v`, `True</arg_v`); a
  streamed string keeps its fragments, a held tag start its text; a string whose key came and
  no byte of it opens, `{"key": "`; a key whose two tags came and whose value tag did not
  opens its member, `"key": `, and a start of the value tag leaves after the call's end as the
  syntax's wrapping; a key tag the cut found open and the template's whitespace between pairs
  leave as wrapping too (`Dropped { Wrapper }`, never `MalformedCall`: a cut is not a failed
  call). A started call then ends as the Qwen assembler's does, after one `Malformed {
  UnterminatedRegion }` event of no bytes that says its arguments are no whole value
  (`tagged::end_cut_call`); a block that named no call has what it held refused.
  `Assembler::stop` is the stream's stop, the model's own end: the call ends as the marker
  would end it, a call not yet named is named from its text, a streamed string still open is
  closed first, with the start of a tag held as its text, and the one `Malformed {
  OpenAtTheStop }` event says so (the third ending of the Qwen assembler, above).
- **Nothing the syntax cannot read is content** (`tagged::refuse`). The block's markers and its
  four tags are the format's own tokens, and the text between them where the syntax has no place
  for text (a value whose key never opened, a tag written twice, a name that is prose, Python or
  JSON) is the model's failed call: the families' own parsers drop such a block whole, and a
  client shown it saw the format's tags at the end of the answer (`</arg_key><arg_value>1975
  </arg_value>`, in 20 of 292 answers of a live cell). Such bytes are refused: a `Malformed`
  event of no bytes with the reason stands where the bytes do, so a reader counts the failure at
  its offset, and the bytes leave as `Dropped { MalformedCall }`, never as content (at a cut
  call's end the bytes held come first and the one event of no bytes stands after them, before
  `ToolCallEnd`, as `tagged::end_cut_call` has it); a call that started keeps the pairs it read
  (`{}` when none), and a block that named no call has every byte refused, the marker that
  closed it included. The attributed assembler (`tagged::attributed`: DSML, XTML, `<parameter
  name="…">`) refuses the same way everything inside an invoke its syntax cannot read: a word
  between tags, a tag out of its place or cut short, a tail a tag has no place for, an invoke
  that named no call (whole, its closing tag included); a call that started keeps the members it
  read. (A parameter tag written without its `<` and a call's arguments written as JSON and
  closed with a region of the model's own, both seen live, reached the client as content with
  the format's tokens in it before.)
- **Every byte of the call lands in exactly one event**, with the Qwen assembler's accounting:
  the name's bytes in `ToolCallStart`, each tag's and value's bytes in the fragments they
  produce, the whitespace the template writes between tags as `Dropped { Wrapper }`, and what
  the syntax cannot read as `Dropped { MalformedCall }` behind its `Malformed` event of no bytes.

## The attributed assembler (`tagged::attributed`)

What the syntax says, and what the assembler does with it:

- **The request's tools type the value** (`Typing::Declared`), as in the keyed assembler: a
  declared string is its text, every byte of it, streamed as it arrives; every other value is
  written whole at `</parameter>` through `json`, which reads the JSON the template writes
  for a number, a boolean, a list, an object or null (`tojson`), and infers the rest. A
  parameter tag with anything between the name's closing quote and `>` is no parameter tag the
  syntax has: the whole tag is refused (a `Malformed` event of no bytes, the bytes as `Dropped {
  MalformedCall }`), and no parameter opens.
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
  tag, which the template never writes, is refused; a value held whole for its close is written
  at the first word of it, as the section has it, so that the refusal never takes the bytes it is
  held in, and the closing tag then leaves as the syntax's wrapping (`Dropped { Wrapper }`).
- **A value may stand on its own line** (`Tags::value_on_its_own_line`): dots3's template
  writes a newline after the parameter tag and one before its closing tag. Those two are the
  template's framing and go into the fragments' sources, not the value: the first byte of a
  value is dropped when it is a newline, and a trailing newline is held until the closing tag
  says it was the framing, or more text says it was the value's.
- **The newlines between the tags are the template's** and go into the next event's source;
  text where the syntax has tags is refused (counted, never content).
- **Two endings**, as DSML's: `Assembler::close` is the invoke's closing tag, or the block's
  end before it, and inside a streamed string closes the string first, as the keyed assembler's
  does (`tagged::left_open`), except that a held tail beginning with one of the family's
  structural tokens is a tag the marker cut short, never the value's text, and is dropped before
  `ToolCallEnd`, whose bytes came after it (at the stop, whose end carries no bytes, it is
  dropped after the call, as above); `Assembler::finish` is a stream that was cut, and closes
  nothing: a started call ends after the one `Malformed { UnterminatedRegion }` event that says
  its arguments are no whole value (`tagged::end_cut_call`), and an invoke the cut found before
  it named a call is refused whole as `UnterminatedRegion`, as the keyed one is, so that the
  reader's release of a refused output that made no call takes the whole slice.
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
  do, up to the nesting the JSON reader allows every syntax (`PartialJson::DEFAULT_MAX_DEPTH`
  containers, the arguments object counted): the tag that would make an element the container
  past that bound is refused with every byte from the element's own tag to the call's end, as
  one `Malformed { InvalidArguments }`, nothing of that member is written, and the containers
  open close whole at the end, so the arguments stay a value a strict reader takes and the
  element stack never grows past the bound; a leaf whose own JSON (a list or an object its
  text spells) would nest past the bound, and an empty element the tool declares a container
  at the bound, are refused the same way, so the bound holds for the value the client reads,
  not only for the tags. Whitespace right after an opening tag is held
  until the element's next byte says what it is: a leaf's text when text or the closing tag
  follows (a value that is one space is one space), the template's when a child tag follows.
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
  closing tag is the leaf's text; a closing tag that matches nothing open is refused, as is text
  where the syntax has tags (`tagged::refuse`: a `Malformed` event of no bytes counts it, the bytes
  leave as `Dropped { MalformedCall }`, never content).
- **The name ends at its quote**, and the invoke tag at `>`; text between the two is refused,
  as for DSML. An invoke with no name starts no call, and it is refused whole, the terminal that
  ended it included.
- **The separator keeps its place.** When nothing is held before it, it is `Dropped { Wrapper }`
  at once; when a value's text or an opening tag is held, its bytes go into the next
  fragment's source, so every event's bytes stay in the output's order.
- **Two endings.** `Assembler::close` is the invoke's closing tag (`INVOKE_CLOSE`, the one
  terminal a started call's end carries), or the next invoke's opening or the block's close,
  which end the call with no bytes of their own: every open container is closed, a tag the
  end cut short, a leaf never written and an element without a child are refused, and the
  arguments object is closed; a streamed string innermost is closed first,
  the start of a tag held as its text, under one `Malformed { OpenAtTheMarker }` event of no
  bytes before its quote (`tagged::left_open`), since the model ended the call and the
  value's bytes are complete, then the containers above it. An invoke that
  named no call takes whichever terminal ended it, so that it is
  refused the same way however it ended. `Assembler::finish` is a stream that was
  cut: nothing is closed, and a call that started still ends, with no bytes of its own, as
  every assembler ends one, after the one `Malformed { UnterminatedRegion }` event that
  carries what was held, or no bytes, and says that its arguments are no whole value
  (`tagged::end_cut_call`); what an invoke that named no call held is refused as
  `UnterminatedRegion` too, so that the reader's release of a refused output that made no call
  takes the whole slice.
- **Every byte of the invoke lands in exactly one event**, with the tagged assemblers'
  accounting: the name's bytes in `ToolCallStart`, each tag's and value's bytes in the
  fragments they produce, text where the syntax has tags refused run by run
  (`Dropped { MalformedCall }` behind its `Malformed` event of no bytes), the whitespace right
  after a tag the syntax read as `Dropped { Wrapper }` before the first run and the whitespace
  between two runs, the model's, in the second run's refused bytes (`Assembler::refuse_between`).

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
- **Two endings, and a cut call closes at neither.** `Assembler::close` is the region's closing
  marker (or the next opener); `Assembler::finish` is a stream that was cut. A call still open
  at either is one the model never finished, since every template writes the call's `)` before
  the marker: nothing is closed for it, the fragments it sent stand, and it ends, with no bytes
  of its own, after the one `Malformed` event that says its arguments are no whole value
  (`tagged::end_cut_call`); so a cut call is never handed over as finished without a word, nor
  with an object the reader closed for it. What was held when the end came differs: at the
  close, the model's own end, the word carries it, a key, a value or a separator cut short, or
  no bytes when the cut came right after the `(`, and the engine names the reason for a region
  a marker closed; at the stream's end (a cut by the token limit, an abort, or the model's own
  stop, which ends a pythonic call as a cut does, as the stop rule above has it), the word has
  no bytes and the call's bytes stay the call's, since a client reads the prefix as written
  under `length` and never the call's bytes as the answer's prose: the member the cut found
  inside its value leaves as the call's own fragment, its key and the JSON the assembler would
  have written for the value as far as that is known (an open string in either quote as an open
  JSON string holding its text so far, decoded as the dialect reads it, `{"city": "Par`, an
  escape the cut left unfinished left out; a value whole before the separator that never came
  written whole, `"days": 3`; a number's digits, the start of a literal word (`Tr` as `tr`)
  and a list or a dict still open as the JSON they stand for so far, which is the text as
  written where that is a prefix of a JSON value and a string holding the text where it is not,
  so that every fragment keeps the arguments a strict prefix of a JSON value; a key and no byte
  of its value as `"key": ` alone), and a key or a separator the cut found open
  leaves as the syntax's wrapping (`Dropped`), as a parameter tag the end cut short does under a
  tagged syntax, since no member can be written for it. Before this the held bytes left as the
  word's text at both ends, and a chat client read `city="Par` as content next to a call whose
  arguments stopped at the member before. Outside a call the two endings differ: at the close a
  name that never got its `(` comes back as `Malformed` with its own reason and the list's
  wrapping before any call is dropped; at the stream's end what was held comes back as
  `Malformed { UnterminatedRegion }`.
- **Python's whitespace is allowed** between a name and its `(`, around the `=`, and around a
  value, as the old parser's regular expression allowed it; a name or a key is the text less
  that whitespace.
- **A list bare in the turn** (`Assembler::bare`, which the engine picks for a table no row of
  which enters the arguments state, as it picks the bare JSON list): a `[` is held with the name
  after it until a `(` says it opens a list, or anything else gives it back as content. At the
  turn's start, after whitespace alone, the list then reads as between markers, as it arrives.
  After content, a list whose first name no tool declares is content from its `(`; any other list
  is held until the `]` that closes it (strings and nested brackets scanned, given back as content
  past 256 KiB) and read whole by an assembler of its own: when every byte is a call's or the list's
  wrapping and every call names a tool the request declares, its events are the turn's; when any
  byte is malformed (a positional argument, an operator, prose between calls) or a name is no
  tool's, the whole is content, since code holds `[name(` of its own. After the list's `]`
  whitespace alone is wrapping; once text follows, every byte after the `]` is the model's prose,
  whitespace included, malformed as after xLAM's list, one event per word with the whitespace
  before it. A stream cut while a `[` or a list is held gives it back as content, since no call
  began.
- **A dialect whose quotes carry no type** (`Values::Quoted`, Llama 4's template writing `"%s"`
  of every value): the text between a value's double quotes, as written and with no escape
  decoded, is typed by the kind the request's tool declares for the parameter, through the value
  module's `json`, or by its text where none is declared; a bare value, a single-quoted Python
  string included, reads as under the other dialects.
- **A parameter without a value is reported and skipped.** `key=` followed by the comma or the
  call's `)` is a member Python refuses and the templates never write: its bytes, the separator
  before it, the key, the `=` and the whitespace around them, come back as
  `Malformed { KeyWithoutAValue }`, nothing is
  written for it, and the object closes whole without it, as the Gemma reader leaves a member
  without a value; no empty string stands for a value the model did not write.

Ported in spirit from the old crate's pythonic parser, which compiled a regular expression per call
and read the arguments with a Python-literal evaluator at the end; here the arguments are read as
the stream arrives, at the value's extent.

## The Gemma assembler (`gemma`)

What the syntax says, and what the assembler does with it:

- **A call is `call:NAME{`, members, `}`** between the table's markers: the header is
  `ToolCallStart`'s source, the closing marker `ToolCallEnd`'s. The template writes a member as a
  key, a colon and a value, the keys bare; a value is a string between two string tokens, a
  number, `true`, `false` or the null word bare, a list or an object; a comma stands between two
  present members only. The assembler writes the members as JSON as their bytes arrive, a string's
  text streamed.
- **A member is written from its value's first byte.** The key and its colon are held until a
  value is there. A member without a key (a colon where a key should stand) or without a value (a
  comma or a close where the value should start), which the template never writes, is reported
  and not written: every byte of it comes back as `Malformed` under the one reason, the comma
  after it and the containers opened inside it included, and the object closes whole without it,
  its next member first, as the tagged assemblers leave a parameter without a name; a list item
  missing before a comma the same. Nothing is made up for such a member: no empty key, no empty
  string.
- **Two endings.** `Assembler::close` is the call's closing marker, the model's own end: a key or
  a bare value cut short, and the key held for a value that never came, come back as `Malformed`
  with the cut's reason before the end, every open container is closed and the arguments are
  whole; a string left open stays open, the fragments stand, and a `Malformed` event of no bytes
  says that the arguments are no whole value. `Assembler::finish` is the stream's end inside the
  call: a cut by the token limit, an abort, or the model's own end of output before the closing
  marker, which ends a Gemma call as a cut does since its braces and tokens are the model's own
  (the stop rule above; neither Gemma table has a stop row), so a call the model left open at
  its own end stays open too, where it used to be closed. Nothing closes, as under every other syntax, since a
  closing brace the model never wrote would hand the client arguments that read as whole, and
  the call's bytes stay the call's, since a client reads the prefix as written under `length`
  and never the call's bytes as the answer's prose. The member the cut found where its value
  stands leaves as the call's own fragment, its key and its bare value as far as it went (`,
  "n": 1`; a word that is no literal as the opening of the string it would be written as, so
  that every fragment keeps the arguments a strict prefix of a JSON value) or the key alone (`,
  "n": `); a string's text so far, the bytes held as a start of
  its closing token included, is the string's, as at the marker; a key cut short, the whitespace
  held after a value and a start of the string token held after a key leave as the syntax's
  wrapping (`Dropped`), since no member can be written for them; the one `Malformed {
  UnterminatedRegion }` event of no bytes stands before the call's end, which carries no bytes.
  An object the model closed before the cut is whole and needs no word. Before this the
  assembler closed the containers at the stream's end and reported what it held, so a client read
  `{"artist": "Taylor"}` beside `n:1` as content for a call cut inside its second member.
- **Every byte lands in exactly one event**: the header, each fragment's bytes with the whitespace
  before them, text after the object closed as `Malformed` run by run, the whitespace right after
  the object and the run before the marker as the template's wrapping (`Dropped { Wrapper }`) and
  the whitespace between two runs, the model's, in the second run's event
  (`event::release_between`), the marker as the end's source.
