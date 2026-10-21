# Architecture

How the crate is built, what each part promises, how a table is added without touching the engine,
and how a failure is classified and surfaced. The other notes go deeper into one part each:
`engine.md` (the decisions the engine makes for every table), `tables.md` (each table), `assemblers.md`
(each call syntax), `grammar.md` (the structural tags), `adapters.md` (the three API shapes). The
README's models table says which checkpoint groups each table reads.

## The shape

Symphony is one state machine, run over data. A model family's output format is a **table**
(`Format`): the marker strings (terminals), the states the text between them is in (content,
reasoning, a call's arguments, the template's wrapping), the rows (`state + terminal = state`), one
call syntax for the arguments state, and a few rows a response template needs (the turn's anchors,
a start state, framing whose text is dropped, stays the template trims, a stop that closes a call).
The **engine** (`Engine`) runs any table: a scanner splits the output into text and terminals, the
table says where a terminal moves and what the text is, and inside an arguments state the call
syntax's **assembler** turns the call's bytes into call events. The **events** (`Event`) are the
contract with every caller. The **grammar** derives an engine's structural tag from the same table
and the request's tools. The three **adapters** render events in the Chat Completions, Responses and
Messages shapes, streamed and whole. The **registry** maps a model id to its table.

```text
                 registry (model id -> Table -> Format)
                               |
  prompt tail ---------------> |            request tools
  decoded deltas + spans ----> Engine(Format) ----------> Events ----> adapt::{chat, responses, messages}
  end of stream (finish) ----> |      |                                     |
                               |   assembler of the call syntax             +--> the API response
                               |
  request tools -------------> Format::grammar / answer_grammar / turn_grammar --> the engine's constraint
```

Dependencies point one way. A table knows nothing of the engine: it is built with a small builder
and read through accessors, and no engine code dispatches on a table's name (checked: the engine
and the grammar core match on states, terminals and the call syntax, never on `Format::name`). The
engine knows the assemblers through one crate-private trait (`CallAssembler`: `feed`, `ignored`,
`start`, `closes`, `close`, `finish`) and picks one from the table's call syntax. The grammar reads
the table's rows (which terminal enters a region, which closes it, which states a block is entered
from) and dispatches on the call syntax for the spelling inside a call. The adapters read events
only: the whole-response shapes share one fold of the sequence (`adapt/fold.rs`), and each stream
folds as it goes. The one crate outside this one the library depends on holds the request's tool
shape and the three APIs' response shapes.

## What each part promises

**The events.** Every byte of the output lands in exactly one event, in the output's order:
content, reasoning, a call's name (`ToolCallStart.source`), a fragment of its arguments, bytes the
format consumed (`Dropped`, with a reason), or bytes of a region the parser gave up on (`Malformed`,
whole, with a reason). `Finish` comes last and once. Calls are numbered from zero as they start, one
start per index, arguments and an end only for a started call, and the fragments of one call form
a valid JSON prefix until a `Malformed { InvalidArguments }` event of no bytes says where they
stopped being one, from which they are the model's bytes as written, and equal the final arguments
when the call ends. What the engine says does not depend on how the output was cut into deltas,
nor on empty deltas between them. With token spans, every token is counted once, in the event that
carries its first byte. Events a caller took are final: later text starts a new event and never
merges into one already drained. These are checked for every registry table by `tests/contract.rs`.

**The table.** `Format::validate` holds every table to the rows the engine can run, and a table is
checked when it is built (the registry's tables in the crate's tests, a compiled response template
by its compiler) and once more by `Engine::new`: a table with no state; a first state that emits
reasoning (a thought the output starts inside is entered by the prompt's replay or by a start row,
never by the table's first state); an arguments state with no call syntax; a terminal with no text;
an empty turn opener or anchor, or an opener beside anchors (a table names one or the other); a
start row where the first state holds a call, or beside a turn opener (which replays the turn's
start itself); a stop row on a state that holds no call; framing, trim or
kept rows on states they do not fit (framing on a call's state, a trimmed state that is framing, a
kept state that is trimmed); a table that thinks first without a thought; a shaped call whose
pattern a stream cannot decide, or that a table could not open through its terminal (the turn may
start inside one, an ignored terminal would break its header, or a row from one call's state into
another's where the header may not match). The error text names the table and the row, so a table
that fails is a programming error caught by the crate's tests, never a request's failure.

**The engine.** A terminal with no row from the current state is text, as the model wrote it,
and where that text is the table's own content a `Malformed { MarkerOutOfPlace }` event of no
bytes says so; a terminal with a row is `Dropped { Wrapper }` (or content, where a table's variant shows the markers
of a channel the request does not separate) and moves the engine. A call region the model closes
early is closed for the client (an open string gets its quote, the object closes); a region the
stream cuts is left open, so cut arguments never look complete. The whitespace a content stay begins
with is held until text follows it (content) or the stay ends (`Dropped { Whitespace }`), so a
stream shows what the whole answer has; a trimmed state drops the whitespace at a stay's edges; a
kept state shows whitespace as written. An ignored terminal is dropped where it stands and moves
nothing, its place among held bytes kept. The prompt is replayed from the last turn anchor the table
names, so a marker quoted in an earlier turn moves nothing. `engine.md` lists each decision with its
reasons.

**The assemblers.** Each turns one call syntax into call events with the same accounting: every
byte of the call in exactly one event (the name's in the start, each tag's and value's in the
fragment it produces, the template's whitespace between tags in the next event's source, anything
else between tags as `Malformed` run by run so that where the chunks were cut changes nothing).
Two endings: `close` for a call the model ended (its closing marker, the next call's opener or the
block's close), which closes the object for the client; `finish` for a stream that was cut, which
closes nothing. A value's type comes from the request's tools (`Declared`) where the syntax writes
values as text, or from the syntax where it writes JSON; a value's bytes are the model's own,
never re-serialized. `feed` returns how many bytes the syntax took, so the engine routes what
follows a JSON object itself. The ids of the calls are the engine's to mint, not the assembler's.

**The adapters.** One fold for the whole shapes, three renderings, and three streams that fold as
they go. The bytes a format consumed (`Dropped`) never reach a client; `ReasoningStart` writes
nothing, and `ReasoningEnd` and `ToolCallEnd` write no bytes of their own: Chat Completions renders
nothing for them, Responses and Messages close their item or block on them. `Malformed` text is
content, since the model wrote it and a client should see it rather than lose it; whitespace-only
content opens nothing; after at least one call, Chat Completions reports a natural stop as
`tool_calls` (`length`, `abort` and an engine's own reasons stay what the engine said), Messages
reports any reason but `length` as `tool_use`, and Responses completes the response as for any stop.
`adapters.md` lists where each API says less than the events do and what the adapter decides there.

## How to add a table

A family of a known call syntax costs a table and evidence, and no engine change:

1. Read the chat template. Write the table under `src/formats/<family>.rs`: the markers as terminals,
   the states and what their text is, the rows, the call syntax, the turn opener; unit tests on the
   recorded shapes (the family's own rendering of a thought, an answer, one call, two calls, a cut
   stream). The module doc says what the table reads; `docs/tables.md` holds what it decides beyond
   the others and what it leaves open.
2. Register it: the `Table` variant, its row in `Table::ALL`, its name and its builder
   (`formats/registry.rs`), its module and re-export (`formats/mod.rs`), and the checkpoint groups
   it reads, one line each in `formats/groups.json` (the table by its name, with each group's own
   turn opener where the template writes one the table does not, and its own text after the
   thought's close where the template writes other than the table's; the build script compiles
   the file into `GROUPS`). A group that names no table fails the build; the registry's tests hold the
   names unique, every table buildable and the file in the registry's form.
3. Give it a subject in `tests/contract.rs` (the test `every_registry_table_has_a_subject` refuses a
   table without one): a handful of outputs in the family's syntax, which the contract then checks
   at every chunking. `tests/robustness.rs` picks the table up on its own and scrambles its markers.
4. Give it a corpus row in `tests/bellwether_parse_fixtures.rs` (the slug of each recorded set and
   how the template ends the generation prompt); the replay reads the table and the opener from the
   registry.
5. Regenerate the README's models table (`SYMPHONY_WRITE_README=1 cargo test -p smg-symphony
   --test readme`); a stale table fails the build.

A new call syntax costs an assembler (under `json`, `tagged`, `pythonic`, or `gemma`), a
`CallSyntax` variant, an arm in the engine's assembler factory, and a grammar module under
`grammar/`, each with the same evidence. A syntax a response template writes as patterns costs
none of them: it is a `shape::CallShape`, data the shaped syntax runs.

## How a failure is classified

Three tiers, two of them at run time:

- **Dropped**: bytes the format consumed on purpose, each with a `DropReason`: a marker that moved
  the engine or a special token (`ControlToken`, `Wrapper`), the template's wrapping between regions
  (`Wrapper`), whitespace the format treats as structural (`Whitespace`), a call after the first
  where the request takes one call only (`LaterCall`), and the bytes of a call block a syntax
  whose block is never prose could not read (`MalformedCall`, behind a `Malformed` event of no
  bytes that counts the failure). Dropped bytes are expected on every well-formed output, the
  last kind excepted.
- **Malformed**: bytes of a region the parser gave up on, returned whole, each with a
  `MalformedReason` from a closed set: the stream ended inside a region (`UnterminatedRegion`),
  arguments that did not parse (`InvalidArguments`), a call to a function the request did not
  declare (`UnknownTool`: the engine's own event of no bytes before the call's start, for every
  syntax, the call delivered as read), and the syntax-level shapes a model gets wrong: text between a call's
  tags, a tag out of place or cut short, a call or a tag without a name, a key written twice or
  without a value, text where a call or an argument should begin, text after the arguments or
  after the list, a block that closed without a complete call, and the rest the enum lists with a
  sentence each (`event.rs`). A malformed region on a well-formed output is a defect of the model's
  or of the table's, and the reason says which shape to look at.
- **Forgiven**: a difference a harness accepts on purpose, never an event: the corpus harness
  carries per-syntax known differences and opt-in allowances for the classes the recorded
  references get wrong. Each run counts them on its diagnostic line.

The adapters show malformed text as content and drop nothing, so a client never loses the bytes a
parser hands back with the event; the keyed syntax, whose block is never prose, hands them back as
`Dropped { MalformedCall }` behind an event of no bytes instead, so that the failure is counted and
the format's own tags never reach the answer (`docs/assemblers.md`). The reason is for the
operator. The set is closed and `Copy` so that a reader can count malformed
regions by table and reason as a bounded label, log a sample with the table, the reason and the
bytes' length, and mark the request's span; how a gateway surfaces that (metric, log, span event,
and whether a request may ask for the reasons on the wire) is the gateway's policy, documented with
its parse-failure telemetry.

## The seam with a caller

A caller (a gateway, a test harness, a tool) does four things: picks the table (`formats::for_model`
for a model id, `Table::named` for a name an operator gives, or compiles one from a response
template); seeds an engine with the prompt's tail (`Input::Prompt`, which `Format::replay` cuts to
the part that matters) and the request's tools (`Declared`); feeds each decoded delta with its token
spans (`Input::Delta`) and the end of the stream with the engine's finish (`Input::End`); and renders
the events with an adapter, or asks the table for a constraint (`Format::grammar`,
`answer_grammar`, `turn_grammar`) before the request goes out. What a caller does not do: read the
events' bytes for structure, match a reason by its text, mint call ids, or decide what a byte is.
Policy about the output (one call only, which bytes are content, how many tokens the reasoning
cost) belongs here, as an option on the engine or a helper over the events, so that every caller
gets the same answer.

## What costs what

Per process: a table (built once per registry table and shared; a compiled response template per
model). Per request: the request's tool typing (`Declared`), one engine per generated choice (the
scanner over the table's terminals, the prompt's replay), and the constraint when the request asks
for one. Per delta: the scan of its bytes against the table's terminals, the events it completes,
and their rendering. The crate's bench (`cargo bench -p smg-symphony --bench throughput`) measures
the three: a turn fed whole and in small pieces per call syntax, prose with no marker in it, and the
construction of a table and of an engine.

## How it is tested

Five layers, each with a different question. Unit tests in every module ask whether one decision
holds. `tests/contract.rs` asks whether every registry table keeps the events' promises over its
outputs at every chunking. `tests/robustness.rs` asks whether any output a model could write, built
from a table's own markers and corrupted, can panic the parser or break the contract.
`tests/bellwether_parse_fixtures.rs` asks whether the tables read the recorded corpus as the
chat templates' own round trip does, byte for byte, at every chunking (opt-in, against a checkout
of the recorded sets). `tests/bellwether_shape_regex.rs` and `tests/bellwether_templates.rs` ask
whether the shaped syntax and the compiled tables agree with the reference implementations they
stand in for, against the shape patterns' oracle bellwether records (opt-in, the same tree). The
bench asks what it costs. A change to the engine or an assembler carries the first three; a change
to a table carries its unit tests and the corpus row; a change to the decode path carries the
bench's rows.
