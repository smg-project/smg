# Notes on the engine

`src/engine.rs` runs any table: its module doc says what the engine is and how it accounts for every
byte. These are the decisions it makes for every format, the questions it leaves open, and how the
prompt decides where an output starts.

## The engine

What the engine decides for every format, and what it leaves open:

- Prose before a call, between two calls, and after the last call is content, and every complete
  call in a chunk is emitted; the old parser dropped the first and stopped after one (smg
  #2788).
- The template's separator bytes stay where the model put them: the newline after `<think>`, the
  two after `</think>`, and so on are reasoning or content, not dropped. Whether they should be
  is bellwether #17; `Dropped { Whitespace }` exists for the other answer. A table whose family's
  reference parser reads a marker together with the separator before it spells the marker that
  way (DeepSeek V4.1's block opener after its blank line, `docs/tables.md`): the separator then
  leaves with the marker as `Dropped { Wrapper }`, and the scanner's hold-back keeps a stream's
  reading the whole's. One exception, so
  that a stream shows the content the whole answer has: whitespace a content stay begins with
  is held while nothing else has arrived in the stay, and leaves as content before the stay's
  first other text, as `Dropped { Whitespace }` when the stay ends without any (a call's opener
  after `</think>` or after a call, the end of the output after a thought, a second thought's
  open). The whole answer's message shows no content for whitespace alone, and a stream that
  showed it as it arrived carried content deltas the whole answer does not have. A trimmed state
  has its own rule; a shown marker (the variants that do not separate a channel) is text, so a
  lead held before it is content, and whitespace after it is content as after any text. A kept
  state (the kept row: a response template's `strip: false` field) holds nothing: whitespace
  alone there is content, as the template's reader gives it. The whole answer's message still
  shows no content for whitespace alone, so for such a table a Chat Completions stream can carry
  content deltas, and their logprobs, that the whole answer does not; the template's reader,
  which keeps the whitespace, is the reference there. The Responses and Messages streams hold
  whitespace alone until text with substance follows and let it go if none does, as their whole
  answers leave it out.
- A terminal with no transition from the current state is text, as the model wrote it:
  `</think>` in content is content. A terminal inside a code fence, or inside a string in a
  call's arguments, is a terminal, as for every parser that reads markers: `</tool_call>` in an
  argument's text ends the call there. Bellwether #16 is where that policy is judged.
- A terminal the table ignores (MiniMax M3's separator token, Inkling's end-of-sampling token)
  is `Dropped { Wrapper }` wherever it stands and moves nothing. Inside a call whose assembler
  holds bytes (a header until its marker, a word until it is spelled) it is placed where it
  arrived once the bytes around it leave: spliced into a call event's source, which no client
  reads, or split out of a text event as a `Dropped` event of its own, so that its bytes keep
  their place and none joins text a client reads; a syntax that reads it as its own separator
  (M3's XML) acts on it and leaves the bytes to the engine.
- A call region that never closes is finished at the end of the stream: a call with what
  arrived, or the bytes as `Malformed { UnterminatedRegion }`. A region whose closing terminal
  comes before a complete call gives what is left as `Malformed` with a reason that says so,
  since `UnterminatedRegion` means the stream ended inside the region; with the tagged syntax
  the call is closed for the client first, as `tagged::Assembler::close` says, unless the close
  fell inside a streamed string. A JSON call whose arguments value the close cut ends with the
  fragments it sent and a `Malformed` event of no bytes, with the region's reason, before its
  end: a cut call is never handed over as a whole one without a word (`json::assembler`), and
  neither is a shaped call whose value never began, the cut or the close right after its header
  (`json::Arguments::finish`). A tagged
  call the end of the stream cut ends the same way whatever stage the cut found, since the object
  the assembler writes closes only at the call's own end, and so does one whose close cut a streamed
  string, in every tag-based syntax (`tagged::end_cut_call`): no closing quote and no closing brace
  are made up for a value the model never finished, and the one event carries the bytes still held,
  when there are any (the keyed syntax refuses them first, as `Dropped { MalformedCall }`, and its
  event carries none: `tagged::refuse`). Gemma's reader does the same in its own way (`gemma::Assembler::end`): its one
  event names the value's reason, which the engine does not rename, and the bytes held at the marker
  are the string's text, not the event's. A pythonic call ends as a tagged one does
  (`tagged::end_cut_call`) whether the stream's end or the region's marker cut it, the cut right
  after its `(` included, since its object closes only at the call's own `)`, which every template
  writes before the marker.
- With the tagged syntax, a value's type comes from the request's tools (`Declared`): a
  declared string streams, every other value is written at its close. A call to a function the
  request did not declare has every value inferred, and the engine announces it, for every
  syntax, with a `Malformed { UnknownTool }` event of no bytes right before the call's start
  (`Engine::push_call_events`): the call goes out as the syntax read it, a reader counts the
  failure at its offset, and a request that declares no tools is not asked. Up to sixteen tools
  are searched by name as they are; more are indexed once per output, at the first call's start,
  and the index is searched. A syntax that reads an undeclared first name as prose (the pythonic
  list after content) makes no call, so there is nothing to announce.
- Call ids are `call_` and 24 lowercase hex characters, the shape the gateway's older parsers
  mint, so a client tells no difference. The engine draws a nonce at random when it is made
  (`CallIds::fresh`: 96 random bits; not a UUIDv7, whose in-process counter steps consecutive
  values by one) and the output's call `index` is the nonce `index` steps on: an id repeats in
  no other request, since a client pairs its tool results with its calls by id across a
  conversation, and is the same for one output however its stream is cut.
  `Engine::with_call_ids` names the nonce for a replay that wants the ids reproducible. No
  assembler chooses an id (`ids.rs`); a table whose call header writes the id and whose
  templates render it back into the prompt (Kimi K2) carries the model's own id on the call
  instead (`Format::call_id(CallId::Header)`: the header's text without its ending literal); a
  header id that repeats one already given in the response falls back to the minted id, so a
  response's ids stay distinct either way.
- Every text event says how many tokens it carries, counted by the `Ledger` from the deltas'
  spans: a token in the event that carries its first byte, a byte-less span (a held half of a
  character, or a hidden special token) into the run that carries the next byte, and the tokens
  left without bytes at the end once as `Dropped { ControlToken }`. `Finish::reasoning_tokens`
  is the count over the reasoning text. Once a delta lacks spans, or its spans do not partition
  its text, the rest of the stream is uncounted, and then `reasoning_tokens` is zero, the one
  place where zero does not mean none (rule 7 keeps `Finish` as it is for now).
- Tool names are not checked against the request's tools; the format has no tool list yet.

The prompt decides where the output starts: its terminals are replayed over the same table from the
initial state, and the output begins in the state they leave. The replay starts at the last turn
opener the table names (`<|im_start|>assistant` for ChatML), so a marker quoted in an earlier turn,
a `<tool_call>` in the user's question or a `<think>` inside an earlier call's arguments, moves
nothing; a table that names no opener is replayed from the prompt's start. Qwen 3.5's template opens
`<think>` in the generation prompt, so the output starts inside the thought, with `ReasoningStart`
pushed for the prompt, and its first `</think>` closes it; Qwen3 writes its own `<think>`; and a
prompt that disables thinking ends with `<think>\n\n</think>\n\n`, which leaves the engine in
content. A prompt that leaves an arguments state open is read as content for now (a prefilled call
is a later step).

## What a table read from a response template adds

- A table can say that the stream's stop closes the call in a state (`Format::closes_at_stop`),
  for a family whose last call ends with a stop token the stop decoder hides: a stream that stops
  there ends the call as its closing terminal would, with no bytes. One cut for length, aborted,
  or ended with any finish but `Stop` is cut as before; a driver hands a stop sequence of the
  request's that matched inside the call over as such a finish, so that it never completes the
  call (the stop row in `format`).
- A table can say that the template discards the text of a state (`Format::framing`): Hugging
  Face's null sink, the space around a response template's fields. Every byte of text there, a
  terminal with no row from the state and what the scanner held at the end included, is
  `Dropped { Wrapper }` at every finish; the state's rows still move the engine.
- A terminal with no row from the state is the state's text, as the model wrote it (dropped where
  the template discards the state's text). Where that text is the table's own content, a
  `Malformed { MarkerOutOfPlace }` event of no bytes stands before it, so a reader counts a frame
  the table did not read (an answer's opener written again inside the answer, a close with nothing
  open) without a byte moving; in a thought the marker is the thought's text alone, as a table may
  read a thought's opener as its text (a turn anchor's table), and so it stays in a state a
  request's variant made content (`Format::is_remade`: the thought of a request that does not
  separate reasoning, the calls of one without tools), where the table counted none: a thought
  reads the same whether the request separates reasoning or not, and a call's header is the
  call syntax's to judge; text in a wrapper state is malformed as text between calls, as before.
- A table can trim a state's text stay by stay (`Format::trims`), as Hugging Face's `text` parser
  strips a field's value: whitespace before a stay's first other character is
  `Dropped { Whitespace }` as it arrives, and a run of it after the last one is held, with any
  terminal the table ignores inside it, until more text in the stay shows it was inside the text,
  which releases it as the state's text, or a row ends the stay or the stream ends, which drops
  it. Held bytes have not reached the ledger, so the tokens are counted in the output's order
  whatever the run turns out to be.
- A call given as data (`CallSyntax::Shaped`) opens only when the header after its terminal
  matches: the engine holds the terminal and every byte after it, and takes no row, until the
  header decides. A header that cannot match gives the terminal back: its first character is the
  text of the state it came in, and the rest, with every byte the header read, is scanned again
  from there, as a regular expression's search goes on after the place its pattern failed at.
  None of those bytes was emitted, so the tokens are still counted in the output's order. A
  closing terminal closes a shaped call only where the call's trailer ends right before it; any
  other is the call's text. The stream's stop, where the table says it closes the call, closes
  one JSON value or object where its trailer ends with the stream, and tagged parameters wherever
  the trailer stands, as a template's reader closes the region at the end of its text.
- A shaped call's values read through the table's reader (`Format::reads_values_with`) with the
  parameter's schema from the request's tools.

A table with turn anchors (`Format::turn_anchor`) is replayed from the end of the last anchor, in
the state that anchor names, and not at all when the prompt holds none. A table with a start row
(`Format::starts_in`) lets the turn's first piece decide: a terminal whose row leads into another
state moves the engine as any terminal does, and anything else starts the turn in the start state,
where it is read; the first piece is the replay's first, or the output's when the prompt left
nothing to replay.

## The shape of a step's events

The engine fills a buffer the caller keeps; it never hands out a vector of its own. One
`Events` per stream, one `feed` per step, the step's events read in place, then the buffer
recycled for the next step:

```rust
let mut spare = Events::new();                      // one buffer for the stream
loop {
    let mut events = spare;                         // this step's view of it
    engine.feed(Input::Delta { token_ids, text: &delta, spans }, &mut events)?;
    for event in &events { /* render it */ }
    spare = events.recycle();                       // empty, allocation kept
}
```

- `Event<'d>` and `Text<'d>` carry the delta's lifetime. A text event borrows the delta's bytes
  where the engine held nothing back: the whole-text path (a delta with nothing held and no byte
  that starts a marker, the common content or reasoning delta) costs no copy. A run the engine
  held (a content stay's lead, a trimmed stay's tail, a shaped call's header), a run merged with
  an earlier one in the same buffer, and the events an assembler builds carry their text owned
  (`Cow::Owned`): an arguments delta takes the whole-text path too, which saves it the scan and
  the piece, but its bytes go to the assembler, whose events own their text and their JSON.
- The buffer's lifetime is the step's: `Events<'d>` cannot outlive the delta it borrows, so a
  delta's text lives with the caller, next to the buffer, and a caller that keeps events past the
  step takes them with `Event::into_owned` (or the whole buffer with `Events::into_owned`). A
  caller that reads an output whole feeds every input into one buffer and takes the events with
  `drain`, as the tests do.
- `recycle` empties the buffer and keeps its allocation for the next step's lifetime: an empty
  vector collected into a vector of the same layout is collected in place by the standard
  library, and `Events::capacity` shows it; a unit test holds the property.
- What was read is final: merging adjacent text of one kind only ever touches events pushed
  into the same buffer since it was last recycled or drained, so a stream sees one `Content`
  event per step for a run of content, and a whole read sees one for the whole run.
