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
  is bellwether #17; `Dropped { Whitespace }` exists for the other answer.
- A terminal with no transition from the current state is text, as the model wrote it:
  `</think>` in content is content. A terminal inside a code fence, or inside a string in a
  call's arguments, is a terminal, as for every parser that reads markers: `</tool_call>` in an
  argument's text ends the call there. Bellwether #16 is where that policy is judged.
- A terminal the table ignores (MiniMax M3's separator token) is `Dropped { Wrapper }` wherever
  it stands and moves nothing; inside a call the assembler accounts for it, so that its bytes
  keep their place among the bytes the assembler holds.
- A call region that never closes is finished at the end of the stream: a call with what
  arrived, or the bytes as `Malformed { UnterminatedRegion }`. A region whose closing terminal
  comes before a complete call gives what is left as `Malformed` with a reason that says so,
  since `UnterminatedRegion` means the stream ended inside the region; with the tagged syntax
  the call is closed for the client first, as `tagged::Assembler::close` says.
- With the tagged syntax, a value's type comes from the request's tools (`Declared`): a
  declared string streams, every other value is written at its close. A call to a function the
  request did not declare has every value inferred.
- Call ids are `call_<index>` for now; the id scheme is the maintainer's decision
  (deterministic or carrying the conversation's history) and changes only this one line.
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
