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
