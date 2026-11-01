# Notes on the tables

Each table under `src/formats/` reads one family's output; its module doc says what the table reads,
names the recorded group and shows one turn. The notes here are the rest: what the family's template
writes around the markers, what the table decides beyond the others, what it leaves open, and the
bellwether issues those decisions wait on. A table whose module doc says everything has no section.

## Qwen3 (`qwen3`)

The definition is the table below; the `Engine` runs it, and decides everything the module doc of
`engine` lists. Two things are Qwen3's own:

- `calls + call_open = calls`: a new `<tool_call>` before the previous block closed ends the
  call that was open and starts the next.
- No row leaves reasoning on a call marker: Qwen3 closes its thought with `</think>` before a
  call, so a `<tool_call>` inside the thought is reasoning text.
- The turn opener is ChatML's `<|im_start|>assistant`.
- Under the tagged syntax with the value between the tags directly (`qwen3_tagged_direct`), MiMo
  V2.5's template writes a newline after `<tool_call>` and after `<function=…>` and the V2.6 line
  (Flash-RL, Pro-RL, Flash-MOPD, Pro-MOPD, Distill-Qwen-9B) writes none; the newline is the tags'
  framing either way, and a value's own bytes stay as written. Both lines' templates write the
  thought alike: every earlier turn opens with `<think>`, closed at once when it has no thought,
  and the generation prompt leaves the model to write `<think>` (`<think></think>` with thinking
  off): the same rows read both, as the recorded sets of the four groups show.
- The table carries its template's two newlines after the thought's close (`</think>\n\n` before
  the content or the first call) as its thought layout, so a forced call written where the close
  left the engine begins with them and not with `<tool_call>` (`grammar`); a prompt with thinking
  switched off has written them itself (`</think>\n\n` at its end), and the tag owes none. MiMo's
  template writes what follows right after the close, and `qwen3_tagged_direct` carries none. The
  NVIDIA Nemotron 3 groups (Nemotron 3, 3.5 Lightning, Labs 3, Cascade 2, Cosmos3 Edge) and Step
  3.5 Flash read with `qwen3_tagged` but write one newline there (`</think>\n<tool_call>`): their
  entries in `groups.json` carry the one newline as the group's `after`, which replaces the
  table's, the close kept (`registry`). The Nemotron 3 groups' thinking-off prompt ends with the
  bare `<think></think>`, so the newline stays the tag's; Step 3.5 Flash's prompt always opens the
  thought.

The tests here are the engine's as much as this format's: they were written against the hand-written
parser this table replaced, and hold the table to the same events.

## DeepSeek V4.1 (`deepseek_v4_1`)

What this table says beyond the Qwen ones:

- **Two states for the calls.** `calls` is the template's wrapping between invokes (a newline,
  dropped; anything else malformed), and `invoke` is one call's arguments, read by the
  attributed assembler with DSML's tags (`tagged::attributed`,
  `Tags::DSML`), the `string` attribute typing each value. A block with several invokes gives
  several calls, each with its own index, through the same two rows.
- **The invoke tag's opening is the terminal** that enters the arguments state, so the
  assembler starts inside the function's name; the closing tag is the terminal that leaves it,
  and the call's end carries its bytes. A second invoke before the first closed ends the first
  and starts the second (`invoke + invoke_open = invoke`), and the block's close ends an invoke
  still open (`invoke + calls_close = content`), so the call closes into an object and the
  prose after the block stays prose.
- **A calls block ends the thought** (`reasoning + calls_open = calls`), as the design's table
  and SMG's V4.1 reasoning parser have it; the recorded outputs close the thought first.
- **The prompt opens the thought.** The template ends the generation prompt with `<think>`, and
  an empty thought writes `</think>` at once, so most recorded outputs start with
  `</think>\n\n`; the engine's prompt replay puts the output inside the thought first. The
  turn opener is `<｜Assistant｜>`, so a marker quoted in an earlier turn moves nothing. A tool
  result renders as a user turn, and the model's turn after it opens with `<｜Assistant｜>` and
  the thought opened or closed (thinking mode, chat mode), so the output always starts in a stay
  of its own.
- **The block's opener is the blank line and the tag.** The template writes two newlines between
  a message's content and its calls block, and the family's reference parser (the checkpoint's
  `encoding` module) looks for the block as `\n\n<｜DSML｜ calls`, reading the content up to
  there and otherwise as written: a blank line inside the content, a third newline or a space
  before the block's blank line, and whitespace at the turn's end stay. The table spells the
  opener the same way, as a second terminal beside the bare tag, so the blank line leaves with
  the tag wherever the block opens (after a text preamble, after the thought's close, at a chat
  turn's start, after a tool result), whole and streamed alike: the scanner takes the longer
  marker where the blank line stands and holds a newline that may begin it. The bare tag still
  opens a block written without the blank line, as it did; what precedes it is content. The V4
  and V3.2 tables spell their openers the same way (their sections); the legacy `deepseek_v41`
  parser strips exactly this blank line too.

## DeepSeek V4 (`deepseek_v4`)

The DSML spelling before V4.1's: no space after the sentinel (`<｜DSML｜invoke name="…">`,
`<｜DSML｜parameter name="…" string="…">`) and `tool_calls` for the block's name. The table is V4.1's
with those terminals and `Tags::DSML_V4`; the V4.1 spelling is content under it, as this one is under
V4.1's, and the registry reads each checkpoint with its own. Beyond that:

- **A calls block ends the thought** (`reasoning + calls_open = calls`), as V4.1's table has it, and
  as the forced-call tag this table writes allows when the prompt leaves the thought open: the free
  text before the block may close the thought or not, so the engine may write the block before
  `</think>`, and the table reads the call it forced. The family's reference encoder renders the close
  first and its parser refuses a block inside the thought; the gateway's legacy reasoning parser for
  this checkpoint ends the thought on `</think>` alone, so a block inside it stays in the thought
  there; this table reads the call.
- **The block's opener is the blank line and the tag.** The template writes two newlines before
  the block (what this table's forced-call tag writes too, `Tags::DSML_V4.before_block`), and the
  family's reference parser (the model repository's `encoding` module) looks for the block as
  `\n\n<｜DSML｜tool_calls`, reading the content up to there and otherwise as written: a blank
  line inside the content, a third newline or a space before the block's blank line, and
  whitespace at the turn's end stay. The table spells the opener the same way, as a second
  terminal beside the bare tag, so the blank line leaves with the tag wherever the block opens
  (after a text preamble, after the thought's close, at a chat turn's start, after a tool result,
  in a turn the prompt leaves unopened), whole and streamed alike: the scanner takes the longer
  marker where the blank line stands and holds a newline that may begin it. The bare tag still
  opens a block written without the blank line, as it did; what precedes it is content (the
  legacy parser trims those newlines too). The prompt ends with `<think>` in thinking mode and
  with `</think>` right after `<｜Assistant｜>` in chat mode, so the replay starts the output inside
  the thought or in the content; bellwether has not recorded the group's set yet.
- **A turn the prompt leaves unopened starts inside the thought.** The template writes
  `<｜Assistant｜>` and the thought's opener after a user message only: a conversation that ends with
  a system message rendered in place (a coding agent's environment notes after the user turn, and
  after every tool result) has no opener for the model's turn, and the model, which thinks unless
  told not to, writes its thought and `</think>` as it would after `<｜Assistant｜><think>`. The table
  names the thought as the state an unopened turn starts in, with `<｜end▁of▁sentence｜>` as the end of
  a turn the model wrote: a prompt with no opener, or whose last opener's turn ended, starts the
  output in the thought, and nothing of it is replayed (an earlier turn's markers and a `</think>`
  quoted in a tool result move nothing). The legacy reasoning parser for this checkpoint starts
  inside the thought whenever the request leaves thinking on, whatever the prompt; the two agree
  on this shape. The V4.1 template opens the turn after a mid-conversation system message too, so
  its table meets no unopened turn. The table is pinned against the turns the
  family's reference encoder (the model repository's `encoding` module) renders for a tool-calling
  conversation, each read as the encoder's own parser reads it: every kind of value, two invokes in
  one block, a call after content, and an invoke with no parameters, which the encoder writes with
  an empty line inside and the table reads as `{}`.

## DeepSeek V3.2 (`deepseek_v3.2`)

DSML as V4 spells it, with `function_calls` for the block's name: the table is V4's with the block's
two terminals changed and the same `Tags::DSML_V4` for the invoke and parameter tags, so what the
V4 section says of the format holds here (a calls block ends the thought; the blank line before
the block is the opener's, `\n\n<｜DSML｜function_calls>` beside the bare tag, as the encoder's
parser reads it; the prompt ends with `<think>` or `</think>` right after `<｜Assistant｜>`), with two
exceptions at the turn's start. The table has no unopened row, and must not take V4's: the encoder
writes `<think>` or `</think>` after every user message and after every tool result, so the turn
is never left unopened, and after a tool result `<｜end▁of▁sentence｜>` follows the last
`<｜Assistant｜>`, which V4's row would read as an unopened turn and start in the thought (in chat
mode the answer would then read as reasoning; the `chat_after_result` assertion of
`the_prompt_decides_where_the_output_starts` guards it). And the table names no turn opener but
the four tails the encoder writes before the model's turn, as turn anchors (`<｜Assistant｜>` and
`</function_results>` with two newlines, each followed by `<think>` or `</think>`): the replay
starts after the last of them, in the state it names, so the turn's earlier block and the tool
result's text are not walked (a DSML marker quoted in a result moves nothing), and the blank line
the model writes before a block after a tool result leaves with the block's opener, as at the
turn's start and after a preamble, as the encoder's parser reads it.
Beyond that:

- **The checkpoint ships an encoder, not a chat template.** The repository's `encoding` module
  renders the prompt and reads a completion, and the gateway's own renderer for the family is a
  port of it. After a tool result the encoder writes `<function_results>` blocks inside the same
  assistant turn and opens the thought again (`</function_results>`, two newlines, `<think>`; or
  `</think>` in chat mode), the tails the table anchors the turn at, so the output starts inside
  the thought, or in the content, as the tail says.
- **Each DSML checkpoint reads with its own table.** The V4 block name and the V4.1 spaced tags are
  content under this table and this block is content under theirs; the registry reads V3.2 and
  V3.2-Speciale here, V3.2-Exp (it writes the earlier V3.1 spelling) with `deepseek_v3.1`, and
  the V4 checkpoints with theirs, and the `deepseek32` alias takes a deployment that names the
  older parser to this table.
- **A string value that holds a closing tag is two parameters.** DSML writes a string raw, so a
  value holding `</｜DSML｜parameter>` and another parameter's opening tag reads as two parameters,
  to the encoder's own parser (it splits at the first closing tag) and to this table alike; the
  format has no escape for it.
- **The forced-call tag writes this block**: the V4 tag with `function_calls` in the block's two
  terminals, byte for byte otherwise.

The module's tests pin the turns the encoder renders where this table reads them its own way,
each read as the encoder's own parser reads it: one call with a string and a number value, at
every cut, and once each a thought and an answer, a calls block inside the thought, a closing tag
inside a string value, and the turn after a tool result in thinking and in chat mode. The other
shapes of a tool-calling conversation (two calls in one block, a call after content, a call with
no parameters, every kind of value, non-ASCII text) are hand-written cases of bellwether's sets,
which its recorder renders through the encoder once the group has a manifest there; the corpus
harness replays them from the fixture pin, and until then nothing in this repository checks what
they parse to; the README row says awaiting fixtures until the pin holds it. What the table
leaves open: the recorded set, and the live reading against the engine's own server.

## DeepSeek V3.1 (`deepseek_v3.1`)

What the table says beyond the DSML ones:

- **The same four rows around the block, another call syntax.** `calls` is the template's wrapping
  between `<｜tool▁call▁end｜>` and the next `<｜tool▁call▁begin｜>` (the template writes nothing
  there; whitespace dropped, anything else malformed), and `call` is one call read by the named
  arguments assembler (`json::named_arguments`): the tool's name up to `<｜tool▁sep｜>`, taken as
  written, then the arguments object alone. The call's close is its own terminal, so the object's
  bytes end at the object and `<｜tool▁call▁end｜>` is wrapping; a new `<｜tool▁call▁begin｜>` before
  the close ends the call that was open and starts the next, and the block's close ends a call
  still open, as the DSML tables have it.
- **A calls block ends the thought** (`reasoning + calls_open = calls`), as the V4 table reads it;
  the template closes the thought before the block.
- **The prompt opens or closes the thought.** The template ends the generation prompt with
  `<｜Assistant｜><think>` when the request thinks and `<｜Assistant｜></think>` when it does not,
  and writes no generation prompt after a tool result: the output then starts where the previous
  turn's `<｜Assistant｜>` leads, which is content once its block has closed. The turn opener is
  `<｜Assistant｜>`, so a marker quoted in an earlier turn moves nothing.
- **The recorded set holds no call.** The template concatenates a call's arguments as a string,
  and bellwether's corpus holds them as an object, so the recorder refused every call case of
  `deepseek-v3.1` (and of `deepseek-v3.2-exp`, the same spelling): the content sets replay; the
  calls stand on the turns the template renders with string arguments, held in the module's tests
  at every cut, as the V4 table stands on its encoder's turns.

- **A forced call's tag is the block around the calls the table reads.** `Format::grammar` writes
  the block's ways in (`<｜tool▁calls▁begin｜>` from the turn's own state, or `</think>` first from
  the thought) around one or more calls, each `<｜tool▁call▁begin｜>`, the tool's name,
  `<｜tool▁sep｜>`, the arguments the tool's schema accepts and `<｜tool▁call▁end｜>`, with nothing
  between the calls, and the block's close: the same rows the table reads a call by, written from
  the named-arguments syntax as the tagged syntaxes' calls are (`docs/grammar.md`). On an engine
  whose grammar starts after its reasoning parser has seen `</think>`, a thinking answer keeps its
  thought and the forced call follows it in the model's own syntax; a prompt with the thought
  closed takes the block at once. Before the builder saw through the block, the gateway handed a
  forced or required request to its older path, whose JSON-schema constraint made the model write
  the bare arguments (or the list of calls) and whose parsers read the answer; the module tests
  pin how the table reads those bytes (a thought and JSON content, no call) beside the tag's
  shape.

What the table leaves open: the recorded set's calls (the recorder refused them, above) and the
live reading of a conversation on an engine beyond the forced call, owed with the family's weights.

## DeepSeek V3 (`deepseek_v3`)

The spelling before V3.1's, the gateway's older `deepseek` parser's: no thought in the template
(the table reads one all the same, below), the call's opener
is `<｜tool▁call▁begin｜>function<｜tool▁sep｜>` (the call's type before the separator, which the
template writes from the message and every message writes as `function`), the name runs to the
`json` fence the template opens on the next line, and the fence's close with `<｜tool▁call▁end｜>`
is the terminal that leaves the call. Beyond that:

- **The newline before the fence's close is wrapping**, dropped as whitespace after the object, and
  so is the newline the template writes before every call after the first. A fence closed without
  `<｜tool▁call▁end｜>` right after it is not a terminal: the bytes stay in the call as malformed
  text until the next opener or the block's close, as the older parser's pattern would not have
  read them either.
- **A thought the template never writes.** V3's template has no thinking switch, yet the table
  keeps the thought rows (`<think>` opens it, `</think>` closes it, a calls block ends it) because
  the checkpoints that share the call spelling and write a thought read with it: the DeepSeek R1
  line, by id, and the gateway's older pairing `deepseek` + `deepseek_r1` under the `deepseek`
  alias. The thought must never land in the content. The cost is a V3-0324 output that spells
  `<think>` itself, which the reference keeps as content and the table reads as a thought; the
  recorded sets hold none.
- **The R1 line is this table by id.** The templates of DeepSeek-R1, R1-Zero, the six distills,
  R1-0528 and R1-0528-Qwen3-8B, and the bf16 mirror of R1, render V3's calls block from the history
  (`function<｜tool▁sep｜>`, the name, the `json` fence) and declare no tool, and keep an assistant
  message from its `</think>` on. They differ in the generation prompt alone: R1's, R1-Zero's and the
  distills' end in `<think>` and a newline after `<｜Assistant｜>`, so the output starts inside the
  thought and the prompt's replay puts it there; R1-0528's, R1-0528-Qwen3-8B's and the bf16
  mirror's end at the header, and the model writes `<think>` itself. The recorded sets (R1-0528, R1-0528-Qwen3-8B, the bf16
  mirror, Distill-Qwen-1.5B, Distill-Llama-8B) hold content alone, as the templates render a
  message, and replay bitwise; the thought's two openings are held by the module's tests. Under
  `--parser symphony` the ids select this table, so a deployment that named only the
  `deepseek_r1` reasoning parser reads with the table by its model id; the reasoning parser's
  name alone selects no table (it reads only a thought, and a table that reads V3's calls under
  that name would be a wider reading than the name says), which an operator serving an R1 under an
  opaque name settles by naming the table on both flags.
- **The recorded set holds no call**, for the reason given under V3.1; the calls stand on the turns
  the template renders, held in the module's tests at every cut.
- **A forced call's tag writes the block**, from the same rows as V3.1's: each call is
  `<｜tool▁call▁begin｜>function<｜tool▁sep｜>`, the tool's name and the fence the template opens,
  the arguments the tool's schema accepts, then the fence's close with `<｜tool▁call▁end｜>` right
  after the object (the template puts a newline between them, which the table drops as wrapping
  either way), between the block's ways in and its close, with the template's newline before
  every call after the first (`CallSyntax::NamedArguments::between_calls`). The fence is ordinary
  text, so the free text before the calls (a thought the prompt opened) may still hold a fenced
  draft of the arguments: only the table's terminals are kept out of it. Not yet read live on an
  engine.
- **The tool outputs are a state of their own.** The template writes the turn opener after a
  user message only, so after a tool result the prompt's replay walks the call turn, and the R1
  distills' template writes `<｜tool▁calls▁end｜><｜end▁of▁sentence｜>` after the second call and
  later only, so a turn with one call ends with neither: without a row for the outputs the walk
  stayed in the block and the next turn read as wrapper text. `<｜tool▁outputs▁begin｜>` and
  `<｜tool▁outputs▁end｜>` are terminals, and the outputs between them a state entered from the
  block or from content and left for content, with no other row; the inner `<｜tool▁output▁begin｜>`
  enters it from content, and from the thought, as well, since R1's, R1-Zero's, the bf16 mirror's
  and the distills' templates write the outer opener, like `<｜Assistant｜>`, for the conversation's
  first tool turn only (R1-0528's and R1-0528-Qwen3-8B's reset them every tool turn, as V3-0324's
  does), so a later turn's text is walked from that first opener in content, where an unbalanced
  `<think>` quoted in a user's question or a call's arguments opens the thought. Nothing quoted inside a tool's
  output moves the walk (a `<think>` or a calls opener in a result's text stays there), and the
  close starts the answer with nothing begun, for every template that ends its block as much as
  for the distills' one-call turn, so whitespace alone ahead of its thought or call is held, not
  shown, and the model's own `<think>`, or one the client prefilled (`continue_final_message`),
  opens the thought from content as it does after `<｜Assistant｜>`. Bellwether's R1-line sets hold
  content alone, so the module test pins the one-call history as that template renders it, the
  quoted markers, the held lead, and the prefilled thought.

## GLM 4.5 and 4.6 (`glm4`)

The template writes a newline before the thought, before the content and before every call, and
strips the content's edges, so those newlines are the model's to write and reach the content stream;
whose they are is bellwether #17's question, as for Qwen3.

The table carries its template's newline after the thought's close (`</think>\n` before the content
or the first call) as its thought layout, so a forced call written where the close left the engine
begins with it and not with `<tool_call>` (`grammar`); a prompt with thinking switched off ends with
the bare `<think></think>`, so the newline stays the tag's to write. A `<tool_call>` inside the
thought stays thought text, as for Qwen3.

## MiniMax M2 (`minimax_m2`)

The prompt opens the thought (`]~b]ai\n<think>\n`, whatever the request says: the template has no
thinking switch), and a turn opens with `]~b]ai`, so the prompt's replay starts there. The template
writes the thought as `<think>\n`, the reasoning, `\n</think>\n\n`, then the content, then
`\n<minimax:tool_call>\n`, each invoke as `<invoke name="NAME">\n`, each parameter on its own line,
`</invoke>\n`, and the block's close; the newlines around the thought and before the block are the
model's to write and reach the reasoning and content streams (bellwether #17's question, as for
Qwen3), the ones inside the block are its wrapping.

## MiniMax M3 (`minimax_m3`)

What this table says beyond the DSML one:

- **An ignored terminal.** The template writes `]<]minimax[>[` before `<tool_call>`, before
  every tag inside an invoke and before `</tool_call>`; it is one token, and it never stands
  inside prose. The table ignores it: the engine drops it in every state and moves nowhere,
  and inside an invoke its bytes keep their place among the invoke's.
- **The model writes the thought, or closes an empty one.** The template renders an assistant
  turn as `<mm:think>…</mm:think>` when it has reasoning and as `</mm:think>` alone when it has
  none, and its generation prompt adds nothing in its default mode (`thinking_mode` unset or
  `adaptive`), so an output starts with one or the other. The table starts in a `start` state
  whose text is content: `<mm:think>` there opens the thought and `</mm:think>` there closes
  the empty one, both dropped; later in the content both are text. With `thinking_mode` set,
  the generation prompt ends with `<mm:think>` (`enabled`) or `</mm:think>` (`disabled`) after
  the turn opener `]~b]ai\n`, and the replay puts the output inside the thought or in content.
- **A calls block ends the thought** (`reasoning + calls_open = calls`), as the design's table
  has it; the recorded outputs close the thought first.
- **A forced call admits the thought first, and the separator.** The structural tag for a forced
  or required call opens with the thought, or the bare `</mm:think>` of an empty one, or nothing,
  then the separator the template writes before `<tool_call>`, optional, then the block; the
  block without the separator, which the grammar forced until now, stays valid. With the thought
  left open by the prompt, free text that may close it comes first instead (`docs/grammar.md`).
  A JSON answer's tag (`response_format`) is led the same way. The thought-closed shape serves
  both the adaptive default and `thinking_mode: disabled` (the generation prompt then ends with
  `</mm:think>`): the thought, or a second close, is admitted in both, and an answer that writes
  one with thinking off reads back as content, as the legacy parser's tag has it.
- **Two states for the calls, as for DSML:** `calls` is the template's wrapping (a newline after
  `<tool_call>` and after `</invoke>`), `invoke` one call's arguments, read by the XML assembler
  (`tagged::xml`); a second invoke before the first closed ends the first,
  and the block's close ends an invoke still open.

## Kimi K3 (`kimi_k3`)

What this table says beyond the DeepSeek one:

- **The turn itself is a state.** Between the regions there is nothing: the template writes
  the regions back to back, and the end-of-turn tag after the last. `message` is that
  wrapping, so a byte there is `Dropped { Wrapper }` when it is whitespace and `Malformed`
  otherwise, and it is the state the turn opener leaves the engine in. Content is only what the
  `response` region holds, every byte of it; the recorded answers match it exactly.
- **Two states for the calls**, as for DSML: `tools` is the wrapping between calls, and `call`
  is one call's arguments, read by the attributed assembler with Kimi's tags (`tagged::attributed`,
  `Tags::XTML`).
  The call tag's opening is the terminal that enters `call`, so the assembler starts inside the
  function's name; the call's closing tag leaves it, and the call's end carries its bytes. A
  second call before the first closed ends the first (`call + call_open = call`), and the
  block's close ends a call still open (`call + tools_close = message`).
- **The end-of-turn tag ends whatever is open.** `<|close|>message<|sep|>` is a token the model
  cannot write by accident, so from any state it ends the region and returns to `message`: a
  thought or an answer cut by it is closed, a call is closed into its object, and the tag
  itself is dropped as wrapping. The same holds for a region's opening inside the thought
  (`reasoning + response_open = content`, `reasoning + tools_open = tools`) and for a tools
  block opened inside the answer (`content + tools_open = tools`): the region ended, however
  the model left it, since the tags are tokens the model cannot write as text.
- **The prompt opens the thought.** The template ends the generation prompt with
  `<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`, so an empty thought writes
  `<|close|>think<|sep|>` at once, and the engine's prompt replay puts the output inside the
  thought first. The turn opener is the message tag with the assistant's role, so an earlier
  turn's regions move nothing.
- **`<|open|>` and `<|close|>` are structural tokens.** Each is one token of the vocabulary,
  so an output cut by `max_tokens` inside a tag (`<|close|>message` with no `<|sep|>` after
  it) ends in a tail the model cannot have written as text: the engine drops it at the end of
  the output, as the engine's own streaming parser does, instead of releasing it as content.

## Inkling (`inkling`)

What the table says beyond the others:

- **Every message starts in the arguments state.** The call's name comes before the marker that
  says the message is a call, so the state after `<|message_model|>` reads with the named-JSON
  assembler (`json::named`): `<|content_thinking|>` or `<|content_text|>`
  there leaves it with nothing fed and no call started; the name and the marker are the
  template's framing, dropped, and the object is the call. The call's end is the object's close;
  `<|end_message|>` is the message's, dropped.
- **The turn's end token is ignored.** The template writes `<|content_model_end_sampling|>`
  after the turn; an engine that passes it through has it dropped wherever it stands, inside
  a message's header or its call included, where its bytes keep their place.
- **A JSON answer's tag leaves the way in to the engine.** The thought is a message of its own
  and the answer another, so an engine that starts a grammar after its reasoning parser hands
  over at `<|content_text|>`, not at the thought's `<|end_message|>`, which ends every message
  alike. `Format::answer_grammar` writes the thought the model may write first (its message,
  closed, and `<|message_model|>`) with `<|content_text|>` owed after it, or `<|content_text|>`
  alone as optional unless the engine says it starts at the first token, then the schema and
  `<|end_message|>`: an engine that consumed the
  model's own marker writes the JSON at once, one that starts before the thought writes the
  marker itself. Before this, the tag's `begin` wrote the marker again after the model's own on
  such an engine, and the answer came back as `<|content_text|>{...}`.
- **A kind said again inside a body is the kind.** The family's own parsers read the blocks as
  self-describing, in any order: `<|content_text|>` inside the answer starts its text over and
  inside the thought starts the answer, `<|content_thinking|>` inside the answer or the thought
  opens a thought, and none of the four is the body's text; the table has the same rows
  (`content` on `text` back to `content`, on `thinking` to `reasoning`; `reasoning` on
  `thinking` to itself, on `text` to `content`), so the markers are framing there, dropped, and
  the whitespace a stay held before one is the template's. The message closes where it did: the
  tag derivation tells a kind said again (a row from one body into another on the marker that
  opens that body from the message's header, `Format::is_kind_switch`) from a close of the
  thought (its row into the space between the messages), so the answer tag and a forced call's
  tag end the thought at `<|end_message|><|message_model|>` as before, and the thought's text
  under them writes no kind marker. Before the fourth row, `<|content_text|>` inside a thought
  was the thought's text: such an answer came back as reasoning, with empty content and nothing
  malformed (no recorded output holds the shape; a model held to a JSON answer could write it).
  The rows are what an engine that starts a JSON answer's grammar after its reasoning parser
  needs: the tag
  above cannot tell that engine's start from the first token, so at the gate, past the model's
  own `<|content_text|>`, its optional thought and marker are open to the model once more, and a
  model held to a JSON it had not planned writes one of the two before the JSON about two times
  in three. Read with the rows, such an answer is the bare JSON, the second thought reasoning
  after the first; read without them, it was `<|content_text|>{...}` or
  `<|content_thinking|>...{...}` in the content, each counted as a marker out of place. A
  `<|message_model|>` inside a body still has no row and is still counted so. The variant for a
  request that keeps the reasoning in the content keeps the four markers dropped
  (`Format::without_reasoning` leaves the message frame's terminals unshown).
- **A forced call's tag is the gateway's.** `Format::grammar` writes the tag the gateway's old
  Inkling parser hand-writes: the call marker triggers it inside whatever message the model is
  writing, the first one included, then `{"name":"NAME","args":`, the schema, `}` and
  `<|end_message|>`; the tool's name before the marker, a thought and the template's empty
  answer are free text, so the recorded thinking turn stays writable. A marker outside a
  message's header is writable under the tag too and reads as that message's text: the
  grammar is wider than the table there, as the gateway's tag is.
- **The group is one template.** Inkling, Inkling-Small and Inkling-NVFP4 ship the same chat
  template byte for byte (one sha256 at the three checkpoints' recorded or current revisions), so
  the three read with the one table and the recorded `inkling` set stands for the group.
- **A request without tools reads the same framing.** The variant the gateway reads such a
  request with (`Format::without_calls`) turns the message header into content, as no syntax
  reads a call there, keeps the ended message a wrapper (the space between two messages, whose
  whitespace is the template's), and keeps every marker dropped: the header is the state the turn
  starts in and returns to, so its four markers frame the thought and the answer as much as a
  call, and are not a call's own; the variant without the thought made of it, for a request that
  keeps the reasoning in the content, drops them too, as that variant does with tools. The
  thought and the answer read as with tools; a call the model writes anyway comes back as content
  between the dropped framing (`search<|content_invoke_tool_json|>{...}`), as the legacy path
  leaves it. Before this, the variant showed `<|end_message|><|message_model|><|content_text|>`
  and the answer's `<|end_message|>` as content on every request without tools.
- **A request with tools and `tool_choice: none` reads with the table itself.** The gateway
  renders such a request's tools as it renders them under `auto`, as the engine's own server does
  (its prompt keeps the definitions unless an operator excludes them; the choice reaches the
  template as a kwarg), so the model may write a call anyway. The table reads it whole, and the
  gateway's reader drops its events (`event::forbidden_calls`: every call's bytes become
  `Dropped { ForbiddenCall }`, the finish counts none and a `tool_calls` finish is `stop` again),
  counted per table as `smg_symphony_calls_forbidden_total`; the text around the call is content
  as under `auto`, and nothing of the call reaches the client as a call or as content, which is
  what the engine's own server delivers for it. A request with no tools anywhere still reads with
  the variant without calls, whatever its `tool_choice`.

What the table leaves out, for want of fixtures: image and audio messages, which the template writes
for the user and never for the model; and the text-mode invocation, `<|content_invoke_tool_text|>`,
which the gateway's older Inkling parsers drop as protocol data. No recorded set holds one. The
table has no terminal for it, so such a message sits in the named-JSON header until
`<|end_message|>` and comes back whole as `Malformed` (a name without a call), no byte lost; a
recording of one decides what it is.

## Mistral v13 (`mistral_v13`)

What the table says beyond the others:

- **A call has no closing marker.** `[TOOL_CALLS]` opens the call state, and the state's own rows
  are the next `[TOOL_CALLS]` and the end of the turn: the arguments object's close ends the call
  (`json::named`, reading the arguments alone), the next opener or the turn's end ends the
  region. Text after a closed object and before the next opener is the template's wrapping:
  whitespace dropped, anything else malformed. The tool's name is the text between the opener and
  `[ARGS]`, taken as written; the object after `[ARGS]` is the arguments alone, so
  `ToolCallStart` carries the name and the marker as its source, and the fragments are the
  object's bytes.
- **The prompt ends at `[/INST]`, or after a tool result.** The template writes no assistant opener
  and no thought in the prompt, whatever the request says: the model writes `[THINK]` itself. After
  a tool result the prompt ends in `[/TOOL_RESULTS]`, and its last `[/INST]` stands before the
  previous turn's calls; the replay walks into the call state, which the prompt never enters, so
  the output starts in content, where the template puts it.
- **A forced call's tag has no end.** `Format::grammar` writes a tag per tool from `[TOOL_CALLS]`,
  the name and `[ARGS]` to the schema, and nothing after it, as the template writes nothing: the
  object the schema accepts ends the tag. The gateway's older `mistral` parser and its tag read
  Mistral 7B v0.3's `[TOOL_CALLS][{…}]` list, another syntax, which the `mistral` table reads;
  the names keep the two apart, `--tool-call-parser mistral` reads with that table under
  `--parser symphony`, and the registry picks this one for the v13 groups.

What the table leaves open: whether every engine takes a tag whose `end` is empty (xgrammar's
structural tag allows it; the Symphony e2e lane says so for the engines it runs).

## Mistral, the list (`mistral`)

The spelling of the Mistral tokenizers before v11, the gateway's older `mistral` parser's: the
calls as one JSON list after `[TOOL_CALLS]`, `[TOOL_CALLS] [{"name": "get_weather", "arguments":
{"city": "Paris"}, "id": "call00000"}]`, the content before it; no thought. Mistral 7B v0.3's
template writes it (a space after the marker, an `id` member of nine characters after each call's
arguments); Mistral Small 3 (2501) and the checkpoints that keep its tokenizer (Small 3.1,
Devstral Small 2505) write the same list through their own tokenizer, since their chat templates
render no tool call at all. What the table says beyond the others:

- **One marker, the list behind it.** `[TOOL_CALLS]` enters the arguments state, and the list
  assembler reads the region framed (`json::list::Assembler::framed`), as Llama Nemotron's does:
  whitespace before the `[` is the template's, each object a call, the brackets and commas
  wrapping, text in place of the list malformed. A second `[TOOL_CALLS]` opens another region
  (`calls + calls_open = calls`), as the v13 table has it. The region ends with the turn: the
  template writes no closing marker, and `</s>` is the stop the decoder hides.
- **The `id` member is the call object's, not an argument.** The template writes it after the
  arguments; the JSON assembler reads a member it does not name as part of the object, so the
  call's arguments are the `arguments` value alone and the `id` is the object's bytes.
- **The prompt ends at `[/INST]`, or after a tool result.** As the v13 table: no assistant
  opener, no thought; after `[/TOOL_RESULTS]` the last `[/INST]` stands before the previous turn's
  calls, the replay walks into the call state, which the prompt never enters, and the output
  starts in content.
- **The recorded set holds no call.** Mistral Small 3's template renders no tool call, so the
  recorder kept its content sets alone (184,972 cases), which replay; 7B v0.3's parse sets are
  not recorded. The calls stand on the turns 7B v0.3's template renders, held in the module's tests
  at every cut; bellwether's hand-written sets record them for the family once its recorder has a
  rule for call ids (the template requires one of nine letters or digits, and a corpus case's
  calls carry none).
- **A request with tools takes a template an operator supplies.** A checkpoint whose own chat
  template renders no tool call (Mistral Small 3 and 3.1, Devstral Small 2505) never shows the
  model the request's tools through it, so such a request is served through a template that
  renders them, handed to the gateway as `--chat-template`; the engine's own HTTP server has the
  same limit, and its example template for the family writes the list the table reads.
- **7B v0.3's template writes a space before the assistant's text** (`[/INST] Hello!</s>`), and
  none before `[TOOL_CALLS]`; the table reads the space as the model's bytes, as the v13 table
  reads what its template writes around the content, and the module tests expect it. Whether
  the family's own tokenizer shows that space to a client is for the live reading to say.
- **A forced call is the list behind the marker, its own `]` the end.** `Format::grammar` writes
  one tag: `[TOOL_CALLS]`, the list of the tools' calls as Llama Nemotron's tag holds it, and the
  list's closing `]` as the tag's end, since the template writes nothing after the list and the
  turn ends the region. The `]` is the end rather than the content's last byte because the
  engines' structural-tag parsers refuse a content that may grow without bound (the further
  calls) behind an empty end (`Grammar::check` holds every table's tag to that rule); the v13
  table's tags, one JSON object each, end with nothing. A list with no marker is content to this
  table, so the tag never writes one; the `id` the template writes after each call's arguments is
  left to the model.

What the table leaves open: the live reading on an engine (owed with the family's weights);
Magistral Small 2506, whose recommended system prompt has the model write its thought as plain
`<think>…</think>` text, which this table has no rows for, and which stays out of the group until
a table reads it (the gateway's older parsers serve it meanwhile); and the v11 tokenizer's
checkpoints (Mistral Small 3.2, Devstral Small 2507, Magistral Small 2509), whose templates write
`[TOOL_CALLS]name[ARGS]{…}`, the v13 table's spelling without `[CALL_ID]`, and are not in the
registry until a set records them.

## JSON (`json`)

What the table says beyond the others:

- **One state, an object, several, or a list.** As `xlam`, the output starts in the arguments
  state and the assembler decides at the first byte that is not whitespace; unlike `xlam`, a `{`
  there is one call alone (`json::list`, `Framing::ObjectOrList`), whose object reads as a
  list's object does, `arguments`, `parameters` or `args` naming the arguments. After the object
  another object is another call, behind a `;` or behind whitespace alone, the `;` and the
  whitespace the template's wrapping: Llama 3.1 writes its parallel calls as `{…}; {…}`, Llama
  3.2 Instruct as consecutive objects with a blank line between them, `{…}\n\n{…}`, and the
  engines' own readers of the family read every object of either sequence. Anything else after
  the object reads as what follows a list's `]`: whitespace that ends the output is the
  template's wrapping, dropped; text is malformed, with the whitespace before and between its
  words, since a model that explains its JSON in a sentence wrote that sentence and the
  reference keeps it; so is a `;` that no object follows.
- **The family's call token opens the output, or does not.** Llama 3.1's template renders a call
  as the bare object, but its models write the call behind `<|python_tag|>`, the token the
  template reserves for a tool call (`CallSyntax::JsonAlone` names it): at the output's start,
  after whitespace or not, the token is dropped as the template's wrapping and the first byte
  after it decides as the output's first did, so an output with the token and one without read
  the same. A delta that ends inside the token is held until the next byte decides. The token is
  read at the start only, as the family's own readers read it: after prose it is text, whole with
  the prose, and after the object it is text after the arguments. The gateway's older `llama`
  parser, whose name the table answers to, took the token anywhere in the text and the prose
  before it as content; that shape reads as content whole here, by the rule that no table scans
  for a marker, and a served Llama 3.1 Instruct checkpoint wrote the token first on every output
  that had it (248 of 248 in the recorded set). A request without tools reads with the table
  without its calls, and then the output is the text it was, token included.
- **Prose before the JSON is content, whole.** The gateway's older `json` parser searches the
  output for the first `{` or `[`; this table does not, as no other table scans for a marker.
- **A forced call is the list.** `Format::grammar` writes the `xlam` list, as the older parser's
  forced calls were written; the call token is not in the grammar, since the template never
  renders it.

What the table leaves open: eight recorded groups write the format (Nemotron Ultra 253B CPT, the
UltraLong 1M, 2M and 4M, Safety Guard 8B v3, the 70B Reward Principle, Llama 3.2 1B/3B Instruct
and 11B/90B Vision Instruct), as the templates render it: one call alone, `parameters` naming the
arguments, and never a list, content beside a call or a thought, since the templates refuse them;
the call token and the `;` between calls are what a served Llama 3.1 Instruct checkpoint writes,
held verbatim in the table's unit tests, and the list and the wrapping after an object rest on the
unit tests until a group writes them.
Llama 3.1 8B/70B/405B Instruct and Llama 3.3 70B Instruct write it under their own ids too: their
one template is, byte for byte, the one the Nemotron Ultra CPT, UltraLong 1M/2M/4M and Safety Guard
8B v3 checkpoints carry, so those five recorded groups replay it already; bellwether has recorded
none of the four, so their rows await fixtures.

## Llama 4 (`llama4`)

The template writes the turn as the content, if the message has one, then the calls as a Python
list, `[get_weather(city="Paris"), get_stock_price()]`, with nothing between the content and the
list and nothing after it; no thought. What the table says beyond the others:

- **One state, no markers, the list bare in the turn.** As `xlam` and `json`, the output starts in
  the arguments state; the pythonic assembler reads the turn bare (`pythonic::Assembler::bare`).
  A turn that starts, after whitespace alone, with a `[` that a name and its `(` follow is the
  list, read as it arrives, the whitespace its wrapping; after the list's `]` whitespace alone is
  dropped, and once text follows every byte after the `]` is the model's prose, whitespace
  included, reported malformed (the rule the `json` table's list reader states). A turn that
  starts with anything else is content, and a
  `[name(` after that content, the shape the template renders for a message that holds both
  (`Let me check.[get_weather(city="Paris")]`), is held until the `]` that closes it and read as
  its calls only when the whole is a well-formed list of calls with keyword arguments to tools
  the request declares: code holds `[name(` of its own (`lim(h→0)[f(a+h) - f(a)]/h`,
  `movies[Math.floor(…)]`, `[set(range(10)) for _ in range(3)]`, `[Authorize(Roles = "Manager")]`,
  `ingredients[exotic_ingredient.lower()]`: twenty-two of the 5,841 content cases a 1-in-100
  sample of the recorded sets holds), and a list opened there as it arrives took those brackets
  for calls; held, checked whole and held to the request's tools, every one of them is content,
  and the recorded messages with content and calls read as the reference has them. A `[` that no
  call follows (`[1]`, `[Note]`, a markdown link, a list of
  names) is content, and so is a list the stream's end cut before its `]`, since no call began.
  The engine's own reader for the family opens the list at the output's start alone; where the
  model writes content before its list, that reader gives content only and this table the calls.
- **Every value is a quoted string in the template, so the quotes carry no type.** The template
  writes each argument as `"%s"` of its value: `days="3"`, `detailed="True"`, a list or an object
  as Python's repr inside the quotes (`tags="['a', 'b']"`). The assembler reads a quoted value by
  the kind the request's tool declares for the parameter, as the tagged tables read a value's
  text (`Dialect::LLAMA4`, `Values::Quoted`): a declared string stays the string, digits or not;
  an integer, an array, an object come back as what the text spells; a parameter no tool
  declares reads by its text, a number or a word as what it is and anything else as the string.
  The text between the quotes is the value's characters as they are: the template decodes no
  escape, so a backslash and an `r` in a value stay those two characters (the recorded patches
  are full of them), and the table reads them so; a backslash before the closing quote is the
  value's last character, not an escape, so a value that ends in one (`command="dir C:\"`, as
  the recorded benchmark sets write it) closes at its quote, whole and at every cut. One
  reading is kept from the model's side: a `"` right after a backslash is the model's own
  escaped quote (`size="5\" screen"`, which the family's own readers decode) unless `,` or `)`
  follows it, whitespace aside, so such a value reads whole with its backslash kept, and only a
  quote the value's end follows closes it. A value the
  model writes bare (`days=3`, `detailed=False`, `city='Paris'`), the spelling its model card and
  the family's parsers take, reads as Python's, a single-quoted string with its escapes. The
  structural tags write Python's values, the spelling the model is told to write.
- **A quote inside a quoted value is ambiguous.** The template does not escape a `"` inside a
  string it quotes, so the bytes it writes for such a value do not say where the value ends; a
  list or an object between the quotes is Python's repr, which quotes an element, or a key, that
  holds an apostrophe with `"`, so a nested apostrophe raises the same question. The table reads
  the value whole when the quotes inside it pair up and nothing between the two of a pair would
  end or reopen the value in the reader's eyes: no `,` or `)`, no `'` and no opening bracket (the
  recorded cases are written so: a word in quotes, a JSON object, a command line); it cannot
  otherwise, since the bytes alone do not say which reading the model meant. The recorded sets
  hold a few such cases (five in a hundred thousand of Scout's under the `"` form alone; the
  apostrophe form was not counted), the recorder refuses the class at record time, and the corpus
  replay counts the ones recorded before that apart, by the reference that holds the quote (a `"`
  at any depth, a `'` below the top level, the keys of a nested object included), rather than
  against the table. A string that holds a newline or a backslash reads as written.
- **`<|python_start|>` and `<|python_end|>` are wrapping wherever they stand.** The family's
  earlier template wrote a tool turn's content between them, and the model may write them around
  its code or its list; the table ignores both terminals, so the text between them reads as what
  it is, content or the list.
- **The prompt ends at the turn's opener.** `<|header_start|>assistant<|header_end|>` and two
  newlines; after a tool result the previous turn's list stands in the prompt, and the replay
  starts at the last opener, in content, where the template puts the answer.

What the table leaves open: a live reading on an engine (owed with the family's weights); the
base checkpoints' template (`llama-4-scout-17b-16e`), which writes JSON objects back to back behind
the python markers, another format no table reads yet; and the structural tag of a forced call,
which writes the bare list as the `xlam` table's does and is not yet proven on an engine.

## The forced call (`forced`)

Not a table an operator names: the gateway picks this format for a request whose `tool_choice`
names a function and whose constraint is the JSON schema the older path builds for it (the
function's `parameters`), so the whole output is that function's arguments, with no marker and no
name in it. The name comes from the request (`Declared::forcing`), and `ToolCallStart` carries it
with no source byte of its own. The value streams through `json::Arguments` (`json::forced` drives
it): the whitespace before it is dropped, a fragment for every byte, a JSON prefix until a
`Malformed { InvalidArguments }` event of no bytes says where it stopped being one, and the call's
end where the value is whole or where the stream ends; the engine reports text after a whole value
as text after the object. A required call
under the same path is a list of `{"name": …, "parameters": …}` objects, which the `json` table
reads.

What the format leaves open: it derives no grammar, since its constraint is the request's own
schema; bellwether #19 records the constrained outputs, which replay here when they land, so the
format stands on its unit tests and on the gateway's reading of the same outputs until then.

## Llama Nemotron (`llama_nemotron`)

What the table says beyond the others:

- **The calls are one list between markers.** `<TOOLCALL>` opens the call state and `</TOOLCALL>`
  closes it; inside, the list assembler (`json::list`) reads `[{"name": …, "arguments": {…}},
  …]` as it reads xLAM's bare list: each object a call, the brackets and the commas the template's
  wrapping. Between markers prose has no place, so text where the list should be, a `[` that opens
  no list of calls included, is malformed where the bare list takes it for content; the engine
  picks that form for a table with a row into the arguments state. After the `]`, whitespace
  before the closing marker is dropped; text is malformed with the whitespace before and between
  its words, as for the bare list.
- **The thought is the Super template's, and the model's own.** Llama-3.3-Nemotron-Super-49B-v1.5's
  template renders an assistant message's content from its `</think>` on, so the model writes
  `<think>…</think>` itself; `/no_think` in the system prompt makes the template close the thought
  in the prompt (`<think>\n\n</think>\n\n`), and the request's `enable_thinking` changes nothing.
  The Nano 8B and DLER templates have no thought. Bellwether records no reasoning case for any of
  the five groups (their templates render no `reasoning_content`, and the Ultra v1's reasoning probe
  was refused at record time), so the thought's rows rest on the Super template and the model
  cards, not on a replay.
- **The tag is one tag around the list.** `Format::grammar` writes `<TOOLCALL>`, the tools' calls
  as the list, `</TOOLCALL>`, and the table's thought takes the prefix the JSON syntax takes. The
  gateway's older parsers have no tag for this syntax (`llama` reads Llama 3.2's pythonic calls,
  `json` the bare object), so nothing is pinned.
- **The v1 checkpoints bring their vendor's syntax, not their template's.** The chat templates of
  Llama-3_3-Nemotron-Super-49B-v1 and Llama-3_1-Nemotron-Ultra-253B-v1 are a Llama 3 header
  template that renders no tool and no call and keeps an assistant message from its `</think>` on:
  the thought is the model's own (`detailed thinking on` in the system prompt), and the calls are
  the `<TOOLCALL>` list the vendor's tool parser reads, written when the system prompt lists the
  tools. Both read with this table; the CPT checkpoint of the Ultra keeps Llama 3.1's own
  template and reads with `json`. The Ultra's recorded set holds content alone, as the template
  writes; the Super v1 has no recorded set.

What the table leaves open: the thought's rows against a recorded reasoning case; Nemotron Nano 4B
v1.1, whose template writes each call one `}` short (`{"name": …, "arguments": {…}]`), which
bellwether's reference heals and this table does not read; and the Ultra's CPT checkpoint
(`Llama-3_1-Nemotron-Ultra-253B-CPT-v1`), whose bare `{"name": …, "parameters": …}` object is
Llama 3.1's own syntax, which the `json` table reads.

## MiniCPM5 (`minicpm5`)

The template writes the turn as `<think>\n`, the reasoning, `\n</think>\n\n`, the content, a newline
when there is content, then the calls one after another with a newline between them, each `<function
name="NAME">`, its parameters with nothing between the tags, `</function>`. A value that holds `<`,
`&` or a newline is a CDATA section, `<![CDATA[…]]>`, which the assembler unwraps
(`tagged::attributed`); every other value is written as Python prints it (`True`, `['a', 'b']`),
which the typed reader takes. The thought is the model's to write: the generation prompt is
`<|im_start|>assistant\n` alone unless the request sets `enable_thinking`, when the template opens
the thought (`<think>\n`) or closes it empty (`<think>\n\n</think>\n\n`), as Qwen3's does; the
newlines around the thought and before the calls reach the reasoning and content streams (bellwether
#17's question). A turn opens with `<|im_start|>assistant`, so the prompt's replay starts there.

A forced call's structural tag is each call as a tag its opener triggers, the parameter tags with
nothing between them (`grammar`). The table carries its template's two newlines after the thought's
close (`</think>\n\n` before the content or the first call) as its thought layout, so a forced call
written where the close left the engine begins with them and not with `<function name="`; a prompt
with thinking switched off has written them itself, and the tag owes none. The table's row out of
the thought on a call's opener reads a call wherever it stands, inside the thought too.

## dots3 (`dots3`)

The template writes the turn as `<think>\n`, the reasoning, `\n</think>\n\n`, the content, then for
each call `\n<dots_function_call>\n<invoke name="NAME">`, for each argument `\n<parameter
name="KEY">\n`, the value, `\n</parameter>`, then `\n</invoke>\n</dots_function_call>`; a string
value as it is, every other value as JSON (`tojson`). The newlines the template writes after a
parameter tag and before its closing tag frame the value and are not part of it; the ones around the
thought and before a block are the model's to write and reach the reasoning and content streams
(bellwether #17's question, as for Qwen3). The thought is the model's to write: the generation
prompt is `<|assistant|>` alone, or with the thought closed empty (`<think>\n\n</think>\n\n`) when
the request turns thinking off, as Qwen3's does; a turn opens with `<|assistant|>`, so the prompt's
replay starts there.

The table carries its template's two newlines after the thought's close (`</think>\n\n` before the
content or the calls) as its thought layout, so a forced call written where the close left the
engine begins with them and not with `<dots_function_call>` (`grammar`); a prompt with thinking
switched off has written them itself, and the tag owes none. The newline the template
writes before every block is the call's own lead, which the tag does not write: a block written
after the two newlines reads as one written after three. The table's row out of the thought on the
block's opener reads a block wherever it stands, inside the thought too.

## Gemma 4 (`gemma4`)

The template writes the thought as `<|channel>thought\n`, the reasoning, `\n<channel|>`, then a
turn's calls one right after another with nothing between them, then the content; the newline before
`<channel|>` is the model's to write and reaches the reasoning stream (bellwether #17's question, as
for Qwen3). The thought is the model's to write: the generation prompt is `<|turn>model\n` alone
whatever the request says (the template opens the thought channel only after a tool response, and
`enable_thinking` is off unless the request sets it), so the output starts in content. A turn opens
with `<|turn>model`, so the prompt's replay starts there.

The template sorts a call's keys, so a call's arguments come back with the keys in that order
whatever order the request gave them: the replay counts such a case as respelled, not as a
difference (bellwether #110). `Format::grammar` writes a forced call's tag in this spelling, the
keys sorted the same way (the grammar's `gemma` module).

## FunctionGemma (`function_gemma`)

What the table says beyond the others:

- **Gemma's shape in the template's own tokens.** Each call stands between `<start_function_call>`
  and `<end_function_call>`, `call:NAME{key:value,…}` inside with the keys sorted as Gemma 4 sorts
  them; a string's text stands between `<escape>` tokens and a null is written `None`, while
  `true` and `false` keep JSON's words (`gemma::Spelling::FUNCTION_GEMMA`). The assembler and the
  grammar take the spelling from the table, so Gemma 4 and FunctionGemma share both. A bare
  `null`, which this template never writes, comes back as the string `"null"`, as any bare text
  the spelling does not know.
- **No thought, content before the calls.** The generation prompt is `<start_of_turn>model\n`
  whatever the request says, and the template renders no `reasoning_content`; the calls follow the
  content one after another with nothing between them, and the turn ends after the last.

What the table leaves open: the spellings come from the recorded common set, since the template is
gated on the Hub: `true`, `None`, integers and a decimal are seen there, `false` and an exponent
are not, so they follow Gemma 4's words and Python's print of a number until a set shows otherwise.

## K2-Horizon (`k2_horizon`)

What the table leaves out, for want of fixtures: the template picks the thought's tag by
`reasoning_effort` (`<ifm|think_fast>` and `<ifm|think_faster>` below the default), and it can write
the calls as JSON objects or with a typed tag per argument (`tool_call_format`); the recorded sets
hold the default thought and the default calls, read here. A thought under one of the other tags is
content until a set shows it.

## Muse Spark (`muse_spark`)

The template opens the turn with `<|start|>assistant` and writes every message as
` to=RECIPIENT<|message|>`, its body, and `<|eom|>` when another message follows or `<|eot|>` when
the turn ends: the thought to `self`, the answer to `user`, an update before or between calls to
`commentary`, and each call to the tool it names, its body `<atem:function_calls>`, a newline,
`<atem:invoke name="NAME">`, each `<atem:parameter name="KEY">` tag on a line of its own, the
invoke's close, a newline and the block's close. The checkpoint's `tokenizer_config.json` declares
the same reading as a response template, and the table spells it the way a compiled one would.
What the table says:

- **The text between the messages is framing.** The turn's own state (`message`) holds the
  headers, the block around a call, `<|start|>assistant` between messages, and whatever the model
  writes outside a message; the template's reader drops all of it, and so does the table
  (`Format::framing`): dropped as wrapping, never malformed, never content. An output with no
  message at all is dropped whole, as the reader gives nothing for it.
- **A call is the invoke, wherever its message put it.** `<atem:invoke` opens a call given as data
  ([`CallSyntax::Shaped`]) from the turn's own state only: its header skips other attributes
  (`id="1"`) and names the tool, and an invoke without a name is framing. Inside the thought or an
  answer the same markup is text, so a call the model quotes to the user is prose, and only a
  message addressed to a tool makes a call. The parameters are tags found as the reader's
  `finditer` finds them, each value read as JSON where it is JSON and as text otherwise, then typed
  by the request's tools. `<|eom|>` or `<|eot|>` before the invoke's close ends the call with what
  arrived, and so does the model's stop (the stop row).
- **Both channels of the answer are content.** `user` and `commentary` open the content state, in
  the order the model writes them, so an update before a call and the answer after the tool's
  result both reach the client as content; the thought is the `self` message. The bodies are the
  model's bytes as written: the template puts nothing between `<|message|>` and the body, and the
  reader's `strip` of a field's edges is bellwether #17's question here as for Qwen3.
- **The prompt's replay starts after the last `<|start|>assistant`** (`Format::turn_anchor`), so
  an earlier turn's call, or a tool result that quotes a header, moves nothing; a prompt that
  continues an assistant message (`to=self<|message|>` or `to=user<|message|>` after the opener)
  leaves the output inside that message.
- **The tags write what the chat template renders** (`Format::with_layout`): a forced or required
  call is the message to the tool, ` to=NAME<|message|><atem:function_calls>`, the invoke, and the
  block's close; a second call is the next message. A JSON answer (`response_format`) is the
  message to `user` with the schema as its body, ended by the turn's stop. The thought is the
  model's to write before them, as gpt-oss's analysis channel is: the template renders a thought
  only when the message has one, so the tags do not require it.

What the table leaves open: the Spark checkpoint's own recorded set, once bellwether has it (the
table replays Muse Glimmer 30B's set, whose template writes the same turn of messages, bitwise);
whether the model ever writes a message to a recipient the table has no row for (its text is
framing today); and whether `commentary` should reach the client as content or as a channel of its
own, which the API has no place for.

## Kimi K2 (`kimi_k2`)

What this table says beyond the DeepSeek V3.1 one, whose rows it has (a thought, content, a
wrapper state for the block, an arguments state per call):

- **One table for the line, and the prompt decides where the output starts.** The six
  templates (Kimi K2 Instruct, Instruct-0905, K2 Thinking, K2.5, K2.6, K2.7-Code) write the same
  block of calls; they differ in the thought alone. K2 Instruct writes none; K2 Thinking's
  generation prompt ends at `<|im_assistant|>assistant<|im_middle|>` and the model writes its
  own `<think>`; K2.5's and K2.6's end with `<think>`, or with `<think></think>` when the request
  turns thinking off; K2.7-Code's end with `<think>`, with no switch. The prompt's replay puts
  the output inside the thought or in content accordingly. Under K2 Instruct a `<think>` the
  model never writes would be read as a thought, the choice the DeepSeek V3 table made for the
  same reason: one table serves the deployments that name the line's older parsers. One
  departure from the older `kimi_thinking` parser, which read everything before `</think>` as
  the thought whether or not `<think>` came first: K2 Thinking starts in content here, so an
  output that closes a thought it never opened is content as written, the marker included (the
  template always writes the opener; the module tests pin the reading).
- **The id is a call given as data.** `<|tool_call_begin|>` opens a call only where
  `functions.NAME:N<|tool_call_argument_begin|>` follows it (the shaped header: `functions.`,
  which the gateway's older `kimik2` parser also reads without, the name, `:`, the index, and
  the whitespace the vendor's reader allows around the id); the name is read from the id, and
  the call carries the id as the model wrote it (`Format::call_id(CallId::Header)`: the line's
  templates render a turn's ids back into the prompt and a tool result names its call by that
  id, so the gateway numbers no K2 call). An id the header does not match
  opens no call: the opener and the id are the block's text (`Malformed`, since the block is a
  wrapper state). The arguments are one JSON value, closed by `<|tool_call_end|>` with nothing
  between, so a closing marker inside a string argument ends the call there (bellwether #16; the
  marker is one token the model cannot write as text), and the block's close ends a call still
  open with what arrived. A second `<|tool_call_begin|>` inside an open call is the call's text:
  a shaped call's opener may open no call, and the engine cannot hand its text back to a state
  that holds calls, so the table has no row from `call` into `call`; the templates close every
  call before the next.
- **Every marker is one token.** `<|tool_calls_section_begin|>`, `<|tool_call_begin|>`,
  `<|tool_call_argument_begin|>`, `<|tool_call_end|>` and `<|tool_calls_section_end|>` are each
  a token of the vocabulary, so a `max_tokens` cut cannot split one; the table declares no
  structural token.
- **What the table stands on.** bellwether holds a manifest for each of the six checkpoints and
  no recorded set yet, so the table stands on the module's tests, which hold a hand-picked set of
  the turns the templates render for the shapes a tool-calling conversation takes (an answer, a
  call, two calls, content before a call, a thought and a call, the turn after a tool result, a
  call with thinking off, arguments that nest and hold text outside ASCII, a brace inside a string
  value), read whole and at every cut, and on the id contract. Those shapes are hand-written cases
  of bellwether's sets, which its recorder renders with each template at its pinned revision; the
  six slugs are listed in the corpus harness (`tests/bellwether_parse_fixtures.rs`), which replays
  the recorded sets from the fixture pin once it holds them, and the README rows say awaiting
  fixtures until then. Two things the templates taught: K2
  Thinking, K2.5 and K2.6 (without `preserve_thinking`) keep the thought of a message that calls
  and render an answer's as `<think></think>`, so the thought-and-answer turn is rendered where
  the template keeps it (K2.7-Code, K2.6 with `preserve_thinking`); and the two Instruct
  templates differ in a block's indentation alone and render the same turns.
- **The forced tag is written from the table's layout.** A shaped call's tag needs the one
  spelling the chat template renders, and the table carries it (`Format::with_layout`): the id as
  `functions.{name}:{index}<|tool_call_argument_begin|>` with the call's number as a place of its
  own (`Segment::Index`, written as a run of digits, since the model counts its calls and no
  request knows the count ahead), the arguments one JSON value (the tool's schema, no parameter
  and nothing around it), `<|tool_call_end|>` as the call's close, nothing between two calls, and
  the thought closed by `</think>` with nothing after it. The calls sit in the table's block, so
  the tag is the block's ways in (from the turn's own state, or `</think>` first) around one or
  more calls and the block's close, as the tagged syntaxes' tags are. On an engine whose grammar
  starts after its reasoning parser has seen `</think>`, a K2 Thinking answer keeps its thought
  and the forced call follows it in the model's own syntax; a prompt that opens the thought (K2.5,
  K2.6, K2.7-Code) takes the thought closed first from an engine that starts at the first token.
  Before the table had its layout, the gateway handed a forced or required request to its older
  path, whose JSON-schema constraint made the model write the bare arguments after the thought,
  and the older parsers read that back as raw text (the module tests pin how the table reads
  those bytes: a thought and JSON content, no call). What the table leaves open: the recorded
  sets, and the live reading against the engine's own server.
