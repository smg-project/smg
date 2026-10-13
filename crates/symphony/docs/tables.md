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
  turn opener is `<｜Assistant｜>`, so a marker quoted in an earlier turn moves nothing.

## GLM 4.5 and 4.6 (`glm4`)

The template writes a newline before the thought, before the content and before every call, and
strips the content's edges, so those newlines are the model's to write and reach the content stream;
whose they are is bellwether #17's question, as for Qwen3.

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
  and inside an invoke the assembler keeps its bytes in place.
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
  is one call's arguments, read by the XTML assembler (`tagged::xtml`).
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
  after the turn; an engine that passes it through has it dropped wherever it stands.
- **A forced call's tag is the gateway's.** `Format::grammar` writes the tag the gateway's old
  Inkling parser hand-writes: the call marker triggers it inside whatever message the model is
  writing, the first one included, then `{"name":"NAME","args":`, the schema, `}` and
  `<|end_message|>`; the tool's name before the marker, a thought and the template's empty
  answer are free text, so the recorded thinking turn stays writable. A marker outside a
  message's header is writable under the tag too and reads as that message's text: the
  grammar is wider than the table there, as the gateway's tag is.

What the table leaves out, for want of fixtures: image and audio messages, which the template writes
for the user and never for the model; and the text-mode invocation, `<|content_invoke_tool_text|>`,
which the gateway's older Inkling parsers drop as protocol data. No recorded set holds one. The
table has no terminal for it, so such a message sits in the named-JSON header until
`<|end_message|>` and comes back whole as `Malformed` (a name without a call), no byte lost; a
recording of one decides what it is.

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
nothing between them (`grammar`).

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

## K2-Horizon (`k2_horizon`)

What the table leaves out, for want of fixtures: the template picks the thought's tag by
`reasoning_effort` (`<ifm|think_fast>` and `<ifm|think_faster>` below the default), and it can write
the calls as JSON objects or with a typed tag per argument (`tool_call_format`); the recorded sets
hold the default thought and the default calls, read here. A thought under one of the other tags is
content until a set shows it.
