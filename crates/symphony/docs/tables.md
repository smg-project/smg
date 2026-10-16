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
- **The template writes two newlines before the block.** They are what the template renders and
  what this table's forced-call tag writes (`Tags::DSML_V4.before_block`), not what its reading
  needs: the table enters the block on its opener wherever it stands, after the thought's text or
  the content's last byte as much as after the newlines, and the newlines the model writes are
  content, the model's own (the legacy parser trims them; bellwether #17's question, as for Qwen3).
  The family's reference parser is the one that looks for the block after the two newlines. The prompt ends with `<think>` in thinking mode and with `</think>` right after
  `<｜Assistant｜>` in chat mode, so the replay starts the output inside the thought or in the content;
  bellwether has not recorded the group's set yet. The table is pinned against the turns the
  family's reference encoder (the model repository's `encoding` module) renders for a tool-calling
  conversation, each read as the encoder's own parser reads it: every kind of value, two invokes in
  one block, a call after content, and an invoke with no parameters, which the encoder writes with
  an empty line inside and the table reads as `{}`.

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

## Mistral v13 (`mistral_v13`)

What the table says beyond the others:

- **A call has no closing marker.** `[TOOL_CALLS]` opens the call state, and the state's own rows
  are the next `[TOOL_CALLS]` and the end of the turn: the arguments object's close ends the call
  (`json::named_arguments`), the next opener or the turn's end ends the region. Text after a closed
  object and before the next opener is the template's wrapping: whitespace dropped, anything else
  malformed. The tool's name is the text between the opener and `[ARGS]`, taken as written; the
  object after `[ARGS]` is the arguments alone, so `ToolCallStart` carries the name and the marker
  as its source, and the fragments are the object's bytes.
- **The prompt ends at `[/INST]`, or after a tool result.** The template writes no assistant opener
  and no thought in the prompt, whatever the request says: the model writes `[THINK]` itself. After
  a tool result the prompt ends in `[/TOOL_RESULTS]`, and its last `[/INST]` stands before the
  previous turn's calls; the replay walks into the call state, which the prompt never enters, so
  the output starts in content, where the template puts it.
- **A forced call's tag has no end.** `Format::grammar` writes a tag per tool from `[TOOL_CALLS]`,
  the name and `[ARGS]` to the schema, and nothing after it, as the template writes nothing: the
  object the schema accepts ends the tag. The gateway's older `mistral` parser and its tag read
  Mistral 7B v0.3's `[TOOL_CALLS][{…}]` list, another syntax, so nothing is pinned to it; the
  table's name keeps that parser's apart, so `--tool-call-parser mistral` stays with it under
  `--parser symphony`, and the registry picks this table for the v13 groups.

What the table leaves open: whether every engine takes a tag whose `end` is empty (xgrammar's
structural tag allows it; the Symphony e2e lane says so for the engines it runs); and the older
list syntax, which no recorded group writes.

## JSON (`json`)

What the table says beyond the others:

- **One state, no markers, an object or a list.** As `xlam`, the output starts in the arguments
  state and the assembler decides at the first byte that is not whitespace; unlike `xlam`, a `{`
  there is one call alone (`json::list::Assembler::object_or_list`), whose object reads as a
  list's object does, `arguments`, `parameters` or `args` naming the arguments. What follows the
  object is the template's wrapping: whitespace dropped, anything else malformed.
- **Prose before the JSON is content, whole.** The gateway's older `json` parser searches the
  output for the first `{` or `[`; this table does not, as no other table scans for a marker.
- **A forced call is the list.** `Format::grammar` writes the `xlam` list, as the older parser's
  forced calls were written.

What the table leaves open: the format is Llama 3.1's own template's, which six recorded groups
write (Nemotron Ultra 253B CPT, the UltraLong 1M, 2M and 4M, Safety Guard 8B v3, the 70B Reward
Principle): one call alone, `parameters` naming the arguments, and never a list, content beside a
call or a thought, since the template refuses them; the list and the wrapping after an object rest
on the unit tests until a group writes them.

## The forced call (`forced`)

Not a table an operator names: the gateway picks this format for a request whose `tool_choice`
names a function and whose constraint is the JSON schema the older path builds for it (the
function's `parameters`), so the whole output is that function's arguments, with no marker and no
name in it. The name comes from the request (`Declared::forcing`), and `ToolCallStart` carries it
with no source byte of its own. The value streams through `json::Arguments` (`json::forced` drives
it): the whitespace before it is dropped, a fragment for every byte that keeps it a JSON prefix,
malformed from the first that does not, and the call's end where the value is whole or where the
stream ends; the engine reports text after a whole value as text after the object. A required call
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
  picks that form for a table with a row into the arguments state. After the `]`, whitespace is
  dropped and other text malformed, as for the bare list.
- **The thought is the Super template's, and the model's own.** Llama-3.3-Nemotron-Super-49B-v1.5's
  template renders an assistant message's content from its `</think>` on, so the model writes
  `<think>…</think>` itself; `/no_think` in the system prompt makes the template close the thought
  in the prompt (`<think>\n\n</think>\n\n`), and the request's `enable_thinking` changes nothing.
  The Nano 8B and DLER templates have no thought. Bellwether records no reasoning case for the three
  groups (their templates render no `reasoning_content`), so the thought's rows rest on the Super
  template and the model cards, not on a replay.
- **The tag is one tag around the list.** `Format::grammar` writes `<TOOLCALL>`, the tools' calls
  as the list, `</TOOLCALL>`, and the table's thought takes the prefix the JSON syntax takes. The
  gateway's older parsers have no tag for this syntax (`llama` reads Llama 3.2's pythonic calls,
  `json` the bare object), so nothing is pinned.

What the table leaves open: the thought's rows against a recorded reasoning case; Nemotron Nano 4B
v1.1, whose template writes each call one `}` short (`{"name": …, "arguments": {…}]`), which
bellwether's reference heals and this table does not read; and Nemotron Ultra's bare
`{"name": …, "parameters": …}` object, Llama 3.1's own syntax, which the `json` table reads.

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

What the table leaves open, for want of fixtures: the recorded sets, once bellwether has them;
whether the model ever writes a message to a recipient the table has no row for (its text is
framing today); and whether `commentary` should reach the client as content or as a channel of its
own, which the API has no place for.
