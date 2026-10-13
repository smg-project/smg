# Notes on the adapters

`src/adapt/` renders events in each API's shape. The module docs say what each adapter is; these
notes say what each one decides where the events say more than its API can, how its shapes match
what the gateway sends today, and where it differs from the older path on purpose.

## Chat Completions (`adapt::chat`)

The shapes are the ones SMG's gateway sends today, the same fields in the same places:

- content and reasoning deltas carry `role: "assistant"`; reasoning goes to `reasoning_content`;
- a call begins with its `id`, `type: "function"` and `name` and no arguments; every later
  fragment carries the call's `index` and `arguments` only;
- the finishing choice has an empty delta without a role;
- the whole message has `content` null when it is only whitespace, `reasoning_content` only when
  there was reasoning, and `tool_calls` only when a call was made, each call with its id, name
  and its argument fragments joined.

One rule for `finish_reason`, in both shapes: the engine's reason, except that `stop` after at least
one tool call becomes `tool_calls`. `length`, `abort` and engine-specific reasons stay what the
engine said even when calls were made. The old streaming path rewrote `stop` only, as here; the old
non-streaming path turned every reason but `length`, `failed` and `error` into `tool_calls`, so the
two differ on `abort` and engine-specific reasons, where the rule keeps the information.

Policy this adapter sets, where the events say more than Chat Completions can:

- `ReasoningStart`, `ReasoningEnd`, `ToolCallEnd` and `Dropped` have no representation and
  produce nothing; the markers a format consumes never reach the client.
- `Malformed` text is content: the model wrote it, and the client sees it rather than losing
  it. This is the opposite of the old driver, which logged a parser error and sent nothing for
  that chunk.
- An empty text or an empty argument fragment produces nothing; the gateway sends no empty deltas.

## Responses (`adapt::responses`)

The items are the ones SMG's gateway builds today from a finished chat message, in OpenAI's order,
so items replayed as the next turn's input rebuild the same turn:

- one `reasoning` item when there was reasoning, its text as one `reasoning_text` part and an
  empty summary;
- one `message` when the content is not whitespace: `role: "assistant"`, one `output_text` part
  with every content and malformed text in order, and no annotations;
- one `function_call` per call in the order the calls started, its argument fragments joined;
  `output` is absent, since running the call belongs to the next turn.

This is the same folding as `chat::message`, so the two adapters agree on what a turn contains.

Ids: the reasoning item is `rs_<response id>` and the message `msg_<response id>`, the prefixes the
Responses API uses; a call's `id` and `call_id` are both the id from its event. The old gateway used
two schemes, `reasoning_` and `msg_` before the response id when it converted a finished message and
fresh `rs_`, `msg_` and `fc_` ids when it streamed; this is one. A second reasoning item, which only
a model that thinks again after answering produces, carries its output index after the response id
(`rs_<response id>_<output index>`), so ids stay unique.

Status, from the `Finish` event: `stop` and `tool_calls` complete the response; `length` leaves it
`incomplete` with `max_output_tokens` as the reason; the engine's `failed` or `error` fails it;
`abort` cancels it, where the old path called an aborted response complete; any other
engine-specific reason completes it. A sequence without `Finish` is `in_progress`.

Item statuses follow. While the response has not finished, and when the engine failed, an item is
`in_progress` unless its own events ended it: a reasoning item whose region closed is `completed`, a
call that ended is `completed`, and the message, which ends only with the response, is
`in_progress`. Once the response finished otherwise, every item is `completed`. In either case a
call whose argument fragments never became a whole JSON value is `incomplete` rather than
`completed`. Three of these differ from the old path, which called the reasoning item `completed`
whatever the response's state, left every item of a failed response `in_progress`, what had finished
included, and marked a call `incomplete` only after `length`; here each status describes its item. A
reasoning region ends at `ReasoningEnd`, or when content with substance or a call comes after it, in
the fold as in the stream. A call with empty arguments is complete: the formats start a call only
once its arguments value has begun or its object has closed, so a call without arguments is one the
model closed.

Policy this adapter sets, where the events say more than the Responses API can:

- `ReasoningStart` and `Dropped` shape nothing, and `ReasoningEnd` and `ToolCallEnd` only end
  their items (the statuses above, and the stream's `done` events): the markers a format
  consumes never reach the client.
- `Malformed` text is content, as in `chat`: the model wrote it, and the client sees it rather
  than losing it.
- Argument fragments for a call that never started cannot occur under the `Event` contract;
  should one arrive, its text joins the content so that no byte is lost. In the stream the same
  goes for a fragment of a call whose item is already complete.
- In the stream, whitespace-only content opens no message item until content with substance
  joins it, and leads that content when it does; held whitespace with nothing after it is let go
  at the end, as the fold leaves whitespace-only content out. The old emitter opened a message
  item for any non-empty content.
- In the stream, a call's item completes when the call ends and the reasoning item when its
  region ends; the old emitter completed every call at the end of the response and the
  reasoning item when the answer began. The items and parts are the typed ones, so a message's
  `output_item.added` carries `status: "in_progress"` and a text part its empty `annotations`,
  as the API's own events do; the old emitter's hand-written JSON left them out.

What stays with the driver, since it depends on the request rather than the output: the `error`
payload of a failed response, encrypted reasoning content, and a function's namespace and the
`custom_tool_call` shape, both read from the request's tool list.

## Messages (`adapt::messages`)

The blocks are the ones SMG's gateway builds today from a finished turn, in the order the Messages
API uses:

- one `thinking` block when there was reasoning, with an empty `signature`: the engines sign
  nothing, and a signature a client could verify needs a decision of its own;
- one `text` block when the content is not whitespace, every content and malformed text in
  order;
- one `tool_use` block per call in the order the calls started, its `input` the joined argument
  fragments parsed as JSON.

This is the same folding as `chat::message` and `responses::output`, so the three adapters agree on
what a turn contains.

Ids: a call id of the form `call_<suffix>` becomes `toolu_<suffix>`, the prefix Anthropic SDKs and
tooling pattern-match; any other shape passes through unchanged, since a model family whose id
format is load-bearing in the prompt must keep it. The rule is deterministic, so a client echoing
the id back names the same call.

Stop reason, from the `Finish` event: `length` is `max_tokens`, and parsed calls do not override a
truncation; `tool_calls`, and any other reason after at least one tool call, are `tool_use`, since a
message with `tool_use` blocks says so whatever stopped the model, as the API itself does; `stop` is
`end_turn`; `abort` and engine-specific reasons are `end_turn` too when no call was made, since the
Messages API has no word for them and the driver decides whether a failed generation is a message at
all. A sequence without `Finish` has no stop reason.

Policy this adapter sets, where the events say more than the Messages API can:

- `ReasoningStart` and `Dropped` shape nothing, and `ReasoningEnd` and `ToolCallEnd` only close
  their blocks in the stream: the markers a format consumes never reach the client.
- `Malformed` text is content, as in `chat`: the model wrote it, and the client sees it rather
  than losing it.
- A call whose argument fragments are not a whole JSON value, because the output was cut or the
  model wrote something else, has `{}` as its `input`: `input` must be a JSON value, and the
  fragments are not one. Chat carries the fragments as a string and Responses marks the call
  `incomplete`; here `max_tokens` is all a client learns. A call the model closed without
  arguments has `{}` too.
- Argument fragments for a call that never started cannot occur under the `Event` contract;
  should one arrive, its text joins the content so that no byte is lost. In the stream the same
  goes for a fragment whose call has no open `tool_use` block because other text, or another
  call, came between the call's start and the fragment: the Messages API streams one block at a
  time and cannot reopen a closed one, so the fragment is text, where the fold still gives it to
  its call. The formats parse a call from one contiguous region, so this does not happen for
  them, and nothing is dropped if it does; the old driver dropped such fragments.
- In the stream, the thinking block stops when the reasoning region ends rather than when the
  next text arrives, as the old driver did, so a client sees the block close as soon as the
  model stops thinking.
- In the stream, whitespace-only text opens no block until text with substance joins it; the old
  driver opened a text block for any non-empty text, so a separator between a thinking block
  and a call became a block holding one newline.
