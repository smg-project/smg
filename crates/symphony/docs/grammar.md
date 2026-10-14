# Notes on the grammar

`src/grammar.rs` derives a structural tag from a table and the request's tools. These notes say how,
syntax by syntax, how the tag is wrapped when the prompt leaves the thought open, and which tags are
pinned to the ones the gateway's older parsers hand-wrote.

## The grammar

`Format::grammar` derives the tree from the table. A region of the table (a call, a block of calls,
the thought, the answer) is opened by the terminal of a row into its state and closed by the
terminal of a row back to a state it is entered from; the terminal every state has a row for is the
end of the turn, which ends whatever is open and closes no region of its own. So the call markers
are the rows into and out of the arguments state, the block markers the rows into and out of the
wrapper state the calls sit in, and the thought's close the row from the reasoning state back to
where the thought opened; the request's tools give each call its name and schema. What the syntax
inside the markers constrains comes from the call syntax: a JSON object around each tool's schema
for the JSON family; the engines' `glm_xml` style of a schema for keyed arguments in the compact
spelling, and the table's own tags written one pair per property for Ling's newlines and Hy4's
suffixed tags (the `keyed` module); `<function=` and `<parameter=` tags, one per property, for Qwen
3.5's and Seed-OSS's tagged syntax (the `tagged` module); a tag grammar per parameter for DSML; for
Kimi K3's XTML, one argument tag per property of the schema (the `xtml` module); `<invoke name="…">`
blocks of `<parameter name="…">` tags for MiniMax M2 (the `attributed` module); Python calls, one
per line or as a list, each value in the template's spelling, for Olmo 3 and LFM2.5 (the `pythonic`
module); xLAM's bare list of JSON objects, written from the tools alone since its table has no
markers; for MiniMax M3, the XML tree of elements with the table's separator before every tag (the
`xml` module); Gemma's `call:NAME{key:value}` spelling, the keys sorted and each string between the
table's tokens, `<|"|>` for Gemma 4 and `<escape>` for FunctionGemma, which also writes a null as
`None` (the `gemma` module); for Inkling, the call marker and a JSON object whose `args` the
schema accepts, wherever the marker stands in a message. The forced call's format (`forced`) derives
no tag: its constraint is the request's own schema, which the gateway writes. The gateway's old
parsers wrote these tags
by hand, one function per parser; where one exists for a format Symphony has, the derived tag is
pinned to it in the tests, so the switch changes nothing an engine sees.

With the prompt's reasoning still open, the tag is wrapped in a prefix: free text that can write
none of the table's markers but the thought's own, nor the call syntax's inner tags, with nothing
owed at its end, so the forced call follows the reasoning instead of replacing it, as the gateway's
`wrap_in_reasoning_prefix` does. The text is the thought and its close where the engine applies the
grammar from the first token, and a newline or nothing where the engine runs a reasoning parser and
applies the grammar only after the model's own close; a prefix the thought's close ended would be
owed a second time there, and the model would fill the gap with a fabricated observation and answer.
A table without a thought has nothing to wrap, and the tag comes back as it is; a thought that does
not close back to where it opened has no prefix yet, and the table gives `None` rather than a tag
the model could write inside its thought. Kimi K3's tag takes no prefix: the block of calls can be
entered from the turn itself, from the thought or from the answer, and the tag has one way in per
state: from a region with a close of its own, that close and then the row into the block from where
the close returns; from any other state, its own row into the block. So the model closes whatever
the prompt left open on its way to the calls, whichever that was.

Inkling's named JSON takes the tag the gateway's old Inkling parser writes, the call marker its
trigger inside whatever message the model is writing. Mistral's named arguments take a tag per tool
from `[TOOL_CALLS]`, the name and `[ARGS]` to the tool's schema, with no end, as the template writes
nothing after the object. Llama Nemotron's list takes one tag around it, `<TOOLCALL>`, the tools'
calls as a JSON list, `</TOOLCALL>`; xLAM's bare list is the same list with no tag, alone in the
turn. A table without calls gives `None`, and the gateway takes the JSON-schema path for it.

## What a table read from a response template adds

A table read from a response template has rows of its own (`Format::framing`,
`Format::stop_tokens`). A framing state, whose text the template drops, lies between the messages
as a wrapper state does: a way in passes through it, and it holds no answer. A stop token ends the
region it closes at the model's stop, and the stop decoder hides it, so no tag writes it: a
region's close is never one, no way in steps on one, and an answer a stop token may close ends with
nothing after its JSON. Such a table may also have a layout (`Format::with_layout`): its tags then
write the thought and the answer with the text the layout writes around them. For a call given as
data, the tag writes its tagged parameters as the table's layout says the model's chat template
writes them (the `shaped` module), and a table without a layout has no tag.

The thought's close a tag writes is the row from the reasoning state back to where the thought
opened, the one the table's layout names where the template closes the thought more than one way.
The rows a way in follows are a path through the template's framing, the shortest the table has:
one row for Kimi K3, and more where a template frames each message (the thought's close, the row
that starts the next message, the row into the block), so a state with no row of its own into the
block still has a way in when the framing leads there.

`Format::grammar_for` takes what the request asks as `Options`, one call among them: a named
function, or `parallel_tool_calls: false`. Every syntax's tag then stops after its first call
(`stop_after_first`), a region that holds calls (a block, a list, Python's calls) holds that one,
and the call's end carries the terminal that ends the turn wherever the model stands, where the
table has one (Kimi K3's message close), since nothing may follow; a turn a stop token ends has
nothing after the call.

`Format::answer_grammar` is the tag for an answer that is JSON a schema accepts, the
`response_format` of a request, from the same table: the rows into the state the answer is written
in, the schema, and the answer's close with the end of the turn. Where the output starts in
content, as in ChatML's tables, it is the schema alone, after the prefix when the prompt left the
model inside its thought.

Whether a thought comes before the call or the answer is the table's, not the syntax's: a table
whose template writes the thought before every call and answer (`Format::thinks_first`) gets it
written first, its opener, its text and its close, where the prompt has not opened it; every other
table leaves the thought to the model, as gpt-oss's template leaves its analysis channel.

A marker that is one token of the model's vocabulary (`Format::marker_tokens`: a terminal, a
structural token, one of the syntax's own tags (`CallSyntax::tag_texts`), or a token inside one) is
written as that token (`Grammar::Token`) wherever a tag writes it inside a tag or between tags, so
the model cannot spell it in ordinary tokens there; a tag's begin and end and its triggers stay
text, as the engines' `triggered_tags` takes them.
