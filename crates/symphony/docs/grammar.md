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
`xml` module); Gemma 4's `call:NAME{key:value}` spelling, the keys sorted and each string between
`<|"|>` tokens (the `gemma` module); for Inkling, the call marker and a JSON object whose `args` the
schema accepts, wherever the marker stands in a message. The gateway's old parsers wrote these tags
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
nothing after the object. A table without calls gives `None`, and the
gateway takes the JSON-schema path for it.
