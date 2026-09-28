# smg-response-template

Validation and matching primitives for the `response_template` object that a
checkpoint can declare in `tokenizer_config.json`. The template describes the
model's output format as fields with a regex opener and literal closes:

| Field | Content | Notes |
| --- | --- | --- |
| `thinking` | `text` | becomes `reasoning_content` |
| `content` | `text` | becomes `content` |
| `tool_calls` | `xml-inline`, repeating | the opener captures `name`; each argument tag captures `key` and `value` and ends with a literal |

`CompiledTemplate::from_value` accepts exactly this subset and returns a
`TemplateError` for anything else. The `defaults` object is accepted but not
used, and `start_anchor_pattern` is only validated.

The crate has no parser of its own. `reasoning-parser` and `tool-parser` build
`TemplateReasoningParser` and `TemplateToolParser` on top of it, and the
gateway selects them when a model's tokenizer declares a supported template.

Matching works on a buffer that grows while a response streams:

- `CompiledTemplate::scan` finds the earliest opener. An opener that reaches
  the end of the buffer is `Pending` only while more input could still change
  it, and complete otherwise.
- `find_close` and `partial_close_len` find literal closes and the part of a
  close that may be split across chunks.
