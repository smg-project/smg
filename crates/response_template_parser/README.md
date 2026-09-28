# Response template parser

Parse assistant output using a `response_template` object from
`tokenizer_config.json`. A template declares the delimiters for reasoning,
content, and repeated function calls. Selection depends on the tokenizer
metadata; no model name or architecture check is required.

## Supported schema

The initial schema requires exactly three fields:

| Field | Content parser | Repeated | Additional requirements |
| --- | --- | --- | --- |
| `thinking` | `text` | No | No `content_args` or `transform` |
| `content` | `text` | No | No `content_args` or `transform` |
| `tool_calls` | `xml-inline` | Yes | Named `name` capture in the opener, `content_args`, and `transform` |

`start_anchor_pattern` must match the rendered prompt prefix. `open_pattern`
and `tag_pattern` use Rust's `regex` syntax. Empty-matching patterns,
lookaround, and backreferences are rejected. `close` is a literal string or
an array of literal strings, rather than a regular expression.

For `xml-inline`, `tag_pattern` must capture `key` and `value` in that order
and end in a literal closing tag. `value_parser.name` is `text`;
`value_parser.args.strip` controls surrounding whitespace. Arguments remain
strings. A transform substitutes `{name}` and `{content}` recursively;
whole-value `{content}` retains the argument object's type.

For example, a tool-call field can use:

```json
{
  "open_pattern": "<tool name=\"(?P<name>[^\"]+)\">",
  "close": "</tool>",
  "content": "xml-inline",
  "content_args": {
    "tag_pattern": "(?s)<arg name=\"(?P<key>[^\"]+)\">(?P<value>.*?)</arg>",
    "value_parser": { "name": "text", "args": { "strip": true } }
  },
  "repeats": true,
  "transform": { "name": "{name}", "arguments": "{content}" }
}
```

## Parsing and limits

`ResponseTemplateParser::from_json` validates and compiles a template.
`parse_complete` parses a complete response. `stream` creates independent
state whose `feed` method accepts byte slices, including partial UTF-8
characters. Call `finish` once at the end.

Fields are emitted after their closing delimiter arrives. A field still
open at the end of the response is an error. Missing reasoning and content
fields use their string defaults; an omitted tool-call field produces no
calls, and `defaults.tool_calls`, when supplied, must be an empty array.
Delimiters and unrecognized wire text are never emitted as content.

Pending delimiter state, each structured value, and each field body have
independent 4 MiB default limits in `ParserConfig`. A failed stream stays in
an error state: subsequent `feed` and `finish` calls return the same error.

## Gateway integration

The gateway validates metadata during local or remote tokenizer registration.
Invalid templates fail registration without retrying or falling back to a
different tokenizer. Ordinary tokenizer I/O failures retain their retry path.

The gRPC Chat Completions path uses the declared template for reasoning,
content, and function calls; Responses consumes that mapped Chat output.
Template delimiters remain available to the parser even when an end delimiter
is also an EOS token. Tokenizers without a template keep their existing
reasoning and tool parser selection. This does not add parsing to HTTP proxy
responses, raw Completions, or the Messages API.

Run the parser and gateway checks with:

```bash
cargo test -p response-template-parser
cargo test -p smg --test response_template_test
```
