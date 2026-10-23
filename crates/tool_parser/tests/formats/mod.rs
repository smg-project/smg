//! Tool-call format integration tests: the JSON, partial-JSON and pythonic parsers, the fallback to
//! plain text, the streaming flush and the edge cases across parsers.

mod tool_parser_edge_cases;
mod tool_parser_fallback;
mod tool_parser_json;
mod tool_parser_mixed_edge_cases;
mod tool_parser_partial_json;
mod tool_parser_pythonic;
mod tool_parser_streaming_flush;
