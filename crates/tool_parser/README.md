# tool-parser

Parser library for extracting tool/function calls from LLM model outputs. Supports both complete parsing and streaming incremental parsing with state management.

## Supported Formats

| Parser | Model Family | Format |
|--------|--------------|--------|
| `CohereParser` | Cohere Command (CMD3/CMD4) | `<\|START_ACTION\|>{...}<\|END_ACTION\|>` |
| `MistralParser` | Mistral, Mixtral | `[TOOL_CALLS][{...}]` |
| `QwenParser` | Qwen 2/2.5/3 | `<tool_call>{...}</tool_call>` |
| `QwenXmlParser` | Qwen3-Coder, Qwen3.5+ | `<tool_call><function=name>...</function></tool_call>` |
| `LlamaParser` | Llama 3.2 | `<\|python_tag\|>{...}` |
| `PythonicParser` | Llama 4, DeepSeek R1 | `[func_name(arg="val")]` |
| `DeepSeekParser` | DeepSeek V3 | `<\|tool▁calls▁begin\|>...<\|tool▁calls▁end\|>` |
| `Glm4MoeParser` | GLM-4.5/4.6/4.7 | `<\|observation\|>...<\|/observation\|>` |
| `Step3Parser` | Step-3 | `<steptml:function_call>...</steptml:function_call>` |
| `KimiK2Parser` | Kimi K2 | `<\|tool_call_begin\|>...<\|tool_call_end\|>` |
| `MinimaxM2Parser` | MiniMax M2 | `<FUNCTION_CALL>{...}</FUNCTION_CALL>` |
| `HyV4Parser` | Hunyuan v4 (Hy4) | `<tool_calls[:suffix]><tool_call[:suffix]>...<arg_key[:suffix]>...<arg_value[:suffix]>...` |
| `JsonParser` | OpenAI, Claude, Gemini | Direct JSON tool calls |

## Usage

### Complete Parsing

```rust
use tool_parser::{CohereParser, ToolParser};

let parser = CohereParser::new();
let input = r#"<|START_RESPONSE|>Let me search.<|END_RESPONSE|>
<|START_ACTION|>
{"tool_name": "search", "parameters": {"query": "rust"}}
<|END_ACTION|>"#;

let (normal_text, tool_calls) = parser.parse_complete(input).await?;
assert_eq!(normal_text, "Let me search.");
assert_eq!(tool_calls[0].function.name, "search");
```

### Streaming Parsing

```rust
use tool_parser::{CohereParser, ToolParser};
use openai_protocol::common::Tool;

let mut parser = CohereParser::new();
let tools: Vec<Tool> = vec![/* tool definitions */];

for chunk in stream {
    let result = parser.parse_incremental(&chunk, &tools).await?;

    // Normal text to display
    if !result.normal_text.is_empty() {
        print!("{}", result.normal_text);
    }

    // Tool call updates (name, arguments delta)
    for call in result.calls {
        // Handle streaming tool call
    }
}

// Get any remaining unstreamed arguments
if let Some(remaining) = parser.get_unstreamed_tool_args() {
    // Handle remaining tool arguments
}

// Reset for next request
parser.reset();
```

### Factory Pattern

```rust
use tool_parser::ParserFactory;

let factory = ParserFactory::new();

// Get parser by model ID (auto-detects format)
let parser = factory.get_pooled("command-r-plus");

// Or create a fresh instance for isolated streaming
let parser = factory.registry().create_for_model("mistral-large");
```

## Architecture

```
tool_parser/
├── src/
│   ├── lib.rs           # Public exports
│   ├── traits.rs        # ToolParser trait
│   ├── types.rs         # ToolCall, StreamingParseResult
│   ├── errors.rs        # ParserError
│   ├── factory.rs       # ParserFactory, ParserRegistry
│   ├── partial_json.rs  # Incomplete JSON handling
│   └── parsers/
│       ├── mod.rs       # Parser re-exports
│       ├── helpers.rs   # Shared utilities
│       ├── cohere.rs    # CohereParser
│       ├── mistral.rs   # MistralParser
│       └── ...          # Other parsers
└── tests/
    └── tool_parser_*.rs # Integration tests per parser
```

## Key Types

```rust
/// Core parsing trait
#[async_trait]
pub trait ToolParser: Send + Sync {
    /// Parse complete output
    async fn parse_complete(&self, output: &str)
        -> ParserResult<(String, Vec<ToolCall>)>;

    /// Parse streaming chunk
    async fn parse_incremental(&mut self, chunk: &str, tools: &[Tool])
        -> ParserResult<StreamingParseResult>;

    /// Check for format markers
    fn has_tool_markers(&self, text: &str) -> bool;

    /// Reset state for reuse
    fn reset(&mut self);
}

/// Parsed tool call
pub struct ToolCall {
    pub function: FunctionCall,
}

pub struct FunctionCall {
    pub name: String,
    pub arguments: String,  // JSON string
}

/// Streaming parse result
pub struct StreamingParseResult {
    pub normal_text: String,
    pub calls: Vec<PartialToolCall>,
}
```

## Adding a New Parser

1. Create `src/parsers/<model>.rs` implementing `ToolParser`
2. Add to `src/parsers/mod.rs` exports
3. Register in `src/factory.rs`:
   ```rust
   registry.register_parser("myparser", || Box::new(MyParser::new()));
   registry.map_model("my-model-*", "myparser");
   ```
4. Add to `src/lib.rs` public exports
5. Create `tests/tool_parser_<model>.rs` with test cases

## License

Apache-2.0
