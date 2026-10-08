# reasoning-parser

A Rust library for detecting and extracting reasoning content (chain-of-thought) from Large Language Model outputs. Handles models that emit explicit thinking blocks delimited by tokens like `<think>` and `</think>`.

## Features

- **Unified Interface** - Single API for multiple model formats
- **Streaming Support** - Incremental parsing with state preservation across chunks
- **Parser Pooling** - Efficient reuse of parser instances for high concurrency
- **Partial Token Handling** - Correctly handles tokens split across chunk boundaries
- **Model Auto-Detection** - Pattern-based automatic parser selection
- **Extensible** - Easy to add support for new model formats

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
reasoning-parser = "1.0"
```

## Quick Start

```rust
use reasoning_parser::{ParserFactory, ReasoningParser};

#[tokio::main]
async fn main() {
    let factory = ParserFactory::new();
    let parser = factory.get_pooled("deepseek-r1");

    let mut p = parser.lock().await;
    let result = p
        .detect_and_parse_reasoning("<think>Let me analyze this...</think>The answer is 42.")
        .unwrap();

    println!("Reasoning: {}", result.reasoning_text);  // "Let me analyze this..."
    println!("Answer: {}", result.normal_text);        // "The answer is 42."
}
```

## Supported Models

| Model | Token Format | Notes |
|-------|--------------|-------|
| DeepSeek-R1 | `<think>`/`</think>` | Starts in reasoning mode |
| Qwen3 | `<think>`/`</think>` | Explicit reasoning blocks |
| Qwen3-Thinking | `<think>`/`</think>` | Starts in reasoning mode |
| GLM-4.5/4.6/4.7 | `<think>`/`</think>` | Explicit reasoning blocks |
| Kimi | `◁think▷`/`◁/think▷` | Unicode delimiters |
| Step3 | `<think>`/`</think>` | Starts in reasoning mode |
| MiniMax M2 | `<think>`/`</think>` | Auto-prepends start token |
| Cohere Command | `<\|START_THINKING\|>`/`<\|END_THINKING\|>` | CMD3/CMD4 format |
| Nemotron-Nano | `<think>`/`</think>` | Qwen3-compatible |
| Hy4 | `<think>`/`</think>` with optional checkpoint suffix | `reasoning_parser: hy_v4` |

Unknown models fall back to a passthrough parser that returns all text as normal output.

## Core Types

### ParserResult

The result of parsing, separating reasoning from normal text:

```rust
pub struct ParserResult {
    pub normal_text: String,    // Text outside reasoning blocks
    pub reasoning_text: String, // Text inside reasoning blocks
}
```

### ReasoningParser Trait

The core interface all parsers implement:

```rust
pub trait ReasoningParser: Send + Sync {
    /// One-shot parsing of complete text
    fn detect_and_parse_reasoning(&mut self, text: &str) -> Result<ParserResult, ParseError>;

    /// Streaming incremental parsing
    fn parse_reasoning_streaming_incremental(&mut self, text: &str) -> Result<ParserResult, ParseError>;

    /// Reset parser state for reuse
    fn reset(&mut self);

    /// Get parser variant identifier
    fn model_type(&self) -> &str;

    /// Check if currently parsing reasoning content
    fn is_in_reasoning(&self) -> bool;
}
```

## Usage Patterns

### One-Shot Parsing

For complete text that doesn't need streaming:

```rust
let factory = ParserFactory::new();
let mut parser = factory.create("qwen3").unwrap();

let input = "<think>Step 1: Consider the problem...</think>The solution is X.";
let result = parser.detect_and_parse_reasoning(input).unwrap();

assert_eq!(result.reasoning_text, "Step 1: Consider the problem...");
assert_eq!(result.normal_text, "The solution is X.");
```

### Streaming Parsing

For processing chunks as they arrive from an LLM:

```rust
let factory = ParserFactory::new();
let parser = factory.get_pooled("deepseek-r1");

let chunks = vec![
    "<think>Let me ",
    "think about this",
    "</think>Here's ",
    "the answer.",
];

let mut p = parser.lock().await;
for chunk in chunks {
    let result = p.parse_reasoning_streaming_incremental(chunk).unwrap();

    if !result.reasoning_text.is_empty() {
        print!("[reasoning] {}", result.reasoning_text);
    }
    if !result.normal_text.is_empty() {
        print!("{}", result.normal_text);
    }
}
```

### Parser Reuse

Reset a parser to process a new request:

```rust
let parser = factory.get_pooled("qwen3");
let mut p = parser.lock().await;

// First request
let result1 = p.detect_and_parse_reasoning("<think>A</think>B").unwrap();

// Reset for next request
p.reset();

// Second request
let result2 = p.detect_and_parse_reasoning("<think>C</think>D").unwrap();
```

### Pooled vs Fresh Parsers

```rust
// Pooled: shared instance, requires lock, efficient for high concurrency
let pooled = factory.get_pooled("deepseek-r1");  // Arc<Mutex<Box<dyn ReasoningParser>>>

// Fresh: new instance each time, no lock needed
let fresh = factory.create("deepseek-r1").unwrap();  // Box<dyn ReasoningParser>
```

## Custom Parser Configuration

Create a parser with custom tokens:

```rust
use reasoning_parser::{BaseReasoningParser, ParserConfig, ReasoningParser};

let config = ParserConfig {
    think_start_token: "<reasoning>".to_string(),
    think_end_token: "</reasoning>".to_string(),
    stream_reasoning: true,
    max_buffer_size: 65536,
    initial_in_reasoning: false,
};

let mut parser = BaseReasoningParser::new(config);
let result = parser
    .detect_and_parse_reasoning("<reasoning>thinking</reasoning>answer")
    .unwrap();
```

## Registering Custom Parsers

Add support for new model patterns:

```rust
let factory = ParserFactory::new();

// Register a creator function
factory.registry().register_parser("myformat", || {
    Box::new(BaseReasoningParser::new(ParserConfig {
        think_start_token: "<<THINK>>".to_string(),
        think_end_token: "<</THINK>>".to_string(),
        stream_reasoning: true,
        max_buffer_size: 65536,
        initial_in_reasoning: false,
    }))
});

// Map model patterns to the parser
factory.registry().register_pattern("my-custom-model", "myformat");
factory.registry().register_pattern("my-model-v2", "myformat");

// Now these work
let parser = factory.get_pooled("my-custom-model-7b");
```

## Error Handling

```rust
use reasoning_parser::ParseError;

match parser.detect_and_parse_reasoning(text) {
    Ok(result) => {
        println!("Reasoning: {}", result.reasoning_text);
        println!("Normal: {}", result.normal_text);
    }
    Err(ParseError::BufferOverflow(size)) => {
        eprintln!("Content too large: {} bytes", size);
    }
    Err(ParseError::Utf8Error(e)) => {
        eprintln!("Invalid UTF-8: {}", e);
    }
    Err(ParseError::UnknownModel(model)) => {
        eprintln!("Unknown model: {}", model);
    }
    Err(ParseError::ConfigError(msg)) => {
        eprintln!("Configuration error: {}", msg);
    }
}
```

## Model Pattern Matching

The factory uses case-insensitive substring matching:

```rust
// All of these match "deepseek-r1" pattern:
factory.get_pooled("deepseek-r1");
factory.get_pooled("DeepSeek-R1-Distill-Qwen-7B");
factory.get_pooled("my-deepseek-r1-finetune");
```

Pattern priority (first match wins):
1. `deepseek-r1` → DeepSeekR1Parser
2. `qwen3-thinking` / `qwen-thinking` → QwenThinkingParser
3. `qwen3` / `qwen` → Qwen3Parser
4. `glm45` / `glm46` / `glm47` → Glm45Parser
5. `kimi` → KimiParser
6. `step3` → Step3Parser
7. `minimax` / `mm-m2` → MiniMaxParser
8. `command-r` / `command-a` / `c4ai-command` / `cohere` → CohereCmdParser
9. `nemotron-nano` / `nano-v3` → Qwen3Parser
10. (fallback) → BaseReasoningParser (passthrough)

## Thread Safety

The crate is designed for high-concurrency scenarios:

- `PooledParser` type is `Arc<Mutex<Box<dyn ReasoningParser>>>`
- Uses `tokio::Mutex` for async-friendly locking
- Registry uses `Arc<RwLock<>>` for safe concurrent access
- Tested with 100 concurrent tasks at 1000+ requests/second

## License

Apache-2.0
