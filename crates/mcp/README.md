# smg-mcp

Model Context Protocol (MCP) client implementation with approval system for SMG.

## Overview

This crate provides:

1. **MCP Orchestration** - Unified entry point for all MCP operations
2. **Tool Inventory** - Cache and query tools with collision handling
3. **Approval System** - Dual-mode approval for tool execution
4. **Response Transformation** - Convert MCP responses to OpenAI formats

## Architecture

```
McpOrchestrator (main entry point)
├── Static Servers (from config, always connected)
├── Connection Pool (LRU, for dynamic servers)
├── Tool Inventory (qualified names, collision-aware)
├── Approval Manager (interactive + policy modes)
└── Response Transformer (MCP → OpenAI formats)

Per-Request Flow:
McpRequestContext → call_tool() → Approval → Execute → Transform
```

## Modules

### Core (`core/`)

- `McpOrchestrator` - Main entry point, coordinates all MCP operations
- `McpRequestContext` - Per-request context with tenant isolation
- `McpConnectionPool` - LRU pool for dynamic server connections
- `SmgClientHandler` - RMCP ClientHandler implementation

### Inventory (`inventory/`)

- `ToolInventory` - Multi-index cache for tools, prompts, resources
- `QualifiedToolName` - Prevents collisions (`server:tool`)
- `ToolEntry` - Tool metadata with annotations and response format

### Approval (`approval/`)

- `ApprovalManager` - Dual-mode approval coordinator
- `PolicyEngine` - Rule-based automatic decisions
- `AuditLog` - Decision logging for compliance

### Transform (`transform/`)

- `ResponseTransformer` - Converts MCP results to OpenAI formats
- `ResponseFormat` - Format specification (Passthrough, WebSearchCall, etc.)

## Configuration

### YAML Configuration File

```yaml
# mcp.yaml

# Static MCP servers (connected at startup)
servers:
  # SSE transport with tool configuration
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    token: "${BRAVE_API_KEY}"
    required: true

    # Tool-level configuration
    tools:
      brave_web_search:
        alias: web_search              # LLM sees "web_search" instead of "brave_web_search"
        response_format: web_search_call  # Transform to OpenAI web_search_call format
        arg_mapping:
          renames:
            q: query                   # Rename "q" argument to "query"
          defaults:
            count: 10                  # Default value for "count" argument

  # Stdio transport (local process)
  - name: filesystem
    protocol: stdio
    command: "npx"
    args: ["-y", "@anthropic/mcp-server-filesystem", "/tmp"]
    envs:
      NODE_ENV: production
    tools:
      search:
        response_format: file_search_call  # Transform to file_search_call format

  # Streamable HTTP transport with custom headers
  - name: custom-server
    protocol: streamable
    url: "https://my-mcp-server.com/mcp"
    token: "my-secret-token"
    headers:
      X-API-Key: "${CUSTOM_API_KEY}"
      X-Tenant-ID: "tenant-123"
    required: false

  # Built-in tool routing: route OpenAI built-in tools to MCP servers
  - name: brave-search
    protocol: sse
    url: "https://mcp.brave.com/sse"
    token: "${BRAVE_API_KEY}"
    builtin_type: web_search_preview    # Route web_search_preview to this server
    builtin_tool_name: brave_web_search # Call this MCP tool
    tools:
      brave_web_search:
        response_format: web_search_call

  - name: code-runner
    protocol: stdio
    command: "code-interpreter-server"
    builtin_type: code_interpreter      # Route code_interpreter to this server
    builtin_tool_name: execute          # Call this MCP tool
    tools:
      execute:
        response_format: code_interpreter_call

# Connection pool for dynamic servers
pool:
  max_connections: 100
  idle_timeout: 300  # seconds

# Tool inventory settings
inventory:
  enable_refresh: true
  tool_ttl: 300           # seconds
  refresh_interval: 60    # seconds
  refresh_on_error: true

# Global proxy (for MCP traffic only, not LLM API)
proxy:
  http: "http://proxy.internal:8080"
  https: "http://proxy.internal:8080"
  no_proxy: "localhost,127.0.0.1,*.internal"

# Pre-warm connections at startup
warmup:
  - url: "https://mcp.example.com/sse"
    label: "example-server"
    token: "optional-token"
```

### Transport Types

| Protocol | Use Case | Example |
|----------|----------|---------|
| `stdio` | Local MCP servers (npx, python, etc.) | Filesystem, Git, Database |
| `sse` | Remote servers with Server-Sent Events | Brave Search, hosted servers |
| `streamable` | Remote servers with HTTP streaming | Custom HTTP MCP servers |

### Loading Configuration

```rust
use smg_mcp::McpConfig;

// From YAML file
let config = McpConfig::from_file("mcp.yaml").await?;

// With environment proxy fallback
let config = config.with_env_proxy();

// Programmatic configuration
let config = McpConfig {
    servers: vec![
        McpServerConfig {
            name: "brave".to_string(),
            transport: McpTransport::Sse {
                url: "https://mcp.brave.com/sse".to_string(),
                token: Some(std::env::var("BRAVE_API_KEY")?),
                headers: HashMap::new(), // Custom headers (e.g., X-API-Key)
            },
            proxy: None,
            required: true,
            tools: None,
        },
    ],
    policy: PolicyConfig {
        default: PolicyDecisionConfig::Allow,
        servers: [("brave".to_string(), ServerPolicyConfig {
            trust_level: TrustLevelConfig::Trusted,
            ..Default::default()
        })].into_iter().collect(),
        ..Default::default()
    },
    ..Default::default()
};
```

## Tool Configuration

### Response Formats

Tools can be configured to transform MCP responses to OpenAI-compatible formats:

| Format | Output Type | Use Case |
|--------|-------------|----------|
| `passthrough` | `mcp_call` | Default, raw MCP response |
| `web_search_call` | `web_search_call` | Search results with URLs |
| `file_search_call` | `file_search_call` | File search results |
| `code_interpreter_call` | `code_interpreter_call` | Code execution results |

### Config-Based Tool Configuration (Recommended)

Configure tools directly in YAML:

```yaml
servers:
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    token: "${BRAVE_API_KEY}"

    tools:
      # Tool name as it exists on the MCP server
      brave_web_search:
        alias: web_search              # Optional: LLM sees this name
        response_format: web_search_call
        arg_mapping:
          renames:
            q: query                   # Rename arguments
          defaults:
            count: 10                  # Default values

      brave_local_search:
        response_format: web_search_call  # No alias, just format

  - name: code-runner
    protocol: stdio
    command: "code-interpreter"
    tools:
      execute:
        alias: run_code
        response_format: code_interpreter_call
```

### Programmatic Tool Configuration

Alternatively, configure tools in code:

```rust
use smg_mcp::{ResponseFormat, ArgMapping};

orchestrator.register_alias(
    "web_search",                      // alias name
    "brave",                           // target server
    "brave_web_search",                // target tool
    Some(ArgMapping::new()
        .with_rename("q", "query")
        .with_default("count", json!(10))),
    ResponseFormat::WebSearchCall,
)?;
```

### How Response Transformation Works

```
MCP Server Response (CallToolResult)
        │
        ▼
┌─────────────────────────────────┐
│ ResponseTransformer.transform() │
│                                 │
│  ResponseFormat::Passthrough    │──► mcp_call output (raw)
│  ResponseFormat::WebSearchCall  │──► web_search_call output
│  ResponseFormat::FileSearchCall │──► file_search_call output
│  ResponseFormat::CodeInterpreter│──► code_interpreter_call output
└─────────────────────────────────┘
        │
        ▼
OpenAI ResponseOutputItem
```

### Built-in Tool Routing

Route OpenAI built-in tool types (`web_search_preview`, `code_interpreter`, `file_search`) to MCP servers instead of passing them to the upstream model.

**Configuration:**

| Field | Type | Description |
|-------|------|-------------|
| `builtin_type` | `web_search_preview` \| `code_interpreter` \| `file_search` | Which built-in tool type this server handles |
| `builtin_tool_name` | string | The MCP tool to call on this server |

**Example:**

```yaml
servers:
  - name: brave-search
    protocol: sse
    url: "https://mcp.brave.com/sse"
    token: "${BRAVE_API_KEY}"
    builtin_type: web_search_preview    # Handle web_search_preview requests
    builtin_tool_name: brave_web_search # Call this tool
    tools:
      brave_web_search:
        response_format: web_search_call  # Transform to OpenAI format
```

**How it works:**

1. Request includes `{"type": "web_search_preview"}` tool
2. Gateway finds server with `builtin_type: web_search_preview`
3. Calls `builtin_tool_name` on that MCP server
4. Transforms response using configured `response_format`

**Programmatic lookup:**

```rust
use smg_mcp::BuiltinToolType;

// Find configured server for a built-in type
if let Some((server_name, tool_name, format)) =
    orchestrator.find_builtin_server(BuiltinToolType::WebSearchPreview)
{
    // Route to MCP server
    let result = orchestrator.call_tool(
        &server_name, &tool_name, args, "web-search", &request_ctx
    ).await?;
}
```

**Note:** Both `builtin_type` and `builtin_tool_name` must be set together. If only one is set, the configuration is invalid.

## Usage

### Basic Usage with Orchestrator

```rust
use smg_mcp::{McpOrchestrator, McpConfig, ApprovalMode, TenantContext};

// Create orchestrator (policy loaded from config.policy)
let config = McpConfig::from_file("mcp.yaml").await?;
let orchestrator = McpOrchestrator::new(config).await?;

// Create per-request context
let tenant_ctx = TenantContext::new("customer-123");
let request_ctx = orchestrator.create_request_context(
    "req-001",
    tenant_ctx,
    ApprovalMode::PolicyOnly,
);

// Call a tool
// server_label is the user-facing name shown in API responses
let result = orchestrator.call_tool(
    "brave",           // server_key (internal identifier)
    "web_search",      // tool_name
    json!({"query": "rust programming"}),
    "brave",           // server_label (user-facing)
    &request_ctx,
).await?;
```

### Interactive Mode (OpenAI Responses API)

```rust
use smg_mcp::{ApprovalMode, ToolCallResult};

// Determine mode based on API capability
let mode = McpOrchestrator::determine_approval_mode(supports_mcp_approval);

let request_ctx = orchestrator.create_request_context(
    "req-002",
    tenant_ctx,
    mode,
);

match orchestrator.call_tool("server", "dangerous_tool", args, "server", &request_ctx).await? {
    ToolCallResult::Success(output) => {
        // Tool executed successfully
    }
    ToolCallResult::PendingApproval(approval_request) => {
        // Send approval_request to client, wait for response
        // Then resolve:
        orchestrator.resolve_approval(
            "req-002",
            &approval_request.server_key,
            &approval_request.elicitation_id,
            true,  // approved
            None,  // reason
            &tenant_ctx,
        ).await?;

        // Continue execution
        let result = orchestrator.continue_tool_execution(
            "server", "dangerous_tool", args, &request_ctx
        ).await?;
    }
    _ => { /* other variants */ }
}
```

### Tool Aliases

```rust
use smg_mcp::{ResponseFormat, ArgMapping};

// Register an alias with argument mapping
orchestrator.register_alias(
    "search",                    // alias name
    "brave",                     // target server
    "brave_web_search",          // target tool
    Some(ArgMapping::new()
        .with_rename("q", "query")
        .with_default("count", json!(10))),
    ResponseFormat::WebSearchCall,
)?;

// In router code, build a session and execute through session mapping.
// The model-visible tool name is resolved to {server_key, tool_name} per request.
```

### Batch Tool Execution

For executing multiple tools efficiently (e.g., parallel tool calls from LLM):

```rust
use smg_mcp::{McpServerBinding, McpToolSession, ToolExecutionInput, ToolExecutionOutput};

// Convert tool calls to inputs
let inputs: Vec<ToolExecutionInput> = tool_calls
    .iter()
    .map(|tc| ToolExecutionInput {
        call_id: tc.id.clone(),
        tool_name: tc.function.name.clone(),
        arguments: serde_json::from_str(&tc.arguments).unwrap_or(json!({})),
    })
    .collect();

let session = McpToolSession::new(
    &orchestrator,
    vec![McpServerBinding {
        label: "my-server".to_string(),
        server_key: "my-server".to_string(),
        allowed_tools: None,
        list_error: None,
        pool_key: None,
    }],
    "req-123",
);

// Execute all tools through session mapping
let outputs: Vec<ToolExecutionOutput> = session.execute_tools(inputs).await;

// Process results
for output in outputs {
    // Get transformed ResponseOutputItem (uses server_label for API response)
    let item = output.to_response_item();

    // Access raw output for conversation history
    let raw_output = &output.output;

    // Check success/error status
    if output.is_error {
        eprintln!("Tool {} failed: {:?}", output.tool_name, output.error_message);
    }
}
```

### Allowed Tools Filtering

To enforce a per-server tool allowlist, set `allowed_tools` on the `McpServerBinding`.
Each router extracts the allowlist from its protocol-specific request type and populates
the binding — the MCP session layer is protocol-agnostic.

```rust
use smg_mcp::{McpServerBinding, McpToolSession};

// No filtering (all tools exposed)
let binding = McpServerBinding {
    label: "my-server".to_string(),
    server_key: "my-server".to_string(),
    allowed_tools: None,
    list_error: None,
    pool_key: None,
};

// Only expose specific tools from this server
let binding = McpServerBinding {
    label: "my-server".to_string(),
    server_key: "my-server".to_string(),
    allowed_tools: Some(vec!["tool_a".to_string(), "tool_b".to_string()]),
    list_error: None,
    pool_key: None,
};

let session = McpToolSession::new(&orchestrator, vec![binding], "req-123");
```

`list_error` is the connect's to fill: why the server's tools could not be listed when the
binding was made, `None` when they were. The `mcp_list_tools` item builders read it as the
item's `error`; a binding built by hand, as above, names it `None`.

**Key types:**
- `ToolExecutionInput`: Tool call ID, name, and arguments
- `ToolExecutionOutput`: Full result including raw output, transformed item, duration, and error info
  - `tool_name`: Name used in transformed output items
- `server_label`: User-facing label for API responses (distinct from internal `server_key`)

Router code should use `McpToolSession` so exposed tool names are collision-safe and execution always resolves through session mapping (exposed name -> `{server_key, tool_name}`).

## Approval System

### Modes

| Mode | Use Case | Behavior |
|------|----------|----------|
| `PolicyOnly` | Batch processing, Chat API | Auto-decide via PolicyEngine |
| `Interactive` | Responses API | Return approval request to client |

On a streamed Responses request through the OpenAI-compatible HTTP router, a tool that requires approval is not run and the stream carries no approval request; a client that wants the approval request sends the request non-streamed.

### Trust Levels

| Level | Behavior |
|-------|----------|
| `trusted` | Allow all tools unconditionally |
| `standard` | Use default policy (default) |
| `untrusted` | Deny destructive operations |
| `sandboxed` | Only allow read-only, no external access |

### Policy Configuration (YAML)

Policy is configured in `mcp.yaml` under the `policy` section:

```yaml
# mcp.yaml
servers:
  - name: brave
    protocol: sse
    url: "https://mcp.brave.com/sse"
    token: "${BRAVE_API_KEY}"

  - name: internal-tools
    protocol: stdio
    command: "internal-mcp"

  - name: external-api
    protocol: sse
    url: "https://untrusted.example.com/sse"

# Approval policy configuration
policy:
  # Default decision when no other rules match (default: allow)
  default: allow

  # Per-server policies with trust levels
  servers:
    brave:
      trust_level: trusted      # Allow all brave tools
    internal-tools:
      trust_level: standard
      default: allow
    external-api:
      trust_level: untrusted    # Deny destructive operations
      default: deny

  # Explicit per-tool policies (qualified name: "server:tool")
  tools:
    "internal-tools:delete_all": deny
    "external-api:execute_code":
      deny_with_reason: "Code execution not allowed on external servers"
```

### Policy Evaluation Order

1. **Explicit tool policy** → `policy.tools["server:tool"]`
2. **Server policy + trust level** → `policy.servers["server"]`
3. **Default policy** → `policy.default`

### Policy Decisions

| Decision | YAML Syntax | Description |
|----------|-------------|-------------|
| Allow | `allow` | Permit tool execution |
| Deny | `deny` | Block tool execution |
| Deny with reason | `deny_with_reason: "message"` | Block with explanation |

### Default Behavior

If no policy is configured, **all tools are allowed** by default. This is equivalent to:

```yaml
policy:
  default: allow
  servers: {}
  tools: {}
```

## File Structure

```
mcp/src/
├── lib.rs              # Public exports
├── error.rs            # McpError, ApprovalError
├── annotations.rs      # ToolAnnotations
├── tenant.rs           # TenantContext, TenantId
│
├── core/
│   ├── orchestrator.rs # McpOrchestrator, McpRequestContext
│   ├── config.rs       # McpConfig, McpServerConfig
│   ├── pool.rs         # McpConnectionPool (LRU)
│   ├── handler.rs      # SmgClientHandler
│   ├── metrics.rs      # McpMetrics
│   ├── proxy.rs        # HTTP proxy resolution
│   └── oauth.rs        # OAuth token refresh
│
├── inventory/
│   ├── index.rs        # ToolInventory
│   ├── types.rs        # QualifiedToolName, ToolEntry
│   └── args.rs         # Argument utilities
│
├── approval/
│   ├── manager.rs      # ApprovalManager
│   ├── policy.rs       # PolicyEngine
│   └── audit.rs        # AuditLog
│
└── transform/
    ├── mod.rs          # ResponseFormat enum
    └── transformer.rs  # ResponseTransformer
```

## Design Decisions

### Why McpOrchestrator?

`McpOrchestrator` provides a unified API that coordinates:
- Server connections (static + dynamic)
- Tool inventory with qualified names
- Approval system integration
- Response transformation
- Metrics collection

`McpOrchestrator` is the main entry point for all MCP integrations.

### Why Qualified Tool Names?

Multiple MCP servers can expose tools with the same name. `QualifiedToolName` stores both: `server-a:run_query`, `server-b:run_query`.

### Why Dual-Mode Approval?

- **Interactive**: OpenAI Responses API supports `mcp_approval_request`/`mcp_approval_response`
- **PolicyOnly**: Chat Completions API, Anthropic Messages API, batch processing

### Why Response Transformation?

MCP returns `CallToolResult` with content arrays. OpenAI expects `ResponseOutputItem`. The transformer bridges this gap with format-specific handling (web search, file search, etc.).

### Why Auth-Aware Connection Pooling?

Connections are keyed by `PoolKey(url, auth_hash, tenant_id)` instead of just URL. This ensures:
- Different auth credentials get different connections (security isolation)
- Different tenants are isolated (multi-tenancy support)
- Credentials are hashed, not stored as plaintext in pool keys

### Server Key vs Server Label

Two distinct identifiers are used for servers:

| Identifier | Purpose | Example |
|------------|---------|---------|
| `server_key` | Internal identifier for connection lookup | URL like `https://mcp.example.com/sse` |
| `server_label` | User-facing label in API responses | `"brave"` or `"my-search-server"` |

For static servers configured in YAML, both are typically the server name. For dynamic servers (connected at runtime via URL), `server_key` is the URL while `server_label` comes from the request's MCP tool configuration.

API responses always use `server_label` so clients see consistent, user-friendly names regardless of the underlying connection mechanism.
