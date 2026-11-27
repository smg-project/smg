use std::collections::HashMap;

use super::{McpServerConfig, McpTransport};

/// An SSE server named `name` at `url`: no token, no headers, no proxy,
/// not required, every tool, not a built-in, not internal.
pub(crate) fn sse_server(name: &str, url: &str) -> McpServerConfig {
    sse_server_with_headers(name, url, HashMap::new())
}

/// [`sse_server`] with `headers` on its transport.
pub(crate) fn sse_server_with_headers(
    name: &str,
    url: &str,
    headers: HashMap<String, String>,
) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport: McpTransport::Sse {
            url: url.to_string(),
            token: None,
            headers,
        },
        proxy: None,
        required: false,
        tools: None,
        builtin_type: None,
        builtin_tool_name: None,
        internal: false,
    }
}
