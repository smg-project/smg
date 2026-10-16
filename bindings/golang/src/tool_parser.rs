//! Tool parser FFI functions

use std::{collections::HashMap, ffi::CStr, os::raw::c_char, ptr, sync::Arc};

use openai_protocol::common::Tool;
use serde_json::{json, Value};
use tool_parser::ToolParser;

use super::{
    args::write_json_result,
    error::{set_error_message, SglErrorCode},
    runtime::{PARSER_FACTORY, RUNTIME},
    utils::generate_tool_call_id,
};

/// Opaque handle for a tool parser instance
/// Note: For streaming, we need mutable access, so we use Arc<Mutex<>> internally
/// Note: This is an opaque handle, C code doesn't access fields directly
pub struct ToolParserHandle {
    parser: Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>,
    model: String,                            // Store model name for ID generation
    history_tool_calls_count: usize,          // Track tool call count for ID generation
    tool_index_to_id: HashMap<usize, String>, // Map tool_index to ID for incremental updates
}

/// Create a tool parser
///
/// # Arguments
/// * `parser_type` - Parser type name (e.g., "json", "llama", "mistral") or model name (e.g., "gpt-4")
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to ToolParserHandle on success, null on failure
///
/// # Safety
/// - `parser_type` must be a valid null-terminated C string
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller owns the returned handle and must free it with `sgl_tool_parser_free`
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_create(
    parser_type: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut ToolParserHandle {
    if parser_type.is_null() {
        set_error_message(error_out, "parser_type cannot be null");
        return ptr::null_mut();
    }

    let type_str = match CStr::from_ptr(parser_type).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in parser_type");
            return ptr::null_mut();
        }
    };

    // Create parser using factory
    // The factory will determine the parser type based on model name or use the provided type
    let parser = if let Some(parser_box) = PARSER_FACTORY.registry().create_for_model(type_str) {
        parser_box
    } else if let Some(parser_box) = PARSER_FACTORY.registry().create_parser(type_str) {
        parser_box
    } else {
        set_error_message(error_out, &format!("Unknown parser type: {type_str}"));
        return ptr::null_mut();
    };

    Box::into_raw(Box::new(ToolParserHandle {
        parser: Arc::new(tokio::sync::Mutex::new(parser)),
        model: type_str.to_string(),
        history_tool_calls_count: 0,
        tool_index_to_id: HashMap::new(),
    }))
}

/// Parse complete tool calls from text
///
/// # Arguments
/// * `handle` - Tool parser handle
/// * `text` - Input text to parse
/// * `result_json_out` - Pointer to receive JSON result (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_tool_parser_create`
/// - `text` must be a valid null-terminated C string
/// - `result_json_out` must be a valid pointer to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free the string written to `result_json_out` using `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_parse_complete(
    handle: *mut ToolParserHandle,
    text: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || text.is_null() || result_json_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in text");
            return SglErrorCode::InvalidArgument;
        }
    };

    let handle_ref = &*handle;
    let parser = Arc::clone(&handle_ref.parser);
    let model = handle_ref.model.clone();
    let history_count = handle_ref.history_tool_calls_count;

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        let parser_guard = parser.lock().await;
        parser_guard.parse_complete(text_str).await
    });

    match result {
        Ok((normal_text, tool_calls)) => {
            // Convert Rust ToolCall to OpenAI format
            let openai_tool_calls: Vec<Value> = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tc)| {
                    // Generate ID for this tool call
                    let id = generate_tool_call_id(&model, &tc.function.name, index, history_count);
                    json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                })
                .collect();

            // Build result JSON
            let result_json = json!({
                "normal_text": normal_text,
                "tool_calls": openai_tool_calls
            });

            match write_json_result(&result_json, result_json_out, error_out) {
                Ok(()) => SglErrorCode::Success,
                Err(code) => code,
            }
        }
        Err(e) => {
            set_error_message(error_out, &format!("Parse error: {e}"));
            SglErrorCode::ParsingError
        }
    }
}

/// Parse tool calls incrementally from streaming chunks
///
/// # Arguments
/// * `handle` - Tool parser handle
/// * `chunk` - New text chunk from stream
/// * `tools_json` - JSON array of available tools (for validation, can be null/empty)
/// * `result_json_out` - Pointer to receive JSON result (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_tool_parser_create`
/// - `chunk` must be a valid null-terminated C string
/// - `tools_json` may be null; if non-null, must be a valid null-terminated C string
/// - `result_json_out` must be a valid pointer to writable memory
/// - `error_out` may be null; if non-null, must point to writable memory
/// - Caller must free the string written to `result_json_out` using `sgl_free_string`
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_parse_incremental(
    handle: *mut ToolParserHandle,
    chunk: *const c_char,
    tools_json: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || chunk.is_null() || result_json_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let chunk_str = match CStr::from_ptr(chunk).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in chunk");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse tools JSON if provided
    let tools: Vec<Tool> = if tools_json.is_null() {
        vec![]
    } else {
        let tools_str = match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_error_message(error_out, "Invalid UTF-8 in tools_json");
                return SglErrorCode::InvalidArgument;
            }
        };
        serde_json::from_str::<Vec<Tool>>(tools_str).unwrap_or_default()
    };

    let handle_ref = &*handle;
    let parser = Arc::clone(&handle_ref.parser);
    let model = handle_ref.model.clone();
    let history_count = handle_ref.history_tool_calls_count;

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        let mut parser_guard = parser.lock().await;
        parser_guard.parse_incremental(chunk_str, &tools).await
    });

    match result {
        Ok(streaming_result) => {
            // Convert StreamingParseResult to OpenAI format
            let handle_mut = &mut *handle;
            let openai_tool_calls: Vec<Value> = streaming_result
                .calls
                .into_iter()
                .map(|item| {
                    // For incremental parsing, we may not have complete tool calls yet
                    // Generate or reuse ID based on tool_index
                    let id = if let Some(ref name) = item.name {
                        // New tool call with name - generate ID and store it
                        let id =
                            generate_tool_call_id(&model, name, item.tool_index, history_count);
                        handle_mut
                            .tool_index_to_id
                            .insert(item.tool_index, id.clone());
                        id
                    } else {
                        // Parameter update - reuse existing ID for this tool_index
                        handle_mut
                            .tool_index_to_id
                            .get(&item.tool_index)
                            .cloned()
                            .unwrap_or_else(|| format!("call_{}", item.tool_index))
                    };

                    json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": item.name.unwrap_or_default(),
                            "arguments": item.parameters
                        }
                    })
                })
                .collect();

            // Build result JSON
            let result_json = json!({
                "normal_text": streaming_result.normal_text,
                "tool_calls": openai_tool_calls
            });

            match write_json_result(&result_json, result_json_out, error_out) {
                Ok(()) => SglErrorCode::Success,
                Err(code) => code,
            }
        }
        Err(e) => {
            set_error_message(error_out, &format!("Parse incremental error: {e}"));
            SglErrorCode::ParsingError
        }
    }
}

/// Reset the parser state for reuse
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_tool_parser_create`, or null
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_reset(handle: *mut ToolParserHandle) {
    if handle.is_null() {
        return;
    }

    let handle_ref = &mut *handle;
    let parser = Arc::clone(&handle_ref.parser);

    // Reset parser state
    RUNTIME.block_on(async {
        let mut parser_guard = parser.lock().await;
        parser_guard.reset();
    });

    // Reset history count and tool index mapping
    handle_ref.history_tool_calls_count = 0;
    handle_ref.tool_index_to_id.clear();
}

/// Free a tool parser handle
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sgl_tool_parser_create`, or null
/// - `handle` must not be used after this call
/// - This function must not be called more than once for the same handle
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_free(handle: *mut ToolParserHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}
