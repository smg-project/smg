//! Shared proto JSON parsing utilities

use serde_json::Value;
use smg_grpc_client::sglang_proto as proto;

/// Parse a JSON value into a proto::GenerateResponse
pub fn parse_proto_response(json_value: &Value) -> Result<proto::GenerateResponse, &'static str> {
    let mut proto_response = proto::GenerateResponse {
        media_identity: None,
        prepared_input: Vec::new(),
        request_id: string_at(json_value, "request_id"),
        response: None,
    };

    if let Some(chunk_json) = json_value.get("chunk") {
        proto_response.response = Some(proto::generate_response::Response::Chunk(parse_chunk(
            chunk_json,
        )));
    } else if let Some(complete_json) = json_value.get("complete") {
        proto_response.response = Some(proto::generate_response::Response::Complete(
            parse_complete(complete_json),
        ));
    } else {
        return Err("Response JSON must contain 'chunk' or 'complete' field");
    }

    Ok(proto_response)
}

/// Check if a proto response is terminal (complete)
pub fn is_terminal_response(json_value: &Value) -> bool {
    json_value.get("complete").is_some()
}

/// The string at `key`; empty when absent or not a string.
fn string_at(json: &Value, key: &str) -> String {
    json.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

/// The count at `key`; zero when absent or not a non-negative integer (a
/// negative number and a float read as absent). The cast keeps the low 32
/// bits of a value beyond `u32`.
fn u32_at(json: &Value, key: &str) -> u32 {
    json.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as u32)
        .unwrap_or(0)
}

/// The token ids at `key`; empty when absent or not an array, a member that
/// is not a non-negative integer skipped.
fn ids_at(json: &Value, key: &str) -> Vec<u32> {
    json.get(key)
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(|id| id.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_chunk(json: &Value) -> proto::GenerateStreamChunk {
    proto::GenerateStreamChunk {
        token_ids: ids_at(json, "token_ids"),
        prompt_tokens: u32_at(json, "prompt_tokens"),
        completion_tokens: u32_at(json, "completion_tokens"),
        cached_tokens: u32_at(json, "cached_tokens"),
        reasoning_tokens: u32_at(json, "reasoning_tokens"),
        // The Go bindings only surface token data over FFI; everything else
        // defaults. Using `..Default::default()` keeps new proto fields from
        // breaking this builder until they are explicitly wired.
        ..Default::default()
    }
}

fn parse_complete(json: &Value) -> proto::GenerateComplete {
    proto::GenerateComplete {
        output_ids: ids_at(json, "output_ids"),
        finish_reason: string_at(json, "finish_reason"),
        prompt_tokens: u32_at(json, "prompt_tokens"),
        completion_tokens: u32_at(json, "completion_tokens"),
        cached_tokens: u32_at(json, "cached_tokens"),
        reasoning_tokens: u32_at(json, "reasoning_tokens"),
        ..Default::default()
    }
}

#[cfg(test)]
#[path = "../tests/unit/proto_parse/tests.rs"]
mod tests;
