use proto::generate_response::Response;
use serde_json::json;

use super::*;

#[test]
fn a_chunk_maps_its_ids_and_every_count() {
    let response = parse_proto_response(&json!({
        "request_id": "r-1",
        "chunk": {
            "token_ids": [1, 2, 3],
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "cached_tokens": 6,
            "reasoning_tokens": 7
        }
    }))
    .unwrap();
    assert_eq!(response.request_id, "r-1");
    let Some(Response::Chunk(chunk)) = response.response else {
        panic!("a chunk response");
    };
    assert_eq!(chunk.token_ids, vec![1, 2, 3]);
    assert_eq!(
        (
            chunk.prompt_tokens,
            chunk.completion_tokens,
            chunk.cached_tokens,
            chunk.reasoning_tokens
        ),
        (4, 5, 6, 7)
    );
    assert!(!is_terminal_response(&json!({ "chunk": {} })));
}

#[test]
fn a_completion_maps_its_ids_reason_and_counts_and_is_terminal() {
    let body = json!({
        "request_id": "r-2",
        "complete": {
            "output_ids": [10, 11],
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "cached_tokens": 3,
            "reasoning_tokens": 4
        }
    });
    assert!(is_terminal_response(&body));
    let response = parse_proto_response(&body).unwrap();
    let Some(Response::Complete(complete)) = response.response else {
        panic!("a completion response");
    };
    assert_eq!(complete.output_ids, vec![10, 11]);
    assert_eq!(complete.finish_reason, "stop");
    assert_eq!(
        (
            complete.prompt_tokens,
            complete.completion_tokens,
            complete.cached_tokens,
            complete.reasoning_tokens
        ),
        (1, 2, 3, 4)
    );
}

#[test]
fn missing_fields_default_and_a_response_with_neither_part_is_refused() {
    let response = parse_proto_response(&json!({ "complete": {} })).unwrap();
    assert_eq!(response.request_id, "");
    let Some(Response::Complete(complete)) = response.response else {
        panic!("a completion response");
    };
    assert!(complete.output_ids.is_empty());
    assert_eq!(complete.finish_reason, "");
    assert_eq!((complete.prompt_tokens, complete.reasoning_tokens), (0, 0));

    assert_eq!(
        parse_proto_response(&json!({ "request_id": "r-3" })),
        Err("Response JSON must contain 'chunk' or 'complete' field")
    );
}

/// The lenient reads: an id entry that is not a number (a string, a
/// negative) is skipped, and a count that is not a number reads as 0.
#[test]
fn entries_and_counts_that_are_not_numbers_are_skipped_or_zero() {
    let response = parse_proto_response(&json!({
        "chunk": {
            "token_ids": [1, "x", -2, 3],
            "prompt_tokens": "4",
            "completion_tokens": -1,
            "cached_tokens": 2.5,
            "reasoning_tokens": 6
        }
    }))
    .unwrap();
    let Some(Response::Chunk(chunk)) = response.response else {
        panic!("a chunk response");
    };
    assert_eq!(chunk.token_ids, vec![1, 3]);
    assert_eq!(
        (
            chunk.prompt_tokens,
            chunk.completion_tokens,
            chunk.cached_tokens,
            chunk.reasoning_tokens
        ),
        (0, 0, 0, 6)
    );
}

#[test]
fn a_string_of_the_wrong_type_reads_as_empty() {
    let response = parse_proto_response(&json!({
        "request_id": 7,
        "complete": {"finish_reason": 5},
    }))
    .unwrap();

    assert_eq!(response.request_id, "");
    let Some(Response::Complete(complete)) = response.response else {
        panic!("a completion response");
    };
    assert_eq!(complete.finish_reason, "");
}
