use super::*;

fn eligible(content_length: u64) -> BodyPathInputs {
    BodyPathInputs {
        routing_key_override: false,
        policy_needs_text: false,
        model_ambiguous: false,
        worker_mutates_body: false,
        wasm_request_hooks: false,
        content_length: Some(content_length),
        retries_enabled: true,
        max_buffered_request_bytes: 1024,
        tenant_rate_limited: false,
        message_thread: false,
    }
}

#[test]
fn hard_buffer_reasons_apply_in_first_match_order() {
    let all = BodyPathInputs {
        routing_key_override: true,
        policy_needs_text: true,
        model_ambiguous: true,
        worker_mutates_body: true,
        wasm_request_hooks: true,
        tenant_rate_limited: true,
        content_length: None,
        ..eligible(0)
    };
    assert_eq!(
        decide_body_path(&all),
        BodyPath::Buffer(REASON_ROUTING_KEY_OVERRIDE)
    );

    for (inputs, reason) in [
        (
            BodyPathInputs {
                routing_key_override: true,
                ..eligible(64)
            },
            REASON_ROUTING_KEY_OVERRIDE,
        ),
        (
            BodyPathInputs {
                policy_needs_text: true,
                ..eligible(64)
            },
            REASON_POLICY_NEEDS_TEXT,
        ),
        (
            BodyPathInputs {
                model_ambiguous: true,
                ..eligible(64)
            },
            REASON_MODEL_AMBIGUOUS,
        ),
        (
            BodyPathInputs {
                worker_mutates_body: true,
                ..eligible(64)
            },
            REASON_WORKER_MUTATES_BODY,
        ),
        (
            BodyPathInputs {
                wasm_request_hooks: true,
                ..eligible(64)
            },
            REASON_WASM_REQUEST_HOOK,
        ),
        (
            BodyPathInputs {
                tenant_rate_limited: true,
                ..eligible(64)
            },
            REASON_TENANT_RATE_LIMIT,
        ),
        (
            BodyPathInputs {
                message_thread: true,
                ..eligible(64)
            },
            REASON_MESSAGE_THREAD,
        ),
        (
            BodyPathInputs {
                content_length: None,
                ..eligible(64)
            },
            REASON_NO_CONTENT_LENGTH,
        ),
    ] {
        assert_eq!(decide_body_path(&inputs), BodyPath::Buffer(reason));
    }
}

/// A message-thread turn buffers under every retry setting and at any
/// size: the thread layer reads the body before the route.
#[test]
fn a_message_thread_turn_buffers_whatever_the_retry_setting() {
    for retries_enabled in [true, false] {
        let inputs = BodyPathInputs {
            message_thread: true,
            retries_enabled,
            ..eligible(u64::MAX)
        };
        assert_eq!(
            decide_body_path(&inputs),
            BodyPath::Buffer(REASON_MESSAGE_THREAD)
        );
    }
}

#[test]
fn retry_buffer_bound_crosses_over_exactly_at_the_cap() {
    assert_eq!(
        decide_body_path(&eligible(1024)),
        BodyPath::Buffer(REASON_RETRYABLE)
    );
    assert_eq!(
        decide_body_path(&eligible(1025)),
        BodyPath::Stream(REASON_RETRY_FORFEITED)
    );
}

#[test]
fn zero_bound_never_buffers_for_retries() {
    let inputs = BodyPathInputs {
        max_buffered_request_bytes: 0,
        ..eligible(1)
    };
    assert_eq!(
        decide_body_path(&inputs),
        BodyPath::Stream(REASON_RETRY_FORFEITED)
    );
}

#[test]
fn pure_forward_streams_at_any_size() {
    for len in [1, 1024, u64::MAX] {
        let inputs = BodyPathInputs {
            retries_enabled: false,
            ..eligible(len)
        };
        assert_eq!(
            decide_body_path(&inputs),
            BodyPath::Stream(REASON_PURE_FORWARD)
        );
    }
}
