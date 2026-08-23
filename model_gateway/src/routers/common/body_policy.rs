//! Request body-path capability shared by every router family: the
//! [`BodyPolicy`] a family declares, the per-request decision matrix a
//! forward-capable router runs, and the counter vocabulary both feed.

/// How a router family treats request bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPolicy {
    /// May forward a raw body verbatim; the per-request buffer-vs-stream
    /// decision applies.
    ForwardCapable,
    /// Must parse every body; the reason feeds the body-path counter.
    MustBuffer(&'static str),
}

/// Per-request body-path decision with its dominant (first-match) reason.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BodyPath {
    Buffer(&'static str),
    Stream(&'static str),
}

pub(crate) const BODY_PATH_STREAMED: &str = "streamed";
pub(crate) const BODY_PATH_BUFFERED: &str = "buffered";

pub(crate) const REASON_POLICY_NEEDS_TEXT: &str = "policy_needs_text";
pub(crate) const REASON_MODEL_AMBIGUOUS: &str = "model_ambiguous";
pub(crate) const REASON_WORKER_MUTATES_BODY: &str = "worker_mutates_body";
pub(crate) const REASON_WASM_REQUEST_HOOK: &str = "wasm_request_hook";
pub(crate) const REASON_NO_CONTENT_LENGTH: &str = "no_content_length";
pub(crate) const REASON_NO_AVAILABLE_WORKER: &str = "no_available_worker";
pub(crate) const REASON_MODEL_SELECTION: &str = "model_selection";
pub(crate) const REASON_RETRYABLE: &str = "retryable";
pub(crate) const REASON_RETRY_FORFEITED: &str = "retry_forfeited";
pub(crate) const REASON_PURE_FORWARD: &str = "pure_forward";

/// Per-request inputs, gathered by the caller before the body arrives.
pub(crate) struct BodyPathInputs {
    pub policy_needs_text: bool,
    pub model_ambiguous: bool,
    pub worker_mutates_body: bool,
    pub wasm_request_hooks: bool,
    pub content_length: Option<u64>,
    pub retries_enabled: bool,
    pub max_buffered_request_bytes: u64,
}

/// Decide the request body path before the body arrives. Buffer when
/// something must read the body here: a text-routing policy without a valid
/// hint-header waiver, a registry serving more than one model (content-blind
/// selection could land the request on the wrong model's worker), a
/// body-mutating worker, a WASM request hook, or a missing/invalid
/// Content-Length. Otherwise, with router retries enabled, buffer up to
/// `max_buffered_request_bytes` to keep the request replayable — a larger
/// body streams and forfeits router retries. Otherwise stream, at any size.
pub(crate) fn decide_body_path(inputs: &BodyPathInputs) -> BodyPath {
    if inputs.policy_needs_text {
        return BodyPath::Buffer(REASON_POLICY_NEEDS_TEXT);
    }
    if inputs.model_ambiguous {
        return BodyPath::Buffer(REASON_MODEL_AMBIGUOUS);
    }
    if inputs.worker_mutates_body {
        return BodyPath::Buffer(REASON_WORKER_MUTATES_BODY);
    }
    if inputs.wasm_request_hooks {
        return BodyPath::Buffer(REASON_WASM_REQUEST_HOOK);
    }
    let Some(content_length) = inputs.content_length else {
        return BodyPath::Buffer(REASON_NO_CONTENT_LENGTH);
    };
    if inputs.retries_enabled {
        if content_length <= inputs.max_buffered_request_bytes {
            return BodyPath::Buffer(REASON_RETRYABLE);
        }
        return BodyPath::Stream(REASON_RETRY_FORFEITED);
    }
    BodyPath::Stream(REASON_PURE_FORWARD)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eligible(content_length: u64) -> BodyPathInputs {
        BodyPathInputs {
            policy_needs_text: false,
            model_ambiguous: false,
            worker_mutates_body: false,
            wasm_request_hooks: false,
            content_length: Some(content_length),
            retries_enabled: true,
            max_buffered_request_bytes: 1024,
        }
    }

    #[test]
    fn hard_buffer_reasons_apply_in_first_match_order() {
        let all = BodyPathInputs {
            policy_needs_text: true,
            model_ambiguous: true,
            worker_mutates_body: true,
            wasm_request_hooks: true,
            content_length: None,
            ..eligible(0)
        };
        assert_eq!(
            decide_body_path(&all),
            BodyPath::Buffer(REASON_POLICY_NEEDS_TEXT)
        );

        for (inputs, reason) in [
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
                    content_length: None,
                    ..eligible(64)
                },
                REASON_NO_CONTENT_LENGTH,
            ),
        ] {
            assert_eq!(decide_body_path(&inputs), BodyPath::Buffer(reason));
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
}
