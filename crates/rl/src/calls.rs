//! RL control over a worker's own engine connection, for a worker with no
//! HTTP control app: a TokenSpeed engine on the gateway's direct ZMQ lane
//! takes the engine-native RL routes as control calls named after them, with
//! the route's JSON body as the call's argument, on every rank it drives.

use std::{fmt, time::Duration};

use async_trait::async_trait;
use serde_json::{json, Value};

/// One rank's answer to a control call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankReply {
    pub success: bool,
    /// The engine's words, empty when it said nothing.
    pub message: String,
}

/// Why a control call did not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallFailure {
    /// A rank did not answer within the control deadline.
    Timeout,
    /// The call could not be made or the engine failed it: not connected,
    /// a dead engine.
    Failed(String),
    /// The connection cannot carry the call at all: the engine's wire has no
    /// control calls (its handshake declared no control reply), so the route
    /// is answered as for a worker without a control endpoint.
    Unsupported(String),
}

/// A worker whose engine connection carries the RL routes as control calls.
#[async_trait]
pub trait RlControlCalls: Send + Sync + fmt::Debug {
    /// Run the route `method` (`pause_generation`, ...) with its JSON
    /// `body` (`None` when the request carried none) on every rank the
    /// worker drives, each within `timeout`; each rank's reply.
    async fn call(
        &self,
        method: &str,
        body: Option<Value>,
        timeout: Duration,
    ) -> Result<Vec<RankReply>, CallFailure>;
}

/// The route's answer from its ranks' replies: `(status, body)` in the shape
/// the engine's HTTP route answers, 200 with `success: true` when every rank
/// succeeded, else 400 (the route's status for `success: false`) with the
/// failing ranks' distinct messages.
pub(crate) fn route_answer(replies: &[RankReply]) -> (u16, Value) {
    let success = replies.iter().all(|reply| reply.success);
    let mut messages: Vec<&str> = Vec::new();
    for reply in replies.iter().filter(|reply| success || !reply.success) {
        if !reply.message.is_empty() && !messages.contains(&reply.message.as_str()) {
            messages.push(&reply.message);
        }
    }
    let status = if success { 200 } else { 400 };
    (
        status,
        json!({"success": success, "message": messages.join("; ")}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(success: bool, message: &str) -> RankReply {
        RankReply {
            success,
            message: message.to_string(),
        }
    }

    /// Every rank must succeed; a failure names the failing ranks' distinct
    /// words, never a succeeding rank's.
    #[test]
    fn the_answer_succeeds_only_when_every_rank_did() {
        let (status, body) = route_answer(&[reply(true, ""), reply(true, "")]);
        assert_eq!(status, 200);
        assert_eq!(body, json!({"success": true, "message": ""}));

        let (status, body) = route_answer(&[
            reply(true, "ok"),
            reply(false, "a pause is already in progress"),
            reply(false, "a pause is already in progress"),
        ]);
        assert_eq!(status, 400);
        assert_eq!(
            body,
            json!({"success": false, "message": "a pause is already in progress"})
        );
    }
}
