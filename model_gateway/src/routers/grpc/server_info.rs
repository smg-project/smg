//! `GET /workers/{worker_id}/server_info`: the server info answer of one
//! worker's engine as a JSON object, under the names of the engine's wire.
//!
//! An engine asked over gRPC answers a message, and the body is that
//! message: every field under its own name. Where the message carries the
//! engine's arguments and its scheduler's facts as structs, each stays an
//! object under the message's name for it, `server_args` and
//! `scheduler_info` (`null` where the engine sent none), so an argument
//! stands one level down: `server_args.tp_size`. The body is not the flat
//! object an HTTP engine answers `/get_server_info` with.
//!
//! Every value of a struct keeps its type, but a whole number below 2 to the
//! 53rd in size reads as an integer, whether the engine holds a count or a
//! float there; a larger one stays a double, and a number that is not finite
//! reads `null`.
//!
//! A field of a struct whose name marks a secret is left out of the body,
//! at every depth.
//!
//! The route stands with the worker routes, behind their guard: the body
//! holds the engine's other arguments as the engine sent them.

use std::time::Duration;

use axum::response::Response;
use prost_types::value::Kind;
use serde_json::{json, Number, Value};
use smg_grpc_client::{engine_labels::ServerInfo, sglang_proto, tokenspeed_proto};
use smg_worker::identity::{
    basic::{ConnectionMode, Worker},
    endpoint::Endpoint,
};
use tonic::Code;
use tracing::warn;

use super::backend_client::get_backend_client;
use crate::routers::{common::json_response, error};

/// The bound of the route's call to a worker: a worker that sends no answer
/// costs the route's caller this long, and no longer.
const BOUND: Duration = Duration::from_secs(5);

/// Answers the route for `worker`: asks its engine, over the gateway's own
/// channel to the worker, once for every request, with nothing kept between
/// two.
///
/// A worker on the ZMQ link is not asked: its client answers from the
/// handshake, not with what the engine would say now. An HTTP worker has no
/// such channel. Both read `501`, as does an engine that does not offer the
/// call.
///
/// A worker that is asked and gives no answer reads by its cause, each said
/// in a warn line: `504` for no answer within [`BOUND`], `502` for any other
/// failure of the way to the worker's client or of the engine's call. The
/// way to the client and the call stand under that bound together, the call
/// carrying no deadline of its own.
pub(crate) async fn answer(worker: &dyn Worker) -> Response {
    within(worker, BOUND).await
}

/// [`answer`], with `bound` as the time the worker has.
async fn within(worker: &dyn Worker, bound: Duration) -> Response {
    if matches!(worker.connection_mode(), ConnectionMode::Zmq) {
        return not_implemented();
    }
    let answered = tokio::time::timeout(bound, async {
        let client = match get_backend_client(worker).await {
            Ok(Some(client)) => client,
            // An HTTP worker: no channel to ask over.
            Ok(None) => return not_implemented(),
            Err(error) => {
                warn!(
                    worker = Endpoint::redacted_str(worker.url()),
                    %error,
                    "Server info: no client for the worker"
                );
                return failed();
            }
        };
        match client.get_server_info().await {
            Ok(info) => body_of(&info),
            Err(status) => {
                warn!(
                    worker = Endpoint::redacted_str(worker.url()),
                    code = ?status.code(),
                    message = status.message(),
                    "Server info: the engine failed the call"
                );
                if status.code() == Code::Unimplemented {
                    not_implemented()
                } else {
                    failed()
                }
            }
        }
    })
    .await;
    answered.unwrap_or_else(|_| {
        warn!(
            worker = Endpoint::redacted_str(worker.url()),
            ?bound,
            "Server info: no answer from the worker in time"
        );
        timed_out()
    })
}

/// The route is not answered for this worker: it is not asked, or its
/// engine does not offer the call.
pub(crate) fn not_implemented() -> Response {
    error::not_implemented("server_info_not_implemented", "Server info not implemented")
}

/// The worker was asked and the way to it or its engine's call failed.
fn failed() -> Response {
    error::bad_gateway(
        "server_info_failed",
        "The worker failed the server info call",
    )
}

/// The worker was asked and sent no answer within the bound.
fn timed_out() -> Response {
    error::gateway_timeout(
        "server_info_timeout",
        "The worker sent no server info in time",
    )
}

/// The route's `200` for one engine's answer: its message, every field under
/// its own name.
fn body_of(info: &ServerInfo) -> Response {
    match info {
        // A message of flat fields: its names are the wire's own.
        ServerInfo::Vllm(info) => json_response(&**info),
        // A message with structs is written by hand, every field named with
        // no rest: a field the message gains fails the build here.
        ServerInfo::TokenSpeed(info) => {
            let tokenspeed_proto::GetServerInfoResponse {
                server_args,
                scheduler_info,
                active_requests,
                is_paused,
                uptime_seconds,
                max_total_num_tokens,
                tokenspeed_version,
                start_time,
                constrained_decoding,
            } = &**info;
            json_response(&json!({
                "server_args": server_args.as_ref().map(record),
                "scheduler_info": scheduler_info.as_ref().map(record),
                "active_requests": active_requests,
                "is_paused": is_paused,
                "uptime_seconds": uptime_seconds,
                "max_total_num_tokens": max_total_num_tokens,
                "tokenspeed_version": tokenspeed_version,
                "start_time": start_time.as_ref().map(ToString::to_string),
                "constrained_decoding": constrained_decoding,
            }))
        }
        // A message of flat fields too.
        ServerInfo::Trtllm(info) => json_response(info),
        ServerInfo::Sglang(info) => {
            let sglang_proto::GetServerInfoResponse {
                server_args,
                scheduler_info,
                active_requests,
                is_paused,
                last_receive_timestamp,
                uptime_seconds,
                sglang_version,
                server_type,
                start_time,
                max_total_num_tokens,
            } = &**info;
            json_response(&json!({
                "server_args": server_args.as_ref().map(record),
                "scheduler_info": scheduler_info.as_ref().map(record),
                "active_requests": active_requests,
                "is_paused": is_paused,
                "last_receive_timestamp": last_receive_timestamp,
                "uptime_seconds": uptime_seconds,
                "sglang_version": sglang_version,
                "server_type": server_type,
                "start_time": start_time.as_ref().map(ToString::to_string),
                "max_total_num_tokens": max_total_num_tokens,
            }))
        }
        ServerInfo::Mlx(info) => json_response(info),
    }
}

/// A struct the engine sent as a JSON object: each value in its JSON type,
/// without the fields whose name marks a secret ([`names_a_secret`]). A
/// struct inside it, or inside a list of it, is written the same way, so
/// such a field is left out at every depth.
fn record(fields: &prost_types::Struct) -> Value {
    fields
        .fields
        .iter()
        .filter(|(name, _)| !names_a_secret(name))
        .map(|(name, value)| (name.clone(), json_of(value)))
        .collect()
}

/// The parts of a name that mark a secret wherever they stand in it.
const SECRET_PARTS: [&str; 3] = ["api_key", "secret", "password"];

/// Whether a field's name marks a secret, in any case of its letters: the
/// name `token`, a name that ends in `_token`, or one that holds a part of
/// [`SECRET_PARTS`]. It is the rule a servicer keeps at the source for the
/// arguments it sends; kept here too, the answer holds no such field
/// whichever servicer speaks.
fn names_a_secret(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name == "token"
        || name.ends_with("_token")
        || SECRET_PARTS.iter().any(|part| name.contains(part))
}

/// Every whole double below this one in size (2 to the 53rd) is an integer
/// the double holds exactly.
const EXACT_WHOLE: f64 = 9_007_199_254_740_992.0;

fn json_of(value: &prost_types::Value) -> Value {
    match &value.kind {
        None | Some(Kind::NullValue(_)) => Value::Null,
        Some(Kind::BoolValue(flag)) => Value::Bool(*flag),
        // A struct carries every number as a double: a whole one that a
        // double holds exactly is written as an integer, so a count reads
        // as it does in an HTTP engine's answer; a float argument of a
        // whole value reads as an integer too, where that answer keeps its
        // point. Any other finite number stays a double.
        Some(Kind::NumberValue(number)) => {
            if number.fract() == 0.0 && number.abs() < EXACT_WHOLE {
                Value::from(*number as i64)
            } else {
                Number::from_f64(*number).map_or(Value::Null, Value::Number)
            }
        }
        Some(Kind::StringValue(text)) => Value::String(text.clone()),
        Some(Kind::ListValue(list)) => list.values.iter().map(json_of).collect(),
        Some(Kind::StructValue(fields)) => record(fields),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/grpc/server_info/tests.rs"]
mod tests;
