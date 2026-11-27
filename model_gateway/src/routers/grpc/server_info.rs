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
//! The route stands with the worker routes, behind their guard: the body
//! holds the engine's arguments as the engine sent them.

use std::time::Duration;

use axum::response::Response;
use prost_types::value::Kind;
use serde_json::{json, Number, Value};
use smg_grpc_client::engine_labels::ServerInfo;
use smg_worker::identity::basic::{ConnectionMode, Worker};

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
/// such channel.
///
/// Every state without an engine's answer reads `501`: a worker that is not
/// asked, no client for the one asked, a call the engine fails or does not
/// offer, and no answer within [`BOUND`]. The way to the worker's client and
/// the engine's call stand under that bound together, the call carrying no
/// deadline of its own.
pub(crate) async fn answer(worker: &dyn Worker) -> Response {
    if matches!(worker.connection_mode(), ConnectionMode::Zmq) {
        return not_implemented();
    }
    let info = tokio::time::timeout(BOUND, async {
        let client = get_backend_client(worker).await.ok()??;
        client.get_server_info().await.ok()
    })
    .await;
    match info {
        Ok(Some(info)) => body_of(&info),
        Ok(None) | Err(_) => not_implemented(),
    }
}

/// No engine's answer: the route is not answered for this worker.
pub(crate) fn not_implemented() -> Response {
    error::not_implemented("server_info_not_implemented", "Server info not implemented")
}

/// The route's `200` for one engine's answer: its message, every field under
/// its own name.
fn body_of(info: &ServerInfo) -> Response {
    match info {
        // A message of flat fields: its names are the wire's own.
        ServerInfo::Vllm(info) => json_response(&**info),
        ServerInfo::TokenSpeed(info) => json_response(&json!({
            "server_args": info.server_args.as_ref().map(record),
            "scheduler_info": info.scheduler_info.as_ref().map(record),
            "active_requests": info.active_requests,
            "is_paused": info.is_paused,
            "uptime_seconds": info.uptime_seconds,
            "max_total_num_tokens": info.max_total_num_tokens,
            "tokenspeed_version": info.tokenspeed_version,
            "start_time": info.start_time.as_ref().map(ToString::to_string),
            "constrained_decoding": info.constrained_decoding,
        })),
        // A message of flat fields too.
        ServerInfo::Trtllm(info) => json_response(info),
        ServerInfo::Sglang(info) => json_response(&json!({
            "server_args": info.server_args.as_ref().map(record),
            "scheduler_info": info.scheduler_info.as_ref().map(record),
            "active_requests": info.active_requests,
            "is_paused": info.is_paused,
            "last_receive_timestamp": info.last_receive_timestamp,
            "uptime_seconds": info.uptime_seconds,
            "sglang_version": info.sglang_version,
            "server_type": info.server_type,
            "start_time": info.start_time.as_ref().map(ToString::to_string),
            "max_total_num_tokens": info.max_total_num_tokens,
        })),
        ServerInfo::Mlx(info) => json_response(info),
    }
}

/// A struct the engine sent as a JSON object: each value in its JSON type.
fn record(fields: &prost_types::Struct) -> Value {
    fields
        .fields
        .iter()
        .map(|(name, value)| (name.clone(), json_of(value)))
        .collect()
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
