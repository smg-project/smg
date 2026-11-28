//! vLLM PD KV-transfer connector handling, shared by both transports.
//!
//! The prefill side is tagged with connector params, the engine returns (or
//! the router mints) the handoff params, and the decode side carries them;
//! only a MoRI-IO WRITE decode side can also go out first, with params minted
//! up front. The gRPC pipeline and the HTTP PD router both implement that
//! flow; the connector vocabulary lives here so the two stay in lockstep.

use std::borrow::Cow;

use serde_json::{value::RawValue, Value};
use smg_worker::{
    identity::basic::{
        Worker, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, MORIIO_CONNECTOR, MORIIO_MODE_LABEL,
        NIXL_CONNECTOR,
    },
    pd::pairing::{connector_profile, KvTransferProfile},
};
use tracing::warn;

use crate::observability::metrics::metrics_labels;

/// KV-transfer params tagged onto the NIXL prefill side so the engine pins its
/// KV blocks and returns the handoff params for the decode worker.
pub(crate) const NIXL_PREFILL_KV_PARAMS: &str =
    r#"{"do_remote_decode":true,"do_remote_prefill":false}"#;

/// PD KV-transfer behavior derived from prefill worker metadata.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum KvConnectorMode {
    /// MooncakeConnector: mint a transfer_id, tag both sides, synthesize decode
    /// params from worker metadata; legacy host/port injection when the
    /// servicer predates kv_engine_id reporting (or DP runs without a pinned rank).
    Mooncake {
        host: String,
        port: u32,
        engine_id: Option<String>,
    },
    /// NixlConnector: tag prefill with do_remote_decode, relay returned params to decode.
    Nixl,
    /// MoRIIOConnector: mint a transfer_id, tag prefill with the decode peer
    /// (see [`moriio_prefill_params`]), relay the validated handoff to decode,
    /// or, dispatched concurrently, tag decode with the prefill peer (see
    /// [`moriio_decode_params`]).
    MoriIo,
    /// A connector whose sides meet in a store the deployment shares (the
    /// Mooncake store, the shared-storage, FlexKV and 3FS connectors; see
    /// [`KvTransferProfile::Store`]): the decode engine reads the prefill's
    /// blocks back by prefix hash, so the gateway mints nothing: neither side
    /// is tagged; params the engine does return are relayed, for a store
    /// that runs a handoff of its own.
    Store,
    /// Unknown/absent connector: relay returned params opportunistically.
    Passthrough,
}

impl KvConnectorMode {
    pub(crate) fn metrics_label(&self) -> &'static str {
        match self {
            Self::Mooncake { .. } => metrics_labels::KV_CONNECTOR_MOONCAKE,
            Self::Nixl => metrics_labels::KV_CONNECTOR_NIXL,
            Self::MoriIo => metrics_labels::KV_CONNECTOR_MORIIO,
            Self::Store => metrics_labels::KV_CONNECTOR_STORE,
            Self::Passthrough => metrics_labels::KV_CONNECTOR_PASSTHROUGH,
        }
    }
}

pub(crate) fn kv_connector_mode(
    kv_connector: Option<&str>,
    bootstrap_host: &str,
    bootstrap_port: Option<u16>,
    kv_engine_id: Option<&str>,
) -> KvConnectorMode {
    match kv_connector {
        Some(MOONCAKE_CONNECTOR) => KvConnectorMode::Mooncake {
            host: bootstrap_host.to_string(),
            port: u32::from(bootstrap_port.unwrap_or(DEFAULT_BOOTSTRAP_PORT)),
            // Empty means unknown (forces the legacy fallback)
            engine_id: kv_engine_id.filter(|s| !s.is_empty()).map(str::to_string),
        },
        Some(NIXL_CONNECTOR) => KvConnectorMode::Nixl,
        Some(MORIIO_CONNECTOR) => KvConnectorMode::MoriIo,
        Some(connector) if connector_profile(connector) == KvTransferProfile::Store => {
            KvConnectorMode::Store
        }
        _ => KvConnectorMode::Passthrough,
    }
}

/// Worker labels carrying a MoRI-IO engine's side channel. HTTP vLLM workers
/// report none of this, so the operator sets them at registration to match
/// the engine's `kv_connector_extra_config`.
pub(crate) const MORIIO_HOST_LABEL: &str = "moriio_host";
pub(crate) const MORIIO_HANDSHAKE_PORT_LABEL: &str = "moriio_handshake_port";
pub(crate) const MORIIO_NOTIFY_PORT_LABEL: &str = "moriio_notify_port";
/// How the two WRITE sides are sent: `sequential` (default) or `concurrent`.
/// Honoured on the decode worker; an invalid value on either worker refuses
/// the pair.
pub(crate) const MORIIO_WRITE_DISPATCH_LABEL: &str = "moriio_write_dispatch";
/// vLLM `MoRIIOConstants.DEFAULT_HANDSHAKE_PORT` / `DEFAULT_NOTIFY_PORT`.
const MORIIO_DEFAULT_HANDSHAKE_PORT: u16 = 6301;
const MORIIO_DEFAULT_NOTIFY_PORT: u16 = 61005;

/// Request-id markers the MoRI-IO connector parses a peer address from. The
/// connector prefers them over explicit peer fields, and vLLM takes the
/// request id from `X-Request-Id` or else the body's `request_id`, so a
/// client-chosen id carrying them would redirect the transfer.
const MORIIO_REQUEST_ID_MARKERS: [&str; 2] = ["___prefill_addr_", "___decode_addr_"];

pub(crate) fn carries_moriio_peer_markers(request_id: &str) -> bool {
    MORIIO_REQUEST_ID_MARKERS
        .iter()
        .any(|marker| request_id.contains(marker))
}

/// MoRI-IO transfer direction; both sides of a pair must run the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoriIoTransfer {
    /// The decode engine pulls KV after the prefill side returns.
    Read,
    /// The prefill engine pushes KV into blocks the decode engine allocates.
    Write,
}

/// A worker's MoRI-IO endpoint, as configured on its labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MoriIoEndpoint {
    pub(crate) transfer: MoriIoTransfer,
    pub(crate) host: String,
    pub(crate) handshake_port: u16,
    pub(crate) notify_port: u16,
    /// From the `tp_size` label. Unset means the peer's TP is unknown, which
    /// the connector treats as equal to its own.
    pub(crate) tp_size: Option<usize>,
    /// From the worker's `dp_size` or the `dp_size` label, 1 when unset.
    pub(crate) dp_size: usize,
    /// Send the WRITE sides at once. Opt-in: a decode request aborted before
    /// the prefill pushes leaves its blocks allocated in the vLLM connector.
    pub(crate) concurrent_write: bool,
}

pub(crate) fn is_moriio_worker(worker: &dyn Worker) -> bool {
    worker.metadata().spec.kv_connector.as_deref() == Some(MORIIO_CONNECTOR)
}

/// Read a MoRIIOConnector worker's endpoint. The transfer mode has no safe
/// default: a READ engine paired with WRITE params returns garbage or hangs.
/// The host defaults to the worker URL's host and the ports to vLLM's.
pub(crate) fn moriio_endpoint(worker: &dyn Worker) -> Result<MoriIoEndpoint, String> {
    let spec = &worker.metadata().spec;
    // The address as the refusals name it: a userinfo in it masked.
    let url = worker.redacted_url();
    if !is_moriio_worker(worker) {
        return Err(format!("{url} is not a {MORIIO_CONNECTOR} worker"));
    }
    let labels = &spec.labels;
    let label = |name: &str| labels.get(name).map(|v| v.trim()).filter(|v| !v.is_empty());
    let transfer = match label(MORIIO_MODE_LABEL)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("read") => MoriIoTransfer::Read,
        Some("write") => MoriIoTransfer::Write,
        other => {
            return Err(format!(
                "{url}: label {MORIIO_MODE_LABEL} must be \"read\" or \"write\", got {other:?}"
            ))
        }
    };
    if worker.dp_rank().is_some() {
        return Err(format!(
            "{url}: MoRI-IO PD pins both sides of a request to one data-parallel rank itself; \
             register the engine once, with a dp_size label, instead of one worker per rank"
        ));
    }
    let dp_size = match (worker.dp_size(), label("dp_size")) {
        (Some(dp_size), _) => dp_size,
        (None, None) => 1,
        (None, Some(v)) => v
            .parse::<usize>()
            .map_err(|_| format!("{url}: label dp_size is not an integer: {v:?}"))?,
    };
    if dp_size == 0 {
        return Err(format!("{url}: dp_size must be at least 1"));
    }
    let port = |name: &str, default: u16| match label(name) {
        None => Ok(default),
        Some(v) => v
            .parse::<u16>()
            .ok()
            .filter(|p| *p > 0)
            .ok_or_else(|| format!("{url}: label {name} is not a port: {v:?}")),
    };
    let tp_size = label("tp_size")
        .map(|v| {
            v.parse::<usize>()
                .ok()
                .filter(|t| *t > 0)
                .ok_or_else(|| format!("{url}: label tp_size is not a positive integer: {v:?}"))
        })
        .transpose()?;
    let host = label(MORIIO_HOST_LABEL).unwrap_or(spec.bootstrap_host.as_str());
    if host.is_empty() {
        return Err(format!(
            "{url}: no MoRI-IO host (set label {MORIIO_HOST_LABEL})"
        ));
    }
    let concurrent_write = match label(MORIIO_WRITE_DISPATCH_LABEL)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("sequential") => false,
        Some("concurrent") => true,
        Some(other) => {
            return Err(format!(
                "{url}: label {MORIIO_WRITE_DISPATCH_LABEL} must be \"sequential\" or \
                 \"concurrent\", got {other:?}"
            ))
        }
    };
    Ok(MoriIoEndpoint {
        transfer,
        host: host.to_string(),
        handshake_port: port(MORIIO_HANDSHAKE_PORT_LABEL, MORIIO_DEFAULT_HANDSHAKE_PORT)?,
        notify_port: port(MORIIO_NOTIFY_PORT_LABEL, MORIIO_DEFAULT_NOTIFY_PORT)?,
        tp_size,
        dp_size,
        concurrent_write,
    })
}

/// The header by which vLLM's API server sends a request to one data-parallel
/// engine core.
pub(crate) const DATA_PARALLEL_RANK_HEADER: &str = "x-data-parallel-rank";

/// Why one data-parallel rank cannot serve both sides. A WRITE decode engine
/// notifies the prefill only from the rank the prefill ran on, so both sides
/// go to the same rank, which both engines must have.
pub(crate) fn moriio_dp_mismatch_error(
    prefill: &MoriIoEndpoint,
    decode: &MoriIoEndpoint,
) -> Option<String> {
    (prefill.dp_size != decode.dp_size).then(|| {
        format!(
            "prefill dp_size={} but decode dp_size={}: MoRI-IO PD sends both sides of a \
             request to the same data-parallel rank, so both engines need the same dp_size",
            prefill.dp_size, decode.dp_size
        )
    })
}

/// Prefill-side params for MoRI-IO. `remote_tp_size` is the decode's TP (the
/// connector reads it as the peer's), sent only when labeled; the prefill
/// engine returns its own as `tp_size` in the handoff. `dp_rank` is the
/// data-parallel rank both sides are pinned to. WRITE also names the decode's
/// side channel: the producer pushes KV before the decode side exists.
pub(crate) fn moriio_prefill_params(
    transfer_id: &str,
    decode: &MoriIoEndpoint,
    dp_rank: Option<usize>,
) -> String {
    let mut params = serde_json::json!({
        "do_remote_decode": true,
        "do_remote_prefill": false,
        "remote_engine_id": null,
        "remote_block_ids": null,
        "transfer_id": transfer_id,
        "remote_dp_size": decode.dp_size,
    });
    if let Some(rank) = dp_rank {
        params["remote_dp_rank"] = Value::from(rank);
    }
    if let Some(tp_size) = decode.tp_size {
        params["remote_tp_size"] = Value::from(tp_size);
    }
    if decode.transfer == MoriIoTransfer::Write {
        params["remote_host"] = Value::from(decode.host.as_str());
        params["remote_handshake_port"] = Value::from(decode.handshake_port);
        params["remote_notify_port"] = Value::from(decode.notify_port);
    }
    params.to_string()
}

/// Decode-side params for MoRI-IO WRITE, minted up front so both sides can go
/// out at once: the decode engine allocates its blocks and tells the prefill
/// engine where to push them, through the side channel of the prefill's
/// `dp_rank`.
pub(crate) fn moriio_decode_params(
    transfer_id: &str,
    prefill: &MoriIoEndpoint,
    dp_rank: Option<usize>,
) -> String {
    let mut params = serde_json::json!({
        "do_remote_decode": false,
        "do_remote_prefill": true,
        "remote_engine_id": null,
        "remote_block_ids": null,
        "transfer_id": transfer_id,
        "remote_dp_size": prefill.dp_size,
        "remote_host": prefill.host,
        "remote_handshake_port": prefill.handshake_port,
        "remote_notify_port": prefill.notify_port,
    });
    if let Some(rank) = dp_rank {
        params["remote_dp_rank"] = Value::from(rank);
    }
    if let Some(tp_size) = prefill.tp_size {
        params["remote_tp_size"] = Value::from(tp_size);
    }
    params.to_string()
}

/// Why a WRITE producer, told to push to `decode`, would not reach the
/// decode engine. A loopback or unspecified host reaches only the producer's
/// own machine, and a side channel equal to the prefill's makes the producer
/// its own peer. Under concurrent dispatch the decode engine also dials the
/// prefill's side channel as named here, so the same holds the other way
/// round.
pub(crate) fn moriio_write_target_error(
    prefill: &MoriIoEndpoint,
    decode: &MoriIoEndpoint,
) -> Option<String> {
    let local_only = |host: &str| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified())
    };
    let (prefill_local, decode_local) = (local_only(&prefill.host), local_only(&decode.host));
    if decode_local && !prefill_local {
        return Some(format!(
            "decode MoRI-IO host {} is loopback or unspecified but the prefill is at {}; \
             set label {MORIIO_HOST_LABEL} on the decode worker",
            decode.host, prefill.host
        ));
    }
    if decode.concurrent_write && prefill_local && !decode_local {
        return Some(format!(
            "prefill MoRI-IO host {} is loopback or unspecified but the decode, dispatched \
             concurrently, is at {}; set label {MORIIO_HOST_LABEL} on the prefill worker",
            prefill.host, decode.host
        ));
    }
    let same_host = prefill.host == decode.host || (prefill_local && decode_local);
    if same_host
        && (prefill.handshake_port == decode.handshake_port
            || prefill.notify_port == decode.notify_port)
    {
        return Some(format!(
            "prefill and decode share the MoRI-IO side channel on {} \
             (handshake {}/{}, notify {}/{}); label distinct ports",
            decode.host,
            prefill.handshake_port,
            decode.handshake_port,
            prefill.notify_port,
            decode.notify_port
        ));
    }
    None
}

/// Check the prefill engine's MoRI-IO handoff before the decode side goes
/// out, or before a concurrent decode side's response is forwarded. A MoRI-IO
/// decode engine without it computes over KV that never arrives (READ) or
/// waits for a push that never comes (WRITE), so a missing, foreign or
/// malformed handoff must fail the request instead. `dp_rank` is the
/// data-parallel rank both sides were pinned to, if any.
pub(crate) fn validate_moriio_handoff(
    handoff: &RawValue,
    transfer_id: &str,
    dp_rank: Option<usize>,
) -> Result<(), String> {
    let params: Value = serde_json::from_str(handoff.get())
        .map_err(|e| format!("unparsable kv_transfer_params: {e}"))?;
    let params = params
        .as_object()
        .ok_or("kv_transfer_params is not an object")?;
    if params.get("do_remote_prefill") != Some(&Value::Bool(true)) {
        return Err("handoff does not ask decode for a remote prefill".to_string());
    }
    let returned_id = params.get("transfer_id").and_then(Value::as_str);
    if returned_id != Some(transfer_id) {
        return Err(format!(
            "handoff transfer_id {returned_id:?} does not match the minted {transfer_id}"
        ));
    }
    for key in [
        "remote_engine_id",
        "remote_block_ids",
        "remote_host",
        "remote_handshake_port",
        "remote_notify_port",
    ] {
        if params.get(key).is_none_or(Value::is_null) {
            return Err(format!("handoff is missing {key}"));
        }
    }
    // The decode engine's connector parses these in its scheduler step, where
    // a value of the wrong shape raises instead of failing one request.
    let malformed = |key: &str| format!("handoff {key} is malformed: {}", params[key]);
    if !params["remote_engine_id"]
        .as_str()
        .is_some_and(|id| !id.is_empty())
    {
        return Err(malformed("remote_engine_id"));
    }
    if !is_block_id_list(&params["remote_block_ids"]) {
        return Err(malformed("remote_block_ids"));
    }
    if !params["remote_host"]
        .as_str()
        .is_some_and(|host| !host.is_empty() && !host.contains(char::is_whitespace))
    {
        return Err(malformed("remote_host"));
    }
    for key in ["remote_handshake_port", "remote_notify_port"] {
        if handoff_port(&params[key]).is_none() {
            return Err(malformed(key));
        }
    }
    // The connector uses some of these without converting them (a string
    // `remote_dp_rank` never equals the engine's own rank), and the producer
    // reports them as integers.
    for key in [
        "remote_dp_size",
        "remote_dp_size_local",
        "remote_dp_rank",
        "tp_size",
        "remote_tp_size",
    ] {
        if params
            .get(key)
            .is_some_and(|value| !value.is_null() && value.as_u64().is_none())
        {
            return Err(malformed(key));
        }
    }
    // A WRITE decode engine notifies the prefill only when it runs on the
    // rank the handoff names, and the decode side goes to the pinned rank.
    if let Some(rank) = dp_rank {
        let named = params.get("remote_dp_rank").and_then(Value::as_u64);
        if named.and_then(|n| usize::try_from(n).ok()) != Some(rank) {
            return Err(format!(
                "handoff remote_dp_rank {} is not the pinned data-parallel rank {rank}",
                params.get("remote_dp_rank").unwrap_or(&Value::Null)
            ));
        }
    }
    Ok(())
}

/// Block ids as a MoRI-IO producer reports them: one list per KV cache group,
/// which the decode engine pairs with its own groups, and at least one block.
fn is_block_id_list(value: &Value) -> bool {
    let Some(groups) = value.as_array() else {
        return false;
    };
    let mut blocks = 0;
    for group in groups {
        let Some(ids) = group.as_array() else {
            return false;
        };
        if !ids.iter().all(|id| id.as_u64().is_some()) {
            return false;
        }
        blocks += ids.len();
    }
    blocks > 0
}

/// A side-channel port in a handoff: the engine reports its configured port
/// as a string or a number.
fn handoff_port(value: &Value) -> Option<u16> {
    let port = match value {
        Value::String(port) => port.trim().parse::<u16>().ok(),
        Value::Number(port) => port.as_u64().and_then(|port| u16::try_from(port).ok()),
        _ => None,
    };
    port.filter(|port| *port != 0)
}

/// Why a valid handoff from `prefill` does not match the side channel a
/// concurrent decode side was sent to notify: that decode engine then waits for
/// a push that never comes. The engine reports its configured base ports,
/// as a string or a number.
pub(crate) fn moriio_side_channel_error(
    handoff: &RawValue,
    prefill: &MoriIoEndpoint,
) -> Option<String> {
    let params: Value = serde_json::from_str(handoff.get()).ok()?;
    let port = |key: &str| params.get(key).and_then(handoff_port);
    let reported = (port("remote_handshake_port"), port("remote_notify_port"));
    if reported == (Some(prefill.handshake_port), Some(prefill.notify_port)) {
        return None;
    }
    let shown = |key: &str| {
        params
            .get(key)
            .map_or_else(|| "none".to_string(), Value::to_string)
    };
    Some(format!(
        "prefill reports MoRI-IO side channel handshake {} / notify {}, but the decode worker \
         was told {} / {}; set labels {MORIIO_HANDSHAKE_PORT_LABEL} / \
         {MORIIO_NOTIFY_PORT_LABEL} on the prefill worker",
        shown("remote_handshake_port"),
        shown("remote_notify_port"),
        prefill.handshake_port,
        prefill.notify_port
    ))
}

/// Connector id of the engine core serving the prefill side. With DP the cores
/// suffix the configured id as `{base}_dp{rank}`, so minting needs a pinned
/// rank; unpinned DP>1 yields None (no mint — decode recomputes locally).
fn effective_kv_engine_id(
    base: Option<&str>,
    dp_size: Option<usize>,
    dp_rank: Option<usize>,
) -> Option<String> {
    let base = base.filter(|s| !s.is_empty())?;
    if dp_size.unwrap_or(1) > 1 {
        dp_rank.map(|rank| format!("{base}_dp{rank}"))
    } else {
        Some(base.to_string())
    }
}

/// The connector mode for a PD pair, read off the prefill worker's metadata.
/// Discovered dp_size matters for a worker that is not a rank worker too:
/// a DP>1 engine behind an unexpanded worker must not be minted for.
pub(crate) fn connector_mode_for_worker(worker: &dyn Worker) -> KvConnectorMode {
    let meta = worker.metadata();
    let engine_id = worker_kv_engine_id(worker).unwrap_or_else(|dp_label| {
        warn!(
            worker = %worker.redacted_url(),
            dp_size = ?dp_label,
            "invalid dp_size label; treating DP topology as unknown and \
             skipping KV engine-id minting"
        );
        None
    });
    kv_connector_mode(
        meta.spec.kv_connector.as_deref(),
        &meta.spec.bootstrap_host,
        meta.spec.bootstrap_port,
        engine_id.as_deref(),
    )
}

/// The KV engine id a handoff to `worker` is minted for: its discovered id
/// under its DP topology ([`effective_kv_engine_id`]), or none. A `dp_size`
/// label that does not parse as a positive integer, where no size was
/// discovered, means the topology is unknown, not absent: minting an
/// unsuffixed id could target the wrong engine core, so nothing is minted,
/// and the label comes back for the caller to report.
fn worker_kv_engine_id(worker: &dyn Worker) -> Result<Option<String>, Option<&String>> {
    let dp_label = worker.metadata().spec.labels.get("dp_size");
    let label_dp = dp_label.and_then(|s| s.parse::<usize>().ok().filter(|v| *v > 0));
    if worker.dp_size().is_none() && dp_label.is_some() && label_dp.is_none() {
        return Err(dp_label);
    }
    let dp_size = worker.dp_size().or(label_dp);
    Ok(effective_kv_engine_id(
        worker.kv_engine_id().as_deref(),
        dp_size,
        worker.dp_rank(),
    ))
}

/// Whether a handoff to `worker` is minted a KV engine id, as
/// [`connector_mode_for_worker`] reads it but without its warning: for a
/// caller that asks of every candidate on every request (selection's
/// probes), where the dispatch's own read reports a bad label once.
pub(crate) fn mints_kv_engine_id(worker: &dyn Worker) -> bool {
    matches!(worker_kv_engine_id(worker), Ok(Some(_)))
}

/// Prefill-side params for Mooncake: the engine pins blocks under the minted id.
pub(crate) fn mooncake_prefill_params(transfer_id: &str) -> String {
    serde_json::json!({
        "do_remote_decode": true,
        "do_remote_prefill": false,
        "transfer_id": transfer_id,
    })
    .to_string()
}

/// `host` as it goes into a URL authority: an IPv6 literal in brackets, so
/// `http://{host}:{port}` parses whether the bootstrap host came bracketed
/// out of a worker URL or bare from a label or a configured value.
fn url_host(host: &str) -> Cow<'_, str> {
    if host.contains(':') && !host.starts_with('[') {
        Cow::Owned(format!("[{host}]"))
    } else {
        Cow::Borrowed(host)
    }
}

/// Decode-side params for Mooncake, synthesized from prefill worker metadata
/// (the engine returns nothing to relay; the connector is push-based).
pub(crate) fn mooncake_decode_params(
    transfer_id: &str,
    engine_id: &str,
    host: &str,
    port: u32,
) -> String {
    serde_json::json!({
        "do_remote_decode": false,
        "do_remote_prefill": true,
        "transfer_id": transfer_id,
        "remote_engine_id": engine_id,
        "remote_bootstrap_addr": format!("http://{}:{port}", url_host(host)),
    })
    .to_string()
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/kv_transfer/tests.rs"]
mod tests;
