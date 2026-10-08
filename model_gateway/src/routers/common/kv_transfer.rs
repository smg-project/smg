//! vLLM PD KV-transfer connector handling, shared by both transports.
//!
//! The prefill leg is tagged with connector params, the engine returns (or
//! the router mints) the handoff params, and the decode leg carries them;
//! only a MoRI-IO WRITE decode leg can also go out first, with params minted
//! up front. The gRPC pipeline and the HTTP PD router both implement that
//! flow; the connector vocabulary lives here so the two stay in lockstep.

use std::borrow::Cow;

use serde_json::{value::RawValue, Value};
use tracing::warn;

use crate::{
    observability::metrics::metrics_labels,
    worker::{
        Worker, DEFAULT_BOOTSTRAP_PORT, MOONCAKE_CONNECTOR, MORIIO_CONNECTOR, MORIIO_MODE_LABEL,
        NIXL_CONNECTOR,
    },
};

/// KV-transfer params tagged onto the NIXL prefill leg so the engine pins its
/// KV blocks and returns the handoff params for the decode worker.
pub(crate) const NIXL_PREFILL_KV_PARAMS: &str =
    r#"{"do_remote_decode":true,"do_remote_prefill":false}"#;

/// PD KV-transfer behavior derived from prefill worker metadata.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum KvConnectorMode {
    /// MooncakeConnector: mint a transfer_id, tag both legs, synthesize decode
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
    /// Unknown/absent connector: relay returned params opportunistically.
    Passthrough,
}

impl KvConnectorMode {
    pub(crate) fn metrics_label(&self) -> &'static str {
        match self {
            Self::Mooncake { .. } => metrics_labels::KV_CONNECTOR_MOONCAKE,
            Self::Nixl => metrics_labels::KV_CONNECTOR_NIXL,
            Self::MoriIo => metrics_labels::KV_CONNECTOR_MORIIO,
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
        _ => KvConnectorMode::Passthrough,
    }
}

/// Worker labels carrying a MoRI-IO engine's side channel. HTTP vLLM workers
/// report none of this, so the operator sets them at registration to match
/// the engine's `kv_connector_extra_config`.
pub(crate) const MORIIO_HOST_LABEL: &str = "moriio_host";
pub(crate) const MORIIO_HANDSHAKE_PORT_LABEL: &str = "moriio_handshake_port";
pub(crate) const MORIIO_NOTIFY_PORT_LABEL: &str = "moriio_notify_port";
/// How the two WRITE legs are sent: `sequential` (default) or `concurrent`.
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

/// MoRI-IO transfer direction; both legs of a pair must run the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoriIoTransfer {
    /// The decode engine pulls KV after the prefill leg returns.
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
    /// Send the WRITE legs at once. Opt-in: a decode request aborted before
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
    let url = worker.url();
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
            "{url}: MoRI-IO PD pins both legs of a request to one data-parallel rank itself; \
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

/// Why one data-parallel rank cannot serve both legs. A WRITE decode engine
/// notifies the prefill only from the rank the prefill ran on, so both legs
/// go to the same rank, which both engines must have.
pub(crate) fn moriio_dp_mismatch_error(
    prefill: &MoriIoEndpoint,
    decode: &MoriIoEndpoint,
) -> Option<String> {
    (prefill.dp_size != decode.dp_size).then(|| {
        format!(
            "prefill dp_size={} but decode dp_size={}: MoRI-IO PD sends both legs of a \
             request to the same data-parallel rank, so both engines need the same dp_size",
            prefill.dp_size, decode.dp_size
        )
    })
}

/// Prefill-leg params for MoRI-IO. `remote_tp_size` is the decode's TP (the
/// connector reads it as the peer's), sent only when labeled; the prefill
/// engine returns its own as `tp_size` in the handoff. `dp_rank` is the
/// data-parallel rank both legs are pinned to. WRITE also names the decode's
/// side channel: the producer pushes KV before the decode leg exists.
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

/// Decode-leg params for MoRI-IO WRITE, minted up front so both legs can go
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

/// Check the prefill engine's MoRI-IO handoff before the decode leg goes
/// out, or before a concurrent decode leg's response is forwarded. A MoRI-IO
/// decode engine without it computes over KV that never arrives (READ) or
/// waits for a push that never comes (WRITE), so a missing, foreign or
/// malformed handoff must fail the request instead. `dp_rank` is the
/// data-parallel rank both legs were pinned to, if any.
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
    // rank the handoff names, and the decode leg goes to the pinned rank.
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
/// concurrent decode leg was sent to notify: that decode engine then waits for
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
        "prefill reports MoRI-IO side channel handshake {} / notify {}, but the decode leg \
         was told {} / {}; set labels {MORIIO_HANDSHAKE_PORT_LABEL} / \
         {MORIIO_NOTIFY_PORT_LABEL} on the prefill worker",
        shown("remote_handshake_port"),
        shown("remote_notify_port"),
        prefill.handshake_port,
        prefill.notify_port
    ))
}

/// Connector id of the engine core serving the prefill leg. With DP the cores
/// suffix the configured id as `{base}_dp{rank}`, so minting needs a pinned
/// rank; unpinned DP>1 yields None (no mint — decode recomputes locally).
pub(crate) fn effective_kv_engine_id(
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
/// Discovered dp_size matters even without `--dp-aware` expansion: a DP>1
/// engine behind an unexpanded worker must not be minted for.
pub(crate) fn connector_mode_for_worker(worker: &dyn Worker) -> KvConnectorMode {
    let meta = worker.metadata();
    let dp_label = meta.spec.labels.get("dp_size");
    let label_dp = dp_label.and_then(|s| s.parse::<usize>().ok().filter(|v| *v > 0));
    let engine_id = if worker.dp_size().is_none() && dp_label.is_some() && label_dp.is_none() {
        // A dp_size label that does not parse as a positive integer means the
        // DP topology is unknown, not absent. Minting an unsuffixed engine id
        // could target the wrong engine core, so fail closed: no mint, decode
        // recomputes the prompt.
        warn!(
            worker = %worker.url(),
            dp_size = ?dp_label,
            "invalid dp_size label; treating DP topology as unknown and \
             skipping KV engine-id minting"
        );
        None
    } else {
        let dp_size = worker.dp_size().or(label_dp);
        effective_kv_engine_id(worker.kv_engine_id().as_deref(), dp_size, worker.dp_rank())
    };
    kv_connector_mode(
        meta.spec.kv_connector.as_deref(),
        &meta.spec.bootstrap_host,
        meta.spec.bootstrap_port,
        engine_id.as_deref(),
    )
}

/// Prefill-leg params for Mooncake: the engine pins blocks under the minted id.
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

/// Decode-leg params for Mooncake, synthesized from prefill worker metadata
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
mod tests {
    use super::*;

    /// The decode leg's bootstrap URL must parse for every spelling the
    /// bootstrap host arrives in: bracketed from a worker URL, bare from a
    /// label or a configured value, IPv4, or a hostname.
    #[test]
    fn mooncake_decode_params_bracket_an_ipv6_bootstrap_host() {
        for (host, expected) in [
            ("fd00::1", "http://[fd00::1]:8998"),
            ("[fd00::1]", "http://[fd00::1]:8998"),
            ("::1", "http://[::1]:8998"),
            ("10.0.0.7", "http://10.0.0.7:8998"),
            ("prefill-host", "http://prefill-host:8998"),
        ] {
            let value: Value =
                serde_json::from_str(&mooncake_decode_params("t", "e", host, 8998)).unwrap();
            assert_eq!(value["remote_bootstrap_addr"], expected, "host {host}");
            let addr = value["remote_bootstrap_addr"].as_str().unwrap();
            let url = url::Url::parse(addr).unwrap_or_else(|e| panic!("host {host}: {e}"));
            assert_eq!(url.port(), Some(8998), "host {host}");
        }
    }

    #[test]
    fn kv_connector_mode_mooncake_uses_bootstrap_metadata() {
        let mode = kv_connector_mode(
            Some(MOONCAKE_CONNECTOR),
            "prefill-host",
            Some(9090),
            Some("engine-1"),
        );
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "prefill-host".to_string(),
                port: 9090,
                engine_id: Some("engine-1".to_string()),
            }
        );
    }

    #[test]
    fn kv_connector_mode_mooncake_defaults_port_and_tolerates_missing_engine_id() {
        let mode = kv_connector_mode(Some(MOONCAKE_CONNECTOR), "prefill-host", None, None);
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "prefill-host".to_string(),
                port: u32::from(DEFAULT_BOOTSTRAP_PORT),
                engine_id: None,
            }
        );
    }

    #[test]
    fn kv_connector_mode_mooncake_empty_engine_id_means_legacy() {
        let mode = kv_connector_mode(Some(MOONCAKE_CONNECTOR), "host", Some(9090), Some(""));
        assert_eq!(
            mode,
            KvConnectorMode::Mooncake {
                host: "host".to_string(),
                port: 9090,
                engine_id: None,
            }
        );
    }

    #[test]
    fn kv_connector_mode_nixl() {
        assert_eq!(
            kv_connector_mode(Some(NIXL_CONNECTOR), "ignored", Some(9090), None),
            KvConnectorMode::Nixl
        );
    }

    fn moriio_worker(url: &str, labels: &[(&str, &str)]) -> crate::worker::BasicWorker {
        let mut builder = crate::worker::BasicWorkerBuilder::new(url)
            .worker_type(crate::worker::WorkerType::Decode)
            .kv_connector(MORIIO_CONNECTOR);
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        builder.build()
    }

    #[test]
    fn moriio_endpoint_defaults_to_the_url_host_and_vllm_ports() {
        let worker = moriio_worker("http://10.0.0.7:8200", &[("moriio_mode", "WRITE")]);
        assert_eq!(
            moriio_endpoint(&worker),
            Ok(MoriIoEndpoint {
                transfer: MoriIoTransfer::Write,
                host: "10.0.0.7".to_string(),
                handshake_port: 6301,
                notify_port: 61005,
                tp_size: None,
                dp_size: 1,
                concurrent_write: false,
            })
        );
        let worker = moriio_worker(
            "http://decode:8200",
            &[
                ("moriio_mode", "read"),
                ("moriio_host", "10.1.1.1"),
                ("moriio_handshake_port", "7301"),
                ("moriio_notify_port", "62005"),
                ("tp_size", "4"),
                ("moriio_write_dispatch", "Concurrent"),
            ],
        );
        assert_eq!(
            moriio_endpoint(&worker),
            Ok(MoriIoEndpoint {
                transfer: MoriIoTransfer::Read,
                host: "10.1.1.1".to_string(),
                handshake_port: 7301,
                notify_port: 62005,
                tp_size: Some(4),
                dp_size: 1,
                concurrent_write: true,
            })
        );
        let worker = moriio_worker(
            "http://decode:8200",
            &[("moriio_mode", "write"), ("dp_size", "8")],
        );
        assert_eq!(moriio_endpoint(&worker).map(|e| e.dp_size), Ok(8));
    }

    #[test]
    fn moriio_endpoint_rejects_missing_mode_dp_and_bad_numbers() {
        for labels in [
            vec![],
            vec![("moriio_mode", "both")],
            vec![("moriio_mode", "read"), ("dp_size", "0")],
            vec![("moriio_mode", "read"), ("dp_size", "one")],
            vec![("moriio_mode", "read"), ("moriio_notify_port", "abc")],
            vec![("moriio_mode", "read"), ("moriio_handshake_port", "0")],
            vec![("moriio_mode", "read"), ("tp_size", "0")],
            vec![
                ("moriio_mode", "write"),
                ("moriio_write_dispatch", "parallel"),
            ],
        ] {
            let worker = moriio_worker("http://decode:8200", &labels);
            assert!(
                moriio_endpoint(&worker).is_err(),
                "{labels:?} must be refused"
            );
        }
        let nixl = crate::worker::BasicWorkerBuilder::new("http://p:1")
            .kv_connector(NIXL_CONNECTOR)
            .label("moriio_mode", "read")
            .build();
        assert!(moriio_endpoint(&nixl).is_err());
        // One worker per rank would let the two legs of a request pick
        // different ranks.
        let per_rank = crate::worker::BasicWorkerBuilder::new("http://decode:8200")
            .worker_type(crate::worker::WorkerType::Decode)
            .kv_connector(MORIIO_CONNECTOR)
            .label("moriio_mode", "write")
            .dp_config(3, 8)
            .build();
        assert!(moriio_endpoint(&per_rank).is_err());
    }

    #[test]
    fn moriio_prefill_params_name_the_decode_side_channel_only_for_write() {
        let mut decode = MoriIoEndpoint {
            transfer: MoriIoTransfer::Read,
            host: "10.1.1.1".to_string(),
            handshake_port: 6301,
            notify_port: 61005,
            tp_size: Some(4),
            dp_size: 1,
            concurrent_write: false,
        };
        let read: Value =
            serde_json::from_str(&moriio_prefill_params("tx-1", &decode, None)).unwrap();
        assert_eq!(
            read,
            serde_json::json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "remote_engine_id": null,
                "remote_block_ids": null,
                "transfer_id": "tx-1",
                "remote_dp_size": 1,
                "remote_tp_size": 4,
            })
        );
        decode.transfer = MoriIoTransfer::Write;
        let write: Value =
            serde_json::from_str(&moriio_prefill_params("tx-2", &decode, None)).unwrap();
        assert_eq!(
            write,
            serde_json::json!({
                "do_remote_decode": true,
                "do_remote_prefill": false,
                "remote_engine_id": null,
                "remote_block_ids": null,
                "transfer_id": "tx-2",
                "remote_dp_size": 1,
                "remote_tp_size": 4,
                "remote_host": "10.1.1.1",
                "remote_handshake_port": 6301,
                "remote_notify_port": 61005,
            })
        );
        // An unlabeled TP stays unknown: the connector then assumes the
        // decode matches the prefill instead of collapsing onto rank 0.
        decode.tp_size = None;
        let unlabeled: Value =
            serde_json::from_str(&moriio_prefill_params("tx-3", &decode, None)).unwrap();
        assert_eq!(unlabeled.get("remote_tp_size"), None);
        decode.dp_size = 8;
        let pinned: Value =
            serde_json::from_str(&moriio_prefill_params("tx-4", &decode, Some(3))).unwrap();
        assert_eq!(pinned["remote_dp_size"], 8);
        assert_eq!(pinned["remote_dp_rank"], 3);
    }

    #[test]
    fn moriio_write_target_must_be_another_engine_the_prefill_can_reach() {
        let endpoint = |host: &str, handshake_port: u16, notify_port: u16| MoriIoEndpoint {
            transfer: MoriIoTransfer::Write,
            host: host.to_string(),
            handshake_port,
            notify_port,
            tp_size: None,
            dp_size: 1,
            concurrent_write: false,
        };
        let remote = endpoint("10.0.0.1", 6301, 61005);
        for (prefill, decode, refused) in [
            (&remote, endpoint("10.0.0.2", 6301, 61005), false),
            (&remote, endpoint("127.0.0.1", 6301, 61005), true),
            (&remote, endpoint("localhost", 7301, 62005), true),
            (&remote, endpoint("[::1]", 7301, 62005), true),
            (&remote, endpoint("0.0.0.0", 7301, 62005), true),
            (&remote, endpoint("[::]", 7301, 62005), true),
            (&remote, endpoint("10.0.0.1", 6301, 62005), true),
            (&remote, endpoint("10.0.0.1", 7301, 61005), true),
            (&remote, endpoint("10.0.0.1", 7301, 62005), false),
            (
                &endpoint("localhost", 6301, 61005),
                endpoint("127.0.0.1", 7301, 62005),
                false,
            ),
            (
                &endpoint("localhost", 6301, 61005),
                endpoint("127.0.0.1", 6301, 62005),
                true,
            ),
            // A sequential decode leg reaches the prefill at the side channel
            // the prefill reports in its handoff ...
            (
                &endpoint("127.0.0.1", 6301, 61005),
                endpoint("10.0.0.2", 6301, 61005),
                false,
            ),
            // ... a concurrent one at the side channel named here.
            (
                &endpoint("127.0.0.1", 6301, 61005),
                MoriIoEndpoint {
                    concurrent_write: true,
                    ..endpoint("10.0.0.2", 6301, 61005)
                },
                true,
            ),
            (
                &endpoint("0.0.0.0", 6301, 61005),
                MoriIoEndpoint {
                    concurrent_write: true,
                    ..endpoint("10.0.0.2", 6301, 61005)
                },
                true,
            ),
            (
                &endpoint("0.0.0.0", 6301, 61005),
                endpoint("10.0.0.2", 6301, 61005),
                false,
            ),
        ] {
            assert_eq!(
                moriio_write_target_error(prefill, &decode).is_some(),
                refused,
                "{prefill:?} -> {decode:?}"
            );
        }
    }

    #[test]
    fn moriio_decode_params_name_the_prefill_side_channel() {
        let mut prefill = MoriIoEndpoint {
            transfer: MoriIoTransfer::Write,
            host: "10.0.0.1".to_string(),
            handshake_port: 7301,
            notify_port: 62005,
            tp_size: None,
            dp_size: 1,
            concurrent_write: false,
        };
        let params: Value =
            serde_json::from_str(&moriio_decode_params("tx-1", &prefill, None)).unwrap();
        assert_eq!(
            params,
            serde_json::json!({
                "do_remote_decode": false,
                "do_remote_prefill": true,
                "remote_engine_id": null,
                "remote_block_ids": null,
                "transfer_id": "tx-1",
                "remote_dp_size": 1,
                "remote_host": "10.0.0.1",
                "remote_handshake_port": 7301,
                "remote_notify_port": 62005,
            })
        );
        prefill.tp_size = Some(8);
        let labeled: Value =
            serde_json::from_str(&moriio_decode_params("tx-2", &prefill, None)).unwrap();
        assert_eq!(labeled.get("remote_tp_size"), Some(&Value::from(8)));
        assert_eq!(labeled.get("remote_dp_rank"), None);
        prefill.dp_size = 8;
        let pinned: Value =
            serde_json::from_str(&moriio_decode_params("tx-3", &prefill, Some(5))).unwrap();
        assert_eq!(pinned["remote_dp_size"], 8);
        assert_eq!(pinned["remote_dp_rank"], 5);
    }

    #[test]
    fn moriio_pair_needs_one_dp_size_for_both_legs() {
        let endpoint = |dp_size| MoriIoEndpoint {
            transfer: MoriIoTransfer::Write,
            host: "10.0.0.1".to_string(),
            handshake_port: 6301,
            notify_port: 61005,
            tp_size: None,
            dp_size,
            concurrent_write: false,
        };
        assert_eq!(moriio_dp_mismatch_error(&endpoint(1), &endpoint(1)), None);
        assert_eq!(moriio_dp_mismatch_error(&endpoint(8), &endpoint(8)), None);
        assert!(moriio_dp_mismatch_error(&endpoint(8), &endpoint(1)).is_some());
        assert!(moriio_dp_mismatch_error(&endpoint(1), &endpoint(8)).is_some());
    }

    #[test]
    fn moriio_handoff_must_name_the_side_channel_the_decode_was_told() {
        let prefill = MoriIoEndpoint {
            transfer: MoriIoTransfer::Write,
            host: "10.0.0.1".to_string(),
            handshake_port: 6301,
            notify_port: 61005,
            tp_size: None,
            dp_size: 1,
            concurrent_write: false,
        };
        let handoff = |handshake: Value, notify: Value| {
            let params = serde_json::json!({
                "do_remote_prefill": true, "remote_host": "10.0.0.1",
                "remote_handshake_port": handshake, "remote_notify_port": notify,
            });
            RawValue::from_string(params.to_string()).unwrap()
        };
        for (handshake, notify, mismatch) in [
            // A vLLM 0.30.1rc1 producer reports its configured ports as strings,
            // the same at TP1 and TP8.
            (Value::from("6301"), Value::from("61005"), false),
            (Value::from(6301), Value::from(61005), false),
            (Value::from("6301"), Value::from("61006"), true),
            (Value::from("7301"), Value::from("61005"), true),
            (Value::from("6301"), Value::from("abc"), true),
            (Value::from("6301"), Value::Null, true),
        ] {
            let error =
                moriio_side_channel_error(&handoff(handshake.clone(), notify.clone()), &prefill);
            assert_eq!(
                error.is_some(),
                mismatch,
                "{handshake} / {notify}: {error:?}"
            );
        }
    }

    #[test]
    fn moriio_handoff_must_carry_the_minted_id_and_the_prefill_peer() {
        // Shape returned by a vLLM 0.30.1rc1 MoRIIOConnector producer.
        let valid = serde_json::json!({
            "do_remote_prefill": true, "do_remote_decode": false,
            "remote_block_ids": [[484, 485]], "remote_engine_id": "10.24.112.125:6301",
            "remote_host": "10.24.112.125", "remote_handshake_port": "6301",
            "remote_notify_port": "61005", "remote_dp_rank": 0,
            "remote_dp_rank_override": true, "remote_dp_size": 1, "tp_size": 1,
            "transfer_id": "tx-abc",
        });
        let raw = |v: &Value| RawValue::from_string(v.to_string()).unwrap();
        assert_eq!(
            validate_moriio_handoff(&raw(&valid), "tx-abc", None),
            Ok(())
        );
        assert!(validate_moriio_handoff(&raw(&valid), "tx-other", None).is_err());
        for (key, value) in [
            ("remote_host", Value::Null),
            ("remote_block_ids", Value::Null),
            ("do_remote_prefill", Value::Bool(false)),
        ] {
            let mut broken = valid.clone();
            broken[key] = value;
            assert!(
                validate_moriio_handoff(&raw(&broken), "tx-abc", None).is_err(),
                "{key} must be required"
            );
        }
        let mut no_port = valid.clone();
        no_port
            .as_object_mut()
            .unwrap()
            .remove("remote_notify_port");
        assert!(validate_moriio_handoff(&raw(&no_port), "tx-abc", None).is_err());
    }

    #[test]
    fn moriio_handoff_must_come_from_the_pinned_dp_rank() {
        let handoff = |rank: Option<Value>| {
            let mut params = serde_json::json!({
                "do_remote_prefill": true, "do_remote_decode": false,
                "remote_block_ids": [[484, 485]], "remote_engine_id": "10.24.112.125:6301",
                "remote_host": "10.24.112.125", "remote_handshake_port": "6301",
                "remote_notify_port": "61005", "remote_dp_rank_override": true,
                "remote_dp_size": 8, "tp_size": 1, "transfer_id": "tx-abc",
            });
            if let Some(rank) = rank {
                params["remote_dp_rank"] = rank;
            }
            RawValue::from_string(params.to_string()).unwrap()
        };
        for (rank, pinned, accepted) in [
            (Some(Value::from(3)), Some(3), true),
            (Some(Value::from(3)), None, true),
            (None, None, true),
            (Some(Value::from(0)), Some(3), false),
            (None, Some(3), false),
            (Some(Value::Null), Some(0), false),
        ] {
            assert_eq!(
                validate_moriio_handoff(&handoff(rank.clone()), "tx-abc", pinned).is_ok(),
                accepted,
                "remote_dp_rank {rank:?} pinned to {pinned:?}"
            );
        }
    }

    #[test]
    fn moriio_handoff_fields_must_have_the_shape_the_decode_engine_parses() {
        let valid = serde_json::json!({
            "do_remote_prefill": true, "do_remote_decode": false,
            "remote_block_ids": [[484, 485]], "remote_engine_id": "10.24.112.125:6301",
            "remote_host": "10.24.112.125", "remote_handshake_port": "6301",
            "remote_notify_port": "61005", "transfer_id": "tx-abc",
        });
        let check = |key: &str, value: &Value| {
            let mut handoff = valid.clone();
            handoff[key] = value.clone();
            validate_moriio_handoff(
                &RawValue::from_string(handoff.to_string()).unwrap(),
                "tx-abc",
                None,
            )
        };
        for (key, value) in [
            ("remote_engine_id", serde_json::json!(42)),
            ("remote_engine_id", serde_json::json!("")),
            (
                "remote_engine_id",
                serde_json::json!(["10.24.112.125:6301"]),
            ),
            ("remote_block_ids", serde_json::json!("484,485")),
            ("remote_block_ids", serde_json::json!({"0": [484]})),
            ("remote_block_ids", serde_json::json!([["484"]])),
            ("remote_block_ids", serde_json::json!([[-1]])),
            ("remote_block_ids", serde_json::json!([[1.5]])),
            ("remote_block_ids", serde_json::json!([484, [485]])),
            // vLLM pairs the groups with its own: a flat list raises there.
            ("remote_block_ids", serde_json::json!([484, 485])),
            ("remote_block_ids", serde_json::json!([])),
            ("remote_block_ids", serde_json::json!([[]])),
            ("remote_host", serde_json::json!("")),
            ("remote_host", serde_json::json!(42)),
            ("remote_host", serde_json::json!("10.24.112.125 ")),
            ("remote_handshake_port", serde_json::json!("abc")),
            ("remote_handshake_port", serde_json::json!(0)),
            ("remote_notify_port", serde_json::json!(70000)),
            ("remote_notify_port", serde_json::json!(true)),
            ("remote_dp_size", serde_json::json!("abc")),
            ("remote_dp_size_local", serde_json::json!(-1)),
            ("remote_dp_rank", serde_json::json!([0])),
            // A string rank never equals the decode engine's own rank.
            ("remote_dp_rank", serde_json::json!("0")),
            ("remote_dp_size", serde_json::json!("1")),
            ("tp_size", serde_json::json!("two")),
            ("tp_size", serde_json::json!("8")),
            ("remote_tp_size", serde_json::json!(1.5)),
        ] {
            assert!(check(key, &value).is_err(), "{key}={value} must be refused");
        }
        // Also valid: a group without blocks next to one with blocks, numeric
        // ports, and any host the decode engine may dial. The host is not
        // compared with the labels: it can be a name, or an address on
        // another interface.
        for (key, value) in [
            ("remote_block_ids", serde_json::json!([[484], []])),
            ("remote_handshake_port", serde_json::json!(6301)),
            ("remote_host", serde_json::json!("prefill-0.pd.svc")),
            ("remote_host", serde_json::json!("10.101.31.101")),
            ("remote_dp_size", serde_json::json!(1)),
            ("remote_dp_size_local", serde_json::json!(0)),
            ("remote_dp_rank", serde_json::json!(0)),
            ("tp_size", serde_json::json!(8)),
            ("remote_tp_size", Value::Null),
        ] {
            assert_eq!(check(key, &value), Ok(()), "{key}={value} must be accepted");
        }
    }

    #[test]
    fn kv_connector_mode_unknown_or_missing_is_passthrough() {
        assert_eq!(
            kv_connector_mode(Some("LMCacheConnector"), "host", None, None),
            KvConnectorMode::Passthrough
        );
        assert_eq!(
            kv_connector_mode(None, "host", None, None),
            KvConnectorMode::Passthrough
        );
    }

    #[test]
    fn invalid_dp_size_label_fails_closed_on_minting() {
        use crate::worker::{BasicWorkerBuilder, WorkerType};

        let worker = BasicWorkerBuilder::new("http://prefill:8000")
            .worker_type(WorkerType::Prefill)
            .kv_connector(MOONCAKE_CONNECTOR)
            .kv_engine_id("eng")
            .label("dp_size", "not-a-number")
            .build();
        let mode = connector_mode_for_worker(&worker);
        // Unknown DP topology must not mint an unsuffixed engine id.
        assert!(matches!(
            mode,
            KvConnectorMode::Mooncake {
                engine_id: None,
                ..
            }
        ));

        let worker = BasicWorkerBuilder::new("http://prefill:8000")
            .worker_type(WorkerType::Prefill)
            .kv_connector(MOONCAKE_CONNECTOR)
            .kv_engine_id("eng")
            .label("dp_size", "1")
            .build();
        let mode = connector_mode_for_worker(&worker);
        assert!(matches!(
            mode,
            KvConnectorMode::Mooncake {
                engine_id: Some(ref id),
                ..
            } if id == "eng"
        ));
    }

    #[test]
    fn effective_engine_id_requires_pinned_rank_under_dp() {
        assert_eq!(
            effective_kv_engine_id(Some("eng"), Some(2), Some(1)),
            Some("eng_dp1".to_string())
        );
        assert_eq!(effective_kv_engine_id(Some("eng"), Some(2), None), None);
        assert_eq!(
            effective_kv_engine_id(Some("eng"), None, None),
            Some("eng".to_string())
        );
        assert_eq!(effective_kv_engine_id(Some(""), None, None), None);
        assert_eq!(effective_kv_engine_id(None, Some(2), Some(0)), None);
    }

    #[test]
    fn a_refreshed_engine_id_is_what_the_handoff_is_minted_for() {
        use crate::worker::{BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerType};

        let worker = BasicWorkerBuilder::new("grpc://p:1")
            .worker_type(WorkerType::Prefill)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .kv_connector("MooncakeConnector")
            .kv_engine_id("eng-old")
            .bootstrap_port(Some(8998))
            .build();
        assert!(matches!(
            connector_mode_for_worker(&worker),
            KvConnectorMode::Mooncake { engine_id: Some(ref id), .. } if id == "eng-old"
        ));
        assert!(worker.refresh_kv_engine_id(Some("eng-new".to_string())));
        assert!(!worker.refresh_kv_engine_id(Some("eng-new".to_string())));
        assert!(matches!(
            connector_mode_for_worker(&worker),
            KvConnectorMode::Mooncake { engine_id: Some(ref id), .. } if id == "eng-new"
        ));
    }
}
