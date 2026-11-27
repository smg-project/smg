use smg_worker::identity::basic::{BasicWorker, WorkerType};

use super::*;
use crate::routers::grpc::backend_client::BasicWorkerBuilder;

/// The decode side's bootstrap URL must parse for every spelling the
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

fn moriio_worker(url: &str, labels: &[(&str, &str)]) -> BasicWorker {
    let mut builder = BasicWorkerBuilder::new(url)
        .worker_type(WorkerType::Decode)
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
    let nixl = BasicWorkerBuilder::new("http://p:1")
        .kv_connector(NIXL_CONNECTOR)
        .label("moriio_mode", "read")
        .build();
    assert!(moriio_endpoint(&nixl).is_err());
    // One worker per rank would let the two sides of a request pick
    // different ranks.
    let per_rank = BasicWorkerBuilder::new("http://decode:8200")
        .worker_type(WorkerType::Decode)
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
    let read: Value = serde_json::from_str(&moriio_prefill_params("tx-1", &decode, None)).unwrap();
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
    let write: Value = serde_json::from_str(&moriio_prefill_params("tx-2", &decode, None)).unwrap();
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
        // A sequential decode side reaches the prefill at the side channel
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
fn moriio_pair_needs_one_dp_size_for_both_sides() {
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

/// A store connector takes the store mode whatever bootstrap metadata the
/// side carries: nothing is minted from it.
#[test]
fn kv_connector_mode_store_connectors_take_the_store_mode() {
    for connector in [
        "MooncakeStoreConnector",
        "SharedStorageConnector",
        "FlexKVConnectorV1",
        "HF3FSKVConnector",
    ] {
        assert_eq!(
            kv_connector_mode(
                Some(connector),
                "prefill-host",
                Some(8998),
                Some("engine-1")
            ),
            KvConnectorMode::Store,
            "{connector}"
        );
    }
    assert_eq!(KvConnectorMode::Store.metrics_label(), "store");
}

/// Unknown connectors, and the connectors whose blocks stay in the engine
/// or whose disaggregated mode the request would have to address, keep
/// the passthrough handling.
#[test]
fn kv_connector_mode_unknown_or_missing_is_passthrough() {
    for connector in [
        "P2pNcclConnector",
        "OffloadingConnector",
        "SimpleCPUOffloadConnector",
        "LMCacheConnectorV1",
        "LMCacheMPConnector",
    ] {
        assert_eq!(
            kv_connector_mode(Some(connector), "host", None, None),
            KvConnectorMode::Passthrough,
            "{connector}"
        );
    }
    assert_eq!(
        kv_connector_mode(None, "host", None, None),
        KvConnectorMode::Passthrough
    );
}

#[test]
fn invalid_dp_size_label_fails_closed_on_minting() {
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
    use smg_worker::identity::basic::{ConnectionMode, RuntimeType};

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
