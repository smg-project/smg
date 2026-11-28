use std::path::Path;

use super::*;
use crate::testing::TestHost;

fn parse(json: &str, roles: Roles) -> Result<Vec<DiscoveredWorker>, String> {
    parse_manifest(json.as_bytes(), roles)
}

fn parse_err(json: &str, roles: Roles) -> String {
    match parse(json, roles) {
        Ok(workers) => panic!("expected a rejection, parsed {workers:?}"),
        Err(reason) => reason,
    }
}

fn manifest(workers: &str) -> String {
    format!(r#"{{"version": 1, "workers": [{workers}]}}"#)
}

#[test]
fn every_field_reaches_the_record() {
    let workers = parse(
        &manifest(
            r#"{"id": "prefill-a", "url": "10.0.0.22:8000", "worker_type": "prefill",
                    "bootstrap_port": 8998, "model_id": "llama", "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer", "kv_engine_id": "engine-a"}"#,
        ),
        Roles::PrefillDecode,
    )
    .unwrap();
    assert_eq!(workers.len(), 1);
    let worker = &workers[0];
    assert_eq!(worker.discovery_id, "prefill-a");
    assert_eq!(worker.endpoint.render(), "10.0.0.22:8000");
    assert_eq!(worker.worker_type, WorkerType::Prefill);
    assert_eq!(worker.bootstrap_port, Some(8998));
    assert_eq!(worker.model_id_override.as_deref(), Some("llama"));
    assert_eq!(worker.kv_connector.as_deref(), Some("MooncakeConnector"));
    assert_eq!(worker.kv_role.as_deref(), Some("kv_producer"));
    assert_eq!(worker.kv_engine_id.as_deref(), Some("engine-a"));
    assert!(worker.compat_labels.is_empty());
}

/// Every EPD role in one manifest, over IPv4, IPv6, gRPC and IPC.
#[test]
fn all_worker_types_and_endpoint_forms_parse() {
    let workers = parse(
        &manifest(
            r#"{"url": "10.0.0.21:8000", "worker_type": "encode", "bootstrap_port": 8998},
                   {"url": "[2001:db8::1]:8000", "worker_type": "prefill"},
                   {"url": "grpc://decode.internal:9000", "worker_type": "decode"},
                   {"url": "ipc:///tmp/engine.sock", "worker_type": "decode"}"#,
        ),
        Roles::EncodePrefillDecode,
    )
    .unwrap();
    let types: Vec<WorkerType> = workers.iter().map(|w| w.worker_type).collect();
    assert_eq!(
        types,
        [
            WorkerType::Encode,
            WorkerType::Prefill,
            WorkerType::Decode,
            WorkerType::Decode
        ]
    );
}

#[test]
fn an_empty_fleet_is_valid() {
    assert!(parse(&manifest(""), Roles::Regular).unwrap().is_empty());
}

/// Without an id the endpoint names the instance, rendered with any
/// userinfo masked: the id is written to a label. Two entries at one
/// address that differ only in their credentials need explicit ids.
#[test]
fn a_missing_id_defaults_to_the_redacted_endpoint() {
    let workers = parse(
        &manifest(r#"{"url": "http://admin:hunter2@10.0.0.11:8000"}"#),
        Roles::Regular,
    )
    .unwrap();
    assert_eq!(workers[0].discovery_id, "http://***@10.0.0.11:8000");
}

#[test]
fn regular_routing_defaults_the_worker_type() {
    let workers = parse(&manifest(r#"{"url": "10.0.0.11:8000"}"#), Roles::Regular).unwrap();
    assert_eq!(workers[0].worker_type, WorkerType::Regular);
}

/// Each malformed manifest is rejected whole, with a reason that names
/// the problem.
#[test]
fn malformed_manifests_are_rejected_with_the_reason() {
    let cases: [(&str, String, Roles, &str); 18] = [
        ("bad JSON", "{".to_string(), Roles::Regular, "EOF"),
        (
            "missing version",
            r#"{"workers": []}"#.to_string(),
            Roles::Regular,
            "missing field `version`",
        ),
        (
            "unsupported version",
            r#"{"version": 2, "workers": []}"#.to_string(),
            Roles::Regular,
            "unsupported version 2",
        ),
        (
            "unknown top-level field",
            r#"{"version": 1, "workers": [], "worker": []}"#.to_string(),
            Roles::Regular,
            "unknown field `worker`",
        ),
        (
            "misspelled worker field",
            manifest(r#"{"url": "10.0.0.11:8000", "worker_typ": "regular"}"#),
            Roles::Regular,
            "unknown field `worker_typ`",
        ),
        (
            "unknown worker type",
            manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "router"}"#),
            Roles::Regular,
            "unknown variant `router`",
        ),
        (
            "type the routing mode does not use",
            manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "prefill"}"#),
            Roles::Regular,
            "'prefill' is not used in regular routing",
        ),
        (
            "encode in prefill-decode routing",
            manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "encode"}"#),
            Roles::PrefillDecode,
            "'encode' is not used in prefill-decode routing",
        ),
        (
            "no type in disaggregated routing",
            manifest(r#"{"url": "10.0.0.11:8000"}"#),
            Roles::PrefillDecode,
            "worker_type is required in prefill-decode routing",
        ),
        (
            "bootstrap port 0",
            manifest(r#"{"url": "10.0.0.11:8000", "worker_type": "prefill", "bootstrap_port": 0}"#),
            Roles::PrefillDecode,
            "bootstrap_port must be between 1 and 65535",
        ),
        (
            "bootstrap port on a decode worker",
            manifest(
                r#"{"url": "10.0.0.11:8000", "worker_type": "decode", "bootstrap_port": 8998}"#,
            ),
            Roles::PrefillDecode,
            "bootstrap_port applies only to encode and prefill workers, not decode",
        ),
        (
            "rank suffix",
            manifest(r#"{"url": "10.0.0.11:8000@1"}"#),
            Roles::Regular,
            "the gateway reads them as a rank suffix",
        ),
        (
            "unparsable url",
            manifest(r#"{"url": "10.0.0.11:http"}"#),
            Roles::Regular,
            "workers[0]: invalid url",
        ),
        (
            "empty id",
            manifest(r#"{"id": "", "url": "10.0.0.11:8000"}"#),
            Roles::Regular,
            "id must not be empty",
        ),
        (
            "duplicate id",
            manifest(
                r#"{"id": "a", "url": "10.0.0.11:8000"}, {"id": "a", "url": "10.0.0.12:8000"}"#,
            ),
            Roles::Regular,
            "workers[1]: duplicate id 'a'",
        ),
        (
            "the same id-less entry twice",
            manifest(r#"{"url": "10.0.0.11:8000"}, {"url": "10.0.0.11:8000"}"#),
            Roles::Regular,
            "workers[1]: duplicate id '10.0.0.11:8000'",
        ),
        (
            "two id-less entries at one address that differ only in their credentials",
            manifest(
                r#"{"url": "http://alice:x@10.0.0.11:8000"}, {"url": "http://bob:y@10.0.0.11:8000"}"#,
            ),
            Roles::Regular,
            "workers[1]: duplicate id 'http://***@10.0.0.11:8000' (the default id, the address \
                 with its userinfo masked; set an explicit id)",
        ),
        (
            "two spellings of one endpoint",
            manifest(
                r#"{"id": "a", "url": "10.0.0.11:8000"}, {"id": "b", "url": "http://10.0.0.11:8000"}"#,
            ),
            Roles::Regular,
            "'b' and 'a' name the same endpoint",
        ),
    ];
    for (case, json, roles, expected) in cases {
        let reason = parse_err(&json, roles);
        assert!(
            reason.contains(expected),
            "{case}: expected '{expected}' in '{reason}'"
        );
    }
    // Exact: the plain duplicate carries no default-id hint (the table's
    // `contains` would accept the hint's longer message as well).
    assert_eq!(
        parse_err(
            &manifest(r#"{"url": "10.0.0.11:8000"}, {"url": "10.0.0.11:8000"}"#),
            Roles::Regular,
        ),
        "workers[1]: duplicate id '10.0.0.11:8000'",
    );
}

/// The worker URL door is asked before the reader's own parse: an
/// address the worker API refuses is refused here in the door's words,
/// which name no part of it behind its scheme. The parse names the host
/// or the port as it reads them: it read the password of the second
/// spelling as a port, and it took the first, which is stored and named
/// without its slash.
#[test]
fn an_address_the_worker_url_door_refuses_is_refused_with_no_part_named() {
    let entry = |url: &str| manifest(&format!(r#"{{"url": "{url}"}}"#));
    for url in [
        "http://adm1n:s3cret@2130706433/",
        "http://adm1n:s3cret@2130706433",
        "http://adm1n:s3cret@99999999999",
        "adm1n:s3cret@2130706433/",
        "adm1n:s3cret@2130706433",
        "http://s3cret@2130706433",
    ] {
        let reason = parse_err(&entry(url), Roles::Regular);
        assert!(
            reason.starts_with("workers[0]: invalid url: "),
            "{url}: {reason}"
        );
        for part in ["adm1n", "s3cret", "2130706433", "99999999999"] {
            assert!(
                !reason.contains(part),
                "{url}: the refusal names '{part}': {reason}"
            );
        }
    }
    assert_eq!(
        parse_err(&entry("http://adm1n:s3cret@2130706433"), Roles::Regular),
        "workers[0]: invalid url: the gateway cannot read this address as a host and a port, \
             though a URL client can: write it as scheme://host:port, with a port from 1 to 65535",
    );

    // The controls: a userinfo before a host and a port, a plain address,
    // a known scheme in upper case, an address with no scheme, and one
    // with blanks around it.
    for (url, stored) in [
        (
            "http://adm1n:s3cret@10.0.0.11:8000",
            "http://adm1n:s3cret@10.0.0.11:8000",
        ),
        ("http://10.0.0.11:8000", "http://10.0.0.11:8000"),
        ("HTTP://10.0.0.11:8000", "http://10.0.0.11:8000"),
        ("adm1n:s3cret@10.0.0.11:8000", "adm1n:s3cret@10.0.0.11:8000"),
        (" 10.0.0.11:8000 ", "10.0.0.11:8000"),
    ] {
        let workers = parse(&entry(url), Roles::Regular).unwrap();
        assert_eq!(workers[0].endpoint.render(), stored, "{url}");
    }
}

/// The door is asked of the address as it is kept, too. The registration
/// is handed the endpoint as it renders, without a path's trailing
/// slashes, and the door reads an address less its slashes under `http`
/// and `https` alone. The first spelling is a userinfo and a host as
/// written, and a host and its rank as kept: it was listed, and named
/// with the token as its host.
#[test]
fn an_address_the_door_refuses_as_it_is_kept_is_refused_with_no_part_named() {
    let entry = |url: &str| manifest(&format!(r#"{{"url": "{url}"}}"#));
    for (url, reason) in [
        (
            "grpc://s3cret-token@2130706433/",
            "the gateway reads them as a rank suffix",
        ),
        (
            "GRPC://s3cret-token@2130706433//",
            "the gateway reads them as a rank suffix",
        ),
        (
            "grpc://adm1n:s3cret@2130706433/",
            "the gateway cannot read this address as a host and a port",
        ),
        (
            "grpc://decode.internal\u{a0}/",
            "the address reads two ways",
        ),
        ("tcp://decode.internal\u{a0}/", "the address reads two ways"),
    ] {
        let refusal = parse_err(&entry(url), Roles::Regular);
        assert!(
            refusal.starts_with("workers[0]: invalid url: ") && refusal.contains(reason),
            "{url}: {refusal}"
        );
        for part in ["adm1n", "s3cret", "2130706433", "decode"] {
            assert!(
                !refusal.contains(part),
                "{url}: the refusal names '{part}': {refusal}"
            );
        }
    }

    // The controls: a trailing slash refuses nothing by itself, and the
    // address is kept without it.
    for (url, stored) in [
        (
            "grpc://decode.internal:9000/",
            "grpc://decode.internal:9000",
        ),
        (
            "grpc://adm1n:s3cret@10.0.0.11:9000/",
            "grpc://adm1n:s3cret@10.0.0.11:9000",
        ),
        ("http://10.0.0.11:8000/", "http://10.0.0.11:8000"),
        ("tcp://10.0.0.11:5555/", "tcp://10.0.0.11:5555"),
    ] {
        let workers = parse(&entry(url), Roles::Regular).unwrap();
        assert_eq!(workers[0].endpoint.render(), stored, "{url}");
    }
}

/// A scheme is a known one at the start of an address, as the address
/// type reads it and the registration keeps it. An address with none
/// may hold a `://` further in, in a path, in a query or behind a
/// userinfo: it is listed and kept as written. Read as the end of a
/// scheme, that `://` had the door refuse the address, and the manifest
/// whole. A scheme the gateway does not know is no scheme: the address
/// is read as one with none, and refused in the door's words, which
/// name no part of it.
#[test]
fn a_scheme_is_read_at_the_start_of_a_listed_address_alone() {
    let entry = |url: &str| manifest(&format!(r#"{{"url": "{url}"}}"#));
    for url in [
        "10.0.0.11:8000/a://b",
        "10.0.0.11:8000/proxy/http://backend:9000",
        "decode.internal:8000/v1?next=http://x",
        "adm1n:s3cret@10.0.0.11:8000/a://b",
        "http://10.0.0.11:8000/a://b",
    ] {
        let workers = parse(&entry(url), Roles::Regular).unwrap();
        assert_eq!(workers[0].endpoint.render(), url, "{url}");
    }

    for (url, reason) in [
        (
            "ftp://10.0.0.11:8000",
            "the gateway cannot read this address as a host and a port",
        ),
        (
            "S3CRET.Token+1://10.0.0.11:8000",
            "the gateway cannot read this address as a host and a port",
        ),
        (
            "adm1n:s3cr://et@10.0.0.11:8000",
            "Invalid URL format: invalid port number",
        ),
    ] {
        let refusal = parse_err(&entry(url), Roles::Regular);
        assert!(
            refusal.starts_with("workers[0]: invalid url: ") && refusal.contains(reason),
            "{url}: {refusal}"
        );
        let lowered = refusal.to_ascii_lowercase();
        for part in ["ftp", "s3cr", "token", "adm1n", "10.0.0.11"] {
            assert!(
                !lowered.contains(part),
                "{url}: the refusal names '{part}': {refusal}"
            );
        }
    }
}

#[test]
fn modes_without_worker_pools_cannot_use_file_discovery() {
    assert!(
        FileProviderConfig::new(PathBuf::from("w.json"), Duration::from_secs(1), None).is_err()
    );
}

#[tokio::test]
async fn an_absent_manifest_is_unreadable_not_empty() {
    let dir = tempfile::tempdir().unwrap();
    let config = FileProviderConfig::new(
        dir.path().join("workers.json"),
        Duration::from_secs(1),
        Some(Roles::Regular),
    )
    .unwrap();
    let err = read_manifest(&config).await.unwrap_err();
    assert!(matches!(err, ManifestError::Unreadable { .. }), "{err}");
    assert_eq!(err.metric_reason(), series::SNAPSHOT_ERROR_UNREADABLE);
}

fn discovery(path: &Path) -> FileDiscovery {
    discovery_checking_every(path, Duration::from_secs(60))
}

fn discovery_checking_every(path: &Path, check_interval: Duration) -> FileDiscovery {
    let config =
        FileProviderConfig::new(path.to_path_buf(), check_interval, Some(Roles::Regular)).unwrap();
    FileDiscovery::new(config, TestHost::without_a_queue())
}

/// The endpoints of the last good snapshot, `None` before the first good
/// read.
fn listed(discovery: &FileDiscovery) -> Option<Vec<String>> {
    discovery.last_good.desired().map(|desired| {
        desired
            .workers
            .values()
            .map(|worker| worker.endpoint.render())
            .collect()
    })
}

/// A read that fails leaves the last good snapshot in place, so a
/// half-written, broken or missing manifest never removes a worker; a
/// valid empty fleet does replace it.
#[tokio::test]
async fn a_failed_read_keeps_the_last_good_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let mut discovery = discovery(&path);

    discovery.reread().await;
    assert_eq!(listed(&discovery), None, "nothing read yet");

    std::fs::write(&path, manifest(r#"{"url": "10.0.0.11:8000"}"#)).unwrap();
    discovery.reread().await;
    assert_eq!(listed(&discovery), Some(vec!["10.0.0.11:8000".to_string()]));

    std::fs::write(&path, "{").unwrap();
    discovery.reread().await;
    assert_eq!(
        listed(&discovery),
        Some(vec!["10.0.0.11:8000".to_string()]),
        "a half-written manifest replaced the snapshot"
    );

    std::fs::remove_file(&path).unwrap();
    discovery.reread().await;
    assert_eq!(
        listed(&discovery),
        Some(vec!["10.0.0.11:8000".to_string()]),
        "a missing manifest replaced the snapshot"
    );

    std::fs::write(&path, manifest("")).unwrap();
    discovery.reread().await;
    assert_eq!(listed(&discovery), Some(vec![]), "an empty fleet is valid");
}

/// One entry the worker URL door refuses refuses its manifest whole, as
/// any entry the reader cannot take does: at a first read the worker
/// listed beside it is not listed either. The warning names the entry's
/// index and the door's reason, and no line names a part of its address.
#[tokio::test]
#[tracing_test::traced_test]
async fn one_entry_the_door_refuses_refuses_its_manifest_whole() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.json");
    let mut discovery = discovery(&path);
    for (refused, reason) in [
        (
            "grpcs://127.0.0.1:9000",
            "grpcs:// workers are not supported",
        ),
        (
            "tcp://adm1n:s3cret@127.0.0.1:31000",
            "a tcp:// worker URL takes no userinfo",
        ),
    ] {
        let workers = format!(r#"{{"url": "10.0.0.11:8000"}}, {{"url": "{refused}"}}"#);
        std::fs::write(&path, manifest(&workers)).unwrap();
        discovery.reread().await;
        assert_eq!(listed(&discovery), None, "{refused}: a worker is listed");
        assert!(
            logs_contain(&format!("workers[1]: invalid url: {reason}")),
            "{refused}: no warning names the entry and the door's reason"
        );
    }
    for part in ["adm1n", "s3cret", "127.0.0.1"] {
        assert!(!logs_contain(part), "a line names '{part}'");
    }
}

/// Until the manifest's directory exists the provider polls only: the
/// watch is retried after the one warning, and starts once the directory
/// is there.
#[tokio::test]
async fn an_unwatchable_directory_leaves_the_provider_polling() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("smg");
    let mut discovery = discovery(&dir.join("workers.json"));
    let mut timers = Timers::new(discovery.config.check_interval);

    discovery.start_watch(&mut timers);
    assert!(discovery.watch.is_none());
    assert!(discovery.warned, "the failure is logged");
    discovery.start_watch(&mut timers);
    assert!(discovery.watch.is_none());
    assert!(discovery.warned, "a failed retry stays quiet");

    std::fs::create_dir(&dir).unwrap();
    discovery.start_watch(&mut timers);
    assert!(discovery.watch.is_some(), "the directory exists now");
    assert!(!discovery.warned, "the watch starting is logged in turn");
}

/// A watch lost with its directory is tried again on its own cadence,
/// under a check interval no longer than that cadence too: the loop, as
/// it runs, sees the directory removed (the watch reports its own loss)
/// and recreated, and has the watch back within a few passes. A retry
/// rebuilt on every pass lost to the tick on every pass and never fired.
#[tokio::test]
async fn a_lost_watch_is_retried_under_a_check_interval_as_short_as_the_retry() {
    for check_interval in [Duration::from_millis(250), REWATCH] {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("smg");
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("workers.json");
        std::fs::write(&path, manifest("")).unwrap();
        let mut discovery = discovery_checking_every(&path, check_interval);
        let mut timers = Timers::new(check_interval);
        discovery.start_watch(&mut timers);
        assert!(discovery.watch.is_some());

        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
        let deadline = time::Instant::now() + Duration::from_secs(10);
        while discovery.watch.is_some() {
            assert!(time::Instant::now() < deadline, "the loss was never seen");
            discovery.pass(&mut timers).await;
        }
        assert!(discovery.warned, "polling only, and said so");

        std::fs::create_dir(&dir).unwrap();
        std::fs::write(&path, manifest("")).unwrap();
        let deadline = time::Instant::now() + Duration::from_secs(5);
        while discovery.watch.is_none() && time::Instant::now() < deadline {
            discovery.pass(&mut timers).await;
        }
        assert!(
            discovery.watch.is_some(),
            "the watch was not tried again under a check interval of {check_interval:?}"
        );
        assert!(!discovery.warned, "the watch starting is logged in turn");
    }
}

/// The export of a fleet, fed back through the parser, yields the same
/// endpoints, types and settings; a key and an address's credentials
/// stay behind and are named in the warnings.
#[test]
fn an_exported_fleet_parses_back_without_its_credentials() {
    use openai_protocol::model_card::ModelCard;
    use smg_worker::{
        identity::basic::{worker_to_info, Worker},
        test_support::BasicWorkerBuilder,
    };

    fn info(worker: BasicWorkerBuilder) -> WorkerInfo {
        let worker: Arc<dyn Worker> = Arc::new(worker.build());
        worker_to_info(&worker)
    }
    let fleet = [
        info(
            BasicWorkerBuilder::new("http://10.0.0.22:8000")
                .worker_type(WorkerType::Prefill)
                .bootstrap_port(Some(8998))
                .kv_connector("MooncakeConnector")
                .kv_role("kv_producer")
                .kv_engine_id("engine-a")
                .model(ModelCard::new("llama")),
        ),
        info(
            BasicWorkerBuilder::new("grpc://decode.internal:9000")
                .worker_type(WorkerType::Decode)
                .model(ModelCard::new("llama")),
        ),
        info(
            BasicWorkerBuilder::new("http://admin:hunter2@10.0.0.23:8001")
                .worker_type(WorkerType::Decode)
                .api_key("s3cr3t"),
        ),
    ];

    let exported = export_manifest(&fleet);
    let text = serde_json::to_string(&exported).unwrap();

    for secret in ["hunter2", "s3cr3t", r#""api_key""#, "admin", r#""id""#] {
        assert!(
            !text.contains(secret),
            "{secret} reached the export: {text}"
        );
    }
    assert_eq!(
        exported.warnings,
        [
            "http://10.0.0.23:8001: url credentials not exported",
            "http://10.0.0.23:8001: api_key not exported"
        ]
    );

    let parsed = parse(&text, Roles::PrefillDecode).unwrap();
    let summary: Vec<_> = parsed
        .iter()
        .map(|w| {
            (
                w.endpoint.render(),
                w.worker_type,
                w.bootstrap_port,
                w.model_id_override.as_deref(),
                w.kv_connector.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            (
                "http://10.0.0.22:8000".to_string(),
                WorkerType::Prefill,
                Some(8998),
                Some("llama"),
                Some("MooncakeConnector"),
            ),
            (
                "grpc://decode.internal:9000".to_string(),
                WorkerType::Decode,
                None,
                Some("llama"),
                None,
            ),
            (
                "http://10.0.0.23:8001".to_string(),
                WorkerType::Decode,
                None,
                None,
                None,
            ),
        ]
    );
    assert_eq!(parsed[0].kv_role.as_deref(), Some("kv_producer"));
    assert_eq!(parsed[0].kv_engine_id.as_deref(), Some("engine-a"));
}

/// A data-parallel group was registered at one address; its export is
/// that one record, not one per rank.
#[test]
fn a_data_parallel_group_exports_as_one_record_at_its_base_address() {
    use smg_worker::{
        identity::basic::{worker_to_info, Worker},
        test_support::BasicWorkerBuilder,
    };

    let fleet: Vec<WorkerInfo> = (0..2)
        .map(|rank| {
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("http://10.0.0.30:8000")
                    .worker_type(WorkerType::Regular)
                    .dp_config(rank, 2)
                    .build(),
            );
            worker_to_info(&worker)
        })
        .collect();

    let exported = export_manifest(&fleet);
    let text = serde_json::to_string(&exported).unwrap();

    assert_eq!(exported.workers.len(), 1, "{text}");
    assert_eq!(exported.workers[0].url, "http://10.0.0.30:8000");
    assert!(exported.workers[0].id.is_none(), "{text}");
    assert!(exported.warnings.is_empty(), "{text}");
    assert_eq!(parse(&text, Roles::Regular).unwrap().len(), 1);
}

/// An export with warnings is still a manifest: the field is known and
/// ignored on read, so the fleet registers minus what was not carried.
#[test]
fn warnings_in_a_manifest_are_read_past() {
    let workers = parse(
        r#"{"version": 1, "workers": [{"url": "10.0.0.5:8000"}], "warnings": ["worker a: api_key not exported"]}"#,
        Roles::Regular,
    )
    .unwrap();
    assert_eq!(workers.len(), 1);
}
