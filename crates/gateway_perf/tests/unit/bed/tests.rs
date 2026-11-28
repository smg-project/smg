use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use super::*;
use crate::http::tests::serve;

#[test]
fn port_block_is_free_and_laid_out_as_the_bed_expects() {
    // A block's ports, the gateway first (the refusal below reads it at the head of
    // the busy list).
    assert_eq!(
        Ports::block(29800, 2, 0),
        (
            Ports {
                gateway: 29800,
                metrics: 29801,
                admin: 29809,
                mock_base: 29810,
                ..Ports::default()
            },
            vec![29800, 29801, 29809, 29810, 29811]
        )
    );
    // The test holds a port of its own, the first of the bed's range it can bind: the
    // bind both finds and holds it, so no other user of the range (another instance of
    // this test included) can take what the test asserts on.
    let holder = (29700..=29999)
        .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
        .unwrap();
    let held = holder.local_addr().unwrap().port();
    // The scan passes the held block by and lays the first free one out as the bed
    // expects.
    let ports = pick_ports(0, 4, 0, 29700, 29999).unwrap();
    let base = ports.gateway;
    assert_ne!(base, held);
    assert_eq!(
        ports,
        Ports {
            gateway: base,
            metrics: base + 1,
            admin: base + 9,
            mock_base: base + 10,
            ..Ports::default()
        }
    );
    let err = pick_ports(0, 4, 0, 29700, 29705).unwrap_err().to_string();
    assert!(err.contains("no free port block"), "{err}");
    // A block whose gateway port is taken is refused by that port's name: the held port
    // heads the error's list whatever else is busy in its block (an ephemeral port
    // would head a block the host's other connections share).
    let err = pick_ports(held, 4, 0, 29700, 29999)
        .unwrap_err()
        .to_string();
    assert!(err.contains(&format!("port(s) [{held}")), "{err}");
}

#[test]
fn choose_ports_hands_the_scan_the_given_base_and_a_held_base_is_refused() {
    // A base the test holds itself: the refusal is certain, and it is the scan's
    // answer for the given base, not a block of the scan's own choice (a success on
    // a fixed base is the probe that went red whenever another process held a port).
    let holder = (29700..=29999)
        .find_map(|port| TcpListener::bind(("127.0.0.1", port)).ok())
        .unwrap();
    let held = holder.local_addr().unwrap().port();
    let mut cfg = config();
    cfg.ports = None;
    cfg.port_base = held;
    let err = cfg.choose_ports().unwrap_err().to_string();
    assert!(err.contains(&format!("port(s) [{held}")), "{err}");
    assert_eq!(cfg.ports(), Ports::default());
}

#[test]
fn the_prefill_decode_block_adds_the_second_admin_the_prefill_engines_and_their_bootstrap_ports() {
    // The decode engines keep the regular fleet's places (+9 admin, +10..), so a
    // script count or a cell that reads them is the same on both fleets; the prefill
    // fleet's admin API takes +8 and its engines and bootstrap ports follow the
    // decode engines.
    let (ports, all) = Ports::block(29800, 4, 1);
    assert_eq!(
        ports,
        Ports {
            gateway: 29800,
            metrics: 29801,
            admin: 29809,
            mock_base: 29810,
            prefill_admin: 29808,
            prefill_base: 29814,
            bootstrap_base: 29815,
        }
    );
    assert_eq!(
        all,
        vec![29800, 29801, 29808, 29809, 29810, 29811, 29812, 29813, 29814, 29815]
    );
    assert_eq!(block_span(4, 1), 16);
    assert_eq!(block_span(4, 0), 14);
    // The span reserves the block (+2..+7 stay free, as on the gRPC fleet); the ports
    // bound are the two admin APIs, the gateway, its metrics and the engines' own.
    let (two, all) = Ports::block(29900, 2, 2);
    assert_eq!((two.prefill_base, two.bootstrap_base), (29912, 29914));
    assert_eq!(all.len(), 4 + 2 + 2 * 2);
    assert!(all.iter().all(|p| *p < 29900 + block_span(2, 2)));
    // A scan for a prefill-decode block steps by its own span.
    let ports = pick_ports(0, 4, 1, 29700, 29999).unwrap();
    assert_eq!(ports.prefill_admin, ports.gateway + 8);
    assert_eq!(ports.bootstrap_base, ports.gateway + 15);
    assert!(pick_ports(0, 4, 1, 29700, 29714).is_err());
}

struct FakeScript {
    calls: Arc<Mutex<Vec<String>>>,
    up_rc: i32,
}

impl BedScript for FakeScript {
    fn run(&self, action: &str, _env: &[(String, String)], log: &Path) -> io::Result<i32> {
        self.calls.lock().unwrap().push(action.to_string());
        if action == "up" {
            fs::write(log, "bed: the probe failed\n")?;
            return Ok(self.up_rc);
        }
        Ok(0)
    }
}

fn config() -> BedConfig {
    let mut cfg = BedConfig::new(Path::new("/bed.sh"), Path::new("/m"), Path::new("/l"), "/t");
    cfg.ports = Some(Ports {
        gateway: 1,
        metrics: 2,
        admin: 3,
        mock_base: 4,
        ..Ports::default()
    });
    cfg
}

#[test]
fn a_failed_bed_up_takes_the_partial_bed_down_and_a_running_bed_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let script = Box::new(FakeScript {
        calls: Arc::clone(&calls),
        up_rc: 1,
    });
    let run_dir = dir.path().join("bed");
    let mut bed = Bed::with_script(&config(), Path::new("/h"), &run_dir, true, script);
    let err = bed.up().unwrap_err().to_string();
    assert!(err.contains("bed up failed"), "{err}");
    assert!(err.contains("the probe failed"), "{err}");
    assert_eq!(calls.lock().unwrap().as_slice(), ["up", "down"]);
    // A script that exits 0 without a readable bed.json is a failed start too: the
    // error names the file and the bed is taken down, so no process or port is left.
    calls.lock().unwrap().clear();
    let script = Box::new(FakeScript {
        calls: Arc::clone(&calls),
        up_rc: 0,
    });
    let mut bed = Bed::with_script(&config(), Path::new("/h"), &run_dir, true, script);
    let err = bed.up().unwrap_err().to_string();
    assert!(err.contains("bed.json"), "{err}");
    assert_eq!(calls.lock().unwrap().as_slice(), ["up", "down"]);
    // A bed that is up already (its pid file under the run directory) is not this
    // call's to stop: the refusal comes before the script runs, nothing is taken down.
    calls.lock().unwrap().clear();
    fs::create_dir_all(run_dir.join("bed")).unwrap();
    fs::write(run_dir.join("bed").join("gateway.pid"), "1\n").unwrap();
    let err = bed.up().unwrap_err().to_string();
    assert!(err.contains("up already"), "{err}");
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn the_environment_carries_the_ports_the_pinning_and_the_allocator_print() {
    let bed = Bed::new(&config(), Path::new("/h"), Path::new("/run"), true);
    let env: BTreeMap<String, String> = bed.env().into_iter().collect();
    assert_eq!(env["GW_PORT"], "1");
    assert_eq!(env["PROM_PORT"], "2");
    assert_eq!(env["ADMIN_PORT"], "3");
    assert_eq!(env["MOCK_BASE"], "4");
    assert_eq!(env["WORKERS"], "4");
    assert_eq!(env["FLEET"], "grpc");
    assert_eq!(env["SCOPE"], "0");
    assert_eq!(env["GW_ENV"], JEMALLOC_ENV);
    assert_eq!(env["GW_ARGS"], DEFAULT_GW_ARGS);
    assert!(!env.contains_key("ENGINE_ARGS"));
    for key in [
        "PREFILL_WORKERS",
        "PREFILL_BASE",
        "BOOTSTRAP_BASE",
        "PREFILL_ADMIN_PORT",
    ] {
        assert!(
            !env.contains_key(key),
            "{key} belongs to the prefill-decode fleet alone"
        );
    }
    let plain = Bed::new(&config(), Path::new("/h"), Path::new("/run"), false);
    let env: BTreeMap<String, String> = plain.env().into_iter().collect();
    assert_eq!(env["GW_ENV"], "");
}

#[test]
fn the_prefill_decode_fleet_hands_the_script_its_second_fleet_and_its_ports() {
    let mut cfg = config();
    cfg.fleet = Fleet::PrefillDecode;
    cfg.prefill_workers = 2;
    assert_eq!(cfg.prefill_engines(), 2);
    // The block's layout, not a probe of the host's ports: the fixed-base probe went
    // red whenever another process held a port of the block.
    cfg.ports = Some(Ports::block(29950, cfg.workers, cfg.prefill_engines()).0);
    let bed = Bed::new(&cfg, Path::new("/h"), Path::new("/run"), true);
    let env: BTreeMap<String, String> = bed.env().into_iter().collect();
    assert_eq!(env["FLEET"], "prefill-decode");
    assert_eq!(env["WORKERS"], "4");
    assert_eq!(env["MOCK_BASE"], "29960");
    assert_eq!(env["ADMIN_PORT"], "29959");
    assert_eq!(env["PREFILL_WORKERS"], "2");
    assert_eq!(env["PREFILL_BASE"], "29964");
    assert_eq!(env["BOOTSTRAP_BASE"], "29966");
    assert_eq!(env["PREFILL_ADMIN_PORT"], "29958");
    assert_eq!(
        cfg.engines_text(),
        "2 prefill and 4 decode scripted mock engines on the prefill-decode path"
    );
    let grpc = config();
    assert_eq!(grpc.prefill_engines(), 0);
    assert_eq!(grpc.engines_text(), "4 scripted mock engines");
    assert_eq!(Fleet::Grpc.as_str(), "grpc");
}

#[test]
fn choose_ports_scans_for_a_block_that_holds_the_prefill_engines_too() {
    // No ports and no base: the scan, handed the config's own counts. A block laid
    // out for the decode engines alone leaves the prefill bases at zero, and only
    // the script would notice at start.
    let mut cfg = config();
    cfg.fleet = Fleet::PrefillDecode;
    cfg.prefill_workers = 2;
    cfg.ports = None;
    cfg.choose_ports().unwrap();
    let ports = cfg.ports();
    assert!((29700..=29999).contains(&ports.gateway), "{ports:?}");
    assert_eq!(ports.prefill_admin, ports.gateway + 8);
    assert_eq!(ports.mock_base, ports.gateway + 10);
    assert_eq!(ports.prefill_base, ports.mock_base + cfg.workers);
    assert_eq!(
        ports.bootstrap_base,
        ports.prefill_base + cfg.prefill_engines()
    );
    // The gRPC fleet's scan reserves no prefill ports.
    let mut grpc = config();
    grpc.ports = None;
    grpc.choose_ports().unwrap();
    let ports = grpc.ports();
    assert_eq!(ports.mock_base, ports.gateway + 10);
    assert_eq!(
        (
            ports.prefill_admin,
            ports.prefill_base,
            ports.bootstrap_base
        ),
        (0, 0, 0)
    );
}

#[test]
fn the_http_fleet_keeps_the_regular_block_and_keys_its_engines_by_the_http_wire() {
    let mut cfg = config();
    cfg.fleet = Fleet::Http;
    assert_eq!(cfg.prefill_engines(), 0);
    assert_eq!(
        cfg.engines_text(),
        "4 scripted mock engines on the HTTP wire"
    );
    assert_eq!((Fleet::Http.as_str(), Fleet::Http.wire()), ("http", "http"));
    for path in ["generate_stream", "generate_stream_spaced_long"] {
        assert!(!Fleet::Http.serves(path), "{path}");
        assert!(Fleet::PrefillDecode.serves(path), "{path}");
    }
    for path in [
        "chat",
        "stream",
        "tools",
        "json",
        "completion_stream",
        "stream_logprobs",
        "messages",
        "messages_stream",
        "messages_stream_spaced",
    ] {
        assert!(Fleet::Http.serves(path), "{path}");
    }
    assert_eq!(Fleet::PrefillDecode.wire(), "grpc");
    assert!(Fleet::Http.serves("chat") && Fleet::Http.serves("completion_stream"));
    assert!(Fleet::Http.serves("messages") && !Fleet::Http.serves("generate_stream_spaced"));
    assert!(Fleet::Grpc.serves("messages") && Fleet::PrefillDecode.serves("messages_stream"));
    assert_eq!(Fleet::Http.default_paths(), DEFAULT_PATHS);
    assert_eq!(Fleet::Grpc.default_paths(), DEFAULT_PATHS);
    assert_eq!(Fleet::PrefillDecode.default_paths(), DEFAULT_PATHS);
    let bed = Bed::new(&cfg, Path::new("/h"), Path::new("/run"), true);
    let env: BTreeMap<String, String> = bed.env().into_iter().collect();
    assert_eq!(env["FLEET"], "http");
    assert_eq!(env["MOCK_BASE"], "4");
    assert!(!env.contains_key("PREFILL_WORKERS"));
}

#[test]
fn bed_json_reads_with_and_without_the_prefill_fleet() {
    let grpc: BedInfo = serde_json::from_str(
        r#"{"gateway_pid": 7, "mock_pid": 8, "gateway_url": "http://127.0.0.1:1", "metrics_url": "m", "admin_url": "a", "workers": 4}"#,
    )
    .unwrap();
    assert_eq!((grpc.prefill_pid, grpc.prefill_workers), (0, 0));
    assert!(grpc.fleet.is_empty() && grpc.prefill_admin_url.is_empty());
    let pd: BedInfo = serde_json::from_str(
        r#"{"gateway_pid": 7, "mock_pid": 8, "gateway_url": "http://127.0.0.1:1", "metrics_url": "m", "admin_url": "a", "workers": 4,
                "fleet": "prefill-decode", "prefill_pid": 9, "prefill_workers": 1, "prefill_admin_url": "p"}"#,
    )
    .unwrap();
    assert_eq!(pd.fleet, "prefill-decode");
    assert_eq!((pd.prefill_pid, pd.prefill_workers), (9, 1));
    assert_eq!(pd.prefill_admin_url, "p");
}

#[test]
fn script_counts_report_the_reads_that_failed_instead_of_zero_misses() {
    let answers = BTreeMap::from([
        (
            "GET /admin/script/grpc:40".to_string(),
            (200, r#"{"hits": 5, "misses": 1}"#.to_string()),
        ),
        (
            "GET /admin/script/all".to_string(),
            (200, r#"{"queued": 3}"#.to_string()),
        ),
    ]);
    let server = serve(answers, false);
    assert_eq!(
        script_counts_at(&server.url, 2, 40, "grpc"),
        ScriptCounts {
            hits: 5,
            misses: 1,
            unreadable: 1,
            queued_left: Some(3)
        }
    );
    // The HTTP fleet's engines are keyed by their wire: the gRPC key reads nothing.
    assert_eq!(script_counts_at(&server.url, 2, 40, "http").unreadable, 2);
    let http_answers = BTreeMap::from([
        (
            "GET /admin/script/http:40".to_string(),
            (200, r#"{"hits": 7, "misses": 0}"#.to_string()),
        ),
        (
            "GET /admin/script/all".to_string(),
            (200, r#"{"queued": 0}"#.to_string()),
        ),
    ]);
    let http_server = serve(http_answers, false);
    let counts = script_counts_at(&http_server.url, 1, 40, "http");
    assert_eq!((counts.hits, counts.misses, counts.unreadable), (7, 0, 0));
    let empty = serve(BTreeMap::new(), false);
    assert_eq!(script_counts_at(&empty.url, 2, 40, "grpc").unreadable, 3);
}

#[test]
fn enqueue_bodies_are_bounded_by_count_and_by_bytes() {
    let small: Vec<Value> = (0..1200)
        .map(|_| serde_json::json!({"output_text": "a"}))
        .collect();
    let sizes: Vec<usize> = enqueue_chunks(&small).iter().map(|c| c.len()).collect();
    assert_eq!(sizes, [500, 500, 200]);

    let long: Vec<Value> = (0..300)
        .map(|_| serde_json::json!({"output_text": "w".repeat(12_000)}))
        .collect();
    let chunks = enqueue_chunks(&long);
    assert!(chunks.len() > 1, "{} chunk(s)", chunks.len());
    assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 300);
    for chunk in &chunks {
        let bytes: usize = chunk.iter().map(|v| v.to_string().len() + 1).sum();
        assert!(bytes <= ENQUEUE_BYTES, "{bytes} bytes in one body");
    }

    let one: Vec<Value> = vec![serde_json::json!({"output_text": "w".repeat(3_000_000)})];
    assert_eq!(enqueue_chunks(&one).len(), 1);
    assert!(enqueue_chunks(&[]).is_empty());
}
