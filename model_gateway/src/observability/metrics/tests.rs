use std::net::TcpListener;

use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};

use super::{test_support::render_with_recorder, *};

#[test]
fn tokenizer_activity_registers_both_layers_on_scrape() {
    let rendered = render_with_recorder(|| {
        init_metrics();
        record_tokenizer_cache_activity();
    });
    for layer in ["l0", "l1"] {
        for name in [
            "smg_tokenizer_cache_lookups_total",
            "smg_tokenizer_cache_evictions_total",
            "smg_tokenizer_cache_reused_bytes_total",
        ] {
            assert!(rendered
                .lines()
                .any(|line| line.starts_with(&format!("{name}{{"))
                    && line.contains(&format!("layer=\"{layer}\""))));
        }
    }
}

#[test]
fn encode_path_activity_registers_every_path_and_reason_on_scrape() {
    let rendered = render_with_recorder(|| {
        init_metrics();
        record_tokenizer_encode_activity();
    });
    for path in ["v1", "native", "tokenizers"] {
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("smg_tokenizer_encode_path_loads_total{")
                    && line.contains(&format!("path=\"{path}\""))
            }),
            "missing path {path}: {rendered}"
        );
    }
    for reason in ["not_nfc", "piece_declined"] {
        assert!(
            rendered.lines().any(|line| {
                line.starts_with("smg_tokenizer_encode_fallbacks_total{")
                    && line.contains(&format!("reason=\"{reason}\""))
            }),
            "missing reason {reason}: {rendered}"
        );
    }
}

#[test]
fn tokenizer_activity_exports_monotonic_totals_without_double_counting() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        init_metrics();
        for layer in ["l0", "l1"] {
            for (hits, misses, evictions, reused_bytes) in [(7, 3, 2, 1024), (9, 4, 3, 2048)] {
                let snapshot = CacheActivityStats {
                    layer,
                    hits,
                    misses,
                    evictions,
                    reused_bytes,
                };
                record_tokenizer_cache_activity_snapshot(snapshot);
                record_tokenizer_cache_activity_snapshot(snapshot);
                // An older concurrent scrape must not decrease counters either.
                record_tokenizer_cache_activity_snapshot(CacheActivityStats {
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                    reused_bytes: 0,
                    ..snapshot
                });
                let rendered = handle.render();
                for (name, result, value) in [
                    ("smg_tokenizer_cache_lookups_total", Some("hit"), hits),
                    ("smg_tokenizer_cache_lookups_total", Some("miss"), misses),
                    ("smg_tokenizer_cache_evictions_total", None, evictions),
                    ("smg_tokenizer_cache_reused_bytes_total", None, reused_bytes),
                ] {
                    assert!(rendered.contains(&format!("# TYPE {name} counter")));
                    assert!(
                        rendered.lines().any(|line| {
                            line.starts_with(&format!("{name}{{"))
                                && line.contains(&format!("layer=\"{layer}\""))
                                && result.is_none_or(|r| line.contains(&format!("result=\"{r}\"")))
                                && line.ends_with(&format!(" {value}"))
                        }),
                        "missing {name} for {layer}: {rendered}"
                    );
                }
            }
        }
    });
}

#[test]
fn prefill_worker_removal_releases_its_admission_series() {
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        None,
        None,
    )));
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    metrics::with_local_recorder(&recorder, || {
        Metrics::set_pd_prefill_admission_inflight("http://removed-prefill", 7);
        Metrics::set_pd_prefill_admission_inflight("http://kept-prefill", 3);
        workers.retire("http://removed-prefill");
        // A release racing the removal must not bring the series back.
        Metrics::set_pd_prefill_admission_inflight("http://removed-prefill", 6);
    });
    let mut rendered = String::new();
    workers.render_into(&mut rendered);
    assert!(
        !rendered.contains("http://removed-prefill"),
        "removed worker still rendered:\n{rendered}"
    );
    assert_metric(
        &rendered,
        "smg_pd_prefill_admission_inflight",
        &[r#"worker="http://kept-prefill""#],
        "3",
    );
}

/// Core engine gauges share these labels for the snapshot fixtures.
const CORE_LABELS: [&str; 3] = ["dp_rank=\"2\"", "model=\"m\"", "worker=\"http://w:1\""];

/// Assert the rendered scrape has a line for `name` carrying every label in
/// `labels` and ending in `value`. Label order is exporter-defined, so this
/// matches on substrings rather than a fixed label set.
fn assert_metric(rendered: &str, name: &str, labels: &[&str], value: &str) {
    let line = rendered
        .lines()
        .find(|l| l.starts_with(&format!("{name}{{")))
        .unwrap_or_else(|| panic!("metric {name} missing; rendered:\n{rendered}"));
    for label in labels {
        assert!(line.contains(label), "{name} missing label {label}: {line}");
    }
    assert!(
        line.ends_with(&format!(" {value}")),
        "{name} expected value {value}: {line}"
    );
}

#[test]
fn record_engine_load_sets_core_gauges_per_dp_rank() {
    let response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 2,
            num_running_reqs: 7,
            num_waiting_reqs: 3,
            token_usage: 0.5,
            gen_throughput: 42.0,
            cache_hit_rate: 0.25,
            ..Default::default()
        }],
        ..Default::default()
    };

    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load("http://w:1", "m", &response);
    });

    // The exporter renders labels in insertion order, so assert on the
    // metric line's components rather than a fixed label ordering.
    assert_metric(&rendered, "smg_engine_running_requests", &CORE_LABELS, "7");
    assert_metric(&rendered, "smg_engine_waiting_requests", &CORE_LABELS, "3");
    assert_metric(&rendered, "smg_engine_gen_throughput", &CORE_LABELS, "42");
    // PD gauges absent when no disagg section was reported.
    assert!(
        !rendered.contains("smg_engine_pd_"),
        "PD gauges must not appear without a disagg section; rendered:\n{rendered}"
    );
}

/// The capacity behind the usage ratio is a series of its own when the
/// engine reports it, and absent when the record carries the ratio alone.
#[test]
fn record_engine_load_sets_the_kv_capacity_when_the_engine_reports_it() {
    let mut response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 2,
            token_usage: 0.5,
            max_total_num_tokens: 262_144,
            ..Default::default()
        }],
        ..Default::default()
    };
    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load("http://w:1", "m", &response);
    });
    assert_metric(
        &rendered,
        "smg_engine_kv_capacity_tokens",
        &CORE_LABELS,
        "262144",
    );
    assert_metric(&rendered, "smg_engine_token_usage", &CORE_LABELS, "0.5");

    response.loads[0].max_total_num_tokens = 0;
    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load("http://w:1", "m", &response);
    });
    assert!(
        !rendered.contains("smg_engine_kv_capacity_tokens"),
        "a record without the capacity sets no series; rendered:\n{rendered}"
    );
    assert_metric(&rendered, "smg_engine_token_usage", &CORE_LABELS, "0.5");
}

/// A model's block size is one gauge, set from what the gateway knows, and
/// 0 once it knows none.
#[test]
fn the_kv_block_size_of_a_model_is_a_gauge() {
    let rendered = render_with_recorder(|| {
        Metrics::set_kv_block_size("m", 16);
        Metrics::set_kv_block_size("m", 64);
    });
    assert_metric(
        &rendered,
        "smg_kv_block_size_tokens",
        &["model=\"m\""],
        "64",
    );
    let rendered = render_with_recorder(|| {
        Metrics::set_kv_block_size("m", 64);
        Metrics::set_kv_block_size("m", 0);
    });
    assert_metric(&rendered, "smg_kv_block_size_tokens", &["model=\"m\""], "0");
}

#[test]
fn record_engine_load_sets_pd_gauges_when_disagg_present() {
    let response = WorkerLoadResponse {
        timestamp: "t".to_string(),
        dp_rank_count: 1,
        loads: vec![SchedulerLoadSnapshot {
            dp_rank: 0,
            disagg_mode: Some("prefill".to_string()),
            kv_transfer_latency_ms: Some(3.5),
            kv_transfer_speed_gb_s: Some(12.0),
            prefill_queue_reqs: Some(9),
            decode_queue_reqs: Some(4),
            ..Default::default()
        }],
        ..Default::default()
    };

    let rendered = render_with_recorder(|| {
        Metrics::record_engine_load("http://w:1", "m", &response);
    });

    let pd_labels = ["role=\"prefill\"", "worker=\"http://w:1\"", "dp_rank=\"0\""];
    assert_metric(
        &rendered,
        "smg_engine_pd_kv_transfer_latency_ms",
        &pd_labels,
        "3.5",
    );
    assert_metric(
        &rendered,
        "smg_engine_pd_prefill_queue_reqs",
        &pd_labels,
        "9",
    );
    assert_metric(
        &rendered,
        "smg_engine_pd_decode_queue_reqs",
        &pd_labels,
        "4",
    );
}

#[test]
fn cache_tree_setters_emit_per_model_gauges() {
    let rendered = render_with_recorder(|| {
        Metrics::set_cache_tree_chars("m", 120);
        Metrics::set_cache_tree_tokens("m", 64);
        Metrics::set_cache_tree_tenants("m", "string", 3);
        Metrics::set_cache_tree_tenants("m", "token", 2);
    });

    assert_metric(&rendered, "smg_cache_tree_chars", &["model=\"m\""], "120");
    assert_metric(&rendered, "smg_cache_tree_tokens", &["model=\"m\""], "64");
    // Two tenant series (one per tree kind); series order within the
    // family is exporter-defined, so match each line independently.
    for (tree, value) in [("string", "3"), ("token", "2")] {
        let label = format!("tree=\"{tree}\"");
        assert!(
            rendered.lines().any(|l| {
                l.starts_with("smg_cache_tree_tenants{")
                    && l.contains("model=\"m\"")
                    && l.contains(&label)
                    && l.ends_with(&format!(" {value}"))
            }),
            "smg_cache_tree_tenants {tree} series missing; rendered:\n{rendered}"
        );
    }
}

/// The match ratio must render as a real histogram (`_bucket{le=...}`
/// lines) once its buckets are registered the way `start_prometheus` does;
/// without them the recorder falls back to a summary.
#[test]
fn cache_aware_decision_metrics_render_counter_and_bucketed_histogram() {
    let recorder = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full(String::from("smg_cache_aware_match_ratio")),
            CACHE_AWARE_MATCH_RATIO_BUCKETS,
        )
        .expect("bucket override")
        .build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_worker_cache_aware_policy_branch("tree_match");
        Metrics::record_worker_cache_aware_policy_branch("tree_match");
        Metrics::record_worker_cache_aware_policy_branch("spill");
        Metrics::record_cache_aware_match_ratio(0.0);
        Metrics::record_cache_aware_match_ratio(0.75);
    });
    let rendered = handle.render();

    for (branch, value) in [("tree_match", "2"), ("spill", "1")] {
        let series = format!("smg_cache_aware_policy_branch_total{{branch=\"{branch}\"}} {value}");
        assert!(
            rendered.lines().any(|l| l == series),
            "{series} missing; rendered:\n{rendered}"
        );
    }
    for (le, count) in [
        ("0", "1"),
        ("0.7", "1"),
        ("0.8", "2"),
        ("1", "2"),
        ("+Inf", "2"),
    ] {
        let series = format!("smg_cache_aware_match_ratio_bucket{{le=\"{le}\"}} {count}");
        assert!(
            rendered.lines().any(|l| l == series),
            "{series} missing; rendered:\n{rendered}"
        );
    }
    assert!(
        rendered
            .lines()
            .any(|l| l == "smg_cache_aware_match_ratio_count 2"),
        "histogram count missing; rendered:\n{rendered}"
    );
    assert!(
        !rendered.contains("smg_cache_aware_match_ratio{quantile="),
        "match ratio rendered as a summary; rendered:\n{rendered}"
    );
}

#[test]
fn test_prometheus_config_default() {
    let config = PrometheusConfig::default();
    assert_eq!(config.port, 29000);
    assert_eq!(config.host, "0.0.0.0");
}

#[test]
fn test_prometheus_config_custom() {
    let config = PrometheusConfig {
        port: 8080,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    };
    assert_eq!(config.port, 8080);
    assert_eq!(config.host, "127.0.0.1");
}

#[test]
fn test_prometheus_config_clone() {
    let config = PrometheusConfig {
        port: 9090,
        host: "192.168.1.1".to_string(),
        duration_buckets: None,
    };
    let cloned = config.clone();
    assert_eq!(cloned.port, config.port);
    assert_eq!(cloned.host, config.host);
}

#[test]
fn test_valid_ipv4_parsing() {
    let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

    for ip_str in test_cases {
        let config = PrometheusConfig {
            port: 29000,
            host: ip_str.to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        assert!(matches!(ip_addr, IpAddr::V4(_)));
    }
}

#[test]
fn test_valid_ipv6_parsing() {
    let test_cases = vec!["::1", "2001:db8::1", "::"];

    for ip_str in test_cases {
        let config = PrometheusConfig {
            port: 29000,
            host: ip_str.to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        assert!(matches!(ip_addr, IpAddr::V6(_)));
    }
}

#[test]
fn test_invalid_ip_parsing() {
    let test_cases = vec!["invalid", "256.256.256.256", "hostname"];

    for ip_str in test_cases {
        let config = PrometheusConfig {
            port: 29000,
            host: ip_str.to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config
            .host
            .parse()
            .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

        assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
    }
}

#[test]
fn test_socket_addr_creation() {
    let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

    for (host, port) in test_cases {
        let config = PrometheusConfig {
            port,
            host: host.to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.port(), port);
        assert_eq!(socket_addr.ip().to_string(), host);
    }
}

#[test]
fn test_socket_addr_with_different_ports() {
    let ports = vec![0, 80, 8080, 65535];

    for port in ports {
        let config = PrometheusConfig {
            port,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.port(), port);
    }
}

#[test]
fn test_duration_bucket_coverage() {
    let test_cases: [(f64, &str); 7] = [
        (0.0005, "sub-millisecond"),
        (0.005, "5ms"),
        (0.05, "50ms"),
        (1.0, "1s"),
        (10.0, "10s"),
        (60.0, "1m"),
        (240.0, "4m"),
    ];

    let buckets: [f64; 20] = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
        60.0, 90.0, 120.0, 180.0, 240.0,
    ];

    for (duration, label) in test_cases {
        let bucket_found = buckets
            .iter()
            .any(|&b| (b - duration).abs() < 0.0001 || b > duration);
        assert!(bucket_found, "No bucket found for {duration} ({label})");
    }
}

#[test]
fn test_duration_suffix_matcher() {
    let matcher = Matcher::Suffix(String::from("duration_seconds"));

    let _matching_metrics = [
        "request_duration_seconds",
        "response_duration_seconds",
        "smg_request_duration_seconds",
    ];

    let _non_matching_metrics = ["duration_total", "duration_seconds_total", "other_metric"];

    match matcher {
        Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
        _ => panic!("Expected Suffix matcher"),
    }
}

#[test]
fn test_prometheus_builder_configuration() {
    let _config = PrometheusConfig::default();

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket = [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
        60.0, 90.0, 120.0, 180.0, 240.0,
    ];

    assert_eq!(duration_bucket.len(), 20);

    match duration_matcher {
        Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
        _ => panic!("Expected Suffix matcher"),
    }
}

#[test]
fn test_upkeep_timeout_duration() {
    let timeout = Duration::from_secs(5 * 60);
    assert_eq!(timeout.as_secs(), 300);
}

#[test]
fn test_custom_buckets_for_different_metrics() {
    let request_buckets = [0.001, 0.01, 0.1, 1.0, 10.0];
    let generate_buckets = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

    assert_eq!(request_buckets.len(), 5);
    assert_eq!(generate_buckets.len(), 6);

    for i in 1..request_buckets.len() {
        assert!(request_buckets[i] > request_buckets[i - 1]);
    }

    for i in 1..generate_buckets.len() {
        assert!(generate_buckets[i] > generate_buckets[i - 1]);
    }
}

#[test]
fn test_port_already_in_use() {
    let port = 29123;

    if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
        let config = PrometheusConfig {
            port,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };

        assert_eq!(config.port, port);
    }
}

#[test]
fn test_metrics_endpoint_accessibility() {
    let config = PrometheusConfig {
        port: 29000,
        host: "127.0.0.1".to_string(),
        duration_buckets: None,
    };

    let ip_addr: IpAddr = config.host.parse().unwrap();
    let socket_addr = SocketAddr::new(ip_addr, config.port);

    assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
}

// ========================================================================
// String interning tests
// ========================================================================

#[test]
fn test_intern_string_returns_same_arc() {
    let s1 = intern_string("test_model");
    let s2 = intern_string("test_model");

    // Should return the same Arc (pointer equality)
    assert!(Arc::ptr_eq(&s1, &s2));
    assert_eq!(&*s1, "test_model");
}

#[test]
fn test_intern_string_different_strings() {
    let s1 = intern_string("model_a");
    let s2 = intern_string("model_b");

    // Different strings should have different Arcs
    assert!(!Arc::ptr_eq(&s1, &s2));
    assert_eq!(&*s1, "model_a");
    assert_eq!(&*s2, "model_b");
}

#[test]
fn test_intern_string_empty() {
    let s1 = intern_string("");
    let s2 = intern_string("");

    assert!(Arc::ptr_eq(&s1, &s2));
    assert_eq!(&*s1, "");
}

#[test]
fn test_interner_size_grows() {
    let initial_size = interner_size();

    // Intern some unique strings
    let unique = format!("unique_test_string_{initial_size}");
    intern_string(&unique);

    assert!(interner_size() > initial_size);
}

#[test]
fn test_bool_to_static_str() {
    assert_eq!(bool_to_static_str(true), "true");
    assert_eq!(bool_to_static_str(false), "false");
}

#[test]
fn test_status_code_to_static_str() {
    // Common codes should return static strings
    assert_eq!(status_code_to_static_str(200), Some("200"));
    assert_eq!(status_code_to_static_str(404), Some("404"));
    assert_eq!(status_code_to_static_str(500), Some("500"));

    // Uncommon codes should return None
    assert_eq!(status_code_to_static_str(418), None);
    assert_eq!(status_code_to_static_str(999), None);
}

#[test]
fn test_status_code_to_cow() {
    // Common codes should be borrowed
    let cow_200 = status_code_to_cow(200);
    assert!(matches!(cow_200, Cow::Borrowed(_)));
    assert_eq!(cow_200, "200");

    // Uncommon codes should be owned
    let cow_418 = status_code_to_cow(418);
    assert!(matches!(cow_418, Cow::Owned(_)));
    assert_eq!(cow_418, "418");
}

#[test]
fn test_method_to_static_str() {
    assert_eq!(method_to_static_str("GET"), "GET");
    assert_eq!(method_to_static_str("POST"), "POST");
    assert_eq!(method_to_static_str("UNKNOWN"), "OTHER");
}

// ========================================================================
// PD disaggregation metric tests
// ========================================================================

/// Run `f` with a Prometheus recorder installed thread-locally and return
/// the rendered /metrics text. Mirrors the helper in `runtime_metrics`.
fn with_test_recorder<T>(f: impl FnOnce() -> T) -> (String, T) {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let result = metrics::with_local_recorder(&recorder, f);
    (handle.render(), result)
}

#[test]
fn test_record_pd_prefill_duration_emits_histogram() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_prefill_duration(
            metrics_labels::BACKEND_PD,
            "test-model",
            "vllm",
            Duration::from_millis(42),
        );
    });
    assert!(
        rendered.contains("smg_pd_prefill_duration_seconds_count{")
            && rendered.contains(r#"backend_type="pd""#)
            && rendered.contains(r#"model="test-model""#)
            && rendered.contains(r#"runtime="vllm""#),
        "prefill duration histogram not emitted; rendered:\n{rendered}"
    );
}

#[test]
fn test_record_pd_kv_transfer_duration_emits_histogram() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_kv_transfer_duration(
            metrics_labels::BACKEND_PD,
            "m",
            "vllm",
            Duration::from_millis(7),
        );
    });
    assert!(
        rendered.contains("smg_pd_kv_transfer_duration_seconds_count"),
        "kv transfer histogram not emitted; rendered:\n{rendered}"
    );
}

#[test]
fn test_record_pd_ttft_emits_histogram() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_ttft(
            metrics_labels::BACKEND_PD,
            "m",
            "sglang",
            Duration::from_millis(123),
        );
    });
    assert!(
        rendered.contains("smg_pd_ttft_seconds_count") && rendered.contains(r#"runtime="sglang""#),
        "pd ttft histogram not emitted; rendered:\n{rendered}"
    );
}

#[test]
fn test_record_pd_decode_ttft_emits_histogram() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_decode_ttft(
            metrics_labels::BACKEND_PD,
            "m",
            "vllm",
            Duration::from_millis(321),
        );
    });
    assert!(
        rendered.contains("smg_pd_decode_ttft_seconds_count")
            && rendered.contains(r#"runtime="vllm""#),
        "pd decode ttft histogram not emitted; rendered:\n{rendered}"
    );
}

/// The PD phase families say on `/metrics` what they bound: the relay
/// family is the gateway's own gap and not the engines' KV transfer, and
/// the prefill phase, the decode phase and the total name their anchors.
#[test]
fn pd_phase_families_describe_what_they_measure() {
    let (rendered, ()) = with_test_recorder(|| {
        init_metrics();
        let sample = Duration::from_millis(5);
        Metrics::record_pd_prefill_duration(metrics_labels::BACKEND_PD, "m", "vllm", sample);
        Metrics::record_pd_kv_transfer_duration(metrics_labels::BACKEND_PD, "m", "vllm", sample);
        Metrics::record_pd_decode_ttft(metrics_labels::BACKEND_PD, "m", "vllm", sample);
        Metrics::record_pd_ttft(metrics_labels::BACKEND_PD, "m", "vllm", sample);
    });
    let help = |family: &str| -> String {
        rendered
            .lines()
            .find_map(|line| line.strip_prefix(&format!("# HELP {family} ")))
            .unwrap_or_else(|| panic!("no # HELP for {family}; rendered:\n{rendered}"))
            .to_owned()
    };
    let prefill = help("smg_pd_prefill_duration_seconds");
    assert!(
        prefill.contains("prefill dispatch to the end of the prefill response"),
        "{prefill}"
    );
    let relay = help("smg_pd_kv_transfer_duration_seconds");
    assert!(
        relay.contains("relay phase") && relay.contains("not the engines' KV transfer"),
        "{relay}"
    );
    let decode = help("smg_pd_decode_ttft_seconds");
    assert!(
        decode.contains("decode dispatch to the first decode token"),
        "{decode}"
    );
    let total = help("smg_pd_ttft_seconds");
    assert!(
        total.contains("prefill dispatch to the first decode token")
            && total.contains("add up to it"),
        "{total}"
    );
}

#[test]
fn test_record_pd_kv_connector_mode_counts_by_mode() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_kv_connector_mode(metrics_labels::KV_CONNECTOR_MOONCAKE);
        Metrics::record_pd_kv_connector_mode(metrics_labels::KV_CONNECTOR_MOONCAKE);
        Metrics::record_pd_kv_connector_mode(metrics_labels::KV_CONNECTOR_NIXL);
    });
    assert!(
        rendered.contains(r#"smg_pd_kv_connector_mode_total{mode="mooncake"} 2"#),
        "mooncake connector counter wrong; rendered:\n{rendered}"
    );
    assert!(
        rendered.contains(r#"smg_pd_kv_connector_mode_total{mode="nixl"} 1"#),
        "nixl connector counter wrong; rendered:\n{rendered}"
    );
}

#[test]
fn test_record_pd_failure_counters() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_bootstrap_failure();
        Metrics::record_pd_kv_transfer_failure();
        Metrics::record_pd_kv_transfer_failure();
    });
    assert!(
        rendered.contains("smg_pd_bootstrap_failures_total 1"),
        "bootstrap failure counter wrong; rendered:\n{rendered}"
    );
    assert!(
        rendered.contains("smg_pd_kv_transfer_failures_total 2"),
        "kv transfer failure counter wrong; rendered:\n{rendered}"
    );
}

#[test]
fn test_record_pd_rendezvous_failure_by_side_and_phase() {
    let (rendered, ()) = with_test_recorder(|| {
        Metrics::record_pd_bootstrap_failure();
        Metrics::record_pd_rendezvous_failure(metrics_labels::WORKER_PREFILL, "probe");
        Metrics::record_pd_rendezvous_failure(metrics_labels::WORKER_DECODE, "bootstrap");
        Metrics::record_pd_rendezvous_failure(metrics_labels::WORKER_DECODE, "bootstrap");
    });
    // Two families: the injection series keeps its unlabelled shape, and
    // the labelled rendezvous counts never join it.
    assert!(
        rendered.contains("smg_pd_bootstrap_failures_total 1"),
        "{rendered}"
    );
    assert!(
        !rendered.contains("smg_pd_bootstrap_failures_total{"),
        "{rendered}"
    );
    let labelled: Vec<&str> = rendered
        .lines()
        .filter(|line| line.starts_with("smg_pd_rendezvous_failures_total{"))
        .collect();
    assert_eq!(labelled.len(), 2, "{rendered}");
    assert!(
        labelled.iter().any(|line| line.contains(r#"leg="prefill""#)
            && line.contains(r#"phase="probe""#)
            && line.ends_with(" 1")),
        "{rendered}"
    );
    assert!(
        labelled.iter().any(|line| line.contains(r#"leg="decode""#)
            && line.contains(r#"phase="bootstrap""#)
            && line.ends_with(" 2")),
        "{rendered}"
    );
}

/// With handles kept, the records update the recorder's own series: the
/// rendered counts are those of the records. The keeping is this
/// thread's say, so the test runs with the others.
#[test]
fn kept_handles_update_the_recorders_series() {
    let rendered = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        for _ in 0..3 {
            Metrics::record_http_request("POST", "/v1/chat/completions");
        }
        Metrics::record_http_request("GET", "/v1/models");
        for _ in 0..2 {
            Metrics::record_http_response("/v1/chat/completions", 200, "");
        }
        Metrics::record_http_response("/v1/chat/completions", 429, "rate_limited");
        Metrics::record_http_duration("POST", "/v1/chat/completions", Duration::from_millis(5));
        Metrics::record_http_rate_limit("allowed");
        Metrics::record_http_rate_limit("allowed");
        Metrics::set_http_connections_active(7);
        Metrics::record_http_connection_rejected();
        handles::reset();
    });
    for expected in [
        "smg_http_requests_total{method=\"POST\",path=\"/v1/chat/completions\"} 3",
        "smg_http_requests_total{method=\"GET\",path=\"/v1/models\"} 1",
        "smg_http_responses_total{path=\"/v1/chat/completions\",status_code=\"200\",error_code=\"\"} 2",
        "smg_http_responses_total{path=\"/v1/chat/completions\",status_code=\"429\",error_code=\"rate_limited\"} 1",
        "smg_http_rate_limit_total{result=\"allowed\"} 2",
        "smg_http_connections_active 7",
        "smg_http_connections_rejected_total 1",
    ] {
        assert!(
            rendered.lines().any(|line| line == expected),
            "missing {expected} in:\n{rendered}"
        );
    }
    assert!(rendered.contains(
        "smg_http_request_duration_seconds_count{method=\"POST\",path=\"/v1/chat/completions\"} 1"
    ));
}

/// The cost of one HTTP record with and without the kept handles, for
/// the Numbers table; run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn time_per_http_record() {
    use std::time::Instant;
    const ROUNDS: u32 = 200_000;
    let recorder = PrometheusBuilder::new().build_recorder();
    let time = |label: &str| {
        metrics::with_local_recorder(&recorder, || {
            let start = Instant::now();
            for _ in 0..ROUNDS {
                Metrics::record_http_request("POST", "/v1/chat/completions");
                Metrics::set_http_connections_active(3);
                Metrics::record_http_duration(
                    "POST",
                    "/v1/chat/completions",
                    Duration::from_millis(2),
                );
                Metrics::record_http_response("/v1/chat/completions", 200, "");
            }
            let per_request = start.elapsed().as_nanos() as f64 / f64::from(ROUNDS);
            println!("ns per request for the HTTP layer's four records, {label}: {per_request:.0}");
        });
    };
    time("keys built per record");
    handles::force_on_this_thread(Some(true));
    time("handles kept per worker");
    handles::reset();
}

/// The lines of a scrape that are series (not `# HELP` or `# TYPE`),
/// sorted, for comparing two recorders' output.
fn sorted_series(rendered: &str) -> Vec<&str> {
    let mut series: Vec<&str> = rendered
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .collect();
    series.sort_unstable();
    series
}

/// The stream-timing set recorded through kept handles renders exactly
/// as it does with a key built per record: the same series with the
/// same values, and no series a stream had nothing for (no TTFT or
/// TPOT for a model whose streams gave no token, no input tokens for an
/// endpoint that reports none).
#[test]
fn the_stream_timing_set_renders_the_same_through_kept_handles() {
    let records = || {
        for (ttft, output_tokens, input_tokens) in [
            (Some(Duration::from_millis(10)), 5, Some(100)),
            (None, 1, None),
            (Some(Duration::from_millis(5)), 1, Some(7)),
        ] {
            Metrics::record_streaming_metrics(StreamingMetricsParams {
                router_type: metrics_labels::ROUTER_GRPC,
                backend_type: metrics_labels::BACKEND_REGULAR,
                model_id: "m1",
                endpoint: metrics_labels::ENDPOINT_CHAT,
                ttft,
                generation_duration: Duration::from_millis(50),
                input_tokens,
                cached_tokens: input_tokens.map(|input| input / 10),
                output_tokens,
            });
        }
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m2",
            endpoint: metrics_labels::ENDPOINT_GENERATE,
            ttft: None,
            generation_duration: Duration::from_millis(20),
            input_tokens: None,
            cached_tokens: None,
            output_tokens: 1,
        });
    };
    let built = render_with_recorder(records);
    let mut kept_label_sets = 0;
    let kept = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        kept_label_sets = handles::STREAMS
            .with_borrow(|by_labels| by_labels.values().map(HashMap::len).sum::<usize>());
        handles::reset();
    });
    assert_eq!(sorted_series(&kept), sorted_series(&built));
    assert_eq!(kept_label_sets, 2, "one bundle per label set and model");
    for expected in [
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\",token_type=\"output\"} 7",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\",token_type=\"input\"} 107",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\",token_type=\"cached\"} 10",
        "smg_router_tokens_total{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\",endpoint=\"generate\",token_type=\"output\"} 1",
        "smg_router_ttft_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 2",
        "smg_router_tpot_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 1",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 3",
        "smg_router_generation_duration_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\",endpoint=\"generate\"} 1",
    ] {
        assert!(
            kept.lines().any(|line| line == expected),
            "missing {expected} in:\n{kept}"
        );
    }
    for absent in [
        "smg_router_ttft_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\"",
        "smg_router_tpot_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\"",
        "model=\"m2\",endpoint=\"generate\",token_type=\"input\"",
        "model=\"m2\",endpoint=\"generate\",token_type=\"cached\"",
    ] {
        assert!(!kept.contains(absent), "unexpected {absent} in:\n{kept}");
    }
}

/// A stage's time renders under its stage and streaming labels, the same
/// through a kept handle and through a key built per record, one
/// observation per record; every stage of the table has its kept slot.
#[test]
fn a_stage_duration_renders_under_its_stage_label() {
    let records = || {
        Metrics::record_stage_duration(RouterStage::Preparation, false, Duration::from_millis(250));
        Metrics::record_stage_duration(RouterStage::Preparation, false, Duration::from_millis(500));
        Metrics::record_stage_duration(RouterStage::Execute, true, Duration::from_millis(125));
        for stage in RouterStage::ALL {
            Metrics::record_stage_duration(stage, true, Duration::from_millis(1));
        }
    };
    let built = render_with_recorder(records);
    let kept = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        handles::reset();
    });
    assert_eq!(sorted_series(&kept), sorted_series(&built));
    for expected in [
        "smg_router_stage_seconds_count{stage=\"preparation\",streaming=\"false\"} 2",
        "smg_router_stage_seconds_sum{stage=\"preparation\",streaming=\"false\"} 0.75",
        "smg_router_stage_seconds_sum{stage=\"execute\",streaming=\"true\"} 0.126",
        "smg_router_stage_seconds_count{stage=\"execute\",streaming=\"true\"} 2",
        "smg_router_stage_seconds_count{stage=\"reselection\",streaming=\"true\"} 1",
    ] {
        assert!(
            kept.lines().any(|line| line == expected),
            "missing {expected} in:\n{kept}"
        );
    }
    let streamed = kept
        .lines()
        .filter(|line| {
            line.starts_with("smg_router_stage_seconds_count{")
                && line.contains("streaming=\"true\"")
        })
        .count();
    assert_eq!(streamed, RouterStage::COUNT, "{kept}");
}

/// The kept worker counters follow the worker set: a removal drops the
/// address's series and the handles kept to them, so a write after the
/// removal lands nowhere, and a re-registration starts the counter over
/// instead of feeding the handle built while the address was retired.
#[test]
fn kept_worker_counters_follow_the_workers_registrations() {
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        None,
        None,
    )));
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    // The rendered `smg_worker_requests_total` of each worker, in order.
    let counts = || {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        ["http://w1", "http://w2"].map(|worker| {
            rendered
                .lines()
                .find(|line| {
                    line.starts_with("smg_worker_requests_total{")
                        && line.contains(&format!("worker=\"{worker}\""))
                })
                .map(|line| line.rsplit(' ').next().unwrap_or_default().to_owned())
        })
    };
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_worker_request("http://w1", "m");
        Metrics::record_worker_request("http://w1", "m");
        Metrics::record_worker_request("http://w2", "m");
        assert_eq!(counts(), [Some("2".to_owned()), Some("1".to_owned())]);

        workers.retire("http://w1");
        handles::worker_set_changed();
        // A send racing the removal: its write is dropped, not kept.
        Metrics::record_worker_request("http://w1", "m");
        assert_eq!(counts(), [None, Some("1".to_owned())]);

        workers.activate("http://w1");
        handles::worker_set_changed();
        Metrics::record_worker_request("http://w1", "m");
        Metrics::record_worker_request("http://w1", "m");
        Metrics::record_worker_request("http://w2", "m");
        assert_eq!(counts(), [Some("2".to_owned()), Some("2".to_owned())]);
    });
    handles::reset();
}

/// With handles kept, the per-request worker gauges set the recorder's
/// own series (the last value written shows), and a retirement drops the
/// kept handle: a set racing the removal feeds nothing, a re-registration
/// gets a fresh handle.
#[test]
fn kept_worker_gauges_set_the_recorders_series() {
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        None,
        None,
    )));
    let recorder = WorkerSeriesRecorder::new(
        PrometheusBuilder::new().build_recorder(),
        Arc::clone(&workers),
    );
    workers.activate("http://w1");
    // The rendered value of one per-worker series.
    let value = |series: &str, worker: &str| -> Option<String> {
        let mut rendered = String::new();
        workers.render_into(&mut rendered);
        rendered
            .lines()
            .find(|line| {
                line.starts_with(&format!("{series}{{"))
                    && line.contains(&format!("worker=\"{worker}\""))
            })
            .map(|line| line.rsplit(' ').next().unwrap_or_default().to_owned())
    };
    handles::force_on_this_thread(Some(true));
    metrics::with_local_recorder(&recorder, || {
        Metrics::set_worker_requests_active("http://w1", 3);
        Metrics::set_worker_requests_active("http://w1", 2);
        Metrics::set_worker_cb_consecutive_failures("http://w1", 1);
        Metrics::set_worker_cb_consecutive_successes("http://w1", 4);
        assert_eq!(
            value("smg_worker_requests_active", "http://w1").as_deref(),
            Some("2")
        );
        assert_eq!(
            value("smg_worker_cb_consecutive_failures", "http://w1").as_deref(),
            Some("1")
        );
        assert_eq!(
            value("smg_worker_cb_consecutive_successes", "http://w1").as_deref(),
            Some("4")
        );

        workers.retire("http://w1");
        handles::worker_set_changed();
        Metrics::set_worker_requests_active("http://w1", 9);
        assert_eq!(value("smg_worker_requests_active", "http://w1"), None);

        workers.activate("http://w1");
        handles::worker_set_changed();
        Metrics::set_worker_requests_active("http://w1", 1);
        assert_eq!(
            value("smg_worker_requests_active", "http://w1").as_deref(),
            Some("1")
        );
    });
    handles::reset();
}

/// The registry's registration and removal hooks are what move the
/// worker-set generation the kept counters are checked against.
#[test]
fn a_registration_or_a_removal_drops_the_kept_worker_counters() {
    let mut kept = handles::WorkerRequests::default();
    kept.for_worker("http://w")
        .insert(Arc::from("m"), metrics::Counter::noop());
    Metrics::remove_worker_metrics("http://gone");
    assert!(kept.for_worker("http://w").is_empty(), "a removal");
    kept.for_worker("http://w")
        .insert(Arc::from("m"), metrics::Counter::noop());
    Metrics::worker_registered("http://new");
    assert!(kept.for_worker("http://w").is_empty(), "a registration");
}

/// The cost of the router's per-request records with and without the
/// kept handles, for the Numbers table: the request counter, the
/// worker counter (through the per-worker store) and the duration
/// histogram of a unary request, and the stream-timing set on top for a
/// streamed one; run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn time_per_router_record() {
    use std::time::Instant;
    const ROUNDS: u32 = 200_000;
    const WORKER: &str = "grpc://worker-1:20000";
    const MODEL: &str = "model-a";
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        None,
        None,
    )));
    let recorder = WorkerSeriesRecorder::new(PrometheusBuilder::new().build_recorder(), workers);
    let request = |streaming: &'static str| {
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            MODEL,
            metrics_labels::ENDPOINT_CHAT,
            streaming,
        );
        Metrics::record_worker_request(WORKER, MODEL);
    };
    let duration = || {
        Metrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            MODEL,
            metrics_labels::ENDPOINT_CHAT,
            Duration::from_millis(20),
        );
    };
    let unary = || {
        request(STREAMING_FALSE);
        duration();
    };
    let streamed = || {
        request(STREAMING_TRUE);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: MODEL,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: Some(Duration::from_millis(50)),
            generation_duration: Duration::from_secs(1),
            input_tokens: Some(128),
            cached_tokens: Some(64),
            output_tokens: 128,
        });
        duration();
    };
    let time = |label: &str| {
        metrics::with_local_recorder(&recorder, || {
            for (name, records) in [
                ("unary (3 records)", &unary as &dyn Fn()),
                ("streamed (8 records)", &streamed as &dyn Fn()),
            ] {
                let start = Instant::now();
                for _ in 0..ROUNDS {
                    records();
                }
                let per_request = start.elapsed().as_nanos() as f64 / f64::from(ROUNDS);
                println!("ns per {name} request, {label}: {per_request:.0}");
            }
        });
    };
    time("keys built per record");
    handles::force_on_this_thread(Some(true));
    time("handles kept per worker");
    handles::reset();
}

/// The same unary records from eight threads at once, as runtime
/// workers record them: the interner's and the registry's lock traffic
/// between workers, which one thread cannot show. Thread time per
/// request (the wall time of the eight threads, times eight, over the
/// requests); run with `--ignored --nocapture` on eight free cores.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn time_per_router_record_contended() {
    use std::time::Instant;
    const ROUNDS: u32 = 400_000;
    const THREADS: u32 = 8;
    const WORKER: &str = "grpc://worker-1:20000";
    const MODEL: &str = "model-a";
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        None,
        None,
    )));
    let recorder = WorkerSeriesRecorder::new(PrometheusBuilder::new().build_recorder(), workers);
    let unary = || {
        Metrics::record_router_request(
            metrics_labels::ROUTER_GRPC,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            MODEL,
            metrics_labels::ENDPOINT_CHAT,
            STREAMING_FALSE,
        );
        Metrics::record_worker_request(WORKER, MODEL);
        Metrics::record_router_duration(
            metrics_labels::ROUTER_GRPC,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_GRPC,
            MODEL,
            metrics_labels::ENDPOINT_CHAT,
            Duration::from_millis(20),
        );
    };
    let time = |label: &str, keep: bool| {
        let start = Instant::now();
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    handles::force_on_this_thread(Some(keep));
                    metrics::with_local_recorder(&recorder, || {
                        for _ in 0..ROUNDS / THREADS {
                            unary();
                        }
                    });
                });
            }
        });
        let per_request =
            start.elapsed().as_nanos() as f64 * f64::from(THREADS) / f64::from(ROUNDS);
        println!(
            "ns of thread time per unary request, {THREADS} threads at once, {label}: {per_request:.0}"
        );
    };
    time("keys built per record", false);
    time("handles kept per worker", true);
}
