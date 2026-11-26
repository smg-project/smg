use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::atomic::{AtomicUsize, Ordering},
};

use super::*;
use crate::testing::http_get;

fn identity(ranks: usize) -> Identity {
    Identity {
        model: "served".to_string(),
        wire: "test",
        ranks,
        max_model_len: 100,
        matches_string_stops: true,
        processes_media: true,
    }
}

fn snapshot() -> Snapshot {
    Snapshot {
        active: 0,
        state: Some(ServicerState::Serving),
    }
}

fn sample(body: &str, name: &str, labels: &str) -> Option<f64> {
    let prefix = format!("{name}{{");
    body.lines()
        .find(|line| {
            line.starts_with(&prefix) && labels.split(',').all(|label| line.contains(label))
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
}

#[test]
fn the_token_count_buckets_follow_the_1_2_5_series_up_to_the_context_length() {
    assert_eq!(
        one_two_five_buckets(100),
        [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    );
    assert_eq!(one_two_five_buckets(0), [1.0]);
    assert_eq!(one_two_five_buckets(4096).len(), 11);
}

/// The servicer's own probe is none of the counters': its observation
/// records nothing, from the chunk to the drop, beside a client's that
/// does.
#[test]
fn an_uncounted_observation_records_nothing() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let probe = metrics.uncounted();
    let mut choice = probe.choice();
    assert_eq!(
        choice.chunk(1, 3, 0),
        None,
        "nothing read, nothing recorded"
    );
    choice.complete("length", 3, 0);
    choice.string_stop();
    probe.failed(Code::Internal);
    drop(choice);
    drop(probe);
    assert_eq!(
        (
            metrics.prompt_tokens_total(),
            metrics.generation_tokens_total()
        ),
        (0, 0)
    );
    let body = metrics.render(snapshot());
    for reason in ["length", "stop", "error", "abort"] {
        assert_eq!(
            sample(
                &body,
                names::REQUESTS_TOTAL,
                &format!("finish_reason=\"{reason}\"")
            ),
            Some(0.0),
            "{reason}"
        );
    }
    assert_eq!(
        sample(&body, names::REQUEST_ERRORS_TOTAL, "code=\"internal\""),
        None,
        "the error counter was never touched"
    );
    let request = metrics.request(Instant::now(), Some(0), 3, Some(8), 1);
    let mut choice = request.choice();
    assert!(choice.chunk(1, 3, 0).is_some(), "a client's is read");
    drop(choice);
    drop(request);
    assert_eq!(
        (
            metrics.prompt_tokens_total(),
            metrics.generation_tokens_total()
        ),
        (3, 1)
    );
}

#[test]
fn a_request_records_its_families_when_the_last_holder_drops() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let arrival = Instant::now();
    let request = metrics.request(arrival, Some(0), 3, Some(8), 1);
    let mut choice = request.choice();
    assert!(choice.chunk(1, 3, 1).is_some());
    choice.chunk(2, 3, 1);
    choice.complete("length", 3, 1);
    drop(choice);
    drop(request);
    let body = metrics.render(snapshot());
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"length\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::PROMPT_TOKENS_TOTAL, "rank=\"0\""),
        Some(3.0)
    );
    assert_eq!(
        sample(&body, names::PROMPT_TOKENS_CACHED_TOTAL, "rank=\"0\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::GENERATION_TOKENS_TOTAL, "rank=\"0\""),
        Some(3.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_count", names::TIME_TO_FIRST_TOKEN_SECONDS),
            "rank=\"0\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_count", names::INTER_TOKEN_LATENCY_SECONDS),
            "rank=\"0\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_count", names::REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS),
            "rank=\"0\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_count", names::E2E_REQUEST_LATENCY_SECONDS),
            "rank=\"0\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_sum", names::REQUEST_PROMPT_TOKENS),
            "rank=\"0\""
        ),
        Some(3.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_sum", names::REQUEST_GENERATION_TOKENS),
            "rank=\"0\""
        ),
        Some(3.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_sum", names::REQUEST_PARAMS_MAX_TOKENS),
            "rank=\"0\""
        ),
        Some(8.0)
    );
    assert_eq!(sample(&body, names::STATE, "state=\"serving\""), Some(1.0));
    assert_eq!(sample(&body, names::STATE, "state=\"starting\""), Some(0.0));
    assert_eq!(
        sample(&body, names::STATE, "state=\"warming_up\""),
        Some(0.0)
    );
    assert!(body.contains("# HELP smg_servicer_requests_total "));
}

/// Without a listener nothing is recorded or read beyond the token
/// totals; the per-token path reads no clock.
#[test]
fn without_a_listener_only_the_token_totals_count() {
    let metrics = ServicerMetrics::new(&identity(1), false).unwrap();
    let request = metrics.request(Instant::now(), Some(0), 3, Some(8), 1);
    let mut choice = request.choice();
    assert!(choice.chunk(4, 3, 1).is_none());
    choice.complete("stop", 3, 1);
    drop(choice);
    drop(request);
    assert_eq!(metrics.generation_tokens_total(), 4);
    assert_eq!(metrics.prompt_tokens_total(), 3, "the stats line's figure");
    assert_eq!(metrics.render(snapshot()), "");
}

/// The prompt goes into the totals at the first output that reports it
/// (a chunk may precede the engine's prefill figures), once per request,
/// with or without a listener.
#[test]
fn the_prompt_counts_once_at_the_first_non_zero_report() {
    for enabled in [false, true] {
        let metrics = ServicerMetrics::new(&identity(1), enabled).unwrap();
        let request = metrics.request(Instant::now(), Some(0), 3, None, 1);
        let mut choice = request.choice();
        choice.chunk(1, 0, 0);
        choice.chunk(1, 5, 2);
        choice.chunk(1, 5, 2);
        choice.complete("stop", 5, 2);
        drop(choice);
        drop(request);
        assert_eq!(metrics.prompt_tokens_total(), 5, "enabled={enabled}");
        assert_eq!(
            metrics
                .rank_totals(0)
                .prompt_tokens_cached
                .load(Ordering::Relaxed),
            2
        );
    }
}

/// A fan-out is one request: its reason is the last choice's, an abort
/// wins, and the first output of any choice is the request's.
#[test]
fn a_fan_out_counts_one_request_with_the_last_reason_and_an_abort_winning() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let request = metrics.request(Instant::now(), Some(0), 3, None, 2);
    let mut first = request.choice();
    let mut second = request.choice();
    first.chunk(1, 3, 0);
    second.chunk(1, 3, 0);
    first.complete("stop", 3, 0);
    second.complete("length", 3, 0);
    drop((first, second, request));
    let request = metrics.request(Instant::now(), Some(0), 3, None, 2);
    let mut first = request.choice();
    let mut second = request.choice();
    first.ended("abort");
    second.complete("stop", 3, 0);
    drop((first, second, request));
    let body = metrics.render(snapshot());
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"length\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"abort\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"stop\""),
        Some(0.0)
    );
    assert_eq!(
        sample(&body, names::PROMPT_TOKENS_TOTAL, "rank=\"0\""),
        Some(6.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_count", names::TIME_TO_FIRST_TOKEN_SECONDS),
            "rank=\"0\""
        ),
        Some(2.0)
    );
}

/// A stream dropped with no choice ended, a cut and a failure each
/// have their reason; a failure also counts under its code.
#[test]
fn a_dropped_cut_or_failed_request_counts_under_its_reason() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    drop(metrics.request(Instant::now(), Some(0), 3, None, 1));
    let cut = metrics.request(Instant::now(), Some(0), 3, None, 1);
    cut.cut();
    drop(cut);
    let failed = metrics.request(Instant::now(), Some(0), 3, None, 1);
    failed.failed(Code::Internal);
    failed.failed(Code::Internal);
    drop(failed);
    metrics.request_not_started(&Status::already_exists("dup"));
    metrics.request_not_started(&Status::unavailable(DRAINING_MESSAGE));
    metrics.request_not_started(&Status::invalid_argument("bad"));
    metrics.request_not_started(&Status::unavailable("engine is still connecting"));
    let body = metrics.render(snapshot());
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"abort\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"cut\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUESTS_TOTAL, "finish_reason=\"error\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUEST_ERRORS_TOTAL, "code=\"internal\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUEST_ERRORS_TOTAL, "code=\"unavailable\""),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &body,
            names::REQUESTS_REFUSED_TOTAL,
            "reason=\"duplicate_id\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUESTS_REFUSED_TOTAL, "reason=\"draining\""),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &body,
            names::REQUESTS_REFUSED_TOTAL,
            "reason=\"invalid_argument\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::REQUESTS_REFUSED_TOTAL, "reason=\"media\""),
        Some(0.0)
    );
}

/// Every rank has its own series; an index past the configured width
/// lands under `other`, which exists only once used.
#[test]
fn ranks_have_their_own_series_and_an_unannounced_one_lands_under_other() {
    let metrics = ServicerMetrics::new(&identity(2), true).unwrap();
    let before = metrics.render(snapshot());
    assert!(!before.contains("rank=\"other\""));
    for rank in [Some(0), Some(1), Some(1), Some(7), None] {
        let request = metrics.request(Instant::now(), rank, 3, None, 1);
        let mut choice = request.choice();
        choice.chunk(1, 3, 0);
        choice.complete("stop", 3, 0);
    }
    let body = metrics.render(snapshot());
    let stops = |rank: &str| {
        sample(
            &body,
            names::REQUESTS_TOTAL,
            &format!("finish_reason=\"stop\",model=\"served\",rank=\"{rank}\""),
        )
    };
    assert_eq!(stops("0"), Some(1.0));
    assert_eq!(stops("1"), Some(2.0));
    assert_eq!(stops("other"), Some(2.0));
    assert_eq!(metrics.generation_tokens_total(), 5);
}

#[test]
fn the_inter_token_sampler_drops_and_counts_above_the_bound() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    for _ in 0..(ITL_PENDING_BOUND + 10) {
        metrics.inter_token(0, Duration::from_millis(1));
    }
    assert_eq!(metrics.inter_token_dropped.load(Ordering::Relaxed), 10);
    assert_eq!(
        metrics.inter_token_pending.load(Ordering::Relaxed),
        ITL_PENDING_BOUND
    );
    let body = metrics.render(snapshot());
    assert_eq!(
        sample(
            &body,
            names::METRICS_DROPPED_TOTAL,
            "family=\"inter_token_latency\""
        ),
        Some(10.0)
    );
    assert_eq!(
        sample(
            &body,
            &format!("{}_count", names::INTER_TOKEN_LATENCY_SECONDS),
            "rank=\"0\""
        ),
        Some(f64::from(ITL_PENDING_BOUND))
    );
    // The scrape drained the samples: the sampler starts over.
    assert_eq!(metrics.inter_token_pending.load(Ordering::Relaxed), 0);
    metrics.inter_token(0, Duration::from_millis(1));
    assert_eq!(metrics.inter_token_dropped.load(Ordering::Relaxed), 10);
}

/// The render time of a scrape at one rank and at eight (the handle
/// sets of a data-parallel group), printed for the Numbers table: a
/// scrape runs on the listener's thread, never on the request runtime,
/// so this is a cost in milliseconds per scrape, not per request.
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the render times are the test's report"
)]
fn a_scrape_renders_in_milliseconds_at_one_and_eight_ranks() {
    for ranks in [1usize, 8] {
        let metrics = ServicerMetrics::new(&identity(ranks), true).unwrap();
        for rank in 0..ranks {
            for _ in 0..16 {
                let request = metrics.request(Instant::now(), Some(rank as u32), 3, Some(8), 1);
                let mut choice = request.choice();
                choice.chunk(1, 3, 1);
                choice.chunk(1, 3, 1);
                choice.complete("stop", 3, 1);
            }
        }
        let mut renders = Vec::new();
        let mut bytes = 0;
        for _ in 0..20 {
            let started = Instant::now();
            bytes = metrics.render(snapshot()).len();
            renders.push(started.elapsed());
        }
        renders.sort();
        eprintln!(
            "scrape at {ranks} rank(s): median {:.3} ms, max {:.3} ms, {bytes} bytes",
            renders[renders.len() / 2].as_secs_f64() * 1e3,
            renders[renders.len() - 1].as_secs_f64() * 1e3
        );
        assert!(renders[renders.len() / 2] < Duration::from_millis(100));
    }
}

/// Two servicers in one process keep two registries: nothing is
/// process-global.
#[test]
fn a_process_hosting_two_servicers_keeps_two_registries() {
    let first = ServicerMetrics::new(&identity(1), true).unwrap();
    let second = ServicerMetrics::new(
        &Identity {
            model: "other".to_string(),
            ..identity(1)
        },
        true,
    )
    .unwrap();
    drop(first.request(Instant::now(), Some(0), 1, None, 1));
    let first_body = first.render(snapshot());
    let second_body = second.render(snapshot());
    assert_eq!(
        sample(
            &first_body,
            names::REQUESTS_TOTAL,
            "finish_reason=\"abort\",model=\"served\""
        ),
        Some(1.0)
    );
    assert_eq!(
        sample(
            &second_body,
            names::REQUESTS_TOTAL,
            "finish_reason=\"abort\",model=\"other\""
        ),
        Some(0.0)
    );
    assert!(!second_body.contains("model=\"served\""));
}

#[test]
fn the_listener_serves_a_scrape_and_an_ipv6_host_binds_with_or_without_brackets() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let probe: Probe = Arc::new(snapshot);
    let listener =
        MetricsListener::start("127.0.0.1:0", Arc::clone(&metrics), Arc::clone(&probe)).unwrap();
    let (status, body) = http_get(listener.address(), "/metrics");
    assert_eq!(status, 200);
    assert!(body.contains("smg_servicer_state{"));
    assert_eq!(http_get(listener.address(), "/other").0, 404);
    listener.stop(Duration::from_secs(5)).unwrap();
    // Nothing to assert on a host without an IPv6 loopback.
    if std::net::TcpListener::bind("[::1]:0").is_err() {
        return;
    }
    for bind in ["[::1]:0", "::1:0"] {
        let listener =
            MetricsListener::start(bind, Arc::clone(&metrics), Arc::clone(&probe)).unwrap();
        assert!(
            listener.address().is_ipv6(),
            "{bind} bound {}",
            listener.address()
        );
        assert_eq!(http_get(listener.address(), "/metrics").0, 200);
        listener.stop(Duration::from_secs(5)).unwrap();
    }
}

#[test]
fn a_busy_metrics_port_is_a_startup_error() {
    let busy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let error = MetricsListener::start(
        &busy.local_addr().unwrap().to_string(),
        metrics,
        Arc::new(snapshot),
    )
    .err()
    .expect("a busy port is refused");
    let text = error.to_string();
    assert!(text.contains("metrics listener: failed to bind"), "{text}");
}

/// A scraper holding a connection does not hold the stop: the listener
/// closes at once, inside the budget, and the port refuses after.
#[test]
fn stop_closes_the_listener_inside_the_budget_with_a_scraper_connected() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let listener = MetricsListener::start("127.0.0.1:0", metrics, Arc::new(snapshot)).unwrap();
    let address = listener.address();
    let mut held = TcpStream::connect(address).unwrap();
    write!(held, "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n").unwrap();
    let mut first = [0u8; 16];
    held.read_exact(&mut first).unwrap();
    assert!(first.starts_with(b"HTTP/1.1 200"));
    let started = Instant::now();
    listener.stop(Duration::from_secs(5)).unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
    held.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut rest = Vec::new();
    let _ = held.read_to_end(&mut rest);
    assert!(TcpStream::connect(address).is_err(), "the port is closed");
}

/// A probe that panics fails that scrape alone with a 500; the next
/// scrape renders again and the failure is counted.
#[test]
fn a_failed_scrape_answers_500_once_and_the_next_renders_again() {
    let metrics = ServicerMetrics::new(&identity(1), true).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let probe_calls = Arc::clone(&calls);
    let probe: Probe = Arc::new(move || {
        assert_ne!(
            probe_calls.fetch_add(1, Ordering::Relaxed),
            0,
            "the first scrape fails"
        );
        snapshot()
    });
    let listener = MetricsListener::start("127.0.0.1:0", metrics, probe).unwrap();
    let (status, body) = http_get(listener.address(), "/metrics");
    assert_eq!(status, 500);
    assert!(body.contains("the next scrape renders again"));
    let (status, body) = http_get(listener.address(), "/metrics");
    assert_eq!(status, 200);
    assert_eq!(
        sample(&body, names::METRICS_SCRAPES_TOTAL, "code=\"500\""),
        Some(1.0)
    );
    assert_eq!(
        sample(&body, names::METRICS_SCRAPES_TOTAL, "code=\"200\""),
        Some(0.0),
        "the current scrape counts once it is rendered"
    );
    let (_, body) = http_get(listener.address(), "/metrics");
    assert_eq!(
        sample(&body, names::METRICS_SCRAPES_TOTAL, "code=\"200\""),
        Some(1.0)
    );
    assert!(sample(&body, names::METRICS_SCRAPE_SECONDS, "wire=\"test\"").is_some());
    listener.stop(Duration::from_secs(5)).unwrap();
}
