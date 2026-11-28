use std::{
    net::TcpListener,
    pin::pin,
    sync::atomic::{AtomicU8, Ordering},
    task::Poll,
};

use metrics_exporter_prometheus::PrometheusBuilder;
use smg_grand_central::{
    document::{AppliedSpec, DecodePick, PolicySpec, ShadowSpec},
    engine::{Candidate, Decision, Members, RequestFacts, Scorer, TieBand, POLICY_FAULTS_HELP},
    stages::{Lifecycle, StageRegistry},
    tracking::{EventSet, Lease},
};
use smg_worker::{
    identity::basic::Worker,
    metrics::writers::{Metrics as WorkerMetrics, WorkerHandles},
    registry::{event::WorkerEvent, WorkerId as ObjectId},
    test_support::{gone, held, ready_worker},
};
use tokio::sync::broadcast::Receiver;

use super::{
    test_support::{
        assert_metric, fault_kinds, render_with_recorder, shadow_kinds, shadowing_spec, side_kinds,
        sorted_series,
    },
    *,
};
use crate::{
    config::types::{PolicyConfig, RouterConfig},
    policies::PolicyRegistry,
    routers::common::{
        placement::{
            decode_again, select_pair, select_single,
            tests::{engine_on, one_pair, pairs_of, MODEL},
            PlacementInputs, Stated,
        },
        snapshot::{RoutingPool, WorkerRegistry},
    },
};

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
    assert!(
        rendered.lines().any(|line| {
            line.starts_with("smg_tokenizer_encode_fallbacks_total{")
                && line.contains("reason=\"piece_declined\"")
        }),
        "missing reason piece_declined: {rendered}"
    );
}

/// One ready worker of the model `m`, and a gateway's policies with an
/// engine over a policy document whose second binding names a shadow
/// ([`shadowing_spec`]). `place` has the engine place requests of that
/// model.
struct Shadowing {
    workers: WorkerRegistry,
    policies: PolicyRegistry,
    engine: &'static InProcess,
}

impl Shadowing {
    fn new() -> Self {
        let workers = WorkerRegistry::new();
        let worker = ready_worker("http://h:1", "m");
        workers.register(worker).expect("the worker registers");
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_engine(Box::leak(Box::new(InProcess::new(shadowing_spec()))));
        let engine = policies.engine().expect("the engine that was set");
        Self {
            workers,
            policies,
            engine,
        }
    }

    fn place(&self, requests: usize) {
        for _ in 0..requests {
            let (inputs, pool) = (PlacementInputs::default(), RoutingPool::HttpRegular);
            let placed = select_single(
                &self.workers,
                &self.policies,
                "m",
                pool,
                None,
                inputs,
                Stated::nothing,
            );
            assert!(placed.is_ok(), "the one worker is selectable");
        }
    }
}

/// Grand Central's shadow counts on a scrape. The one binding that
/// names a shadow has a zero series of each kind from the first scrape, by
/// its place in the policy document and beside a single request. The
/// binding ahead of it names none and has no series. A shadow that places
/// on the one worker agrees with the live decision, and each such decision
/// is added once, however many scrapes read it. A count read under another
/// policy document starts from zero and is added whole: the series goes on
/// from where it stood.
///
/// The series is new with Grand Central: a classic gateway counts no
/// shadow, and publishes no such family.
#[test]
fn the_engines_shadow_counts_are_published_from_zero_and_added_by_their_growth() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        init_metrics();
        let published = ShadowsPublished::default();
        let scraped = |shadowing: &Shadowing| {
            record_policy_shadow_activity(shadowing.engine, &published);
            shadow_kinds(&handle.render())
        };
        let agreed = |count| [Some(count), Some(0), Some(0), Some(0)];
        assert!(!handle
            .render()
            .contains("smg_policy_shadow_decisions_total"));

        let first = Shadowing::new();
        assert_eq!(scraped(&first), agreed(0));
        let rendered = handle.render();
        let help = "# HELP smg_policy_shadow_decisions_total Shadow decisions of Grand Central";
        assert!(rendered.contains(help));
        let series = |line: &&str| line.starts_with("smg_policy_shadow_decisions_total{");
        assert_eq!(rendered.lines().filter(series).count(), 4, "{rendered}");
        first.place(3);
        assert_eq!(scraped(&first), agreed(3));
        assert_eq!(scraped(&first), agreed(3));
        first.place(2);
        assert_eq!(scraped(&first), agreed(5));

        let second = Shadowing::new();
        second.place(2);
        assert_eq!(scraped(&second), agreed(7));
        assert_eq!(scraped(&second), agreed(7));
    });
}

/// A pair binding's shadows on a scrape, by side. The policy document
/// names a shadow for each side of its pair's binding and picks the decode
/// behind its prefill, and the gateway's own loader takes it. From the
/// first scrape the binding has a zero series of each kind beside its
/// prefill side, beside its decode side and beside its decode decided
/// again, and none beside a single request. A pair placed adds one
/// decision beside each side, and its decode asked again adds one under
/// `decode-again`. Each shadow is its live profile's twin over the one
/// worker of its side, so it agrees.
///
/// Classic picks a pair's decode once, with its prefill
/// (`model_gateway/src/routers/common/placement.rs:3071`, on main
/// `9bfcf9bd2`). It asks no decode again and runs no second policy beside
/// a side, so the series is the engine's own.
#[test]
fn a_pair_bindings_shadow_counts_are_published_by_side() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        init_metrics();
        let pairs = PolicySpec::pair_presets("round_robin", "round_robin");
        let mut document = pairs.expect("the presets of a pair");
        document.plans.pd_pair.decode = DecodePick::AfterPrefill;
        let sides = ["prefill", "decode"].map(|side| {
            let twin = document.profiles[side].clone();
            let next = format!("{side}-next");
            document.profiles.insert(next.clone(), twin);
            (side.to_owned(), next)
        });
        document.bindings[0].shadow = Some(ShadowSpec::Sides(sides.into()));
        let config = RouterConfig::default();
        let spec = config.hosted_policy(&document, &StageRegistry::builtin());
        let registry = one_pair();
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        engine_on(&policies, spec.expect("the gateway loads a pair's shadows"));
        let engine = policies.engine().expect("the engine that was set");

        let published = ShadowsPublished::default();
        // The binding's series beside a single request, beside each side
        // and beside the decode decided again.
        let scraped = || {
            record_policy_shadow_activity(engine, &published);
            let rendered = handle.render();
            ["single", "prefill", "decode", "decode-again"]
                .map(|side| side_kinds(&rendered, 0, side))
        };
        let agreed = |count| [Some(count), Some(0), Some(0), Some(0)];
        assert_eq!(scraped(), [[None; 4], agreed(0), agreed(0), agreed(0)]);

        let pairs = || pairs_of(&registry, &policies);
        let pair = select_pair(
            &registry,
            &policies,
            MODEL,
            &pairs(),
            None,
            None,
            PlacementInputs::default(),
            Stated::nothing,
        );
        let pair = pair.expect("the engine places the pair");
        assert_eq!(scraped(), [[None; 4], agreed(1), agreed(1), agreed(0)]);

        // The prefill is sent and its guard let go, and the decode is asked
        // again behind it, as a route that sends the sides in turn asks.
        drop(pair.prefill.guard(None));
        let ahead = pair.decode.ahead(None);
        let (moved, guard) = decode_again(
            &registry,
            &policies,
            MODEL,
            &pairs,
            None,
            PlacementInputs::default(),
            &pair.decode,
            ahead,
            false,
        );
        assert!(moved.is_none(), "the one decode stands");
        assert_eq!(scraped(), [[None; 4], agreed(1), agreed(1), agreed(1)]);
        drop(pair.decode.sent(guard));
    });
}

/// The stage of the fault counts' case, as its table and its document name
/// it.
const FAULTY: &str = "of-the-case";

/// Where [`PanicsAt`] panics: nowhere, at its score, or at the issue of a
/// request's lease.
const NOWHERE: u8 = 0;
const AT_ITS_SCORE: u8 = 1;
const AT_AN_ISSUE: u8 = 2;

/// A scorer that panics where the case says. It takes every word of a
/// lease, so the engine leases each pick under it and tells it the issue.
#[derive(Debug)]
struct PanicsAt(Arc<AtomicU8>);

impl Lifecycle for PanicsAt {
    fn hears(&self) -> EventSet {
        EventSet::LEASE
    }

    fn on_decision_issued(&self, _: &Lease, _: &Decision) {
        let at = self.0.load(Ordering::Relaxed);
        assert_ne!(at, AT_AN_ISSUE, "the case's panic at an issue");
    }
}

impl Scorer for PanicsAt {
    fn name(&self) -> &'static str {
        FAULTY
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        let at = self.0.load(Ordering::Relaxed);
        assert_ne!(at, AT_ITS_SCORE, "the case's panic at a score");
        out.fill(1.0);
        TieBand::NORMALIZED
    }
}

/// A host's registry whose own code panics where the engine's upkeep
/// subscribes to it.
struct PanicsAtItsEvents;

impl Members for PanicsAtItsEvents {
    fn events(&self) -> Receiver<WorkerEvent> {
        panic!("the case's panic in the host's own code")
    }

    fn standing(&self) -> Vec<(ObjectId, Arc<dyn Worker>)> {
        Vec::new()
    }
}

/// Two ready workers of the model `m`, and a gateway's policies with an
/// engine whose one scorer is [`PanicsAt`] and whose host's registry is
/// [`PanicsAtItsEvents`]. `at` says where the scorer panics next.
struct Faulty {
    workers: WorkerRegistry,
    policies: PolicyRegistry,
    engine: &'static InProcess,
    at: Arc<AtomicU8>,
}

impl Faulty {
    fn new() -> Self {
        let workers = WorkerRegistry::new();
        for url in ["http://h:1", "http://h:2"] {
            let worker = ready_worker(url, "m");
            workers.register(worker).expect("the worker registers");
        }
        let at = Arc::new(AtomicU8::new(NOWHERE));
        let asked = Arc::clone(&at);
        let mut stages = StageRegistry::builtin();
        stages
            .scorer(FAULTY, move || Box::new(PanicsAt(Arc::clone(&asked))))
            .expect("a free name");
        let mut document = PolicySpec::preset("round_robin").expect("a stock preset");
        for profile in document.profiles.values_mut() {
            profile.score = vec![FAULTY.into()];
            profile.pick = "best".into();
        }
        let spec = AppliedSpec::validate_in(&document, &stages, &[]);
        let spec = spec.expect("the stages hold the scorer");
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        policies.set_engine(Box::leak(Box::new(InProcess::new(spec))));
        let engine = policies.engine().expect("the engine that was set");
        engine.set_registry(Arc::new(PanicsAtItsEvents));
        Self {
            workers,
            policies,
            engine,
            at,
        }
    }

    /// Whether the engine places one request of the model.
    fn places(&self) -> bool {
        let (inputs, pool) = (PlacementInputs::default(), RoutingPool::HttpRegular);
        let placed = select_single(
            &self.workers,
            &self.policies,
            "m",
            pool,
            None,
            inputs,
            Stated::nothing,
        );
        placed.is_ok()
    }

    /// One poll of the engine's own task: its first upkeep runs at that
    /// poll, and subscribes to the host's registry.
    fn runs_one_upkeep(&self) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("a runtime");
        runtime.block_on(async {
            let mut task = pin!(self.engine.run());
            assert!(matches!(futures::poll!(task.as_mut()), Poll::Pending));
        });
    }
}

/// The panics Grand Central held and counted, on a scrape. Each kind
/// has a zero series from the first scrape. A decision that panics is
/// refused, and raises `decision`. A stage that panics at the issue of a
/// request's lease costs that request nothing, and raises `event`. A panic
/// of the host's own code in a step of the engine's upkeep raises `upkeep`.
/// Each series stands at its count as read, however many scrapes read it.
///
/// The series is new with Grand Central: a classic gateway starts no
/// engine, and publishes no such family.
#[test]
fn the_engines_fault_counts_are_published_from_zero_each_under_its_kind() {
    let recorder = PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        init_metrics();
        let faulty = Faulty::new();
        let scraped = || {
            record_policy_fault_activity(faulty.engine);
            fault_kinds(&handle.render())
        };
        assert!(!handle.render().contains("smg_policy_faults_total"));

        assert_eq!(scraped(), [Some(0); 3]);
        let rendered = handle.render();
        let help = format!("# HELP smg_policy_faults_total {POLICY_FAULTS_HELP}\n");
        assert!(rendered.contains(&help), "{rendered}");
        let series = |line: &&str| line.starts_with("smg_policy_faults_total{");
        assert_eq!(rendered.lines().filter(series).count(), 3, "{rendered}");

        faulty.at.store(AT_ITS_SCORE, Ordering::Relaxed);
        assert!(!faulty.places(), "a decision that panicked places nothing");
        assert_eq!(scraped(), [Some(1), Some(0), Some(0)]);

        faulty.at.store(AT_AN_ISSUE, Ordering::Relaxed);
        for _ in 0..2 {
            assert!(faulty.places(), "the request keeps its decision");
        }
        assert_eq!(scraped(), [Some(1), Some(2), Some(0)]);

        faulty.at.store(NOWHERE, Ordering::Relaxed);
        for _ in 0..3 {
            faulty.runs_one_upkeep();
        }
        assert!(faulty.places(), "no panic is left behind");
        assert_eq!(scraped(), [Some(1), Some(2), Some(3)]);
        assert_eq!(scraped(), [Some(1), Some(2), Some(3)]);
    });
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
        workers.retire("http://removed-prefill", gone("http://removed-prefill"));
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

/// The first-token and per-output-token families render on their own edges
/// once the buckets are registered the way `start_prometheus` does: an 18 ms
/// per-token time lands under the 20 ms edge there and under the 25 ms edge
/// of the request-duration list `smg_router_tpot_seconds` keeps; a 120 ms
/// first token under 150 ms there and under 250 ms on `smg_router_ttft_seconds`.
#[test]
fn first_token_and_per_output_token_families_render_on_their_own_edges() {
    let mut builder = PrometheusBuilder::new();
    for (matcher, buckets) in histogram_bucket_overrides(None) {
        builder = builder
            .set_buckets_for_metric(matcher, &buckets)
            .expect("bucket override");
    }
    let recorder = builder.build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m1",
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: Some(Duration::from_millis(120)),
            dispatch_wait: None,
            dispatch_wait_after_body: None,
            generation_duration: Duration::from_millis(156),
            input_tokens: Some(10),
            cached_tokens: None,
            output_tokens: 3,
        });
    });
    let rendered = handle.render();

    let labels = "router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"";
    for (family, le, count) in [
        (REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS, "0.015", "0"),
        (REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS, "0.02", "1"),
        (TIME_TO_FIRST_TOKEN_SECONDS, "0.1", "0"),
        (TIME_TO_FIRST_TOKEN_SECONDS, "0.15", "1"),
        ("smg_router_tpot_seconds", "0.01", "0"),
        ("smg_router_tpot_seconds", "0.025", "1"),
        ("smg_router_ttft_seconds", "0.1", "0"),
        ("smg_router_ttft_seconds", "0.25", "1"),
    ] {
        let series = format!("{family}_bucket{{{labels},le=\"{le}\"}} {count}");
        assert!(
            rendered.lines().any(|l| l == series),
            "{series} missing; rendered:\n{rendered}"
        );
    }
    for (family, le) in [
        ("smg_router_tpot_seconds", "0.015"),
        ("smg_router_ttft_seconds", "0.15"),
    ] {
        let edge = format!("{family}_bucket{{{labels},le=\"{le}\"}}");
        assert!(
            !rendered.contains(&edge),
            "{edge} is not a request-duration edge; rendered:\n{rendered}"
        );
    }
    for family in [
        TIME_TO_FIRST_TOKEN_SECONDS,
        REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS,
    ] {
        let count = format!("{family}_count{{{labels}}} 1");
        assert!(
            rendered.lines().any(|l| l == count),
            "{count} missing; rendered:\n{rendered}"
        );
        let summary = format!("{family}{{");
        assert!(
            !rendered.contains(&summary),
            "{family} rendered as a summary; rendered:\n{rendered}"
        );
    }
}

/// The dispatch wait renders as a histogram on the request-duration edges
/// once the buckets are registered the way `start_prometheus` does. Its name
/// ends in `_wait_seconds`, which no suffix matcher covers: without its own
/// matcher the exporter renders a summary. A 7 ms wait lands under the 10 ms
/// edge and not under the 5 ms one, beside the TTFT it is recorded with.
#[test]
fn the_dispatch_wait_renders_on_the_request_duration_edges() {
    let mut builder = PrometheusBuilder::new();
    for (matcher, buckets) in histogram_bucket_overrides(None) {
        builder = builder
            .set_buckets_for_metric(matcher, &buckets)
            .expect("bucket override");
    }
    let recorder = builder.build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m1",
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: Some(Duration::from_millis(120)),
            dispatch_wait: Some(Duration::from_millis(7)),
            dispatch_wait_after_body: None,
            generation_duration: Duration::from_millis(156),
            input_tokens: Some(10),
            cached_tokens: None,
            output_tokens: 3,
        });
    });
    let rendered = handle.render();

    let labels = "router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"";
    for series in [
        format!("{DISPATCH_WAIT_SECONDS}_bucket{{{labels},le=\"0.005\"}} 0"),
        format!("{DISPATCH_WAIT_SECONDS}_bucket{{{labels},le=\"0.01\"}} 1"),
        format!("{DISPATCH_WAIT_SECONDS}_count{{{labels}}} 1"),
        format!("smg_router_ttft_seconds_count{{{labels}}} 1"),
    ] {
        assert!(
            rendered.lines().any(|l| l == series),
            "{series} missing; rendered:\n{rendered}"
        );
    }
    let summary = format!("{DISPATCH_WAIT_SECONDS}{{");
    assert!(
        !rendered.contains(&summary),
        "{DISPATCH_WAIT_SECONDS} rendered as a summary; rendered:\n{rendered}"
    );
}

/// The dispatch wait without the upload renders beside the
/// dispatch wait, a histogram on the same edges: its name has a matcher of
/// its own, as the dispatch wait's has, and an operator's list replaces the
/// edges of both. A 7 ms wait of which 1 ms is the body's upload lands
/// under the 10 ms edge and not under the 5 ms one in both families.
#[test]
fn the_dispatch_wait_after_body_renders_on_the_dispatch_wait_edges() {
    let mut builder = PrometheusBuilder::new();
    for (matcher, buckets) in histogram_bucket_overrides(None) {
        builder = builder
            .set_buckets_for_metric(matcher, &buckets)
            .expect("bucket override");
    }
    let recorder = builder.build_recorder();
    let handle = recorder.handle();
    metrics::with_local_recorder(&recorder, || {
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m1",
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: Some(Duration::from_millis(120)),
            dispatch_wait: Some(Duration::from_millis(7)),
            dispatch_wait_after_body: Some(Duration::from_millis(6)),
            generation_duration: Duration::from_millis(156),
            input_tokens: Some(10),
            cached_tokens: None,
            output_tokens: 3,
        });
    });
    let rendered = handle.render();

    let labels = "router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"";
    for (family, sum) in [
        (DISPATCH_WAIT_SECONDS, "0.007"),
        (DISPATCH_WAIT_AFTER_BODY_SECONDS, "0.006"),
    ] {
        for series in [
            format!("{family}_bucket{{{labels},le=\"0.005\"}} 0"),
            format!("{family}_bucket{{{labels},le=\"0.01\"}} 1"),
            format!("{family}_count{{{labels}}} 1"),
            format!("{family}_sum{{{labels}}} {sum}"),
        ] {
            assert!(
                rendered.lines().any(|l| l == series),
                "{series} missing; rendered:\n{rendered}"
            );
        }
        let summary = format!("{family}{{");
        assert!(
            !rendered.contains(&summary),
            "{family} rendered as a summary; rendered:\n{rendered}"
        );
    }

    let after_body = Matcher::Full(String::from(DISPATCH_WAIT_AFTER_BODY_SECONDS));
    let dispatch_wait = Matcher::Full(String::from(DISPATCH_WAIT_SECONDS));
    let defaults = histogram_bucket_overrides(None);
    assert_eq!(defaults[&after_body], defaults[&dispatch_wait]);
    let custom = vec![0.5, 1.0];
    let overridden = histogram_bucket_overrides(Some(custom.clone()));
    assert_eq!(overridden[&after_body], custom);
}

/// An operator's `--prometheus-duration-buckets` list replaces the
/// request-duration edges, the ones `smg_router_ttft_seconds` and
/// `smg_router_tpot_seconds` share with it, and leaves the first-token and
/// per-output-token families on their own edges.
#[test]
fn duration_bucket_override_leaves_the_token_timing_families_alone() {
    let custom = vec![0.5, 1.0];
    let overridden = histogram_bucket_overrides(Some(custom.clone()));
    for suffix in ["duration_seconds", "ttft_seconds", "tpot_seconds"] {
        assert_eq!(
            overridden[&Matcher::Suffix(String::from(suffix))],
            custom,
            "{suffix}"
        );
    }
    // The dispatch wait ends in `_wait_seconds`: it is matched by its full
    // name and follows the same list.
    let dispatch_wait = Matcher::Full(String::from(DISPATCH_WAIT_SECONDS));
    assert_eq!(overridden[&dispatch_wait], custom);
    let defaults = histogram_bucket_overrides(None);
    let duration = &defaults[&Matcher::Suffix(String::from("duration_seconds"))];
    assert_eq!(&defaults[&dispatch_wait], duration);
    assert_eq!(duration.len(), 29);
    assert_eq!(duration.first(), Some(&0.001));
    assert_eq!(duration.last(), Some(&7200.0));
    assert_eq!(
        &defaults[&Matcher::Suffix(String::from("ttft_seconds"))],
        duration
    );
    assert_eq!(
        &defaults[&Matcher::Suffix(String::from("tpot_seconds"))],
        duration
    );
    for (family, edges) in [
        (TIME_TO_FIRST_TOKEN_SECONDS, TIME_TO_FIRST_TOKEN_BUCKETS),
        (
            REQUEST_TIME_PER_OUTPUT_TOKEN_SECONDS,
            REQUEST_TIME_PER_OUTPUT_TOKEN_BUCKETS,
        ),
    ] {
        let matcher = Matcher::Full(String::from(family));
        assert_eq!(defaults[&matcher], edges, "{family}");
        assert_eq!(overridden[&matcher], edges, "{family} under the override");
        assert!(
            edges[0] > 0.0 && edges.windows(2).all(|w| w[0] < w[1]),
            "{family}: edges ascend strictly from above zero"
        );
    }
    let within =
        |edges: &[f64], lo: f64, hi: f64| edges.iter().filter(|e| (lo..=hi).contains(*e)).count();
    assert_eq!(within(duration, 0.005, 0.05), 4);
    assert_eq!(
        within(REQUEST_TIME_PER_OUTPUT_TOKEN_BUCKETS, 0.005, 0.05),
        7
    );
    assert_eq!(within(duration, 0.05, 0.5), 4);
    assert_eq!(within(TIME_TO_FIRST_TOKEN_BUCKETS, 0.05, 0.5), 7);
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

/// The router's first-token and generation families say on `/metrics` where
/// their clock starts: at the answering attempt's RPC dispatch (the prefill
/// dispatch of a pair, after a pushed encode call), so a refused attempt's
/// wait and its backoff are in the request duration (the backoff alone in the
/// backoff histogram, the attempt in the retry count), not in them.
#[test]
fn router_stream_families_name_the_attempt_anchor() {
    let (rendered, ()) = with_test_recorder(|| {
        init_metrics();
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m",
            endpoint: "chat",
            ttft: Some(Duration::from_millis(5)),
            dispatch_wait: None,
            dispatch_wait_after_body: None,
            generation_duration: Duration::from_millis(9),
            input_tokens: None,
            cached_tokens: None,
            output_tokens: 2,
        });
    });
    let help = |family: &str| -> String {
        rendered
            .lines()
            .find_map(|line| line.strip_prefix(&format!("# HELP {family} ")))
            .unwrap_or_else(|| panic!("no # HELP for {family}; rendered:\n{rendered}"))
            .to_owned()
    };
    for family in [
        "smg_router_ttft_seconds",
        TIME_TO_FIRST_TOKEN_SECONDS,
        "smg_router_generation_duration_seconds",
    ] {
        let text = help(family);
        assert!(
            text.contains("answering attempt's RPC dispatch"),
            "{family}: {text}"
        );
    }
    for family in [
        "smg_router_ttft_seconds",
        "smg_router_generation_duration_seconds",
    ] {
        let text = help(family);
        assert!(
            text.contains("smg_router_request_duration_seconds")
                && text.contains("after the encode call when encode pushes")
                && text.contains("smg_worker_retry_backoff_seconds")
                && text.contains("smg_worker_retries_total"),
            "{family}: {text}"
        );
    }
}

/// The dispatch wait says on `/metrics` what its clock is (the request's
/// entry at the edge to the answering attempt's RPC dispatch, where the TTFT
/// clock starts), which answers it observes (streamed and collected, under
/// the TTFT's labels) and which it does not (a Responses answer on the
/// Harmony pipeline, a background Responses request that is not streamed),
/// that for a streamed answer the two add up to the time to first token from
/// the request's receipt, and that its count also holds the collected
/// answers. It says where a later turn of a tool loop counts from: that
/// turn's start.
#[test]
fn the_dispatch_wait_help_names_its_clock_its_answers_and_its_sum_with_the_ttft() {
    let (rendered, ()) = with_test_recorder(|| {
        init_metrics();
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m",
            endpoint: "chat",
            ttft: Some(Duration::from_millis(5)),
            dispatch_wait: Some(Duration::from_millis(2)),
            dispatch_wait_after_body: None,
            generation_duration: Duration::from_millis(9),
            input_tokens: None,
            cached_tokens: None,
            output_tokens: 2,
        });
    });
    let help = rendered
        .lines()
        .find_map(|line| line.strip_prefix(&format!("# HELP {DISPATCH_WAIT_SECONDS} ")))
        .unwrap_or_else(|| panic!("no # HELP for {DISPATCH_WAIT_SECONDS}; rendered:\n{rendered}"));
    for phrase in [
        "of streamed and collected answers by router_type, backend_type, model, endpoint",
        "a Responses answer on the Harmony pipeline records none, streamed or collected, nor does a background Responses request that is not streamed",
        "from the request's entry at the edge to the answering attempt's RPC dispatch",
        "the prefill dispatch of a prefill-decode pair, after the encode call when encode pushes",
        "a refused attempt's wait and its backoff are here",
        "observed once per answer, under the labels of smg_router_ttft_seconds",
        "for a streamed answer the two add up to the time to first token from the request's receipt",
        "this family's count also holds the collected answers and is not the count of smg_router_ttft_seconds",
        "a Responses tool loop observes one per turn, streamed or collected",
        "a later turn's from that turn's start",
    ] {
        assert!(help.contains(phrase), "{phrase:?} is not in: {help}");
    }
}

/// The dispatch wait without the upload says on `/metrics` what it takes off
/// the dispatch wait (the time the read of the body waited for the client),
/// that it is observed where the dispatch wait is under the same labels and
/// buckets, what both families hold, what the dispatch wait alone holds and
/// which requests read the same value in both.
#[test]
fn the_dispatch_wait_after_body_help_names_what_it_takes_off_and_what_each_family_holds() {
    let (rendered, ()) = with_test_recorder(|| {
        init_metrics();
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: "m",
            endpoint: "chat",
            ttft: Some(Duration::from_millis(5)),
            dispatch_wait: Some(Duration::from_millis(2)),
            dispatch_wait_after_body: Some(Duration::from_millis(1)),
            generation_duration: Duration::from_millis(9),
            input_tokens: None,
            cached_tokens: None,
            output_tokens: 2,
        });
    });
    let family = DISPATCH_WAIT_AFTER_BODY_SECONDS;
    let help = rendered
        .lines()
        .find_map(|line| line.strip_prefix(&format!("# HELP {family} ")))
        .unwrap_or_else(|| panic!("no # HELP for {family}; rendered:\n{rendered}"));
    for phrase in [
        "by router_type, backend_type, model, endpoint (gRPC only; observed wherever smg_router_dispatch_wait_seconds is, under the same labels and buckets)",
        "smg_router_dispatch_wait_seconds less the time the read of the request body waited for the client",
        "authentication, the wait for an admission permit, the run of WASM modules, parsing, tokenizing, templating, worker selection, a refused attempt's wait and its backoff are in both",
        "the body's upload is in smg_router_dispatch_wait_seconds alone",
        "a body that came whole and a later turn of a Responses tool loop take nothing off and observe the same value in both",
        "a multipart upload is measured as any other body",
    ] {
        assert!(help.contains(phrase), "{phrase:?} is not in: {help}");
    }
}

/// The dispatch wait without the upload is recorded beside the
/// dispatch wait: once per answer that carries one, streamed or collected,
/// under the same four labels, through a key built per record and through
/// kept handles alike. An answer dispatched outside a request's entry stamp
/// carries neither wait and records neither, and the dispatch wait reads
/// what it is handed, whatever the other family reads.
#[test]
fn the_dispatch_wait_after_body_is_recorded_beside_the_dispatch_wait() {
    let records = || {
        let waits = Some((Duration::from_millis(17), Duration::from_millis(6)));
        for (model_id, ttft, waits) in [
            ("streamed", Some(Duration::from_millis(40)), waits),
            ("collected", None, waits),
            ("unstamped", Some(Duration::from_millis(40)), None),
        ] {
            Metrics::record_streaming_metrics(StreamingMetricsParams {
                router_type: metrics_labels::ROUTER_GRPC,
                backend_type: metrics_labels::BACKEND_PD,
                model_id,
                endpoint: metrics_labels::ENDPOINT_MESSAGES,
                ttft,
                dispatch_wait: waits.map(|(since_entry, _)| since_entry),
                dispatch_wait_after_body: waits.map(|(_, since_body)| since_body),
                generation_duration: Duration::from_millis(90),
                input_tokens: Some(4),
                cached_tokens: None,
                output_tokens: 2,
            });
        }
    };
    let built = render_with_recorder(records);
    let kept = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        handles::reset();
    });
    assert_eq!(sorted_series(&kept), sorted_series(&built));
    let labels = |model: &str| {
        format!("router_type=\"grpc\",backend_type=\"pd\",model=\"{model}\",endpoint=\"messages\"")
    };
    for rendered in [&built, &kept] {
        for model in ["streamed", "collected"] {
            let labels = labels(model);
            for series in [
                format!("{DISPATCH_WAIT_SECONDS}_count{{{labels}}} 1"),
                format!("{DISPATCH_WAIT_SECONDS}_sum{{{labels}}} 0.017"),
                format!("{DISPATCH_WAIT_AFTER_BODY_SECONDS}_count{{{labels}}} 1"),
                format!("{DISPATCH_WAIT_AFTER_BODY_SECONDS}_sum{{{labels}}} 0.006"),
            ] {
                assert!(
                    rendered.lines().any(|line| line == series),
                    "{series} missing; rendered:\n{rendered}"
                );
            }
        }
        let unstamped = labels("unstamped");
        for family in [DISPATCH_WAIT_SECONDS, DISPATCH_WAIT_AFTER_BODY_SECONDS] {
            let wait = format!("{family}_count{{{unstamped}}}");
            assert!(
                !rendered.contains(&wait),
                "an answer with no wait recorded {family}; rendered:\n{rendered}"
            );
        }
    }
}

/// The dispatch wait is recorded whenever an answer carries one: once, under
/// the TTFT's four labels. A streamed answer with a first token records both
/// families. An answer with no first token (a collected one) records its
/// wait alone, and one dispatched outside a request's entry stamp records
/// its TTFT alone.
#[test]
fn the_dispatch_wait_is_recorded_whenever_an_answer_carries_one() {
    let record = |model_id: &str, ttft: Option<Duration>, dispatch_wait: Option<Duration>| {
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_PD,
            model_id,
            endpoint: metrics_labels::ENDPOINT_MESSAGES,
            ttft,
            dispatch_wait,
            dispatch_wait_after_body: None,
            generation_duration: Duration::from_millis(90),
            input_tokens: Some(4),
            cached_tokens: None,
            output_tokens: 2,
        });
    };
    let (rendered, ()) = with_test_recorder(|| {
        let (ttft, wait) = (Duration::from_millis(40), Duration::from_millis(17));
        record("stamped", Some(ttft), Some(wait));
        record("collected", None, Some(wait));
        record("unstamped", Some(ttft), None);
    });
    let labels = |model: &str| {
        format!("router_type=\"grpc\",backend_type=\"pd\",model=\"{model}\",endpoint=\"messages\"")
    };
    let (stamped, collected) = (labels("stamped"), labels("collected"));
    for series in [
        format!("smg_router_ttft_seconds_count{{{stamped}}} 1"),
        format!("{DISPATCH_WAIT_SECONDS}_count{{{stamped}}} 1"),
        format!("{DISPATCH_WAIT_SECONDS}_sum{{{stamped}}} 0.017"),
        format!("{DISPATCH_WAIT_SECONDS}_count{{{collected}}} 1"),
        format!("{DISPATCH_WAIT_SECONDS}_sum{{{collected}}} 0.017"),
        format!("smg_router_ttft_seconds_count{{{}}} 1", labels("unstamped")),
    ] {
        assert!(
            rendered.lines().any(|line| line == series),
            "{series} missing; rendered:\n{rendered}"
        );
    }
    let wait = format!("{DISPATCH_WAIT_SECONDS}_count{{{}}}", labels("unstamped"));
    assert!(
        !rendered.contains(&wait),
        "an answer with no wait recorded one; rendered:\n{rendered}"
    );
    let ttft = format!("smg_router_ttft_seconds_count{{{collected}}}");
    assert!(
        !rendered.contains(&ttft),
        "an answer with no first token recorded a TTFT; rendered:\n{rendered}"
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
        WorkerMetrics::record_pd_rendezvous_failure(metrics_labels::WORKER_PREFILL, "probe");
        WorkerMetrics::record_pd_rendezvous_failure(metrics_labels::WORKER_DECODE, "bootstrap");
        WorkerMetrics::record_pd_rendezvous_failure(metrics_labels::WORKER_DECODE, "bootstrap");
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

/// The stream-timing set recorded through kept handles renders exactly
/// as it does with a key built per record: the same series with the
/// same values, and no series a stream had nothing for (no TTFT or
/// TPOT for a model whose streams gave no token, no input tokens for an
/// endpoint that reports none, no dispatch wait for an answer that
/// carries none).
#[test]
fn the_stream_timing_set_renders_the_same_through_kept_handles() {
    let records = || {
        // The dispatch wait rides the first record, which has a first
        // token, and the second, which has none and records its wait all
        // the same; the third is a stream dispatched outside a request's
        // entry stamp.
        for (ttft, dispatch_wait, output_tokens, input_tokens) in [
            (
                Some(Duration::from_millis(10)),
                Some(Duration::from_millis(3)),
                5,
                Some(100),
            ),
            (None, Some(Duration::from_millis(4)), 1, None),
            (Some(Duration::from_millis(5)), None, 1, Some(7)),
        ] {
            Metrics::record_streaming_metrics(StreamingMetricsParams {
                router_type: metrics_labels::ROUTER_GRPC,
                backend_type: metrics_labels::BACKEND_REGULAR,
                model_id: "m1",
                endpoint: metrics_labels::ENDPOINT_CHAT,
                ttft,
                dispatch_wait,
                dispatch_wait_after_body: None,
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
            dispatch_wait: None,
            dispatch_wait_after_body: None,
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
        "smg_router_dispatch_wait_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 2",
        "smg_router_dispatch_wait_seconds_sum{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 0.007",
        "smg_router_tpot_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 1",
        "smg_router_time_to_first_token_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 2",
        "smg_router_request_time_per_output_token_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m1\",endpoint=\"chat\"} 1",
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
        "smg_router_dispatch_wait_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\"",
        "smg_router_tpot_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\"",
        "smg_router_time_to_first_token_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\"",
        "smg_router_request_time_per_output_token_seconds_count{router_type=\"grpc\",backend_type=\"regular\",model=\"m2\"",
        "model=\"m2\",endpoint=\"generate\",token_type=\"input\"",
        "model=\"m2\",endpoint=\"generate\",token_type=\"cached\"",
    ] {
        assert!(!kept.contains(absent), "unexpected {absent} in:\n{kept}");
    }
}

/// The upkeep forgets a retired address once nothing holds the removed
/// worker: the set is bounded by the removed workers still referenced, not
/// by the process's history of removals.
#[test]
fn upkeep_forgets_the_retired_addresses_whose_workers_are_gone() {
    let workers = Arc::new(WorkerSeries::new(DistributionBuilder::new(
        parse_quantiles(DEFAULT_SUMMARY_QUANTILES),
        None,
        None,
        None,
        None,
        None,
    )));
    for n in 0..3 {
        let url = format!("http://gone-{n}");
        workers.retire(&url, gone(&url));
    }
    let (held_worker, witness) = held("http://held");
    workers.retire("http://held", witness);
    assert_eq!(workers.retired_len(), 4);
    workers.run_upkeep();
    assert_eq!(workers.retired_len(), 1, "the held worker's address stays");
    drop(held_worker);
    workers.run_upkeep();
    assert_eq!(workers.retired_len(), 0);
}

/// The cost of the router's per-request records with and without the
/// kept handles, for the Numbers table: the request counter, the
/// worker counter (through the per-worker store) and the duration
/// histogram of a unary request, and the stream-timing set on top for a
/// streamed one, without and with the dispatch wait a request under the
/// edge's entry stamp records; run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn time_per_router_record() {
    use std::time::Instant;
    const ROUNDS: u32 = 200_000;
    const WORKER: &str = "grpc://worker-1:20000";
    const MODEL: &str = "model-a";
    let worker = held(WORKER).0;
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
        WorkerMetrics::record_worker_request(worker.as_ref());
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
    let streamed = |dispatch_wait: Option<Duration>| {
        request(STREAMING_TRUE);
        Metrics::record_streaming_metrics(StreamingMetricsParams {
            router_type: metrics_labels::ROUTER_GRPC,
            backend_type: metrics_labels::BACKEND_REGULAR,
            model_id: MODEL,
            endpoint: metrics_labels::ENDPOINT_CHAT,
            ttft: Some(Duration::from_millis(50)),
            dispatch_wait,
            dispatch_wait_after_body: None,
            generation_duration: Duration::from_secs(1),
            input_tokens: Some(128),
            cached_tokens: Some(64),
            output_tokens: 128,
        });
        duration();
    };
    let unstamped = || streamed(None);
    let stamped = || streamed(Some(Duration::from_millis(2)));
    let time = |label: &str| {
        metrics::with_local_recorder(&recorder, || {
            for (name, records) in [
                ("unary (3 records)", &unary as &dyn Fn()),
                ("streamed (8 records)", &unstamped as &dyn Fn()),
                (
                    "streamed (8 records and the dispatch wait)",
                    &stamped as &dyn Fn(),
                ),
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
    let worker = held(WORKER).0;
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
        WorkerMetrics::record_worker_request(worker.as_ref());
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

/// The chat path's remaining per-request records (the selection, the
/// cache-aware decision, the index lookup, the routing-key source, the
/// admission gauges, the body path, the early release) render the same
/// through kept handles as through a key built per record: the same
/// series with the same values. A second pass through kept handles,
/// under another recorder, renders them again: the reset between the
/// passes dropped every handle of the first.
#[test]
fn the_per_request_records_render_the_same_through_kept_handles() {
    let records = || {
        for _ in 0..2 {
            Metrics::record_worker_selection(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_GRPC,
                "m1",
                "cache_aware",
            );
        }
        Metrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_GRPC,
            "m2",
            "random",
        );
        Metrics::record_worker_cache_aware_policy_branch("tree_match");
        Metrics::record_worker_cache_aware_policy_branch("tree_match");
        Metrics::record_worker_cache_aware_policy_branch("spill");
        Metrics::record_worker_manual_policy_branch("occupied_hit");
        Metrics::record_worker_manual_policy_branch("occupied_hit");
        Metrics::record_worker_consistent_hashing_policy_branch("routing_key_hit");
        Metrics::record_worker_prefix_hash_policy_branch("ring_hit");
        Metrics::record_cache_aware_match_ratio(0.25);
        Metrics::record_cache_aware_match_ratio(0.75);
        Metrics::record_kv_index_lookup("chain", 0.000_002);
        Metrics::record_kv_index_lookup("chain", 0.000_004);
        Metrics::record_kv_index_lookup("reference", 0.000_008);
        Metrics::record_routing_key_source("header");
        Metrics::record_admission_queue_entered();
        Metrics::record_admission_inflight_acquired();
        Metrics::record_admission_inflight_acquired();
        Metrics::record_admission_inflight_released();
        Metrics::record_admission_queue_exited();
        Metrics::record_admission_queue_entered();
        Metrics::record_request_body_path("buffered", "no_available_worker");
        Metrics::record_request_body_path("streamed", "single_worker");
        Metrics::record_request_body_path("buffered", "no_available_worker");
        Metrics::record_request_buffers_released_early(300);
        Metrics::record_request_buffers_released_early(200);
    };
    let built = render_with_recorder(|| {
        handles::force_on_this_thread(Some(false));
        records();
        handles::reset();
    });
    let kept = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        handles::reset();
    });
    assert_eq!(sorted_series(&kept), sorted_series(&built));
    let kept_again = render_with_recorder(|| {
        handles::force_on_this_thread(Some(true));
        records();
        handles::reset();
    });
    assert_eq!(sorted_series(&kept_again), sorted_series(&built));
    for expected in [
        "smg_worker_selection_total{worker_type=\"regular\",connection_mode=\"grpc\",model=\"m1\",policy=\"cache_aware\"} 2",
        "smg_worker_selection_total{worker_type=\"prefill\",connection_mode=\"grpc\",model=\"m2\",policy=\"random\"} 1",
        "smg_cache_aware_policy_branch_total{branch=\"tree_match\"} 2",
        "smg_cache_aware_policy_branch_total{branch=\"spill\"} 1",
        "smg_manual_policy_branch_total{branch=\"occupied_hit\"} 2",
        "smg_consistent_hashing_policy_branch_total{branch=\"routing_key_hit\"} 1",
        "smg_prefix_hash_policy_branch_total{branch=\"ring_hit\"} 1",
        "smg_cache_aware_match_ratio_count 2",
        "smg_kv_index_lookup_seconds_count{index=\"chain\"} 2",
        "smg_kv_index_lookup_seconds_count{index=\"reference\"} 1",
        "smg_kv_index_lookup_seconds_sum{index=\"reference\"} 0.000008",
        "smg_routing_key_source_total{source=\"header\"} 1",
        "smg_admission_queue_depth 1",
        "smg_admission_inflight 1",
        "smg_router_request_body_path_total{path=\"buffered\",reason=\"no_available_worker\"} 2",
        "smg_router_request_body_path_total{path=\"streamed\",reason=\"single_worker\"} 1",
        "smg_router_request_buffers_released_early_bytes_total 500",
    ] {
        assert!(
            kept.lines().any(|line| line == expected),
            "missing {expected} in:\n{kept}"
        );
    }
}

/// The cost of the chat path's per-request records with and without the
/// kept handles, for the Numbers table; run with `--ignored --nocapture`.
#[test]
#[ignore = "a timing, not a check"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn time_per_request_records() {
    use std::time::Instant;
    const ROUNDS: u32 = 200_000;
    let recorder = PrometheusBuilder::new().build_recorder();
    let breaker = WorkerHandles::new("http://w1");
    let time = |label: &str| {
        metrics::with_local_recorder(&recorder, || {
            let start = Instant::now();
            for _ in 0..ROUNDS {
                Metrics::record_admission_inflight_acquired();
                Metrics::record_worker_selection(
                    metrics_labels::WORKER_REGULAR,
                    metrics_labels::CONNECTION_GRPC,
                    "m1",
                    "cache_aware",
                );
                Metrics::record_worker_cache_aware_policy_branch("tree_match");
                Metrics::record_cache_aware_match_ratio(0.5);
                Metrics::record_request_body_path("buffered", "single_worker");
                Metrics::record_request_buffers_released_early(300);
                WorkerMetrics::record_worker_cb_outcome(&breaker, "success");
                Metrics::record_admission_inflight_released();
            }
            let per_request = start.elapsed().as_nanos() as f64 / f64::from(ROUNDS);
            println!("ns per request for the eight per-request records, {label}: {per_request:.0}");
        });
    };
    handles::force_on_this_thread(Some(false));
    time("keys built per record");
    handles::force_on_this_thread(Some(true));
    time("handles kept");
    handles::reset();
}

#[test]
fn a_duration_bucket_list_is_refused_unless_finite_positive_and_strictly_ascending() {
    for (buckets, reason) in [
        (vec![0.5, f64::NAN], "finite number above 0"),
        (vec![0.5, f64::INFINITY], "finite number above 0"),
        (vec![0.0, 1.0], "finite number above 0"),
        (vec![-1.0], "finite number above 0"),
        (vec![1.0, 1.0], "ascend strictly"),
        (vec![2.0, 1.0], "ascend strictly"),
    ] {
        let err = PrometheusConfig::validate_duration_buckets(&buckets)
            .expect_err("the list is refused")
            .to_string();
        assert!(err.contains(reason), "{buckets:?}: {err}");
    }
    for buckets in [&[][..], &[0.1, 0.5, 2.0][..]] {
        PrometheusConfig::validate_duration_buckets(buckets)
            .expect("an empty or strictly ascending list is accepted");
    }
}

#[test]
fn the_multimodal_harness_repeats_the_malloc_conf() {
    // The harness sits below this crate and cannot import the constant: it
    // repeats the bytes by hand, and this is what keeps that copy equal.
    let harness = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../crates/multimodal/examples/mm_baseline.rs"
    ));
    let conf = std::str::from_utf8(&SERVER_MALLOC_CONF[..SERVER_MALLOC_CONF.len() - 1])
        .expect("the malloc conf is ASCII");
    assert!(
        harness.contains(conf),
        "crates/multimodal/examples/mm_baseline.rs no longer carries SERVER_MALLOC_CONF ({conf})"
    );
}
