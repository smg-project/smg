//! Cross-router shared utilities.
//!
//! This module collects helpers that every router (HTTP, gRPC,
//! OpenAI, Anthropic, Gemini, etc.) needs but no individual router
//! owns. Putting them here keeps `routers/mod.rs` focused on the
//! `RouterTrait` definition and the per-protocol submodules.
//!
//! Submodules:
//! - [`body_policy`] — per-family [`body_policy::BodyPolicy`] capability,
//!   the per-request buffer-vs-stream decision matrix and its counter
//!   vocabulary
//! - [`header_utils`] — request header parsing helpers
//!   (`extract_routing_key`, `extract_target_worker`, etc.)
//! - [`mcp_utils`] — Model Context Protocol tool-call orchestration
//! - [`persistence_utils`] — response/conversation persistence
//!   helpers shared across the chat / responses / messages routes
//! - [`realtime`] — Realtime API transport (WS/WebRTC/REST relay +
//!   session registry) shared by the OpenAI and HTTP routers
//! - [`overload`] — shed responses for the absolute worker-overload
//!   guard, shared by the HTTP and gRPC selection paths
//! - `pair_index` — the prefill and decode workers that may pair, decided
//!   once for a membership generation and read by every placement
//! - [`pd_admission`] — the disaggregated dispatch's admission gate: never
//!   post more bootstrap rooms to a pair than its decode engine can admit
//! - `pd_failure` — the one vocabulary a prefill or decode side's failure is
//!   named by (the side, the phase, the failure's own status), the HTTP and
//!   gRPC prefill-decode paths' answers built from it
//! - `placement` — policy-driven worker placement over the routing
//!   pools (single worker and prefill/decode pairs), the one sequence the
//!   HTTP and gRPC families used to each carry a copy of
//! - [`worker_selection`] — per-request worker-selection helpers used
//!   by every routing path (regular, PD, fallback, external provider)
//! - `test_support` (test builds only) — the fast retry policy the HTTP
//!   and gRPC routers' tests retry under
//! - `bench_support` (test builds and the `test-util` feature) — the
//!   regular and the pair selection over a registered fleet, for a
//!   benchmark of each path
//! - `attempt_ledger` — one request's bookkeeping across its retry
//!   attempts: the circuit breaker charged once per request per worker,
//!   and a worker's definitive answer never replayed on it
//! - `rate_limit_cell` — one logical request's tenant rate-limit
//!   reservation: reserved once before any dispatch, then settled, handed
//!   to a streaming body, or closed unsettled
//! - [`request_lease`] — dispatch-phase owner of a request's parsed
//!   body, routing derivatives and serialized upstream bytes, with a
//!   retry-aware release point
//! - [`retry`] — generic async retry executor + backoff calculator,
//!   used by every router for transport-level retries. Has zero
//!   coupling to the `Worker` trait — it lived in `worker/` for
//!   historical reasons before this extraction.
//! - [`sglang_fields`] — the engine-specific request fields the HTTP proxy
//!   strips at their defaults and the provider transformers drop outright
//! - [`snapshot`] — the gateway's value of a membership generation (the
//!   routing pools, the pairs, the window floors and the labels kept with
//!   the members), and the registry and the load monitor named over it
//! - [`sse`] — shared SSE codec (encoder + decoder) for streaming
//!   responses to clients and parsing upstream SSE byte streams

pub(crate) mod attempt_ledger;
pub mod body_policy;
pub(crate) mod decisions;
pub use smg_external_router::header_utils;
pub(crate) mod kv_transfer;
pub use smg_external_router::{mcp_utils, openai_bridge};
pub mod overload;
pub(crate) mod pair_classes;
pub(crate) mod pair_index;
pub mod pd_admission;
pub(crate) mod pd_failure;
pub use smg_external_router::persistence_utils;
pub(crate) mod placement;
pub mod prefill_admission;
pub(crate) mod rate_limit_cell;
pub use smg_external_router::realtime;
pub mod request_lease;
pub(crate) use smg_external_router::sglang_fields;
pub use smg_external_router::{retry, sse};
pub mod snapshot;
pub mod sse_rechunk;
pub mod stream;
#[cfg(test)]
mod worker_health_tests;
pub mod worker_selection;

/// Threshold above which upstream request bodies are sent as one-shot
/// streams. A streamed body's `try_clone()` is `None` in every layer that
/// would otherwise hold a refcount until response headers (the stale-conn
/// resend guard, reqwest's internal retry and redirect clones), so the
/// allocation frees at upload completion instead of living for the whole
/// generation on buffered upstreams. The explicit Content-Length keeps h1
/// framing non-chunked; h2 is unaffected.
pub(crate) const STREAM_UPSTREAM_BODY_OVER: usize = 1 << 20;

/// Buffer capacity for reserializing a parsed request of `raw_len` incoming
/// bytes: the round trip stays close to raw size, and the 1/16 + 512B slack
/// absorbs injected fields (bootstrap, kv_transfer_params, dp ranks).
fn serialized_capacity(raw_len: usize) -> usize {
    raw_len + raw_len / 16 + 512
}

/// `serde_json::to_vec` into a buffer pre-sized from the raw request length,
/// avoiding doubling growth (and its final huge memcpy) for large bodies.
pub(crate) fn serialize_json_sized<T: serde::Serialize>(
    value: &T,
    raw_len: Option<usize>,
) -> serde_json::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(raw_len.map_or(128, serialized_capacity));
    serde_json::to_writer(&mut buf, value)?;
    Ok(buf)
}

/// The capacity a thread's JSON answer buffer keeps between answers: a chat
/// completion, a Messages answer or a JSON-schema answer of a few hundred
/// bytes fits it many times over; the buffer a larger answer grew goes with
/// that answer.
const JSON_ANSWER_RETAINED: usize = 64 * 1024;

thread_local! {
    /// The buffer a thread's typed JSON answers are serialized into.
    static JSON_ANSWER_BUFFER: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A typed answer as a JSON response: serialized once into the thread's
/// buffer (serde's own sink, one `extend_from_slice` per write) and copied
/// once into a body of exactly its size, with the `application/json`
/// content type. That is one allocation per answer: a buffer sized per
/// answer and handed to the body by refcount cost two, the buffer and the
/// shared header `Bytes::from` boxes for a buffer with spare capacity, and
/// grew again for a body past its size. `axum::Json` writes the same bytes
/// through `BytesMut::with_capacity(128).writer()`: the `bytes` writer
/// adapter slices every write to the buffer's remaining room before it
/// appends, which prices the one-byte JSON tokens (2.5 % of a chat
/// request's user instructions on the mock bed), and the 128-byte start
/// reallocates and copies a 700-byte body three times. A value that does
/// not serialize answers as axum does: a 500 with the error's text.
pub fn json_response<T: serde::Serialize>(value: &T) -> axum::response::Response {
    JSON_ANSWER_BUFFER.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buffer) => {
            buffer.clear();
            let result = serde_json::to_writer(&mut *buffer, value);
            let response = json_answer(&buffer, result);
            if buffer.capacity() > JSON_ANSWER_RETAINED {
                *buffer = Vec::new();
            }
            response
        }
        // A serializer that answers while this thread's buffer is borrowed
        // (none does today): a buffer of this answer's own.
        Err(_) => {
            let mut buffer = Vec::new();
            let result = serde_json::to_writer(&mut buffer, value);
            json_answer(&buffer, result)
        }
    })
}

/// The response of a serialized answer, compiled once for every `T`.
fn json_answer(body: &[u8], result: serde_json::Result<()>) -> axum::response::Response {
    use axum::{
        http::{header, HeaderValue, StatusCode},
        response::IntoResponse,
    };
    match result {
        Ok(()) => (
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            )],
            bytes::Bytes::copy_from_slice(body),
        )
            .into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            )],
            error.to_string(),
        )
            .into_response(),
    }
}

/// A typed request as a `Value`, for the forwarding paths that edit it:
/// `openai_protocol::common::to_value_exact` with the buffer pre-sized from
/// the raw request length. Every typed `f32` goes out as the client wrote
/// it (`serde_json::to_value` would widen `0.95` to `0.949999988079071`).
pub(crate) fn request_to_value<T: serde::Serialize>(
    value: &T,
    raw_len: Option<usize>,
) -> serde_json::Result<serde_json::Value> {
    serde_json::from_slice(&serialize_json_sized(value, raw_len)?)
}

/// `Bytes::from(Vec)` keeps the Vec's capacity, so unshrunk doubling growth
/// (up to 2x len) would be retained for the buffer's whole lifetime; trade
/// one bounded memcpy to free it. The threshold is double the intended
/// pre-size slack so well-hinted buffers are never re-copied.
pub(crate) fn trim_serialization_slack(buf: &mut Vec<u8>) {
    if buf.capacity() > buf.len() + buf.len() / 8 + 1024 {
        buf.shrink_to_fit();
    }
}

pub(crate) fn attach_sized_body(
    builder: reqwest::RequestBuilder,
    body: bytes::Bytes,
) -> reqwest::RequestBuilder {
    let len = body.len();
    let builder = builder.header(reqwest::header::CONTENT_LENGTH, len);
    if len >= STREAM_UPSTREAM_BODY_OVER {
        builder.body(reqwest::Body::wrap_stream(futures::stream::once(
            async move { Ok::<_, std::convert::Infallible>(body) },
        )))
    } else {
        builder.body(body)
    }
}

/// What a benchmark needs of the routers' selection: the regular placement
/// over one model's registered pool and the default pair selection over its
/// registered prefill-decode fleet, each as a router runs it for a request,
/// and whether the process captures every request's cache trace.
#[cfg(any(test, feature = "test-util"))]
pub mod bench_support {
    use std::{
        future::Future,
        pin::pin,
        sync::Arc,
        task::{Context, Poll, Waker},
        time::Instant,
    };

    use axum::http::{HeaderMap, HeaderValue};
    use rand::RngExt;
    use smg_policy_controller::{
        engine::{
            decide, Constraints, DecisionClock, Identity, Live, Phase, PolicyEngine, RequestFacts,
            WireModes,
        },
        stages::Lifecycle,
        tracking::{Lease, Outcome, Side as EngineSide, TransportClass},
    };
    use smg_worker::{
        identity::basic::{Worker, WorkerLoadGuard},
        load::state::LoadReceiver,
    };

    use super::{
        attempt_ledger::AttemptLedger,
        overload,
        placement::{select_pair, select_single, PlacementInputs},
    };
    use crate::{
        config::RouterConfig,
        observability::cache_trace,
        policies::{EnginePolicy, PolicyRegistry, Side},
        policy_controller::{
            engine::{mark, Ask, End, PolicyEngine as EngineObjectOf, Prompt},
            start_in_process,
        },
        routers::common::{
            pair_index::{PdPairIndex, PdWire},
            snapshot::{RoutingPool, WorkerRegistry, WorkerRegistryExt},
        },
    };

    /// The route a request is placed on: the pool its router places over.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Route {
        /// The regular HTTP router.
        Http,
        /// The gRPC pipeline's regular selection stage.
        Grpc,
    }

    impl Route {
        /// The pool the route's router places a single request over.
        fn pool(self) -> RoutingPool {
            match self {
                Self::Http => RoutingPool::HttpRegular,
                Self::Grpc => RoutingPool::GrpcPipelineRegular,
            }
        }
    }

    /// Bytes the calling thread has allocated since it started, by the
    /// allocator's own count of the thread: a free does not subtract.
    /// `None` in a build that links no such count.
    pub fn thread_allocated_bytes() -> Option<u64> {
        #[cfg(all(
            feature = "jemalloc-stats",
            not(target_env = "msvc"),
            not(target_env = "musl")
        ))]
        {
            tikv_jemalloc_ctl::thread::allocatedp::read()
                .ok()
                .map(|allocated| allocated.get())
        }
        #[cfg(not(all(
            feature = "jemalloc-stats",
            not(target_env = "msvc"),
            not(target_env = "musl")
        )))]
        {
            None
        }
    }

    /// The worker a route placed a request on, as a number a bench compares
    /// with another request's: the place of the worker's object in memory,
    /// which stands for as long as the worker is registered. Nothing else is
    /// read of it.
    fn placed_on(worker: &Arc<dyn Worker>) -> usize {
        Arc::as_ptr(worker).addr()
    }

    /// Hands the classic policies the load feed the monitor would publish.
    pub fn attach_loads(policies: &PolicyRegistry, loads: Option<LoadReceiver>) {
        policies.set_load_receiver(loads);
    }

    /// Sets a pair gateway's side policies on `policies` as the router
    /// factory sets them at the start: each side's own policy of `config`,
    /// or its main policy.
    pub fn set_pair_sides(policies: &PolicyRegistry, config: &RouterConfig) {
        let main = &config.policy;
        policies.set_side_policy(Side::Prefill, config.mode.get_prefill_policy(main));
        policies.set_side_policy(Side::Decode, config.mode.get_decode_policy(main));
    }

    /// The classic policy registry of `config`'s policy, its policies
    /// asking the policy engine of the document in force under `config`,
    /// built as a gateway's start builds both under the setting, whatever
    /// `config` says of the setting: [`route_classic`] then runs the
    /// classic placement, and the pass asks the policy object that asks the
    /// engine where it asks a classic policy. Under a pair mode the
    /// registry holds a pair's two side policies ([`set_pair_sides`]), and
    /// [`route_pair`] runs the classic pair placement over them.
    ///
    /// # Errors
    ///
    /// A configuration no policy document says, a document the engine
    /// refuses, and a policy the engine does not pick by itself: the
    /// registry would hand out the classic policy, and a cell named for
    /// the object would time that one.
    pub fn policies_asking_the_engine(config: &RouterConfig) -> Result<PolicyRegistry, String> {
        let picked = RouterConfig {
            policy_controller: true,
            ..config.clone()
        };
        let started = start_in_process(&picked, None)
            .map_err(|refused| refused.to_string())?
            .ok_or("this configuration places its requests by no routing policy")?;
        let whole = started.decides_whole();
        let policies = PolicyRegistry::with_policy_engine(
            config.policy.clone(),
            config.routing_key_override.clone(),
            started.engine,
            started.origin,
            whole,
        );
        let asked = if config.mode.is_pd_mode() {
            set_pair_sides(&policies, config);
            [policies.get_prefill_policy(), policies.get_decode_policy()]
                .iter()
                .all(|side| side.as_any().is::<EnginePolicy>())
        } else {
            policies.get_default_policy().as_any().is::<EnginePolicy>()
        };
        if asked {
            Ok(policies)
        } else {
            Err(format!(
                "the policy engine does not pick `{}` by itself: the registry builds the classic policy",
                config.policy.name()
            ))
        }
    }

    /// The gateway's own policy engine object, over the engine a gateway's
    /// start builds for `config`'s policy under the setting, whatever
    /// `config` says of the setting, and over `registry`'s lists: what the
    /// routers ask where the engine decides every request of the document
    /// whole.
    pub struct EngineObject(EngineObjectOf);

    impl EngineObject {
        /// The object a gateway's start builds for `config` over `registry`,
        /// handed `loads`, the load feed the monitor would publish, as the
        /// start hands it the monitor's.
        ///
        /// # Errors
        ///
        /// A configuration no policy document says, a document the engine
        /// refuses, and a policy the engine does not decide whole yet: the
        /// routers would keep their classic placement pass, and a cell
        /// named for the object would time that one.
        pub fn of(
            config: &RouterConfig,
            registry: &Arc<WorkerRegistry>,
            loads: Option<LoadReceiver>,
        ) -> Result<Self, String> {
            let picked = RouterConfig {
                policy_controller: true,
                ..config.clone()
            };
            let started = start_in_process(&picked, None)
                .map_err(|refused| refused.to_string())?
                .ok_or("this configuration places its requests by no routing policy")?;
            if !started.decides_whole() {
                return Err(format!(
                    "the policy engine does not decide `{}` whole yet: the routers keep their classic placement pass",
                    config.policy.name()
                ));
            }
            let object = EngineObjectOf::own(started.engine, Arc::clone(registry));
            if let Some(loads) = loads {
                object.read_loads(loads);
            }
            Ok(Self(object))
        }

        /// How many times the reports of `model`'s workers on `route` were
        /// converted for the engine's records: once for each load snapshot
        /// a decision converted the list under. Zero for a pool no request
        /// was routed over, and under a document that reads no report.
        pub fn conversions(registry: &WorkerRegistry, model: &str, route: Route) -> u64 {
            registry
                .get_routing_snapshot(model)
                .pool_members_cell(route.pool())
                .get()
                .map_or(0, |members| members.conversions())
        }
    }

    /// One request's routing by the gateway's engine object on `route`, as
    /// a regular router's arm runs it: the prompt's hashes, the ask over
    /// the route's pool and the commit of the first target the gateway can
    /// send to, the dispatch-time read of the chosen worker's overload
    /// veto, the dispatch's word, the worker's load guard from the send to
    /// the request's end, and the end's word. The worker the request went
    /// to (`placed_on`), and `None` where it had no worker.
    ///
    /// What [`route_classic`] does under a classic policy, with the engine
    /// deciding the whole request through the object the routers hold: no
    /// classic policy and no classic placement pass runs.
    pub fn route_engine(
        registry: &WorkerRegistry,
        object: &EngineObject,
        model: &str,
        route: Route,
        tokens: Option<&[u32]>,
    ) -> Option<usize> {
        let (pool, wires) = match route {
            Route::Http => (RoutingPool::HttpRegular, WireModes::NONE),
            Route::Grpc => (
                RoutingPool::GrpcPipelineRegular,
                WireModes::GRPC.union(WireModes::ZMQ),
            ),
        };
        let prompt = Prompt::of(tokens, None);
        let hashes = object.0.hashes(model, &prompt);
        let id = mark();
        let ask = Ask {
            wires,
            hashes: Some(&hashes),
            prompt: Some(&prompt),
            pool: Some(pool),
            ..Ask::of(&id, model)
        };
        let now = Instant::now();
        // The engine of this process decides on the calling thread: the
        // arm's future is ready at its first poll.
        let placing = pin!(object
            .0
            .place_where(&ask, |worker| pool.matches(worker), now));
        let Poll::Ready(Ok(mut placed)) = placing.poll(&mut Context::from_waker(Waker::noop()))
        else {
            return None;
        };
        let shed = overload::shed_if_worker_overloaded(
            placed.worker().as_ref(),
            model,
            registry.overload_shed_enabled(),
        );
        if shed.is_some() {
            return None;
        }
        placed.dispatched(now);
        let guard = WorkerLoadGuard::with_key(Arc::clone(placed.worker()), None);
        drop(guard);
        placed.completed(
            End {
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: 0,
                tokens_out: 0,
                duration_micros: 0,
            },
            now,
        );
        Some(placed_on(placed.worker()))
    }

    /// One request's routing under the classic policies on `route`: the
    /// placement over the route's own pool, the dispatch-time read of the
    /// chosen worker's overload veto, and the worker's load guard from the
    /// send to the request's end, whose drop tells the model's policy.
    /// `headers` are the request's, where it carries any: a routing key is
    /// read from them as a router reads it, for the placement and for the
    /// load guard. The worker the request went to (`placed_on`), and
    /// `None` where it had no worker.
    pub fn route_classic(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        model: &str,
        route: Route,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
    ) -> Option<usize> {
        let pool = route.pool();
        let inputs = PlacementInputs {
            tokens,
            headers,
            ..Default::default()
        };
        let worker = select_single(registry, policies, model, pool, None, inputs)?;
        let shed = overload::shed_if_worker_overloaded(
            worker.as_ref(),
            model,
            registry.overload_shed_enabled(),
        );
        if shed.is_some() {
            return None;
        }
        let went = placed_on(&worker);
        let guard = WorkerLoadGuard::with_key(worker, policies.sticky_key(headers, None));
        drop(guard);
        Some(went)
    }

    /// One request's routing through a pair gateway's classic placement
    /// on the gRPC wire: the pair index of the model's snapshot, the pair
    /// placed over it by the registry's two side policies, and each side's
    /// load guard from the send to the request's end. The prefill worker
    /// the request went to (`placed_on`), and `None` where it had no
    /// pair.
    pub fn route_pair(
        registry: &WorkerRegistry,
        policies: &PolicyRegistry,
        model: &str,
        tokens: Option<&[u32]>,
    ) -> Option<usize> {
        let pairs = registry
            .get_routing_snapshot(model)
            .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode());
        let inputs = PlacementInputs {
            tokens,
            ..Default::default()
        };
        let pair = select_pair(registry, policies, model, &pairs, None, None, inputs).ok()?;
        let went = placed_on(&pair.prefill);
        let prefill = WorkerLoadGuard::with_key(pair.prefill, policies.sticky_key(None, None));
        let decode = WorkerLoadGuard::with_key(pair.decode, policies.sticky_key(None, None));
        drop((prefill, decode));
        Some(went)
    }

    /// The policy engine of one routing policy, asked for its whole
    /// decision over records written from the gateway's own data: the
    /// engine of the document the routing settings give for the policy, its
    /// clock, and the load feed the monitor would publish.
    pub struct EngineOverRecords {
        engine: Arc<PolicyEngine>,
        clock: DecisionClock,
        loads: Option<LoadReceiver>,
    }

    impl EngineOverRecords {
        /// The engine a gateway's start builds for `config`'s policy under
        /// the setting, whatever `config` says of the setting, reading the
        /// workers' reports from `loads`.
        ///
        /// # Errors
        ///
        /// A configuration no policy document says, and a document the
        /// engine refuses.
        pub fn of(config: &RouterConfig, loads: Option<LoadReceiver>) -> Result<Self, String> {
            let picked = RouterConfig {
                policy_controller: true,
                ..config.clone()
            };
            let started = start_in_process(&picked, None)
                .map_err(|refused| refused.to_string())?
                .ok_or("this configuration places its requests by no routing policy")?;
            Ok(Self {
                engine: started.engine,
                clock: DecisionClock::new(),
                loads,
            })
        }

        /// How many times the reports of `model`'s workers on `route` were
        /// converted for the engine's records: once for each load snapshot
        /// a decision wrote every record under. Zero for a pool no request
        /// was routed over.
        pub fn conversions(&self, registry: &WorkerRegistry, model: &str, route: Route) -> u64 {
            registry
                .get_routing_snapshot(model)
                .pool_members_cell(route.pool())
                .get()
                .map_or(0, |members| {
                    members
                        .records(self.engine.spec(), &self.clock)
                        .conversions()
                })
        }
    }

    /// One request's routing by the policy engine's whole decision on
    /// `route`, over records written for this decision from the route's own
    /// pool as the registry lists it, the worker objects and the load
    /// snapshot: the decision, the word of its issue, its dispatch and its
    /// end to the engine's stages, the dispatch-time read of the chosen
    /// worker's overload veto, and the worker's load guard from the send to
    /// the request's end. `headers` are the request's, where it carries
    /// any: its routing key is read from them by the sources its binding
    /// names, as a router's ask reads it. The worker the request went to
    /// (`placed_on`), and `None` where it had no worker.
    ///
    /// What [`route_classic`] does under a classic policy, with the engine
    /// deciding: no classic policy and no classic placement pass runs.
    pub fn route_records(
        registry: &WorkerRegistry,
        engine: &EngineOverRecords,
        model: &str,
        route: Route,
        tokens: Option<&[u32]>,
        headers: Option<&HeaderMap>,
    ) -> Option<usize> {
        let pool = route.pool();
        // The decision's instant is read before its snapshot is borrowed: a
        // decision that began later holds a snapshot no older.
        let now = Instant::now();
        let snapshot = registry.get_routing_snapshot(model);
        let workers = snapshot.pool(pool);
        let members = snapshot
            .pool_members_cell(pool)
            .get_or_init(|| Box::new(engine.engine.pool_members(workers)));
        let spec = engine.engine.spec();
        let records = members.records(spec, &engine.clock);
        // Borrowed once a decision, as a classic policy borrows it.
        let loads = engine.loads.as_ref().map(|loads| loads.borrow().clone());
        // The request's two keys by the sources its binding names: its
        // routing key from the first source that holds a value, and its
        // client's key where it carries no routing key.
        let (key, client_key) = match spec.key_sources_for(model, Phase::Single) {
            Some(sources) => {
                let header = |name: &str| headers?.get(name).map(HeaderValue::as_bytes);
                match sources.derive(header, None) {
                    Some(derived) => (derived.key, 0),
                    None => (Identity::NONE, sources.derive_client(header)),
                }
            }
            None => (Identity::NONE, 0),
        };
        let mut facts = RequestFacts {
            model,
            phase: Phase::Single,
            prompt_tokens: tokens
                .map_or(0, |tokens| u32::try_from(tokens.len()).unwrap_or(u32::MAX)),
            session_key: key,
            client_key,
            seed: rand::rng().random(),
            ..RequestFacts::default()
        };
        let live = Live::at(spec.engine(), &engine.clock, now, loads.as_ref(), key.plain);
        let decision = decide(
            spec.engine(),
            records,
            workers,
            &live,
            &mut facts,
            &Constraints::default(),
        );
        let pick = decision.pick()?;
        let target = pick.targets.first()?;
        let worker = workers.get(target.position as usize)?;
        // The engine's stages hear the request as a gateway's guard tells
        // it: its issue and its dispatch here, its end below.
        let stages = spec.engine();
        let lease = Lease {
            side: EngineSide::Single,
            booked: target.id,
            generation: spec.generation(),
            session: facts.session(),
            session_verdict: pick.session_verdict,
            issued_micros: facts.now_micros,
            tokens: facts.prompt_tokens,
            ..Lease::default()
        };
        stages.on_decision_issued(&lease, &decision);
        stages.on_dispatched(&lease, target, EngineSide::Single, 0);
        let shed = overload::shed_if_worker_overloaded(
            worker.as_ref(),
            model,
            registry.overload_shed_enabled(),
        );
        if shed.is_some() {
            return None;
        }
        let guard = WorkerLoadGuard::with_key(Arc::clone(worker), key.plain);
        drop(guard);
        stages.on_completed(
            &lease,
            &Outcome {
                side: EngineSide::Single,
                served: target.id,
                status: 200,
                class: TransportClass::Ok,
                ttft_micros: 0,
                tokens_in: facts.prompt_tokens,
                tokens_out: 0,
                duration_micros: 0,
            },
        );
        Some(placed_on(worker))
    }

    /// Whether a process switch captures every request's cache trace
    /// (`SMG_CACHE_TRACE=1` or `SMG_CACHE_TRACE_HEADER=1`). A benchmark cell
    /// labelled untraced is not untraced in such a process.
    pub fn process_trace_configured() -> bool {
        cache_trace::configured()
    }

    /// One model's regular pool over the gRPC pipeline and its policies.
    pub struct RegularPath {
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        model: String,
    }

    impl RegularPath {
        /// The regular pool of `model` as `registry` holds it at each
        /// placement.
        pub fn new(
            registry: Arc<WorkerRegistry>,
            policies: Arc<PolicyRegistry>,
            model: &str,
        ) -> Self {
            Self {
                registry,
                policies,
                model: model.to_string(),
            }
        }

        /// One placement of a request carrying `tokens`, with no filter, no
        /// wire pin and no tried engine: the pool drawn from the registry
        /// and one policy pick over it.
        pub fn place(&self, tokens: Option<&[u32]>) -> Option<Arc<dyn Worker>> {
            self.select(PlacementInputs {
                tokens,
                ..Default::default()
            })
        }

        /// [`Self::place`] under a candidate filter that refuses nobody and
        /// costs nothing itself: what the selection adds when a router hands
        /// it a filter, as the gRPC pipeline does on a request's first
        /// attempt for its media and context demands.
        pub fn place_filtered(&self, tokens: Option<&[u32]>) -> Option<Arc<dyn Worker>> {
            let everyone = |_: &dyn Worker| true;
            self.select(PlacementInputs {
                tokens,
                candidate_filter: Some(&everyone),
                ..Default::default()
            })
        }

        /// [`Self::place`] as the HTTP router runs a request's first
        /// attempt: under the candidate filter a new request's attempt
        /// ledger hands. The filter and the entry are that router's; the
        /// pool is this path's, the gRPC pipeline's regular one.
        pub fn place_first_attempt(&self, tokens: Option<&[u32]>) -> Option<Arc<dyn Worker>> {
            AttemptLedger::default().with_candidate_filter(|candidate_filter| {
                self.select(PlacementInputs {
                    tokens,
                    candidate_filter,
                    ..Default::default()
                })
            })
        }

        fn select(&self, inputs: PlacementInputs<'_>) -> Option<Arc<dyn Worker>> {
            select_single(
                &self.registry,
                &self.policies,
                &self.model,
                RoutingPool::GrpcPipelineRegular,
                None,
                inputs,
            )
        }
    }

    /// One model's fleet, its policies and its pair index over the gRPC wire.
    pub struct PairPath {
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        model: String,
        pairs: Arc<PdPairIndex>,
    }

    impl PairPath {
        /// The pair index of `model` over the workers `registry` holds now.
        pub fn new(
            registry: Arc<WorkerRegistry>,
            policies: Arc<PolicyRegistry>,
            model: &str,
        ) -> Self {
            let pairs = registry
                .get_routing_snapshot(model)
                .pd_pairs(PdWire::Grpc, policies.pd_pairing_mode());
            Self {
                registry,
                policies,
                model: model.to_string(),
                pairs,
            }
        }

        /// One placement of a request carrying `tokens`, with no admission
        /// window: the prefill and the decode paired with it. With `traced`
        /// it runs inside a cache-trace capture, as a request that asked for
        /// its decision record does.
        pub async fn place(
            &self,
            tokens: Option<&[u32]>,
            traced: bool,
        ) -> Option<(Arc<dyn Worker>, Arc<dyn Worker>)> {
            cache_trace::scope(traced, async {
                let inputs = PlacementInputs {
                    tokens,
                    ..Default::default()
                };
                select_pair(
                    &self.registry,
                    &self.policies,
                    &self.model,
                    &self.pairs,
                    None,
                    None,
                    inputs,
                )
                .ok()
                .map(|pair| (pair.prefill, pair.decode))
            })
            .await
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! What the HTTP and gRPC routers' tests share.

    use crate::config::types::RetryConfig;

    /// `max_retries` attempts with one to two milliseconds between them and
    /// no jitter: a retrying test's whole schedule fits in its own bound.
    pub(crate) fn fast_retry_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            initial_backoff_ms: 1,
            max_backoff_ms: 2,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sized_serialization_never_doubles_for_large_bodies() {
        let raw = serde_json::to_vec(&serde_json::json!({
            "text": "x".repeat(8 << 20),
            "stream": false,
        }))
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();

        let body = serialize_json_sized(&parsed, Some(raw.len())).unwrap();

        assert_eq!(body.len(), raw.len());
        assert_eq!(body.capacity(), serialized_capacity(raw.len()));
        assert!(body.capacity() <= body.len() + body.len() / 8 + 1024);
    }

    #[test]
    fn a_json_answer_is_the_serializer_bytes_under_the_json_content_type() {
        use axum::http::{header, StatusCode};
        let value = serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "x".repeat(600)}}],
        });
        let response = json_response(&value);
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let body =
            futures::executor::block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
                .unwrap();
        assert_eq!(body, serde_json::to_vec(&value).unwrap());
    }

    #[test]
    fn a_json_answer_that_does_not_serialize_is_a_plain_text_500() {
        use axum::http::{header, StatusCode};
        // A map keyed by a tuple has no JSON key: the serializer refuses it.
        let value: std::collections::BTreeMap<(u8, u8), u8> = [((1, 2), 3)].into_iter().collect();
        let response = json_response(&value);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/plain; charset=utf-8"
        );
    }

    #[test]
    fn trim_frees_doubling_slack_but_keeps_presized_buffers() {
        // Mimic a 5MiB body landing in an 8MiB doubling-growth buffer.
        let mut grown = Vec::with_capacity(8 << 20);
        grown.resize(5 << 20, b'x');
        trim_serialization_slack(&mut grown);
        assert!(grown.capacity() <= grown.len() + grown.len() / 8 + 1024);
        assert_eq!(grown.len(), 5 << 20);

        let mut sized = Vec::with_capacity(serialized_capacity(1 << 20));
        sized.resize(1 << 20, b'y');
        let capacity = sized.capacity();
        trim_serialization_slack(&mut sized);
        assert_eq!(sized.capacity(), capacity);
    }
}

#[cfg(test)]
#[expect(clippy::print_stdout, reason = "a recorder says where its sheet went")]
mod classic_pair_select_rows;
