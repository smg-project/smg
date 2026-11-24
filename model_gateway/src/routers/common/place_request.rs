//! What a placement function states of its request to the policy engine.
//!
//! The placement functions take one [`PlacementInputs`] from their route:
//! what a classic policy reads of a request. Where the engine decides, they
//! ask it with a [`PlaceRequest`] instead. This file fills that request from
//! the same inputs, each under the name the engine reads it by, and from
//! what the route states for the engine alone ([`Stated`]), which a classic
//! placement is handed nothing of.
//!
//! A request pinned to a wire stays on it by its demand: a retry pinned to
//! its retained plan's wire, and a first attempt its route pins. One
//! function gives each runtime and transport this gateway dials a number
//! ([`wire_number`]). The gateway states that number of every worker of a
//! list it lends ([`stated_facts`]) and as the demand of a pinned request
//! ([`place_request`]), and the engine compares the two.

use std::sync::Arc;

use smg_policy_controller::engine::{
    worker_id, Demands, FixedFacts, HashSeeds, HeaderRead, PlaceRequest, Prompt, TokenEntry,
    TokenKey, WorkerId, WorkerLists, WorkerRule,
};
use smg_worker::identity::basic::{ConnectionMode, RuntimeType, Worker};

use super::placement::{was_tried, PlacementInputs, Stated, WireConstraint};
use crate::policies::CacheNamespace;

/// The workers at the engines a request's failed attempts were sent to,
/// each by the id the engine knows it by ([`tried_ids`]): every rank of a
/// tried engine is among them. A first attempt leaves it empty and
/// allocates nothing.
pub(crate) type TriedIds = Vec<WorkerId>;

/// Writes into `ids` the workers of `workers` at the engines `tried` names
/// ([`PlacementInputs::tried`]), each by the id the engine knows it by: the
/// fingerprint of the worker's address.
///
/// A route names a tried engine by its base URL, which the ranks of a
/// DP-aware worker share: such an engine names every worker at it, as
/// classic placement steps aside from all of them. An engine that left the
/// list names nobody. A first attempt reads no worker.
pub(crate) fn tried_ids(workers: &[Arc<dyn Worker>], tried: &[String], ids: &mut TriedIds) {
    ids.clear();
    if tried.is_empty() {
        return;
    }
    ids.extend(
        workers
            .iter()
            .filter(|worker| was_tried(worker.as_ref(), tried))
            .map(|worker| worker_id(worker.as_ref())),
    );
}

/// A request's entry in the token tree the gateway lends the engine: the
/// key the classic policy matches and writes its own tree by
/// ([`CacheNamespace::keyed_tokens`]), under the request's cache partition
/// and at the tree's page. A request that holds no token ids names none.
impl TokenEntry for PlacementInputs<'_> {
    fn token_key(&self, page: usize) -> Option<TokenKey<'_>> {
        let (tokens, marker) =
            CacheNamespace::keyed_tokens(self.cache_namespace, self.tokens?, page);
        Some(TokenKey { tokens, marker })
    }
}

/// The number the policy engine knows a runtime and transport by: one for
/// each pair this gateway dials, and never zero, which states none. Two
/// workers have one number exactly where a retained plan's wire holds them
/// alike ([`WireConstraint::admits`]): the engine compares the numbers and
/// reads no more of them.
#[must_use]
pub(crate) const fn wire_number(runtime: RuntimeType, connection: ConnectionMode) -> u32 {
    // Every runtime and transport by its name and no rest: one added later
    // does not build until it has a number of its own.
    let runtime = match runtime {
        RuntimeType::Unspecified => 0,
        RuntimeType::Sglang => 1,
        RuntimeType::Vllm => 2,
        RuntimeType::Trtllm => 3,
        RuntimeType::Mlx => 4,
        RuntimeType::TokenSpeed => 5,
        RuntimeType::Generic => 6,
        RuntimeType::External => 7,
    };
    let connection = match connection {
        ConnectionMode::Http => 1,
        ConnectionMode::Grpc => 2,
        ConnectionMode::Zmq => 3,
    };
    // The transport stands in the low byte: one added later takes the next
    // number, and no two pairs share one.
    (runtime << 8) | connection
}

/// What this gateway states of each worker of `workers` beside its object,
/// one a worker in the list's order: the number of the runtime and
/// transport it is dialed on ([`wire_number`]), and nothing else. A list is
/// lent with them, made once a list at its first lending: the engine reads
/// them where it makes its value of the list, under each document.
#[must_use]
pub(crate) fn stated_facts(workers: &[Arc<dyn Worker>]) -> Box<[FixedFacts]> {
    let stated = |worker: &Arc<dyn Worker>| {
        let mut facts = FixedFacts::default();
        facts.demands.runtime = wire_number(
            worker.metadata().spec.runtime_type,
            *worker.connection_mode(),
        );
        facts
    };
    workers.iter().map(stated).collect()
}

/// The request `inputs` and `stated` state for `model_id` over `lists`, as
/// the engine reads it. `tried` are the workers at the engines the
/// request's failed attempts were sent to ([`tried_ids`]), and `wire` the
/// wire the request is pinned to, where it is pinned to one.
///
/// - The prompt's text and token ids, and the two seeds of its cache
///   partition. The engine hashes the prompt under them, where a classic
///   policy keys its trees under the partition's marker.
/// - The prompt's entry in the token tree the gateway lends the engine
///   ([`TokenEntry`]): the inputs themselves, which name the classic
///   policy's key. The engine asks for it where a decision reads that tree,
///   and of no other request.
/// - The headers, and the body's `rid` as the request states it
///   ([`PlacementInputs::rid`]). The engine derives the request's routing
///   value from the sources its document names, and takes the id's retry
///   and turn suffixes off by its document's own rule. The key classic
///   placement derives of the id ([`PlacementInputs::rid_key`]) is not
///   handed: it lost its suffixes once, and stands only under the sticky
///   override. The one exception is the HTTP pair route: its attempt holds
///   the derived key alone, and states it as the id.
/// - The route's own rule for a worker (the candidate filter). The engine
///   asks it of the worker it picked, where classic placement narrows the
///   list by it.
/// - The retry number, as the count of engines the request tried: the
///   inputs hold no other. It is the retry number where each attempt took
///   an engine of its own, and less where an engine was tried twice. A
///   pair's failed attempt can name two engines, so a pair's caller states
///   the number over this request's.
/// - The most tokens the answer may hold, as the route states the body's
///   limit for the engine alone ([`Stated::max_output_tokens`]). Zero
///   states none: a body with no limit, a route that parsed no body.
/// - The wire the request is pinned to, as the one hard demand it states:
///   the number of the wire's runtime and transport ([`wire_number`]). The
///   engine holds the request to the workers that state the same number
///   ([`stated_facts`]), a pair's request a side at a time, and decides
///   among them. A request pinned to no wire states no demand.
///
/// Neither the inputs nor what a route states hold a tenant, a priority, a
/// media count or a budget, so the request states none.
#[must_use]
pub(crate) fn place_request<'a>(
    model_id: &'a str,
    lists: WorkerLists<'a>,
    inputs: &'a PlacementInputs<'a>,
    tried: &'a [WorkerId],
    wire: Option<WireConstraint>,
    stated: Stated,
) -> PlaceRequest<'a> {
    let headers: &'a dyn HeaderRead = match inputs.headers {
        Some(headers) => headers,
        None => &(),
    };
    PlaceRequest {
        prompt: Prompt {
            tokens: inputs.tokens,
            text: inputs.text,
            seeds: seeds(inputs.cache_namespace),
            entry: Some(inputs),
        },
        headers,
        request_id: inputs.rid,
        attempt: u8::try_from(inputs.tried.len()).unwrap_or(u8::MAX),
        tried,
        admits: inputs
            .candidate_filter
            .map(|rule| -> WorkerRule<'a> { rule }),
        max_output_tokens: stated.max_output_tokens,
        demands: Demands {
            runtime: wire.map_or(0, |wire| wire_number(wire.runtime, wire.connection)),
            ..Demands::default()
        },
        ..PlaceRequest::of(model_id, lists)
    }
}

/// The two seeds of a request's cache partition, and the plain ones for a
/// request with none: the seed its workers hash the partition's blocks
/// under, and the partition's whole identity.
fn seeds(namespace: Option<CacheNamespace>) -> HashSeeds {
    namespace.map_or(HashSeeds::PLAIN, |namespace| HashSeeds {
        event: namespace.event_seed(),
        partition: namespace.partition_seed(),
    })
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};
    use openai_protocol::common::CachePartition;
    use smg_policy_controller::engine::{LentList, ListCell};

    use super::*;
    use crate::routers::{
        common::placement::PipelinePin, grpc::backend_client::BasicWorkerBuilder,
    };

    const MODEL: &str = "m";

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new(url).build())
    }

    fn ranked(url: &str, rank: usize) -> Arc<dyn Worker> {
        Arc::new(BasicWorkerBuilder::new(url).dp_config(rank, 2).build())
    }

    /// Each input a placement function takes reaches the engine under the
    /// field that names it, over the list its caller hands, with the output
    /// limit its route states for the engine alone, and the request states
    /// nothing else.
    #[test]
    fn a_request_states_what_its_placement_inputs_hold() {
        let tokens = [7_u32, 8, 9];
        let mut headers = HeaderMap::new();
        headers.insert("x-smg-routing-key", HeaderValue::from_static("conv"));
        let partition = CachePartition {
            cache_salt: Some("tenant"),
            extra_key: Some("k"),
            lora_path: None,
        };
        let namespace = CacheNamespace::derive(&partition).expect("a partition");
        let fleet: Arc<[Arc<dyn Worker>]> = Arc::from([worker("http://a:1"), worker("http://b:1")]);
        let cell = ListCell::default();
        let lent = LentList {
            workers: &fleet,
            cell: &cell,
            facts: None,
        };
        let filter = |candidate: &dyn Worker| candidate.url() == "http://b:1";
        let engines = ["http://a:1".to_owned()];
        let mut ids = TriedIds::new();
        tried_ids(&fleet, &engines, &mut ids);
        let inputs = PlacementInputs {
            text: Some("a prompt"),
            tokens: Some(&tokens),
            headers: Some(&headers),
            rid_key: Some("session-7"),
            rid: Some("session-7_t2"),
            cache_namespace: Some(namespace),
            candidate_filter: Some(&filter),
            tried: &engines,
        };
        let stated = Stated {
            max_output_tokens: 256,
        };
        let request = place_request(
            MODEL,
            WorkerLists::Single(lent),
            &inputs,
            &ids,
            None,
            stated,
        );

        assert_eq!(request.model, MODEL);
        assert!(
            matches!(
                request.lists,
                WorkerLists::Single(list)
                    if std::ptr::eq(list.workers, &fleet) && std::ptr::eq(list.cell, &cell)
            ),
            "the list its caller handed, with its cell"
        );
        assert_eq!(request.prompt.tokens, Some(&tokens[..]));
        assert_eq!(request.prompt.text, Some("a prompt"));
        let seeds = HashSeeds {
            event: namespace.event_seed(),
            partition: namespace.partition_seed(),
        };
        assert_eq!(request.prompt.seeds, seeds);
        assert_eq!(
            request.headers.header("x-smg-routing-key"),
            Some(&b"conv"[..])
        );
        assert_eq!(request.headers.header("x-absent"), None);
        // The id as the request states it, and not the key classic derives.
        assert_eq!(request.request_id, Some("session-7_t2"));
        assert_eq!(request.attempt, 1);
        assert_eq!(request.tried, [worker_id(fleet[0].as_ref())]);
        // The candidate filter is the route's rule: the same function.
        let admits = request.admits.expect("a rule");
        assert!(!admits(fleet[0].as_ref()) && admits(fleet[1].as_ref()));

        // The body's output limit, as the route stated it for the engine.
        assert_eq!(request.max_output_tokens, 256);

        // What neither the inputs nor the route's statement hold is not
        // stated.
        assert!(request.tenant.is_none() && request.priority == 0 && !request.streaming);
        assert_eq!(request.budget_micros, 0);
        assert!(request.demands.is_none() && request.encode.is_none() && !request.decode_again);

        // A prompt of text alone is hashed under its partition's seeds too.
        let worded = PlacementInputs {
            tokens: None,
            ..inputs
        };
        let nothing = Stated::nothing();
        let request = place_request(MODEL, WorkerLists::None, &worded, &ids, None, nothing);
        assert!(request.prompt.tokens.is_none());
        assert_eq!(request.prompt.seeds, seeds);
        assert_eq!(request.max_output_tokens, 0, "a route that states no limit");

        // A request pinned to a wire states the number of that wire's
        // runtime and transport as its one demand. What else the wire holds
        // of a retained plan is the route's own rule to keep.
        let pinned = WireConstraint {
            runtime: RuntimeType::Vllm,
            connection: ConnectionMode::Grpc,
            requires_media_refs: true,
            pipeline: PipelinePin::Only,
        };
        let request = place_request(
            MODEL,
            WorkerLists::None,
            &inputs,
            &ids,
            Some(pinned),
            nothing,
        );
        let demanded = Demands {
            runtime: wire_number(RuntimeType::Vllm, ConnectionMode::Grpc),
            ..Demands::default()
        };
        assert_eq!(request.demands, demanded);
        assert!(request.admits.is_some() && request.attempt == 1);
    }

    /// One number names one wire. Two workers have one number exactly where
    /// a pinned wire holds them alike ([`WireConstraint::admits`]), over
    /// every runtime and transport, and no number is zero, which states
    /// none. The gateway states that number of each worker of a list, in
    /// the list's order, and nothing else.
    #[test]
    fn one_number_names_one_wire() {
        const RUNTIMES: [RuntimeType; 8] = [
            RuntimeType::Unspecified,
            RuntimeType::Sglang,
            RuntimeType::Vllm,
            RuntimeType::Trtllm,
            RuntimeType::Mlx,
            RuntimeType::TokenSpeed,
            RuntimeType::Generic,
            RuntimeType::External,
        ];
        const CONNECTIONS: [(ConnectionMode, &str); 3] = [
            (ConnectionMode::Http, "http"),
            (ConnectionMode::Grpc, "grpc"),
            (ConnectionMode::Zmq, "zmq"),
        ];
        // Every runtime and transport by its name and no rest: one added
        // later does not build until the two lists above name it.
        let listed = |runtime: RuntimeType, connection: ConnectionMode| {
            match runtime {
                RuntimeType::Unspecified
                | RuntimeType::Sglang
                | RuntimeType::Vllm
                | RuntimeType::Trtllm
                | RuntimeType::Mlx
                | RuntimeType::TokenSpeed
                | RuntimeType::Generic
                | RuntimeType::External => {}
            }
            match connection {
                ConnectionMode::Http | ConnectionMode::Grpc | ConnectionMode::Zmq => {}
            }
        };
        let mut wires = Vec::new();
        let mut workers: Vec<Arc<dyn Worker>> = Vec::new();
        for runtime in RUNTIMES {
            for (connection, scheme) in CONNECTIONS {
                listed(runtime, connection);
                let url = format!("{scheme}://w:{}", wires.len());
                let built = BasicWorkerBuilder::new(url)
                    .connection_mode(connection)
                    .runtime_type(runtime)
                    .build();
                wires.push((runtime, connection));
                workers.push(Arc::new(built));
            }
        }
        let facts = stated_facts(&workers);
        assert_eq!(facts.len(), workers.len());
        for (at, (runtime, connection)) in wires.iter().copied().enumerate() {
            let number = wire_number(runtime, connection);
            assert_ne!(number, 0, "{runtime:?} over {connection:?}");
            let mut stated = FixedFacts::default();
            stated.demands.runtime = number;
            assert_eq!(facts[at], stated, "{runtime:?} over {connection:?}");
            let pinned = Some(WireConstraint {
                runtime,
                connection,
                requires_media_refs: false,
                pipeline: PipelinePin::Any,
            });
            for (other, worker) in workers.iter().enumerate() {
                let (theirs, their_connection) = wires[other];
                assert_eq!(
                    WireConstraint::admits(pinned, worker.as_ref()),
                    number == wire_number(theirs, their_connection),
                    "{runtime:?} over {connection:?} beside {theirs:?} over {their_connection:?}"
                );
            }
        }
    }

    /// A request's entry in a lent token tree is the key the classic policy
    /// keys its own tree by ([`CacheNamespace::keyed_tokens`]): the token
    /// ids behind the partition's marker, which fills whole pages of the
    /// tree asked for, and the ids alone for a request in no partition. A
    /// request that holds no token ids names no entry.
    ///
    /// The classic policy keys its tree by the same function at its own
    /// page (`model_gateway/src/policies/cache_aware.rs`, `keyed_tokens`,
    /// read by `select_worker_with_tokens`).
    #[test]
    fn a_requests_entry_is_the_classic_policys_key() {
        let tokens: Vec<u32> = (0..40).collect();
        let partition = CachePartition {
            cache_salt: Some("tenant"),
            extra_key: None,
            lora_path: None,
        };
        let namespace = CacheNamespace::derive(&partition).expect("a partition");
        let key_of = |inputs: &PlacementInputs<'_>, page: usize| {
            let nothing = Stated::nothing();
            let request = place_request(MODEL, WorkerLists::None, inputs, &[], None, nothing);
            let entry = request.prompt.entry.expect("a request states its entry");
            let key = entry.token_key(page)?;
            Some((key.tokens.into_owned(), key.marker))
        };

        let partitioned = PlacementInputs {
            tokens: Some(&tokens),
            cache_namespace: Some(namespace),
            ..PlacementInputs::default()
        };
        for page in [1, 16, 64] {
            let marker = CacheNamespace::token_marker_len(page);
            assert_eq!(marker % page, 0, "the marker fills whole pages");
            let keyed = namespace.prefixed_tokens(&tokens, page);
            assert_eq!(key_of(&partitioned, page), Some((keyed, marker)));
        }

        let plain = PlacementInputs {
            tokens: Some(&tokens),
            ..PlacementInputs::default()
        };
        assert_eq!(key_of(&plain, 16), Some((tokens.clone(), 0)));

        let worded = PlacementInputs {
            text: Some("a prompt"),
            cache_namespace: Some(namespace),
            ..PlacementInputs::default()
        };
        assert_eq!(key_of(&worded, 16), None);
    }

    /// A first attempt that hands nothing states its model alone, as the
    /// engine's own first attempt does: no key classic derived stands in
    /// for an id the request does not state.
    #[test]
    fn a_request_with_no_input_states_its_model_alone() {
        let (first, nothing) = (PlacementInputs::default(), Stated::nothing());
        let request = place_request(MODEL, WorkerLists::None, &first, &[], None, nothing);
        assert_eq!(request.model, MODEL);
        assert!(matches!(request.lists, WorkerLists::None));
        assert!(request.prompt.tokens.is_none() && request.prompt.text.is_none());
        assert_eq!(request.prompt.seeds, HashSeeds::PLAIN);
        assert_eq!(request.headers.header("x-smg-routing-key"), None);
        assert!(request.request_id.is_none() && request.admits.is_none());
        assert!(request.attempt == 0 && request.tried.is_empty());
        assert!(request.demands.is_none(), "pinned to no wire, no demand");

        let keyed = PlacementInputs {
            rid_key: Some("session-7"),
            ..PlacementInputs::default()
        };
        let request = place_request(MODEL, WorkerLists::None, &keyed, &[], None, nothing);
        assert!(request.request_id.is_none());
    }

    /// The retry number is the count of engines tried, and the most the
    /// engine's counter holds behind it.
    #[test]
    fn the_retry_number_counts_the_engines_tried() {
        let engines: Vec<String> = (0..256).map(|at| format!("http://e{at}:1")).collect();
        let attempt = |tried: &[String]| {
            let inputs = PlacementInputs {
                tried,
                ..PlacementInputs::default()
            };
            place_request(
                MODEL,
                WorkerLists::None,
                &inputs,
                &[],
                None,
                Stated::nothing(),
            )
            .attempt
        };
        assert_eq!(attempt(&engines[..3]), 3);
        assert_eq!(attempt(&engines[..255]), 255);
        assert_eq!(attempt(&engines), 255, "no engine past the counter's last");
    }

    /// A tried engine names every worker at its base URL, by the id the
    /// engine knows each by. An engine outside the list names nobody,
    /// wherever it stands among the tried, and the buffer is written anew
    /// at each call.
    #[test]
    fn a_tried_engine_names_the_workers_at_its_base_url() {
        let fleet = [
            worker("http://a:1"),
            ranked("http://b:1", 0),
            ranked("http://b:1", 1),
            worker("http://c:1"),
        ];
        let id = |at: usize| worker_id(fleet[at].as_ref());
        let mut ids = TriedIds::new();

        tried_ids(&fleet, &["http://b:1".to_owned()], &mut ids);
        assert_eq!(ids, [id(1), id(2)], "both ranks of the tried engine");
        assert_ne!(id(1), id(2));

        let two = ["http://c:1".to_owned(), "http://gone:1".to_owned()];
        tried_ids(&fleet, &two, &mut ids);
        assert_eq!(ids, [id(3)]);

        let behind = ["http://gone:1".to_owned(), "http://a:1".to_owned()];
        tried_ids(&fleet, &behind, &mut ids);
        assert_eq!(ids, [id(0)], "an engine named behind one that left");

        tried_ids(&fleet, &[], &mut ids);
        assert!(ids.is_empty());
    }
}
