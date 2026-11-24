//! What a placement function states of its request to the policy engine.
//!
//! The placement functions take one [`PlacementInputs`] from their route:
//! what a classic policy reads of a request. Where the engine decides, they
//! ask it with a [`PlaceRequest`] instead. This file fills that request from
//! the same inputs, each under the name the engine reads it by, and from
//! what the route states for the engine alone ([`Stated`]), which a classic
//! placement is handed nothing of.

use std::sync::Arc;

use smg_policy_controller::engine::{
    worker_id, HashSeeds, HeaderRead, PlaceRequest, Prompt, TokenEntry, TokenKey, WorkerId,
    WorkerLists, WorkerRule,
};
use smg_worker::identity::basic::Worker;

use super::placement::{was_tried, PlacementInputs, Stated};
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

/// The request `inputs` and `stated` state for `model_id` over `lists`, as
/// the engine reads it. `tried` are the workers at the engines the
/// request's failed attempts were sent to ([`tried_ids`]).
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
///
/// Neither the inputs nor what a route states hold a tenant, a priority, a
/// media count, a demand or a budget, so the request states none.
#[must_use]
pub(crate) fn place_request<'a>(
    model_id: &'a str,
    lists: WorkerLists<'a>,
    inputs: &'a PlacementInputs<'a>,
    tried: &'a [WorkerId],
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
    use crate::routers::grpc::backend_client::BasicWorkerBuilder;

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
        let request = place_request(MODEL, WorkerLists::Single(lent), &inputs, &ids, stated);

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
        let request = place_request(MODEL, WorkerLists::None, &worded, &ids, Stated::nothing());
        assert!(request.prompt.tokens.is_none());
        assert_eq!(request.prompt.seeds, seeds);
        assert_eq!(request.max_output_tokens, 0, "a route that states no limit");
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
            let request = place_request(MODEL, WorkerLists::None, inputs, &[], Stated::nothing());
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
        let nothing = PlacementInputs::default();
        let request = place_request(MODEL, WorkerLists::None, &nothing, &[], Stated::nothing());
        assert_eq!(request.model, MODEL);
        assert!(matches!(request.lists, WorkerLists::None));
        assert!(request.prompt.tokens.is_none() && request.prompt.text.is_none());
        assert_eq!(request.prompt.seeds, HashSeeds::PLAIN);
        assert_eq!(request.headers.header("x-smg-routing-key"), None);
        assert!(request.request_id.is_none() && request.admits.is_none());
        assert!(request.attempt == 0 && request.tried.is_empty());

        let keyed = PlacementInputs {
            rid_key: Some("session-7"),
            ..PlacementInputs::default()
        };
        let request = place_request(MODEL, WorkerLists::None, &keyed, &[], Stated::nothing());
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
            place_request(MODEL, WorkerLists::None, &inputs, &[], Stated::nothing()).attempt
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
