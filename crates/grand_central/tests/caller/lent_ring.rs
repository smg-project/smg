//! The program's own ring, lent with each request.
//!
//! The program keeps a hash ring of a model's workers for each generation
//! ([`Ring`]) and lends it with every request. Under `consistent_hashing` a
//! request's routing key then goes to the worker that ring's walk reaches
//! from where the key stands. Each case reads the ring the program holds
//! and expects the sends its walk says.
//!
//! [`Ring`]: crate::ring::Ring

use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

use http::{HeaderMap, HeaderValue};
use smg_grand_central::{
    document::PolicySettings,
    engine::{Identity, WorkerId},
    policies::consistent_hashing::LentRing,
};
use smg_worker::{identity::basic::Worker, worker::WorkerType};

use crate::{
    pair::{self, typed, PLAIN},
    program::{Answer, Asked, Program},
    served, under, under_encode, under_pairs, workers, MODEL,
};

/// The header a gateway's settings read a routing key from.
const KEY_HEADER: &str = "x-smg-routing-key";

/// Sixteen routing keys.
fn keys() -> Vec<String> {
    (0..16).map(|key| format!("session-{key}")).collect()
}

/// The addresses the program's ring walks from where `value` stands, over
/// the generation that stands.
fn walked(program: &Program, value: &str) -> Vec<String> {
    let lists = program.published(MODEL).expect("the model's generation");
    lists.ring().walked(value).map(str::to_owned).collect()
}

/// One request under the routing key `key`, as the program answered it.
fn served_under(program: &Program, key: &str) -> Answer {
    let mut headers = HeaderMap::new();
    let value = HeaderValue::from_str(key).expect("a header's value");
    headers.insert(KEY_HEADER, value);
    let asked = Asked {
        headers: Some(&headers),
        ..PLAIN
    };
    program.serve(&asked)
}

/// Under `consistent_hashing` a routing key goes to the worker the program's
/// ring puts it on: the first worker of the ring's walk from where the key
/// stands. Sixteen keys over four workers each go there, every time, and
/// they spread over the workers.
///
/// The gateway lends the ring its own ring policy walks
/// (`impl LentRing for HashRing`,
/// `model_gateway/src/policies/hash_ring.rs:343-359`), read for each request
/// (`ModelRing`, `model_gateway/src/routers/common/placement.rs:1057-1133`),
/// so a key keeps the worker that policy sends it to: its case
/// `the_engine_places_a_keyed_request_where_the_classic_ring_does`
/// (`:8138-8167`).
#[test]
fn a_routing_key_goes_to_the_worker_the_programs_ring_puts_it_on() {
    let fleet = workers(4);
    let (program, _loads, _task) = under(PolicySettings::ConsistentHashing, &fleet, served);
    let mut owners = BTreeSet::new();
    for key in keys() {
        let owner = walked(&program, &key).swap_remove(0);
        for _ in 0..2 {
            let answer = served_under(&program, &key);
            assert_eq!((answer.status, &answer.sent), (200, &vec![owner.clone()]));
        }
        owners.insert(owner);
    }
    assert!(owners.len() > 1, "sixteen keys on one worker");
}

/// A worker that leaves hands its keys to the next worker of the ring, and
/// every other key stays where it was: the next generation's ring is the
/// old one without the worker.
///
/// The gateway's ring is one of a generation too: a request is lent the ring
/// of the generation its list is read from (`ModelRing::of`,
/// `model_gateway/src/routers/common/placement.rs:1087-1101`).
#[test]
fn a_worker_that_leaves_hands_its_keys_to_the_next_on_the_ring() {
    let fleet = workers(4);
    let (program, _loads, _task) = under(PolicySettings::ConsistentHashing, &fleet, served);
    let keys = keys();
    let before: Vec<Vec<String>> = keys.iter().map(|key| walked(&program, key)).collect();
    let left = before[0][0].clone();
    assert!(program.leave(&left), "a worker that joined leaves");
    let mut moved = 0;
    for (key, walk) in keys.iter().zip(&before) {
        let expected = if walk[0] == left {
            moved += 1;
            &walk[1]
        } else {
            &walk[0]
        };
        let answer = served_under(&program, key);
        assert_eq!(
            (answer.status, &answer.sent),
            (200, &vec![expected.clone()])
        );
    }
    assert!((1..keys.len()).contains(&moved), "{moved} keys moved");
}

/// A retry walks on from the worker it tried: a key whose worker answers a
/// server error is asked again at the next worker of the ring's walk.
///
/// The gateway's ring policy falls through so, and the engine behind the
/// lent ring with it: its cases
/// `a_retry_under_consistent_hashing_falls_through_to_the_next_ring_node`
/// and `the_engine_places_a_keyed_request_where_the_classic_ring_does`
/// (`model_gateway/src/routers/common/placement.rs:5394-5456`, `:8138-8167`).
#[test]
fn a_retry_walks_on_from_the_worker_it_tried() {
    let fleet = workers(4);
    let failing = Arc::new(Mutex::new(String::new()));
    let fails = Arc::clone(&failing);
    let sends = move |worker: &dyn Worker| {
        let failing = fails.lock().expect("no case panicked");
        if worker.url() == *failing {
            502
        } else {
            200
        }
    };
    let (program, _loads, _task) = under(PolicySettings::ConsistentHashing, &fleet, sends);
    for key in keys() {
        let walk = walked(&program, &key);
        failing
            .lock()
            .expect("no case panicked")
            .clone_from(&walk[0]);
        let answer = served_under(&program, &key);
        assert_eq!(
            (answer.status, &answer.sent[..]),
            (200, &walk[..2]),
            "the key {key}"
        );
    }
}

/// The ring's two walks agree: from where each key stands, the ids its walk
/// by ids hands are the ids of the addresses its walk by address hands, in
/// that order, each worker once. The engine asks a lent ring the walk by
/// ids alone, and this ring answers it from the ids it kept, so no request
/// reads the walk by address: the trait holds the two to one order
/// (`LentRing::walk_ids`).
#[test]
fn the_rings_two_walks_name_the_same_workers() {
    let fleet = workers(4);
    let (program, _loads, _task) = under(PolicySettings::ConsistentHashing, &fleet, served);
    let lists = program.published(MODEL).expect("the model's generation");
    let ring = lists.ring();
    for key in keys() {
        let position = ring.position(&key);
        let (mut by_address, mut by_id) = (Vec::new(), Vec::new());
        ring.walk(position, &mut |address| {
            by_address.push(WorkerId::from_address(address));
            false
        });
        ring.walk_ids(position, &mut |id| {
            by_id.push(id);
            false
        });
        assert_eq!(by_id.len(), fleet.len(), "each worker once: the key {key}");
        assert_eq!(by_id, by_address, "the key {key}");
    }
}

/// One ring serves a pair's two lists: the prefill is the first prefill of
/// the key's walk and the decode the first decode of it. A worker of the
/// other kind that stands on the way is passed over.
///
/// The gateway reads the model's one ring once for a pair and hands it to
/// both sides' policies
/// (`model_gateway/src/routers/common/placement.rs:3000-3014`), and lends the
/// engine that ring: its case
/// `the_engine_places_a_pair_of_two_ring_words_where_its_classic_policies_do`
/// (`:8660-8771`).
#[test]
fn a_pairs_two_sides_are_walked_on_the_one_ring() {
    let fleet = pair::fleet(3, 3);
    let (program, _loads, _task) = under_pairs(PolicySettings::ConsistentHashing, &fleet, served);
    for key in keys() {
        let walk = walked(&program, &key);
        let first = |kind: &str| {
            let of_kind = walk.iter().find(|address| address.contains(kind));
            of_kind.expect("a worker of the kind").clone()
        };
        let expected = vec![first("prefill"), first("decode")];
        let answer = served_under(&program, &key);
        assert_eq!((answer.status, &answer.sent), (200, &expected), "{key}");
    }
}

/// An encode request's item goes to the encoder the ring reaches from where
/// the item's text stands: the first encoder of that walk. The sides' own
/// policy reads no ring, and the items are lent one all the same.
///
/// The gateway hands each item's selection the model's ring, with the item's
/// content hash as its routing key (`assign_encode_workers`,
/// `model_gateway/src/routers/grpc/common/stages/worker_selection.rs:2791-2811`).
#[test]
fn an_item_goes_to_the_encoder_the_ring_reaches_from_its_text() {
    let encoders = (0..3).map(|at| {
        let address = format!("http://encode-{at}:8000");
        typed(&address, WorkerType::Encode)
    });
    let pairs = pair::fleet(1, 1);
    let fleet: Vec<Arc<dyn Worker>> = pairs.into_iter().chain(encoders).collect();
    let (program, _loads, _task) = under_encode(PolicySettings::RoundRobin, &fleet, served);
    let mut taken = BTreeSet::new();
    for content in 0..16 {
        let text = format!("content-{content}");
        let walk = walked(&program, &text);
        let encoder = walk.iter().find(|address| address.contains("encode"));
        let encoder = encoder.expect("an encoder").clone();
        let keys = [Identity::plain(&text)];
        let asked = Asked {
            items: &keys,
            ..PLAIN
        };
        let answer = program.serve(&asked);
        assert_eq!((answer.status, &answer.sent[0]), (200, &encoder), "{text}");
        taken.insert(encoder);
    }
    assert!(taken.len() > 1, "sixteen contents on one encoder");
}
