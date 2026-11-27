//! The cases of a frame's key on a hash ring: where a client's key places
//! its request, and what the prefix ring and the plain ring read of who
//! handed a session key.

use super::*;

/// A frame's client key places its request on the ring: one client's
/// requests take one worker whatever their seeds, and clients spread
/// over the workers.
#[tokio::test]
async fn a_frames_client_key_places_its_request_on_the_ring() {
    let mut bed = bed(Some(preset("consistent_hashing", 1)), true, (2, 8), []);
    let from = |id: u8, client: u64| {
        let mut frame = frame(id, u64::from(id));
        frame
            .facts
            .as_mut()
            .expect("the frame has facts")
            .client_key = client;
        frame
    };
    let mut of_one = Vec::new();
    for id in 1..=12u8 {
        of_one.push(picked(&bed.ask(from(id, 0x9e37_79b9_7f4a_7c15)).await));
    }
    assert!(
        of_one.windows(2).all(|pair| pair[0] == pair[1]),
        "{of_one:?}"
    );
    let mut of_many = Vec::new();
    for id in 13..=28u8 {
        let client = u64::from(id).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        of_many.push(picked(&bed.ask(from(id, client)).await));
    }
    assert!(
        of_many.windows(2).any(|pair| pair[0] != pair[1]),
        "{of_many:?}"
    );
}

/// The worker `frame` takes on `bed`. Its lease is given up where it
/// stands, so the request weighs on no later one.
async fn placed(bed: &mut Bed, frame: proto::DecideRequest) -> WorkerId {
    use crate::service::controller::lifecycle::tests::a_released;
    let answer = bed.ask(frame).await;
    let released = a_released(lease_of(&answer).id);
    lifecycle::released(&bed.store, &Hand::new(GATEWAY), &released).expect("the frame reads");
    picked(&answer)
}

/// A frame says who handed its session key, and the prefix form of a
/// ring reads the word. The document is the settings' for `prefix_hash`
/// under the sticky override. Twelve keys under one prompt, each
/// handed by its request's own id, take the worker a request of that
/// prompt takes with no key: the ring places them by the prompt. Twelve
/// keys under the same prompt whose frames say nothing lead it, and
/// stand on more workers than that one: a header's keys, and every key
/// of a gateway older than the field. The id's key names the session
/// still: once dispatched, its next request follows its pin under a
/// prompt another worker owns.
#[tokio::test]
async fn the_prefix_ring_places_a_frame_by_its_prompt_where_its_id_handed_the_key() {
    use crate::{
        document::{PolicySettings, RoutingKeyOverrideSettings, RoutingSettings},
        engine::sticky_sessions::SessionVerdict,
        service::controller::lifecycle::tests::a_dispatched,
    };
    let settings = RoutingSettings {
        policy: PolicySettings::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: Vec::new(),
        },
        routing_key_override: RoutingKeyOverrideSettings {
            enabled: true,
            ..RoutingKeyOverrideSettings::default()
        },
        ..RoutingSettings::default()
    };
    let mut document = settings.policy_document().expect("the settings' document");
    document.generation = 1;
    let mut bed = bed(Some(applied(&document)), true, (1, 8), []);
    // A frame of four blocks from `prompt` on, sixteen tokens a block,
    // under `key` where it has one, and `by_id` is the word it says of
    // that key.
    let asked = |key: Option<u64>, by_id: bool, prompt: u64, seed: u64| {
        let mut one = match key {
            Some(key) => keyed(key.wrapping_mul(0x9e37_79b9_7f4a_7c15), seed),
            None => frame(u8::try_from(seed % 251).expect("under 251"), seed),
        };
        let facts = one.facts.as_mut().expect("the case's frame has facts");
        facts.block_hashes = (prompt..prompt + 4).collect();
        facts.prompt_tokens = 64;
        facts.session_key_from_rid = by_id;
        one
    };
    // Two prompts that two workers own: a request with no key goes by
    // its prompt.
    const NEAR: u64 = 1_000;
    let here = placed(&mut bed, asked(None, false, NEAR, 1)).await;
    let (mut far, mut there) = (NEAR, here);
    while there == here {
        far += 100;
        assert!(far < 4_000, "every prompt stands on one worker");
        there = placed(&mut bed, asked(None, false, far, far)).await;
    }
    // The requests' own ids handed these keys: each goes where its
    // prompt goes.
    for key in 1..=12 {
        let by_id = placed(&mut bed, asked(Some(key), true, NEAR, 100 + key)).await;
        assert_eq!(by_id, here, "key {key}");
    }
    // These frames say nothing of their keys, and each key leads the
    // prompt.
    let mut led = Vec::new();
    for key in 13..=24 {
        led.push(placed(&mut bed, asked(Some(key), false, NEAR, 100 + key)).await);
    }
    assert!(led.iter().any(|&worker| worker != here), "{led:?}");
    // An id's key is its session's name: dispatched, it pins its
    // worker, and its next request follows the pin under the prompt
    // the other worker owns.
    let first = bed.ask(asked(Some(25), true, NEAR, 125)).await;
    assert_eq!(picked(&first), here);
    let (hand, at) = (Hand::new(GATEWAY), bed.script.taken);
    lifecycle::dispatched(&bed.store, &hand, &a_dispatched(lease_of(&first).id), at)
        .expect("the frame reads");
    let next = bed.ask(asked(Some(25), true, far, 126)).await;
    assert_eq!(picked(&next), here, "its pin, and not {there:?}");
    let filed = LeasesStore::get(&*bed.store, GATEWAY, LeaseId(lease_of(&next).id));
    let filed = filed.expect("the lease is on file").lease;
    assert_eq!(filed.session_verdict, SessionVerdict::Pinned);
}

/// The plain ring reads no word of who handed a key. Under
/// `consistent_hashing` a frame that says its request's own id handed
/// its key takes the worker that key takes with nothing said, and the
/// keys stand on more than one worker.
#[tokio::test]
async fn the_plain_ring_places_a_frame_by_its_key_whoever_handed_it() {
    let mut bed = bed(Some(preset("consistent_hashing", 1)), true, (2, 8), []);
    let said = |key: u64, by_id: bool| {
        let mut one = keyed(key.wrapping_mul(0x9e37_79b9_7f4a_7c15), key);
        let facts = one.facts.as_mut().expect("the case's frame has facts");
        facts.session_key_from_rid = by_id;
        one
    };
    let mut by_key = Vec::new();
    for key in 1..=12 {
        let silent = placed(&mut bed, said(key, false)).await;
        let by_id = placed(&mut bed, said(key, true)).await;
        assert_eq!(by_id, silent, "key {key}");
        by_key.push(silent);
    }
    assert!(
        by_key.windows(2).any(|pair| pair[0] != pair[1]),
        "{by_key:?}"
    );
}
