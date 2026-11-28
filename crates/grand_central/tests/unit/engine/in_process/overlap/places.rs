//! A request its host holds to places: the holders and the deepest holder it
//! is named among them, and the workers a string tree's match hands among
//! them.

use super::*;

/// A request its host holds to places is named its holders among the
/// places: the index is asked within them, so holders outside cannot fill
/// the sample. Of twenty workers that hold the whole prompt, a request held
/// to two is named both under every seed, where the sample of sixteen
/// would leave each out for some. The classic policy narrows its workers
/// first (`model_gateway/src/routers/common/placement.rs:474-484`) and
/// gathers the deepest holders among them, one or both by its cap's draw
/// (`top_group`, `holders_at`, `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_held_request_is_named_its_holders_among_its_places() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 2);
    for at in 0..20 {
        store(&index, at, &hashes_of(&asked));
    }
    let bed = Bed::new(affinity, preset("cache_aware"), 20);
    for seed in 0..32 {
        let named = bed.held_to(&request(&asked, seed, NOW), &[3, 17]);
        assert_eq!(named.holders, [(3, 2, 0), (17, 2, 0)], "seed {seed}");
    }
    // A request that states no places is named the sample of the list.
    let named = bed.decide(&request(&asked, 7, NOW), None).expect("rows");
    assert_eq!(named.holders.len(), 16);
}

/// A shorter beginning held among the places is named where deeper holders
/// outside them are more than the sample: it is the deepest among the
/// places. The classic policy's whole gather finds it where none of its
/// drawn holders is a candidate
/// (`top_group`, `model_gateway/src/policies/cache_aware.rs`).
#[test]
fn a_held_request_is_named_the_deepest_holder_among_its_places() {
    let (affinity, _, index) = hosted(true);
    let index = index.expect("the index");
    let asked = prompt(1_000, 8);
    let hashes = hashes_of(&asked);
    for at in 0..17 {
        store(&index, at, &hashes);
    }
    store(&index, 19, &hashes[..3]);
    // The index knows the other place by a number alone.
    index.intern_worker(&address(18)).expect("a number");
    let bed = Bed::new(affinity, preset("cache_aware"), 20);
    for seed in 0..32 {
        let named = bed.held_to(&request(&asked, seed, NOW), &[18, 19]);
        assert_eq!(named.holders, [(19, 3, 0)], "seed {seed}");
        // Stated to no place, the seventeen deeper holders fill the sample.
        let named = bed.decide(&request(&asked, seed, NOW), None).expect("rows");
        assert!(named
            .holders
            .iter()
            .all(|&(row, held, _)| row < 17 && held == 8));
    }
    // The request's mask is the table's, kept to the places: a worker that
    // gains its first blocks while the table stands is not the index's to
    // name in that second, for a request held to it as for any other.
    store(&index, 18, &hashes);
    let named = bed.held_to(&request(&asked, 0, NOW), &[18, 19]);
    assert_eq!(named.holders, [(19, 3, 0)]);
    let named = bed.decide(&request(&asked, 0, NOW), None).expect("rows");
    assert!(named.holders.iter().all(|&(row, _, _)| row < 17));
}

/// A request of text its host holds to places is named the workers its
/// string tree's match hands among those places, as a tokenized one is
/// ([`a_held_request_is_named_a_trees_workers_among_its_places`]).
#[test]
fn a_held_request_of_text_is_named_a_trees_workers_among_its_places() {
    let (affinity, _) = lending_text(false);
    let bed = Bed::new(affinity, sampled(4), 20);
    let asked = text(1_000, 4);
    for at in 0..8 {
        bed.worded(&asked, 0, Some(at));
    }
    let hashes = worded(&asked);
    let key = || TextKey {
        text: Cow::Borrowed(&asked),
        marker: 0,
    };
    for seed in 0..32 {
        let facts = request(&hashes, seed, NOW);
        let named = bed.decide_entered(&facts, Some(&[2, 5]), (None, Some(key())), None);
        let named = named.expect("rows named");
        assert_eq!(named.holders, [(2, 0, 4), (5, 0, 4)], "seed {seed}");
    }
    // Stated to no place, the sample cuts the eight to four.
    assert_eq!(bed.worded(&asked, 0, None).len(), 4);
}
