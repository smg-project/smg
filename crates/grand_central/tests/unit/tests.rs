use super::Draw;
use crate::engine::fingerprint;

#[test]
fn the_draw_is_deterministic_per_seed_and_stays_below_the_bound() {
    let mut a = Draw::new(7);
    let mut b = Draw::new(7);
    for _ in 0..64 {
        let x = a.below(13);
        assert_eq!(x, b.below(13));
        assert!(x < 13);
    }
    assert_eq!(Draw::new(1).below(0), 0);
}

/// A stage's stream shares no value, place for place, with the seed's
/// own stream, which the picker reads: under the empty name too. It is
/// the seed's stream under the fingerprint of the stage's name: the
/// equality a stage that keeps its fingerprint opens its stream by.
#[test]
fn a_stages_stream_is_the_seeds_under_its_names_fingerprint() {
    for seed in [0, 1, 7, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
        for name in ["pack", "spread", "a", ""] {
            let mut own = Draw::for_stage(seed, name);
            let mut pickers = Draw::new(seed);
            let mut by_hand = Draw::new(seed ^ fingerprint(name.as_bytes()));
            for place in 0..8 {
                let value = own.next_u64();
                assert_ne!(value, pickers.next_u64(), "{name} at {seed}, {place}");
                assert_eq!(value, by_hand.next_u64(), "{name} at {seed}, {place}");
            }
        }
    }
}

/// The two names, of two fingerprints, open two streams from one seed.
/// One name opens the same stream from one seed every time: the call
/// keeps nothing between two calls.
#[test]
fn two_names_open_two_streams_and_one_name_replays() {
    for seed in 0..256u64 {
        let first = |name| Draw::for_stage(seed, name).next_u64();
        assert_ne!(first("pack"), first("spread"), "seed {seed}");
        assert_eq!(first("pack"), first("pack"), "seed {seed}");
    }
}
