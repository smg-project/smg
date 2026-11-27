//! A host's standing vetoes beside a pair's two lists
//! ([`PairLists::prefill_stalled`], [`PairLists::prefill_held`],
//! [`PairLists::decode_stalled`], [`PairLists::decode_held`]): each side
//! reads the places stated beside its own list, and no place of the other.

use super::*;
use crate::engine::{decision::LiftedVeto, framework::DecisionScope};

/// The places a host's vetoes mask and hold, for the prefill list and then
/// for the decode list: what [`Host::decide_vetoed`] takes.
type Vetoes<'a> = [[&'a [u32]; 2]; 2];

/// No veto on either side.
const NONE: Vetoes<'static> = [[&[], &[]], [&[], &[]]];

/// Marks in `seen` each place a target of `decision` names: its prefill's
/// in the first list, its decode's in the second. Every target is read, a
/// pick's alternates too, and a decision that picks no pair is refused.
fn note<const P: usize, const D: usize>(seen: &mut ([bool; P], [bool; D]), decision: &Decision) {
    let targets = named(decision);
    assert!(!targets.is_empty(), "no pick: {decision:?}");
    for target in targets {
        seen.0[target.position as usize] = true;
        let pair = target.pair.expect("a pair");
        seen.1[pair.decode_position as usize] = true;
    }
}

/// A host's standing vetoes are places of one list. The second prefill and
/// the third decode are masked: no target of a pick names either, and the
/// same place of the other list, the second decode and the third prefill,
/// still takes requests. The engine's entry for a pair that issues no
/// lease reads the same slices.
#[test]
fn each_sides_vetoes_mask_that_sides_list_alone() {
    let host = Host::of(3, 3);
    let vetoes: Vetoes<'_> = [[&[1], &[]], [&[2], &[]]];
    let free = Constraints::default();
    let mut seen = ([false; 3], [false; 3]);
    for seed in 0..64 {
        let decision = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], vetoes);
        note(&mut seen, &decision);
    }
    assert_eq!(seen, ([true, false, true], [true, true, false]));

    let (classes, now) = (every(3, 3), Instant::now());
    let mut seen = ([false; 3], [false; 3]);
    for seed in 0..64 {
        let mut stated = host.asks_by(&classes, (seed, false), None);
        stated.1.prefill_stalled = &[1];
        stated.1.decode_stalled = &[2];
        let decided = host.engine.decide_without_lease(&stated, now);
        let (decision, _) = decided.expect("a decision");
        note(&mut seen, &decision);
    }
    assert_eq!(seen, ([true, false, true], [true, true, false]));
}

/// Under each preset a pair's two sides can take, a place a standing veto
/// masks is in no target of a pick, on either side, with a hold on it or
/// without. Four prefills and four decodes. The two places masked are
/// places the preset names with no veto: on each side, the lowest place
/// among the targets of its first 32 turns. No preset is left out: the one
/// whose picker leaves the placement to the sessions rule places each side
/// by the rule's assignment.
#[test]
fn a_masked_place_is_in_no_pick_under_any_pair_preset() {
    use crate::document::spec::PRESETS;
    let free = Constraints::default();
    for name in PRESETS {
        let mut document = PolicySpec::pair_presets(name, name).expect("presets");
        document.generation = 3;
        let spec = AppliedSpec::validate(&document).unwrap_or_else(|err| panic!("{name}: {err}"));
        let host = Host::under((spec, Arc::default()), 4, 4);
        // Each side's places that a target of 32 turns names under `vetoes`.
        let named_under = |vetoes: Vetoes<'_>| {
            let mut seen = ([false; 4], [false; 4]);
            for seed in 0..32 {
                let decision = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], vetoes);
                note(&mut seen, &decision);
            }
            seen
        };
        let lowest = |seen: [bool; 4]| seen.iter().position(|&seen| seen).expect("a place");
        let unmasked = named_under(NONE);
        let (prefill, decode) = (lowest(unmasked.0), lowest(unmasked.1));
        let (on_prefill, on_decode) = ([prefill as u32], [decode as u32]);
        for held in [false, true] {
            let (holds_prefill, holds_decode): (&[u32], &[u32]) = if held {
                (&on_prefill, &on_decode)
            } else {
                (&[], &[])
            };
            let vetoes: Vetoes<'_> = [[&on_prefill, holds_prefill], [&on_decode, holds_decode]];
            let masked = named_under(vetoes);
            assert!(!masked.0[prefill], "{name}, held {held}: the prefill");
            assert!(!masked.1[decode], "{name}, held {held}: the decode");
        }
    }
}

/// A side's mask is read among that side's places. A prefill list held to
/// its first two places with the second masked hands the first alone, and
/// a decode list held to its last two with the last masked hands the
/// second: no target names another place.
#[test]
fn a_sides_mask_is_read_among_that_sides_places() {
    let host = Host::of(3, 3);
    let places = [Some(&[0, 1][..]), Some(&[1, 2][..])];
    let vetoes: Vetoes<'_> = [[&[1], &[]], [&[2], &[]]];
    let mut seen = ([false; 3], [false; 3]);
    for seed in 0..32 {
        let free = Constraints::default();
        let decision = host.decide_vetoed(&OPEN, (seed, None), free, places, vetoes);
        note(&mut seen, &decision);
    }
    assert_eq!(seen, ([true, false, false], [false, true, false]));
}

/// A side's mask stands over the request's constraints and replaces
/// nothing of them but the places and the mask. The request names the
/// first decode as a worker its host just failed to reach, and the third
/// decode is masked: the pair's decode is the second at every turn, at the
/// first decision and at the decode asked again.
#[test]
fn a_sides_mask_keeps_what_the_request_states() {
    let host = Host::under(presets("round_robin"), 1, 3);
    let excluded = [worker_id(host.decodes[0].as_ref())];
    let stated = Constraints {
        excluded: &excluded,
        ..Constraints::default()
    };
    let vetoes: Vetoes<'_> = [[&[], &[]], [&[2], &[]]];
    let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
    for seed in 0..16 {
        let first = host.decide_vetoed(&OPEN, (seed, None), stated, [None; 2], vetoes);
        assert_eq!(host.picked(&first).1, 1, "seed {seed}");
        let again = host.decide_vetoed(&OPEN, (seed, Some(&held)), stated, [None; 2], vetoes);
        assert_eq!(host.picked(&again).1, 1, "seed {seed}, asked again");
    }
}

/// A side masked whole is paired while a pair can be made. The prefill
/// side's own run sheds, and the refused pool goes to the pair's default
/// ([`Engine::decide_pair`]): it lifts the liveness veto where nothing
/// else pairs, so it reads each side's mask as its holds alone, and the
/// trace says the lifted veto. It never takes a place a hold stands on.
/// With a hold on one place of the side, every target names the other.
/// With a hold on every place the decision is shed at the health stage.
/// So it is for the prefills and for the decodes, each by its own slices.
#[test]
fn a_side_masked_whole_is_paired_past_its_mask_and_never_onto_a_hold() {
    let host = Host::of(2, 2);
    let (all, first): (&[u32], &[u32]) = (&[0, 1], &[0]);
    let free = Constraints::default();
    // The side's own places a target names over 16 turns under `vetoes`,
    // each decision made by the default.
    let named = |vetoes: Vetoes<'_>| {
        let mut seen = ([false; 2], [false; 2]);
        for seed in 0..16 {
            let decision = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], vetoes);
            note(&mut seen, &decision);
            assert_eq!(decision.trace().lifted, LiftedVeto::Liveness, "{vetoes:?}");
        }
        seen
    };
    let (prefills, decodes) = (0, 1);
    for side in [prefills, decodes] {
        let on = |holds: &'static [u32]| -> Vetoes<'static> {
            let mut vetoes = NONE;
            vetoes[side] = [&[0, 1], holds];
            vetoes
        };
        // No hold: the default pairs past the mask.
        named(on(&[]));
        // A hold on the first place: the second alone is named.
        let seen = named(on(first));
        let of_side = if side == prefills { seen.0 } else { seen.1 };
        assert_eq!(of_side, [false, true], "side {side}");
        // A hold on every place: nothing pairs.
        let decision = host.decide_vetoed(&OPEN, (1, None), free, [None; 2], on(all));
        let shed = (ShedReason::NoEndpoints, "healthy");
        assert_eq!(shed_for(&decision), shed, "side {side}");
    }
}

/// The decode asked again reads the decode side's mask and no place of the
/// prefills'. Its held prefill is found though a hold stands on it by
/// then, and the decode is the one place of its list no veto masks.
#[test]
fn the_decode_asked_again_reads_the_decode_sides_mask_alone() {
    let host = Host::of(2, 3);
    let (sent, _) = host.picked(&host.decide(&OPEN, 3, None));
    let held = prefill_alone(worker_id(host.prefills[sent].as_ref()));
    let on_sent = [sent as u32];
    let vetoes: Vetoes<'_> = [[&on_sent, &on_sent], [&[0, 2], &[]]];
    let mut seen = ([false; 2], [false; 3]);
    for seed in 0..16 {
        let free = Constraints::default();
        let again = host.decide_vetoed(&OPEN, (seed, Some(&held)), free, [None; 2], vetoes);
        note(&mut seen, &again);
        assert_eq!(again.trace().lifted, LiftedVeto::None, "seed {seed}");
    }
    let mut sent_alone = [false; 2];
    sent_alone[sent] = true;
    assert_eq!(seen, (sent_alone, [false, true, false]));
}

/// A decode held back between a pair's two calls is not kept. Under round
/// robin the decode asked again is handed the decode the first decision
/// named, and answers it again while the profile's filters keep it. A
/// veto that stands on that decode by the second call, with a hold or
/// without, is read by them: no target of the answer names it. With no
/// veto the first decode is the answer.
#[test]
fn a_decode_held_back_between_the_two_calls_is_not_kept() {
    let host = Host::under(presets("round_robin"), 1, 3);
    let free = Constraints::default();
    for seed in 0..6 {
        let held = first_target(&host.decide(&OPEN, seed, None));
        let chosen = held.pair.expect("a decode").decode_position;
        let kept = host.decide_vetoed(&OPEN, (seed, Some(&held)), free, [None; 2], NONE);
        assert_eq!(host.picked(&kept).1, chosen as usize, "seed {seed}");
        let on_chosen = [chosen];
        for holds in [&[][..], &on_chosen[..]] {
            let vetoes: Vetoes<'_> = [[&[], &[]], [&on_chosen, holds]];
            let again = host.decide_vetoed(&OPEN, (seed, Some(&held)), free, [None; 2], vetoes);
            let mut seen = ([false; 1], [false; 3]);
            note(&mut seen, &again);
            assert!(!seen.1[chosen as usize], "seed {seed}: {again:?}");
        }
    }
}

/// Where every decode of the held prefill is masked, the decode asked
/// again is put to the all-vetoed rule. The rule lifts the mask as its
/// last resort and leaves masked a place a hold stands on: no target names
/// that place, and the trace says the lifted veto. Where a hold stands on
/// every place the call is shed at the health stage.
#[test]
fn a_hold_keeps_its_place_out_where_the_rule_lifts_the_decodes_mask() {
    let host = Host::of(1, 3);
    let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
    let all: &[u32] = &[0, 1, 2];
    let free = Constraints::default();
    let one_held: Vetoes<'_> = [[&[], &[]], [all, &[1]]];
    let mut seen = ([false; 1], [false; 3]);
    for seed in 0..32 {
        let again = host.decide_vetoed(&OPEN, (seed, Some(&held)), free, [None; 2], one_held);
        note(&mut seen, &again);
        assert_eq!(again.trace().lifted, LiftedVeto::Liveness, "seed {seed}");
    }
    assert_eq!(seen.1, [true, false, true]);
    let all_held: Vetoes<'_> = [[&[], &[]], [all, all]];
    let again = host.decide_vetoed(&OPEN, (1, Some(&held)), free, [None; 2], all_held);
    assert_eq!(shed_for(&again), (ShedReason::NoEndpoints, "healthy"));
}

/// A pair's request that states a mask of its own is declined under the
/// stage `places`: its places are of one list, and a pair's are stated
/// beside its two. So is the decode asked again behind a prefill its list
/// holds. What a pair is not narrowed by is answered first: a hint of 0
/// sheds the same request under its own stage.
#[test]
fn a_pairs_request_that_states_a_mask_of_its_own_is_declined_under_the_places_name() {
    let host = Host::of(2, 2);
    let declined = (ShedReason::PolicyDeclined, "places");
    let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
    let free = Constraints::default();
    let stated: [(&[u32], &[u32]); 3] = [(&[1], &[]), (&[1], &[1]), (&[], &[1])];
    for (stalled, holds) in stated {
        let own = Constraints {
            stalled,
            held: holds,
            ..free
        };
        let first = host.decide_vetoed(&OPEN, (1, None), own, [None; 2], NONE);
        assert_eq!(shed_for(&first), declined, "{stalled:?} {holds:?}");
        let again = host.decide_vetoed(&OPEN, (1, Some(&held)), own, [None; 2], NONE);
        assert_eq!(shed_for(&again), declined, "{stalled:?} {holds:?}");
        let hinted = Constraints {
            subset_hint: Some(0),
            ..own
        };
        let first = host.decide_vetoed(&OPEN, (1, None), hinted, [None; 2], NONE);
        assert_eq!(shed_for(&first), (ShedReason::NoEndpoints, SUBSET_HINT));
    }
    // The same request with no mask of its own is decided.
    let first = host.decide_vetoed(&OPEN, (1, None), free, [None; 2], NONE);
    assert!(first.pick().is_some(), "{first:?}");
}

/// What a scope was shown when it last opened: each place whose own stalled
/// bit was set, with its breaker bit.
type Seen = Arc<Mutex<Vec<(u32, bool)>>>;

/// A filter of a host's own build that opens a scope over its side's rows,
/// as a plugin's stage does. It keeps every row and notes the vetoes the
/// rows themselves carried.
#[derive(Debug)]
struct Shown(Seen);

impl Filter for Shown {
    fn name(&self) -> &'static str {
        "shown-rows"
    }

    fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
        true
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        Some(self)
    }
}

impl DecisionScope for Shown {
    fn open(&self, _: &RequestFacts<'_>, rows: &[Candidate], _: &PoolFacts<'_>) {
        let doubted = rows.iter().filter(|row| row.health.stalled);
        let bits = doubted.map(|row| (row.position, row.health.breaker_open));
        *self.0.lock() = bits.collect();
    }

    fn close(&self) {}
}

/// Two prefills and three decodes under the round-robin pair document
/// whose profile `side` shows its rows, with what its scope was last shown.
fn showing(side: &str) -> (Host, Seen) {
    let seen = Seen::default();
    let kept = Arc::clone(&seen);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter("shown-rows", move || Box::new(Shown(Arc::clone(&kept))))
        .expect("a free filter name");
    let presets = PolicySpec::pair_presets("round_robin", "round_robin");
    let mut document = presets.expect("the pair presets");
    document.generation = 3;
    let profile = document.profiles.get_mut(side).expect("the side's profile");
    profile.filter.push("shown-rows".into());
    let digest = document.digest().expect("a digest");
    let spec = AppliedSpec::validate_digested(&document, digest, &catalog, &[]);
    let spec = spec.expect("a valid document");
    (Host::under((spec, Arc::default()), 2, 3), seen)
}

/// A side whose profile shows its rows to a stage is handed its mask in
/// the rows: the stalled bit on each masked record, and the breaker bit
/// beside it under a hold. A plugin is shown a record's own fields, and
/// the engine declines such a profile over a mask beside its rows. Here
/// the decode profile shows its rows. The decodes' mask is written into
/// them: the scope is shown the bits, and no target names a masked decode,
/// at the pair's first decision and at the decode asked again. The
/// prefills' mask stays beside their rows, which no stage is shown.
#[test]
fn a_decode_side_that_shows_its_rows_is_handed_its_mask_in_them() {
    let (host, seen) = showing("decode");
    let free = Constraints::default();
    let shown = || std::mem::take(&mut *seen.lock());
    // The first decode is masked, and the third under a hold.
    let decodes: Vetoes<'_> = [[&[], &[]], [&[0, 2], &[2]]];
    let mut named_by = ([false; 2], [false; 3]);
    for seed in 0..8 {
        let first = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], decodes);
        note(&mut named_by, &first);
        assert_eq!(shown(), [(0, false), (2, true)], "seed {seed}");
    }
    assert_eq!(named_by, ([true; 2], [false, true, false]));
    // The prefills' mask is read beside their rows: no decode's record
    // carries a bit, and no target names the masked prefill.
    let prefills: Vetoes<'_> = [[&[0], &[]], [&[], &[]]];
    let mut named_by = ([false; 2], [false; 3]);
    for seed in 0..8 {
        let first = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], prefills);
        note(&mut named_by, &first);
        assert_eq!(shown(), [], "seed {seed}");
    }
    assert!(!named_by.0[0] && named_by.0[1]);
    // The decode asked again is handed the decodes' mask the same way.
    let held = prefill_alone(worker_id(host.prefills[1].as_ref()));
    let again = host.decide_vetoed(&OPEN, (1, Some(&held)), free, [None; 2], decodes);
    let mut named_by = ([false; 2], [false; 3]);
    note(&mut named_by, &again);
    assert_eq!(named_by, ([false, true], [false, true, false]));
    assert_eq!(shown(), [(0, false), (2, true)]);
}

/// The prefill side likewise: its profile shows its rows, and its mask is
/// written into them with a hold's breaker bit. No target names the masked
/// prefill, and the decodes' mask stays beside rows no stage is shown. The
/// decode asked again shows no prefill a mask: the held prefill ran.
#[test]
fn a_prefill_side_that_shows_its_rows_is_handed_its_mask_in_them() {
    let (host, seen) = showing("prefill");
    let free = Constraints::default();
    let shown = || std::mem::take(&mut *seen.lock());
    for (holds, bits) in [(&[][..], [(0, false)]), (&[0][..], [(0, true)])] {
        let prefills: Vetoes<'_> = [[&[0], holds], [&[], &[]]];
        let mut named_by = ([false; 2], [false; 3]);
        for seed in 0..8 {
            let first = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], prefills);
            note(&mut named_by, &first);
            assert_eq!(shown(), bits, "seed {seed}");
        }
        assert_eq!(named_by, ([false, true], [true; 3]));
    }
    let decodes: Vetoes<'_> = [[&[], &[]], [&[1], &[]]];
    let mut named_by = ([false; 2], [false; 3]);
    for seed in 0..8 {
        let first = host.decide_vetoed(&OPEN, (seed, None), free, [None; 2], decodes);
        note(&mut named_by, &first);
        assert_eq!(shown(), [], "seed {seed}");
    }
    assert_eq!(named_by, ([true; 2], [true, false, true]));
    // The decode asked again behind the first prefill, masked by then.
    let held = prefill_alone(worker_id(host.prefills[0].as_ref()));
    let masked: Vetoes<'_> = [[&[0], &[0]], [&[], &[]]];
    let again = host.decide_vetoed(&OPEN, (1, Some(&held)), free, [None; 2], masked);
    assert_eq!(host.picked(&again).0, 0);
    assert_eq!(shown(), []);
}
