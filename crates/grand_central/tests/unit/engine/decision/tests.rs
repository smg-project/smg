use super::*;
use crate::engine::view::Role;

/// A target made of a record names the record's position, id and rank
/// and no pair. A pair's target names its prefill so and its decode in
/// the pair, with no verdict on the decode's key: the value the engine
/// makes where it puts a decode's target under its prefill's.
#[test]
fn a_target_names_the_records_it_is_made_of() {
    let prefill = Candidate {
        dp_rank: 2,
        ..Candidate::ready(4, WorkerId(7), Role::Prefill)
    };
    let decode = Candidate {
        dp_rank: 1,
        ..Candidate::ready(6, WorkerId(9), Role::Decode)
    };
    let alone = Target {
        position: 4,
        id: WorkerId(7),
        dp_rank: 2,
        pair: None,
    };
    assert_eq!(Target::of(&prefill), alone);
    let side = Pair {
        decode_position: 6,
        decode_id: WorkerId(9),
        decode_dp_rank: 1,
        decode_verdict: DecodeVerdict::NONE,
        decode_placing: DecodePlacing::STAGES,
    };
    assert_eq!(Pair::of(&decode), side);
    let pair = Target {
        pair: Some(side),
        ..alone
    };
    assert_eq!(Target::paired(&prefill, &decode), pair);

    let mut targets = Targets::new();
    targets.push(Target::of(&decode));
    targets.pair_under(&Target::of(&prefill), SessionVerdict::None, Placing::Stages);
    assert_eq!(targets.first(), Some(&pair));
}

#[test]
fn every_shadow_kind_takes_a_place_of_its_own_under_the_count() {
    use ShadowKind::{Agreed, OtherOutcome, OtherTarget, Skipped};
    let places = [Agreed, OtherTarget, OtherOutcome, Skipped].map(ShadowKind::index);
    assert_eq!(places, [0, 1, 2, 3]);
    assert_eq!(places.len(), ShadowKind::COUNT);
    assert_eq!(ShadowKind::ALL.map(ShadowKind::index), places);
}

#[test]
fn shadow_kind_compares_the_outcome_and_first_target() {
    let first = Target {
        id: WorkerId(73),
        ..Target::default()
    };
    let second = Target {
        id: WorkerId(91),
        position: 1,
        ..Target::default()
    };
    let rank = Target {
        dp_rank: 5,
        ..first
    };
    let pick = |items: &[Target]| {
        let mut targets = Targets::new();
        for &target in items {
            targets.push(target);
        }
        Decision::Pick(Pick {
            targets,
            ..Pick::default()
        })
    };
    let shed = Decision::Shed(Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 0,
        wait_allowed: false,
        side: None,
        trace: Trace::default(),
    });
    let queue = |ticket| {
        Decision::Queue(Queue {
            ticket,
            max_wait_micros: 113,
            trace: Trace::default(),
        })
    };
    for (a, b, kind) in [
        (pick(&[first]), pick(&[first, second]), ShadowKind::Agreed),
        (
            pick(&[first, second]),
            pick(&[second, first]),
            ShadowKind::OtherTarget,
        ),
        (pick(&[first]), pick(&[rank]), ShadowKind::OtherTarget),
        (pick(&[first]), shed, ShadowKind::OtherOutcome),
        (pick(&[first]), queue(67), ShadowKind::OtherOutcome),
        (shed, queue(67), ShadowKind::OtherOutcome),
        (
            shed,
            Decision::Shed(Shed {
                reason: ShedReason::Saturated,
                emptied_by: "not-overloaded",
                retry_after_secs: 3,
                wait_allowed: false,
                side: None,
                trace: Trace::default(),
            }),
            ShadowKind::Agreed,
        ),
        (queue(67), queue(79), ShadowKind::Agreed),
    ] {
        assert_eq!(ShadowKind::of(&a, &b), kind);
        assert_eq!(ShadowKind::of(&b, &a), kind);
    }
    for (kind, label) in [
        (ShadowKind::Agreed, "agreed"),
        (ShadowKind::OtherTarget, "other-target"),
        (ShadowKind::OtherOutcome, "other-outcome"),
        (ShadowKind::Skipped, "skipped"),
    ] {
        assert_eq!(kind.as_str(), label);
    }
}

#[test]
fn a_default_trace_marks_no_profile_and_keeps_its_other_defaults() {
    let trace = Trace::default();
    assert_eq!(
        trace,
        Trace {
            profile: u16::MAX,
            survivors: Survivors::counted(0),
            top: [(WorkerId(0), 0.0); TRACE_TOP],
            top_len: 0,
            seed: 0,
            generation: 0,
            lifted: LiftedVeto::None,
        }
    );
    assert_eq!(trace.made_under(), None);
    let last = Trace {
        profile: u16::MAX - 1,
        ..trace
    };
    assert_eq!(last.made_under(), Some(u16::MAX - 1));
}

/// The mark of a lifted veto costs the trace no byte: it sits in one the
/// record had spare, and a trace no rule wrote reads none.
#[test]
fn the_mark_of_a_lifted_veto_costs_the_trace_no_byte() {
    assert_eq!(size_of::<LiftedVeto>(), 1);
    assert_eq!(size_of::<Trace>(), 72);
    assert_eq!(Trace::default().lifted, LiftedVeto::None);
}

/// The side of a pair costs the refusal no byte, and so the decision
/// none: it stands in a byte the refusal had spare. A side is put on a
/// refusal alone: a pick and a queue ticket pass as they are.
#[test]
fn the_side_of_a_pair_costs_the_refusal_no_byte() {
    assert_eq!(size_of::<Option<PairSide>>(), 1);
    assert_eq!(size_of::<Shed>(), 96);
    assert_eq!(size_of::<Decision>(), 256);
    let refused = Decision::Shed(Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 0,
        wait_allowed: false,
        side: None,
        trace: Trace::default(),
    });
    let named = refused.of_side(PairSide::Decode);
    assert!(matches!(named, Decision::Shed(shed) if shed.side == Some(PairSide::Decode)));
    let pick = Decision::Pick(Pick::default());
    assert_eq!(pick.of_side(PairSide::Prefill), pick);
}

/// The overlap costs the pick no byte: it shares the eight bytes the
/// targets' count stood alone in, a pick nothing is stated of reads
/// none, and a list led anew or cleared says nothing of a first target.
#[test]
fn the_overlap_costs_the_pick_no_byte() {
    assert_eq!(size_of::<Overlap>(), 6);
    assert_eq!(size_of::<Targets>(), MAX_TARGETS * size_of::<Target>() + 8);
    assert_eq!(
        size_of::<Pick>(),
        size_of::<Targets>() + size_of::<Trace>() + size_of::<SessionVerdict>()
    );
    assert_eq!(Pick::default().overlap(), Overlap::NONE);
    assert_eq!(Overlap::default(), Overlap::NONE);
    let held = Overlap {
        blocks: 3,
        of: 4,
        source: OverlapSource::Index,
        margin_held: false,
    };
    let mut targets = Targets::new();
    for position in 0..2u32 {
        targets.push(Target {
            position,
            id: WorkerId(u64::from(position) + 1),
            dp_rank: 0,
            pair: None,
        });
    }
    targets.state(held);
    let mut pick = Pick {
        targets,
        ..Pick::default()
    };
    assert_eq!(pick.overlap(), held);
    let second = *pick.targets.as_slice().last().expect("two targets");
    pick.targets.lead(second);
    assert_eq!(pick.overlap(), Overlap::NONE);
    pick.targets.state(held);
    pick.targets.clear();
    assert_eq!(pick.overlap(), Overlap::NONE);
}

/// An entry of a decision's rows is stated by the larger of its two
/// readings, the index's on a tie, with no word of a margin. An entry
/// that holds nothing is a miss of the request's blocks, not "nothing
/// stated". A request past sixteen bits halves both counts together.
#[test]
fn an_entry_is_stated_by_its_larger_reading_of_the_requests_blocks() {
    let entry = |index: u32, predicted: u32| AffinityFacts {
        overlap_blocks: index,
        predicted_blocks: predicted,
        ..AffinityFacts::default()
    };
    let stated = |blocks: u16, of: u16, source: OverlapSource| Overlap {
        blocks,
        of,
        source,
        margin_held: false,
    };
    let cases = [
        (entry(3, 0), 8, stated(3, 8, OverlapSource::Index)),
        (entry(0, 5), 8, stated(5, 8, OverlapSource::Predictor)),
        (entry(2, 5), 8, stated(5, 8, OverlapSource::Predictor)),
        (entry(5, 2), 8, stated(5, 8, OverlapSource::Index)),
        // A tie reads as the index's.
        (entry(4, 4), 8, stated(4, 8, OverlapSource::Index)),
        // A miss: zero of the request's blocks, which is not `NONE`.
        (entry(0, 0), 8, stated(0, 8, OverlapSource::None)),
        // 65,536 blocks are one bit past sixteen: one halving.
        (
            entry(40_000, 0),
            65_536,
            stated(20_000, 32_768, OverlapSource::Index),
        ),
        // 262,143 blocks are two bits past: two halvings, rounded down.
        (
            entry(0, 262_143),
            262_143,
            stated(65_535, 65_535, OverlapSource::Predictor),
        ),
    ];
    for (held, request_blocks, said) in cases {
        assert_eq!(
            Overlap::held(held, request_blocks),
            said,
            "{held:?} of {request_blocks}"
        );
    }
    assert_ne!(Overlap::held(entry(0, 0), 8), Overlap::NONE);
    // A lent tree's match is stated in the tree's own unit, and halved
    // as blocks are: 200,000 chars are two bits past sixteen.
    let in_tree = |matched: u32, of: u32| Overlap::in_tree(matched, of, false);
    assert_eq!(in_tree(40, 100), stated(40, 100, OverlapSource::Tree));
    let halved = stated(17_500, 50_000, OverlapSource::Tree);
    assert_eq!(in_tree(70_000, 200_000), halved);
    assert_eq!(in_tree(0, 100), stated(0, 100, OverlapSource::None));
}

/// The word of who placed a pick costs the pick no byte: it sits in the
/// one a list's count and overlap left spare between them, with the
/// rule's own word of a pick it placed. A pick no rule placed reads the
/// stages, and is not kept from them. Both words are the decision's: a
/// list led anew or emptied keeps them, and a list turned into a
/// pair's hands them to its pairs (the next case).
#[test]
fn the_word_of_who_placed_a_pick_costs_the_pick_no_byte() {
    assert_eq!(size_of::<Placing>(), 1);
    assert_eq!(size_of::<Targets>(), MAX_TARGETS * size_of::<Target>() + 8);
    assert_eq!(size_of::<Pick>(), 256);
    assert_eq!(Pick::default().placed_by(), PlacedBy::Stages);
    let target = |position: u32| Target {
        position,
        id: WorkerId(u64::from(position) + 1),
        dp_rank: 0,
        pair: None,
    };
    let mut pick = Pick::default();
    pick.targets.push(target(0));
    pick.targets.push(target(1));
    let by_stages = pick;
    assert!(!by_stages.untold());
    for kept in [RulePlaced::Told, RulePlaced::Untold] {
        let untold = kept == RulePlaced::Untold;
        let mut pick = by_stages;
        pick.targets.by_rule(kept);
        assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
        assert_ne!(pick, by_stages);
        pick.targets.lead(target(1));
        assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
        pick.targets.clear();
        assert_eq!((pick.placed_by(), pick.untold()), (PlacedBy::Rule, untold));
    }
    // The two words of a rule's pick are two picks.
    let (mut told, mut kept) = (by_stages, by_stages);
    told.targets.by_rule(RulePlaced::Told);
    kept.targets.by_rule(RulePlaced::Untold);
    assert_ne!(told, kept);
}

/// Each side of a pair states its own word of who placed it, and the
/// word costs the pair no byte: the decode side's sits in the last
/// byte the pair had spare, beside that side's verdict, and the
/// prefill side's in the list's own. A pair, a target and a pick are
/// the size they were. A list turned into a pair's hands its own word,
/// the decode side's, to each of its pairs, and states the prefill
/// side's. Each of the three words reads back on each side, whatever
/// the other side states: a sessions rule may place one side of a pair
/// alone and leave the other to its stages. A pair no decision spoke
/// of reads the stages. A decode asked again keeps the word a held
/// list states of the prefill side. Of the decode side it keeps the
/// word of the pair it is held by where it names that pair's decode
/// first, by its worker and its rank, and its own where it names
/// another. A gateway's sticky map places each role of a pair under
/// that role's own key
/// (`model_gateway/src/policies/registry.rs:762-769`), so one role's
/// request may follow its pin where the other's is its policy's. It
/// selects a request's decode once
/// (`model_gateway/src/routers/common/placement.rs:3088`): the policy
/// that placed it hears of it, however often the decode is asked for.
#[test]
fn each_side_of_a_pair_states_its_own_word_of_who_placed_it() {
    assert_eq!(size_of::<DecodePlacing>(), 1);
    assert_eq!((size_of::<Pair>(), size_of::<Target>()), (16, 40));
    assert_eq!(size_of::<Pick>(), 256);
    assert_eq!(Pair::default().decode_placing, DecodePlacing::STAGES);
    let target = |position: u32| Target {
        position,
        id: WorkerId(u64::from(position) + 1),
        dp_rank: 0,
        pair: None,
    };
    // A side's word as its list states it and as a pair does, with
    // who placed the side and whether it is kept from the profile.
    let words = [
        (
            Placing::Stages,
            DecodePlacing::STAGES,
            (PlacedBy::Stages, false),
        ),
        (Placing::Rule, DecodePlacing::RULE, (PlacedBy::Rule, false)),
        (
            Placing::RuleUntold,
            DecodePlacing::RULE_UNTOLD,
            (PlacedBy::Rule, true),
        ),
    ];
    for (prefill, _, prefill_word) in words {
        for (decode, in_pair, decode_word) in words {
            // The decode side's own list, as its decision made it.
            let mut pick = Pick::default();
            pick.targets.push(target(0));
            pick.targets.push(target(1));
            match decode {
                Placing::Stages => {}
                Placing::Rule => pick.targets.by_rule(RulePlaced::Told),
                Placing::RuleUntold => pick.targets.by_rule(RulePlaced::Untold),
            }
            pick.targets
                .pair_under(&target(2), SessionVerdict::None, prefill);
            let pairs: Vec<Pair> = pick.targets.iter().filter_map(|t| t.pair).collect();
            assert_eq!(pairs.len(), 2);
            assert!(pairs.iter().all(|pair| pair.decode_placing == in_pair));
            let stated = (in_pair.placed_by(), in_pair.untold());
            assert_eq!(stated, decode_word, "the decode side's");
            assert_eq!(pick.decode_untold(), decode_word.1);
            let own = (pick.placed_by(), pick.untold());
            assert_eq!(own, prefill_word, "the prefill side's");
            // A decode asked again: its own word, and the held list's
            // of the prefill side.
            let mut again = Pick::default();
            again.targets.push(target(3));
            again.targets.by_rule(RulePlaced::Untold);
            again
                .targets
                .pair_under(&target(2), SessionVerdict::None, Placing::Stages);
            assert_eq!(again.placed_by(), PlacedBy::Stages);
            again.targets.placed_as(&pick.targets);
            assert_eq!((again.placed_by(), again.untold()), prefill_word);
            assert!(again.decode_untold(), "its own side's word stands");
            // Held by a pair of another decode, or of another rank of
            // its decode, the call placed the side anew.
            let own = again;
            for held in [pairs[0], pairs[1]] {
                again.targets.decode_placed_as(&held);
                let other_rank = Pair {
                    decode_id: WorkerId(4),
                    decode_dp_rank: 1,
                    ..held
                };
                again.targets.decode_placed_as(&other_rank);
                assert_eq!(again, own);
            }
            // Held by a pair of the decode it names first, it keeps
            // that pair's word on every pair, and the prefill side's.
            let mut found = Pick::default();
            found.targets.push(target(3));
            found.targets.push(target(4));
            found.targets.by_rule(RulePlaced::Untold);
            found
                .targets
                .pair_under(&target(2), SessionVerdict::None, prefill);
            let held = Pair {
                decode_id: WorkerId(4),
                ..pairs[1]
            };
            found.targets.decode_placed_as(&held);
            let kept: Vec<Pair> = found.targets.iter().filter_map(|t| t.pair).collect();
            assert!(kept.iter().all(|pair| pair.decode_placing == in_pair));
            assert_eq!(found.decode_untold(), decode_word.1);
            assert_eq!((found.placed_by(), found.untold()), prefill_word);
        }
    }
    // A pick that names no pair has one side, and that side's word.
    let mut single = Pick::default();
    single.targets.push(target(0));
    assert!(!single.decode_untold());
    single.targets.by_rule(RulePlaced::Untold);
    assert!(single.untold() && single.decode_untold());
    // Every other value of the byte reads as the stages'.
    let other = DecodePlacing(9);
    assert_eq!(
        (other.placed_by(), other.untold()),
        (PlacedBy::Stages, false)
    );
}

/// The decode side's verdict costs a pair no byte: it sits in one the
/// pair had spare. A target and a pick are the size they were, and a
/// pair no decision spoke of reads none.
#[test]
fn the_decode_sides_verdict_costs_the_pair_no_byte() {
    assert_eq!(size_of::<DecodeVerdict>(), 1);
    assert_eq!((size_of::<Pair>(), size_of::<Target>()), (16, 40));
    assert_eq!(size_of::<Pick>(), 256);
    assert_eq!(Pair::default().decode_verdict, DecodeVerdict::NONE);
}

#[test]
fn targets_keep_order_and_refuse_a_fifth() {
    let mut targets = Targets::new();
    for position in 0..6u32 {
        targets.push(Target {
            position,
            id: WorkerId(u64::from(position)),
            dp_rank: 0,
            pair: None,
        });
    }
    assert_eq!(targets.len(), MAX_TARGETS);
    let positions: Vec<u32> = targets.iter().map(|t| t.position).collect();
    assert_eq!(positions, vec![0, 1, 2, 3]);
    assert_eq!(targets.first().map(|t| t.position), Some(0));
    targets.clear();
    assert!(targets.is_empty());
}

#[test]
fn every_outcome_carries_its_trace_and_only_a_pick_has_targets() {
    let trace = Trace {
        seed: 9,
        ..Trace::default()
    };
    let shed = Decision::Shed(Shed {
        reason: ShedReason::NoEndpoints,
        emptied_by: "healthy",
        retry_after_secs: 1,
        wait_allowed: false,
        side: None,
        trace,
    });
    let queue = Decision::Queue(Queue {
        ticket: 1,
        max_wait_micros: 50_000,
        trace,
    });
    assert!(shed.pick().is_none());
    assert!(queue.pick().is_none());
    assert_eq!(shed.trace().seed, 9);
    assert_eq!(queue.trace().seed, 9);
    let pick = Decision::Pick(Pick {
        targets: Targets::new(),
        trace,
        session_verdict: SessionVerdict::None,
    });
    assert!(pick.pick().is_some());
}

#[test]
fn a_list_cleared_and_filled_again_equals_a_new_one() {
    let target = |position: u32| Target {
        position,
        id: WorkerId(u64::from(position) + 1),
        dp_rank: 0,
        pair: None,
    };
    let mut used = Targets::new();
    used.push(target(0));
    used.push(target(1));
    used.clear();
    used.push(target(2));
    let mut new = Targets::new();
    new.push(target(2));
    assert_eq!(used, new);
    // A count and what a list states of its first target are parts
    // of it still.
    let mut longer = new;
    longer.push(target(1));
    assert_ne!(used, longer);
    let mut stated = new;
    stated.state(Overlap {
        blocks: 1,
        of: 2,
        source: OverlapSource::Index,
        margin_held: false,
    });
    assert_ne!(used, stated);
    // A list prints what its equality reads.
    assert_eq!(format!("{used:?}"), format!("{new:?}"));
    assert_ne!(format!("{used:?}"), format!("{longer:?}"));
}

#[test]
fn lead_puts_a_target_first_and_drops_its_copy_or_the_last_one() {
    let target = |id: u64| Target {
        position: id as u32,
        id: WorkerId(id),
        dp_rank: 0,
        pair: None,
    };
    let mut targets = Targets::new();
    for id in 1..=3 {
        targets.push(target(id));
    }
    targets.lead(target(3));
    let ids = |t: &Targets| t.as_slice().iter().map(|t| t.id.0).collect::<Vec<_>>();
    assert_eq!(ids(&targets), [3, 1, 2], "a copy among them leaves");
    let mut full = Targets::new();
    for id in 1..=MAX_TARGETS as u64 {
        full.push(target(id));
    }
    full.lead(target(9));
    assert_eq!(full.len(), MAX_TARGETS);
    assert_eq!(ids(&full)[0], 9, "the newcomer leads");
    assert_eq!(
        *ids(&full).last().expect("full"),
        MAX_TARGETS as u64 - 1,
        "the last one left"
    );
}

#[test]
fn lead_removes_only_the_equal_target_not_another_rank() {
    let x_rank_0 = Target {
        position: 0,
        id: WorkerId(1),
        dp_rank: 0,
        pair: None,
    };
    let w_rank_1 = Target {
        position: 1,
        id: WorkerId(2),
        dp_rank: 1,
        pair: None,
    };
    let w_rank_0 = Target {
        position: 2,
        dp_rank: 0,
        ..w_rank_1
    };
    let mut targets = Targets::new();
    for target in [x_rank_0, w_rank_1, w_rank_0] {
        targets.push(target);
    }

    targets.lead(w_rank_0);

    assert_eq!(targets.as_slice(), [w_rank_0, x_rank_0, w_rank_1]);
}

/// A decode side's targets turn into a pair's where they stand. Each
/// names the prefill with the decode it held, in the order they had,
/// and the first pair alone states the decode side's verdict. The list
/// equals the one a new list takes push by push, and nothing is stated
/// of its first target.
#[test]
fn a_decodes_targets_turn_into_a_pairs_where_they_stand() {
    let prefill = Target {
        position: 7,
        id: WorkerId(70),
        dp_rank: 3,
        pair: None,
    };
    let decode = |position: u32| Target {
        position,
        id: WorkerId(u64::from(position) + 1),
        dp_rank: position as u16 + 5,
        pair: None,
    };
    let mut turned = Targets::new();
    let mut pushed = Targets::new();
    for position in 0..2u32 {
        turned.push(decode(position));
        pushed.push(Target {
            pair: Some(Pair {
                decode_position: position,
                decode_id: decode(position).id,
                decode_dp_rank: decode(position).dp_rank,
                decode_verdict: [DecodeVerdict::CLAIMED, DecodeVerdict::NONE][position as usize],
                decode_placing: DecodePlacing::STAGES,
            }),
            ..prefill
        });
    }
    turned.state(Overlap {
        blocks: 3,
        of: 4,
        source: OverlapSource::Index,
        margin_held: false,
    });
    turned.pair_under(
        &prefill,
        SessionVerdict::Claimed(decode(0).id),
        Placing::Stages,
    );
    assert_eq!(turned, pushed);
    // The first pair reads its side's whole verdict back with its decode.
    let first = turned.as_slice()[0].pair.expect("a pair");
    assert_eq!(
        first.decode_verdict.on(first.decode_id),
        SessionVerdict::Claimed(decode(0).id)
    );
    let pick = Pick {
        targets: turned,
        ..Pick::default()
    };
    assert_eq!(pick.overlap(), Overlap::NONE);
    // Each verdict of the decode side is read back whole from the
    // first pair, with the decode it names.
    for verdict in [
        SessionVerdict::None,
        SessionVerdict::Pinned,
        SessionVerdict::Respill,
        SessionVerdict::Claimed(decode(0).id),
    ] {
        let mut list = Targets::new();
        list.push(decode(0));
        list.pair_under(&prefill, verdict, Placing::Stages);
        let pair = list.as_slice()[0].pair.expect("a pair");
        assert_eq!(pair.decode_verdict.on(pair.decode_id), verdict);
    }
    // An empty list stays empty.
    let mut none = Targets::new();
    none.pair_under(&prefill, SessionVerdict::None, Placing::Stages);
    assert_eq!(none, Targets::new());
}

/// The count and its absence are one word: a trace is the size it was
/// when it held a bare count, and no count is never a zero.
#[test]
fn survivors_hold_a_count_or_none_in_one_word() {
    assert_eq!(size_of::<Survivors>(), size_of::<u32>());
    assert_eq!(Survivors::default().get(), Some(0), "the default counted");
    assert_eq!(Survivors::counted(5).get(), Some(5));
    assert_eq!(Survivors::NOT_COUNTED.get(), None);
    assert_eq!(
        Survivors::counted(u32::MAX).get(),
        Some(u32::MAX - 1),
        "a count is never the mark of none"
    );
    assert_ne!(Survivors::NOT_COUNTED, Survivors::counted(0));
    assert_eq!(format!("{:?}", Survivors::counted(5)), "Some(5)");
    assert_eq!(format!("{:?}", Survivors::NOT_COUNTED), "None");
    let counted = Trace::default();
    assert!(!counted.drew_first(), "a trace that counted did not draw");
    let drew = Trace {
        survivors: Survivors::NOT_COUNTED,
        ..counted
    };
    assert!(drew.drew_first());
}
