//! The draw that claims its worker for a key: the look-ups it pays, and
//! the room it makes for the list's rows.

use super::*;

/// A document whose sessions rule leaves a key with no pin to a profile that
/// takes turns: over a list of the draw's floor or more, such a key's
/// decision is a draw that claims ([`PoolRecords::claim`]).
fn turns_under_a_rule_that_delegates() -> AppliedSpec {
    let mut document = PolicySpec::preset("manual").expect("a built-in preset");
    let turns = PolicySpec::preset("round_robin").expect("a built-in preset");
    document.profiles = turns.profiles;
    let binding = &mut document.bindings[0];
    binding.profile = Some("round_robin".into());
    let rule = binding.sessions.as_mut().expect("the preset's rule");
    rule.assignment = Assignment::Delegate;
    applied_of(document)
}

/// One decision of `spec`'s engine for the session key `key` over `pool`
/// under `loads`, with the list's own `word` lent to it.
fn keyed(
    spec: &AppliedSpec,
    clock: &DecisionClock,
    pool: &PoolRecords,
    workers: &[Arc<dyn Worker>],
    loads: &Arc<LoadSnapshot>,
    word: &AtomicU64,
    key: u64,
) -> Decision {
    let live = Live::at(spec.engine(), clock, Instant::now(), Some(loads), None).turning(word);
    let mut facts = RequestFacts {
        seed: key,
        session_key: crate::engine::Identity::hashed(key),
        ..single()
    };
    let cx = Constraints::default();
    decide(spec.engine(), pool, workers, &live, &mut facts, &cx)
}

/// Draws that claim pay their look-ups as plain draws do: a turn asks about
/// four workers, so the fourth new key pays the last of the 16 and converts
/// the list's reports. The keys behind it copy the list's rows.
#[test]
fn draws_that_claim_convert_a_lists_reports_once_they_looked_up_a_lists_length() {
    let (spec, clock) = (turns_under_a_rule_that_delegates(), DecisionClock::new());
    let workers = fleet(DRAW_FROM);
    let pool = PoolRecords::new(&workers, &spec, &clock);
    let (loads, _) = reports(&[0; DRAW_FROM], Duration::from_millis(5));
    let word = AtomicU64::new(0);
    for key in 1..=3 {
        let decision = keyed(&spec, &clock, &pool, &workers, &loads, &word, key);
        let pick = decision.pick().expect("a pick");
        assert!(pick.trace.drew_first(), "key {key}: no pass over the list");
        assert_eq!(
            pick.session_verdict,
            SessionVerdict::Claimed(picked(&decision).1)
        );
        assert_eq!(pick.targets.len(), 4);
        assert_eq!(pool.conversions(), 0, "key {key}");
    }
    let _ = keyed(&spec, &clock, &pool, &workers, &loads, &word, 4);
    assert_eq!(pool.conversions(), 1, "the draw that pays the last");
    assert!(held(&pool).is_some_and(|rows| rows.is_of(&loads)));
    for key in 5..=40 {
        let _ = keyed(&spec, &clock, &pool, &workers, &loads, &word, key);
    }
    assert_eq!(pool.conversions(), 1, "the rows are copied");
}

/// A draw that claims may be its thread's first decision: it makes room for
/// the list's rows in the thread's buffer, as a plain draw does.
#[test]
fn a_draw_that_claims_makes_room_for_its_lists_rows_on_a_new_thread() {
    // A thread of its own: no earlier decision made room in its buffers.
    let claimed = std::thread::spawn(|| {
        let (spec, clock) = (turns_under_a_rule_that_delegates(), DecisionClock::new());
        let workers = fleet(DRAW_FROM);
        let pool = PoolRecords::new(&workers, &spec, &clock);
        let (loads, _) = reports(&[0; DRAW_FROM], Duration::from_millis(5));
        let word = AtomicU64::new(0);
        let decision = keyed(&spec, &clock, &pool, &workers, &loads, &word, 41);
        let pick = decision.pick().expect("a pick");
        (
            pick.trace.drew_first(),
            pick.session_verdict,
            picked(&decision).1,
        )
    })
    .join()
    .expect("the draw made room for the list's rows");
    let (drew_first, verdict, first) = claimed;
    assert!(drew_first, "no pass over the list ran");
    assert_eq!(verdict, SessionVerdict::Claimed(first));
}
