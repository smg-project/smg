use std::{
    ops::Deref,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use parking_lot::Mutex;
use serde_json::json;

use crate::{
    document::{AppliedSpec, PolicySpec, SpecError, API_VERSION},
    engine::{
        decision, decision_engine::DecisionEngine, framework, Candidate, Constraints, Engine,
        Filter, LoadRow, Pair, PoolFacts, RequestFacts, Scorer, Scratch, Target, TieBand, WorkerId,
    },
    stages::{pickers, scorers::take_lane, Lifecycle, StageRegistry},
    tracking::{
        decision_guard::DecisionGuard, CreditedDispatches, DispatchCounts, DispatchCredit, Lease,
        Membership, Side,
    },
};

const POOL: &str = "work";
const START: u64 = 91_300;
const DELAY: u32 = 47;
const TOKENS: u32 = 317;

fn one(tokens: u32) -> CreditedDispatches {
    CreditedDispatches {
        requests: 1,
        tokens: u64::from(tokens),
        ..CreditedDispatches::default()
    }
}

#[derive(Debug)]
struct AvoidSent(Arc<DispatchCredit>);

impl Lifecycle for AvoidSent {}

impl Scorer for AvoidSent {
    fn name(&self) -> &'static str {
        "avoid-sent"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let sent = self.0.pass(facts);
        for (&row, score) in kept.iter().zip(out) {
            *score = if sent.of(&candidates[row as usize]).requests == 0 {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }
}

type Readers = Arc<Mutex<Vec<Arc<DispatchCredit>>>>;

fn catalog() -> (StageRegistry, Readers) {
    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = StageRegistry::builtin();
    catalog
        .scorer_with("avoid-sent", move |build| {
            let sent = build.dispatch_credit();
            kept.lock().push(Arc::clone(&sent));
            Ok(Box::new(AvoidSent(sent)))
        })
        .expect("a free name");
    (catalog, readers)
}

fn document(limit: Option<u32>) -> PolicySpec {
    serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": {
            "work": { "sent_ttl_ms": limit, "score": ["avoid-sent"], "pick": "best" }
        }
    }))
    .expect("a document")
}

fn built(limit: Option<u32>) -> (Engine, Arc<DispatchCredit>) {
    counting(document(limit))
}

/// The engine of `document` over the catalog with the registered reader,
/// and the count that reader was handed.
fn counting(document: PolicySpec) -> (Engine, Arc<DispatchCredit>) {
    let (catalog, readers) = catalog();
    let engine = document
        .validate_in(&catalog, &[])
        .expect("the profile validates");
    let sent = Arc::clone(&readers.lock()[0]);
    (engine, sent)
}

/// The report delay most cases set: five milliseconds, as a profile
/// writes it and as the count's clock reads it.
const REPORT_DELAY_MS: u32 = 5;
const REPORT_DELAY: u64 = 5_000;

/// [`document`] with a report delay of `delay` milliseconds on its profile.
fn delayed(limit: Option<u32>, delay: u32) -> PolicySpec {
    let mut document = document(limit);
    document
        .profiles
        .get_mut(POOL)
        .expect("the profile")
        .sent_report_delay_ms = Some(delay);
    document
}

fn row(id: u64, dp_rank: u16) -> Candidate {
    Candidate {
        id: WorkerId(id),
        dp_rank,
        ..Candidate::default()
    }
}

fn target(row: &Candidate) -> Target {
    Target {
        id: row.id,
        dp_rank: row.dp_rank,
        ..Target::default()
    }
}

fn send(engine: &Engine, row: &Candidate, issued_micros: u64, tokens: u32) {
    engine.on_dispatched(
        &Lease {
            issued_micros,
            tokens,
            ..Lease::default()
        },
        &target(row),
        Side::Single,
        DELAY,
    );
}

fn clock(now_micros: u64) -> RequestFacts<'static> {
    RequestFacts {
        now_micros,
        ..RequestFacts::default()
    }
}

#[test]
fn a_registered_scorer_reads_the_requests_and_tokens_sent_to_each_row() {
    let (engine, sent) = built(None);
    let a = row(101, 3);
    let b = row(103, a.dp_rank);
    send(&engine, &a, START, TOKENS);

    let facts = clock(START + 700);
    let reads = sent.pass(&facts);
    assert_eq!(reads.of(&a), one(TOKENS));
    assert_eq!(reads.of(&b), CreditedDispatches::default());
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
}

#[test]
fn a_pass_keeps_one_tally_map_and_a_later_pass_reads_new_rows() {
    let (engine, sent) = built(None);
    let a = row(107, 3);
    let facts = clock(START + 700);
    let pass = sent.pass(&facts);
    send(&engine, &a, START, TOKENS);
    assert_eq!(pass.of(&a), CreditedDispatches::default());
    assert_eq!(sent.pass(&facts).of(&a), one(TOKENS));
}

#[test]
fn a_send_with_no_stated_size_counts_only_as_a_request() {
    let (engine, sent) = built(None);
    let a = row(109, 5);
    send(&engine, &a, START, 0);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            unsized_requests: 1,
            ..one(0)
        }
    );
}

/// A send of text alone states no token count, and the count keeps its
/// bytes over four as an estimate. The documents' terms read the estimate
/// as the send's tokens, and the whole read states it apart from the
/// stated tokens, beside the count of such sends. A send that states a
/// token count is counted by it. A report that reflects a send takes its
/// estimate with it.
#[test]
fn a_send_of_text_alone_counts_by_its_bytes_over_four_in_the_documents_terms() {
    let (engine, sent) = built(None);
    let mut a = row(113, 5);
    let text = |issued_micros: u64, bytes: u32| {
        let lease = Lease {
            issued_micros,
            bytes,
            ..Lease::default()
        };
        engine.on_dispatched(&lease, &target(&a), Side::Single, DELAY);
    };
    text(START, 404);
    text(START + 9, 39);
    send(&engine, &a, START + 20, TOKENS);
    send(&engine, &a, START + 30, 0);
    // 404 bytes are 101 tokens and 39 bytes are 9.
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 4,
            tokens: u64::from(TOKENS),
            unsized_requests: 3,
            estimated_tokens: 110,
        }
    );
    let counts = sent.pass(&clock(0)).counts(&a);
    assert_eq!(
        counts,
        DispatchCounts {
            requests: 4,
            tokens: u64::from(TOKENS) + 110,
        }
    );
    // A stage that reads the sends one by one is handed each with its
    // stated tokens: none for a send of text.
    assert_eq!(
        instants(&sent, &clock(0), &a),
        [
            (sent_at(START), 0),
            (sent_at(START + 9), 0),
            (sent_at(START + 20), TOKENS),
            (sent_at(START + 30), 0),
        ]
    );
    // A report that reflects the first send, the text of 404 bytes.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 3,
            tokens: u64::from(TOKENS),
            unsized_requests: 2,
            estimated_tokens: 9,
        }
    );
    assert_eq!(
        sent.pass(&clock(0)).counts(&a).tokens,
        u64::from(TOKENS) + 9
    );
}

#[test]
fn a_send_of_no_stated_size_leaves_the_count_with_its_release() {
    let (engine, sent) = built(None);
    let mut a = row(179, 5);
    send(&engine, &a, START, 0);
    send(&engine, &a, START + 9, 0);
    send(&engine, &a, START + 20, TOKENS);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 3,
            tokens: u64::from(TOKENS),
            unsized_requests: 2,
            estimated_tokens: 0,
        }
    );
    // A report that reflects the first two sends, each of no stated size.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + 9 + u64::from(DELAY);
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
}

#[test]
fn a_retired_send_of_no_stated_size_leaves_that_count_too() {
    let (engine, sent) = built(None);
    let a = row(181, 5);
    for _ in 0..300 {
        send(&engine, &a, START, 0);
    }
    for _ in 300..8201 {
        send(&engine, &a, START + 1, 13);
    }
    // The nine oldest sends went, each of no stated size.
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 8192,
            tokens: 7901 * 13,
            unsized_requests: 291,
            estimated_tokens: 0,
        }
    );
}

#[test]
fn a_busy_rows_tally_lets_its_oldest_go_past_the_bound() {
    let (engine, sent) = built(None);
    let a = row(111, 3);
    for _ in 0..8037 {
        send(&engine, &a, START, 13);
    }
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 8037,
            tokens: 8037 * 13,
            ..CreditedDispatches::default()
        }
    );
    for _ in 8037..8201 {
        send(&engine, &a, START + 1, 19);
    }
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 8192,
            tokens: 8028 * 13 + 164 * 19,
            ..CreditedDispatches::default()
        }
    );
}

#[test]
fn the_row_report_ends_sends_at_its_sample_and_earlier() {
    let (engine, sent) = built(None);
    let mut a = row(113, 3);
    let at = START + u64::from(DELAY);
    send(&engine, &a, START, TOKENS);
    a.load_sampled_at_micros = at;
    let facts = clock(at + 400);
    assert_eq!(sent.read(&facts, &a), one(TOKENS));

    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = at - 1;
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
    send(&engine, &a, START + 7, 293);
    a.load_sampled_at_micros = at;
    assert_eq!(sent.read(&facts, &a), one(293));
    a.load_sampled_at_micros = at + 8;
    assert_eq!(sent.read(&facts, &a), CreditedDispatches::default());
}

#[test]
fn the_limit_ends_sends_at_its_cutoff_and_earlier() {
    let (engine, sent) = built(Some(17));
    let mut a = row(127, 3);
    let at = START + u64::from(DELAY);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = at - 1;
    send(&engine, &a, START - 2, 281);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 4, 331);

    assert_eq!(sent.read(&clock(at + 17_000), &a), one(331));
    a.load_sampled_at_micros = at + 4;
    assert_eq!(
        sent.read(&clock(at + 17_001), &a),
        CreditedDispatches::default()
    );
}

#[test]
fn a_report_sampled_sooner_than_the_delay_behind_a_send_leaves_it_counted() {
    let (engine, sent) = counting(delayed(None, REPORT_DELAY_MS));
    // One row is read whole and the other by its two numbers.
    let mut rows = [row(211, 3), row(223, 3)];
    for row in &mut rows {
        send(&engine, row, START, TOKENS);
        row.load = Some(LoadRow::default());
    }
    let sent_at = START + u64::from(DELAY);
    let facts = clock(sent_at + 9_000);
    // Three milliseconds behind the send, a microsecond short of the
    // delay, and the delay itself.
    for (behind, counted) in [(3_000, 1), (REPORT_DELAY - 1, 1), (REPORT_DELAY, 0)] {
        for row in &mut rows {
            row.load_sampled_at_micros = sent_at + behind;
        }
        assert_eq!(sent.read(&facts, &rows[0]).requests, counted, "{behind}");
        let numbers = sent.pass(&facts).counts(&rows[1]);
        assert_eq!(numbers.requests, counted, "{behind}");
    }
}

#[test]
fn a_delayed_report_ends_the_sends_its_delay_reaches_and_keeps_the_newer() {
    let (engine, sent) = counting(delayed(None, REPORT_DELAY_MS));
    let mut a = row(227, 3);
    send(&engine, &a, START, 281);
    send(&engine, &a, START + 4_000, TOKENS);
    // Six milliseconds behind the first send and two behind the second.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY) + 6_000;
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
}

#[test]
fn a_sample_earlier_on_the_clock_than_the_delay_ends_no_send() {
    let (engine, sent) = counting(delayed(None, REPORT_DELAY_MS));
    let mut a = row(229, 3);
    send(&engine, &a, 0, TOKENS);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = REPORT_DELAY - 2_000;
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
}

#[test]
fn the_limit_ends_a_send_whose_report_is_too_soon_for_the_delay() {
    let (engine, sent) = counting(delayed(Some(2), REPORT_DELAY_MS));
    let mut a = row(233, 3);
    send(&engine, &a, START, TOKENS);
    let sent_at = START + u64::from(DELAY);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = sent_at + 1_000;
    assert_eq!(sent.read(&clock(sent_at + 1_999), &a), one(TOKENS));
    assert_eq!(
        sent.read(&clock(sent_at + 2_000), &a),
        CreditedDispatches::default()
    );
}

#[test]
fn a_read_under_the_shorter_of_two_documents_delays_ends_a_send_for_both() {
    let (standing, long) = counting(delayed(None, 9));
    let mut a = row(239, 3);
    send(&standing, &a, START, TOKENS);
    let (next, short) = counting(delayed(None, REPORT_DELAY_MS));
    next.succeed(&standing);
    // Six milliseconds behind the send: under the delay of nine and past
    // the delay of five.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY) + 6_000;
    let facts = clock(0);
    assert_eq!(long.read(&facts, &a), one(TOKENS));
    assert_eq!(short.read(&facts, &a), CreditedDispatches::default());
    assert_eq!(long.read(&facts, &a), CreditedDispatches::default());
}

#[test]
fn a_delay_longer_than_a_u32_of_microseconds_holds_to_its_end() {
    // 705 microseconds past the most a `u32` of microseconds holds.
    const LONG_MS: u32 = 4_294_968;
    let (engine, sent) = counting(delayed(None, LONG_MS));
    let mut a = row(241, 3);
    send(&engine, &a, START, TOKENS);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY) + u64::from(LONG_MS) * 1000 - 1;
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
    a.load_sampled_at_micros += 1;
    assert_eq!(sent.read(&clock(0), &a), CreditedDispatches::default());
}

/// What [`DispatchCreditPass::sends`](super::DispatchCreditPass::sends)
/// hands of `row` under `facts`: each send's instant and its stated tokens,
/// in the order handed.
fn instants(sent: &DispatchCredit, facts: &RequestFacts<'_>, row: &Candidate) -> Vec<(u64, u32)> {
    let mut handed = Vec::new();
    let each = |at_micros, tokens| handed.push((at_micros, tokens));
    sent.pass(facts).sends(row, each);
    handed
}

/// The instant the count keeps of a send whose lease was issued at
/// `issued_micros`: [`send`] reports it [`DELAY`] microseconds later.
fn sent_at(issued_micros: u64) -> u64 {
    issued_micros + u64::from(DELAY)
}

/// The sends a stage reads one by one are the sends the two numbers count,
/// to the send, across a release: each with its own instant and its stated
/// tokens, none for a send of no stated size. A report ends the sends it
/// reaches for both reads, and the read of the instants lets them go
/// itself. Each rank of a worker hands its own sends, and a row the count
/// holds nothing of hands nothing.
#[test]
fn the_instants_read_are_the_sends_the_two_numbers_count_across_a_release() {
    let (engine, sent) = built(None);
    let mut a = row(251, 3);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 1, 0);
    send(&engine, &a, START + 20, 293);
    // Another rank of the worker holds a send of its own.
    let other = row(a.id.0, 7);
    send(&engine, &other, START + 40, 17);
    let facts = clock(0);
    let all = [
        (sent_at(START), TOKENS),
        (sent_at(START + 1), 0),
        (sent_at(START + 20), 293),
    ];
    assert_eq!(instants(&sent, &facts, &a), all);
    let others = [(sent_at(START + 40), 17)];
    assert_eq!(instants(&sent, &facts, &other), others);
    let counts = DispatchCounts {
        requests: 3,
        tokens: u64::from(TOKENS) + 293,
    };
    assert_eq!(sent.pass(&facts).counts(&a), counts);
    // A sample instant on a row that holds no report ends nothing.
    a.load_sampled_at_micros = sent_at(START + 1);
    assert_eq!(instants(&sent, &facts, &a), all);
    // A report that reflects the first two sends: the read of the
    // instants lets them go, with no other read ahead of it.
    a.load = Some(LoadRow::default());
    assert_eq!(instants(&sent, &facts, &a), all[2..]);
    assert_eq!(sent.credits.held_at(a.id, a.dp_rank), 1);
    let counts = DispatchCounts {
        requests: 1,
        tokens: 293,
    };
    assert_eq!(sent.pass(&facts).counts(&a), counts);
    assert!(instants(&sent, &facts, &row(257, 3)).is_empty());
}

/// The read of the instants follows the pass's rule, as the totals do.
/// Under a report delay a report sampled sooner behind a send leaves the
/// send handed, and one sampled the delay behind it ends it. The profile's
/// time limit ends a send that no report reaches.
#[test]
fn the_instants_follow_the_profiles_report_delay_and_its_time_limit() {
    let (engine, sent) = counting(delayed(None, REPORT_DELAY_MS));
    let mut a = row(263, 3);
    send(&engine, &a, START, TOKENS);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = sent_at(START) + REPORT_DELAY - 1;
    assert_eq!(instants(&sent, &clock(0), &a), [(sent_at(START), TOKENS)]);
    a.load_sampled_at_micros += 1;
    assert!(instants(&sent, &clock(0), &a).is_empty());

    let (engine, sent) = built(Some(2));
    let a = row(269, 3);
    send(&engine, &a, START, TOKENS);
    let within = clock(sent_at(START) + 1_999);
    assert_eq!(instants(&sent, &within, &a), [(sent_at(START), TOKENS)]);
    assert!(instants(&sent, &clock(sent_at(START) + 2_000), &a).is_empty());
}

/// The sends come lane by lane as the count keeps them, oldest first
/// within a lane. Two threads' sends are not merged by instant, so a stage
/// that wants one order sorts.
#[test]
fn the_instants_come_lane_by_lane_and_oldest_first_within_a_lane() {
    let (engine, sent) = built(None);
    let a = row(271, 3);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            take_lane(0);
            // The later send is recorded first: its lane keeps the two by
            // instant.
            send(&engine, &a, START + 30, 13);
            send(&engine, &a, START, 11);
        });
        first.join().expect("the first lane's sends");
        let second = scope.spawn(|| {
            take_lane(1);
            send(&engine, &a, START + 20, 17);
        });
        second.join().expect("the second lane's send");
    });
    let handed = [
        (sent_at(START), 11),
        (sent_at(START + 30), 13),
        (sent_at(START + 20), 17),
    ];
    assert_eq!(instants(&sent, &clock(0), &a), handed);
}

/// A send reported while the instants are read is not lost. The read holds
/// every lane of the row while it calls, so the send's record waits and
/// lands behind the read: the read hands the sends that stood before it,
/// and the next read hands the new one too. The read stays in its first
/// call until the sender's thread runs, and a thousand turns of the
/// scheduler more: the send is under way while the read runs and has not
/// landed when the read goes on.
#[test]
fn a_send_reported_while_the_instants_are_read_lands_behind_the_read() {
    let (engine, sent) = built(None);
    let a = row(277, 3);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 3, 281);
    let entry = sent.credits.tally.load().get(&(a.id, a.dp_rank)).cloned();
    let entry = entry.expect("the row's entry");
    let facts = clock(0);
    let (started, landed) = (AtomicBool::new(false), AtomicBool::new(false));
    let (mut read, mut held, mut waited) = (Vec::new(), true, true);
    std::thread::scope(|scope| {
        let mut sender = None;
        sent.pass(&facts).sends(&a, |at_micros, tokens| {
            held &= entry.lanes_held_for_test();
            // The first call starts a send to the row on another thread.
            sender.get_or_insert_with(|| {
                let sender = scope.spawn(|| {
                    started.store(true, Ordering::Release);
                    send(&engine, &a, START + 9, 331);
                    landed.store(true, Ordering::Release);
                });
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                (0..1_000).for_each(|_| std::thread::yield_now());
                sender
            });
            waited &= !landed.load(Ordering::Acquire);
            read.push((at_micros, tokens));
        });
    });
    assert!(held, "the read holds every lane of the row while it calls");
    assert!(waited, "the send has not landed while the read runs");
    let before = [(sent_at(START), TOKENS), (sent_at(START + 3), 281)];
    assert_eq!(read, before);
    let mut all = instants(&sent, &facts, &a);
    all.sort_unstable();
    assert_eq!(all[..2], before);
    assert_eq!(all[2..], [(sent_at(START + 9), 331)]);
    assert_eq!(sent.pass(&facts).counts(&a).requests, 3);
}

#[test]
fn the_two_number_read_says_the_requests_and_stated_tokens_of_the_whole_read() {
    let (engine, sent) = built(None);
    let a = row(331, 3);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 1, 0);
    send(&engine, &a, START + 2, 293);

    let facts = clock(START + 700);
    let reads = sent.pass(&facts);
    let whole = reads.of(&a);
    assert_eq!(whole.unsized_requests, 1);
    let counts = DispatchCounts {
        requests: 3,
        tokens: u64::from(TOKENS) + 293,
    };
    assert_eq!(reads.counts(&a), counts);
    assert_eq!(
        (whole.requests, whole.tokens),
        (counts.requests, counts.tokens)
    );
    assert_eq!(
        reads.counts(&row(337, a.dp_rank)),
        DispatchCounts::default()
    );
}

#[test]
fn the_two_number_read_releases_the_sends_a_report_reflects() {
    let (engine, sent) = built(None);
    let mut a = row(347, 3);
    let at = START + u64::from(DELAY);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START + 7, 293);
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = at;

    // No other read ran first: this one releases the send the report
    // reflects, and the whole read finds it gone.
    let facts = clock(at + 400);
    let later = DispatchCounts {
        requests: 1,
        tokens: 293,
    };
    assert_eq!(sent.pass(&facts).counts(&a), later);
    assert_eq!(sent.read(&facts, &a), one(293));
}

#[test]
fn a_release_leaves_a_later_send_of_no_stated_size_in_the_read_that_ran_it() {
    let (engine, sent) = built(None);
    let mut a = row(353, 5);
    send(&engine, &a, START, 0);
    send(&engine, &a, START + 9, 0);
    // A report that reflects the first send: the read that releases it
    // still counts the second, and counts it as of no stated size.
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY);
    assert_eq!(
        sent.read(&clock(0), &a),
        CreditedDispatches {
            requests: 1,
            tokens: 0,
            unsized_requests: 1,
            estimated_tokens: 0,
        }
    );
}

#[test]
fn the_two_number_read_counts_no_send_past_the_time_limit() {
    let (engine, sent) = built(Some(17));
    let a = row(349, 3);
    let at = START + u64::from(DELAY);
    send(&engine, &a, START, TOKENS);

    let held = DispatchCounts {
        requests: 1,
        tokens: u64::from(TOKENS),
    };
    assert_eq!(sent.pass(&clock(at + 16_999)).counts(&a), held);
    assert_eq!(
        sent.pass(&clock(at + 17_000)).counts(&a),
        DispatchCounts::default()
    );
}

#[test]
fn an_unset_limit_uses_the_report_alone() {
    let (engine, sent) = built(None);
    let mut a = row(131, 3);
    let facts = clock(START + 7_300_000);
    send(&engine, &a, START, TOKENS);
    assert_eq!(sent.read(&facts, &a), one(TOKENS));
    a.load = Some(LoadRow::default());
    a.load_sampled_at_micros = START + u64::from(DELAY);
    assert_eq!(sent.read(&facts, &a), CreditedDispatches::default());
}

#[test]
fn a_clock_at_zero_or_before_the_limit_does_not_age_sends() {
    let (engine, sent) = built(Some(19));
    let a = row(137, 3);
    send(&engine, &a, 71, TOKENS);
    assert_eq!(sent.read(&clock(0), &a), one(TOKENS));
    assert_eq!(sent.read(&clock(271), &a), one(TOKENS));
}

#[test]
fn removing_a_worker_clears_its_ranks_sends() {
    let (engine, sent) = built(None);
    let a = row(139, 3);
    let other_rank = row(a.id.0, 7);
    let b = row(149, 3);
    for candidate in [&a, &other_rank, &b] {
        send(&engine, candidate, START, TOKENS);
    }
    engine.on_membership(&Membership::Removed(a.id));
    let facts = clock(0);
    let reads = sent.pass(&facts);
    assert_eq!(reads.of(&a), CreditedDispatches::default());
    assert_eq!(reads.of(&other_rank), CreditedDispatches::default());
    assert_eq!(reads.of(&b), one(TOKENS));
}

#[cfg_attr(
    not(feature = "in-process"),
    expect(
        dead_code,
        reason = "the cases that read it are built under `in-process`"
    )
)]
fn validated(document: &PolicySpec, catalog: &StageRegistry) -> AppliedSpec {
    let digest = document.digest().expect("a digest");
    AppliedSpec::validate_digested(document, digest, catalog, &[]).expect("the policy validates")
}

/// The first target of a held decision, as the candidate a reader is asked
/// of, whichever engine decided it.
#[cfg_attr(
    not(feature = "in-process"),
    expect(
        dead_code,
        reason = "the cases that read it are built under `in-process`"
    )
)]
fn chosen<D: Deref<Target: DecisionEngine>>(decision: &DecisionGuard<D>) -> Candidate {
    let target = decision
        .decision()
        .pick()
        .and_then(|pick| pick.targets.first())
        .expect("a pick");
    row(target.id.0, target.dp_rank)
}

#[test]
fn each_side_of_a_pair_credits_its_own_rank() {
    let (engine, sent) = built(None);
    let prefill = row(151, 3);
    let decode = row(157, 7);
    let target = Target {
        pair: Some(Pair {
            decode_id: decode.id,
            decode_dp_rank: decode.dp_rank,
            ..Pair::default()
        }),
        ..target(&prefill)
    };
    let mut lease = Lease {
        issued_micros: START,
        tokens: TOKENS,
        ..Lease::default()
    };
    let facts = clock(0);
    engine.on_dispatched(&lease, &target, Side::Decode, DELAY);
    assert_eq!(sent.read(&facts, &decode), one(TOKENS));
    assert_eq!(sent.read(&facts, &prefill), CreditedDispatches::default());
    lease.tokens = 349;
    engine.on_dispatched(&lease, &target, Side::Prefill, DELAY);
    assert_eq!(sent.read(&facts, &prefill), one(349));
    assert_eq!(sent.read(&facts, &decode), one(TOKENS));
    assert_eq!(
        sent.read(&facts, &row(decode.id.0, 3)),
        CreditedDispatches::default()
    );
}

#[test]
fn a_profiles_limit_does_not_end_another_profiles_sends() {
    let (catalog, readers) = catalog();
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "profiles": {
            "brief": { "sent_ttl_ms": 23, "score": ["avoid-sent"], "pick": "best" },
            "long": { "sent_ttl_ms": 3719, "score": ["avoid-sent"], "pick": "best" }
        },
        "bindings": [{ "match": { "model": "*" }, "profiles": { "single": "brief" } }]
    }))
    .expect("a document");
    let engine = document.validate_in(&catalog, &[]).expect("validates");
    let a = row(163, 3);
    send(&engine, &a, START, TOKENS);
    let facts = clock(START + 29_000);
    let readers = readers.lock();
    assert_eq!(readers[0].read(&facts, &a), CreditedDispatches::default());
    assert_eq!(readers[1].read(&facts, &a), one(TOKENS));
}

#[test]
fn two_scorers_in_one_profile_share_one_tally() {
    let (catalog, readers) = catalog();
    let mut document = document(None);
    document
        .profiles
        .get_mut("work")
        .expect("the profile")
        .score
        .push("avoid-sent".into());
    let engine = document.validate_in(&catalog, &[]).expect("validates");
    let readers = readers.lock();
    assert!(Arc::ptr_eq(&readers[0], &readers[1]));
    let a = row(167, 3);
    send(&engine, &a, START, TOKENS);
    assert_eq!(readers[1].read(&clock(0), &a), one(TOKENS));
}

/// A profile of `least-load` under `name`, with the registered reader
/// beside it where `reader` says so.
fn load_document(name: &str, limit: Option<u32>, reader: bool) -> PolicySpec {
    let score = if reader {
        json!(["least-load", "avoid-sent"])
    } else {
        json!(["least-load"])
    };
    serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": { (name): { "sent_ttl_ms": limit, "score": score, "pick": "best" } }
    }))
    .expect("a document")
}

fn load_engine(name: &str, limit: Option<u32>) -> Engine {
    load_document(name, limit, false)
        .validate()
        .expect("the profile validates")
}

/// Two rows with equal reports, sampled ahead of every send of a case.
fn reporting() -> [Candidate; 2] {
    [(0, 191), (1, 193)].map(|(position, id)| Candidate {
        position,
        load: Some(LoadRow::default()),
        load_sampled_at_micros: START - 1,
        ..row(id, 3)
    })
}

/// The scores of `rows` in a decision at `now_micros`, in the rows' order.
fn scores(engine: &Engine, rows: &[Candidate; 2], now_micros: u64) -> [f32; 2] {
    let pool = PoolFacts {
        pool_size: 2,
        ..PoolFacts::default()
    };
    let mut scratch = Scratch::new();
    let decision = engine.decide(
        &clock(now_micros),
        &pool,
        rows,
        &Constraints::default(),
        &mut scratch,
    );
    let pick = decision.pick().expect("the rows route");
    let top = &pick.trace.top[..usize::from(pick.trace.top_len)];
    rows.each_ref().map(|row| {
        let scored = top.iter().find(|(id, _)| *id == row.id);
        scored.expect("each row is scored").1
    })
}

#[test]
fn least_load_reads_its_profiles_count_and_keeps_none_of_its_own() {
    let engine = load_engine(POOL, None);
    assert_eq!(engine.sent_credits.len(), 1);
    let rows = reporting();
    let lease = Lease {
        issued_micros: START,
        tokens: TOKENS,
        ..Lease::default()
    };
    // The counts alone hear this send: a stage with a tally of its own
    // would score the two rows alike.
    engine.dispatched_sent(&lease, &target(&rows[0]), Side::Single, DELAY, || true);
    let [sent_to, other] = scores(&engine, &rows, START + 700);
    assert!(sent_to < other, "{sent_to} against {other}");
}

#[test]
fn least_load_and_a_reader_share_one_count_and_a_send_is_counted_once() {
    let (catalog, readers) = catalog();
    let engine = load_document(POOL, None, true)
        .validate_in(&catalog, &[])
        .expect("the profile validates");
    assert_eq!(engine.sent_credits.len(), 1);
    assert!(Arc::ptr_eq(&engine.sent_credits[0], &readers.lock()[0]));
    let rows = reporting();
    send(&engine, &rows[0], START, TOKENS);
    assert_eq!(
        engine.sent_credits[0].read(&clock(0), &rows[0]),
        one(TOKENS)
    );
}

#[test]
fn least_load_prices_a_send_of_no_stated_size_at_its_mean() {
    let mut document = load_document(POOL, None, false);
    document.profiles.get_mut(POOL).expect("the profile").score =
        vec![
            serde_json::from_value(json!({ "least-load": { "mean_prefill_tokens": 701 } }))
                .expect("a stage"),
        ];
    let engine = document.validate().expect("the profile validates");
    let rows = reporting();
    send(&engine, &rows[0], START, 0);
    send(&engine, &rows[1], START, 701);
    let sent = &engine.sent_credits[0];
    assert_eq!(
        sent.read(&clock(0), &rows[0]),
        CreditedDispatches {
            unsized_requests: 1,
            ..one(0)
        }
    );
    let [unsized_send, stated] = scores(&engine, &rows, START + 700);
    assert_eq!(unsized_send, stated);
    send(&engine, &rows[1], START, 1);
    let [unsized_send, stated] = scores(&engine, &rows, START + 700);
    assert!(stated < unsized_send, "{stated} against {unsized_send}");
}

/// A send of text alone states no size to `least-load` either. The stage
/// prices it at its mean prefill and adds nothing for the text's bytes,
/// which the count keeps apart for the documents' terms.
#[test]
fn least_load_prices_a_send_of_text_alone_at_its_mean() {
    let mut document = load_document(POOL, None, false);
    document.profiles.get_mut(POOL).expect("the profile").score =
        vec![
            serde_json::from_value(json!({ "least-load": { "mean_prefill_tokens": 701 } }))
                .expect("a stage"),
        ];
    let engine = document.validate().expect("the profile validates");
    let rows = reporting();
    // 404 bytes of text: 101 by the bytes over four.
    let text = Lease {
        issued_micros: START,
        bytes: 404,
        ..Lease::default()
    };
    engine.on_dispatched(&text, &target(&rows[0]), Side::Single, DELAY);
    send(&engine, &rows[1], START, 701);
    assert_eq!(
        engine.sent_credits[0].read(&clock(0), &rows[0]),
        CreditedDispatches {
            unsized_requests: 1,
            estimated_tokens: 101,
            ..one(0)
        }
    );
    let [text_send, stated] = scores(&engine, &rows, START + 700);
    assert_eq!(text_send, stated);
}

#[test]
fn the_profiles_limit_ends_least_loads_credit() {
    let limited = load_engine(POOL, Some(17));
    let unlimited = load_engine(POOL, None);
    let rows = reporting();
    for engine in [&limited, &unlimited] {
        send(engine, &rows[0], START, TOKENS);
    }
    let ends = START + u64::from(DELAY) + 17_000;
    let [sent_to, other] = scores(&limited, &rows, ends - 1);
    assert!(sent_to < other, "{sent_to} against {other}");
    let [sent_to, other] = scores(&limited, &rows, ends);
    assert_eq!(sent_to, other);
    // With no limit the credit stands until a report reflects the send.
    let [sent_to, other] = scores(&unlimited, &rows, ends + 7_300_000);
    assert!(sent_to < other, "{sent_to} against {other}");
    let reflected = rows.map(|row| Candidate {
        load_sampled_at_micros: START + u64::from(DELAY),
        ..row
    });
    let [sent_to, other] = scores(&unlimited, &reflected, ends + 7_300_000);
    assert_eq!(sent_to, other);
}

/// [`load_engine`] under the report delay and the time limit `limit`.
fn delayed_load_engine(limit: Option<u32>) -> Engine {
    let mut document = load_document(POOL, limit, false);
    document
        .profiles
        .get_mut(POOL)
        .expect("the profile")
        .sent_report_delay_ms = Some(REPORT_DELAY_MS);
    document.validate().expect("the profile validates")
}

/// [`reporting`]'s rows, their reports sampled `behind` microseconds after
/// the send [`send`] makes of a lease issued at [`START`].
fn sampled(behind: u64) -> [Candidate; 2] {
    reporting().map(|row| Candidate {
        load_sampled_at_micros: START + u64::from(DELAY) + behind,
        ..row
    })
}

#[test]
fn the_profiles_report_delay_keeps_least_loads_credit_past_a_report_sampled_sooner() {
    let delayed = delayed_load_engine(None);
    let plain = load_engine(POOL, None);
    let rows = reporting();
    for engine in [&delayed, &plain] {
        send(engine, &rows[0], START, TOKENS);
    }
    let now = START + u64::from(DELAY) + 9_000;
    // A report three milliseconds behind the send reflects it where the
    // profile sets no delay, and leaves the credit under a delay of five.
    let [sent_to, other] = scores(&plain, &sampled(3_000), now);
    assert_eq!(sent_to, other);
    let [sent_to, other] = scores(&delayed, &sampled(3_000), now);
    assert!(sent_to < other, "{sent_to} against {other}");
    // A report five behind it ends the credit.
    let [sent_to, other] = scores(&delayed, &sampled(REPORT_DELAY), now);
    assert_eq!(sent_to, other);
}

#[test]
fn least_load_under_a_limit_and_a_delay_follows_both() {
    let engine = delayed_load_engine(Some(20));
    send(&engine, &reporting()[0], START, TOKENS);
    let sent_at = START + u64::from(DELAY);
    // A report three milliseconds behind the send: the delay keeps the
    // credit.
    let [sent_to, other] = scores(&engine, &sampled(3_000), sent_at + 4_000);
    assert!(sent_to < other, "{sent_to} against {other}");
    // The limit ends it at twenty, under the same report.
    let [sent_to, other] = scores(&engine, &sampled(3_000), sent_at + 19_999);
    assert!(sent_to < other, "{sent_to} against {other}");
    let [sent_to, other] = scores(&engine, &sampled(3_000), sent_at + 20_000);
    assert_eq!(sent_to, other);
}

#[test]
fn a_new_documents_least_load_keeps_the_sends_of_the_standing_one() {
    let standing = load_engine(POOL, None);
    let rows = reporting();
    send(&standing, &rows[0], START, TOKENS);
    let next = load_engine(POOL, None);
    next.succeed(&standing);
    let [sent_to, other] = scores(&next, &rows, START + 700);
    assert!(sent_to < other, "{sent_to} against {other}");
}

#[test]
fn least_load_under_a_new_profile_name_starts_with_no_sends() {
    let standing = load_engine(POOL, None);
    let rows = reporting();
    send(&standing, &rows[0], START, TOKENS);
    let next = load_engine("renamed", None);
    next.succeed(&standing);
    let [sent_to, other] = scores(&next, &rows, START + 700);
    assert_eq!(sent_to, other);
}

#[test]
fn presence_is_asked_only_where_a_send_enters_a_rank() {
    let (engine, sent) = built(None);
    let a = row(197, 3);
    let lease = Lease {
        issued_micros: START,
        tokens: TOKENS,
        ..Lease::default()
    };
    // A rank the count does not know is asked for, and a row that left
    // enters none.
    engine.dispatched_sent(&lease, &target(&a), Side::Single, DELAY, || false);
    assert!(!sent.credits.tally.load().contains_key(&(a.id, a.dp_rank)));
    engine.dispatched_sent(&lease, &target(&a), Side::Single, DELAY, || true);
    // A rank it knows takes the send unasked.
    engine.dispatched_sent(&lease, &target(&a), Side::Single, DELAY, || {
        panic!("a known rank was asked for")
    });
    assert_eq!(sent.read(&clock(0), &a).requests, 2);
}

#[test]
fn a_limit_is_taken_where_least_load_reads_the_sends() {
    assert_eq!(load_engine(POOL, Some(71)).sent_credits.len(), 1);
}

#[test]
fn a_zero_limit_is_refused_at_the_profiles_field() {
    let (catalog, _) = catalog();
    assert_eq!(
        document(Some(0)).validate_in(&catalog, &[]).err(),
        Some(SpecError::BadField {
            path: "profiles.work.sent_ttl_ms".into(),
            value: "0".into(),
            reason: "above zero",
        })
    );
}

#[test]
fn a_limit_without_a_reader_is_refused_at_the_profiles_field() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document
        .profiles
        .get_mut("random")
        .expect("the profile")
        .sent_ttl_ms = Some(71);
    assert_eq!(
        document.validate().err(),
        Some(SpecError::BadField {
            path: "profiles.random.sent_ttl_ms".into(),
            value: "71".into(),
            reason: "a limit any stage of this profile reads",
        })
    );
}

#[test]
fn a_zero_report_delay_is_refused_at_the_profiles_field() {
    let (catalog, _) = catalog();
    assert_eq!(
        delayed(None, 0).validate_in(&catalog, &[]).err(),
        Some(SpecError::BadField {
            path: "profiles.work.sent_report_delay_ms".into(),
            value: "0".into(),
            reason: "above zero",
        })
    );
}

#[test]
fn a_report_delay_without_a_reader_is_refused_at_the_profiles_field() {
    let mut document = PolicySpec::preset("random").expect("a preset");
    document
        .profiles
        .get_mut("random")
        .expect("the profile")
        .sent_report_delay_ms = Some(71);
    assert_eq!(
        document.validate().err(),
        Some(SpecError::BadField {
            path: "profiles.random.sent_report_delay_ms".into(),
            value: "71".into(),
            reason: "a delay any stage of this profile reads",
        })
    );
}

#[test]
fn a_document_with_no_reader_keeps_no_send_tallies() {
    let engine = PolicySpec::preset("random")
        .expect("a preset")
        .validate()
        .expect("a valid document");
    assert!(engine.sent_credits.is_empty());
    send(&engine, &row(173, 3), START, TOKENS);
    assert!(engine.sent_credits.is_empty());
}

#[test]
fn an_unset_limit_leaves_a_documents_canonical_form_and_digest_intact() {
    let canonical = format!(
        r#"{{"apiVersion":"{API_VERSION}","generation":0,"profiles":{{"work":{{"filter":["healthy"],"score":["least-requests"],"pick":"best"}}}},"bindings":[]}}"#
    );
    let document: PolicySpec = serde_json::from_str(&canonical).expect("an existing document");
    assert_eq!(
        serde_json::to_string(&document).expect("the form"),
        canonical
    );
    assert_eq!(
        document.digest().expect("a digest").to_string(),
        "sha256:5d615ed114ca79046283cd72596dd3428baa2ec110170a5bb72ea4bc525b1ade"
    );
}

#[test]
fn a_written_report_delay_stands_in_a_documents_canonical_form() {
    let canonical = format!(
        r#"{{"apiVersion":"{API_VERSION}","generation":0,"profiles":{{"work":{{"filter":["healthy"],"score":["least-load"],"pick":"best","sent_report_delay_ms":5}}}},"bindings":[]}}"#
    );
    let document: PolicySpec = serde_json::from_str(&canonical).expect("a document");
    assert_eq!(
        serde_json::to_string(&document).expect("the form"),
        canonical
    );
    let profile = document.profiles.get(POOL).expect("the profile");
    assert_eq!(profile.sent_report_delay_ms, Some(5));
}

/// Holds `sent` to a report delay of nine milliseconds, over the two
/// sends [`send`] made `row` at [`START`]: the count a stage's factory was
/// handed is under its profile's delay, whichever kind of stage asked.
fn held_to_a_delay_of_nine(sent: &DispatchCredit, row: &Candidate) {
    let sent_at = START + u64::from(DELAY);
    let reported = |behind: u64| Candidate {
        load: Some(LoadRow::default()),
        load_sampled_at_micros: sent_at + behind,
        ..*row
    };
    let facts = clock(sent_at + 9_700);
    assert_eq!(sent.read(&facts, &reported(8_999)).requests, 2);
    assert_eq!(sent.read(&facts, &reported(9_000)).requests, 0);
}

/// Keeps the rows its profile sent fewer than `most` requests.
#[derive(Debug)]
struct SentUnder {
    sent: Arc<DispatchCredit>,
    most: u32,
}

impl Filter for SentUnder {
    fn name(&self) -> &'static str {
        "sent-under"
    }

    fn keep(&self, facts: &RequestFacts<'_>, candidate: &Candidate, _: &Constraints<'_>) -> bool {
        self.sent.read(facts, candidate).requests < self.most
    }
}

/// A registered filter holds its profile's count of sends, as a registered
/// scorer does. The count is made where the filter's factory asks for it,
/// ahead of any scorer, so the profile's limit has its reader.
#[test]
fn a_registered_filter_reads_the_sends_of_its_profile() {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Most {
        requests: u32,
    }

    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = StageRegistry::builtin();
    catalog
        .filter_with("sent-under", move |build| {
            let Most { requests } = build.params()?;
            let sent = build.dispatch_credit();
            kept.lock().push(Arc::clone(&sent));
            Ok(Box::new(SentUnder {
                sent,
                most: requests,
            }))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": {
            "work": {
                "sent_ttl_ms": 24_001,
                "sent_report_delay_ms": 9,
                "filter": [{ "sent-under": { "requests": 2 } }],
                "pick": "random"
            }
        }
    }))
    .expect("a document");
    let engine = document
        .validate_in(&catalog, &[])
        .expect("the filter reads the sends the limit ages");
    assert_eq!(engine.sent_credits.len(), 1);

    let a = row(127, 3);
    let b = row(131, a.dp_rank);
    let facts = clock(START + 700);
    let filter = &engine.profiles()[0].filters[0];
    let keeps = |row: &Candidate| filter.keep(&facts, row, &Constraints::default());
    send(&engine, &a, START, TOKENS);
    assert!(keeps(&a) && keeps(&b), "one send is under two");
    send(&engine, &a, START, TOKENS);
    assert!(!keeps(&a), "two sends are not under two");
    assert!(keeps(&b), "another row's sends are its own");
    // The count the factory was handed is the one the engine feeds.
    assert_eq!(
        readers.lock()[0].read(&facts, &a),
        CreditedDispatches {
            requests: 2,
            tokens: 2 * u64::from(TOKENS),
            ..CreditedDispatches::default()
        }
    );
    held_to_a_delay_of_nine(&readers.lock()[0], &a);
}

/// Picks the kept row its profile sent the fewest requests, the first of
/// them at a tie.
#[derive(Debug)]
struct LeastSent {
    sent: Arc<DispatchCredit>,
}

impl Lifecycle for LeastSent {}

impl framework::Picker for LeastSent {
    fn name(&self) -> &'static str {
        "least-sent"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut decision::Targets,
    ) {
        let sent = |row: &&u32| self.sent.read(facts, &candidates[**row as usize]).requests;
        if let Some(&row) = kept.iter().min_by_key(sent) {
            out.push(pickers::target_of(&candidates[row as usize]));
        }
    }
}

/// A registered picker holds its profile's count of sends, as a registered
/// filter does. A profile whose only reader of sends is its picker keeps
/// its limit: the picker is built ahead of the limit's check. The count its
/// factory was handed is the profile's own, the one the engine feeds.
#[test]
fn a_registered_picker_reads_the_sends_of_its_profile() {
    let readers = Readers::default();
    let kept = Arc::clone(&readers);
    let mut catalog = StageRegistry::builtin();
    catalog
        .picker_with("least-sent", move |build| {
            let sent = build.dispatch_credit();
            kept.lock().push(Arc::clone(&sent));
            Ok(Box::new(LeastSent { sent }))
        })
        .expect("a free name");
    let document: PolicySpec = serde_json::from_value(json!({
        "apiVersion": API_VERSION,
        "generation": 4,
        "profiles": {
            "work": {
                "sent_ttl_ms": 24_001,
                "sent_report_delay_ms": 9,
                "filter": ["healthy"],
                "score": ["least-requests"],
                "pick": "least-sent"
            }
        }
    }))
    .expect("a document");
    let engine = document
        .validate_in(&catalog, &[])
        .expect("the picker reads the sends the limit ages");
    assert_eq!(engine.sent_credits.len(), 1);

    let a = row(127, 3);
    let facts = clock(START + 700);
    send(&engine, &a, START, TOKENS);
    send(&engine, &a, START, TOKENS);
    assert_eq!(
        readers.lock()[0].read(&facts, &a),
        CreditedDispatches {
            requests: 2,
            tokens: 2 * u64::from(TOKENS),
            ..CreditedDispatches::default()
        }
    );
    held_to_a_delay_of_nine(&readers.lock()[0], &a);
}

/// A send told through a decision's handle, over the engine that decides
/// over worker lists.
#[cfg(feature = "in-process")]
mod over_lists {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_worker::{
        identity::basic::Worker,
        load::state::{one_rank_load, LoadSnapshot},
        test_support::ready_worker,
    };

    use super::{
        catalog, chosen, clock, document, one, send, validated, Readers, DELAY, POOL, TOKENS,
    };
    use crate::{
        document::AppliedSpec,
        engine::{
            decision_engine::{Request, RequestMaker, WorkerLists},
            list_cell::tests::PublishedList,
            policy_engine::PolicyEngine,
            Constraints, Live, RequestFacts,
        },
        store::GatewayId,
        tracking::{decision_guard::DecisionGuard, CreditedDispatches},
    };

    /// A request stated ahead, with the list its host hands for it.
    struct Asks<'a>(Request<'a>, &'a PublishedList);

    impl RequestMaker for Asks<'_> {
        fn model(&self) -> &str {
            self.0.facts.model
        }

        fn make<'w>(&'w self, _: &'w AppliedSpec) -> Request<'w> {
            self.0
        }

        fn workers(&self) -> WorkerLists<'_> {
            WorkerLists::Single(self.1.lent())
        }
    }

    /// The engine of the standing document, two ready workers as it reads
    /// their list, and the document's readers.
    fn bed() -> (PolicyEngine, PublishedList, Readers) {
        let (catalog, readers) = catalog();
        let engine = PolicyEngine::new(validated(&document(None), &catalog));
        let list: Arc<[Arc<dyn Worker>]> = ["http://first:8000", "http://second:8000"]
            .iter()
            .map(|url| ready_worker(url, POOL))
            .collect();
        let members = PublishedList::new(list);
        (engine, members, readers)
    }

    /// A decision for a request of [`TOKENS`] tokens over `members`, held
    /// by its handle.
    fn decide<'a>(
        engine: &'a PolicyEngine,
        members: &PublishedList,
        id: &[u8],
        now: Instant,
    ) -> DecisionGuard<&'a PolicyEngine> {
        let request = Request {
            id,
            facts: RequestFacts {
                model: POOL,
                prompt_tokens: TOKENS,
                ..RequestFacts::default()
            },
            constraints: Constraints::default(),
        };
        let asks = Asks(request, members);
        DecisionGuard::decide_under(engine, GatewayId::LOCAL, &asks, now).expect("a decision")
    }

    /// A send told through a decision's handle is read by the profile's
    /// reader, and the engine's next decision goes to the other worker.
    #[test]
    fn a_handle_dispatch_changes_the_next_in_process_pick() {
        let (engine, members, readers) = bed();
        let now = Instant::now();
        let mut first = decide(&engine, &members, b"first", now);
        let sent_to = chosen(&first);
        let sent = Arc::clone(&readers.lock()[0]);
        assert_eq!(
            sent.read(&clock(0), &sent_to),
            CreditedDispatches::default()
        );
        first.dispatched(now);
        let second = decide(&engine, &members, b"second", now);
        assert_ne!(chosen(&second).id, sent_to.id);
        assert_eq!(sent.read(&clock(0), &sent_to), one(TOKENS));
    }

    /// A handle says its dispatch once, so a second word of the same send
    /// counts nothing. The engine files no lease: the handle is the one
    /// that tells it a word, and no entry takes a report by a lease's id.
    #[test]
    fn a_second_dispatch_word_of_one_handle_counts_nothing() {
        let (engine, members, readers) = bed();
        let now = Instant::now();
        let mut first = decide(&engine, &members, b"first", now);
        let sent_to = chosen(&first);
        first.dispatched(now);
        first.dispatched(now);
        assert_eq!(readers.lock()[0].read(&clock(0), &sent_to), one(TOKENS));
    }

    /// A report sampled behind a send ends the send's credit. A report
    /// sampled ahead of the send, read later, leaves the credit ended: no
    /// older report brings a send back.
    #[test]
    fn a_released_send_stays_retired_when_an_older_report_arrives() {
        let (engine, members, readers) = bed();
        let sent = Arc::clone(&readers.lock()[0]);
        let url = "http://first:8000";
        let time = engine.clock();
        let records = engine
            .members_of(members.lent())
            .records(engine.spec(), time);
        let stages = engine.spec().engine();

        let issued = Instant::now();
        let dispatched = issued + Duration::from_micros(u64::from(DELAY));
        let newer = dispatched + Duration::from_millis(13);
        let later = newer + Duration::from_millis(1);
        send(stages, &records.fixed()[0], time.micros(issued), TOKENS);

        // The first worker's report, sampled at `sampled_at` and read at
        // `now`: what the profile's reader counts for the worker then.
        let read = |sampled_at, running: u16, now| {
            let mut report = one_rank_load(0, 0.0, 0.0);
            report.loads[0].num_running_reqs = i32::from(running);
            report.sampled_at = Some(sampled_at);
            let loads = LoadSnapshot::from_loads_for_test(vec![(url.to_owned(), report)]);
            let live = Live::at(stages, time, now, Some(&loads), None);
            let row = records
                .read(members.workers(), &live, 0)
                .expect("the worker record");
            assert_eq!(row.load.expect("the report").running, u32::from(running));
            assert_eq!(row.load_sampled_at_micros, time.micros(sampled_at));
            sent.read(&clock(live.now_micros()), &row)
        };

        let empty = CreditedDispatches::default();
        assert_eq!(read(issued, 19, dispatched), one(TOKENS));
        assert_eq!(read(newer, 7, newer), empty);
        assert_eq!(read(issued, 19, later), empty, "the send stays retired");
    }
}
