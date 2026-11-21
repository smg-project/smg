//! A decision over live rows: the rows of a gateway that keeps a worker's
//! health and its requests in flight on the worker object. Such rows are
//! written once a membership, with no live fact in them, and their reader
//! is handed beside them ([`LiveRows`]): a row the decision asks about is
//! one read of its worker, a row it does not ask about is not read, and
//! what it decides it decides from one read of each worker.
//!
//! The decision is the profile's own, by the same tests and the same
//! picker. The draw and the whole pass by the profile's one test of a row
//! are written here for live rows ([`run_drawn_live`],
//! [`keep_folded_live`]). Every pass of a decision over rows that carry
//! their facts is left as it was: none is entered from here, and none
//! holds anything of these rows or asks which rows it was handed.
//!
//! What live rows do not decide is handed back ([`Engine::decide_live`]
//! returns `None`), and its caller writes rows for that decision
//! ([`Engine::decide`]):
//!
//! - A profile whose stages read a live fact off a row
//!   ([`Engine::decides_live`]).
//! - A pool the whole pass leaves nobody. The stages one by one, the soft
//!   filter that steps aside and the rule for a pool every worker of which
//!   is vetoed read a row's bits and its count off the row, so they are
//!   asked over rows that carry them.

use super::{
    combine, debug_assert_indexed, score_survivors, shed, trace_by, trace_ordered, Buffers, Engine,
    Entry, Marks, Plan, Profile, Scratch,
};
use crate::{
    document::spec::Needs,
    engine::{
        decision::{Decision, Pick, ShedReason, Targets, Trace},
        sticky_sessions::SessionVerdict,
        view::{Candidate, Constraints, LiveRows, Phase, PoolFacts, RequestFacts},
    },
    stages::builtins::RowTests,
    Draw,
};

/// The smallest pool at which a decision over live rows draws first
/// ([`Engine::decide_live`]), set by run. There a row asked about is a read
/// of its worker, where a row that carries its facts is a read of the row
/// beside the last: the whole pass pays that read a row at any size, and
/// the draw pays it for the rows it draws alone. So every pool that holds a
/// row is drawn from.
pub const LIVE_DRAW_FROM: usize = 1;

/// The draws a picker makes for one row of a decision over live rows before
/// the decision reads the pool whole ([`Picker::draw_live`]): the bound at
/// which the gateway's own sampled policies give a draw up. A draw reads
/// one worker and the pass reads them all, so a decision draws again where
/// one over rows that carry their facts would go to the pass
/// ([`DRAW_BOUND`]). With half of a large pool refused, sixteen draws all
/// miss once in 65,536 decisions.
///
/// [`Picker::draw_live`]: super::Picker::draw_live
/// [`DRAW_BOUND`]: super::DRAW_BOUND
pub const LIVE_DRAWS: u32 = 16;

/// The refused rows a draw over live rows names to the whole pass it widens
/// to ([`Refused`]): the most draws a stock picker makes for one row
/// ([`LIVE_DRAWS`]).
const REFUSED_NAMED: usize = LIVE_DRAWS as usize;

/// The rows a draw over live rows landed on and refused, kept on the draw's
/// own stack for the whole pass it widens to ([`LiveRows::pass`]): a
/// picker's own bound of them at the most. A draw that refuses more names
/// none, and the pass then tests each row by the read the decision holds.
struct Refused {
    rows: [u32; REFUSED_NAMED],
    len: usize,
}

impl Refused {
    const fn new() -> Self {
        Self {
            rows: [0; REFUSED_NAMED],
            len: 0,
        }
    }

    /// One more row the draw refused.
    fn push(&mut self, row: u32) {
        if let Some(slot) = self.rows.get_mut(self.len) {
            *slot = row;
        }
        // A count past the list's room says a refused row is not named.
        self.len += 1;
    }

    /// The rows in rising order, where every refused row is named: what the
    /// whole pass leaves out. A row the draw landed on twice stands twice.
    fn named(&mut self) -> Option<&[u32]> {
        let rows = self.rows.get_mut(..self.len)?;
        rows.sort_unstable();
        Some(rows)
    }
}

/// [`LIVE_DRAW_FROM`] as the run that sets it turns it: a bench asks the
/// same decision under each floor in one build.
#[cfg(feature = "test-util")]
mod turned {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LIVE_DRAW_FROM: AtomicUsize = AtomicUsize::new(super::LIVE_DRAW_FROM);

    pub(super) fn live_draw_from() -> usize {
        LIVE_DRAW_FROM.load(Ordering::Relaxed)
    }

    pub(super) fn turn(rows: usize) {
        LIVE_DRAW_FROM.store(rows, Ordering::Relaxed);
    }
}

/// Turns the floor a decision over live rows draws first from, for every
/// engine of the process: the knob of the run that sets [`LIVE_DRAW_FROM`],
/// and no test's.
#[cfg(feature = "test-util")]
pub fn turn_live_draw_from(rows: usize) {
    turned::turn(rows);
}

/// The smallest pool at which a decision over live rows draws first:
/// [`LIVE_DRAW_FROM`], or what the run that sets it turned it to.
#[inline]
fn live_draw_from() -> usize {
    #[cfg(feature = "test-util")]
    {
        turned::live_draw_from()
    }
    #[cfg(not(feature = "test-util"))]
    {
        LIVE_DRAW_FROM
    }
}

/// Whether a profile's stages decide over rows that carry no live fact: the
/// build folded its filters into the one test of a row the passes here ask
/// ([`RowTests::of`]), and its picker and every scorer say they read no
/// live fact off a row ([`Picker::reads_no_live_fact`],
/// [`Scorer::reads_no_live_fact`]).
///
/// [`Picker::reads_no_live_fact`]: super::Picker::reads_no_live_fact
/// [`Scorer::reads_no_live_fact`]: super::Scorer::reads_no_live_fact
pub(super) fn reads_no_live_fact(profile: &Profile, tests: Option<RowTests>) -> bool {
    tests.is_some()
        && profile.picker.reads_no_live_fact()
        && profile
            .scorers
            .iter()
            .all(|weighted| weighted.scorer.reads_no_live_fact())
}

impl Engine {
    /// Whether a decision for `model` in `phase` is one live rows make
    /// ([`Self::decide_live`]). Its profile's stages read no live fact off
    /// a row ([`Picker::reads_no_live_fact`]) and ask nothing of a decision
    /// beyond the profile's own run but the draw: no scope to open, no rows
    /// named, no ranks, no engines listed. Its binding is of the single
    /// plan and holds no sessions rule. A model no binding matches reads so
    /// too: its decision is a decline, and reads no row.
    ///
    /// A caller asks once a membership, and writes rows for every decision
    /// this says no to.
    ///
    /// [`Picker::reads_no_live_fact`]: super::Picker::reads_no_live_fact
    #[must_use]
    pub fn decides_live(&self, model: &str, phase: Phase) -> bool {
        self.resolve(model, phase)
            .is_none_or(|(entry, index, _)| self.live_at(entry, index))
    }

    /// Whether a decision for `model` in `phase` reads no clock: its profile
    /// asks its caller for nothing ([`Engine::reads_for`]), shows no row to
    /// a plugin, refuses by no report, and its binding holds no sessions
    /// rule. Such a decision reads its rows' bits and counts, its pool's
    /// rotation and its seed, and may be handed no instant
    /// ([`RequestFacts::now_micros`] of `0`). A caller that keeps its own
    /// clock asks once a membership, not once a request.
    #[must_use]
    pub fn clock_free(&self, model: &str, phase: Phase) -> bool {
        let Some((entry, _, _)) = self.resolve(model, phase) else {
            return true;
        };
        let (needs, shown, by_reports) = self.reads_for(model, phase);
        needs == Needs::default() && !shown && !by_reports && entry.binding.sessions.is_none()
    }

    /// [`Self::decides_live`] of the profile at `index` under `entry`'s
    /// binding.
    fn live_at(&self, entry: &Entry, index: u16) -> bool {
        let at = usize::from(index);
        self.reads_no_live_fact[at]
            && self.marks[at].0 & !Marks::DRAWS_FIRST == 0
            && entry.binding.plan == Plan::Single
            && entry.binding.sessions.is_none()
    }

    /// [`Self::decide`] over rows that carry no live fact: `candidates`
    /// hold no health bit and no count, and `live` reads both of a row the
    /// decision asks about. No mask and no constraint stand beside such
    /// rows: a worker's vetoes are its own bits, as its reader reads them.
    ///
    /// The decision is the profile's own. A row is asked about through the
    /// profile's one test of a row, over the one read of its worker. Over a
    /// pool of [`LIVE_DRAW_FROM`] rows or more the picker draws its pick
    /// ([`Picker::draw_live`]). A draw that keeps nobody widens to one pass
    /// over the pool, a read of every worker not read yet, whose survivors
    /// the scorers and the picker are handed as [`Self::decide`] hands
    /// them. No row is written.
    ///
    /// `None` where live rows do not make the decision: its profile is not
    /// one they decide ([`Self::decides_live`]), or the whole pass left
    /// nobody. The caller writes rows for it and asks [`Self::decide`],
    /// whose stages name the stage that emptied the pool, let a soft filter
    /// step aside and ask the all-vetoed rule, each over a row's own facts.
    /// A pool is then read once more for them, as the gateway's own
    /// placement reads a pool its policy placed nobody in.
    ///
    /// [`Picker::draw_live`]: super::Picker::draw_live
    pub fn decide_live(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        live: &dyn LiveRows,
        scratch: &mut Scratch,
    ) -> Option<Decision> {
        #[cfg(debug_assertions)]
        {
            scratch.kept_last = false;
        }
        debug_assert_indexed(candidates);
        let Some((entry, index, profile)) = self.resolve(facts.model, facts.phase) else {
            let bare = Trace {
                seed: facts.seed,
                generation: pool.generation,
                ..Trace::default()
            };
            return Some(shed(ShedReason::PolicyDeclined, "binding", bare));
        };
        if !self.live_at(entry, index) {
            return None;
        }
        let at = usize::from(index);
        let cx = Constraints::default();
        let tests = self.row_tests[at]?.asked(facts, &cx);
        let asked = Asked {
            profile,
            index,
            tests,
            live,
        };
        let buf = &mut scratch.single;
        if self.marks[at].ahead() == Marks::DRAWS_FIRST && candidates.len() >= live_draw_from() {
            run_drawn_live(&asked, facts, pool, candidates, &cx, buf)
        } else {
            // No row is read yet.
            run_whole_live(&asked, facts, pool, candidates, &cx, buf, Some(&[]))
        }
    }
}

/// What a decision over live rows asks its rows by: the profile, its place,
/// its one test of a row as this request asks it ([`RowTests::asked`]), and
/// the reader of the rows' live facts.
struct Asked<'a> {
    profile: &'a Profile,
    index: u16,
    tests: RowTests,
    live: &'a dyn LiveRows,
}

/// The whole pass over live rows, by the profile's one test of a row: the
/// rows every filter keeps, in the pool's order, are left in `kept`. Says
/// whether a row stands.
///
/// These are the rows [`keep_folded`](super::keep_folded) leaves of rows
/// written with what the reader reads: the test is the same set, in two
/// halves. The reader asks the two tests of a row's health bits where it
/// reads the row's worker, in one call for the pool ([`LiveRows::pass`]):
/// a worker the decision has not read yet is read once there, and a row
/// that is refused costs that read and that test and nothing more. The
/// rows the decision's draw refused (`refused`) are named to it, so it
/// asks nothing of them. The tests that read the row itself are then asked
/// of the rows that passed. Out of line, as that pass is.
#[inline(never)]
fn keep_folded_live(
    tests: RowTests,
    live: &dyn LiveRows,
    refused: Option<&[u32]>,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    kept: &mut Vec<u32>,
) -> bool {
    kept.clear();
    live.pass(tests.live_test(cx), refused, kept);
    if tests.reads_rows() {
        kept.retain(|&row| {
            candidates
                .get(row as usize)
                .is_some_and(|c| tests.keeps_row(facts, c, cx))
        });
    }
    !kept.is_empty()
}

/// The whole pass of a decision over live rows, and the one its draw widens
/// to ([`run_drawn_live`]): the rows the profile's one test keeps
/// ([`keep_folded_live`]), then the scoring half of the profile's run, as
/// over rows that carry their facts. `None` where the test keeps no row:
/// what decides such a pool reads a row's facts off the row, and the pass
/// kept no read of a row it refused, so the decision is handed back for
/// rows written for it ([`Engine::decide_live`]).
///
/// `refused` is [`LiveRows::pass`]'s: the rows a draw read before this
/// pass, where it refused each and the decision read no other.
#[inline(never)]
fn run_whole_live(
    asked: &Asked<'_>,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    buf: &mut Buffers,
    refused: Option<&[u32]>,
) -> Option<Decision> {
    let &Asked {
        profile,
        index,
        tests,
        live,
    } = asked;
    buf.reserve(candidates.len());
    keep_folded_live(tests, live, refused, facts, candidates, cx, &mut buf.kept)
        .then(|| score_survivors(profile, index, facts, pool, candidates, buf))
}

/// A decision over live rows by a picker that draws first
/// ([`Picker::draw_live`](super::Picker::draw_live)). The picker draws its
/// pick, and every filter of the profile is asked about each row it draws
/// in the profile's one test, over one read of the row's worker. The
/// scorers are handed the row drawn and the picker hands it out. A draw
/// that keeps nobody says nothing of the pool: the decision widens to the
/// whole pass ([`run_whole_live`]), under the seed's stream past the values
/// the draw read, as a draw over rows that carry their facts widens
/// ([`run_drawn`](super::run_drawn)). The pass is told the rows the draw
/// refused and reads the others: a row the draw landed on is not read
/// again, and no row of the pool is looked up for a read it does not hold.
/// `None` where that pass keeps nobody, as it says.
#[inline(never)]
fn run_drawn_live(
    asked: &Asked<'_>,
    facts: &RequestFacts<'_>,
    pool: &PoolFacts<'_>,
    candidates: &[Candidate],
    cx: &Constraints<'_>,
    buf: &mut Buffers,
) -> Option<Decision> {
    let &Asked {
        profile,
        index,
        tests,
        live,
    } = asked;
    buf.reserve(candidates.len());
    buf.kept.clear();
    let mut refused = Refused::new();
    let mut draw = Draw::new(facts.seed);
    // Whether the draw kept a row it asked about: one that keeps a row and
    // gives up holds a read the refused rows do not name.
    let mut kept_one = false;
    let drew = {
        let refused = &mut refused;
        let mut ask = |c: &Candidate| {
            let keeps = tests.keeps_live(live.read(c.position), facts, c, cx);
            if keeps {
                kept_one = true;
            } else {
                refused.push(c.position);
            }
            keeps
        };
        profile
            .picker
            .draw_live(facts, pool, candidates, &mut ask, &mut buf.kept, &mut draw)
    };
    // A picker that read no value of the stream left the seed as it was.
    let resumed;
    let after = if draw.state() == facts.seed {
        facts
    } else {
        resumed = RequestFacts {
            seed: draw.state(),
            ..*facts
        };
        &resumed
    };
    if drew {
        let mut targets = Targets::new();
        let band = combine(
            profile,
            after,
            candidates,
            &buf.kept,
            &mut buf.scores,
            &mut buf.stage,
        );
        profile.picker.order_drawn(
            after,
            candidates,
            &buf.kept,
            &buf.scores,
            band,
            &mut targets,
        );
        if !targets.is_empty() {
            // Rows no scorer ordered all score zero, and their best are
            // the first drawn: the trace a draw reads off the scores, with
            // no pass over them. Scored rows are read as a draw's, by the
            // body the draw over rows that carry their facts enters through
            // its own one caller (`trace_drawn`), which stays that.
            let trace = if profile.scorers.is_empty() {
                let drawn = buf
                    .kept
                    .iter()
                    .map(|&row| (candidates[row as usize].id, 0.0));
                trace_ordered(index, facts, pool, drawn)
            } else {
                trace_by::<true>(index, facts, pool, candidates, &buf.kept, &buf.scores)
            };
            return Some(Decision::Pick(Pick {
                targets,
                trace,
                session_verdict: SessionVerdict::None,
            }));
        }
    }
    let named = if kept_one { None } else { refused.named() };
    let mut decision = run_whole_live(asked, after, pool, candidates, cx, buf, named);
    // The trace keeps the request's seed, from which the decision replays.
    match &mut decision {
        Some(Decision::Pick(pick)) => pick.trace.seed = facts.seed,
        Some(Decision::Shed(shed)) => shed.trace.seed = facts.seed,
        Some(Decision::Queue(queue)) => queue.trace.seed = facts.seed,
        None => {}
    }
    decision
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    use super::{
        super::{keep_folded, Engine},
        keep_folded_live, LIVE_DRAWS,
    };
    use crate::{
        document::spec::PolicySpec,
        engine::{
            decision::{Decision, ShedReason},
            view::{
                Candidate, Constraints, Health, LiveCell, LiveRow, LiveRows, LiveTest, Phase,
                PoolFacts, RequestFacts, WorkerId,
            },
            Scratch,
        },
        stages::builtins::{filters_of, live_turns, RowTests},
    };

    const MODEL: &str = "model-a";

    /// A pool's live facts by position, read as a gateway's reader reads
    /// its workers: a row's first read in a decision is kept, and the reads
    /// of the workers are counted.
    #[derive(Debug)]
    struct Rows {
        rows: Vec<LiveRow>,
        cells: Vec<LiveCell>,
        mark: AtomicU32,
        reads: AtomicUsize,
    }

    impl Rows {
        fn of(rows: Vec<LiveRow>) -> Self {
            Self {
                cells: rows.iter().map(|_| LiveCell::default()).collect(),
                rows,
                mark: AtomicU32::new(1),
                reads: AtomicUsize::new(0),
            }
        }

        /// Opens the next decision: no row is read yet. Returns the
        /// workers the last one read.
        fn next(&self) -> usize {
            self.mark.fetch_add(1, Ordering::Relaxed);
            self.reads.swap(0, Ordering::Relaxed)
        }

        fn fresh(&self, at: usize) -> LiveRow {
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.rows[at]
        }
    }

    impl LiveRows for Rows {
        fn read(&self, position: u32) -> LiveRow {
            let at = position as usize;
            let Some(cell) = self.cells.get(at) else {
                return LiveRow::default();
            };
            let mark = self.mark.load(Ordering::Relaxed);
            cell.kept(mark).unwrap_or_else(|| {
                let row = self.fresh(at);
                cell.keep(mark, row);
                row
            })
        }

        fn whole(&self) -> &[LiveCell] {
            let mark = self.mark.load(Ordering::Relaxed);
            for (at, cell) in self.cells.iter().enumerate() {
                if cell.kept(mark).is_none() {
                    cell.keep(mark, self.fresh(at));
                }
            }
            &self.cells
        }

        // As a gateway's reader passes over its workers: the rows a draw
        // refused are left out, every other row is read with no look at
        // its cell, and the read of a row that passes is kept.
        fn pass(&self, test: LiveTest, refused: Option<&[u32]>, kept: &mut Vec<u32>) {
            let Some(refused) = refused else {
                return test.pass_whole(self.whole(), kept);
            };
            let mark = self.mark.load(Ordering::Relaxed);
            for (at, cell) in (0u32..).zip(&self.cells) {
                if refused.contains(&at) {
                    continue;
                }
                let row = self.fresh(at as usize);
                if test.passes(row.health()) {
                    cell.keep(mark, row);
                    kept.push(at);
                }
            }
        }
    }

    const READY: Health = Health {
        healthy: true,
        breaker_open: false,
        stalled: false,
        draining: false,
        overloaded: false,
    };

    fn ready() -> LiveRow {
        LiveRow::new(READY, 0)
    }

    fn not_ready() -> LiveRow {
        let health = Health {
            healthy: false,
            ..READY
        };
        LiveRow::new(health, 0)
    }

    fn overloaded(in_flight: u32) -> LiveRow {
        let health = Health {
            overloaded: true,
            ..READY
        };
        LiveRow::new(health, in_flight)
    }

    /// The rows of a pool of `rows` workers as they are written once a
    /// membership: no live fact in them.
    fn bare(rows: u32) -> Vec<Candidate> {
        (0..rows)
            .map(|position| Candidate {
                position,
                id: WorkerId(1_000 + u64::from(position)),
                ..Candidate::default()
            })
            .collect()
    }

    /// `bare` with each row's live facts written into it.
    fn written(bare: &[Candidate], live: &[LiveRow]) -> Vec<Candidate> {
        let rows = bare.iter().zip(live);
        rows.map(|(row, live)| Candidate {
            health: live.health(),
            in_flight: live.in_flight(),
            ..*row
        })
        .collect()
    }

    fn preset(name: &str) -> Engine {
        let mut document = PolicySpec::preset(name).expect("a built-in preset");
        document.generation = 1;
        document.validate().expect("the preset validates")
    }

    fn facts(seed: u64) -> RequestFacts<'static> {
        RequestFacts {
            model: MODEL,
            seed,
            ..RequestFacts::default()
        }
    }

    fn pool_of(rows: &[Candidate]) -> PoolFacts<'static> {
        PoolFacts {
            pool_size: rows.len() as u32,
            membership_epoch: 1,
            generation: 1,
            ..PoolFacts::default()
        }
    }

    /// One decision of `engine` over `bare` beside `rows`, and the workers
    /// it read.
    fn decide(
        engine: &Engine,
        rows: &Rows,
        bare: &[Candidate],
        facts: &RequestFacts<'_>,
        scratch: &mut Scratch,
    ) -> (Option<Decision>, usize) {
        let decision = engine.decide_live(facts, &pool_of(bare), bare, rows, scratch);
        (decision, rows.next())
    }

    /// The row a decision names first.
    fn named(decision: &Decision) -> Option<u32> {
        match decision {
            Decision::Pick(pick) => pick.targets.as_slice().first().map(|t| t.position),
            _ => None,
        }
    }

    /// A word holds every health bit and the count beside them, and a count
    /// past the word's width reads as the width's most. The word's own test
    /// of the routing bits is the health bits' test.
    #[test]
    fn a_word_holds_the_bits_and_the_count() {
        for bits in 0u32..32 {
            let health = Health {
                healthy: bits & 1 != 0,
                breaker_open: bits & 2 != 0,
                stalled: bits & 4 != 0,
                draining: bits & 8 != 0,
                overloaded: bits & 16 != 0,
            };
            let row = LiveRow::new(health, 7 + bits);
            assert_eq!((row.health(), row.in_flight()), (health, 7 + bits));
            assert_eq!(
                row.routable_ignoring_overload(),
                health.routable_ignoring_overload()
            );
            assert_eq!(row.overloaded(), health.overloaded);
        }
        let most = LiveRow::new(READY, u32::MAX);
        assert_eq!(most.in_flight(), LiveRow::MAX_IN_FLIGHT);
        assert_eq!(most.health(), READY);
        assert!(!LiveRow::default().routable_ignoring_overload());
    }

    /// Every set of the folded tests, over every word of a row's health
    /// bits, tried and not, under the overload veto and with it lifted:
    /// the test over a read of the worker keeps the row exactly where the
    /// test of a row written with those bits keeps it.
    #[test]
    fn the_test_over_a_read_keeps_the_rows_the_test_of_a_written_row_keeps() {
        let row = Candidate {
            position: 3,
            id: WorkerId(7),
            has_capacity: true,
            has_room: false,
            ..Candidate::default()
        };
        let tried = [WorkerId(7)];
        let untried = [WorkerId(8)];
        let lifted = Constraints {
            overload_lifted: true,
            ..Constraints::default()
        };
        let mut kept = 0;
        for set in 0..32u8 {
            let tests = RowTests::of(&filters_of(set)).expect("the stock filters fold");
            for bits in 0u8..32 {
                let health = Health {
                    healthy: bits & 1 != 0,
                    breaker_open: bits & 2 != 0,
                    stalled: bits & 4 != 0,
                    draining: bits & 8 != 0,
                    overloaded: bits & 16 != 0,
                };
                let live = LiveRow::new(health, 5);
                let carried = Candidate {
                    health,
                    in_flight: 5,
                    ..row
                };
                for tried in [&tried[..], &untried[..], &[]] {
                    let facts = RequestFacts { tried, ..facts(1) };
                    for cx in [Constraints::default(), lifted] {
                        let by_read = tests.keeps_live(live, &facts, &row, &cx);
                        let by_row = tests.keeps(&facts, &carried, &cx);
                        assert_eq!(by_read, by_row, "set {set:#07b}, bits {bits:#07b}, {cx:?}");
                        // The whole pass asks the same test in two halves:
                        // the reader's, of the bits, and the row's own.
                        let pass = tests.live_test(&cx);
                        assert_eq!(pass.passes(health), pass.passes_row(live));
                        let halves = pass.passes(health) && tests.keeps_row(&facts, &row, &cx);
                        assert_eq!(halves, by_read, "set {set:#07b}, bits {bits:#07b}, {cx:?}");
                        assert!(tests.reads_rows() || tests.keeps_row(&facts, &row, &cx));
                        kept += usize::from(by_read);
                    }
                }
            }
        }
        // Both answers stand among the cases.
        assert!(kept > 0 && kept < 32 * 32 * 6);
    }

    /// The whole pass over live rows leaves the rows the pass over rows
    /// written with the same facts leaves, in the same order, and reads
    /// each worker once: with no row read before it, behind a draw that
    /// refused some rows and names them, and for a decision that names no
    /// row it read.
    #[test]
    fn the_pass_over_live_rows_leaves_the_written_rows_pass_survivors() {
        let live: Vec<LiveRow> = (0..40u32)
            .map(|at| {
                let health = Health {
                    healthy: at % 5 != 1,
                    breaker_open: at % 7 == 2,
                    stalled: at % 11 == 3,
                    draining: at % 13 == 4,
                    overloaded: at % 3 == 0,
                };
                LiveRow::new(health, at)
            })
            .collect();
        let bare = bare(40);
        let carried = written(&bare, &live);
        let rows = Rows::of(live);
        let tried = [bare[9].id, bare[22].id];
        let cx = Constraints::default();
        for set in 0..32u8 {
            let tests = RowTests::of(&filters_of(set)).expect("the stock filters fold");
            for tried in [&tried[..], &[]] {
                let facts = RequestFacts { tried, ..facts(1) };
                let tests = tests.asked(&facts, &cx);
                let (mut by_read, mut by_row) = (Vec::new(), Vec::new());
                let none = Some(&[][..]);
                let stands = keep_folded_live(tests, &rows, none, &facts, &bare, &cx, &mut by_read);
                assert_eq!(rows.next(), 40, "one read a worker");
                assert_eq!(
                    stands,
                    keep_folded(tests, &facts, &carried, &cx, &mut by_row)
                );
                assert_eq!(by_read, by_row, "set {set:#07b}");
                // A draw read the rows it refused, and names them.
                let refused: Vec<u32> = (0..12u32)
                    .filter(|&at| !tests.keeps(&facts, &carried[at as usize], &cx))
                    .collect();
                for &at in &refused {
                    let _ = rows.read(at);
                }
                let named = Some(&refused[..]);
                assert_eq!(
                    keep_folded_live(tests, &rows, named, &facts, &bare, &cx, &mut by_read),
                    stands
                );
                assert_eq!(rows.next(), 40, "one read a worker");
                assert_eq!(by_read, by_row, "set {set:#07b}, {refused:?} named");
                // A decision that read rows and names none.
                for at in 0..12u32 {
                    let _ = rows.read(at);
                }
                assert_eq!(
                    keep_folded_live(tests, &rows, None, &facts, &bare, &cx, &mut by_read),
                    stands
                );
                assert_eq!(rows.next(), 40, "one read a worker");
                assert_eq!(by_read, by_row, "set {set:#07b}, no row named");
            }
        }
    }

    /// Which presets a pool's own rows decide today: the two whose stages
    /// read no live fact off a row.
    #[test]
    fn the_presets_that_read_no_row_fact_decide_over_live_rows() {
        for (name, live) in [
            ("round_robin", true),
            ("random", true),
            ("passthrough", false),
            ("power_of_two", false),
            ("least_load", false),
            ("consistent_hashing", false),
            ("prefix_hash", false),
            ("manual", false),
            ("cache_aware", false),
            ("bucket", false),
        ] {
            let engine = preset(name);
            assert_eq!(engine.decides_live(MODEL, Phase::Single), live, "{name}");
        }
    }

    /// With every row ready round robin reads one worker a decision and
    /// turns one strict rotation at every pool size: each row is named once
    /// a turn of the pool.
    #[test]
    fn round_robin_reads_one_worker_a_decision_and_names_each_row_once_a_turn() {
        for size in [1u32, 2, 3, 8, 16, 64] {
            let engine = preset("round_robin");
            let bare = bare(size);
            let rows = Rows::of(vec![ready(); size as usize]);
            let mut scratch = Scratch::default();
            let mut turn = |seed| {
                let (decision, reads) = decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
                assert_eq!(reads, 1, "{size} rows");
                named(&decision.expect("live rows decide it")).expect("a row is named")
            };
            let first = turn(1);
            for step in 1..2 * size {
                assert_eq!(turn(u64::from(step) + 1), (first + step) % size);
            }
        }
    }

    /// Some rows are not ready: one in eight, then three in four, of a
    /// pool large enough to earn every turn ahead. No such row is named,
    /// and a turn that lands on one takes the pool's next turn. So the
    /// ready rows are named in the pool's order, each exactly once a turn
    /// of the ready rows, as the exact rotation among the ready rows names
    /// them. No decision reads the pool whole: it reads the worker of each
    /// turn it takes.
    #[test]
    fn a_turn_on_a_row_not_ready_takes_the_next_turn_and_the_shares_stay_exact() {
        const ROWS: u32 = 256;
        assert_eq!(live_turns(ROWS as usize), 9);
        for (ready_at, most) in [
            ((|at: u32| at % 8 != 3) as fn(u32) -> bool, 2),
            ((|at: u32| at % 4 == 1) as fn(u32) -> bool, 4),
        ] {
            let engine = preset("round_robin");
            let bare = bare(ROWS);
            let live = (0..ROWS).map(|at| if ready_at(at) { ready() } else { not_ready() });
            let rows = Rows::of(live.collect());
            let serving: Vec<u32> = (0..ROWS).filter(|&at| ready_at(at)).collect();
            let mut scratch = Scratch::default();
            let mut times = [0u32; ROWS as usize];
            let mut order = Vec::new();
            for seed in 1..=serving.len() as u64 * 25 {
                let (decision, reads) = decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
                let row = named(&decision.expect("live rows decide it")).expect("a row is named");
                assert!(reads <= most, "a decision read {reads} workers");
                times[row as usize] += 1;
                order.push(row);
            }
            for (at, &times) in (0u32..).zip(&times) {
                let expected = if ready_at(at) { 25 } else { 0 };
                assert_eq!(times, expected, "row {at}");
            }
            // The pool's order: each pick is the ready row behind the last.
            let place = |row| {
                serving
                    .iter()
                    .position(|&at| at == row)
                    .expect("a ready row")
            };
            for pair in order.windows(2) {
                assert_eq!(place(pair[1]), (place(pair[0]) + 1) % serving.len());
            }
        }
    }

    /// A pool that earns turns ahead, with a run of rows not ready longer
    /// than those turns: 64 rows, 16 in a row not ready, three turns a
    /// decision. A decision whose turns all land in the run reads the pool
    /// whole and takes the survivors' own turn, and every other is placed
    /// by the pool's turn. Both rotations are exact, so the shares are.
    /// Each turn of the pool places 53 requests: one on each of the 48
    /// ready rows by the pool's turn, and five by the survivors' rotation,
    /// whose turns were the first fifteen rows of the run. Over 48 turns of
    /// the pool the survivors' rotation comes round five times, so each
    /// ready row is named 53 times and no row of the run once.
    #[test]
    fn a_run_of_rows_not_ready_longer_than_the_turns_keeps_the_shares_exact() {
        const ROWS: u32 = 64;
        const RUN: std::ops::Range<u32> = 20..36;
        assert_eq!(live_turns(ROWS as usize), 3);
        let engine = preset("round_robin");
        let bare = bare(ROWS);
        let live = (0..ROWS).map(|at| {
            if RUN.contains(&at) {
                not_ready()
            } else {
                ready()
            }
        });
        let rows = Rows::of(live.collect());
        let mut scratch = Scratch::default();
        let mut seed = 0;
        let mut next = || {
            seed += 1;
            let (decision, reads) = decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
            let row = named(&decision.expect("live rows decide it")).expect("a row is named");
            (row, reads)
        };
        // Up to the pool's turn on the row in front of the run: the pool's
        // next turn enters the run.
        while next() != (RUN.start - 1, 1) {}
        let mut times = [0u32; ROWS as usize];
        let (mut by_turn, mut by_pass) = (0u32, 0u32);
        for _ in 0..48 * 53 {
            let (row, reads) = next();
            times[row as usize] += 1;
            // A decision that reads the pool whole reads every worker once:
            // its turns' three, and the pass's others.
            if reads == ROWS as usize {
                by_pass += 1;
            } else {
                assert!(
                    reads <= 2,
                    "a decision placed by a turn read {reads} workers"
                );
                by_turn += 1;
            }
        }
        assert_eq!((by_turn, by_pass), (48 * 48, 48 * 5));
        for (at, &times) in (0u32..).zip(&times) {
            let expected = if RUN.contains(&at) { 0 } else { 53 };
            assert_eq!(times, expected, "row {at}");
        }
    }

    /// The turns a pool's size earns: a small pool goes to the whole pass
    /// at once, as the survivors' rotation always did, and a larger pool
    /// takes one more turn for each 32 rows, eight more at the most.
    #[test]
    fn a_pool_earns_its_turns_by_its_size() {
        for (rows, turns) in [
            (1, 1),
            (8, 1),
            (31, 1),
            (32, 2),
            (64, 3),
            (128, 5),
            (256, 9),
        ] {
            assert_eq!(live_turns(rows), turns, "{rows} rows");
        }
        assert_eq!(live_turns(4_096), 9);
        // A small pool with a row not ready: the turn that lands on it
        // reads the pool whole, and every ready row is named evenly.
        let engine = preset("round_robin");
        let bare = bare(8);
        let live = (0..8u32).map(|at| if at == 5 { not_ready() } else { ready() });
        let rows = Rows::of(live.collect());
        let mut scratch = Scratch::default();
        let mut times = [0u32; 8];
        for seed in 1..=7 * 8 * 20 {
            let (decision, reads) = decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
            let row = named(&decision.expect("live rows decide it")).expect("a row is named");
            assert!(reads == 1 || reads == 8, "{reads} reads");
            times[row as usize] += 1;
        }
        assert_eq!(times[5], 0);
        let ready_times: Vec<u32> = (0..8).filter(|&at| at != 5).map(|at| times[at]).collect();
        let (least, most) = (ready_times.iter().min(), ready_times.iter().max());
        assert_eq!(least, most, "{ready_times:?}");
    }

    /// Past a run of refused rows longer than the turns a decision takes
    /// the decision reads the pool whole, and the survivors' own rotation
    /// decides: one ready row among 64 is named by every decision. A
    /// decision whose turns reach the row reads the workers of those turns,
    /// and any other reads each worker of the pool once: the turns' reads
    /// are kept, and the pass reads the rest.
    #[test]
    fn a_pool_the_turns_miss_is_read_whole_and_no_worker_twice() {
        let engine = preset("round_robin");
        let bare = bare(64);
        let mut live = vec![not_ready(); 64];
        live[41] = ready();
        let rows = Rows::of(live);
        let mut scratch = Scratch::default();
        let turns = live_turns(64);
        let (mut whole, mut by_turn) = (0, 0);
        for seed in 1..=128 {
            let (decision, reads) = decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
            assert_eq!(named(&decision.expect("live rows decide it")), Some(41));
            if reads == 64 {
                whole += 1;
            } else {
                assert!(reads <= turns, "{reads} reads");
                by_turn += 1;
            }
        }
        assert!(whole > 0 && by_turn > 0, "{whole} passes, {by_turn} turns");
    }

    /// A random pick over live rows names no row that is not ready, names
    /// every ready row, and reads the rows it draws alone.
    #[test]
    fn random_draws_over_the_ready_rows_alone() {
        const ROWS: u32 = 64;
        const DECISIONS: u64 = 56_000;
        let engine = preset("random");
        let bare = bare(ROWS);
        let ready_at = |at: u32| at % 8 != 3;
        let live = (0..ROWS).map(|at| if ready_at(at) { ready() } else { not_ready() });
        let rows = Rows::of(live.collect());
        let mut scratch = Scratch::default();
        let mut times = [0u64; ROWS as usize];
        for seed in 1..=DECISIONS {
            let (decision, reads) = decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
            let row = named(&decision.expect("live rows decide it")).expect("a row is named");
            times[row as usize] += 1;
            assert!(reads <= LIVE_DRAWS as usize);
        }
        let even = DECISIONS as f64 / 56.0;
        for (at, &times) in (0u32..).zip(&times) {
            if !ready_at(at) {
                assert_eq!(times, 0, "row {at} is not ready");
                continue;
            }
            let off = (times as f64 - even).abs() / even;
            assert!(off < 0.2, "row {at} took {times} of an even {even:.0}");
        }
    }

    /// A pool none of whose rows passes every filter is handed back: what
    /// decides such a pool reads a row's facts off the row, so the caller
    /// writes rows for it. Each worker was read once by then: the rows the
    /// draw landed on by the draw, the others by the pass. So with a pool
    /// nobody serves, a pool over its thresholds on every row and a pool
    /// the liveness doubt alone refuses.
    #[test]
    fn a_pool_every_row_of_which_is_refused_is_handed_back_after_one_read_a_worker() {
        for name in ["round_robin", "random"] {
            let engine = preset(name);
            let bare = bare(24);
            let mut scratch = Scratch::default();
            let doubted = Health {
                stalled: true,
                ..READY
            };
            let pools = [
                ("nobody serves", vec![not_ready(); 24]),
                ("over its thresholds", (0..24).map(overloaded).collect()),
                ("doubted", vec![LiveRow::new(doubted, 3); 24]),
            ];
            for (pool, live) in pools {
                let rows = Rows::of(live);
                for seed in 1..=16 {
                    let (decision, reads) =
                        decide(&engine, &rows, &bare, &facts(seed), &mut scratch);
                    assert!(decision.is_none(), "{name}, {pool}: handed back");
                    assert_eq!(reads, 24, "{name}, {pool}: one read a worker");
                }
            }
        }
    }

    /// What live rows do not decide is handed back: a profile whose stages
    /// read a row's facts off the row, and no worker is read for it. A
    /// model no binding matches is declined at once, and no worker is read
    /// for it either.
    #[test]
    fn a_decision_live_rows_do_not_make_is_handed_back() {
        let bare = bare(8);
        let mut scratch = Scratch::default();
        let rows = Rows::of(vec![ready(); 8]);
        let by_load = preset("least_load");
        let (asked, reads) = decide(&by_load, &rows, &bare, &facts(1), &mut scratch);
        assert!(asked.is_none(), "least load reads a row's reports");
        assert_eq!(reads, 0);

        let mut bound = PolicySpec::preset("round_robin").expect("a built-in preset");
        bound.generation = 1;
        for binding in &mut bound.bindings {
            binding.matcher.model = "another-model".to_owned();
        }
        let bound = bound.validate().expect("the document validates");
        let (declined, reads) = decide(&bound, &rows, &bare, &facts(1), &mut scratch);
        let Some(Decision::Shed(shed)) = declined else {
            panic!("a model no binding matches is declined");
        };
        assert_eq!(
            (shed.reason, shed.emptied_by),
            (ShedReason::PolicyDeclined, "binding")
        );
        assert_eq!(reads, 0);
        assert!(bound.decides_live(MODEL, Phase::Single));
    }

    /// A retry names the rows it tried and is placed on another while one
    /// stands. Where it tried every row the test keeps nobody and the
    /// decision is handed back: over rows written for it the soft filter
    /// steps aside, and a tried row takes the request.
    #[test]
    fn a_retry_over_live_rows_is_placed_on_a_row_it_did_not_try() {
        let engine = preset("round_robin");
        let bare = bare(3);
        let rows = Rows::of(vec![ready(); 3]);
        let mut scratch = Scratch::default();
        let some = [bare[0].id, bare[2].id];
        let every = [bare[0].id, bare[1].id, bare[2].id];
        for seed in 1..=12 {
            let retry = RequestFacts {
                tried: &some,
                ..facts(seed)
            };
            let (decision, _) = decide(&engine, &rows, &bare, &retry, &mut scratch);
            assert_eq!(named(&decision.expect("live rows decide it")), Some(1));
            let last = RequestFacts {
                tried: &every,
                ..facts(seed)
            };
            let (decision, reads) = decide(&engine, &rows, &bare, &last, &mut scratch);
            assert!(decision.is_none(), "every row is tried: handed back");
            assert_eq!(reads, 3, "one read a worker");
        }
    }
}
