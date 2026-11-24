//! The filters written for no one policy: each keeps a candidate row or
//! refuses it. None allocates per decision.

use std::{any::Any, sync::Arc};

use crate::{
    document::spec::Needs,
    engine::{
        framework::Filter,
        view::{Candidate, Constraints, LoadReportCutoff, LoadRow, Phase, RequestFacts},
    },
    load::signals::{SignalColumn, SignalSums, SignalTerm},
    stages::cost_terms::{Coefficients, CostTerm},
    tracking::dispatch_credit::{DispatchCounts, DispatchCredit},
};

/// Ready, breaker closed, not stalled, not draining.
#[derive(Debug)]
pub struct Healthy;

impl Filter for Healthy {
    fn name(&self) -> &'static str {
        "healthy"
    }

    // Refuses by a row's health bits and its host's mask.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        c.health.routable_ignoring_overload() && !cx.masks(c.position)
    }

    /// The host's mask is asked about once: a pass under no standing veto
    /// pays one test for it, and is then the pass over the rows' own bits.
    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        if !cx.stalled.is_empty() {
            return retain_masked(candidates, cx, kept);
        }
        OwnBits.retain(facts, candidates, cx, kept);
    }
}

/// The health filter over a row's own bits alone. A filter of its own, so
/// that the pass under no standing veto is the loop the trait gives a
/// filter: written out beside the masked pass, the same test compiled to
/// two instructions a row more past the first refused row.
#[derive(Debug)]
struct OwnBits;

impl Filter for OwnBits {
    fn name(&self) -> &'static str {
        "healthy"
    }

    // Refuses by a row's health bits.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        c.health.routable_ignoring_overload()
    }
}

/// The health filter's pass under a standing veto: a row's own bits, then
/// the host's mask. Out of line, so the pass under no veto stays the short
/// one.
#[inline(never)]
fn retain_masked(candidates: &[Candidate], cx: &Constraints<'_>, kept: &mut Vec<u32>) {
    kept.retain(|&i| candidates[i as usize].health.routable_ignoring_overload() && !cx.masks(i));
}

/// Not over the overload thresholds, unless the host lifted the veto.
#[derive(Debug)]
pub struct NotOverloaded;

impl Filter for NotOverloaded {
    fn name(&self) -> &'static str {
        "not-overloaded"
    }

    // Refuses by the overload bit of a row's health, unless its host lifts it.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        cx.overload_lifted || !c.health.overloaded
    }
}

/// The prefill capacity window has a slot (a host-supplied bit).
#[derive(Debug)]
pub struct HasCapacity;

impl Filter for HasCapacity {
    fn name(&self) -> &'static str {
        "has-capacity"
    }

    // Refuses by a bit its host supplies.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        c.has_capacity
    }
}

/// The decode's running window has a free room (a host-supplied bit). Soft:
/// with every candidate full the list stands and the dispatch waits at its
/// pick's gate, as today.
#[derive(Debug)]
pub struct HasRoom;

impl Filter for HasRoom {
    fn name(&self) -> &'static str {
        "has-room"
    }

    // Refuses by a bit its host supplies.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        c.has_room
    }

    fn soft(&self) -> bool {
        true
    }
}

/// Asks `$rule.$method` with the three flags of a report's sum
/// ([`Coefficients::add_report`]) settled from the names the rule reads. A
/// flag left on for a name it does not read would drop a row for a value
/// nobody named. With a stage named behind the rule, the stage is asked
/// under the rule's flags: a stage that wraps the rule.
macro_rules! by_names {
    ($rule:expr, $method:ident($($arg:expr),*)) => {
        by_names!($rule, $rule, $method($($arg),*))
    };
    ($rule:expr, $stage:expr, $method:ident($($arg:expr),*)) => {
        match ($rule.resident, $rule.waiting, $rule.kv_or_running) {
            (false, false, false) => $stage.$method::<false, false, false>($($arg),*),
            (false, false, true) => $stage.$method::<false, false, true>($($arg),*),
            (false, true, false) => $stage.$method::<false, true, false>($($arg),*),
            (false, true, true) => $stage.$method::<false, true, true>($($arg),*),
            (true, false, false) => $stage.$method::<true, false, false>($($arg),*),
            (true, false, true) => $stage.$method::<true, false, true>($($arg),*),
            (true, true, false) => $stage.$method::<true, true, false>($($arg),*),
            (true, true, true) => $stage.$method::<true, true, true>($($arg),*),
        }
    };
}

/// `has-room` by a rule the document writes: a row has room where what it
/// holds is at most its capacity less the headroom. What a row holds is the
/// sum of the rule's named fields times their coefficients, in name order.
/// A row that lacks a named value has no room. Soft and row-wise, as the
/// host's bit is: where no row has room the list stands.
#[derive(Debug)]
pub(crate) struct HasRoomRule {
    terms: Coefficients,
    /// The coefficient of the request's size: zero where the rule does not
    /// name it.
    request_tokens: f64,
    /// The coefficient of the request's output limit: zero where the rule
    /// does not name it.
    output_tokens: f64,
    /// The capacity less the headroom.
    bound: f64,
    /// Whether a term reads the row's count in flight.
    in_flight: bool,
    /// Whether a term reads the row's report.
    loads: bool,
    // The optional values of a report, each tested where the rule names it.
    resident: bool,
    waiting: bool,
    kv_or_running: bool,
    /// The profile's count of sends, where a term reads it.
    sent: Option<Arc<DispatchCredit>>,
}

impl HasRoomRule {
    pub(crate) const NAME: &str = "has-room";
    /// The request's size among the rule's fields. Every request states it.
    pub(crate) const REQUEST_TOKENS: &str = "request.prompt_tokens";
    /// The request's output limit among the rule's fields: the most tokens
    /// its answer may hold. A request that states none reads zero.
    pub(crate) const OUTPUT_TOKENS: &str = "request.max_output_tokens";

    pub(crate) fn new(
        terms: &[(CostTerm, f64)],
        request_tokens: f64,
        output_tokens: f64,
        capacity: f64,
        headroom: f64,
        sent: Option<Arc<DispatchCredit>>,
    ) -> Self {
        let coefficients = Coefficients::of(terms);
        Self {
            request_tokens,
            output_tokens,
            bound: (1.0 - headroom) * capacity,
            in_flight: coefficients.in_flight != 0.0,
            loads: terms.iter().any(|&(term, _)| term.reads_load()),
            resident: coefficients.resident_tokens != 0.0,
            waiting: coefficients.waiting_uncached_tokens != 0.0,
            kv_or_running: coefficients.names_kv_or_running(),
            terms: coefficients,
            sent,
        }
    }

    /// The request's part of what a row holds, in name order: its output
    /// limit times the rule's coefficient for it, then its size times the
    /// size's. Under a coefficient above zero the limit's product reserves
    /// room for the answer ahead of its first token. A request that states
    /// no limit reserves nothing, and under a rule that names no limit the
    /// product is zero: the sum equals the size's product, so such a rule
    /// keeps the rows it kept. Every row of a pass is handed the same, so a
    /// pass works it once.
    #[inline(always)]
    fn asked(&self, facts: &RequestFacts<'_>) -> f64 {
        f64::from(facts.max_output_tokens) * self.output_tokens
            + f64::from(facts.prompt_size()) * self.request_tokens
    }

    /// Whether a row has room, by its report as `load` hands it and its
    /// sends as `sent` counts them: the requests and their tokens, stated
    /// or estimated ([`DispatchCreditPass::counts`]), or `None` where the
    /// rule reads no count. The sum runs in name order: the count in
    /// flight, the report's fields, the request's part as `asked`
    /// ([`Self::asked`]), the sends. `within` says whether that sum is
    /// within the row's bound: the rule's own test, or the signal form's
    /// behind its signals.
    ///
    /// [`DispatchCreditPass::counts`]: crate::tracking::dispatch_credit::DispatchCreditPass::counts
    #[inline(always)]
    fn fits_with<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        asked: f64,
        facts: &RequestFacts<'_>,
        row: &Candidate,
        load: Option<&LoadRow>,
        sent: impl FnOnce() -> Option<DispatchCounts>,
        within: impl FnOnce(f64) -> bool,
    ) -> bool {
        let mut used = 0.0;
        if self.in_flight {
            used += f64::from(facts.in_flight(row)) * self.terms.in_flight;
        }
        if self.loads {
            // A row with no report, or with one that lacks a named value,
            // has no room.
            let Some(sum) = load.and_then(|report| {
                self.terms
                    .add_report::<RESIDENT, WAITING, KV_OR_RUNNING>(used, report, true)
            }) else {
                return false;
            };
            used = sum;
        }
        used += asked;
        if let Some(sent) = sent() {
            self.terms.add_sent(&mut used, sent);
        }
        within(used)
    }

    /// [`Self::fits_with`] under the rule's own bound.
    #[inline(always)]
    fn fits<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        asked: f64,
        facts: &RequestFacts<'_>,
        row: &Candidate,
        load: Option<&LoadRow>,
        sent: impl FnOnce() -> Option<DispatchCounts>,
    ) -> bool {
        self.fits_with::<RESIDENT, WAITING, KV_OR_RUNNING>(asked, facts, row, load, sent, |used| {
            used <= self.bound
        })
    }

    /// [`Filter::keep`] of one row, by the report `load` hands.
    fn keep_with(&self, facts: &RequestFacts<'_>, row: &Candidate, load: Option<&LoadRow>) -> bool {
        // One row's word reads the count as it stands, behind the rule's
        // flag, so the two numbers come back in registers. Mapped straight
        // into the `Option`, this closure hands the pair back through
        // memory: 12 instructions more for each row drawn.
        let sent = || {
            self.sent
                .as_ref()
                .map_or_else(DispatchCounts::default, |sent| sent.pass(facts).counts(row))
        };
        let asked = self.asked(facts);
        by_names!(
            self,
            fits(asked, facts, row, load, || self.sent.is_some().then(sent))
        )
    }

    /// One pass over the kept rows, in one loop with the test in it: the
    /// request's part is worked once, and the profile's sends are read
    /// through one load of its tally. The rows that fit are written over
    /// the list's front, in the list's order.
    #[inline(always)]
    fn rows<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &mut Vec<u32>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) {
        let asked = self.asked(facts);
        // A count that holds no row's entry adds nothing to any sum, each
        // coefficient being finite: the pass reads no count then.
        let sends = self
            .sent
            .as_ref()
            .map(|sent| sent.pass(facts))
            .filter(|sends| sends.holds_any());
        let mut held = 0;
        for at in 0..kept.len() {
            let i = kept[at];
            let row = &candidates[i as usize];
            let sent = || sends.as_ref().map(|sends| sends.counts(row));
            let fits =
                self.fits::<RESIDENT, WAITING, KV_OR_RUNNING>(asked, facts, row, load(row), sent);
            kept[held] = i;
            held += usize::from(fits);
        }
        kept.truncate(held);
    }

    /// [`Filter::retain`], by the report `load` hands of each row: the
    /// flags are settled once a pass.
    fn retain_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &mut Vec<u32>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) {
        by_names!(self, rows(facts, candidates, kept, load));
    }
}

impl Filter for HasRoomRule {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    // Soft: it turns no request away, so a late report can only place a
    // request elsewhere.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn soft(&self) -> bool {
        true
    }

    // A report ends the sends before it, so the sends need the reports
    // though no term may read one. A rule that names the request's size
    // needs it counted, and so does one that prices the sends by their
    // tokens: a send of text alone is counted by its text's bytes. The
    // output limit asks nothing of a host: a request states it, or zero
    // for none.
    fn needs(&self) -> Needs {
        Needs {
            loads: self.loads || self.sent.is_some(),
            prompt_size: self.request_tokens != 0.0 || self.terms.sizes_sends(),
            ..Needs::default()
        }
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        self.keep_with(facts, c, c.load.as_ref())
    }

    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        self.retain_with(facts, candidates, kept, |row| row.load.as_ref());
    }
}

/// [`HasRoomRule`] under the view's report age limit: a report past the
/// limit reads as none, by the cutoff the scorers read under.
#[derive(Debug)]
struct HasRoomWithin {
    rule: HasRoomRule,
    max_age_ms: u64,
}

impl Filter for HasRoomWithin {
    fn name(&self) -> &'static str {
        self.rule.name()
    }

    fn refuses_by_reports(&self) -> bool {
        self.rule.refuses_by_reports()
    }

    fn row_wise(&self) -> bool {
        self.rule.row_wise()
    }

    fn soft(&self) -> bool {
        self.rule.soft()
    }

    fn needs(&self) -> Needs {
        self.rule.needs()
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        let cutoff = LoadReportCutoff::new(facts.now_micros, self.max_age_ms);
        self.rule.keep_with(facts, c, cutoff.load(c))
    }

    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        let cutoff = LoadReportCutoff::new(facts.now_micros, self.max_age_ms);
        self.rule
            .retain_with(facts, candidates, kept, |row| cutoff.load(row));
    }
}

/// The rule as a profile's filter: under the view's report age limit where
/// the document sets one and a term reads a report, and as it stands
/// otherwise. So a document with no limit runs no test of an age.
pub(crate) fn has_room_rule(rule: HasRoomRule, max_age_ms: Option<u64>) -> Box<dyn Filter> {
    match max_age_ms.filter(|_| rule.loads) {
        Some(max_age_ms) => Box::new(HasRoomWithin { rule, max_age_ms }),
        None => Box::new(rule),
    }
}

/// [`HasRoomRule`] where the document names a signal: in the rule's sum, as
/// its capacity, or both. A type of its own, chosen where the stage is
/// built, so a rule that names no signal runs the pass that reads none.
///
/// The signals close the sum, in name order. A row with no fresh value for
/// one of them has no room. Under a signal's capacity the bound is worked
/// a row: one less the headroom, times the row's capacity. A row whose
/// capacity has no fresh value has no room, and neither has one whose
/// capacity reads zero or less.
///
/// The view's report age limit reaches the report's fields, as it does in
/// [`HasRoomWithin`]. A signal's maximum age reaches that signal alone.
#[derive(Debug)]
pub(crate) struct HasRoomSignals {
    /// The rule's other terms. Under a signal's capacity its bound is one
    /// less the headroom: the bound of a capacity of one.
    rule: HasRoomRule,
    /// The signals the rule sums, in name order.
    signals: Box<[SignalTerm]>,
    /// The capacity's signal, where the rule names one.
    capacity: Option<SignalColumn>,
    /// The view's report age limit, where the document sets one and a term
    /// reads a report.
    max_age_ms: Option<u64>,
}

impl HasRoomSignals {
    pub(crate) fn new(
        rule: HasRoomRule,
        signals: Box<[SignalTerm]>,
        capacity: Option<SignalColumn>,
        max_age_ms: Option<u64>,
    ) -> Self {
        Self {
            max_age_ms: max_age_ms.filter(|_| rule.loads),
            rule,
            signals,
            capacity,
        }
    }

    /// Whether `used` is within the row's bound. `capacity` reads the
    /// row's capacity under the signal that holds it.
    #[inline(always)]
    fn within(&self, used: f64, capacity: impl FnOnce(&SignalColumn) -> Option<f64>) -> bool {
        match &self.capacity {
            // A capacity of zero or less holds nothing, by a test of its
            // own: a plain compare would pass a row that holds nothing.
            Some(signal) => capacity(signal)
                .is_some_and(|capacity| capacity > 0.0 && used <= self.rule.bound * capacity),
            None => used <= self.rule.bound,
        }
    }

    /// [`Filter::keep`] of one row, by the report `load` hands. A drawn
    /// row reads each signal through [`SignalColumn::read`].
    fn keep_with(&self, facts: &RequestFacts<'_>, row: &Candidate, load: Option<&LoadRow>) -> bool {
        let rule = &self.rule;
        // The count is read behind the rule's flag, as
        // [`HasRoomRule::keep_with`] reads it and for its reason.
        let sent = || {
            rule.sent
                .as_ref()
                .map_or_else(DispatchCounts::default, |sent| sent.pass(facts).counts(row))
        };
        let asked = rule.asked(facts);
        by_names!(
            rule,
            fits_with(
                asked,
                facts,
                row,
                load,
                || rule.sent.is_some().then(sent),
                |mut used| {
                    for (signal, coefficient) in &self.signals {
                        let Some(value) = signal.read(facts, row) else {
                            return false;
                        };
                        used += value * coefficient;
                    }
                    self.within(used, |capacity| capacity.read(facts, row))
                }
            )
        )
    }

    /// One pass over the kept rows, as [`HasRoomRule::rows`] makes it. A
    /// count that holds no row's entry is not read, as there. The pass is
    /// made once for a count to read and once for none, so a row tests
    /// neither.
    #[inline(always)]
    fn rows<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &mut Vec<u32>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) {
        let sends = self
            .rule
            .sent
            .as_ref()
            .map(|sent| sent.pass(facts))
            .filter(|sends| sends.holds_any());
        match &sends {
            Some(sends) => self.rows_by::<RESIDENT, WAITING, KV_OR_RUNNING>(
                facts,
                candidates,
                kept,
                load,
                |row| Some(sends.counts(row)),
            ),
            None => self.rows_by::<RESIDENT, WAITING, KV_OR_RUNNING>(
                facts,
                candidates,
                kept,
                load,
                |_| None,
            ),
        }
    }

    /// [`Self::rows`] by the sends `sent` counts of each row. The facts'
    /// cells and each signal's oldest instant still read are taken once a
    /// pass, and a row's cells once a row.
    #[inline(always)]
    fn rows_by<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &mut Vec<u32>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
        sent: impl Fn(&Candidate) -> Option<DispatchCounts>,
    ) {
        let rule = &self.rule;
        let asked = rule.asked(facts);
        let sums = SignalSums::new(facts, &self.signals);
        let capacity_oldest = self
            .capacity
            .as_ref()
            .map_or(0, |capacity| capacity.oldest(facts));
        let mut held = 0;
        for at in 0..kept.len() {
            let i = kept[at];
            let row = &candidates[i as usize];
            let fits = rule.fits_with::<RESIDENT, WAITING, KV_OR_RUNNING>(
                asked,
                facts,
                row,
                load(row),
                || sent(row),
                |used| {
                    let cells = sums.row(row);
                    let (used, whole) = sums.add(used, cells);
                    whole && self.within(used, |capacity| capacity.among(cells?, capacity_oldest))
                },
            );
            kept[held] = i;
            held += usize::from(fits);
        }
        kept.truncate(held);
    }

    /// [`Filter::retain`], by the report `load` hands of each row.
    fn retain_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &mut Vec<u32>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) {
        by_names!(self.rule, self, rows(facts, candidates, kept, load));
    }
}

impl Filter for HasRoomSignals {
    fn name(&self) -> &'static str {
        self.rule.name()
    }

    fn refuses_by_reports(&self) -> bool {
        self.rule.refuses_by_reports()
    }

    fn row_wise(&self) -> bool {
        self.rule.row_wise()
    }

    fn soft(&self) -> bool {
        self.rule.soft()
    }

    // The cells stand beside the rows of a profile whose stage says it
    // reads a signal. A signal alone asks for no load report.
    fn needs(&self) -> Needs {
        Needs {
            signal_cells: true,
            ..self.rule.needs()
        }
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        match self.max_age_ms {
            Some(max_age_ms) => {
                let cutoff = LoadReportCutoff::new(facts.now_micros, max_age_ms);
                self.keep_with(facts, c, cutoff.load(c))
            }
            None => self.keep_with(facts, c, c.load.as_ref()),
        }
    }

    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        match self.max_age_ms {
            Some(max_age_ms) => {
                let cutoff = LoadReportCutoff::new(facts.now_micros, max_age_ms);
                self.retain_with(facts, candidates, kept, |row| cutoff.load(row));
            }
            None => self.retain_with(facts, candidates, kept, |row| row.load.as_ref()),
        }
    }
}

/// Not tried by an earlier attempt and not excluded by the host. Soft: the
/// engines a retry already tried step aside while another remains.
#[derive(Debug)]
pub struct NotTried;

impl NotTried {
    /// Whether the request names no row tried and its host none excluded:
    /// the filter then keeps every row. The one test behind both
    /// shortcuts that read it: the filter's own pass and the fold's set
    /// ([`RowTests::asked`]).
    #[inline(always)]
    fn names_none(facts: &RequestFacts<'_>, cx: &Constraints<'_>) -> bool {
        facts.tried.is_empty() && cx.excluded.is_empty()
    }
}

impl Filter for NotTried {
    fn name(&self) -> &'static str {
        "not-tried"
    }

    // Refuses by the request's own tried list and its host's exclusions.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        !facts.tried.contains(&c.id) && !cx.excluded.contains(&c.id)
    }

    fn retain(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        cx: &Constraints<'_>,
        kept: &mut Vec<u32>,
    ) {
        if Self::names_none(facts, cx) {
            return;
        }
        kept.retain(|&i| self.keep(facts, &candidates[i as usize], cx));
    }

    fn soft(&self) -> bool {
        true
    }
}

/// The filters of this file as one test of a row: the set of them a
/// profile holds. The engine folds a profile made of these filters alone
/// into one at its build ([`RowTests::of`]), and a decision that draws first
/// asks it about each row drawn: one call a row, where the filters' own
/// `keep`s are a call a filter. A whole pass asks it about every row of
/// the single plan's pool, in one loop where the filters are a loop each.
///
/// The set is the word a row is tested by. Each filter that reads a row's
/// own bits names the bytes of the row's word ([`row_word`]) it refuses a
/// row by, so the set asks all of them of a row in one compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowTests(u64);

impl RowTests {
    /// Not healthy, the breaker open, stalled or draining: a byte each.
    const HEALTHY: u64 = 0x0101_0101;
    const NOT_OVERLOADED: u64 = 1 << 32;
    const HAS_CAPACITY: u64 = 1 << 48;
    const HAS_ROOM: u64 = 1 << 56;
    /// A bit no row's word holds: this test reads the request's lists.
    const NOT_TRIED: u64 = 1 << 1;
    /// The bits of a row's word a row is kept by: a filter's other bits
    /// refuse.
    const KEPT_BY: u64 = 1 | Self::HAS_CAPACITY | Self::HAS_ROOM;

    /// `filters` as one set, where each is a filter of this file, known by
    /// its type. A host's own filter is no test the engine holds, one that
    /// wraps a filter of this file too: a profile with one folds into
    /// nothing, and is asked through each filter's `keep`.
    pub(crate) fn of(filters: &[Box<dyn Filter>]) -> Option<Self> {
        let mut tests = 0;
        for filter in filters {
            let filter: &dyn Any = filter.as_ref();
            tests |= if filter.is::<Healthy>() {
                Self::HEALTHY
            } else if filter.is::<NotOverloaded>() {
                Self::NOT_OVERLOADED
            } else if filter.is::<HasCapacity>() {
                Self::HAS_CAPACITY
            } else if filter.is::<HasRoom>() {
                Self::HAS_ROOM
            } else if filter.is::<NotTried>() {
                Self::NOT_TRIED
            } else {
                return None;
            };
        }
        Some(Self(tests))
    }

    /// The empty set.
    pub(crate) const fn none() -> Self {
        Self(0)
    }

    /// The set with `other`'s tests added.
    pub(crate) const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// The tests a placement pass that asks for one pick holds itself at a
    /// place of `phase`, so that the pick leaves them to the pass: the
    /// test of the workers a retry tried at every place, the prefill
    /// window's at a prefill place and the decode rooms' at a decode
    /// place. The two windows are bits a host supplies, and such a pass
    /// is the host: it narrows a side's candidates by them before it asks.
    /// At an encode place no pass narrows by the workers a retry tried,
    /// and none has to: a retry's list names the prefill and the decode it
    /// tried and never an encoder, so that test passes every candidate
    /// there.
    pub(crate) const fn held_by_a_pass(phase: Phase) -> Self {
        let side = match phase {
            Phase::Prefill => Self::HAS_CAPACITY,
            Phase::Decode => Self::HAS_ROOM,
            Phase::Single | Phase::Encode => 0,
        };
        Self(Self::NOT_TRIED | side)
    }

    /// Whether every test of the set is a test of a row's health bits or
    /// one of `held`.
    pub(crate) const fn health_or(self, held: Self) -> bool {
        self.0 & !(Self::HEALTHY | Self::NOT_OVERLOADED | held.0) == 0
    }

    /// The test of a row's health bits the set leaves out, by its filter's
    /// name. A caller that tests a worker as the worker tests itself for
    /// routing asks both, so it would add the one left out.
    pub(crate) fn health_test_left_out(self) -> Option<&'static str> {
        if self.0 & Self::HEALTHY == 0 {
            Some(Healthy.name())
        } else if self.0 & Self::NOT_OVERLOADED == 0 {
            Some(NotOverloaded.name())
        } else {
            None
        }
    }

    /// The set as one decision asks it: without the overload test where
    /// the host lifted that veto, and without the tried test where the
    /// request names no row tried and the host none excluded. A test left
    /// out keeps every row under these facts, so the set that is left
    /// keeps the rows the whole set keeps, at fewer tests a row.
    pub(crate) fn asked(self, facts: &RequestFacts<'_>, cx: &Constraints<'_>) -> Self {
        let mut tests = self.0;
        if cx.overload_lifted {
            tests &= !Self::NOT_OVERLOADED;
        }
        if NotTried::names_none(facts, cx) {
            tests &= !Self::NOT_TRIED;
        }
        Self(tests)
    }

    /// Whether a decision under `cx` asks the host's mask about a row
    /// beside [`RowTests::keeps`]: the health filter is in the set and a
    /// veto stands.
    pub(crate) fn masked(self, cx: &Constraints<'_>) -> bool {
        self.0 & Self::HEALTHY != 0 && !cx.stalled.is_empty()
    }

    /// Whether every filter of the set keeps the row, as each one's own
    /// `keep` answers. The four that read a row's own bits are asked in
    /// one compare of the row's word with the set, whatever the set holds.
    /// The health filter answers here for the row's own bits: the
    /// host's mask, its other half, is the caller's to ask where
    /// [`RowTests::masked`] says a veto stands.
    #[inline(always)]
    pub(crate) fn keeps(
        self,
        facts: &RequestFacts<'_>,
        c: &Candidate,
        cx: &Constraints<'_>,
    ) -> bool {
        // The overload test asks nothing where the host lifted that veto.
        let lifted = if cx.overload_lifted {
            Self::NOT_OVERLOADED
        } else {
            0
        };
        (row_word(c) ^ Self::KEPT_BY) & self.0 & !lifted == 0
            && (self.0 & Self::NOT_TRIED == 0 || NotTried.keep(facts, c, cx))
    }
}

/// The bits of a row that four filters of this file read, as one word: a
/// byte a bit, in the row's own order. The byte between them that none of
/// the four reads is read with them: the eight are one load.
#[inline(always)]
fn row_word(c: &Candidate) -> u64 {
    u64::from(c.health.healthy)
        | u64::from(c.health.breaker_open) << 8
        | u64::from(c.health.stalled) << 16
        | u64::from(c.health.draining) << 24
        | u64::from(c.health.overloaded) << 32
        | u64::from(c.session_hit) << 40
        | u64::from(c.has_capacity) << 48
        | u64::from(c.has_room) << 56
}

/// The five filters of this file a set names, by bit: what the cases of the
/// folded test walk, here and in the engine's.
#[cfg(test)]
pub(crate) fn filters_of(set: u8) -> Vec<Box<dyn Filter>> {
    let all: [fn() -> Box<dyn Filter>; 5] = [
        || Box::new(Healthy),
        || Box::new(NotOverloaded),
        || Box::new(HasCapacity),
        || Box::new(HasRoom),
        || Box::new(NotTried),
    ];
    (0..all.len())
        .filter(|bit| set & (1 << bit) != 0)
        .map(|bit| all[bit]())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::view::Health,
        stages::scorers::tests::{candidate, kept},
    };

    #[test]
    fn the_filters_read_their_bits() {
        let facts = RequestFacts::default();
        let cx = Constraints::default();
        let mut c = candidate(0, 0);
        assert!(Healthy.keep(&facts, &c, &cx));
        for down in [
            Health {
                healthy: false,
                ..c.health
            },
            Health {
                breaker_open: true,
                ..c.health
            },
            Health {
                stalled: true,
                ..c.health
            },
            Health {
                draining: true,
                ..c.health
            },
        ] {
            let unroutable = Candidate { health: down, ..c };
            assert!(!Healthy.keep(&facts, &unroutable, &cx));
        }
        assert!(NotOverloaded.keep(&facts, &c, &cx));
        c.health.overloaded = true;
        assert!(!NotOverloaded.keep(&facts, &c, &cx));
        let lifted = Constraints {
            overload_lifted: true,
            ..cx
        };
        assert!(NotOverloaded.keep(&facts, &c, &lifted));
        assert!(!HasCapacity.keep(&facts, &c, &cx));
        c.has_capacity = true;
        assert!(HasCapacity.keep(&facts, &c, &cx));
        assert!(!HasRoom.keep(&facts, &c, &cx));
        c.has_room = true;
        assert!(HasRoom.keep(&facts, &c, &cx));
        let tried = [c.id];
        let retry = RequestFacts {
            tried: &tried,
            ..facts
        };
        assert!(!NotTried.keep(&retry, &c, &cx));
        let excluded = Constraints {
            excluded: &tried,
            ..cx
        };
        assert!(!NotTried.keep(&facts, &c, &excluded));
        assert!(NotTried.soft() && HasRoom.soft() && !Healthy.soft() && !HasCapacity.soft());

        let pool = [c];
        let mut kept = vec![0];
        NotTried.retain(&facts, &pool, &cx, &mut kept);
        assert_eq!(kept, [0]);
        NotTried.retain(&facts, &pool, &excluded, &mut kept);
        assert!(kept.is_empty());
        kept.push(0);
        NotTried.retain(&retry, &pool, &cx, &mut kept);
        assert!(kept.is_empty());
    }

    #[test]
    fn a_profiles_filters_as_one_row_test_answer_as_each_filter_asked() {
        // One row every filter keeps, then a row each refusal takes alone:
        // the four health bits, the overload bit, the capacity bit, the
        // room bit, a row the request tried, one the host excluded and one
        // the host's mask names.
        let mut pool: Vec<Candidate> = (0..11)
            .map(|position| Candidate {
                has_capacity: true,
                has_room: true,
                ..candidate(position, 0)
            })
            .collect();
        // The ready row holds the session hit: a flag of the row's word that
        // none of these filters reads.
        pool[0].session_hit = true;
        pool[1].health.healthy = false;
        pool[2].health.breaker_open = true;
        pool[3].health.stalled = true;
        pool[4].health.draining = true;
        pool[5].health.overloaded = true;
        pool[6].has_capacity = false;
        pool[7].has_room = false;
        let (tried, excluded, mask) = ([pool[8].id], [pool[9].id], [10u32]);
        let facts = RequestFacts {
            tried: &tried,
            ..RequestFacts::default()
        };
        let plain = Constraints {
            excluded: &excluded,
            ..Constraints::default()
        };
        let lifted = Constraints {
            overload_lifted: true,
            ..plain
        };
        let scopes = [
            plain,
            lifted,
            Constraints {
                stalled: &mask,
                ..plain
            },
            Constraints {
                stalled: &mask,
                ..lifted
            },
        ];
        let mut refused = [0u32; 11];
        for set in 0..32u8 {
            let filters = filters_of(set);
            let tests = RowTests::of(&filters).expect("the filters of this file fold");
            for cx in &scopes {
                let masked = tests.masked(cx);
                assert_eq!(
                    masked,
                    set & 1 != 0 && !cx.stalled.is_empty(),
                    "set {set:#07b}"
                );
                for c in &pool {
                    let each = filters.iter().all(|f| f.keep(&facts, c, cx));
                    let folded = tests.keeps(&facts, c, cx) && !(masked && cx.masks(c.position));
                    assert_eq!(folded, each, "set {set:#07b}, row {}, {cx:?}", c.position);
                    refused[c.position as usize] += u32::from(!each);
                }
            }
        }
        // The ready row is kept under every set and scope, and each other
        // row is refused under some.
        assert_eq!(refused[0], 0);
        assert!(refused[1..].iter().all(|&times| times > 0), "{refused:?}");
    }

    #[test]
    fn a_filter_of_a_hosts_folds_its_profile_into_no_row_test() {
        /// A host's filter around the health filter: the same word on every
        /// row, and a type the engine does not hold.
        #[derive(Debug)]
        struct Hosted(Healthy);
        impl Filter for Hosted {
            fn name(&self) -> &'static str {
                self.0.name()
            }

            fn row_wise(&self) -> bool {
                true
            }

            fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
                self.0.keep(facts, c, cx)
            }
        }
        assert!(RowTests::of(&[]).is_some(), "no filter keeps every row");
        let mut filters = filters_of(0b10011);
        assert!(RowTests::of(&filters).is_some());
        filters.insert(1, Box::new(Hosted(Healthy)));
        assert_eq!(RowTests::of(&filters), None);
    }

    #[test]
    fn the_healthy_filter_reads_the_hosts_mask_beside_the_rows_own_bits() {
        let facts = RequestFacts::default();
        let mut pool: Vec<Candidate> = (0..6).map(|position| candidate(position, 0)).collect();
        pool[4].health.breaker_open = true;
        let mask = [1u32, 3];
        let cx = Constraints {
            stalled: &mask,
            ..Constraints::default()
        };
        assert!(!cx.masks(0) && cx.masks(1) && cx.masks(3) && !cx.masks(4));
        // Row by row and over the whole pool, one answer: a row's own bits
        // refuse it, and so does the mask.
        let row_by_row: Vec<u32> = pool
            .iter()
            .filter(|row| Healthy.keep(&facts, row, &cx))
            .map(|row| row.position)
            .collect();
        let mut whole = kept(6);
        Healthy.retain(&facts, &pool, &cx, &mut whole);
        assert_eq!(row_by_row, [0, 2, 5]);
        assert_eq!(whole, row_by_row);
        // Under no veto the mask refuses nobody.
        let mut whole = kept(6);
        Healthy.retain(&facts, &pool, &Constraints::default(), &mut whole);
        assert_eq!(whole, [0, 1, 2, 3, 5]);
    }
}
