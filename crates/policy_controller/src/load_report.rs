//! A worker's load row from its report, shared by the load driver and the
//! worker host's load adapter.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use smg_worker::worker::{NamedValue, WorkerLoadResponse};

use crate::{
    signals::SignalColumn,
    spec::{AcrossRanks, MAX_SIGNALS},
    view::{permille, LoadRow, Role, SignalSource},
};

/// One worker's load row from its report, in one pass over the ranks: the
/// request and token counts summed, the KV ratios averaged, the drain rate
/// as the policies read it today. A prefill's drain is its measured input
/// capacity, which only a complete report with every rank's rate states: a
/// partial report or a rank without a rate leaves it unknown (zero), and
/// the scorer prices the worker at the fleet's nominal rate. Any other
/// role's drain is the generation rate of the ranks with a request running.
/// `source` is where the report came from; `report_age_ms` is `now` less
/// `sampled_at`.
pub fn load_row(
    report: &WorkerLoadResponse,
    role: Role,
    now: Instant,
    sampled_at: Instant,
    source: SignalSource,
) -> LoadRow {
    let mut running: i64 = 0;
    let mut queued: i64 = 0;
    let mut uncached: i64 = 0;
    let mut max_running: i64 = 0;
    let mut kv_usage = 0.0f64;
    let mut active_kv = 0.0f64;
    let mut live_rate = 0.0f64;
    let complete = !report.loads.is_empty()
        && report.loads.len() >= usize::try_from(report.dp_rank_count).unwrap_or(0);
    let mut prefill_rate: Option<f64> = complete.then_some(0.0);
    let mut resident: Option<i64> = Some(0);
    for rank in &report.loads {
        running += i64::from(rank.num_running_reqs);
        queued += i64::from(rank.num_waiting_reqs);
        uncached += i64::from(rank.num_waiting_uncached_tokens);
        max_running += i64::from(rank.max_running_requests);
        kv_usage += rank.token_usage;
        active_kv += rank.active_token_usage.unwrap_or(rank.token_usage);
        match role {
            Role::Prefill => {
                prefill_rate = prefill_rate
                    .zip(
                        rank.prefill_throughput
                            .filter(|rate| rate.is_finite() && *rate > 0.0),
                    )
                    .map(|(sum, rate)| sum + rate);
            }
            _ if rank.num_running_reqs > 0 => live_rate += rank.gen_throughput,
            _ => {}
        }
        resident = resident.zip(rank.num_resident_tokens).map(|(a, b)| a + b);
    }
    if role == Role::Prefill {
        live_rate = prefill_rate.unwrap_or(0.0);
    }
    let ranks = report.loads.len().max(1) as f64;
    LoadRow {
        running: narrow(running),
        queued: narrow(queued),
        waiting_uncached_tokens: narrow(uncached),
        kv_usage_permille: permille(kv_usage / ranks),
        active_kv_permille: permille(active_kv / ranks),
        live_rate: if live_rate.is_finite() && live_rate > 0.0 {
            live_rate.min(f64::from(u32::MAX)) as u32
        } else {
            0
        },
        resident_tokens: resident.map_or(0, narrow),
        // Counted when the report has ranks and every one of them sends it.
        resident_counted: resident.is_some() && !report.loads.is_empty(),
        report_age_ms: u32::try_from(now.saturating_duration_since(sampled_at).as_millis())
            .unwrap_or(u32::MAX),
        source,
        max_running: narrow(max_running),
    }
}

/// A summed engine count as the row's field: negative reads as none.
fn narrow(count: i64) -> u32 {
    u32::try_from(count.max(0)).unwrap_or(u32::MAX)
}

/// One row's values of the declared signals from its report, each handed
/// to `each` under its column, with the unit the report states, the value
/// and its sample instant: what every source of a report writes to the
/// signal sink behind the report's row ([`load_row`]), so each makes the
/// same values of the same report. The column is the one of `columns` the
/// value was made under, and the sink's write names it.
///
/// `columns` are the declared signals in name order
/// ([`SignalSink::signal_columns`](crate::store::SignalSink::signal_columns)):
/// a name the document does not declare is not offered. A row has a value
/// only where every rank of the report states the name, under one unit: a
/// sum over some of the ranks would understate the row, as the resident
/// tokens are counted. The ranks' values are made one as the signal
/// declares ([`AcrossRanks`]), under the oldest sample among them. A rank
/// that states a name twice is read at its first statement. A report of
/// one rank, as a rank's own record and a scoped answer are, hands that
/// rank's values as they are and allocates nothing.
///
/// The report is read as one sample. A pushed record of a worker over
/// several ranks is not one: its report lists each rank at its latest
/// record. A source makes that worker's values through [`RankValues`].
///
/// `sampled_at` is the sample instant of the report's row: a record's
/// own, a polled answer's stamp, else the poll's start. A value is
/// sampled then, less its own age where the report states one: the age
/// is the engine's, as of the report it made. A polled value with no
/// stamp is so never dated after its poll began, as its row is not. A
/// stamped one is dated no later than its answer's receipt.
pub fn signal_values(
    report: &WorkerLoadResponse,
    columns: &[SignalColumn],
    sampled_at: Instant,
    mut each: impl FnMut(&SignalColumn, &str, f64, Instant),
) {
    if columns.is_empty() {
        return;
    }
    let ranks = match report.loads.as_slice() {
        [] => return,
        // One rank: nothing is kept. Each declared name has a bit, set at
        // the name's first statement.
        [rank] if columns.len() <= u64::BITS as usize => {
            let mut read = 0u64;
            for stated in rank.named_values() {
                let Some(at) = column_of(columns, &stated.name) else {
                    continue;
                };
                let bit = 1u64 << at;
                if read & bit == 0 {
                    read |= bit;
                    let sampled = sampled(&stated, sampled_at);
                    each(&columns[at], &stated.unit, stated.value, sampled);
                }
            }
            return;
        }
        ranks => ranks,
    };
    let mut made: Vec<Option<Made>> = vec![None; columns.len()];
    for (nth, rank) in (0u32..).zip(ranks) {
        for stated in rank.named_values() {
            let Some(at) = column_of(columns, &stated.name) else {
                continue;
            };
            let sampled = sampled(&stated, sampled_at);
            let across = columns[at].across_ranks();
            Made::join(
                &mut made[at],
                nth,
                across,
                &stated.unit,
                stated.value,
                sampled,
            );
        }
    }
    hand(columns, made, ranks.len() as u32, each);
}

/// The place of the declared signal `name` among `columns`, which stand in
/// name order.
fn column_of(columns: &[SignalColumn], name: &str) -> Option<usize> {
    let found = columns.binary_search_by(|column| column.name().cmp(name));
    found.ok()
}

/// When `stated` was sampled, by a report sampled at `sampled_at`: then,
/// less the value's own age where the report states one.
fn sampled(stated: &NamedValue, sampled_at: Instant) -> Instant {
    stated.age_ms.map_or(sampled_at, |age| {
        let age = Duration::from_millis(u64::from(age));
        sampled_at.checked_sub(age).unwrap_or(sampled_at)
    })
}

/// Hands `each` the value of every column that all `count` ranks joined
/// ([`Made::join`]), made one as the column's signal declares.
fn hand(
    columns: &[SignalColumn],
    made: impl IntoIterator<Item = Option<Made>>,
    count: u32,
    mut each: impl FnMut(&SignalColumn, &str, f64, Instant),
) {
    for (column, made) in columns.iter().zip(made) {
        let Some(made) = made.filter(|made| made.ranks == count) else {
            continue;
        };
        let value = match column.across_ranks() {
            AcrossRanks::Mean => made.value / f64::from(count),
            AcrossRanks::Sum | AcrossRanks::Max | AcrossRanks::Min => made.value,
        };
        each(column, &made.unit, value, made.oldest);
    }
}

/// One declared signal's value over the ranks read so far.
#[derive(Clone, Debug)]
struct Made {
    /// How many ranks stated the name, each rank from the first.
    ranks: u32,
    unit: Arc<str>,
    value: f64,
    oldest: Instant,
}

impl Made {
    /// Joins the `nth` rank's statement of a name to the name's `slot`:
    /// its `value` under its `unit`, sampled at `sampled`. A rank's later
    /// statement of the same name is passed over.
    fn join(
        slot: &mut Option<Self>,
        nth: u32,
        across: AcrossRanks,
        unit: &Arc<str>,
        value: f64,
        sampled: Instant,
    ) {
        match slot {
            None if nth == 0 => {
                *slot = Some(Self {
                    ranks: 1,
                    unit: Arc::clone(unit),
                    value,
                    oldest: sampled,
                });
            }
            // Every rank before this one stated the name, and this is
            // the rank's first statement of it. Under another unit the
            // row has no value: its slot is emptied, and a slot is
            // filled at the first rank alone, so no later statement
            // is read.
            Some(row) if row.ranks == nth => {
                if row.unit == *unit {
                    row.ranks += 1;
                    row.value = joined(across, row.value, value);
                    row.oldest = row.oldest.min(sampled);
                } else {
                    *slot = None;
                }
            }
            // A rank before this one left the name out or stated
            // another unit, or this rank states the name once more.
            _ => {}
        }
    }
}

/// The value over one more rank. A rank's value that is no finite number
/// makes the row's none, under every reduction: the sink then counts it,
/// where a maximum would pass it over.
fn joined(across: AcrossRanks, so_far: f64, next: f64) -> f64 {
    match across {
        AcrossRanks::Sum | AcrossRanks::Mean => so_far + next,
        AcrossRanks::Max | AcrossRanks::Min if !(so_far.is_finite() && next.is_finite()) => {
            f64::NAN
        }
        AcrossRanks::Max => so_far.max(next),
        AcrossRanks::Min => so_far.min(next),
    }
}

/// What each rank of one worker last stated of the declared signals, each
/// statement under its own sample instant: what the row's values of a
/// worker over several ranks are made of, report after report.
///
/// Such a worker's pushed record is one rank's, and the report it comes
/// with lists every rank at its latest record. The entries are of as many
/// instants, and a record may leave out a name an earlier one stated.
/// Read as one sample ([`signal_values`]), a rank that fell silent would
/// count as fresh under the newest record's instant, and one record
/// without a name would leave the row with no new value. So a source
/// keeps one of these a worker and makes the row's values through it
/// ([`Self::values`]).
///
/// Each entry of a pushed report says when its own sample was
/// ([`sampled_at`](smg_worker::worker::SchedulerLoadSnapshot::sampled_at)):
/// the collector writes that instant where it writes the entry, at a
/// rank's record and at a polled answer laid on the report. The keeper
/// reads an entry under the instant it says, and works out no instant by
/// a rule of its own. An entry that says none is read as the report's own
/// sample. So a source can make a keeper at any report and nothing stated
/// before is lost: the report holds each rank's latest entry. A source
/// makes one at a worker's first report of several ranks. A worker of one
/// rank needs none: its report is one sample, handed as it is, and the
/// store's cell holds its last value.
///
/// What is kept is one statement a declared signal and a rank, for
/// [`MAX_RANKS`] ranks at the most. A statement ends with its keeper, and
/// with the list of columns it was read under: a list handed anew ends
/// them all, though it declares the same signals, and the next report's
/// entries are read again, each under its own instant. A name a rank's
/// latest record left out then counts when the rank states it again. No
/// report's list ends a statement: an answer may list fewer ranks than
/// its engine has, and a worker's first pushed reports list fewer than an
/// answer.
#[derive(Debug, Default)]
pub struct RankValues {
    /// The columns the statements were read under.
    columns: Option<Arc<[SignalColumn]>>,
    /// The ranks the worker's reports listed, in rank order.
    ranks: Vec<RankStated>,
    /// The slots of one reduction, kept for their room.
    made: Vec<Option<Made>>,
}

/// What one rank last stated: one statement a declared name at the most.
#[derive(Debug)]
struct RankStated {
    rank: i32,
    /// The instant the rank's entry of a pushed report said when the
    /// keeper last read it. The entry is read again where it says another.
    read_at: Option<Instant>,
    stated: Vec<Stated>,
}

/// One rank's last statement of one declared name.
#[derive(Debug)]
struct Stated {
    /// The name's place among the columns.
    column: usize,
    unit: Arc<str>,
    value: f64,
    sampled: Instant,
}

/// Whose sample a report is: what says under which instant a keeper reads
/// each of its entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportOf {
    /// A polled answer's: one sample of every rank it lists, under the
    /// answer's instant.
    EveryRank,
    /// A pushed record's: its own rank's, under the record's instant.
    /// Every other entry is an older write, read under the instant the
    /// entry says.
    OneRank {
        /// The rank the record came from.
        rank: i32,
    },
}

/// The ranks one keeper holds at the most. An engine's ranks are far
/// fewer. A rank past the bound is not kept, so a report that lists it
/// makes no value through the keeper.
pub const MAX_RANKS: usize = 1_024;

// A rank's first statement of each declared name is marked in the bits of
// one word ([`RankValues::keep`]): a document's cap of names fits them.
const _: () = assert!(MAX_SIGNALS <= u64::BITS as usize);

/// Whether `report` lists its ranks in rank order, each once, as a pushed
/// record's report does.
fn in_rank_order(report: &WorkerLoadResponse) -> bool {
    let ranks = || report.loads.iter().map(|rank| rank.dp_rank);
    ranks().zip(ranks().skip(1)).all(|(rank, next)| rank < next)
}

/// What `rank` last stated among `ranks`, which stand in rank order.
fn of_rank(ranks: &[RankStated], rank: i32) -> Option<&RankStated> {
    let at = ranks.binary_search_by_key(&rank, |kept| kept.rank).ok()?;
    ranks.get(at)
}

/// Whether `report` lists each of its ranks once, in whatever order.
fn lists_each_rank_once(report: &WorkerLoadResponse) -> bool {
    let mut rest = report.loads.iter();
    while let Some(rank) = rest.next() {
        if rest.clone().any(|later| later.dp_rank == rank.dp_rank) {
            return false;
        }
    }
    true
}

impl RankStated {
    /// The rank's statement of the name at `column`, sampled at `sampled`,
    /// in the place of its last one of that name unless that one was
    /// sampled later.
    fn state(&mut self, column: usize, stated: NamedValue, sampled: Instant) {
        let NamedValue { unit, value, .. } = stated;
        let stated = Stated {
            column,
            unit,
            value,
            sampled,
        };
        match self.stated.iter_mut().find(|kept| kept.column == column) {
            Some(kept) if kept.sampled > sampled => {}
            Some(kept) => *kept = stated,
            None => self.stated.push(stated),
        }
    }
}

impl RankValues {
    /// Whether a keeper makes the values of `report`: one of several ranks
    /// of one engine, each listed once, under a document that declares a
    /// signal. Any other report is one sample, and [`signal_values`] makes
    /// its values. A gateway's rollup of its own workers is one: its
    /// entries are workers, and they repeat a rank, so one worker's
    /// statement would be kept in the place of another's.
    pub fn keeps(report: &WorkerLoadResponse, columns: &[SignalColumn]) -> bool {
        !columns.is_empty() && report.loads.len() > 1 && Self::lists_ranks(report)
    }

    /// Whether `report` lists ranks of one engine, each once. One pass
    /// says so of a report in rank order, as a pushed record's is, and an
    /// answer out of order is read pair by pair.
    fn lists_ranks(report: &WorkerLoadResponse) -> bool {
        !report.loads.is_empty()
            && report.ranks_are_dp_ranks()
            && (in_rank_order(report) || lists_each_rank_once(report))
    }

    /// One row's values of the declared signals from a report of its
    /// worker, each handed to `each` as [`signal_values`] hands it.
    ///
    /// A report of several ranks is kept first ([`Self::keep`]). The row's
    /// values are then made over what each of the report's ranks last
    /// stated, by the rule of one report: every rank states the name,
    /// under one unit, and the ranks' values are made one as the signal
    /// declares. Each value is dated by the oldest of its statements. So a
    /// rank that fell silent ages the row's value, and a name one record
    /// leaves out keeps the statement its rank made before.
    pub fn values(
        &mut self,
        report: &WorkerLoadResponse,
        of: ReportOf,
        columns: &Arc<[SignalColumn]>,
        sampled_at: Instant,
        each: impl FnMut(&SignalColumn, &str, f64, Instant),
    ) {
        if !self.keep(report, of, columns, sampled_at) {
            signal_values(report, columns, sampled_at, each);
            return;
        }
        self.made.clear();
        self.made.resize(columns.len(), None);
        for (nth, rank) in (0u32..).zip(&report.loads) {
            let kept = of_rank(&self.ranks, rank.dp_rank);
            for stated in kept.into_iter().flat_map(|kept| &kept.stated) {
                let slot = self.made.get_mut(stated.column);
                let (Some(slot), Some(column)) = (slot, columns.get(stated.column)) else {
                    continue;
                };
                let across = column.across_ranks();
                Made::join(
                    slot,
                    nth,
                    across,
                    &stated.unit,
                    stated.value,
                    stated.sampled,
                );
            }
        }
        let ranks = report.loads.len() as u32;
        hand(columns, self.made.drain(..), ranks, each);
    }

    /// Keeps what a report states, and says whether the row's values are
    /// made through the keeper ([`Self::keeps`]). A report of one rank is
    /// kept too: its statements stand for a later report of several ranks,
    /// and its values are its own. Nothing is kept of a report that lists
    /// no ranks of one engine, or under a document that declares no
    /// signal.
    ///
    /// `of` says whose sample the report is. A polled answer is one sample
    /// of every rank it lists, at `sampled_at`. A pushed record is its own
    /// rank's, at `sampled_at`. Every other entry of its report is read
    /// under the instant the entry says, where that is not the instant the
    /// keeper last read of the rank: the entry was written since, by the
    /// rank's record or by an answer laid on the report. An entry that
    /// says no instant is read as the record's own sample, at `sampled_at`.
    ///
    /// A rank's statement takes the place of its last one of that name,
    /// unless that one was sampled later: a later report of an older
    /// sample leaves it, as the store's cell does. A name the rank does
    /// not state keeps its last statement. A rank that states a name twice
    /// is read at its first statement.
    pub fn keep(
        &mut self,
        report: &WorkerLoadResponse,
        of: ReportOf,
        columns: &Arc<[SignalColumn]>,
        sampled_at: Instant,
    ) -> bool {
        if columns.is_empty() || !Self::lists_ranks(report) {
            return false;
        }
        self.read_under(columns);
        for entry in &report.loads {
            let own = match of {
                ReportOf::EveryRank => true,
                ReportOf::OneRank { rank } => rank == entry.dp_rank,
            };
            // The entry's sample instant: the report's where the entry is
            // the report's own sample or says none, else the one it says.
            let at = match entry.sampled_at {
                Some(at) if !own => at,
                _ => sampled_at,
            };
            let Some(kept) = self.rank_mut(entry.dp_rank) else {
                continue;
            };
            if let ReportOf::OneRank { .. } = of {
                // A pushed report's entry: another rank's is read where it
                // says another instant than the one last read of the rank.
                if !own && kept.read_at == Some(at) {
                    continue;
                }
                kept.read_at = Some(at);
            }
            // Each declared name has a bit, set at the name's first
            // statement. A document declares no more names than the bits.
            let mut read = 0u64;
            for stated in entry.named_values() {
                let column = column_of(columns, &stated.name);
                let bit = column.and_then(|at| 1u64.checked_shl(u32::try_from(at).ok()?));
                let (Some(column), Some(bit)) = (column, bit) else {
                    continue;
                };
                if read & bit != 0 {
                    continue;
                }
                read |= bit;
                let sampled = sampled(&stated, at);
                kept.state(column, stated, sampled);
            }
        }
        report.loads.len() > 1
    }

    /// The statements under `columns`, the list of the document in force.
    /// A statement stands for the column it was read under: another list
    /// ends them all, though it declares the same signals, and the next
    /// report's entries are read again. No statement so passes a document
    /// that stood between two reports.
    fn read_under(&mut self, columns: &Arc<[SignalColumn]>) {
        let same = |stood: &Arc<[SignalColumn]>| Arc::ptr_eq(stood, columns);
        if self.columns.as_ref().is_some_and(same) {
            return;
        }
        for rank in &mut self.ranks {
            rank.stated.clear();
            rank.read_at = None;
        }
        self.columns = Some(Arc::clone(columns));
    }

    /// What is kept of `rank`, started where nothing is. `None` past the
    /// keeper's bound of ranks.
    fn rank_mut(&mut self, rank: i32) -> Option<&mut RankStated> {
        let at = match self.ranks.binary_search_by_key(&rank, |kept| kept.rank) {
            Ok(at) => at,
            Err(_) if self.ranks.len() >= MAX_RANKS => return None,
            Err(at) => {
                let fresh = RankStated {
                    rank,
                    read_at: None,
                    stated: Vec::new(),
                };
                self.ranks.insert(at, fresh);
                at
            }
        };
        self.ranks.get_mut(at)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use smg_worker::worker::SchedulerLoadSnapshot;

    use super::*;

    fn rank(
        running: i32,
        waiting: i32,
        usage: f64,
        gen: f64,
        prefill: Option<f64>,
    ) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            token_usage: usage,
            gen_throughput: gen,
            prefill_throughput: prefill,
            max_running_requests: 256,
            num_resident_tokens: Some(1_000),
            ..Default::default()
        }
    }

    fn report(loads: Vec<SchedulerLoadSnapshot>) -> WorkerLoadResponse {
        WorkerLoadResponse {
            dp_rank_count: i32::try_from(loads.len()).unwrap_or(i32::MAX),
            loads,
            ..Default::default()
        }
    }

    #[test]
    fn counts_add_over_the_ranks_and_ratios_average() {
        let now = Instant::now();
        let two = report(vec![
            rank(3, 1, 0.2, 100.0, None),
            rank(5, 2, 0.6, 50.0, None),
        ]);
        let row = load_row(&two, Role::Single, now, now, SignalSource::Poll);
        assert_eq!((row.running, row.queued), (8, 3));
        assert_eq!(row.kv_usage_permille, 400);
        assert_eq!(row.active_kv_permille, 400);
        assert_eq!(row.live_rate, 150);
        assert_eq!(row.resident_tokens, 2_000);
        assert!(row.resident_counted);
        assert_eq!(row.max_running, 512);
        assert_eq!(row.source, SignalSource::Poll);
        assert_eq!(row.report_age_ms, 0);
    }

    /// One rank without the count, or a report with no rank, is a report
    /// that does not count resident tokens, not a count of zero.
    #[test]
    fn resident_tokens_are_counted_only_when_every_rank_sends_them() {
        let now = Instant::now();
        let mut silent = rank(1, 0, 0.1, 40.0, None);
        silent.num_resident_tokens = None;
        let one_silent = report(vec![rank(1, 0, 0.1, 40.0, None), silent]);
        let row = load_row(&one_silent, Role::Single, now, now, SignalSource::Poll);
        assert_eq!((row.resident_tokens, row.resident_counted), (0, false));
        let no_rank = load_row(&report(vec![]), Role::Single, now, now, SignalSource::Poll);
        assert_eq!(
            (no_rank.resident_tokens, no_rank.resident_counted),
            (0, false)
        );
    }

    #[test]
    fn an_idle_rank_adds_no_generation_rate() {
        let now = Instant::now();
        let one_idle = report(vec![
            rank(0, 0, 0.0, 999.0, None),
            rank(1, 0, 0.1, 40.0, None),
        ]);
        let row = load_row(&one_idle, Role::Decode, now, now, SignalSource::Events);
        assert_eq!(row.live_rate, 40);
    }

    #[test]
    fn a_prefill_drain_needs_every_rank_rate_of_a_complete_report() {
        let now = Instant::now();
        let complete = report(vec![
            rank(1, 0, 0.1, 0.0, Some(2_000.0)),
            rank(1, 0, 0.1, 0.0, Some(3_000.0)),
        ]);
        assert_eq!(
            load_row(&complete, Role::Prefill, now, now, SignalSource::Poll).live_rate,
            5_000
        );
        let one_rate_missing = report(vec![
            rank(1, 0, 0.1, 0.0, Some(2_000.0)),
            rank(1, 0, 0.1, 0.0, None),
        ]);
        assert_eq!(
            load_row(
                &one_rate_missing,
                Role::Prefill,
                now,
                now,
                SignalSource::Poll
            )
            .live_rate,
            0
        );
        let mut partial = report(vec![rank(1, 0, 0.1, 0.0, Some(2_000.0))]);
        partial.dp_rank_count = 2;
        assert_eq!(
            load_row(&partial, Role::Prefill, now, now, SignalSource::Poll).live_rate,
            0
        );
    }

    #[test]
    fn the_age_is_the_decision_instant_less_the_sample_instant() {
        let sampled = Instant::now();
        let now = sampled + Duration::from_millis(1_500);
        let row = load_row(
            &report(vec![rank(0, 0, 0.0, 0.0, None)]),
            Role::Single,
            now,
            sampled,
            SignalSource::Scrape,
        );
        assert_eq!(row.report_age_ms, 1_500);
        assert_eq!(row.source, SignalSource::Scrape);
    }

    #[test]
    fn negative_counts_and_ratios_over_one_are_clamped() {
        let now = Instant::now();
        let odd = report(vec![rank(-4, -1, 1.7, -5.0, None)]);
        let row = load_row(&odd, Role::Single, now, now, SignalSource::Poll);
        assert_eq!((row.running, row.queued), (0, 0));
        assert_eq!(row.kv_usage_permille, 1_000);
        assert_eq!(row.live_rate, 0);
    }

    /// The declared signals `names`, each under `unit` and its reduction,
    /// in name order as the applied document lists them.
    fn columns(names: &[(&str, &str, AcrossRanks)]) -> Vec<SignalColumn> {
        let mut names = names.to_vec();
        names.sort_by_key(|&(name, ..)| name);
        let column = |((name, unit, across), at): ((&str, &str, AcrossRanks), u16)| {
            SignalColumn::new(name, unit, at, None).across(across)
        };
        names.into_iter().zip(0..).map(column).collect()
    }

    fn stated(name: &str, unit: &str, value: f64, age_ms: Option<u32>) -> NamedValue {
        NamedValue {
            name: name.into(),
            unit: unit.into(),
            value,
            age_ms,
        }
    }

    /// One rank that states `values` by name and no fixed count.
    fn stating(values: Vec<NamedValue>) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            values,
            ..Default::default()
        }
    }

    /// What the conversion hands of `report` under `columns`, received at
    /// `received` from a record sampled 50 ms before: each value by its
    /// name and unit, with its age at the receipt in milliseconds. A
    /// stated age counts back from the record's sample.
    fn made(
        report: &WorkerLoadResponse,
        columns: &[SignalColumn],
        received: Instant,
    ) -> Vec<(String, String, f64, u128)> {
        let sampled_at = received - Duration::from_millis(50);
        let mut made = Vec::new();
        signal_values(report, columns, sampled_at, |column, unit, value, at| {
            let age = received.duration_since(at).as_millis();
            made.push((column.name().to_owned(), unit.to_owned(), value, age));
        });
        made
    }

    fn value(name: &str, unit: &str, value: f64, age: u128) -> (String, String, f64, u128) {
        (name.to_owned(), unit.to_owned(), value, age)
    }

    #[test]
    fn several_ranks_make_one_rows_value_as_the_signal_declares_under_the_oldest_sample() {
        let received = Instant::now() + Duration::from_secs(5);
        let columns = columns(&[
            ("largest", "ms", AcrossRanks::Max),
            ("middle", "ms", AcrossRanks::Mean),
            ("smallest", "ms", AcrossRanks::Min),
            ("total", "ms", AcrossRanks::Sum),
        ]);
        let names = ["largest", "middle", "smallest", "total"];
        // Three ranks state 2, 5 and 11 of each name. The first states no
        // age, so its value is as old as its record: 50 ms. The second's
        // is 300 ms older than the record.
        let rank = |value, age| stating(names.map(|name| stated(name, "ms", value, age)).into());
        let whole = report(vec![
            rank(2.0, None),
            rank(5.0, Some(300)),
            rank(11.0, Some(120)),
        ]);
        assert_eq!(
            made(&whole, &columns, received),
            [
                value("largest", "ms", 11.0, 350),
                value("middle", "ms", 6.0, 350),
                value("smallest", "ms", 2.0, 350),
                value("total", "ms", 18.0, 350),
            ]
        );
        // Scoped to one rank, a report hands that rank's values as they are.
        let scoped = whole.for_dp_rank(None).map(|mut all| {
            all.loads.truncate(1);
            all
        });
        let scoped = scoped.expect("a report with ranks");
        let alone = made(&scoped, &columns, received);
        assert_eq!(alone, names.map(|name| value(name, "ms", 2.0, 50)));
    }

    #[test]
    fn a_row_has_a_value_only_where_every_rank_states_the_name_under_one_unit() {
        let received = Instant::now() + Duration::from_secs(5);
        let columns = columns(&[
            ("mixed", "ms", AcrossRanks::Sum),
            ("partial", "ms", AcrossRanks::Sum),
            ("turned", "ms", AcrossRanks::Sum),
            ("twice", "ms", AcrossRanks::Sum),
            ("whole", "ms", AcrossRanks::Sum),
        ]);
        let first = stating(vec![
            stated("mixed", "ms", 1.0, None),
            stated("partial", "ms", 1.0, None),
            stated("turned", "ms", 1.0, None),
            stated("twice", "ms", 1.0, None),
            // A second statement by one rank: its first one is read.
            stated("twice", "ms", 64.0, None),
            stated("whole", "ms", 1.0, None),
            // A name the document does not declare is not offered.
            stated("other", "ms", 1.0, None),
        ]);
        let second = stating(vec![
            stated("mixed", "seconds", 2.0, None),
            // A rank's first statement rules: under another unit the row
            // has no value, whatever the rank states of the name next.
            stated("turned", "seconds", 9.0, None),
            stated("turned", "ms", 2.0, None),
            stated("twice", "ms", 2.0, None),
            stated("whole", "ms", 2.0, None),
        ]);
        assert_eq!(
            made(&report(vec![first.clone(), second]), &columns, received),
            [value("twice", "ms", 3.0, 50), value("whole", "ms", 3.0, 50)]
        );
        // The ranks in the other order: the names the first one leaves out
        // are none of the row's, and the one both state is.
        let late = stating(vec![stated("partial", "ms", 2.0, None)]);
        assert_eq!(
            made(&report(vec![late, first]), &columns, received),
            [value("partial", "ms", 3.0, 50)]
        );
        // No declared signal and no rank hand nothing.
        let one = report(vec![stating(vec![stated("whole", "ms", 1.0, None)])]);
        assert_eq!(made(&one, &[], received), []);
        assert_eq!(made(&report(vec![]), &columns, received), []);
    }

    #[test]
    fn a_report_of_one_rank_hands_each_declared_name_at_its_first_statement() {
        let received = Instant::now() + Duration::from_secs(5);
        let columns = columns(&[
            ("depth", "items", AcrossRanks::Mean),
            ("wait", "ms", AcrossRanks::Max),
        ]);
        let alone = report(vec![stating(vec![
            stated("wait", "ms", 7.0, Some(20)),
            stated("other", "ms", 1.0, None),
            stated("depth", "items", 3.0, None),
            stated("wait", "ms", 64.0, None),
        ])]);
        // In the report's order: the one rank's values are handed as they
        // are read, each under its own age or its record's.
        assert_eq!(
            made(&alone, &columns, received),
            [
                value("wait", "ms", 7.0, 70),
                value("depth", "items", 3.0, 50)
            ]
        );
        // A value that is no number is handed on: the sink counts it.
        let odd = report(vec![stating(vec![stated("wait", "ms", f64::NAN, None)])]);
        let made = made(&odd, &columns, received);
        assert!(matches!(made.as_slice(), [(_, _, value, _)] if value.is_nan()));
    }

    #[test]
    fn a_ranks_value_that_is_no_number_makes_the_rows_none_under_every_reduction() {
        let received = Instant::now() + Duration::from_secs(5);
        for across in [
            AcrossRanks::Sum,
            AcrossRanks::Max,
            AcrossRanks::Min,
            AcrossRanks::Mean,
        ] {
            let columns = columns(&[("depth", "items", across)]);
            for odd in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let rank = |value| stating(vec![stated("depth", "items", value, None)]);
                let made = made(
                    &report(vec![rank(4.0), rank(odd), rank(9.0)]),
                    &columns,
                    received,
                );
                let [(_, _, value, _)] = made.as_slice() else {
                    panic!("one value, {made:?}");
                };
                assert!(!value.is_finite(), "{across:?} of {odd}: {value}");
            }
        }
    }

    #[test]
    fn the_fixed_token_counts_are_offered_under_their_names() {
        let received = Instant::now() + Duration::from_secs(5);
        let columns = columns(&[("kv_tokens_used", "tokens", AcrossRanks::Sum)]);
        let rank = |used| SchedulerLoadSnapshot {
            num_used_tokens: used,
            max_total_num_tokens: 8_192,
            ..Default::default()
        };
        assert_eq!(
            made(&report(vec![rank(1_000), rank(24)]), &columns, received),
            [value("kv_tokens_used", "tokens", 1_024.0, 50)]
        );
        // A rank that drains states zero: the row's sum still stands, and a
        // drained rank alone reads zero where its last count stood.
        assert_eq!(
            made(&report(vec![rank(1_000), rank(0)]), &columns, received),
            [value("kv_tokens_used", "tokens", 1_000.0, 50)]
        );
        assert_eq!(
            made(&report(vec![rank(0)]), &columns, received),
            [value("kv_tokens_used", "tokens", 0.0, 50)]
        );
    }

    /// One rank's entry of a worker's report: `rank` states `values`.
    fn ranked(rank: i32, values: Vec<NamedValue>) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            dp_rank: rank,
            ..stating(values)
        }
    }

    /// As [`ranked`], as the entry stands on a pushed report: it says
    /// when its own sample was.
    fn ranked_at(rank: i32, values: Vec<NamedValue>, at: Instant) -> SchedulerLoadSnapshot {
        SchedulerLoadSnapshot {
            sampled_at: Some(at),
            ..ranked(rank, values)
        }
    }

    /// A worker's pushed report as a collector keeps it: each rank's entry
    /// at its latest record, under that record's sample instant.
    #[derive(Default)]
    struct Pushed {
        report: WorkerLoadResponse,
    }

    impl Pushed {
        /// The report behind a record of `rank` that states `values`,
        /// sampled at `at`.
        fn record(
            &mut self,
            rank: i32,
            values: Vec<NamedValue>,
            at: Instant,
        ) -> &WorkerLoadResponse {
            let entry = ranked_at(rank, values, at);
            let loads = &mut self.report.loads;
            match loads.binary_search_by_key(&rank, |load| load.dp_rank) {
                Ok(place) => loads[place] = entry,
                Err(place) => loads.insert(place, entry),
            }
            &self.report
        }
    }

    fn depth(value: f64) -> Vec<NamedValue> {
        vec![stated("depth", "items", value, None)]
    }

    /// The depth alone, summed over a worker's ranks, as an applied
    /// document shares its columns.
    fn depth_alone() -> Arc<[SignalColumn]> {
        columns(&[("depth", "items", AcrossRanks::Sum)]).into()
    }

    /// What `keeper` hands of `report`, sampled at `sampled_at`: a record
    /// of the rank `record_of`, or a polled answer. Each value by its name
    /// and unit, with its age at `now` in milliseconds.
    fn kept(
        keeper: &mut RankValues,
        report: &WorkerLoadResponse,
        record_of: Option<i32>,
        columns: &Arc<[SignalColumn]>,
        (sampled_at, now): (Instant, Instant),
    ) -> Vec<(String, String, f64, u128)> {
        let of = record_of.map_or(ReportOf::EveryRank, |rank| ReportOf::OneRank { rank });
        let mut handed = Vec::new();
        keeper.values(
            report,
            of,
            columns,
            sampled_at,
            |column, unit, value, at| {
                let age = now.duration_since(at).as_millis();
                handed.push((column.name().to_owned(), unit.to_owned(), value, age));
            },
        );
        handed
    }

    /// A worker over two ranks. Rank 1 states its depth once and falls
    /// silent, and rank 0 states its own each second. The row's value
    /// stays as old as rank 1's statement, where one sample of the whole
    /// report dates both ranks by the newest record.
    #[test]
    fn a_silent_ranks_statement_keeps_its_own_sample_instant() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let columns = depth_alone();
        let mut pushed = Pushed::default();
        let mut keeper = RankValues::default();
        // Rank 1's record alone is one sample of one rank.
        let alone = pushed.record(1, depth(100.0), at(0));
        let first = kept(&mut keeper, alone, Some(1), &columns, (at(0), at(0)));
        assert_eq!(first, [value("depth", "items", 100.0, 0)]);
        for second in 1..=9 {
            let now = at(second * 1_000);
            let file = pushed.record(0, depth(1.0), now);
            let handed = kept(&mut keeper, file, Some(0), &columns, (now, now));
            let age = u128::from(second) * 1_000;
            assert_eq!(handed, [value("depth", "items", 101.0, age)]);
        }
        let mut one_sample = Vec::new();
        signal_values(
            &pushed.report,
            &columns,
            at(9_000),
            |_, _, value, sampled| {
                one_sample.push((value, at(9_000).duration_since(sampled).as_millis()));
            },
        );
        assert_eq!(one_sample, [(101.0, 0)]);
    }

    /// A record may leave a name out: the entry then holds no value of it,
    /// and the rank's last statement stands for the row.
    #[test]
    fn a_record_that_leaves_a_name_out_keeps_its_ranks_last_statement() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let columns = depth_alone();
        let mut pushed = Pushed::default();
        let mut keeper = RankValues::default();
        let alone = pushed.record(1, depth(2.0), at(0));
        kept(&mut keeper, alone, Some(1), &columns, (at(0), now));
        let both = pushed.record(0, depth(1.0), at(10));
        let whole = kept(&mut keeper, both, Some(0), &columns, (at(10), now));
        assert_eq!(whole, [value("depth", "items", 3.0, 1_000)]);
        // Rank 1's next record states no value, then rank 0 states 3.
        let bare = pushed.record(1, Vec::new(), at(20));
        let stands = kept(&mut keeper, bare, Some(1), &columns, (at(20), now));
        assert_eq!(stands, [value("depth", "items", 3.0, 1_000)]);
        let moved = pushed.record(0, depth(3.0), at(30));
        let later = kept(&mut keeper, moved, Some(0), &columns, (at(30), now));
        assert_eq!(later, [value("depth", "items", 5.0, 1_000)]);
        // One sample of that report makes no value: rank 1 states none.
        let mut one_sample = 0;
        let file = &pushed.report;
        signal_values(file, &columns, at(30), |_, _, _, _| one_sample += 1);
        assert_eq!(one_sample, 0);
    }

    /// A record states its own rank under its own instant, and another
    /// rank's entry is read under the instant the entry says. An entry
    /// that says none is read as the record's own sample. A polled answer
    /// is one sample of every rank.
    #[test]
    fn a_record_states_its_own_rank_and_a_polled_answer_every_rank() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(10_000);
        let columns = depth_alone();
        // Rank 1's entry was written at the first second, and says so.
        let mut keeper = RankValues::default();
        let first = ranked_at(1, depth(50.0), at(1_000));
        let file = report(vec![ranked(0, depth(1.0)), first.clone()]);
        let pushed = kept(&mut keeper, &file, Some(0), &columns, (at(9_000), now));
        assert_eq!(pushed, [value("depth", "items", 51.0, 9_000)]);
        // An entry that says no instant is one sample with the record.
        let mut unsaid = RankValues::default();
        let bare = report(vec![ranked(0, depth(1.0)), ranked(1, depth(50.0))]);
        let one = kept(&mut unsaid, &bare, Some(0), &columns, (at(9_000), now));
        assert_eq!(one, [value("depth", "items", 51.0, 1_000)]);
        // A polled answer is one sample of every rank.
        let answer = report(vec![ranked(0, depth(4.0)), ranked(1, depth(6.0))]);
        let polled = kept(&mut keeper, &answer, None, &columns, (at(9_500), now));
        assert_eq!(polled, [value("depth", "items", 10.0, 500)]);
        // Rank 0's next record: its own statement, beside the answer's of
        // rank 1. Rank 1's entry says the instant it said before.
        let next = report(vec![ranked(0, depth(5.0)), first]);
        let own = kept(&mut keeper, &next, Some(0), &columns, (at(9_800), now));
        assert_eq!(own, [value("depth", "items", 11.0, 500)]);
    }

    #[test]
    fn a_ranks_statement_stands_until_a_newer_one_of_its_name() {
        let start = Instant::now() + Duration::from_secs(10);
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(10_000);
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        // Rank 1 stated 10 at the eighth second. Rank 0 states `own` by a
        // record sampled at `ms`.
        let mut pushed = Pushed::default();
        let alone = pushed.record(1, depth(10.0), at(8_000));
        kept(&mut keeper, alone, Some(1), &columns, (at(8_000), now));
        let mut rank_zero = |own: Vec<NamedValue>, ms| {
            let file = pushed.record(0, own, at(ms));
            kept(&mut keeper, file, Some(0), &columns, (at(ms), now))
        };
        let first = rank_zero(depth(1.0), 5_000);
        assert_eq!(first, [value("depth", "items", 11.0, 5_000)]);
        // A later record of a sample two seconds old: the newer one stands.
        let older = rank_zero(vec![stated("depth", "items", 9.0, Some(2_000))], 6_000);
        assert_eq!(older, [value("depth", "items", 11.0, 5_000)]);
        let newer = rank_zero(vec![stated("depth", "items", 2.0, Some(500))], 7_000);
        assert_eq!(newer, [value("depth", "items", 12.0, 3_500)]);
        // A rank that states a name twice is read at its first statement.
        let twice = [3.0, 7.0].map(|value| stated("depth", "items", value, None));
        let read = rank_zero(twice.into(), 7_500);
        assert_eq!(read, [value("depth", "items", 13.0, 2_500)]);
    }

    #[test]
    fn the_kept_statements_are_made_one_by_the_rule_of_one_report() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let columns: Arc<[SignalColumn]> = columns(&[
            ("largest", "ms", AcrossRanks::Max),
            ("middle", "ms", AcrossRanks::Mean),
            ("mixed", "ms", AcrossRanks::Sum),
            ("partial", "ms", AcrossRanks::Sum),
        ])
        .into();
        let zero = vec![
            stated("largest", "ms", 2.0, None),
            stated("middle", "ms", 2.0, None),
            stated("mixed", "ms", 1.0, None),
            stated("partial", "ms", 1.0, None),
        ];
        let one = vec![
            stated("largest", "ms", 5.0, None),
            stated("middle", "ms", 6.0, None),
            stated("mixed", "s", 1.0, None),
        ];
        let mut pushed = Pushed::default();
        let mut keeper = RankValues::default();
        let alone = pushed.record(1, one, at(0));
        kept(&mut keeper, alone, Some(1), &columns, (at(0), now));
        // Two units make no value, and neither does a name one rank never
        // stated.
        let file = pushed.record(0, zero, at(100));
        assert_eq!(
            kept(&mut keeper, file, Some(0), &columns, (at(100), now)),
            [
                value("largest", "ms", 5.0, 1_000),
                value("middle", "ms", 4.0, 1_000),
            ]
        );
    }

    /// No report's list ends a statement. An answer may list fewer ranks
    /// than its engine has: a rank it leaves out stands, as old as it
    /// stated, and counts when a report lists it again.
    #[test]
    fn a_rank_a_report_leaves_out_keeps_its_statements() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        let both = report(vec![ranked(0, depth(1.0)), ranked(1, depth(2.0))]);
        let whole = kept(&mut keeper, &both, None, &columns, (at(0), now));
        assert_eq!(whole, [value("depth", "items", 3.0, 1_000)]);
        // An answer and a record list rank 0 alone, with no depth: a
        // report of one rank is one sample.
        let one = report(vec![ranked(0, Vec::new())]);
        assert!(kept(&mut keeper, &one, None, &columns, (at(10), now)).is_empty());
        assert!(kept(&mut keeper, &one, Some(0), &columns, (at(20), now)).is_empty());
        // Rank 0 states 5 beside a rank 1 with no depth: rank 1 counts
        // what the first answer stated.
        let next = report(vec![ranked(0, depth(5.0)), ranked(1, Vec::new())]);
        let sum = kept(&mut keeper, &next, None, &columns, (at(30), now));
        assert_eq!(sum, [value("depth", "items", 7.0, 1_000)]);
    }

    /// A keeper holds no more ranks than its bound: a rank past it is not
    /// kept, and a report that lists it makes no value.
    #[test]
    fn a_rank_past_the_keepers_bound_is_not_kept() {
        let now = Instant::now();
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        let bound = i32::try_from(MAX_RANKS).expect("a bound of this build");
        let full = report((0..bound).map(|rank| ranked(rank, depth(1.0))).collect());
        let handed = kept(&mut keeper, &full, None, &columns, (now, now));
        assert_eq!(handed, [value("depth", "items", f64::from(bound), 0)]);
        let past = report(vec![ranked(0, depth(1.0)), ranked(bound, depth(1.0))]);
        assert!(kept(&mut keeper, &past, None, &columns, (now, now)).is_empty());
        assert_eq!(keeper.ranks.len(), MAX_RANKS);
    }

    /// A statement stands for the column it was read under. Another list
    /// of columns ends them all, though it declares the same signal, and
    /// the next record reads each rank's entry again, under the instant
    /// the entry says. A name an entry leaves out is gone with the list
    /// its statement was read under.
    #[test]
    fn another_list_of_columns_ends_the_statements_and_the_entries_are_read_again() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let first = depth_alone();
        let mut pushed = Pushed::default();
        let mut keeper = RankValues::default();
        let alone = pushed.record(1, depth(2.0), at(0));
        kept(&mut keeper, alone, Some(1), &first, (at(0), now));
        let both = pushed.record(0, depth(1.0), at(0));
        kept(&mut keeper, both, Some(0), &first, (at(0), now));
        // Rank 0 states 4 under another list: rank 1's entry is read
        // again, as old as it says.
        let again = depth_alone();
        let moved = pushed.record(0, depth(4.0), at(40));
        let read = kept(&mut keeper, moved, Some(0), &again, (at(40), now));
        assert_eq!(read, [value("depth", "items", 6.0, 1_000)]);
        // Rank 1's next record leaves the depth out, and its statement
        // stands under the same list.
        let bare = pushed.record(1, Vec::new(), at(50));
        let stands = kept(&mut keeper, bare, Some(1), &again, (at(50), now));
        assert_eq!(stands, [value("depth", "items", 6.0, 1_000)]);
        // Under a third list the entry states no depth: what rank 1
        // stated is gone with the list it was read under.
        let third = depth_alone();
        let next = pushed.record(0, depth(4.0), at(70));
        let gone = kept(&mut keeper, next, Some(0), &third, (at(70), now));
        assert!(gone.is_empty());
    }

    /// A worker's first records list one rank, and a keeper is made at the
    /// first report of several. Nothing stated before is lost: the report
    /// holds the first rank's entry, which says its own instant. An entry
    /// is read again where it says another instant, as one an answer was
    /// laid over does, and not while it says the same.
    #[test]
    fn a_ranks_entry_is_read_under_the_instant_it_says() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(10_000);
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        // Rank 1 stated 100 at the start. Rank 0's first record comes
        // nine seconds later.
        let file = report(vec![
            ranked(0, depth(1.0)),
            ranked_at(1, depth(100.0), at(0)),
        ]);
        let joined = kept(&mut keeper, &file, Some(0), &columns, (at(9_500), now));
        assert_eq!(joined, [value("depth", "items", 101.0, 10_000)]);
        // The entry says the same instant: it is not read again.
        let same = report(vec![ranked(0, depth(1.0)), ranked_at(1, depth(7.0), at(0))]);
        let stands = kept(&mut keeper, &same, Some(0), &columns, (at(9_900), now));
        assert_eq!(stands, [value("depth", "items", 101.0, 10_000)]);
        // An answer laid over the entry says the poll's start.
        let laid = report(vec![
            ranked(0, depth(1.0)),
            ranked_at(1, depth(7.0), at(9_000)),
        ]);
        let read = kept(&mut keeper, &laid, Some(0), &columns, (at(9_950), now));
        assert_eq!(read, [value("depth", "items", 8.0, 1_000)]);
    }

    /// A polled answer may list its ranks out of rank order: it is kept
    /// all the same, and the kept ranks stand in rank order.
    #[test]
    fn an_answer_out_of_rank_order_is_kept_as_one_in_order() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        let answer = report(vec![
            ranked(2, depth(4.0)),
            ranked(0, depth(1.0)),
            ranked(1, depth(2.0)),
        ]);
        assert!(RankValues::keeps(&answer, &columns));
        let whole = kept(&mut keeper, &answer, None, &columns, (at(0), now));
        assert_eq!(whole, [value("depth", "items", 7.0, 1_000)]);
        let in_order: Vec<i32> = keeper.ranks.iter().map(|kept| kept.rank).collect();
        assert_eq!(in_order, [0, 1, 2]);
        // The next answer lists two of them: their statements make its
        // value, and the third stands.
        let fewer = report(vec![ranked(2, depth(4.0)), ranked(0, depth(3.0))]);
        let two = kept(&mut keeper, &fewer, None, &columns, (at(10), now));
        assert_eq!(two, [value("depth", "items", 7.0, 990)]);
        assert_eq!(keeper.ranks.len(), 3);
    }

    /// A report of one rank is one sample: it is handed as it is, by a
    /// record and by an answer. A keeper that is told one keeps its
    /// statement for a later report of several ranks. A document that
    /// declares no signal keeps nothing and takes no room.
    #[test]
    fn a_report_of_one_rank_is_handed_as_it_is_and_kept_for_a_later_one() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let columns = depth_alone();
        // Under no declared signal nothing is read, of one rank or of two.
        let none: Arc<[SignalColumn]> = Vec::new().into();
        let two = report(vec![ranked(0, depth(1.0)), ranked(1, depth(2.0))]);
        let mut idle = RankValues::default();
        assert!(!RankValues::keeps(&two, &none));
        assert!(kept(&mut idle, &two, Some(0), &none, (now, now)).is_empty());
        assert!(idle.columns.is_none());
        assert_eq!((idle.ranks.capacity(), idle.made.capacity()), (0, 0));
        assert!(RankValues::keeps(&two, &columns));
        // Rank 1 alone, by a record and then by an answer.
        let mut keeper = RankValues::default();
        let one = report(vec![ranked(1, depth(2.0))]);
        assert!(!RankValues::keeps(&one, &columns));
        for (record_of, ms) in [(Some(1), 0), (None, 10)] {
            let handed = kept(&mut keeper, &one, record_of, &columns, (at(ms), now));
            let age = 1_000 - u128::from(ms);
            assert_eq!(handed, [value("depth", "items", 2.0, age)]);
        }
        // Rank 0's first record counts what rank 1 stated alone, as old as
        // the answer that stated it last and was laid on the report.
        let laid = ranked_at(1, depth(2.0), at(10));
        let two = report(vec![ranked(0, depth(1.0)), laid]);
        let both = kept(&mut keeper, &two, Some(0), &columns, (at(500), now));
        assert_eq!(both, [value("depth", "items", 3.0, 990)]);
    }

    /// A gateway's rollup lists its workers, and they repeat a rank: kept
    /// by rank, one worker's statement would stand in the place of
    /// another's. An answer that lists a rank twice is as little an
    /// engine's ranks. Each is one sample, handed as it is, and nothing of
    /// it is kept.
    #[test]
    fn a_report_that_repeats_a_rank_is_one_sample_and_keeps_nothing() {
        let now = Instant::now();
        let columns = depth_alone();
        let of = |worker: &str, rank: i32, value: f64| SchedulerLoadSnapshot {
            worker: Some(worker.to_owned()),
            ..ranked(rank, depth(value))
        };
        let rollup = report(vec![of("a", 0, 3.0), of("b", 0, 4.0)]);
        // Two of a gateway's workers at ranks of their own are no engine's
        // ranks either.
        let apart = report(vec![of("a", 0, 3.0), of("b", 1, 4.0)]);
        let twice = report(vec![ranked(0, depth(3.0)), ranked(0, depth(4.0))]);
        let mut keeper = RankValues::default();
        for one_sample in [&rollup, &apart, &twice] {
            assert!(!RankValues::keeps(one_sample, &columns));
            let handed = kept(&mut keeper, one_sample, None, &columns, (now, now));
            assert_eq!(handed, [value("depth", "items", 7.0, 0)]);
        }
        assert!(keeper.columns.is_none());
        assert!(keeper.ranks.is_empty());
    }
}
