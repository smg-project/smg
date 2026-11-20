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
                stated.unit,
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
        unit: Arc<str>,
        value: f64,
        sampled: Instant,
    ) {
        match slot {
            None if nth == 0 => {
                *slot = Some(Self {
                    ranks: 1,
                    unit,
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
                if row.unit == unit {
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
/// A report of one rank keeps nothing and is handed as it is: the store's
/// cell holds that worker's last value. It ends no statement either: a
/// worker's first records list one rank until the others report. What is
/// kept is bounded by the declared signals times the report's ranks. A
/// rank that a report of several ranks lists no more takes its statements
/// with it. A statement stands for the column it was read under, so a
/// document that declares other columns ends them all, as it starts the
/// store's cells with no value.
#[derive(Debug, Default)]
pub struct RankValues {
    /// The columns the statements were read under.
    columns: Option<Arc<[SignalColumn]>>,
    /// The ranks that stated a declared name, in rank order.
    ranks: Vec<RankStated>,
    /// The slots of one reduction, kept for their room.
    made: Vec<Option<Made>>,
}

/// What one rank last stated: one statement a declared name at the most.
#[derive(Debug)]
struct RankStated {
    rank: i32,
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

// A rank's first statement of each declared name is marked in the bits of
// one word ([`RankValues::keep`]): a document's cap of names fits them.
const _: () = assert!(MAX_SIGNALS <= u64::BITS as usize);

/// Whether `report` lists each of its ranks once.
fn lists_each_rank_once(report: &WorkerLoadResponse) -> bool {
    let mut rest = report.loads.iter();
    while let Some(rank) = rest.next() {
        if rest.clone().any(|later| later.dp_rank == rank.dp_rank) {
            return false;
        }
    }
    true
}

impl RankValues {
    /// Whether a keeper keeps `report`: one of several ranks of one
    /// engine, each listed once, under a document that declares a signal.
    /// Any other report is one sample, and [`signal_values`] makes its
    /// values with nothing kept. A gateway's rollup of its own workers is
    /// one: its entries are workers, and they repeat a rank, so one
    /// worker's statement would be kept in the place of another's.
    pub fn keeps(report: &WorkerLoadResponse, columns: &[SignalColumn]) -> bool {
        !columns.is_empty()
            && report.loads.len() > 1
            && report.ranks_are_dp_ranks()
            && lists_each_rank_once(report)
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
        record_of: Option<i32>,
        columns: &Arc<[SignalColumn]>,
        sampled_at: Instant,
        each: impl FnMut(&SignalColumn, &str, f64, Instant),
    ) {
        if !self.keep(report, record_of, columns, sampled_at) {
            signal_values(report, columns, sampled_at, each);
            return;
        }
        self.made.clear();
        self.made.resize(columns.len(), None);
        for (nth, rank) in (0u32..).zip(&report.loads) {
            let kept = self.ranks.iter().find(|kept| kept.rank == rank.dp_rank);
            for stated in kept.into_iter().flat_map(|kept| &kept.stated) {
                let slot = self.made.get_mut(stated.column);
                let (Some(slot), Some(column)) = (slot, columns.get(stated.column)) else {
                    continue;
                };
                let unit = Arc::clone(&stated.unit);
                let across = column.across_ranks();
                Made::join(slot, nth, across, unit, stated.value, stated.sampled);
            }
        }
        let ranks = report.loads.len() as u32;
        hand(columns, self.made.drain(..), ranks, each);
    }

    /// Keeps what a report of several ranks states, and says whether
    /// `report` is one ([`Self::keeps`]): nothing is kept of any other.
    ///
    /// `record_of` is the rank a pushed record came from. That rank's
    /// entry alone is the record's own, sampled at `sampled_at`: the other
    /// entries are what their ranks' earlier records left, and are not
    /// read. `None` is a polled answer, one sample of every rank.
    ///
    /// A rank's statement takes the place of its last one of that name,
    /// unless that one was sampled later: a later report of an older
    /// sample leaves it, as the store's cell does. A name the rank does
    /// not state keeps its last statement. A rank that states a name twice
    /// is read at its first statement.
    pub fn keep(
        &mut self,
        report: &WorkerLoadResponse,
        record_of: Option<i32>,
        columns: &Arc<[SignalColumn]>,
        sampled_at: Instant,
    ) -> bool {
        if !Self::keeps(report, columns) {
            return false;
        }
        let read_under = |kept: &Arc<[SignalColumn]>| Arc::ptr_eq(kept, columns);
        if !self.columns.as_ref().is_some_and(read_under) {
            self.ranks.clear();
            self.columns = Some(Arc::clone(columns));
        }
        let listed = |kept: &RankStated| report.loads.iter().any(|rank| rank.dp_rank == kept.rank);
        self.ranks.retain(listed);
        for rank in &report.loads {
            if record_of.is_some_and(|own| own != rank.dp_rank) {
                continue;
            }
            // Each declared name has a bit, set at the name's first
            // statement. A document declares no more names than the bits.
            let mut read = 0u64;
            for stated in rank.named_values() {
                let column = column_of(columns, &stated.name);
                let bit = column.and_then(|at| 1u64.checked_shl(u32::try_from(at).ok()?));
                let (Some(column), Some(bit)) = (column, bit) else {
                    continue;
                };
                if read & bit != 0 {
                    continue;
                }
                read |= bit;
                let sampled = sampled(&stated, sampled_at);
                self.state(rank.dp_rank, column, stated, sampled);
            }
        }
        true
    }

    /// `rank`'s statement of the name at `column`, sampled at `sampled`,
    /// in the place of the rank's last one of that name unless that one
    /// was sampled later.
    fn state(&mut self, rank: i32, column: usize, stated: NamedValue, sampled: Instant) {
        let at = match self.ranks.binary_search_by_key(&rank, |kept| kept.rank) {
            Ok(at) => at,
            Err(at) => {
                let stated = Vec::new();
                self.ranks.insert(at, RankStated { rank, stated });
                at
            }
        };
        let Some(kept) = self.ranks.get_mut(at) else {
            return;
        };
        let NamedValue { unit, value, .. } = stated;
        match kept.stated.iter_mut().find(|kept| kept.column == column) {
            Some(kept) if kept.sampled > sampled => {}
            Some(kept) => {
                *kept = Stated {
                    column,
                    unit,
                    value,
                    sampled,
                };
            }
            None => kept.stated.push(Stated {
                column,
                unit,
                value,
                sampled,
            }),
        }
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
        let mut handed = Vec::new();
        keeper.values(
            report,
            record_of,
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
        let file = report(vec![ranked(0, depth(1.0)), ranked(1, depth(100.0))]);
        let mut keeper = RankValues::default();
        // Rank 1's record states its own rank alone: the row has no value
        // until rank 0 states the name too.
        let first = kept(&mut keeper, &file, Some(1), &columns, (at(0), at(0)));
        assert!(first.is_empty());
        for second in 1..=9 {
            let now = at(second * 1_000);
            let handed = kept(&mut keeper, &file, Some(0), &columns, (now, now));
            let age = u128::from(second) * 1_000;
            assert_eq!(handed, [value("depth", "items", 101.0, age)]);
        }
        let mut one_sample = Vec::new();
        signal_values(&file, &columns, at(9_000), |_, _, value, sampled| {
            one_sample.push((value, at(9_000).duration_since(sampled).as_millis()));
        });
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
        let mut keeper = RankValues::default();
        let both = report(vec![ranked(0, depth(1.0)), ranked(1, depth(2.0))]);
        kept(&mut keeper, &both, Some(1), &columns, (at(0), now));
        let whole = kept(&mut keeper, &both, Some(0), &columns, (at(10), now));
        assert_eq!(whole, [value("depth", "items", 3.0, 1_000)]);
        // Rank 1's next record states no value, then rank 0 states 3.
        let bare = report(vec![ranked(0, depth(1.0)), ranked(1, Vec::new())]);
        let stands = kept(&mut keeper, &bare, Some(1), &columns, (at(20), now));
        assert_eq!(stands, [value("depth", "items", 3.0, 1_000)]);
        let moved = report(vec![ranked(0, depth(3.0)), ranked(1, Vec::new())]);
        let later = kept(&mut keeper, &moved, Some(0), &columns, (at(30), now));
        assert_eq!(later, [value("depth", "items", 5.0, 1_000)]);
        // One sample of that report makes no value: rank 1 states none.
        let mut one_sample = 0;
        signal_values(&moved, &columns, at(30), |_, _, _, _| one_sample += 1);
        assert_eq!(one_sample, 0);
    }

    #[test]
    fn a_record_states_its_own_rank_alone_and_a_polled_answer_every_rank() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(10_000);
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        // The report lists rank 1 at a record of another instant: rank
        // 0's record does not read that entry.
        let file = report(vec![ranked(0, depth(1.0)), ranked(1, depth(50.0))]);
        let alone = kept(&mut keeper, &file, Some(0), &columns, (at(0), now));
        assert!(alone.is_empty());
        // A polled answer is one sample of every rank.
        let answer = report(vec![ranked(0, depth(4.0)), ranked(1, depth(6.0))]);
        let polled = kept(&mut keeper, &answer, None, &columns, (at(1_000), now));
        assert_eq!(polled, [value("depth", "items", 10.0, 9_000)]);
        // Rank 0's next record: its own statement, beside the answer's of
        // rank 1.
        let next = report(vec![ranked(0, depth(5.0)), ranked(1, depth(50.0))]);
        let pushed = kept(&mut keeper, &next, Some(0), &columns, (at(2_000), now));
        assert_eq!(pushed, [value("depth", "items", 11.0, 9_000)]);
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
        let both = report(vec![ranked(0, depth(1.0)), ranked(1, depth(10.0))]);
        kept(&mut keeper, &both, Some(1), &columns, (at(8_000), now));
        let mut rank_zero = |own: Vec<NamedValue>, ms| {
            let file = report(vec![ranked(0, own), ranked(1, depth(10.0))]);
            kept(&mut keeper, &file, Some(0), &columns, (at(ms), now))
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
        let file = report(vec![ranked(0, zero), ranked(1, one)]);
        let mut keeper = RankValues::default();
        kept(&mut keeper, &file, Some(1), &columns, (at(0), now));
        // Two units make no value, and neither does a name one rank never
        // stated.
        assert_eq!(
            kept(&mut keeper, &file, Some(0), &columns, (at(100), now)),
            [
                value("largest", "ms", 5.0, 1_000),
                value("middle", "ms", 4.0, 1_000),
            ]
        );
    }

    #[test]
    fn a_rank_that_leaves_the_report_and_another_list_of_columns_end_the_statements() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let now = at(1_000);
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        let both = report(vec![ranked(0, depth(1.0)), ranked(1, depth(2.0))]);
        kept(&mut keeper, &both, Some(1), &columns, (at(0), now));
        kept(&mut keeper, &both, Some(0), &columns, (at(0), now));
        // The report lists rank 2 where rank 1 stood: rank 1's statement
        // went with it, and is not there when the rank is listed again.
        let other = report(vec![ranked(0, depth(1.0)), ranked(2, depth(4.0))]);
        let swapped = kept(&mut keeper, &other, Some(2), &columns, (at(10), now));
        assert_eq!(swapped, [value("depth", "items", 5.0, 1_000)]);
        let back = kept(&mut keeper, &both, Some(0), &columns, (at(20), now));
        assert!(back.is_empty());
        let whole = kept(&mut keeper, &both, Some(1), &columns, (at(30), now));
        assert_eq!(whole, [value("depth", "items", 3.0, 980)]);
        // Another list of columns ends every statement, though it declares
        // the same signal: a statement stands for the column it was read
        // under.
        let again = depth_alone();
        let ended = kept(&mut keeper, &both, Some(0), &again, (at(40), now));
        assert!(ended.is_empty());
        let restated = kept(&mut keeper, &both, Some(1), &again, (at(50), now));
        assert_eq!(restated, [value("depth", "items", 3.0, 960)]);
    }

    /// A worker of one rank has its last value in the store's cell: its
    /// report is handed as it is, and the keeper takes no room.
    #[test]
    fn a_report_of_one_rank_keeps_nothing_and_takes_no_room() {
        let now = Instant::now();
        let columns = depth_alone();
        let mut keeper = RankValues::default();
        let one = report(vec![ranked(3, depth(2.0))]);
        assert!(!RankValues::keeps(&one, &columns));
        for record_of in [Some(3), None] {
            let handed = kept(&mut keeper, &one, record_of, &columns, (now, now));
            assert_eq!(handed, [value("depth", "items", 2.0, 0)]);
        }
        // Several ranks under a document that declares no signal keep
        // nothing either.
        let none: Arc<[SignalColumn]> = Vec::new().into();
        let two = report(vec![ranked(0, depth(1.0)), ranked(1, depth(2.0))]);
        assert!(!RankValues::keeps(&two, &none));
        assert!(kept(&mut keeper, &two, Some(0), &none, (now, now)).is_empty());
        assert!(keeper.columns.is_none());
        assert_eq!((keeper.ranks.capacity(), keeper.made.capacity()), (0, 0));
        assert!(RankValues::keeps(&two, &columns));
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
