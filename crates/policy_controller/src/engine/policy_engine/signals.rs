//! A document's named signals, read beside the load snapshot.
//!
//! A document can declare named values of a worker's load report
//! ([`SignalSpec`](crate::document::spec::SignalSpec)), and a stage can read
//! one for each worker it is asked about
//! ([`SignalColumn::read`](crate::load::SignalColumn::read)). The stage reads
//! them in the request's facts
//! ([`RequestFacts::signal_cells`](crate::engine::RequestFacts::signal_cells)).
//!
//! The engine makes those cells itself. A worker's report stands in the load
//! snapshot a decision holds, the one its request gives or the one the
//! engine's own task keeps, and the named values are part of that report.
//! So a list's cells are made where its load reports are converted: once a
//! snapshot a list ([`PoolRecords::cells`]). A host hands its load channel
//! and nothing more, and cells a caller put on a request are not read.
//!
//! **Who is lent them.** A single request whose profile reads a signal
//! ([`decide`]), and each side of a pair whose profile reads one: the
//! prefill side its prefill list's cells, the decode side its decode
//! list's ([`sides`]). An encode request decided over its three lists is
//! lent none, for its encoders or for the two sides of its pair. Decided
//! as its pair, over the pair's two lists, it is lent what a pair is.
//!
//! **What a value is.** The cells hold what the snapshot's reports state,
//! and nothing of an earlier snapshot. A worker with no report, a name its
//! report leaves out, a value under another unit than the document's and a
//! value out of range all read as no value. A value ends by its signal's
//! maximum age, counted from its own sample: the report's sample instant
//! as the snapshot states it, less the age the report states for the
//! value. That instant is the one the worker's load row is dated by. For a
//! polled answer it is the instant the load monitor read the answer.
//!
//! **A worker that pushes its load.** Such a worker's report holds the named
//! values of its last record alone, and a snapshot takes the report once a
//! publication. So its value is read in the snapshots behind a record that
//! stated it, and reads as no value behind a record that left it out. A
//! polled worker's every answer states its values whole, and is read so.
//!
//! The two fixed token counts are the report's to keep. Behind a record
//! that leaves one out, a report of one rank offers the count as it was
//! last stated, as old as that statement
//! ([`SchedulerLoadSnapshot::named_values`]): it is read until its signal's
//! maximum age, as any value is. A signal that declares none reads it
//! until a record states the count again.
//!
//! [`SchedulerLoadSnapshot::named_values`]: smg_worker::worker::SchedulerLoadSnapshot::named_values
//!
//! **A limit: a worker object that stands for several ranks.** Its report
//! states one sample instant for all its ranks. Where it pushes, the report
//! lists each rank at that rank's latest record under the newest record's
//! instant: a rank that stopped reporting would keep its last values under
//! a new date for as long as another rank reports, and no maximum age would
//! end them. Where it is polled, the answer carries the entries of the
//! ranks that pushed meanwhile. A snapshot states no instant a rank. So the
//! engine reads no named value of a report of several ranks: the worker
//! reads as one with no value, and the signal's rule across ranks has
//! nothing to make one. Where each rank is a row of the list, a worker
//! object of its own, the row's report holds its rank alone: each rank's
//! row reads its own values, however they arrive.
//!
//! [`PoolRecords::cells`]: crate::engine::records::PoolRecords::cells

use std::{sync::Arc, time::Instant};

use smg_worker::load::state::LoadSnapshot;

use super::{PolicyEngine, PoolMembers};
use crate::{
    document::spec::AppliedSpec,
    engine::{
        decision::Decision,
        decision_engine::{Request, RequestMaker},
        records::{DecisionClock, HeldCells, Live},
        shadow::Shadowed,
        view::{Phase, RequestFacts},
    },
};

impl PoolMembers {
    /// Whether a decision of `phase` for `model` reads a named signal: a
    /// stage of its profile does, or one of its binding's shadow
    /// ([`needs_for`]). The decode phase answers for the decode profile of
    /// the model's pair too, as the needs do. Found once a list, as the
    /// profile the engine picks alone is: a request of the list's model at
    /// the list's phase pays no second read of its binding. Any other model
    /// or phase asks the document.
    ///
    /// [`needs_for`]: crate::engine::framework::Engine::needs_for
    pub(super) fn reads_signal(&self, spec: &AppliedSpec, model: &str, phase: Phase) -> bool {
        let find = || spec.engine().needs_for(model, phase).signal_cells;
        let (named, asked, reads) = self.signal.get_or_init(|| (model.into(), phase, find()));
        if **named == *model && *asked == phase {
            *reads
        } else {
            find()
        }
    }
}

/// One made single request's decision where its profile reads a named
/// signal: the decision every made single request takes over its snapshot
/// ([`PolicyEngine::decide_with_loads`]), made over a copy of the request
/// with its list's signal cells in its facts. The cells a caller put on the
/// request are not read: under no snapshot the copy carries none.
///
/// The cells are made of `loads`, the one snapshot the decision reads its
/// records from, and held here to the decision's end: the copy borrows
/// them, so it ends here too, and the request's entry issues by the
/// request it made. The cells are in the copy's facts from the decision's
/// first read. So the keyed read, which asks the profile's filters of a
/// pin's worker, reads them, and so does a shadow profile behind the live
/// one. A request the decision declines was lent them all the same.
///
/// Out of line: a document that reads no signal pays one test for it, in
/// the function that decides a made single request. A request whose
/// profile reads none is not decided here ([`PoolMembers::reads_signal`]):
/// it is decided as its maker made it, and no stage of it holds a signal's
/// column.
///
/// `maker` is the request's own, handed on with the copy: where the
/// request's host lends a token tree, the copy's decision asks it for the
/// request's entry there. It is taken as a trait object, so this function
/// is compiled once, and the copy's decision is no second caller of the
/// decision a request with no cells takes: as one, it put that decision
/// out of line for every request.
#[inline(never)]
pub(super) fn decide(
    own: &PolicyEngine,
    request: &mut Request<'_>,
    members: &PoolMembers,
    maker: &dyn RequestMaker,
    loads: Option<&Arc<LoadSnapshot>>,
    now: Instant,
    said: &mut Option<Shadowed>,
) -> Decision {
    let spec = &own.spec;
    let live = Live::at(spec.engine(), &own.clock, now, loads, None);
    let records = members.records(spec, &own.clock);
    let held = records.cells(members.workers(), &live);
    let mut lent = Request {
        facts: RequestFacts {
            signal_cells: held.as_ref().and_then(HeldCells::cells),
            ..request.facts
        },
        ..*request
    };
    let decision = own.decide_with_loads(&mut lent, members, maker, loads, now, said);
    hand_back(request, &lent);
    decision
}

/// The signal cells one decision of a pair lends its two sides, the
/// prefills' first: for each side whose profile reads a named signal, its
/// own list's cells of the decision's snapshot ([`PoolRecords::cells`]).
/// The prefill list is asked for the prefill phase and the decode list for
/// the decode phase ([`PoolMembers::reads_signal`]). `None` for a side that
/// reads no signal, and for both under no snapshot, where every worker
/// reads no value.
///
/// `first` says the decision is the pair's first. The decode decided again
/// asks no prefill stage, so its prefill side is lent nothing. Its decode
/// side is lent the cells of `live`, the snapshot of its own call, and
/// never those the pair's first decision read.
///
/// The cells are held by the caller to the decision's end, and the pair's
/// pools carry them
/// ([`PairPools`](crate::engine::framework::PairPools)): the engine writes
/// each side's facts itself, so no copy of the request is made, and cells
/// a caller put on the request are read by neither side.
///
/// Called behind the two lists' records' writes. A write converts its
/// list's reports where the list holds none of the decision's snapshot,
/// and the cells are made by that conversion: so a side's cells are mostly
/// its list's own, by one read. A decision that was not the one to convert
/// writes the cells of its own snapshot. The calling thread keeps one
/// buffer for that: where both sides write their own, the second
/// allocates.
///
/// Out of line: a document that reads no signal pays one test for it, in
/// the pair's decision ([`PairLists::decide`]), which hands the engine's
/// values of its two lists as `members`, the prefills' first.
///
/// [`PairLists::decide`]: super::pair::PairLists::decide
/// [`PoolRecords::cells`]: crate::engine::records::PoolRecords::cells
#[inline(never)]
pub(super) fn sides(
    [prefill, decode]: [&PoolMembers; 2],
    spec: &AppliedSpec,
    clock: &DecisionClock,
    live: &Live<'_>,
    model: &str,
    first: bool,
) -> [Option<HeldCells>; 2] {
    let lent = |members: &PoolMembers, phase| {
        if !members.reads_signal(spec, model, phase) {
            return None;
        }
        let records = members.records(spec, clock);
        records.cells(members.workers(), live)
    };
    let prefills = if first {
        lent(prefill, Phase::Prefill)
    } else {
        None
    };
    [prefills, lent(decode, Phase::Decode)]
}

/// Writes into `request` every fact a decision writes, from `lent`, the
/// copy the decision was made over: the request's entry issues by the
/// request, so what a decision wrote must stand there.
///
/// A decision writes one fact today, its instant
/// ([`RequestFacts::now_micros`]), which the issue reads as the lease's. A
/// decision that comes to write another fact hands it back here.
fn hand_back(request: &mut Request<'_>, lent: &Request<'_>) {
    request.facts.now_micros = lent.facts.now_micros;
}

#[cfg(test)]
mod tests;
