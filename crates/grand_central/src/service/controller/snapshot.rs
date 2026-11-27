//! The operators' snapshot (design 5.1 and 8): the control scope's read of
//! what this replica holds, a slice at a time, under the header every
//! snapshot carries. This build serves two slices: the worker rows, and the
//! spec's status.
//!
//! **A slice asked for and not served is refused by its name, never
//! answered empty.** An empty slice reads as a controller that holds
//! nothing of the kind. A request that names no slice asks for every one,
//! so it is refused too until every slice is served.
//!
//! The answer is redacted ([`snapshot`]). Identifiers in clear are
//! an admin's to ask and this build holds no role, so that request is
//! refused by its name: it is never answered redacted in silence.
//!
//! The rows are bounded twice: by the caller's `limit`, and by the bytes
//! of a watch's page whatever the caller asked, so the answer fits a frame.
//! `truncated` says that a bound left a row out. The rows come in the order
//! of their registration keys, so a cut answer names the same rows at each
//! call while the rows stand.
//!
//! The rows come by one of three roads ([`read`]). A cut kept at the log's
//! last watched write is shared with no copy, as a watch's start shares it
//! ([`Feed::rows_at_head`](super::watch_workers::Feed::rows_at_head)).
//! Where another call is let in or a watch is open, the first read at a
//! place cuts every row from the worker objects and keeps the copy, and
//! each read that starts there shares it: calls that come together take the
//! holder's turn once a write, as watches do. A call with no limit and no
//! models takes that road too, alone or not: it would copy every row
//! anyway, and a call let in while it runs shares the copy. The last call
//! out lets the copy go where no watch is open to share it
//! ([`Readers::snapshot`]): a copy of every row is held only while a reader
//! can share it. Any other call that is alone with no watch open reads the
//! rows it may tell and keeps nothing ([`Objects::rows_serving`]): its
//! limit's worth, or every row its models serve where it states no limit.
//!
//! A copy holds two locks, one after the other taken. The holder's turn
//! is held for the rows: a registration call's write, a standby's applier
//! and a door's read of a row wait for the copy there. The store's
//! membership lock is held inside it for the log's place and one status a
//! row: a status tell and the store's other writes of that lock wait for
//! that alone. No decision takes either. Whatever waits does so on the
//! thread that brought it.
//! **One snapshot is taken at a time, off the runtime's workers**
//! ([`in_turn`]): one operator's copy is made at a time, and the take itself
//! holds no thread a stream of the listener runs on. A watch's open and a
//! standby's base copy by themselves, in no turn.
//!
//! No standby is restored from the answer: redacted, and cut where a bound
//! bit, it names no place of the log.

use std::{sync::Arc, time::Instant};

use prost::encoding::message;
use tonic::Status;

use super::{generations::Generations, watch_workers::Readers, wire};
use crate::{
    document::operator::snapshot::{self, Rows, WorkerRowsQuery},
    service::proto::{self, snapshot_request::Slice},
    store::MemoryStore,
    worker_host::{Members, Objects},
};

/// Every slice of the contract, in its order.
const SLICES: [Slice; 9] = [
    Slice::Workers,
    Slice::Health,
    Slice::Loads,
    Slice::Sessions,
    Slice::Leases,
    Slice::Spec,
    Slice::Plugins,
    Slice::KvIndex,
    Slice::Gateways,
];

/// The refusal of identifiers in clear.
const PLAIN: &str =
    "plain_identifiers: identifiers in clear are an admin's to ask, and this build holds no role";

/// The refusal of the spec slice by a controller that was given no spec
/// source, in the spec calls' own words.
const NO_SOURCE: &str =
    "slices: SLICE_SPEC is not served by a controller started with no spec source";

/// How many snapshots may wait their turn behind the one that is taken: a
/// call past them is told to ask again.
pub(crate) const SNAPSHOTS_WAITING: usize = 32;

/// Whether this build serves `slice`.
const fn served(slice: Slice) -> bool {
    matches!(slice, Slice::Workers | Slice::Spec)
}

/// The bit of `slice` in a set of slices: its number in the contract.
fn bit(slice: Slice) -> u32 {
    1 << i32::from(slice)
}

/// The slices `keep` keeps, by the contract's names and in its order.
fn names(keep: impl Fn(Slice) -> bool) -> String {
    SLICES
        .into_iter()
        .filter(|slice| keep(*slice))
        .map(|slice| slice.as_str_name())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Takes a snapshot off the runtime's workers, in the turn of `readers`
/// ([`Readers::snapshot`]), which is held until the take ends. The
/// readers let their kept cut go when the last place goes back, including
/// where a caller leaves while it waits.
pub(super) async fn in_turn<T: Send + 'static>(
    readers: &Arc<Readers>,
    take: impl FnOnce() -> T + Send + 'static,
) -> Result<T, Status> {
    let Some(held) = readers.snapshot().await else {
        return Err(Status::resource_exhausted(format!(
            "{SNAPSHOTS_WAITING} snapshots wait their turn: ask again later"
        )));
    };
    tokio::task::spawn_blocking(move || {
        let _held = held;
        take()
    })
    .await
    .map_err(|join| Status::internal(format!("the snapshot did not end: {join}")))
}

/// What a snapshot's caller asked for, once the request holds this build's
/// rules ([`asked`]).
#[derive(Debug)]
pub(crate) struct SnapshotQuery {
    workers: bool,
    spec: bool,
    rows: WorkerRowsQuery,
}

impl SnapshotQuery {
    /// The refusal of the spec slice, by its name, where the controller was
    /// given no spec source (`bound` says whether it was). The handler
    /// reads it ahead of the turn, as it reads each rule of the request: a
    /// refused call takes no place.
    pub(crate) fn sourced(&self, bound: bool) -> Result<(), Status> {
        if self.spec && !bound {
            return Err(Status::unimplemented(NO_SOURCE));
        }
        Ok(())
    }
}

/// What `request` asks for, or the refusal of a field by its name, the
/// fields in the contract's order. A number that names no slice is the
/// caller's error. A slice this build does not serve, and identifiers in
/// clear, are this build's: `UNIMPLEMENTED`. The models are held to a
/// watch's bounds: the list is read at each row.
pub(crate) fn asked(request: proto::SnapshotRequest) -> Result<SnapshotQuery, Status> {
    let mut named = 0;
    for &number in &request.slices {
        match Slice::try_from(number) {
            Ok(slice) if slice != Slice::Unspecified => named |= bit(slice),
            _ => {
                return Err(Status::invalid_argument(format!(
                    "slices: {number} names no slice"
                )));
            }
        }
    }
    let every = request.slices.is_empty();
    if every {
        named = SLICES.into_iter().fold(0, |set, slice| set | bit(slice));
    }
    let unserved = names(|slice| named & bit(slice) != 0 && !served(slice));
    if !unserved.is_empty() {
        let none = if every {
            "none named is every slice, and "
        } else {
            ""
        };
        return Err(Status::unimplemented(format!(
            "slices: {none}this build does not serve {unserved}: it serves {}",
            names(served)
        )));
    }
    if let Some(why) = WorkerRowsQuery::refused(&request.models) {
        return Err(Status::invalid_argument(format!("models: {why}")));
    }
    if request.plain_identifiers {
        return Err(Status::unimplemented(PLAIN));
    }
    Ok(SnapshotQuery {
        workers: named & bit(Slice::Workers) != 0,
        spec: named & bit(Slice::Spec) != 0,
        rows: WorkerRowsQuery {
            models: request.models,
            // Zero leaves the bound of a page's bytes alone.
            limit: match request.limit {
                0 => usize::MAX,
                limit => usize::try_from(limit).unwrap_or(usize::MAX),
            },
        },
    })
}

/// The rows of one take, by the cheapest road that keeps calls that come
/// together at one copy a write. The readers of the feed say whether the
/// take shares a cut ([`Readers::cut`]): one kept at the log's head, with
/// no copy, or the one taken and kept where another call is let in or a
/// watch is open. A call with no limit and no models shares it too: it
/// would copy every row anyway, and a call let in while it runs shares the
/// copy. Any other call that is alone with no watch open reads the rows it
/// may tell and keeps nothing. Each read is the worker objects': their
/// holder cuts it.
pub(crate) fn read(
    asked: &SnapshotQuery,
    readers: &Readers,
    (objects, store): (&Objects<Members>, &MemoryStore),
) -> Rows {
    let every = asked.rows.limit == usize::MAX && asked.rows.models.is_empty();
    if let Some(cut) = readers.cut(every, objects, store) {
        return Rows::Shared(cut);
    }
    let (rows, serving) = objects.rows_serving(store, &asked.rows.models, asked.rows.limit);
    Rows::Told(rows, serving)
}

/// The snapshot `asked` for, as this replica holds it at `now` under
/// `term`: the header, then each slice asked. The rows are `read`'s, which
/// is called where the worker rows are asked and never else. The spec slice
/// of a controller with no spec source is refused ahead of the turn
/// ([`SnapshotQuery::sourced`]), and the take keeps its own read of the source,
/// before a row is copied. The header's generation is the spec slice's own
/// where that slice is told: one read. The answer names no place of the
/// log, no id table and no document: it is no base a standby starts from.
pub(crate) fn taken(
    asked: &SnapshotQuery,
    term: u64,
    controller_id: &str,
    read: impl FnOnce() -> Rows,
    generations: &Generations,
    now: Instant,
) -> Result<proto::Snapshot, Status> {
    let spec = if asked.spec {
        let status = generations
            .status()
            .ok_or_else(|| Status::unimplemented(NO_SOURCE))?;
        Some(wire::spec_status(status))
    } else {
        None
    };
    let (workers, truncated) = if asked.workers {
        rows(&asked.rows, read(), now)
    } else {
        (Vec::new(), false)
    };
    Ok(proto::Snapshot {
        term,
        generation: spec
            .as_ref()
            .map_or_else(|| generations.generation(), |spec| spec.generation),
        controller_id: controller_id.to_owned(),
        workers,
        spec,
        truncated,
        ..proto::Snapshot::default()
    })
}

/// A controller page keeps its bound on the encoded rows' bytes.
fn rows(asked: &WorkerRowsQuery, read: Rows, now: Instant) -> (Vec<proto::WorkerRow>, bool) {
    snapshot::worker_rows_page_with(asked, read, now, true, |row| {
        let told = wire::snapshot_row(&row, now);
        let framed = message::encoded_len(1, &told);
        (told, framed)
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/service/controller/snapshot/tests.rs"]
mod tests;
