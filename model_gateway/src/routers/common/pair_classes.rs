//! The pair classes of a snapshot's prefill and decode pools, as Grand
//! Central reads them.
//!
//! Which prefill pairs with which decode is decided once a membership by the
//! snapshot's pair index ([`PdPairIndex`]). The engine's pair decision reads
//! that as a class on each worker's record and a table of the classes that
//! pair. This module numbers the classes from the index's partner lists, so
//! the pairing rule is written once, in the index, whoever places the pair.
//! It says too which partners a prefill is paired with first, by the order
//! the classic selection gives its candidates on the index's wire.

use std::{ptr, sync::Arc};

use smg_grand_central::engine::PartnerClasses;
use smg_worker::identity::basic::{RuntimeType, Worker};
use tracing::warn;

use super::pair_index::PdPairIndex;

/// A list of workers as the index holds it: a pool, or one prefill's
/// partners.
type Workers = Arc<[Arc<dyn Worker>]>;

/// Whether two handles name one worker object.
fn same(one: &Arc<dyn Worker>, other: &Arc<dyn Worker>) -> bool {
    ptr::addr_eq(Arc::as_ptr(one), Arc::as_ptr(other))
}

/// The places in `pool` of the workers of `list`, which holds workers of
/// `pool` in the pool's order.
fn places(pool: &Workers, list: &Workers) -> Vec<u32> {
    if Arc::ptr_eq(pool, list) {
        return (0..pool.len() as u32).collect();
    }
    let mut listed = list.iter().peekable();
    let places: Vec<u32> = (0u32..)
        .zip(pool.iter())
        .filter(|(_, worker)| listed.next_if(|next| same(next, worker)).is_some())
        .map(|(place, _)| place)
        .collect();
    debug_assert_eq!(
        places.len(),
        list.len(),
        "a partner list in its pool's order"
    );
    places
}

/// One partner list as the prefills of one runtime hold it.
struct Listed<'a> {
    list: &'a Workers,
    /// The runtime its prefills are paired on first, where they have one.
    runtime: Option<RuntimeType>,
    /// The list's places in the decode pool.
    places: Vec<u32>,
    /// Those of the places whose decode is on `runtime`.
    first: Vec<u32>,
}

/// The class of every worker of `index`'s two pools, in each pool's order,
/// and the table of the classes that pair, with the partners each prefill
/// is paired with first.
///
/// On a wire that carries one runtime's request
/// ([`PdWire::runtime_specific`](super::pair_index::PdWire::runtime_specific))
/// a prefill of a known runtime is paired first with its partners on that
/// runtime: a pair on one known runtime goes ahead of a pair with a worker
/// whose runtime is unknown, as the classic selection orders its candidates
/// (`placement.rs:2479-2502`, and the pick's decode at `:2615-2618`, on main
/// `30f636706`). The other wire states no first partner, and neither does a
/// prefill of no known runtime.
///
/// Called once an index, at Grand Central's first pair decision over it
/// ([`PdPairIndex::lent`]). Prefills of one descriptor share one partner
/// list, so a list is turned into places once, however many prefills hold
/// it.
pub(super) fn of(index: &PdPairIndex) -> PartnerClasses {
    let apart = index.wire.runtime_specific();
    let runtime_of = |worker: &Arc<dyn Worker>| worker.metadata().spec.runtime_type;
    let mut lists: Vec<Listed<'_>> = Vec::new();
    // Each prefill with a partner, by its list's place in `lists`.
    let listed: Vec<usize> = index
        .prefill
        .iter()
        .zip(&index.partners)
        .map(|(prefill, list)| {
            let runtime = Some(runtime_of(prefill)).filter(|own| apart && own.is_specified());
            let known = lists
                .iter()
                .position(|held| Arc::ptr_eq(held.list, list) && held.runtime == runtime);
            known.unwrap_or_else(|| {
                let places = places(&index.decode_pool, list);
                // A place is a decode of the pool, so no decode's runtime
                // equals a prefill's that has none.
                let on_it = |place: &u32| {
                    let decode = index.decode_pool.get(*place as usize);
                    decode.map(runtime_of) == runtime
                };
                let first = places.iter().copied().filter(on_it).collect();
                lists.push(Listed {
                    list,
                    runtime,
                    places,
                    first,
                });
                lists.len() - 1
            })
        })
        .collect();
    // The index lists the prefills that have a partner, in the pool's
    // order: every other prefill of the pool pairs with nobody.
    let mut paired = index.prefill.iter().zip(&listed).peekable();
    let partners = index.prefill_pool.iter().map(|worker| {
        let held = paired.next_if(|(prefill, _)| same(prefill, worker));
        held.map_or((&[][..], &[][..]), |(_, &list)| {
            (lists[list].places.as_slice(), lists[list].first.as_slice())
        })
    });
    let classes = PartnerClasses::number_first(index.decode_pool.len(), partners);
    // Asked once, where the table is built, in every build: a decision of a
    // release build checks none.
    if let Err(clause) = classes.table().validate() {
        warn!(
            %clause,
            "The pair classes' table is not one Grand Central honours whole"
        );
    }
    if let Err(clause) = classes.first().within(&classes.table()) {
        warn!(
            %clause,
            "The pair classes' first partners do not stand within their table"
        );
    }
    classes
}

#[cfg(test)]
#[path = "../../../tests/unit/routers/common/pair_classes/tests.rs"]
mod tests;
