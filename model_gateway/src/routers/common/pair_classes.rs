//! The pair classes of a snapshot's prefill and decode pools, as the policy
//! engine reads them.
//!
//! Which prefill pairs with which decode is decided once a membership by the
//! snapshot's pair index ([`PdPairIndex`]). The engine's pair decision reads
//! that as a class on each worker's record and a table of the classes that
//! pair. This module numbers the classes from the index's partner lists, so
//! the pairing rule is written once, in the index, whoever places the pair.
//! It says too which partners a prefill is paired with first, by the order
//! the classic selection gives its candidates on the index's wire.

use std::{ptr, sync::Arc};

use smg_policy_controller::engine::PartnerClasses;
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
/// Called once an index, at the policy engine's first pair decision over it
/// ([`PdPairIndex::classes`]). Prefills of one descriptor share one partner
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
            "The pair classes' table is not one the policy engine honours whole"
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
mod tests {
    use openai_protocol::model_card::ModelCard;
    use smg_worker::{
        identity::basic::{ConnectionMode, RuntimeType, WorkerType},
        pd::pairing::PdPairingMode,
        test_support::no_health_check,
    };

    use super::*;
    use crate::routers::{common::pair_index::PdWire, grpc::backend_client::BasicWorkerBuilder};

    fn worker(url: &str, worker_type: WorkerType, connector: Option<&str>) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .health_config(no_health_check());
        if let Some(connector) = connector {
            builder = builder.kv_connector(connector);
        }
        Arc::new(builder.build())
    }

    fn prefill(url: &str, connector: Option<&str>) -> Arc<dyn Worker> {
        worker(url, WorkerType::Prefill, connector)
    }

    /// A worker on `runtime` that names no transport.
    fn on(url: &str, worker_type: WorkerType, runtime: RuntimeType) -> Arc<dyn Worker> {
        let builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new("m"))
            .worker_type(worker_type)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(runtime)
            .health_config(no_health_check());
        Arc::new(builder.build())
    }

    /// The pairs the classes make first, as (prefill place, decode place).
    fn first(classes: &PartnerClasses) -> Vec<(usize, usize)> {
        let (prefills, decodes) = (classes.prefills().len(), classes.decodes().len());
        (0..prefills)
            .flat_map(|prefill| (0..decodes).map(move |decode| (prefill, decode)))
            .filter(|&(prefill, decode)| classes.pairs_first(prefill, decode))
            .collect()
    }

    fn decode(url: &str, connector: Option<&str>) -> Arc<dyn Worker> {
        worker(url, WorkerType::Decode, connector)
    }

    /// The pairs the classes say, as (prefill place, decode place), under
    /// a table the engine honours whole.
    fn classed(classes: &PartnerClasses) -> Vec<(usize, usize)> {
        assert_eq!(classes.table().validate(), Ok(()));
        let (prefills, decodes) = (classes.prefills().len(), classes.decodes().len());
        (0..prefills)
            .flat_map(|prefill| (0..decodes).map(move |decode| (prefill, decode)))
            .filter(|&(prefill, decode)| classes.pairs(prefill, decode))
            .collect()
    }

    /// The pairs the index says, as (prefill place, decode place) in the
    /// two pools.
    fn indexed(index: &PdPairIndex) -> Vec<(usize, usize)> {
        let place = |pool: &[Arc<dyn Worker>], worker: &Arc<dyn Worker>| {
            pool.iter().position(|held| same(held, worker))
        };
        let mut pairs = Vec::new();
        for (prefill, partners) in index.prefill.iter().zip(&index.partners) {
            for decode in partners.iter() {
                let pair =
                    place(&index.prefill_pool, prefill).zip(place(&index.decode_pool, decode));
                pairs.extend(pair);
            }
        }
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn a_partner_list_reads_as_places_in_the_decode_pool() {
        let pool: Arc<[Arc<dyn Worker>]> = Arc::from([
            decode("grpc://d:0", None),
            decode("grpc://d:1", None),
            decode("grpc://d:2", None),
        ]);
        let list: Arc<[Arc<dyn Worker>]> = Arc::from([Arc::clone(&pool[0]), Arc::clone(&pool[2])]);
        assert_eq!(places(&pool, &list), [0, 2]);
        assert_eq!(places(&pool, &pool), [0, 1, 2]);
        assert_eq!(places(&pool, &Arc::from([])), [0u32; 0]);
    }

    #[test]
    fn a_fleet_of_one_transport_is_one_class_a_side() {
        let index = PdPairIndex::build(
            Arc::from([prefill("grpc://p:0", None), prefill("grpc://p:1", None)]),
            Arc::from([decode("grpc://d:0", None), decode("grpc://d:1", None)]),
            PdWire::Grpc,
            PdPairingMode::Lenient,
        );
        let classes = index.classes();
        assert_eq!(classes.prefills(), [1, 1]);
        assert_eq!(classes.decodes(), [1, 1]);
        assert_eq!(classed(classes), indexed(&index));
        assert_eq!(classed(classes).len(), 4);
        // One known runtime on every worker: every partner would go first,
        // so none does.
        assert!(first(classes).is_empty());
    }

    /// On the wire that carries one runtime's request, a prefill of a known
    /// runtime is paired first with its partners on that runtime, ahead of
    /// its partners of no known runtime. A prefill of no known runtime has
    /// no first partner, and the other wire states none. The classic
    /// selection orders its candidates so (`placement.rs:2479-2502` on main
    /// `30f636706`). Under the mode that compares nothing of the handoff
    /// both prefills hold the decode pool itself as their list, and stand
    /// apart by their runtime alone.
    #[test]
    fn a_known_runtimes_partners_go_first_on_a_runtime_specific_wire() {
        let index = |wire: PdWire, mode: PdPairingMode| {
            let prefills = [
                on("grpc://p:known", WorkerType::Prefill, RuntimeType::Vllm),
                on(
                    "grpc://p:unknown",
                    WorkerType::Prefill,
                    RuntimeType::Unspecified,
                ),
            ];
            let decodes = [
                on(
                    "grpc://d:unknown",
                    WorkerType::Decode,
                    RuntimeType::Unspecified,
                ),
                on("grpc://d:known", WorkerType::Decode, RuntimeType::Vllm),
            ];
            PdPairIndex::build(Arc::from(prefills), Arc::from(decodes), wire, mode)
        };
        for (mode, name) in [
            (PdPairingMode::Lenient, "lenient"),
            (PdPairingMode::Off, "off"),
        ] {
            let specific = index(PdWire::Grpc, mode);
            let classes = specific.classes();
            assert_eq!(indexed(&specific), [(0, 0), (0, 1), (1, 0), (1, 1)]);
            assert_eq!(classed(classes), indexed(&specific), "{name}");
            assert_eq!(classes.prefills(), [1, 2], "{name}");
            assert_eq!(classes.decodes(), [1, 2], "{name}");
            assert_eq!(classes.first().within(&classes.table()), Ok(()));
            assert_eq!(first(classes), [(0, 1)], "{name}");
        }
        // The wire that forwards one body to both sides tells no runtime
        // apart: the same pairs, and none of them first.
        let plain = index(PdWire::Http, PdPairingMode::Lenient);
        assert_eq!(classed(plain.classes()), indexed(&plain));
        assert_eq!(plain.classes().prefills(), [1, 1]);
        assert!(first(plain.classes()).is_empty());
    }

    /// Two transports and a prefill that reports none: under the lenient
    /// mode that prefill pairs with both decodes, and each of the others
    /// with its own.
    #[test]
    fn the_classes_say_the_pairs_of_the_index() {
        let index = PdPairIndex::build(
            Arc::from([
                prefill("grpc://p:nixl", Some("NixlConnector")),
                prefill("grpc://p:mooncake", Some("MooncakeConnector")),
                prefill("grpc://p:silent", None),
            ]),
            Arc::from([
                decode("grpc://d:nixl", Some("NixlConnector")),
                decode("grpc://d:mooncake", Some("MooncakeConnector")),
            ]),
            PdWire::Grpc,
            PdPairingMode::Lenient,
        );
        let classes = index.classes();
        assert_eq!(classes.prefills(), [1, 2, 3]);
        assert_eq!(classes.decodes(), [1, 2]);
        assert_eq!(indexed(&index), [(0, 0), (1, 1), (2, 0), (2, 1)]);
        assert_eq!(classed(classes), indexed(&index));
    }

    /// A worker the index pairs with nobody stands in its pool all the
    /// same, as class 0: a record of it is written, and the engine passes
    /// it over.
    #[test]
    fn a_worker_with_no_partner_is_class_zero_at_its_place() {
        let index = PdPairIndex::build(
            Arc::from([
                prefill("grpc://p:mooncake", Some("MooncakeConnector")),
                prefill("grpc://p:nixl:0", Some("NixlConnector")),
                prefill("grpc://p:nixl:1", Some("NixlConnector")),
            ]),
            Arc::from([decode("grpc://d:nixl", Some("NixlConnector"))]),
            PdWire::Grpc,
            PdPairingMode::Lenient,
        );
        let classes = index.classes();
        assert_eq!(classes.prefills(), [0, 1, 1]);
        assert_eq!(classes.decodes(), [1]);
        assert_eq!(indexed(&index), [(1, 0), (2, 0)]);
        assert_eq!(classed(classes), indexed(&index));

        let index = PdPairIndex::build(
            Arc::from([prefill("grpc://p:nixl", Some("NixlConnector"))]),
            Arc::from([
                decode("grpc://d:mooncake", Some("MooncakeConnector")),
                decode("grpc://d:nixl", Some("NixlConnector")),
            ]),
            PdWire::Grpc,
            PdPairingMode::Lenient,
        );
        let classes = index.classes();
        assert_eq!(classes.prefills(), [1]);
        assert_eq!(classes.decodes(), [0, 1]);
        assert_eq!(classed(classes), indexed(&index));
    }

    /// Under the mode that compares nothing every prefill shares the decode
    /// pool itself as its list: the pool is read as its own places.
    #[test]
    fn the_pool_itself_as_a_partner_list_pairs_every_decode() {
        let index = PdPairIndex::build(
            Arc::from([
                prefill("grpc://p:nixl", Some("NixlConnector")),
                prefill("grpc://p:mooncake", Some("MooncakeConnector")),
            ]),
            Arc::from([
                decode("grpc://d:nixl", Some("NixlConnector")),
                decode("grpc://d:mooncake", Some("MooncakeConnector")),
            ]),
            PdWire::Grpc,
            PdPairingMode::Off,
        );
        assert!(Arc::ptr_eq(&index.partners[0], &index.decode_pool));
        let classes = index.classes();
        assert_eq!(classes.prefills(), [1, 1]);
        assert_eq!(classes.decodes(), [1, 1]);
        assert_eq!(classed(classes), indexed(&index));
    }

    #[test]
    fn two_empty_pools_have_no_class() {
        let index = PdPairIndex::empty(PdWire::Http);
        let classes = index.classes();
        assert!(classes.prefills().is_empty());
        assert!(classes.decodes().is_empty());
        assert!(classed(classes).is_empty());
    }

    /// The classes are numbered once an index and kept with it.
    #[test]
    fn an_index_numbers_its_classes_once() {
        let index = PdPairIndex::build(
            Arc::from([prefill("grpc://p:0", None)]),
            Arc::from([decode("grpc://d:0", None)]),
            PdWire::Grpc,
            PdPairingMode::Off,
        );
        assert!(ptr::eq(index.classes(), index.classes()));
    }
}
