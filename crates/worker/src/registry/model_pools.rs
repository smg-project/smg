//! A generation's members with their lists by pool ([`ModelPools`]): the
//! value of a generation for a host that places a request among a part of a
//! model's workers.
//!
//! A pool is the host's own: the part of a generation's members one of its
//! routes places among. The host names each pool by a key ([`PoolKey`]) that
//! states the pool's rule of membership. This module holds no pool and no
//! rule.

use std::{
    fmt,
    marker::PhantomData,
    sync::{Arc, OnceLock},
};

use super::Generation;
use crate::identity::basic::Worker;

/// A list of workers as the registry publishes one: shared, never changed.
type Workers = Arc<[Arc<dyn Worker>]>;

/// A host's name for one of its `N` pools: the pool's place among them and
/// the one rule of its membership.
pub trait PoolKey<const N: usize>: Copy {
    /// The pool's place among the host's pools: below `N`, and no two pools
    /// at one place. [`ModelPools`] keeps the pool's list and its cell
    /// there, and panics at a place of `N` or more.
    fn index(self) -> usize;

    /// Whether `worker` is a member of the pool. A list is projected by
    /// this rule once a generation, so it reads what stands for the worker
    /// object's life (its role, its transport, its registration) and
    /// nothing a running worker changes: health, circuit-breaker and
    /// overload state stay on the worker, for the reader of the list.
    fn matches(self, worker: &dyn Worker) -> bool;
}

/// One membership generation's members, their lists by pool, and one cell
/// beside each list for what the host keeps of that list.
///
/// A list is projected at its pool's first read and then shared by every
/// reader until a membership change publishes another generation
/// ([`Self::pool`]). A cell is the host's (`C`): the host fills it with what
/// it derives from the pool's list, and it falls with the generation
/// ([`Self::cell`]). A reader that needs several pools together reads each
/// from the one value it took: two reads of the registry can straddle a
/// membership change and pair workers that never stood together.
///
/// It is a [`Generation`] by itself, for a host that keeps nothing more of
/// one, and a field of the host's own value where the host does. A host
/// that uses it bare as its registry's value puts it on cache lines of its
/// own, apart from the count of its handles, which every request writes.
#[derive(Debug)]
pub struct ModelPools<K, C, const N: usize> {
    members: Workers,
    lists: [OnceLock<Workers>; N],
    cells: [OnceLock<C>; N],
    /// The one key the value is read by.
    key: PhantomData<fn(K)>,
}

impl<K: PoolKey<N>, C, const N: usize> ModelPools<K, C, N> {
    /// The workers of `key`'s pool, in the members' order: projected once a
    /// generation and borrowed from it, so a length or membership read costs
    /// no refcount step. A pool that takes every member is the members' own
    /// list, and nothing is made for it. A caller that keeps the list past
    /// the value clones the `Arc` itself.
    #[inline]
    pub fn pool(&self, key: K) -> &Arc<[Arc<dyn Worker>]> {
        self.lists[key.index()].get_or_init(|| {
            if self
                .members
                .iter()
                .all(|worker| key.matches(worker.as_ref()))
            {
                return Arc::clone(&self.members);
            }
            self.members
                .iter()
                .filter(|worker| key.matches(worker.as_ref()))
                .cloned()
                .collect()
        })
    }

    /// The cell beside `key`'s list. It is empty until the host fills it:
    /// nothing here reads or writes it.
    #[inline]
    pub fn cell(&self, key: K) -> &OnceLock<C> {
        &self.cells[key.index()]
    }
}

impl<K, C, const N: usize> Generation for ModelPools<K, C, N>
where
    K: PoolKey<N> + fmt::Debug + 'static,
    C: fmt::Debug + Send + Sync + 'static,
{
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            members,
            lists: std::array::from_fn(|_| OnceLock::new()),
            cells: std::array::from_fn(|_| OnceLock::new()),
            key: PhantomData,
        }
    }

    #[inline]
    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.members
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, sync::Arc};

    use super::{ModelPools, PoolKey};
    use crate::{
        identity::basic::{ConnectionMode, Worker},
        model_card::ModelCard,
        registry::{Generation, WorkerRegistry},
        test_support::{no_health_check, BasicWorkerBuilder},
    };

    thread_local! {
        /// How many workers this thread's projections asked the rule of.
        static ASKED: Cell<usize> = const { Cell::new(0) };
    }

    /// A host's two pools, by its workers' transport.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Wire {
        Http,
        Grpc,
    }

    impl PoolKey<2> for Wire {
        fn index(self) -> usize {
            match self {
                Self::Http => 0,
                Self::Grpc => 1,
            }
        }

        fn matches(self, worker: &dyn Worker) -> bool {
            ASKED.set(ASKED.get() + 1);
            let mode = match self {
                Self::Http => ConnectionMode::Http,
                Self::Grpc => ConnectionMode::Grpc,
            };
            *worker.connection_mode() == mode
        }
    }

    /// The host's value of a generation: what it keeps of a list is the
    /// list's length.
    type Pools = ModelPools<Wire, usize, 2>;

    fn worker(url: &str, mode: ConnectionMode) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .model(ModelCard::new("m"))
                .connection_mode(mode)
                .health_config(no_health_check())
                .build(),
        )
    }

    fn generation_of(members: Vec<Arc<dyn Worker>>) -> Pools {
        Pools::of(Arc::from(members))
    }

    fn urls(list: &[Arc<dyn Worker>]) -> Vec<&str> {
        list.iter().map(|worker| worker.url()).collect()
    }

    /// A pool that takes every member hands the members' own list: no list
    /// is made. A pool that takes none is empty.
    #[test]
    fn a_pool_of_every_member_is_the_members_own_list() {
        let pools = generation_of(vec![
            worker("http://a:8000", ConnectionMode::Http),
            worker("http://b:8000", ConnectionMode::Http),
        ]);
        assert!(Arc::ptr_eq(pools.pool(Wire::Http), pools.members()));
        assert!(pools.pool(Wire::Grpc).is_empty());
        // A generation without a member makes no list either.
        let empty = generation_of(Vec::new());
        assert!(Arc::ptr_eq(empty.pool(Wire::Http), empty.members()));
        assert!(Arc::ptr_eq(empty.pool(Wire::Grpc), empty.members()));
    }

    /// A pool's list holds its members in the generation's order, by the
    /// members' own handles, and is projected at the first read alone: the
    /// rule is asked of no worker after it.
    #[test]
    fn a_pool_is_projected_once_in_the_members_order() {
        let pools = generation_of(vec![
            worker("http://a:8000", ConnectionMode::Http),
            worker("grpc://b:9000", ConnectionMode::Grpc),
            worker("http://c:8000", ConnectionMode::Http),
            worker("grpc://d:9000", ConnectionMode::Grpc),
        ]);
        let before = ASKED.get();
        let http = Arc::clone(pools.pool(Wire::Http));
        assert_eq!(urls(&http), ["http://a:8000", "http://c:8000"]);
        assert!(Arc::ptr_eq(&http[0], &pools.members()[0]));
        assert!(Arc::ptr_eq(&http[1], &pools.members()[2]));
        let asked = ASKED.get();
        assert!(asked > before);

        for _read in 0..3 {
            assert!(Arc::ptr_eq(pools.pool(Wire::Http), &http));
        }
        assert_eq!(ASKED.get(), asked);
        assert_eq!(
            urls(pools.pool(Wire::Grpc)),
            ["grpc://b:9000", "grpc://d:9000"]
        );
    }

    /// A cell is empty until the host fills it, and is read back as filled
    /// beside its own pool's list. Reading a cell asks the rule of no worker
    /// and projects no list.
    #[test]
    fn a_cell_is_the_hosts_to_fill_beside_its_own_list() {
        let pools = generation_of(vec![
            worker("http://a:8000", ConnectionMode::Http),
            worker("grpc://b:9000", ConnectionMode::Grpc),
            worker("grpc://c:9000", ConnectionMode::Grpc),
        ]);
        let before = ASKED.get();
        assert!(pools.cell(Wire::Http).get().is_none());
        assert!(pools.cell(Wire::Grpc).get().is_none());
        assert_eq!(ASKED.get(), before);

        let kept = pools
            .cell(Wire::Grpc)
            .get_or_init(|| pools.pool(Wire::Grpc).len());
        assert_eq!(*kept, 2);
        assert_eq!(pools.cell(Wire::Grpc).get(), Some(&2));
        assert!(pools.cell(Wire::Http).get().is_none());
    }

    /// The registry builds the value at every publish, as it builds any
    /// host's: a membership change hands the next reader empty lists and
    /// empty cells, and a reader that holds the old value keeps its list
    /// and its cell.
    #[test]
    fn a_membership_change_publishes_empty_lists_and_cells() {
        let registry = WorkerRegistry::<Pools>::new();
        registry
            .register(worker("http://a:8000", ConnectionMode::Http))
            .expect("a registers");
        let held = registry.generation("m");
        let list = Arc::clone(held.pool(Wire::Http));
        assert_eq!(*held.cell(Wire::Http).get_or_init(|| list.len()), 1);

        registry
            .register(worker("http://b:8000", ConnectionMode::Http))
            .expect("b registers");
        let next = registry.generation("m");
        assert!(next.cell(Wire::Http).get().is_none());
        assert_eq!(
            urls(next.pool(Wire::Http)),
            ["http://a:8000", "http://b:8000"]
        );
        assert!(Arc::ptr_eq(held.pool(Wire::Http), &list));
        assert_eq!(held.cell(Wire::Http).get(), Some(&1));
        // A model the registry does not hold reads the empty generation.
        assert!(registry.generation("nobody").pool(Wire::Http).is_empty());
    }
}
