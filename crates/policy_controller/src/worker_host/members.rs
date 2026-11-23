//! What a holder keeps of a registry's generation: the members, and where
//! the build holds the engine over worker lists, the members by role, each
//! role's list with its cell, its facts and its reach, the pair classes of
//! its prefill and decode lists, and the columns an encode request's
//! decision reads beside its encoder and prefill lists.

use std::sync::Arc;
#[cfg(feature = "in-process")]
use std::{array, mem, sync::OnceLock};

use smg_worker::{identity::basic::Worker, registry::Generation};

#[cfg(feature = "in-process")]
use super::reach::{ListReach, Reach};
#[cfg(feature = "in-process")]
use crate::engine::{FixedFacts, LentList, ListCell, Role};

/// What a host keeps of a registry's generation: the members.
///
/// Where the build holds the engine over worker lists (the `in-process`
/// feature), the value also keeps the members by role, for the holder to
/// lend (`Objects::lent`): each role's list in the registry's order, with a
/// cell of its own and with what the holder's rows say of each worker. The
/// reach of each list is cut with it (`Objects::reach`). They
/// stand for as long as the generation does. A membership change publishes
/// another generation, whose cells are empty.
///
/// The registry makes the value under its own guard, where no member is
/// read and no lock taken. So the lists are cut behind that, at the
/// generation's first ask.
#[derive(Debug)]
pub struct Members {
    members: Arc<[Arc<dyn Worker>]>,
    /// The pair classes of the prefill and decode lists, one set a pairing
    /// mode.
    #[cfg(feature = "in-process")]
    pairs: pairs::PairSets,
    /// The columns an encode request's decision reads beside the encoder
    /// and prefill lists: derived at the first ask, once.
    #[cfg(feature = "in-process")]
    encode: OnceLock<EncodeColumns>,
    #[cfg(feature = "in-process")]
    by_role: ByRole,
}

impl Generation for Members {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            members,
            #[cfg(feature = "in-process")]
            pairs: pairs::PairSets::default(),
            #[cfg(feature = "in-process")]
            encode: OnceLock::new(),
            #[cfg(feature = "in-process")]
            by_role: ByRole::default(),
        }
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.members
    }
}

/// What a holder states of one member where the generation's lists are
/// cut: what its row says of the worker, and who can dial it.
#[cfg(feature = "in-process")]
#[derive(Clone, Copy, Debug)]
pub(super) struct Stated {
    /// What is lent with the worker's list.
    pub(super) facts: FixedFacts,
    /// The wires the worker states and the gateways that dial it.
    pub(super) reach: Reach,
}

/// One role's workers of a generation that the holder answered for and, at
/// the same places, what it stated of each.
#[cfg(feature = "in-process")]
type RoleList = (Arc<[Arc<dyn Worker>]>, Box<[FixedFacts]>);

/// What one pass over a generation's members cuts: each role's list at the
/// role's place, and the reach of the same workers at the same places.
#[cfg(feature = "in-process")]
#[derive(Debug)]
struct Cut {
    lists: [RoleList; 4],
    reach: [ListReach; 4],
}

/// A generation's members by role.
#[cfg(feature = "in-process")]
#[derive(Debug, Default)]
struct ByRole {
    /// Each role's list and its reach, at the role's place: cut at the
    /// generation's first ask, all four by one pass over the members.
    cut: OnceLock<Cut>,
    /// The cell beside each role's list, at the role's place.
    cells: [ListCell; 4],
}

/// A role's place among a generation's lists and cells.
#[cfg(feature = "in-process")]
const fn place(role: Role) -> usize {
    match role {
        Role::Single => 0,
        Role::Prefill => 1,
        Role::Decode => 2,
        Role::Encode => 3,
    }
}

#[cfg(feature = "in-process")]
impl Members {
    /// The members of `role`, lent with the cell beside them and with what
    /// was stated of each. At the generation's first ask `state` says what
    /// is known of the members, one answer a member in their order, and the
    /// lists are cut with it: a member it answers `None` for is in no list.
    /// Asks that meet there wait for the one that cuts. No later ask calls
    /// `state`.
    pub(super) fn lent(
        &self,
        role: Role,
        state: impl FnOnce(&[Arc<dyn Worker>]) -> Box<[Option<Stated>]>,
    ) -> LentList<'_> {
        let cut = || self.cut(&state(&self.members));
        let (workers, facts) = &self.by_role.cut.get_or_init(cut).lists[place(role)];
        LentList {
            workers,
            cell: &self.by_role.cells[place(role)],
            facts: Some(facts),
        }
    }

    /// The reach of the list [`Self::lent`] lends for `role`: one worker's
    /// reach at each place of that list, and as many as the list. The two
    /// are cut by one pass, at the generation's first ask of either, and
    /// `state` is asked there as `lent` asks it.
    pub(super) fn reach(
        &self,
        role: Role,
        state: impl FnOnce(&[Arc<dyn Worker>]) -> Box<[Option<Stated>]>,
    ) -> &ListReach {
        let cut = || self.cut(&state(&self.members));
        &self.by_role.cut.get_or_init(cut).reach[place(role)]
    }

    /// Each role's members that `stated` answers for, in the members'
    /// order, with what it says of each, and the reach of each role's
    /// list. Answers of another count than the members are refused, in
    /// every build: a pass cut short would leave workers out unseen.
    fn cut(&self, stated: &[Option<Stated>]) -> Cut {
        assert_eq!(stated.len(), self.members.len(), "one answer a member");
        let mut workers: [Vec<Arc<dyn Worker>>; 4] = Default::default();
        let mut facts: [Vec<FixedFacts>; 4] = Default::default();
        let mut reach: [Vec<Reach>; 4] = Default::default();
        for (member, stated) in self.members.iter().zip(stated) {
            let Some(stated) = stated else {
                continue;
            };
            let at = place(Role::from(*member.worker_type()));
            workers[at].push(Arc::clone(member));
            facts[at].push(stated.facts);
            reach[at].push(stated.reach);
        }
        Cut {
            lists: array::from_fn(|at| {
                let (workers, facts) = (mem::take(&mut workers[at]), mem::take(&mut facts[at]));
                (workers.into(), facts.into())
            }),
            reach: reach.map(ListReach::new),
        }
    }
}

#[cfg(feature = "in-process")]
mod encode;
#[cfg(feature = "in-process")]
mod pairs;
#[cfg(feature = "in-process")]
pub use encode::EncodeColumns;
#[cfg(feature = "in-process")]
pub use pairs::OpenAdmission;

#[cfg(test)]
mod tests;
