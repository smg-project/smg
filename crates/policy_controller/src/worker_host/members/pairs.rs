//! The pair classes a generation keeps for its prefill and decode lists, and
//! a pair's admission as a controller reads it.

use std::sync::OnceLock;

use super::Members;
use crate::{
    document::spec::Pairing,
    engine::{PairAdmission, PartnerClasses},
};

/// A generation's sets of pair classes, each at its pairing mode's place:
/// a set is derived at the first ask under its mode and stands for as long
/// as the generation does.
#[derive(Debug, Default)]
pub(super) struct PairSets([OnceLock<PartnerClasses>; 3]);

/// A pairing mode's place among a generation's sets.
const fn place(mode: Pairing) -> usize {
    match mode {
        Pairing::Off => 0,
        Pairing::Lenient => 1,
        Pairing::Strict => 2,
    }
}

impl Members {
    /// The pair classes of the generation's prefill and decode lists under
    /// the pairing mode `mode`. `derive` derives them where no set stands
    /// under the mode. Asks that meet there wait for the one that derives,
    /// and read the same set.
    pub(in crate::worker_host) fn pair_classes(
        &self,
        mode: Pairing,
        derive: impl FnOnce() -> PartnerClasses,
    ) -> &PartnerClasses {
        self.pairs.0[place(mode)].get_or_init(derive)
    }
}

/// A pair's admission as a controller reads it: a prefill always has
/// capacity and a decode always has a room.
///
/// A controller knows no admission window of a gateway. So it reads both
/// bits open, as the rows its store builds state them, and the gateway that
/// asks gates capacity at its own commit.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAdmission;

impl PairAdmission for OpenAdmission {
    fn has_capacity(&self, _: usize) -> bool {
        true
    }

    fn has_room(&self, _: usize) -> bool {
        true
    }
}
