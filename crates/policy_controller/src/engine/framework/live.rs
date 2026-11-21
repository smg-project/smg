//! Which profiles one pick of their picker decides.

use super::{Engine, Marks};

impl Engine {
    /// Whether one pick of the profile at `index` over a pool's workers is
    /// the profile's whole decision, made by its picker alone
    /// ([`Picker::pick_live`]): the picker says so, the profile holds no
    /// scorer, its filters are the engine's own two tests of a worker's
    /// health, with or without the test of the workers a retry tried and
    /// with no other, and its stages ask nothing else of a decision.
    ///
    /// A caller that holds every other rule of a placement itself (a
    /// gateway's own placement pass) asks such a profile for its pick and
    /// reads no row for it. That caller tests a worker as the worker tests
    /// itself for routing, which is the two health tests, and steps past
    /// the workers a retry tried by its own rule. Any filter beyond those
    /// is one the pick would drop (a pair's admission, a pair's room), so a
    /// profile that holds one is not picked so. `false` past the profiles.
    ///
    /// [`Picker::pick_live`]: super::Picker::pick_live
    #[must_use]
    pub fn picks_live(&self, index: u16) -> bool {
        let at = usize::from(index);
        let (Some(profile), Some(Some(tests)), Some(marks)) = (
            self.profiles.get(at),
            self.row_tests.get(at),
            self.marks.get(at),
        ) else {
            return false;
        };
        profile.picker.picks_live()
            && profile.scorers.is_empty()
            && tests.health_alone()
            && marks.0 & !Marks::DRAWS_FIRST == 0
    }
}
