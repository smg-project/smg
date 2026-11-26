//! Membership rules for every door that writes the store's rows. The
//! controller's door uses them behind the wire. A host can use them over
//! a store it holds in its own process.

#![expect(
    clippy::exhaustive_enums,
    clippy::exhaustive_structs,
    reason = "the store's table of worker records leaves the crate, and its types with it"
)]

use std::{collections::hash_map::RandomState, fmt, hash::BuildHasher, sync::Arc, time::Instant};

use super::workers::{
    identity, MembershipChange, Registrant, WorkerFacts, WorkerPatch, WorkerRow, WorkerStatus,
};

/// A write of a row by a caller that did not register it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotRegistrant<'a> {
    /// A control call made the row.
    Control,
    /// The caller's own provider made the row, under another key.
    AnotherKey { provider: &'a str },
    /// Another provider made the row, or the caller is a control call.
    Provider { provider: &'a str },
}

impl fmt::Display for NotRegistrant<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Control => {
                f.write_str("the row was made by a control call: its maker alone changes it")
            }
            Self::AnotherKey { provider } => write!(
                f,
                "the row is provider {provider}'s under another key: that key's caller changes it"
            ),
            Self::Provider { provider } => write!(
                f,
                "the row is provider {provider}'s: its worker is changed at the provider's source"
            ),
        }
    }
}

impl WorkerRow {
    /// How a registration of `facts` changes this row, by whichever door it
    /// comes. A drain is of the registration it named and ends with it: a
    /// registration over a draining row is the URL's next life, whatever
    /// its facts, and a removal that named the draining row then names a
    /// row that left. A patch and a second drain keep a drain. Any other
    /// row is changed as its facts are ([`WorkerFacts::compare_to_standing`]).
    pub fn compare_new_facts(&self, facts: &WorkerFacts) -> WorkerFactsChange {
        if self.status == WorkerStatus::Draining {
            WorkerFactsChange::Another
        } else {
            facts.compare_to_standing(&self.facts)
        }
    }

    /// Whether `caller` made the row. Check before writing a standing row.
    pub fn made_by(&self, caller: &Registrant) -> Result<(), NotRegistrant<'_>> {
        if self.maker == *caller {
            return Ok(());
        }
        Err(match (&self.maker, caller) {
            (Registrant::Control, _) => NotRegistrant::Control,
            (Registrant::Provider(found), Registrant::Provider(own))
                if found.provider() == own.provider() =>
            {
                NotRegistrant::AnotherKey {
                    provider: found.provider(),
                }
            }
            (Registrant::Provider(found), _) => NotRegistrant::Provider {
                provider: found.provider(),
            },
        })
    }
}

/// What a registration's facts make of the row that stands under their key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerFactsChange {
    /// The facts are equal.
    Unmoved,
    /// The facts change within the same life.
    InPlace,
    /// The facts start another life.
    Another,
}

impl WorkerFacts {
    /// How these facts change a standing registration under the same key.
    ///
    /// The facts carry the standing row's incarnation, as a registration
    /// laid over the row does: the one write that asks this copies it first.
    /// The facts are compared whole, so facts that state another number are
    /// never `Unmoved`.
    ///
    /// Another life starts where the facts change which requests the worker
    /// can take at all: its role, its wires, its models, whether it takes
    /// images, or what it states beside that for a request's hard demands.
    /// The store pays a publication at the write of another life, so a
    /// decision that starts once the write is answered reads none of the old
    /// life's facts. A decision that loaded its publication before the write
    /// still decides over the old life.
    ///
    /// A life its vision or its demand facts alone start is handed the
    /// standing life's status: the store hands a status on where role,
    /// models and wires match, and it is the same reachable worker.
    pub fn compare_to_standing(&self, standing: &Self) -> WorkerFactsChange {
        if self == standing {
            WorkerFactsChange::Unmoved
        } else if self.role != standing.role
            || self.wire_modes != standing.wire_modes
            || self.vision != standing.vision
            || self.demand_facts != standing.demand_facts
            || !self.same_models(standing)
        {
            WorkerFactsChange::Another
        } else {
            WorkerFactsChange::InPlace
        }
    }
}

/// Whether `new` registers the worker `old` was: its role, its models and
/// its wires. A registration that differs in one is another worker.
pub(super) fn same_worker(old: &WorkerFacts, new: &WorkerFacts) -> bool {
    old.role == new.role && old.same_models(new) && old.wire_modes == new.wire_modes
}

impl WorkerRow {
    /// What a registration of `facts` by `maker` makes of `standing`, the
    /// row that stands under their key, at `now`: the change, and the
    /// row the change leaves. One rule for every writer of a row, whatever
    /// the writer reads the standing row from.
    ///
    /// The facts carry the incarnation their door gave them. Under the
    /// standing row's own they change the row in place: it keeps its id,
    /// its status, its maker and its admission instant, and its revision
    /// steps. The change is a resize where the width moved. Under another
    /// incarnation they are the key's next life: a new id, at revision 1,
    /// admitted at `now`. The key's registration stands through its new
    /// incarnation, and so does its maker, and the standing row hands its
    /// status on by its own rule ([`WorkerRow::hands_on`]). A key no row
    /// stands under is registered: pending, at revision 1, by `maker`.
    ///
    /// A blank value of a descriptor or of an encode offer is no report:
    /// the row holds what the two rules read. A descriptor that stands so
    /// is kept where it is, shared or not: a row written again copies none.
    pub(crate) fn laid_over(
        mut facts: WorkerFacts,
        standing: Option<&Self>,
        maker: Registrant,
        now: Instant,
    ) -> (MembershipChange, Self) {
        if let Some(descriptor) = facts
            .pairing
            .as_mut()
            .filter(|descriptor| !descriptor.is_normalized())
        {
            Arc::make_mut(descriptor).normalize();
        }
        facts.encode = facts.encode.normalized();
        let Some(standing) = standing else {
            let id = identity(&facts.key, facts.incarnation);
            let row = Self {
                id,
                facts,
                status: WorkerStatus::Pending,
                revision: 1,
                admitted_at: now,
                maker,
            };
            return (MembershipChange::Registered { id }, row);
        };
        if standing.facts.incarnation == facts.incarnation {
            let id = standing.id;
            let change = if standing.facts.dp_size == facts.dp_size {
                MembershipChange::Relabelled { id }
            } else {
                MembershipChange::Resized { id }
            };
            let row = Self {
                id,
                facts,
                status: standing.status,
                revision: standing.revision + 1,
                admitted_at: standing.admitted_at,
                maker: standing.maker.clone(),
            };
            return (change, row);
        }
        let new = identity(&facts.key, facts.incarnation);
        let row = Self {
            id: new,
            status: standing.hands_on(&facts),
            facts,
            revision: 1,
            admitted_at: now,
            maker: standing.maker.clone(),
        };
        let old = standing.id;
        (MembershipChange::Replaced { old, new }, row)
    }

    /// The status the key's next life starts in behind this row, where
    /// `facts` are that life's. The row's own carries over where the
    /// registration matches ([`same_worker`]), as the registry inherits a
    /// replaced worker's runtime state. A drain is of the registration it
    /// named and ends with it: a draining incarnation hands no status on,
    /// and neither does another worker's. The next life is pending then.
    /// Whether a worker takes images and what it states for a request's
    /// hard demands are not asked, though a change of either starts another
    /// life: it is the same reachable worker.
    pub(super) fn hands_on(&self, facts: &WorkerFacts) -> WorkerStatus {
        if self.status != WorkerStatus::Draining && same_worker(&self.facts, facts) {
            self.status
        } else {
            WorkerStatus::Pending
        }
    }
}

impl WorkerPatch {
    /// Writes the patch over `facts`: each fact it names, and no other.
    /// Every field is named: one more stops the build here until it says
    /// which fact it moves. A patch moves no part of a worker's offer.
    pub(crate) fn laid_on(self, facts: &mut WorkerFacts) {
        let Self {
            labels,
            priority,
            overload,
        } = self;
        if let Some(labels) = labels {
            facts.labels = labels;
        }
        if let Some(priority) = priority {
            facts.priority = priority;
        }
        if let Some(overload) = overload {
            facts.overload = overload;
        }
    }
}

/// A fact past its bound, named with the rule it broke.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactPastBound {
    pub fact: &'static str,
    pub why: &'static str,
}

impl WorkerFacts {
    /// What a row's numbers are charged in [`Self::weight`].
    pub const NUMBERS: usize = 160;
    /// What a row is charged beside its numbers where its worker states a
    /// fact for a request's hard demands: the wire's block at its widest.
    pub const DEMAND_FACTS: usize = 22;
    /// What a text is charged beside its own bytes.
    pub const FRAME: usize = 5;
    /// The most a row weighs.
    pub const ROW_BYTES: usize = 16 * 1024;

    /// An upper bound on the row's wire size. A row within [`Self::ROW_BYTES`]
    /// fits a controller's frame, including when a host writes it in process.
    pub fn weight(&self, maker: &Registrant) -> usize {
        let text = |value: &str| value.len() + Self::FRAME;
        let mut bytes =
            Self::NUMBERS + 2 * text(&self.key) + text(&self.engine) + text(&self.engine_kind);
        if self.signals.scrape_interval_ms > 0 {
            bytes += text(&self.engine_kind);
        }
        if self.demand_facts != crate::engine::DemandFacts::default() {
            bytes += Self::DEMAND_FACTS;
        }
        if let Some(zone) = &self.zone {
            bytes += text(zone);
        }
        for model in &self.models {
            bytes += text(model);
        }
        for (name, value) in &self.labels {
            bytes += text(name) + text(value);
            if name == "policy" {
                bytes += text(value);
            }
        }
        if let Registrant::Provider(found) = maker {
            bytes += text(found.provider()) + text(found.key());
        }
        bytes
    }

    /// Check each fact's own bound, then the size of the row with its maker.
    pub fn bounded(&self, maker: &Registrant) -> Result<(), FactPastBound> {
        let past = |fact, why| Err(FactPastBound { fact, why });
        if self.key.is_empty() {
            return past("key", "a registration names its worker's URL");
        }
        if self.key.len() > Self::URL_BYTES {
            return past("key", "a URL is at most 2048 bytes");
        }
        if blank(&self.key) {
            return past("key", "a URL holds no blank and no control character");
        }
        if self.engine.len() > Self::URL_BYTES {
            return past("engine", "a URL is at most 2048 bytes");
        }
        if blank(&self.engine) {
            return past("engine", "a URL holds no blank and no control character");
        }
        if self.engine_kind.len() > Self::NAME_BYTES {
            return past("engine_kind", "a runtime is at most 256 bytes");
        }
        if blank(&self.engine_kind) {
            return past(
                "engine_kind",
                "a runtime holds no blank and no control character",
            );
        }
        if self
            .models
            .iter()
            .any(|model| model.len() > Self::NAME_BYTES)
        {
            return past("models", "a model id is at most 256 bytes");
        }
        if self
            .models
            .iter()
            .any(|model| model.is_empty() || blank(model))
        {
            return past(
                "models",
                "a model id is not empty and holds no blank and no control character",
            );
        }
        if self
            .zone
            .as_ref()
            .is_some_and(|zone| zone.len() > Self::NAME_BYTES)
        {
            return past("zone", "a zone is at most 256 bytes");
        }
        if self.labels.len() > Self::LABELS {
            return past("labels", "a row holds at most 128 labels");
        }
        for (name, value) in &self.labels {
            if name.len() > Self::LABEL_KEY_BYTES {
                return past("labels", "a label's key is at most 128 bytes");
            }
            if value.len() > Self::LABEL_VALUE_BYTES {
                return past("labels", "a label's value is at most 4096 bytes");
            }
        }
        if self.weight(maker) > Self::ROW_BYTES {
            return past("row", "a row is at most 16384 bytes on the wire");
        }
        Ok(())
    }
}

/// Whether a stated name holds a blank or a control character.
pub(crate) fn blank(name: &str) -> bool {
    name.chars().any(|c| c.is_whitespace() || c.is_control())
}

/// The low bits of a life's number hold the store's epoch, counted from the
/// door's base; the leader's term stands above them.
pub const EPOCH_BITS: u32 = 40;

/// The bound of a base and of an epoch, each: their sum and one stay inside
/// the low bits.
pub const HALF: u64 = 1 << (EPOCH_BITS - 1);

/// The number of the life a registration makes under `term` in a store that
/// stands at `epoch`, at a door whose count starts at `base`.
///
/// Every life the store held was numbered at an earlier epoch of this term
/// or under an earlier term, and a registration moves the epoch, so no two
/// lives share a number, the later of two lives of one run under one term
/// has the higher one, and the id of a URL's next life is one no life of it
/// had. Another run under the same term counts from another base. A term
/// past 2^24, or a base or an epoch past 2^39, would carry into the other's
/// bits, and a number that wraps repeats: a debug build asserts the three
/// bounds (a term a second for half a year, a thousand writes a second for
/// seventeen years).
pub fn life(term: u64, base: u64, epoch: u64) -> u64 {
    debug_assert!(
        term >> (u64::BITS - EPOCH_BITS) == 0,
        "a term past its 24 bits: {term}"
    );
    debug_assert!(base < HALF, "a base past its 39 bits: {base}");
    debug_assert!(epoch < HALF, "an epoch past its 39 bits: {epoch}");
    (term << EPOCH_BITS) | (base.wrapping_add(epoch).wrapping_add(1) & ((1 << EPOCH_BITS) - 1))
}

/// A draw of where one run's count of lives starts, below [`HALF`]: 39 bits
/// of a hash under the keys the standard library draws for a process's maps,
/// so the crate gains no generator for one number. Each call draws anew: a
/// run calls it once and hands the value to every door it builds.
pub fn base() -> u64 {
    RandomState::new().hash_one(0u8) % HALF
}

#[cfg(test)]
#[path = "../../tests/unit/store/membership/tests.rs"]
mod tests;
