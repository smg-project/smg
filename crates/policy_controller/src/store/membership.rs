//! Membership rules for every door that writes the store's rows. The
//! controller's door uses them behind the wire. A host can use them over
//! a store it holds in its own process.

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

impl WorkerFacts {
    /// What a registration of these facts by `maker` makes of `standing`,
    /// the row that stands under their key, at `now`: the change, and the
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
        mut self,
        standing: Option<&WorkerRow>,
        maker: Registrant,
        now: Instant,
    ) -> (MembershipChange, WorkerRow) {
        if let Some(descriptor) = self
            .pairing
            .as_mut()
            .filter(|descriptor| !descriptor.is_normalized())
        {
            Arc::make_mut(descriptor).normalize();
        }
        self.encode = self.encode.normalized();
        let Some(standing) = standing else {
            let id = identity(&self.key, self.incarnation);
            let row = WorkerRow {
                id,
                facts: self,
                status: WorkerStatus::Pending,
                revision: 1,
                admitted_at: now,
                maker,
            };
            return (MembershipChange::Registered { id }, row);
        };
        if standing.facts.incarnation == self.incarnation {
            let id = standing.id;
            let change = if standing.facts.dp_size == self.dp_size {
                MembershipChange::Relabelled { id }
            } else {
                MembershipChange::Resized { id }
            };
            let row = WorkerRow {
                id,
                facts: self,
                status: standing.status,
                revision: standing.revision + 1,
                admitted_at: standing.admitted_at,
                maker: standing.maker.clone(),
            };
            return (change, row);
        }
        let new = identity(&self.key, self.incarnation);
        let row = WorkerRow {
            id: new,
            status: standing.hands_on(&self),
            facts: self,
            revision: 1,
            admitted_at: now,
            maker: standing.maker.clone(),
        };
        let old = standing.id;
        (MembershipChange::Replaced { old, new }, row)
    }
}

impl WorkerRow {
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
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use super::{base, NotRegistrant, WorkerFactsChange, HALF};
    use crate::{
        engine::{
            demands::DemandFacts,
            view::{Role, WireModes, WorkerId},
        },
        store::{
            identity, pairing::PairingDescriptor, EncodeOffer, MembershipChange, ProviderKeyError,
            Registrant, WorkerFacts, WorkerRow, WorkerStatus,
        },
    };

    #[test]
    fn the_rows_maker_passes_and_any_other_caller_is_told_who_made_it(
    ) -> Result<(), ProviderKeyError> {
        let callers = [
            Registrant::Control,
            Registrant::provider("pool", "first")?,
            Registrant::provider("pool", "second")?,
            Registrant::provider("other", "first")?,
        ];
        let control = Err((
            NotRegistrant::Control,
            "the row was made by a control call: its maker alone changes it",
        ));
        let another_key = Err((
            NotRegistrant::AnotherKey { provider: "pool" },
            "the row is provider pool's under another key: that key's caller changes it",
        ));
        let provider = Err((
            NotRegistrant::Provider { provider: "pool" },
            "the row is provider pool's: its worker is changed at the provider's source",
        ));
        for (maker, expected) in [
            (Registrant::Control, [Ok(()), control, control, control]),
            (
                Registrant::provider("pool", "first")?,
                [provider, Ok(()), another_key, provider],
            ),
        ] {
            let row = WorkerRow {
                maker,
                id: WorkerId(1),
                facts: WorkerFacts::default(),
                status: WorkerStatus::Pending,
                revision: 1,
                admitted_at: Instant::now(),
            };
            for (caller, expected) in callers.iter().zip(expected) {
                assert_eq!(
                    row.made_by(caller).map_err(|not| (not, not.to_string())),
                    expected.map_err(|(not, sentence)| (not, sentence.to_owned())),
                    "maker {:?}, caller {caller:?}",
                    row.maker
                );
            }
        }
        Ok(())
    }

    #[test]
    fn facts_under_another_incarnation_are_never_unmoved() {
        let standing = WorkerFacts {
            key: "http://worker:8000".to_owned(),
            incarnation: 7,
            ..WorkerFacts::default()
        };
        let fresh = WorkerFacts {
            incarnation: 0,
            ..standing.clone()
        };
        assert_eq!(
            standing.compare_to_standing(&standing),
            WorkerFactsChange::Unmoved
        );
        assert_eq!(
            fresh.compare_to_standing(&standing),
            WorkerFactsChange::InPlace
        );
    }

    #[test]
    fn equal_facts_move_nothing_a_label_changes_in_place_and_a_role_a_wire_or_a_model_is_another_life(
    ) {
        let standing = WorkerFacts {
            role: Role::Single,
            wire_modes: WireModes::HTTP,
            models: vec!["one".to_owned(), "two".to_owned()],
            ..WorkerFacts::default()
        };
        assert_eq!(
            standing.clone().compare_to_standing(&standing),
            WorkerFactsChange::Unmoved
        );

        let mut relabelled = standing.clone();
        relabelled
            .labels
            .insert("pool".to_owned(), "blue".to_owned());
        assert_eq!(
            relabelled.compare_to_standing(&standing),
            WorkerFactsChange::InPlace
        );

        for (field, changed) in [
            (
                "role",
                WorkerFacts {
                    role: Role::Decode,
                    ..standing.clone()
                },
            ),
            (
                "wire modes",
                WorkerFacts {
                    wire_modes: WireModes::GRPC,
                    ..standing.clone()
                },
            ),
            (
                "models",
                WorkerFacts {
                    models: vec!["one".to_owned(), "three".to_owned()],
                    ..standing.clone()
                },
            ),
        ] {
            assert_eq!(
                changed.compare_to_standing(&standing),
                WorkerFactsChange::Another,
                "{field}"
            );
        }

        let mut reordered = standing.clone();
        reordered.models.reverse();
        assert_eq!(
            reordered.compare_to_standing(&standing),
            WorkerFactsChange::InPlace
        );
    }

    /// What a worker states for a request's hard demands says which
    /// requests it can take at all, as its wires do: a registration that
    /// changes it starts another life, whichever way the number moves.
    #[test]
    fn a_changed_demand_fact_is_another_life() {
        let stating = |runtime| WorkerFacts {
            demand_facts: DemandFacts {
                runtime,
                ..DemandFacts::default()
            },
            ..WorkerFacts::default()
        };
        for (standing, fresh) in [(7, 9), (7, 0), (0, 7)] {
            assert_eq!(
                stating(fresh).compare_to_standing(&stating(standing)),
                WorkerFactsChange::Another,
                "{standing} to {fresh}"
            );
        }
        assert_eq!(
            stating(7).compare_to_standing(&stating(7)),
            WorkerFactsChange::Unmoved
        );
    }

    /// Whether a worker takes images says which requests it can take at
    /// all, as a demand fact does: the door reads the bit for a request
    /// that needs vision. A registration that changes it starts another
    /// life, whichever way it turns.
    #[test]
    fn a_changed_vision_is_another_life() {
        let seeing = |vision| WorkerFacts {
            vision,
            ..WorkerFacts::default()
        };
        for (standing, fresh) in [(true, false), (false, true)] {
            assert_eq!(
                seeing(fresh).compare_to_standing(&seeing(standing)),
                WorkerFactsChange::Another,
                "{standing} to {fresh}"
            );
        }
        assert_eq!(
            seeing(true).compare_to_standing(&seeing(true)),
            WorkerFactsChange::Unmoved
        );
    }

    #[test]
    fn a_run_draws_its_base_below_the_bound() {
        assert!(base() < HALF);
        assert!(base() < HALF);
    }

    fn stated() -> WorkerFacts {
        WorkerFacts {
            key: "url".to_owned(),
            ..WorkerFacts::default()
        }
    }

    #[test]
    fn a_fact_past_its_bound_is_refused_by_its_name() {
        use super::FactPastBound;

        type Set = fn(&mut WorkerFacts, usize);
        let limits: [(&str, usize, &str, Set); 8] = [
            ("key", 2_048, "a URL is at most 2048 bytes", |f, n| {
                f.key = "a".repeat(n);
            }),
            ("engine", 2_048, "a URL is at most 2048 bytes", |f, n| {
                f.engine = "a".repeat(n);
            }),
            (
                "engine_kind",
                256,
                "a runtime is at most 256 bytes",
                |f, n| {
                    f.engine_kind = "a".repeat(n);
                },
            ),
            ("models", 256, "a model id is at most 256 bytes", |f, n| {
                f.models = vec!["a".repeat(n)];
            }),
            ("zone", 256, "a zone is at most 256 bytes", |f, n| {
                f.zone = Some(" ".repeat(n));
            }),
            ("labels", 128, "a row holds at most 128 labels", |f, n| {
                f.labels = (0..n).map(|key| (key.to_string(), String::new())).collect();
            }),
            (
                "labels",
                128,
                "a label's key is at most 128 bytes",
                |f, n| {
                    f.labels = [(" ".repeat(n), String::new())].into();
                },
            ),
            (
                "labels",
                4_096,
                "a label's value is at most 4096 bytes",
                |f, n| {
                    f.labels = [(String::new(), " ".repeat(n))].into();
                },
            ),
        ];
        let maker = Registrant::Control;
        for (fact, bound, why, set) in limits {
            let mut facts = stated();
            set(&mut facts, bound);
            assert_eq!(facts.bounded(&maker), Ok(()), "{fact} at {bound}");
            set(&mut facts, bound + 1);
            assert_eq!(facts.bounded(&maker), Err(FactPastBound { fact, why }));
        }

        let mut facts = stated();
        facts.key.clear();
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "key",
                why: "a registration names its worker's URL"
            })
        );
        facts.engine = "é".repeat(1_025);
        facts.engine_kind = "two words".to_owned();
        facts.models = vec![String::new()];
        facts.zone = Some("a".repeat(257));
        facts.labels.insert("a".to_owned(), "a".repeat(4_097));

        for bad in ["url\u{2003}other", "url\u{7f}"] {
            facts.key = bad.to_owned();
            assert_eq!(
                facts.bounded(&maker),
                Err(FactPastBound {
                    fact: "key",
                    why: "a URL holds no blank and no control character",
                })
            );
        }
        facts.key = stated().key;
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "engine",
                why: "a URL is at most 2048 bytes",
            })
        );
        for bad in ["one\u{2003}two", "one\u{7f}"] {
            facts.engine = bad.to_owned();
            assert_eq!(
                facts.bounded(&maker),
                Err(FactPastBound {
                    fact: "engine",
                    why: "a URL holds no blank and no control character",
                })
            );
            facts.engine.clear();
            facts.engine_kind = bad.to_owned();
            assert_eq!(
                facts.bounded(&maker),
                Err(FactPastBound {
                    fact: "engine_kind",
                    why: "a runtime holds no blank and no control character",
                })
            );
        }
        facts.engine_kind.clear();
        for bad in ["", "one\u{2003}two", "one\u{7f}"] {
            facts.models = vec!["m".to_owned(), bad.to_owned()];
            assert_eq!(
                facts.bounded(&maker),
                Err(FactPastBound {
                    fact: "models",
                    why: "a model id is not empty and holds no blank and no control character",
                })
            );
        }
        facts.models.clear();
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "zone",
                why: "a zone is at most 256 bytes",
            })
        );
        facts.zone = None;
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "labels",
                why: "a label's value is at most 4096 bytes",
            })
        );
    }

    #[test]
    fn a_row_weighs_its_texts_as_often_as_it_tells_them() -> Result<(), ProviderKeyError> {
        let mut facts = WorkerFacts {
            engine: "engine".to_owned(),
            engine_kind: "kind".to_owned(),
            zone: Some("zone-a".to_owned()),
            models: vec!["m".to_owned(), "other".to_owned()],
            labels: [
                ("tier".to_owned(), "gold".to_owned()),
                ("policy".to_owned(), "fast".to_owned()),
            ]
            .into(),
            ..stated()
        };
        assert_eq!(facts.weight(&Registrant::Control), 270);
        facts.signals.scrape_interval_ms = 1;
        assert_eq!(facts.weight(&Registrant::Control), 279);
        assert_eq!(facts.weight(&Registrant::provider("pool", "work")?), 297);
        Ok(())
    }

    #[test]
    fn a_long_list_of_model_ids_is_bounded_by_the_rows_size_alone() {
        use super::FactPastBound;

        let mut facts = stated();
        let maker = Registrant::Control;
        facts.models = (0..129).map(|id| id.to_string()).collect();
        assert_eq!(facts.bounded(&maker), Ok(()));

        // 186 bytes before the ids, then 62 ids of 261 bytes and one of 16.
        facts.models = vec!["m".repeat(256); 62];
        facts.models.push("m".repeat(11));
        assert_eq!(facts.weight(&maker), 16_384);
        assert_eq!(facts.bounded(&maker), Ok(()));
        facts.models[62].push('m');
        assert_eq!(
            facts.bounded(&maker),
            Err(FactPastBound {
                fact: "row",
                why: "a row is at most 16384 bytes on the wire",
            })
        );
        facts.models[62].pop();
        assert_eq!(
            facts.bounded(&Registrant::provider("pool", "work").expect("a maker")),
            Err(FactPastBound {
                fact: "row",
                why: "a row is at most 16384 bytes on the wire",
            })
        );
    }

    /// A row that stands under the key [`stated`] names, in `status`,
    /// admitted at `admitted_at` and made by a provider.
    fn standing(status: WorkerStatus, admitted_at: Instant) -> Result<WorkerRow, ProviderKeyError> {
        Ok(WorkerRow {
            id: WorkerId(9),
            facts: WorkerFacts {
                incarnation: 1,
                role: Role::Single,
                wire_modes: WireModes::HTTP,
                models: vec!["one".to_owned(), "two".to_owned()],
                dp_size: 1,
                ..stated()
            },
            status,
            revision: 4,
            admitted_at,
            maker: Registrant::provider("pool", "first")?,
        })
    }

    /// The rule's first arm: a key no row stands under.
    #[test]
    fn facts_laid_over_no_row_register_a_pending_row_by_their_maker() -> Result<(), ProviderKeyError>
    {
        let (maker, now) = (Registrant::provider("pool", "first")?, Instant::now());
        let facts = WorkerFacts {
            incarnation: 7,
            ..stated()
        };
        let id = identity("url", 7);
        let (change, row) = facts.clone().laid_over(None, maker.clone(), now);
        assert_eq!(change, MembershipChange::Registered { id });
        let registered = WorkerRow {
            id,
            facts,
            status: WorkerStatus::Pending,
            revision: 1,
            admitted_at: now,
            maker,
        };
        assert_eq!(row, registered);
        Ok(())
    }

    /// The rule's second arm: the standing row's own incarnation. The row
    /// keeps what is no fact of a registration: its id, its status, when it
    /// was admitted and who made it.
    #[test]
    fn facts_laid_over_their_own_incarnation_change_the_row_in_place(
    ) -> Result<(), ProviderKeyError> {
        let earlier = Instant::now();
        let now = earlier + Duration::from_secs(5);
        for status in [WorkerStatus::Ready, WorkerStatus::Draining] {
            let standing = standing(status, earlier)?;
            let id = standing.id;
            let mut facts = standing.facts.clone();
            facts.labels.insert("zone".to_owned(), "b".to_owned());
            let (change, row) = facts
                .clone()
                .laid_over(Some(&standing), Registrant::Control, now);
            assert_eq!(change, MembershipChange::Relabelled { id });
            let relabelled = WorkerRow {
                facts: facts.clone(),
                revision: 5,
                ..standing.clone()
            };
            assert_eq!(row, relabelled, "over a row that is {status:?}");
            // A width that moved is told as a resize.
            facts.dp_size = 2;
            let (change, row) = facts
                .clone()
                .laid_over(Some(&standing), Registrant::Control, now);
            assert_eq!(change, MembershipChange::Resized { id });
            let resized = WorkerRow {
                facts,
                revision: 5,
                ..standing
            };
            assert_eq!(row, resized, "over a row that is {status:?}");
        }
        Ok(())
    }

    /// The rule's third arm: another incarnation under the key. The same
    /// worker's status is handed on, a drain never, and the key's maker
    /// stands through its new life, whoever states it.
    #[test]
    fn facts_laid_over_another_incarnation_are_the_keys_next_life() -> Result<(), ProviderKeyError>
    {
        type State = fn(&mut WorkerFacts);
        // What each next life states, and whether it is the worker the row
        // was: its role, its models however they are ordered, its wires.
        let lives: [(State, bool); 6] = [
            (|_| {}, true),
            (|facts| facts.models.reverse(), true),
            (|facts| facts.vision = true, true),
            (|facts| facts.role = Role::Decode, false),
            (|facts| facts.wire_modes = WireModes::GRPC, false),
            (|facts| facts.models.push("three".to_owned()), false),
        ];
        let earlier = Instant::now();
        let now = earlier + Duration::from_secs(5);
        let new = identity("url", 2);
        for stood in [
            WorkerStatus::Ready,
            WorkerStatus::NotReady,
            WorkerStatus::Draining,
        ] {
            let standing = standing(stood, earlier)?;
            for (life, (state, same)) in lives.into_iter().enumerate() {
                let mut facts = WorkerFacts {
                    incarnation: 2,
                    ..standing.facts.clone()
                };
                state(&mut facts);
                let (change, row) =
                    facts
                        .clone()
                        .laid_over(Some(&standing), Registrant::Control, now);
                let old = standing.id;
                assert_eq!(change, MembershipChange::Replaced { old, new });
                let carried = same && stood != WorkerStatus::Draining;
                let next = WorkerRow {
                    id: new,
                    facts,
                    status: if carried {
                        stood
                    } else {
                        WorkerStatus::Pending
                    },
                    revision: 1,
                    admitted_at: now,
                    maker: standing.maker.clone(),
                };
                assert_eq!(row, next, "life {life} over a row that is {stood:?}");
            }
        }
        Ok(())
    }

    /// Ahead of every arm: a blank value of a descriptor or of an encode
    /// offer is no report, and a descriptor that holds none is shared with
    /// the row, never copied.
    #[test]
    fn facts_laid_over_any_row_hold_no_blank_of_a_descriptor_or_of_an_encode_offer(
    ) -> Result<(), ProviderKeyError> {
        let now = Instant::now();
        let standing = standing(WorkerStatus::Ready, now)?;
        let blank = PairingDescriptor {
            protocol: Some(" nixl ".to_owned()),
            transport: Some(" ".to_owned()),
            ..PairingDescriptor::default()
        };
        let read = Arc::new(PairingDescriptor {
            protocol: Some("nixl".to_owned()),
            ..PairingDescriptor::default()
        });
        // A new key, the standing row's own incarnation, and another.
        for (standing, incarnation) in [(None, 1), (Some(&standing), 1), (Some(&standing), 2)] {
            let facts = WorkerFacts {
                incarnation,
                pairing: Some(Arc::new(blank.clone())),
                encode: EncodeOffer {
                    connector: Some("  ".to_owned()),
                    ..EncodeOffer::default()
                },
                ..stated()
            };
            let (_, row) = facts.laid_over(standing, Registrant::Control, now);
            assert_eq!(row.facts.pairing.as_ref(), Some(&read));
            assert_eq!(row.facts.encode.connector, None);
            let facts = WorkerFacts {
                incarnation,
                pairing: Some(Arc::clone(&read)),
                ..stated()
            };
            let (_, row) = facts.laid_over(standing, Registrant::Control, now);
            let held = row.facts.pairing.expect("the descriptor stated");
            assert!(Arc::ptr_eq(&held, &read), "shared, not copied");
        }
        Ok(())
    }
}
