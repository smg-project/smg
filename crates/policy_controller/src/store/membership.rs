//! Membership rules for every door that writes the store's rows. The
//! controller's door uses them behind the wire. A host can use them over
//! a store it holds in its own process.

use std::{collections::hash_map::RandomState, fmt, hash::BuildHasher};

use super::workers::{Maker, WorkerFacts, WorkerRow};

/// A write of a row by a caller that did not make it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotMaker<'a> {
    /// A control call made the row.
    Control,
    /// The caller's own provider made the row, under another key.
    AnotherKey { provider: &'a str },
    /// Another provider made the row, or the caller is a control call.
    Provider { provider: &'a str },
}

impl fmt::Display for NotMaker<'_> {
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
    /// Whether `caller` made the row. Check before writing a standing row.
    pub fn made_by(&self, caller: &Maker) -> Result<(), NotMaker<'_>> {
        if self.maker == *caller {
            return Ok(());
        }
        Err(match (&self.maker, caller) {
            (Maker::Control, _) => NotMaker::Control,
            (Maker::Provider(found), Maker::Provider(own))
                if found.provider() == own.provider() =>
            {
                NotMaker::AnotherKey {
                    provider: found.provider(),
                }
            }
            (Maker::Provider(found), _) => NotMaker::Provider {
                provider: found.provider(),
            },
        })
    }
}

/// What a registration's facts make of the row that stands under their key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Laid {
    /// The facts are equal.
    Unmoved,
    /// The facts change within the same life.
    InPlace,
    /// The facts start another life.
    Another,
}

impl WorkerFacts {
    /// How these facts change a standing registration under the same key.
    pub fn laid_over(&self, standing: &Self) -> Laid {
        if self == standing {
            Laid::Unmoved
        } else if self.role != standing.role
            || self.wire_modes != standing.wire_modes
            || !self.same_models(standing)
        {
            Laid::Another
        } else {
            Laid::InPlace
        }
    }
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

/// Where one run's count of lives starts: below `HALF`, drawn once a run.
pub fn base() -> u64 {
    RandomState::new().hash_one(0u8) % HALF
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{base, Laid, NotMaker, HALF};
    use crate::{
        store::{Maker, Unmade, WorkerFacts, WorkerRow, WorkerStatus},
        view::{Role, WireModes, WorkerId},
    };

    #[test]
    fn the_rows_maker_passes_and_any_other_caller_is_told_who_made_it() -> Result<(), Unmade> {
        let callers = [
            Maker::Control,
            Maker::provider("pool", "first")?,
            Maker::provider("pool", "second")?,
            Maker::provider("other", "first")?,
        ];
        let control = Err((
            NotMaker::Control,
            "the row was made by a control call: its maker alone changes it",
        ));
        let another_key = Err((
            NotMaker::AnotherKey { provider: "pool" },
            "the row is provider pool's under another key: that key's caller changes it",
        ));
        let provider = Err((
            NotMaker::Provider { provider: "pool" },
            "the row is provider pool's: its worker is changed at the provider's source",
        ));
        for (maker, expected) in [
            (Maker::Control, [Ok(()), control, control, control]),
            (
                Maker::provider("pool", "first")?,
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
    fn equal_facts_move_nothing_a_label_changes_in_place_and_a_role_a_wire_or_a_model_is_another_life(
    ) {
        let standing = WorkerFacts {
            role: Role::Single,
            wire_modes: WireModes::HTTP,
            models: vec!["one".to_owned(), "two".to_owned()],
            ..WorkerFacts::default()
        };
        assert_eq!(standing.clone().laid_over(&standing), Laid::Unmoved);

        let mut relabelled = standing.clone();
        relabelled
            .labels
            .insert("pool".to_owned(), "blue".to_owned());
        assert_eq!(relabelled.laid_over(&standing), Laid::InPlace);

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
            assert_eq!(changed.laid_over(&standing), Laid::Another, "{field}");
        }

        let mut reordered = standing.clone();
        reordered.models.reverse();
        assert_eq!(reordered.laid_over(&standing), Laid::InPlace);
    }

    #[test]
    fn a_run_draws_its_base_below_the_bound() {
        assert!(base() < HALF);
        assert!(base() < HALF);
    }
}
