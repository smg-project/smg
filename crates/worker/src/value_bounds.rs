//! The bounds a load report keeps an engine's ranks and named values
//! under, and the read of a `values` key off the wire. Every path a list
//! arrives by is held to the one test here: a pushed record's entries where
//! the record is read, a polled answer's list where the load monitor takes
//! it.

use std::{fmt, sync::Arc};

use serde::{
    de::{DeserializeSeed, Error, MapAccess, SeqAccess, Visitor},
    Deserializer,
};

use crate::worker::NamedValue;
#[cfg(feature = "runtime")]
use crate::worker::SchedulerLoadSnapshot;

/// How many ranks of a worker a report considers: the ranks below it, and
/// the one rank of its engine a worker is registered for. A record or a
/// row of any other rank is dropped unread.
#[cfg(feature = "runtime")]
pub(crate) const MAX_RANKS: usize = 64;
/// How many entries of a list a report considers. Those past them are
/// dropped unread.
pub(crate) const MAX_VALUES: usize = 64;
const MAX_NAME_BYTES: usize = 63;
const MAX_UNIT_BYTES: usize = 31;

/// Whether a report of a worker considers `dp_rank`: a rank below
/// [`MAX_RANKS`], or the one rank of its engine the worker is registered
/// for (`pinned`), whatever its number.
#[cfg(feature = "runtime")]
pub(crate) fn considers(pinned: Option<usize>, dp_rank: i32) -> bool {
    usize::try_from(dp_rank).is_ok_and(|rank| rank < MAX_RANKS || pinned == Some(rank))
}

/// Puts the rows of a polled answer under the bound of its ranks, in
/// place, and answers how many rows left it for their rank: the rows of a
/// rank the report does not consider. Where the rows are one engine's
/// ranks (`one_engine`), a row of a rank an earlier row stated leaves as
/// well, uncounted: it stated no rank of its own, and the first of a rank
/// stands as the first of a name does. The kept rows keep their order. A
/// list that holds room for [`MAX_RANKS`] rows or fewer is bound where it
/// lies; one that holds more gives back the room past its rows.
#[cfg(feature = "runtime")]
pub(crate) fn bound_ranks(
    loads: &mut Vec<SchedulerLoadSnapshot>,
    pinned: Option<usize>,
    one_engine: bool,
) -> usize {
    let mut left = 0;
    // Whether a kept row stated a rank: each rank below the bound, and the
    // registered one past it.
    let mut below = [false; MAX_RANKS];
    let mut past = false;
    loads.retain(|row| {
        if !considers(pinned, row.dp_rank) {
            left += 1;
            return false;
        }
        let seen = match usize::try_from(row.dp_rank) {
            Ok(rank) if rank < MAX_RANKS => &mut below[rank],
            _ => &mut past,
        };
        !(std::mem::replace(seen, true) && one_engine)
    });
    loads.shrink_to(MAX_RANKS);
    left
}

/// Whether an entry among a list's first [`MAX_VALUES`] stands beside the
/// entries kept ahead of it: its name is not empty, its name and its unit
/// are within their bounds, its value is finite, and no kept entry states
/// its name. So the first of a name stands, and an entry that is dropped
/// keeps no name from a later one.
pub(crate) fn stands(name: &str, unit: &str, value: f64, kept: &[NamedValue]) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && unit.len() <= MAX_UNIT_BYTES
        && value.is_finite()
        && !kept.iter().any(|kept| *kept.name == *name)
}

/// Puts a list as an answer stated it under the bounds, in place, and
/// answers how many entries left it. The kept entries keep their order and
/// their strings. A list that holds room for [`MAX_VALUES`] entries or
/// fewer is bound where it lies, and nothing is allocated. One that holds
/// more gives the rest back, so a report holds the bound's room at the
/// most, as a pushed record's list does, whatever its engine stated.
pub(crate) fn bound(values: &mut Vec<NamedValue>) -> usize {
    let stated = values.len();
    values.truncate(MAX_VALUES);
    let mut kept = 0;
    for at in 0..values.len() {
        let entry = &values[at];
        if stands(&entry.name, &entry.unit, entry.value, &values[..kept]) {
            values.swap(kept, at);
            kept += 1;
        }
    }
    values.truncate(kept);
    values.shrink_to(MAX_VALUES);
    stated - kept
}

/// Reads a well-formed `values` key as any answer states it, and fails no
/// answer over it. A list keeps its entries in their order: a named value
/// as it stands, any other entry as one with no name, which the bounds
/// drop and the load monitor counts. A key of any other shape is some
/// other key of the answer's own, and reads as no list. A token no JSON
/// reader takes, a bare `NaN` among them, fails the whole body here as
/// anywhere in it: the load monitor keeps a route that had answered over
/// such a body.
///
/// The list is read where it lies on the wire: no entry is set aside to
/// be tried, so a named value costs its own text and its two strings.
pub(crate) fn stated<'de, D>(source: D) -> Result<Vec<NamedValue>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(match source.deserialize_any(Read::Values)? {
        Stated::List(values) => values,
        _ => Vec::new(),
    })
}

/// What an answer states at one place of a `values` key, taken whole.
enum Stated {
    Text(Arc<str>),
    Unsigned(u64),
    /// A whole number stated as present: an age alone takes it.
    Present(u64),
    Signed(i64),
    Float(f64),
    Null,
    List(Vec<NamedValue>),
    Named(NamedValue),
    Key(Key),
    Other,
}

/// The place read: the key's list, one entry of it, a key of an entry, a
/// name or a unit, or any other value. Each takes the shape it looks for
/// and reads any other to its end as [`Stated::Other`]. A value that is
/// left is still read as a value, so a token that fails a body fails it
/// here too, and only a name and a unit are made strings.
#[derive(Clone, Copy)]
enum Read {
    Values,
    Entry,
    Key,
    Text,
    Rest,
}

/// A field of an entry. `Other` is a key the entry states for itself:
/// its value is read and left.
#[derive(Clone, Copy)]
enum Key {
    Name,
    Unit,
    Value,
    AgeMs,
    Other,
}

impl Key {
    /// The field at a place: of an entry stated as a list, or of a key
    /// stated as a number.
    fn at(place: u64) -> Self {
        match place {
            0 => Self::Name,
            1 => Self::Unit,
            2 => Self::Value,
            3 => Self::AgeMs,
            _ => Self::Other,
        }
    }
}

/// An entry's fields as far as they are stated. A field stated twice, or
/// as something its type does not read, breaks the entry: it is still
/// read to its end, and stands as one with no name.
#[derive(Default)]
struct Fields {
    name: Option<Arc<str>>,
    unit: Option<Arc<str>>,
    value: Option<f64>,
    age_ms: Option<Option<u32>>,
    broken: bool,
}

impl Fields {
    /// The place that reads a field's value: a name and a unit are made
    /// strings where the entry still stands and states none yet.
    fn place(&self, key: Key) -> Read {
        match key {
            Key::Name if !self.broken && self.name.is_none() => Read::Text,
            Key::Unit if !self.broken && self.unit.is_none() => Read::Text,
            _ => Read::Rest,
        }
    }

    fn take(&mut self, key: Key, stated: Stated) {
        let fresh = match (key, stated) {
            (Key::Name, Stated::Text(name)) => self.name.replace(name).is_none(),
            (Key::Unit, Stated::Text(unit)) => self.unit.replace(unit).is_none(),
            (Key::Value, Stated::Unsigned(value)) => self.value.replace(value as f64).is_none(),
            (Key::Value, Stated::Signed(value)) => self.value.replace(value as f64).is_none(),
            (Key::Value, Stated::Float(value)) => self.value.replace(value).is_none(),
            (Key::AgeMs, Stated::Null) => self.age_ms.replace(None).is_none(),
            (Key::AgeMs, Stated::Unsigned(age) | Stated::Present(age)) => {
                u32::try_from(age).is_ok_and(|age| self.age_ms.replace(Some(age)).is_none())
            }
            (Key::Other, _) => true,
            _ => false,
        };
        self.broken |= !fresh;
    }

    /// The named value, where the entry stands and states its name, its
    /// unit and its value.
    fn named(self) -> Stated {
        match self {
            Self {
                name: Some(name),
                unit: Some(unit),
                value: Some(value),
                age_ms,
                broken: false,
            } => Stated::Named(NamedValue {
                name,
                unit,
                value,
                age_ms: age_ms.flatten(),
            }),
            _ => Stated::Other,
        }
    }
}

impl<'de> DeserializeSeed<'de> for Read {
    type Value = Stated;

    fn deserialize<D: Deserializer<'de>>(self, source: D) -> Result<Stated, D::Error> {
        source.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Read {
    type Value = Stated;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any value")
    }

    fn visit_bool<E>(self, _: bool) -> Result<Stated, E> {
        Ok(Stated::Other)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Stated, E> {
        Ok(match (self, u64::try_from(value)) {
            // A key reads by place as an unsigned number alone.
            (Self::Key, _) | (_, Err(_)) => Stated::Signed(value),
            (_, Ok(value)) => Stated::Unsigned(value),
        })
    }

    fn visit_u64<E>(self, value: u64) -> Result<Stated, E> {
        Ok(Stated::Unsigned(value))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Stated, E> {
        Ok(Stated::Float(value))
    }

    fn visit_str<E>(self, text: &str) -> Result<Stated, E> {
        Ok(match self {
            Self::Key => Stated::Key(match text {
                "name" => Key::Name,
                "unit" => Key::Unit,
                "value" => Key::Value,
                "age_ms" => Key::AgeMs,
                _ => Key::Other,
            }),
            Self::Text => Stated::Text(Arc::from(text)),
            Self::Values | Self::Entry | Self::Rest => Stated::Other,
        })
    }

    fn visit_unit<E>(self) -> Result<Stated, E> {
        Ok(Stated::Null)
    }

    // The rest is what a source other than a JSON text may state. Each is
    // read as the buffered read gave it to the derived readers, so none
    // fails an answer: a key by place is a number of 8 or of 64 bits, a
    // character is no text, bytes that are text are text, a value left
    // out is a null, one stated as present is an age alone, and a wrapped
    // one is read and left.
    fn visit_u16<E: Error>(self, value: u16) -> Result<Stated, E> {
        self.visit_u32(value.into())
    }

    fn visit_u32<E>(self, value: u32) -> Result<Stated, E> {
        Ok(match self {
            Self::Key => Stated::Other,
            _ => Stated::Unsigned(value.into()),
        })
    }

    fn visit_char<E>(self, _: char) -> Result<Stated, E> {
        Ok(Stated::Other)
    }

    fn visit_bytes<E: Error>(self, bytes: &[u8]) -> Result<Stated, E> {
        match (str::from_utf8(bytes), self) {
            (Ok(text), _) => self.visit_str(text),
            // Bytes that are no text are a key of the entry's own.
            (Err(_), Self::Key) => Ok(Stated::Key(Key::Other)),
            (Err(_), _) => Ok(Stated::Other),
        }
    }

    fn visit_none<E>(self) -> Result<Stated, E> {
        Ok(Stated::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, source: D) -> Result<Stated, D::Error> {
        Ok(match source.deserialize_any(Self::Rest)? {
            Stated::Unsigned(age) => Stated::Present(age),
            _ => Stated::Other,
        })
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, source: D) -> Result<Stated, D::Error> {
        source.deserialize_any(Self::Rest).map(|_| Stated::Other)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut list: A) -> Result<Stated, A::Error> {
        match self {
            Self::Values => {
                let room = list.size_hint().map_or(0, |stated| stated.min(MAX_VALUES));
                let mut values = Vec::with_capacity(room);
                while let Some(entry) = list.next_element_seed(Self::Entry)? {
                    values.push(match entry {
                        Stated::Named(value) => value,
                        _ => NamedValue {
                            name: Arc::default(),
                            unit: Arc::default(),
                            value: 0.0,
                            age_ms: None,
                        },
                    });
                }
                Ok(Stated::List(values))
            }
            // An entry stated by place: its name, unit, value and age,
            // and nothing past them.
            Self::Entry => {
                let mut fields = Fields::default();
                let mut place = 0;
                while let Some(stated) = list.next_element_seed(fields.place(Key::at(place)))? {
                    fields.take(Key::at(place), stated);
                    place += 1;
                }
                fields.broken |= place != 4;
                Ok(fields.named())
            }
            Self::Key | Self::Text | Self::Rest => {
                while list.next_element_seed(Self::Rest)?.is_some() {}
                Ok(Stated::Other)
            }
        }
    }

    fn visit_map<A: MapAccess<'de>>(self, mut entry: A) -> Result<Stated, A::Error> {
        let Self::Entry = self else {
            while entry.next_key_seed(Self::Rest)?.is_some() {
                entry.next_value_seed(Self::Rest)?;
            }
            return Ok(Stated::Other);
        };
        let mut fields = Fields::default();
        while let Some(key) = entry.next_key_seed(Self::Key)? {
            let key = match key {
                Stated::Key(key) => key,
                Stated::Unsigned(place) => Key::at(place),
                _ => {
                    fields.broken = true;
                    Key::Other
                }
            };
            fields.take(key, entry.next_value_seed(fields.place(key))?);
        }
        Ok(fields.named())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::SchedulerLoadSnapshot;

    fn value(name: &str, unit: &str, value: f64) -> NamedValue {
        NamedValue {
            name: name.into(),
            unit: unit.into(),
            value,
            age_ms: None,
        }
    }

    fn names(values: &[NamedValue]) -> Vec<&str> {
        values.iter().map(|value| value.name.as_ref()).collect()
    }

    #[test]
    fn a_list_keeps_its_first_entries_and_a_dropped_ones_place_goes_to_none() {
        let mut values: Vec<_> = (0..=64)
            .map(|index| value(&format!("ready_{index}"), "items", f64::from(index)))
            .collect();
        let mut whole = values.clone();
        assert_eq!(bound(&mut whole), 1);
        assert_eq!(whole.len(), 64);
        assert_eq!(whole.last().map(|last| last.value), Some(63.0));
        // An entry dropped among the first 64 gives no later one its place.
        values[0].value = f64::NAN;
        assert_eq!(bound(&mut values), 2);
        assert_eq!(values.len(), 63);
        assert_eq!(names(&values)[0], "ready_1");
    }

    #[test]
    fn a_list_stated_past_the_bounds_gives_the_room_of_what_left_back() {
        let entries = |count: u32| -> Vec<_> {
            (0..count)
                .map(|index| value(&format!("ready_{index}"), "items", f64::from(index)))
                .collect()
        };
        let mut past = entries(4_096);
        assert_eq!(bound(&mut past), 4_096 - MAX_VALUES);
        assert_eq!(past.len(), MAX_VALUES);
        assert_eq!(past.capacity(), MAX_VALUES);
        // One entry kept of a list stated as long holds the bound's room,
        // as a pushed record's list is sized.
        let mut one = entries(4_096);
        for entry in &mut one[1..] {
            entry.value = f64::NAN;
        }
        assert_eq!(bound(&mut one), 4_095);
        assert_eq!((one.len(), one.capacity()), (1, MAX_VALUES));
        // A list within the bound is bound where it lies.
        let mut within = entries(MAX_VALUES as u32);
        let lies = (within.as_ptr(), within.capacity());
        assert_eq!(bound(&mut within), 0);
        assert_eq!((within.as_ptr(), within.capacity()), lies);
    }

    #[test]
    fn a_list_drops_each_entry_past_a_bound_and_keeps_the_rest_as_stated() {
        let name = "é".repeat(31) + "e";
        let unit = "ø".repeat(15) + "o";
        let mut values = vec![
            value("", "items", 1.0),
            value(&"é".repeat(32), "items", 2.0),
            value("ready", &"ø".repeat(16), 3.0),
            value(&name, &unit, 4.0),
            value("depth", "items", f64::NEG_INFINITY),
            value("depth", "items", 5.0),
            value("depth", "items", 6.0),
            value("rate", "items/s", 7.0),
        ];
        let kept = [values[3].clone(), values[5].clone(), values[7].clone()];
        assert_eq!(bound(&mut values), 5);
        assert_eq!(values, kept);
        // The kept entries are the stated ones: no string is copied.
        assert!(values.iter().zip(&kept).all(|(left, stated)| {
            Arc::ptr_eq(&left.name, &stated.name) && Arc::ptr_eq(&left.unit, &stated.unit)
        }));
        // A list within the bounds stands as it is, and so does none.
        assert_eq!(bound(&mut values), 0);
        assert_eq!(values, kept);
        assert_eq!(bound(&mut Vec::new()), 0);
    }

    #[test]
    fn a_well_formed_values_key_fails_no_answer_whatever_it_states() {
        let read = |values: &str| -> SchedulerLoadSnapshot {
            let answer = format!(
                r#"{{"dp_rank":2,"num_running_reqs":5,"values":{values},"num_waiting_reqs":7}}"#
            );
            serde_json::from_str(&answer).expect("the answer is read")
        };
        // A key that is no list is the answer's own: no list, and the
        // fields on both sides of it are read.
        for other in [r#"{"depth":3}"#, r#""deep""#, "3", "null", "true"] {
            let snapshot = read(other);
            assert!(snapshot.values.is_empty(), "{other}");
            assert_eq!(
                (
                    snapshot.dp_rank,
                    snapshot.num_running_reqs,
                    snapshot.num_waiting_reqs
                ),
                (2, 5, 7),
                "{other}"
            );
        }
        // A list keeps each entry's place. One that is no named value is
        // read with no name: a number, a value of another type, a missing
        // unit, an age no sample has.
        let mut snapshot = read(
            r#"[{"name":"depth","unit":"items","value":3,"age_ms":41},
                7,
                {"name":"rate","unit":"items/s","value":"fast"},
                {"name":"lag","value":2.5},
                {"name":"wait","unit":"ms","value":1.5,"age_ms":-1},
                {"name":"rate","unit":"items/s","value":0.5,"more":[1]}]"#,
        );
        assert_eq!(names(&snapshot.values), ["depth", "", "", "", "", "rate"]);
        assert_eq!(snapshot.values[0].age_ms, Some(41));
        assert_eq!(snapshot.num_waiting_reqs, 7);
        // The bounds drop what was read with no name, and say how many.
        assert_eq!(snapshot.bound_values(), 4);
        assert_eq!(names(&snapshot.values), ["depth", "rate"]);
        assert_eq!(snapshot.values[1].value, 0.5);
        // A list of four states an entry by place: its name, unit, value
        // and age. A list of three or of five, a key stated twice and a
        // unit that is no text are read with no name, each to its end.
        let snapshot = read(
            r#"[["place","items",-4,null],
                ["short","items",4],
                ["long","items",4,null,5],
                {"name":"twice","unit":"items","value":1,"value":2},
                {"name":"deep","unit":{"of":["items"]},"value":1},
                {"name":"last","unit":"items","value":1}]"#,
        );
        assert_eq!(names(&snapshot.values), ["place", "", "", "", "", "last"]);
        assert_eq!(snapshot.values[0].value, -4.0);
        assert_eq!(snapshot.num_waiting_reqs, 7);
        // What the key keeps nothing of is read as a value all the same:
        // a number past any float, or half a pair of escapes, fails the
        // body under a key of the entry's own, in a unit that is a list,
        // and in a key that is no list.
        for refused in [
            r#"[{"name":"depth","unit":"items","value":3,"more":1e400}]"#,
            r#"[{"name":"depth","unit":"items","value":3,"more":"\udc00"}]"#,
            r#"[{"name":"depth","unit":[1e400],"value":3}]"#,
            r#"{"depth":1e400}"#,
        ] {
            let answer = format!(r#"{{"dp_rank":2,"values":{refused}}}"#);
            assert!(
                serde_json::from_str::<SchedulerLoadSnapshot>(&answer).is_err(),
                "{refused}"
            );
        }
    }
}
