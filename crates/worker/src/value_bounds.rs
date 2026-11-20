//! The bounds a load report keeps an engine's named values under, and the
//! read of a `values` key off the wire. Every path a list arrives by is
//! held to the one test here: a pushed record's entries where the record
//! is read, a polled answer's list where the load monitor takes it.

use std::sync::Arc;

use serde::{de::IgnoredAny, Deserialize, Deserializer};

use crate::worker::NamedValue;

/// How many entries of a list a report considers. Those past them are
/// dropped unread.
pub(crate) const MAX_VALUES: usize = 64;
const MAX_NAME_BYTES: usize = 63;
const MAX_UNIT_BYTES: usize = 31;

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
/// their strings, and nothing is allocated.
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
pub(crate) fn stated<'de, D>(source: D) -> Result<Vec<NamedValue>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Entry {
        Named(NamedValue),
        Other(IgnoredAny),
    }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stated {
        List(Vec<Entry>),
        Other(IgnoredAny),
    }
    match Stated::deserialize(source)? {
        Stated::List(entries) => Ok(entries
            .into_iter()
            .map(|entry| match entry {
                Entry::Named(value) => value,
                Entry::Other(IgnoredAny) => NamedValue {
                    name: Arc::default(),
                    unit: Arc::default(),
                    value: 0.0,
                    age_ms: None,
                },
            })
            .collect()),
        Stated::Other(IgnoredAny) => Ok(Vec::new()),
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
    }
}
