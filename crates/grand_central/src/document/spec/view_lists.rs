//! A generation's label selectors, zones and engine families as lists
//! ([`ViewLists`]), and what one worker's registration reads as under
//! them ([`ViewListPositions`]).

use std::{cmp::Ordering, collections::BTreeMap};

use rustc_hash::FxHashMap;

use super::{selector, ViewSpec};

/// The lists a generation declares, as the writer of a worker's candidate
/// row ([`Candidate`](crate::engine::view::Candidate)) reads them: the row names its
/// engine family and its zone by a position from 1, and holds one bit for
/// each label selector the worker meets. Built once, at validation.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ViewLists {
    /// The label selectors in their keys' order, each with its bit: a
    /// worker's labels are in their keys' order too, so one pass over the
    /// two finds every selector the worker meets.
    labels: Vec<LabelSelector>,
    zones: FxHashMap<Box<str>, u16>,
    engines: FxHashMap<Box<str>, u16>,
}

/// One label selector: its key, the value it asks for when it names one,
/// and its bit, which is its place in the document's list.
#[derive(Debug, PartialEq, Eq)]
struct LabelSelector {
    key: Box<str>,
    value: Option<Box<str>>,
    bit: u8,
}

/// What one worker's registration reads as under a generation's lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewListPositions {
    /// The position of the worker's engine family, from 1; `0` when the
    /// list does not name it.
    pub engine_kind: u16,
    /// The position of the worker's zone, from 1; `0` for a worker with no
    /// zone and for a zone the list does not name.
    pub zone: u16,
    /// One bit for each label selector the worker meets.
    pub labels: u64,
}

impl ViewLists {
    pub(in crate::document) fn of(view: &ViewSpec) -> Self {
        let positions = |list: &[String]| {
            let named = list.iter().zip(1u16..);
            named.map(|(name, at)| (name.as_str().into(), at)).collect()
        };
        let sides = |(label, bit): (&String, u8)| {
            let (key, value) = selector(label);
            LabelSelector {
                key: key.into(),
                value: value.map(Into::into),
                bit,
            }
        };
        let mut labels: Vec<LabelSelector> = view.labels.iter().zip(0u8..).map(sides).collect();
        // The sort is stable: selectors of one key keep the list's order.
        labels.sort_by(|a, b| a.key.cmp(&b.key));
        Self {
            labels,
            zones: positions(&view.zones),
            engines: positions(&view.engines),
        }
    }

    /// Whether the generation declares no list: every row reads 0 in the
    /// three fields, and a store looks nothing up.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty() && self.zones.is_empty() && self.engines.is_empty()
    }

    /// Whether these lists can leave a worker outside them: they name an
    /// engine family or a zone. Label selectors alone leave none outside.
    #[must_use]
    pub(crate) fn names_a_family_or_zone(&self) -> bool {
        !self.zones.is_empty() || !self.engines.is_empty()
    }

    /// A worker's three fields under these lists, from the engine family,
    /// the zone and the labels of its registration.
    #[must_use]
    pub fn positions(
        &self,
        engine: &str,
        zone: Option<&str>,
        labels: &BTreeMap<String, String>,
    ) -> ViewListPositions {
        ViewListPositions {
            engine_kind: self.engine(engine).0.unwrap_or(0),
            zone: self.zone(zone).0.unwrap_or(0),
            labels: self.met(labels),
        }
    }

    /// Whether a worker's engine family and its zone stand outside these
    /// lists, the family first: the worker names one, and its list holds
    /// entries and not that one. No label is read.
    #[must_use]
    pub(crate) fn outside(&self, engine: &str, zone: Option<&str>) -> (bool, bool) {
        (self.engine(engine).1, self.zone(zone).1)
    }

    /// The position of an engine family, and whether it stands outside the
    /// list: the worker names one, and the list holds entries and not that
    /// one.
    fn engine(&self, engine: &str) -> (Option<u16>, bool) {
        let at = self.engines.get(engine).copied();
        let outside = at.is_none() && !engine.is_empty() && !self.engines.is_empty();
        (at, outside)
    }

    /// The position of a zone and whether it stands outside the list, as
    /// [`Self::engine`] reads a family. An empty zone is no zone.
    fn zone(&self, zone: Option<&str>) -> (Option<u16>, bool) {
        let zone = zone.filter(|zone| !zone.is_empty());
        let at = zone.and_then(|zone| self.zones.get(zone).copied());
        let outside = at.is_none() && zone.is_some() && !self.zones.is_empty();
        (at, outside)
    }

    /// The bits of the selectors a worker's labels meet, in one pass over
    /// the two, each in its keys' order.
    fn met(&self, labels: &BTreeMap<String, String>) -> u64 {
        let mut bits = 0;
        let mut wanted = self.labels.iter().peekable();
        'held: for (key, held) in labels {
            while let Some(want) = wanted.peek() {
                match (*want.key).cmp(key.as_str()) {
                    Ordering::Less => {}
                    Ordering::Greater => continue 'held,
                    Ordering::Equal => {
                        let value = want.value.as_deref();
                        let met = value.is_none_or(|value| value == held.as_str());
                        bits |= u64::from(met) << want.bit;
                    }
                }
                wanted.next();
            }
            break;
        }
        bits
    }
}
