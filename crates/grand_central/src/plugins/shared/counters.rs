//! The counters a plugin adds to through `host.counter`: one per id the spec
//! declared for it, shared by every instance of the plugin, read by the
//! host's metrics. An undeclared id adds to nothing and is counted as such.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

#[derive(Clone, Debug)]
pub struct Counters {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    names: Vec<String>,
    values: Vec<AtomicU64>,
    undeclared: AtomicU64,
}

impl Counters {
    /// Counters named by `names`, in id order, all zero.
    pub fn new(names: Vec<String>) -> Self {
        let values = names.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            inner: Arc::new(Inner {
                names,
                values,
                undeclared: AtomicU64::new(0),
            }),
        }
    }

    /// Adds `by` to counter `id`; an undeclared id is counted apart.
    pub fn add(&self, id: u32, by: u64) {
        match usize::try_from(id)
            .ok()
            .and_then(|i| self.inner.values.get(i))
        {
            Some(counter) => {
                counter.update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.saturating_add(by)
                });
            }
            None => {
                self.inner
                    .undeclared
                    .update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                        value.saturating_add(1)
                    });
            }
        }
    }

    /// The declared counters as (name, value), in id order.
    pub fn read(&self) -> impl Iterator<Item = (&str, u64)> + '_ {
        self.inner
            .names
            .iter()
            .zip(&self.inner.values)
            .map(|(n, v)| (n.as_str(), v.load(Ordering::Relaxed)))
    }

    /// Calls with an id outside the declared list.
    pub fn undeclared(&self) -> u64 {
        self.inner.undeclared.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/shared/counters/tests.rs"]
mod tests;
