//! The mechanism the worker's metric writers share with a host's, behind the
//! `metrics` feature: the bounded label set, the one process-wide set of model
//! labels, and the handles a record keeps ([`handles`]). `smg-external-router`
//! re-exports each item at the path it had there.
//!
//! The `runtime` feature adds the worker's own writers (`writers`) and the
//! store that keeps their series (`series`).

#[cfg(feature = "runtime")]
pub mod series;
#[cfg(feature = "runtime")]
pub mod writers;

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, LazyLock,
};

use dashmap::{mapref::entry::Entry, DashMap};

use crate::metrics_labels::MODEL_OTHER;

const BOUNDED_LABEL_SENTINEL: &str = MODEL_OTHER;

/// Max distinct client-supplied model labels retained before collapsing to the
/// sentinel. A gateway fronts far fewer real models than this; the cap only bites
/// on adversarial or unvalidated input.
const MAX_MODEL_LABELS: usize = 1024;

static MODEL_LABELS: LazyLock<BoundedLabels> =
    LazyLock::new(|| BoundedLabels::new(MAX_MODEL_LABELS));
static BOUNDED_LABEL_SENTINEL_ARC: LazyLock<Arc<str>> =
    LazyLock::new(|| Arc::from(BOUNDED_LABEL_SENTINEL));

/// A label set that admits at most `cap` distinct values; everything past
/// the cap maps to the sentinel `other`.
///
/// Distinct values beyond `cap` collapse to the shared sentinel, so untrusted
/// input (client-supplied model names, model-generated tool names, tenant
/// keys) cannot grow the interner, or the metric's Prometheus series set,
/// without bound. Unlike an LRU, admitted values are never evicted and
/// re-admitted, which would keep minting new series in the recorder even as
/// the map churned. Admission reserves a slot before touching the map, so a
/// burst of distinct labels cannot overshoot the cap. The model and tool
/// labels are two process-wide sets; a label another part of the gateway
/// bounds by a setting of its own (the scheduler's tenant label) gets a set
/// of its own.
pub struct BoundedLabels {
    map: DashMap<String, Arc<str>>,
    admitted: AtomicUsize,
    cap: usize,
}

impl BoundedLabels {
    /// A set that admits `cap` distinct values.
    pub fn new(cap: usize) -> Self {
        Self {
            map: DashMap::new(),
            admitted: AtomicUsize::new(0),
            cap,
        }
    }

    /// `s` as the label value: its own interned copy while the set has room,
    /// the sentinel `other` once it has none.
    pub fn intern(&self, s: &str) -> Arc<str> {
        if let Some(entry) = self.map.get(s) {
            return Arc::clone(entry.value());
        }
        let mut admitted = self.admitted.load(Ordering::Relaxed);
        loop {
            if self.is_full(admitted) {
                return Arc::clone(&BOUNDED_LABEL_SENTINEL_ARC);
            }
            match self.admitted.compare_exchange_weak(
                admitted,
                admitted + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(now) => admitted = now,
            }
        }
        match self.map.entry(s.to_string()) {
            Entry::Occupied(entry) => {
                // Someone else admitted this label first; give the slot back.
                self.admitted.fetch_sub(1, Ordering::AcqRel);
                Arc::clone(entry.get())
            }
            Entry::Vacant(entry) => Arc::clone(&*entry.insert(Arc::from(s))),
        }
    }

    /// Whether the set admits no value more: `admitted` is at its cap, or
    /// a test reads the set as full on its own thread. Asked only for a
    /// value the set does not hold.
    fn is_full(&self, admitted: usize) -> bool {
        #[cfg(test)]
        if labels_full::on_this_thread() {
            return true;
        }
        admitted >= self.cap
    }
}

/// Intern a client-supplied model label, bounded by `MAX_MODEL_LABELS`.
pub fn intern_model_label(model_id: &str) -> Arc<str> {
    MODEL_LABELS.intern(model_id)
}

/// For this crate's tests: a thread's say that the bounded label sets are
/// full. A value a set does not hold then gets the sentinel, as it does
/// past the set's cap, and no process-wide set is filled under the other
/// tests.
#[cfg(test)]
pub(crate) mod labels_full {
    use std::cell::Cell;

    thread_local! {
        static FULL: Cell<bool> = const { Cell::new(false) };
    }

    /// The say, held: it ends when this drops, so a case that fails leaves
    /// none behind on its thread.
    #[must_use = "the say ends when the guard drops"]
    pub(crate) struct Full(());

    impl Drop for Full {
        fn drop(&mut self) {
            FULL.set(false);
        }
    }

    /// Have this thread read the sets as full until the guard drops.
    pub(crate) fn hold_on_this_thread() -> Full {
        FULL.set(true);
        Full(())
    }

    pub(super) fn on_this_thread() -> bool {
        FULL.get()
    }
}

/// Per-thread caches of the handles the per-request records go through.
///
/// The `metrics` macros build a key on every call (the labels in a `Vec`,
/// the hash) and search the recorder's registry for the series under one of
/// its shard locks, after the dynamic labels were interned through shared
/// maps. A request records a handful of metrics that way from whichever
/// runtime worker is polling it, against the same few label sets, so at a
/// thousand requests per second the interners' and the registry's locks are
/// taken thousands of times per second from every worker. A handle the
/// recorder hands out refers to the series' shared value, so each worker
/// keeps the handles it has used in a map of its own and a record is a local
/// lookup plus the atomic update of the value. The label sets are bounded
/// (static values, the route templates, the capped model labels), and so is
/// every map. Nothing is kept before the recorder is installed: a handle
/// from the no-op recorder would be a dead end for the series.
pub mod handles {
    use std::{
        cell::Cell,
        collections::HashMap,
        hash::BuildHasher,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
    };

    use rustc_hash::FxHashMap;

    use super::{intern_model_label, BOUNDED_LABEL_SENTINEL_ARC, MAX_MODEL_LABELS};

    static RECORDER_INSTALLED: AtomicBool = AtomicBool::new(false);

    /// The most handles one map keeps: the model labels the interner admits
    /// and its sentinel. The other dynamic labels (route templates, status
    /// codes, worker addresses) are far fewer. Past the cap a record builds
    /// its handle and uses it without keeping it, so a flood of distinct
    /// labels, which the interner collapses to its sentinel, cannot grow a
    /// map.
    pub const MAX_KEPT_PER_MAP: usize = MAX_MODEL_LABELS + 1;

    /// The process's Prometheus recorder is installed: records keep their
    /// handles from here on.
    pub fn recorder_installed() {
        RECORDER_INSTALLED.store(true, Ordering::Release);
    }

    /// Whether a record on this thread keeps the handle it builds.
    pub fn keep_handles() -> bool {
        FORCED
            .get()
            .unwrap_or_else(|| RECORDER_INSTALLED.load(Ordering::Acquire))
    }

    /// For tests: have the records of this thread keep their handles (or
    /// build keys), whatever the installed flag says; `None` gives the flag
    /// its say back. A handle kept under a test's local recorder belongs to
    /// that recorder, so a test that forces keeping stays on its own thread
    /// (the harness gives every test one) and its handles die with it.
    pub fn force_on_this_thread(keep: Option<bool>) {
        FORCED.set(keep);
    }

    /// Handles by a static label set, then by the model label as the
    /// interner gave it: the key is the interner's own allocation, shared
    /// with every thread that kept the handle. The static level hashes
    /// with FxHash: its keys are the gateway's own label constants, hashed
    /// on every record; the model level keeps SipHash, its key is the
    /// client's.
    pub type ByLabels<L, H> = FxHashMap<L, HashMap<Arc<str>, H>>;

    thread_local! {
        /// A test's say for this thread, over the installed flag.
        static FORCED: Cell<Option<bool>> = const { Cell::new(None) };
    }

    /// Uses the handle under `key`, building it on first sight and keeping
    /// it while the map is under [`MAX_KEPT_PER_MAP`]. The key is only
    /// allocated for a new entry.
    pub fn with_handle<H, S: BuildHasher>(
        map: &mut HashMap<String, H, S>,
        key: &str,
        build: impl FnOnce() -> H,
        use_handle: impl FnOnce(&H),
    ) {
        if let Some(handle) = map.get(key) {
            use_handle(handle);
            return;
        }
        let handle = build();
        use_handle(&handle);
        if map.len() < MAX_KEPT_PER_MAP {
            map.insert(key.to_owned(), handle);
        }
    }

    /// Whether `label` is the sentinel the interner gives a name past its
    /// cap (or refused in a race at the cap).
    pub fn is_sentinel(label: &Arc<str>) -> bool {
        Arc::ptr_eq(label, &BOUNDED_LABEL_SENTINEL_ARC)
    }

    /// Whether a handle built under `label` is kept in a map of `len`
    /// entries: under the cap, and never for the sentinel. A name the
    /// interner refused is used and dropped, so a flood of distinct names
    /// grows no map, and a name refused in a race does not pin the
    /// sentinel's series under its own spelling for the thread's life.
    pub fn kept(len: usize, label: &Arc<str>) -> bool {
        len < MAX_KEPT_PER_MAP && !is_sentinel(label)
    }

    fn find_model_handle<'a, H, S: BuildHasher>(
        map: &'a HashMap<Arc<str>, H, S>,
        model_id: &str,
    ) -> Option<&'a H> {
        // One model is the usual deployment, and comparing its name costs a
        // fraction of hashing it with SipHash: the hash runs only once this
        // map keeps a second label, whatever request brought it.
        if map.len() == 1 {
            map.iter()
                .next()
                .filter(|&(label, _)| **label == *model_id)
                .map(|(_, handle)| handle)
        } else {
            map.get(model_id)
        }
    }

    /// Uses the handle under the model label of `model_id`, building it on
    /// first sight with the label the interner gives and keeping it, when
    /// [`kept`] says so, under that same `Arc<str>`: the map holds the
    /// interner's allocation, not a copy of a client-controlled name.
    pub fn with_model_handle<H>(
        map: &mut HashMap<Arc<str>, H>,
        model_id: &str,
        build: impl FnOnce(Arc<str>) -> H,
        use_handle: impl FnOnce(&H),
    ) {
        if let Some(handle) = find_model_handle(map, model_id) {
            use_handle(handle);
            return;
        }
        let label = intern_model_label(model_id);
        let keep = kept(map.len(), &label);
        let handle = build(Arc::clone(&label));
        use_handle(&handle);
        if keep {
            map.insert(label, handle);
        }
    }

    /// The inner map under `key`, created on first sight; the key is only
    /// allocated for a new entry.
    #[expect(
        clippy::expect_used,
        reason = "the key is inserted just above when missing"
    )]
    pub fn nested<'m, V: Default, S: BuildHasher>(
        map: &'m mut HashMap<String, V, S>,
        key: &str,
    ) -> &'m mut V {
        if !map.contains_key(key) {
            map.insert(key.to_owned(), V::default());
        }
        map.get_mut(key).expect("inserted above")
    }

    #[cfg(test)]
    mod tests {
        use std::{cell::Cell, collections::hash_map::DefaultHasher, hash::BuildHasher, rc::Rc};

        use super::*;

        #[derive(Clone)]
        struct CountingBuildHasher(Rc<Cell<usize>>);

        impl BuildHasher for CountingBuildHasher {
            type Hasher = DefaultHasher;

            fn build_hasher(&self) -> Self::Hasher {
                self.0.set(self.0.get() + 1);
                DefaultHasher::new()
            }
        }

        #[test]
        fn a_handle_is_built_once_per_key_and_used_every_time() {
            let mut map: HashMap<String, usize> = HashMap::new();
            let builds = Cell::new(0);
            let mut used = Vec::new();
            for key in ["/a", "/a", "/b", "/a"] {
                with_handle(
                    &mut map,
                    key,
                    || {
                        builds.set(builds.get() + 1);
                        key.len() * 10
                    },
                    |handle| used.push(*handle),
                );
            }
            assert_eq!(builds.get(), 2);
            assert_eq!(map.len(), 2);
            assert_eq!(used, [20, 20, 20, 20]);
        }

        #[test]
        fn a_full_map_uses_a_new_handle_without_keeping_it() {
            let mut map: HashMap<String, usize> = HashMap::new();
            for i in 0..MAX_KEPT_PER_MAP {
                with_handle(&mut map, &format!("m{i}"), || i, |_| {});
            }
            assert_eq!(map.len(), MAX_KEPT_PER_MAP);
            let mut used = None;
            with_handle(
                &mut map,
                "one-too-many",
                || 7,
                |handle| used = Some(*handle),
            );
            assert_eq!(used, Some(7));
            assert_eq!(map.len(), MAX_KEPT_PER_MAP);
            assert!(!map.contains_key("one-too-many"));
            // A kept key is still found when the map is full.
            with_handle(&mut map, "m3", || 99, |handle| used = Some(*handle));
            assert_eq!(used, Some(3));
        }

        #[test]
        fn a_nested_map_is_created_once() {
            let mut map: HashMap<String, Vec<u8>> = HashMap::new();
            nested(&mut map, "/a").push(1);
            nested(&mut map, "/a").push(2);
            assert_eq!(map.len(), 1);
            assert_eq!(map["/a"], [1, 2]);
        }

        #[test]
        fn a_kept_model_handle_is_keyed_by_the_interners_label() {
            let mut map: HashMap<Arc<str>, usize> = HashMap::new();
            let mut built = None;
            with_model_handle(
                &mut map,
                "kept-model",
                |label| {
                    built = Some(Arc::clone(&label));
                    label.len()
                },
                |_| {},
            );
            let key = map.keys().next().expect("kept");
            assert!(Arc::ptr_eq(key, &intern_model_label("kept-model")));
            assert!(Arc::ptr_eq(key, built.as_ref().expect("built once")));
            let mut used = None;
            with_model_handle(&mut map, "kept-model", |_| 0, |handle| used = Some(*handle));
            assert_eq!(used, Some("kept-model".len()), "found by its text");
            assert_eq!(map.len(), 1);
        }

        #[test]
        fn one_model_lookup_skips_hashing_but_two_model_lookup_does_not() {
            let hashes = Rc::new(Cell::new(0));
            let mut map = HashMap::with_hasher(CountingBuildHasher(Rc::clone(&hashes)));
            map.insert(Arc::from("model-a"), 70);
            hashes.set(0);

            assert_eq!(find_model_handle(&map, "model-a"), Some(&70));
            assert_eq!(hashes.get(), 0);
            assert_eq!(find_model_handle(&map, "model-b"), None);
            assert_eq!(hashes.get(), 0);

            map.insert(Arc::from("model-b"), 71);
            hashes.set(0);
            assert_eq!(find_model_handle(&map, "model-a"), Some(&70));
            assert!(hashes.get() > 0);
        }

        #[test]
        fn a_map_of_one_model_finds_that_model_and_no_other() {
            let mut map: HashMap<Arc<str>, usize> = HashMap::new();
            let builds = Cell::new(0);
            let mut used = Vec::new();
            // The two names have one length: only their last byte tells
            // them apart.
            for model in ["model-a", "model-a", "model-b", "model-a", "model-b"] {
                with_model_handle(
                    &mut map,
                    model,
                    |label| {
                        builds.set(builds.get() + 1);
                        label.len() * 10 + usize::from(label.ends_with('b'))
                    },
                    |handle| used.push(*handle),
                );
            }
            assert_eq!(builds.get(), 2);
            assert_eq!(map.len(), 2);
            assert_eq!(used, [70, 70, 71, 70, 71]);
        }

        #[test]
        fn the_sentinel_is_used_but_never_kept() {
            let sentinel = Arc::clone(&BOUNDED_LABEL_SENTINEL_ARC);
            assert!(is_sentinel(&sentinel));
            assert!(!kept(0, &sentinel));
            let admitted = intern_model_label("an-admitted-model");
            assert!(!is_sentinel(&admitted));
            assert!(kept(0, &admitted));
            assert!(!kept(MAX_KEPT_PER_MAP, &admitted));
        }

        #[test]
        fn the_forced_say_of_a_thread_outranks_the_installed_flag() {
            assert!(!keep_handles());
            force_on_this_thread(Some(true));
            assert!(keep_handles());
            force_on_this_thread(Some(false));
            assert!(!keep_handles());
            force_on_this_thread(None);
            assert!(!keep_handles());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_labels_cap_cardinality_with_a_sentinel() {
        let labels = BoundedLabels::new(2);

        let a = labels.intern("m1");
        let b = labels.intern("m2");
        assert_eq!(labels.map.len(), 2);

        // Repeats return the same interned Arc without growing the map.
        let a2 = labels.intern("m1");
        assert!(Arc::ptr_eq(&a, &a2));
        assert_eq!(labels.map.len(), 2);

        // A distinct value past the cap collapses to the sentinel and does not
        // grow the map, so no new Prometheus series is minted for it.
        let c = labels.intern("m3");
        assert_eq!(&*c, BOUNDED_LABEL_SENTINEL);
        assert_eq!(labels.map.len(), 2);

        // Already-admitted values still resolve normally after the cap is hit.
        let b2 = labels.intern("m2");
        assert!(Arc::ptr_eq(&b, &b2));
        assert_ne!(&*a, BOUNDED_LABEL_SENTINEL);
    }

    #[test]
    fn a_thread_reads_the_label_sets_as_full_until_the_say_drops() {
        let admitted = intern_model_label("a-model-admitted-ahead");
        {
            let _full = labels_full::hold_on_this_thread();
            let again = intern_model_label("a-model-admitted-ahead");
            assert!(Arc::ptr_eq(&again, &admitted));
            let met = intern_model_label("a-model-met-while-full");
            assert_eq!(&*met, BOUNDED_LABEL_SENTINEL);
        }
        let met = intern_model_label("a-model-met-while-full");
        assert_eq!(&*met, "a-model-met-while-full");
    }
}
