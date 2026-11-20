//! What a host's value of a generation can count on, proved with a value
//! that counts itself: every publish builds one, a generation has one value,
//! a value answers the members it was built from, a replaced value falls
//! once its write let the index go, the model-less generation carries a
//! value like any other, and a superseded one of those falls at the next
//! membership change.

use std::{
    cell::{Cell, RefCell},
    sync::Weak,
};

use super::*;
use crate::test_support::ready_worker;

/// The registry and the model a case watches.
type Watched = (Weak<WorkerRegistry<Counted>>, &'static str);

thread_local! {
    /// The values built on this thread, and the values standing.
    static BUILT: Cell<usize> = const { Cell::new(0) };
    static STANDING: Cell<usize> = const { Cell::new(0) };
    /// What a case watches, and the values that fell while the watched
    /// model's shard of the index was locked.
    static WATCHED: RefCell<Option<Watched>> = const { RefCell::new(None) };
    static FELL_LOCKED: Cell<usize> = const { Cell::new(0) };
}

/// A host's value that keeps the members and counts itself.
#[derive(Debug)]
struct Counted(Arc<[Arc<dyn Worker>]>);

impl Generation for Counted {
    fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
        BUILT.set(BUILT.get() + 1);
        STANDING.set(STANDING.get() + 1);
        Self(members)
    }

    fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.0
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        STANDING.set(STANDING.get() - 1);
        let watched = WATCHED.with_borrow(|watched| {
            let (registry, model) = watched.as_ref()?;
            Some((registry.upgrade()?, *model))
        });
        if let Some((registry, model)) = watched {
            if registry.model_index.try_get(model).is_locked() {
                FELL_LOCKED.set(FELL_LOCKED.get() + 1);
            }
        }
    }
}

fn a_changed_card() -> Vec<ModelCard> {
    vec![ModelCard::new("m").with_context_length(4_096)]
}

#[test]
fn every_publish_builds_one_value() {
    let start = BUILT.get();
    let built = || BUILT.get() - start;
    let registry = WorkerRegistry::<Counted>::new();
    assert_eq!(built(), 1, "the empty generation, with the registry");

    let first = ready_worker("http://a:8000", "m");
    let first_id = registry.register(Arc::clone(&first)).unwrap();
    assert_eq!(built(), 2, "a model's first member");
    let second_id = registry
        .register(ready_worker("http://b:8000", "m"))
        .unwrap();
    assert_eq!(built(), 3, "a member joins");

    registry.global_generation();
    assert_eq!(built(), 4, "the global generation, at its first read");
    registry.global_generation();
    assert_eq!(built(), 4, "and at no later read while membership stands");

    assert!(registry.refresh_models(&first, a_changed_card()));
    assert_eq!(built(), 5, "a refresh that changes a card");
    assert!(registry.replace(&first_id, ready_worker("http://a:8000", "m")));
    assert_eq!(built(), 6, "a property rebuild");

    registry.remove(&second_id).unwrap();
    assert_eq!(built(), 7, "a member leaves");
    registry.remove(&first_id).unwrap();
    assert_eq!(built(), 8, "the last member leaves");
}

#[test]
fn a_generation_has_one_value_built_from_its_members() {
    let registry = WorkerRegistry::<Counted>::new();
    registry
        .register(ready_worker("http://a:8000", "m"))
        .unwrap();

    let taken = registry.model_generation("m").unwrap();
    assert!(Arc::ptr_eq(&taken, &registry.generation("m")));
    assert!(
        Arc::ptr_eq(taken.members(), &registry.get_by_model("m")),
        "the members it answers are the index's own"
    );

    registry
        .register(ready_worker("http://b:8000", "m"))
        .unwrap();
    let next = registry.generation("m");
    assert!(!Arc::ptr_eq(&taken, &next), "a change ends the generation");
    assert_eq!(
        taken.members().len(),
        1,
        "the value taken keeps its members"
    );
    assert_eq!(next.members().len(), 2);
}

#[test]
fn a_replaced_value_falls_once_its_write_let_the_index_go() {
    let (standing, fell_locked) = (STANDING.get(), FELL_LOCKED.get());
    let registry = Arc::new(WorkerRegistry::<Counted>::new());
    WATCHED.set(Some((Arc::downgrade(&registry), "m")));

    let first = ready_worker("http://a:8000", "m");
    let first_id = registry.register(Arc::clone(&first)).unwrap();
    let second_id = registry
        .register(ready_worker("http://b:8000", "m"))
        .unwrap();
    assert_eq!(
        STANDING.get() - standing,
        2,
        "the empty one and the model's"
    );
    assert!(registry.refresh_models(&first, a_changed_card()));
    registry.remove(&second_id).unwrap();
    assert_eq!(STANDING.get() - standing, 2);
    registry.remove(&first_id).unwrap();
    assert_eq!(
        STANDING.get() - standing,
        1,
        "the model's went with its entry"
    );

    assert_eq!(
        FELL_LOCKED.get() - fell_locked,
        0,
        "none fell under the guard"
    );
    WATCHED.set(None);
}

#[test]
fn the_model_less_generation_carries_a_value_like_any_other() {
    let start = BUILT.get();
    let registry = WorkerRegistry::<Counted>::new();
    let unknown = registry.generation("no-such-model");
    assert!(unknown.members().is_empty());
    assert!(Arc::ptr_eq(&unknown, &registry.generation("another")));
    assert_eq!(
        BUILT.get() - start,
        1,
        "one empty generation, the registry's"
    );

    registry
        .register(ready_worker("http://a:8000", "m"))
        .unwrap();
    registry
        .register(ready_worker("http://b:8000", "n"))
        .unwrap();
    let global = registry.generation(UNKNOWN_MODEL_ID);
    assert_eq!(global.members().len(), 2, "every registered worker");
    assert!(Arc::ptr_eq(&global, &registry.global_generation()));

    registry
        .register(ready_worker("http://c:8000", "m"))
        .unwrap();
    let next = registry.global_generation();
    assert!(
        !Arc::ptr_eq(&global, &next),
        "a change ends it like any other"
    );
    assert_eq!(next.members().len(), 3);
}

#[test]
fn the_global_epoch_wrap_keeps_the_live_generation() {
    let registry = WorkerRegistry::<Counted>::new();
    registry
        .register(ready_worker("http://a:8000", "m"))
        .unwrap();
    registry
        .global_epoch
        .store(usize::MAX - 2, Ordering::Relaxed);

    registry
        .register(ready_worker("http://b:8000", "m"))
        .unwrap();
    let before_wrap = registry.global_generation();
    let mut before_urls: Vec<_> = before_wrap.members().iter().map(|w| w.url()).collect();
    before_urls.sort_unstable();
    assert_eq!(before_urls, ["http://a:8000", "http://b:8000"]);

    registry
        .register(ready_worker("http://c:8000", "m"))
        .unwrap();
    let after_wrap = registry.global_generation();
    let mut after_urls: Vec<_> = after_wrap.members().iter().map(|w| w.url()).collect();
    after_urls.sort_unstable();
    assert_eq!(
        after_urls,
        ["http://a:8000", "http://b:8000", "http://c:8000"]
    );
}

#[test]
fn a_superseded_model_less_value_falls_at_the_next_membership_change() {
    let standing = STANDING.get();
    let registry = WorkerRegistry::<Counted>::new();
    registry
        .register(ready_worker("http://a:8000", "m"))
        .unwrap();
    assert_eq!(
        STANDING.get() - standing,
        2,
        "the empty one and the model's"
    );

    drop(registry.global_generation());
    assert_eq!(
        STANDING.get() - standing,
        3,
        "the model-less one stands once it is read"
    );

    registry
        .register(ready_worker("http://b:8000", "m"))
        .unwrap();
    assert_eq!(
        STANDING.get() - standing,
        2,
        "and falls at the change, with no read of it to come"
    );
}
