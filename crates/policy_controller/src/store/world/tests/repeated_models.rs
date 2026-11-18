//! The cases of a row that lists a model more than once: the model's pool
//! holds the worker's one row, wherever the repeat stands in the list and
//! whichever row was read before it.

use super::facts;
use crate::{
    store::{MemoryStore, WorkersStore},
    view::Role,
};

#[test]
fn a_model_listed_again_anywhere_in_a_row_is_its_pools_one_row() {
    let store = MemoryStore::new();
    // Each repeat stands apart from its first listing, and two rows repeat
    // one model: the row read first leaves its entry in the pool the next
    // row repeats.
    let wide = ["one", "two", "one", "three", "two", "one"];
    store.upsert(facts("http://a:8000", Role::Single, &wide));
    store.publish_now();
    store.upsert(facts("http://b:8000", Role::Single, &["two", "one", "two"]));
    store.publish_now();
    store.upsert(facts("http://c:8000", Role::Single, &["three"]));
    store.publish_now();
    let world = store.world();
    assert_eq!(world.members("one", Role::Single), Some(2));
    assert_eq!(world.members("two", Role::Single), Some(2));
    assert_eq!(world.members("three", Role::Single), Some(2));
}

#[test]
fn a_repeat_stands_once_on_its_own_side_of_a_pair() {
    let store = MemoryStore::new();
    store.upsert(facts("http://p:8000", Role::Prefill, &["one", "one"]));
    store.publish_now();
    store.upsert(facts("http://d:8000", Role::Decode, &["one", "two", "one"]));
    store.publish_now();
    store.upsert(facts("http://e:8000", Role::Decode, &["one"]));
    store.publish_now();
    let world = store.world();
    assert_eq!(world.members("one", Role::Prefill), Some(1));
    assert_eq!(world.members("one", Role::Decode), Some(2));
    assert_eq!(world.members("two", Role::Decode), Some(1));
}

#[test]
fn a_row_of_many_models_each_listed_twice_stands_once_in_every_pool_across_writes() {
    let store = MemoryStore::new();
    let names: Vec<String> = (0..300).map(|model| format!("model-{model}")).collect();
    // The whole list, then the whole list again: each repeat stands 300
    // models behind its first listing.
    let listed: Vec<&str> = names.iter().chain(&names).map(String::as_str).collect();
    store.upsert(facts("http://wide:8000", Role::Single, &listed));
    store.publish_now();
    store.upsert(facts("http://b:8000", Role::Single, &["model-7"]));
    store.publish_now();
    // One membership write more builds every pool again from the standing
    // ones.
    store.upsert(facts("http://c:8000", Role::Single, &["model-299"]));
    store.publish_now();
    let world = store.world();
    for (index, name) in names.iter().enumerate() {
        let members = if index == 7 || index == 299 { 2 } else { 1 };
        assert_eq!(world.members(name, Role::Single), Some(members), "{name}");
    }
}
