//! The named policies, one file each. A file says what its policy is made of,
//! in its `profile()`, and holds the stages and state written for the policy.
//!
//! A `profile()` names its stages by their catalog names, and the
//! [`Catalog`](crate::stages::Catalog) maps each name to its type. Every
//! profile starts with the three filters of `standard_filters`, in this file.
//!
//! A stage written for one policy can serve another. `power_of_two` and
//! `cache_aware` score with [`least_load::LeastLoad`]. `prefix_hash` scores
//! with [`consistent_hashing::ConsistentHash`], which keys a request's prefix
//! through `prefix_hash`. The engine draws with [`random::Random`] under a
//! sessions rule whose assignment is random.
//!
//! Three shared pieces still sit in a policy's file. `bucket` reads the fence
//! of retired workers in `consistent_hashing` ([`consistent_hashing::FENCE`]).
//! `prefix_hash`, `consistent_hashing` and the policy document read the block
//! arithmetic of `cache_aware` ([`cache_aware::Gates`], `blocks_within`). The
//! engine, the store and the shared stages read the warm-up ruling of
//! `cache_aware` ([`cache_aware::WarmRule`], [`cache_aware::Warming`]): the
//! [`Scorer`](crate::engine::Scorer) trait returns it, and the engine's
//! `decide_stated` and `decide_stated_keeping` call `forget_judged` of
//! `cache_aware`.
//!
//! A stage written for no one policy lives in [`stages`](crate::stages).

use crate::document::spec::FilterSpec;

pub mod bucket;
pub mod cache_aware;
pub mod consistent_hashing;
pub mod least_load;
pub mod manual;
pub mod passthrough;
pub mod power_of_two;
pub mod prefix_hash;
pub mod random;
pub mod round_robin;

/// The three filters every named policy starts with.
fn standard_filters() -> Vec<FilterSpec> {
    vec![
        "healthy".into(),
        "not-overloaded".into(),
        "not-tried".into(),
    ]
}
