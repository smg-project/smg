//! The named policies, one file each. A file says what its policy is made of,
//! in its `profile()`, and holds the stages and state written for the policy.
//!
//! A `profile()` names its stages by their catalog names, and the
//! [`StageRegistry`](crate::stages::StageRegistry) maps each name to its type.
//! Every profile starts with the three filters of `standard_filters`, in this
//! file.
//!
//! A stage written for one policy can serve another. `power_of_two` and
//! `cache_aware` score with [`least_load::LeastLoad`]. `prefix_hash` scores
//! with [`consistent_hashing::ConsistentHash`], which keys a request's prefix
//! through `prefix_hash`. `prefix_hash`, `consistent_hashing` and the policy
//! document read the gates of `cache_aware` ([`cache_aware::Gates`]). The
//! engine draws with [`random::Random`] under a sessions rule whose
//! assignment is random.
//!
//! What no one policy owns lives outside this folder. A stage written for no
//! one policy is in [`stages`](crate::stages), and so is the fence of retired
//! workers that `consistent_hashing` and `bucket` each hold. The warm-up
//! ruling that a stage of `cache_aware` applies is in
//! [`engine`](crate::engine), where the engine and the store read it.

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
