//! The named policies, one file each: what a policy is made of, and the stages
//! and state that only it uses. What several policies share lives in `stages`.

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
pub mod sticky_sessions;
