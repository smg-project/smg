use super::*;

/// A key's hash rank is its xxh3 modulo the ranks, pinned here by value:
/// a key keeps its rank in every process and build, and a change of hash
/// would move every live key to another engine's cold cache.
#[test]
fn a_keys_hash_rank_is_fixed_by_its_xxh3() {
    for (key, size, rank) in [
        ("conv-0", 8, 4),
        ("conv-1", 8, 5),
        ("conv-2", 8, 7),
        ("conv-2", 4, 3),
        ("session-a", 3, 1),
        ("session-b", 4, 2),
    ] {
        assert_eq!(hash_rank(key, size), rank, "{key} of {size}");
    }
}
