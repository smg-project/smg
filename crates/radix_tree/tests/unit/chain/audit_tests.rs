use super::*;
use crate::Config;

/// The strengthened audit must reject a double-freed chain index — a
/// duplicate in `free_chains` would let two logical chains later alias
/// one physical `ChainData`, silently corrupting overlap answers.
#[test]
fn audit_rejects_a_double_freed_chain() {
    let mut t = RadixTree::new(Config::default());
    let h = t.create_holder("w1");
    t.store(h, None, &[(1, 10), (2, 20)]).expect("store");
    t.clear(h); // frees the chain(s)
    t.audit().expect("a cleared tree is valid");
    assert!(!t.free_chains.is_empty(), "clear frees the chain");

    let dup = t.free_chains[0];
    t.free_chains.push(dup);
    let err = t.audit().expect_err("a double free must be caught");
    assert!(err.contains("double free"), "unexpected audit error: {err}");
}

/// The strengthened audit must reject an interner set held only by the
/// table (strong_count 1) — the leak class the release path can strand.
#[test]
fn audit_rejects_an_interner_orphan() {
    let mut t = RadixTree::new(Config::default());
    let h = t.create_holder("w1");
    t.store(h, None, &[(1, 10)]).expect("store");
    t.audit().expect("a fresh tree is valid");

    // An Arc no live span references — only the table holds it.
    let orphan: std::sync::Arc<[u32]> = std::sync::Arc::from([99u32].as_slice());
    t.interner.table.entry(0xDEAD).or_default().push(orphan);
    let err = t.audit().expect_err("an interner orphan must be caught");
    assert!(err.contains("orphan"), "unexpected audit error: {err}");
}
