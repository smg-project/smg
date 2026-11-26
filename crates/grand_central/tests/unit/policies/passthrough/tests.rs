use super::*;
use crate::stages::scorers::tests::candidate;

#[test]
fn passthrough_keeps_the_hosts_order() {
    let pool = [candidate(0, 9), candidate(1, 0)];
    let mut out = Targets::new();
    Passthrough.pick(
        &RequestFacts::default(),
        &pool,
        &[1, 0],
        &[0.0; 2],
        TieBand::NORMALIZED,
        &mut out,
    );
    let order: Vec<u32> = out.iter().map(|t| t.position).collect();
    assert_eq!(order, vec![1, 0]);
}
