//! The bootstrap rendezvous a disaggregated request carries: the operator's wins and the engine's
//! fills the rest.

use super::*;

/// What the operator registered wins, field by field; what the engine
/// declared fills the rest; the default port is the last resort.
#[test]
fn the_operators_rendezvous_wins_and_the_engines_fills_the_rest() {
    let declared = Some(("10.0.0.7", 31755));
    assert_eq!(
        bootstrap_rendezvous("", None, declared),
        ("10.0.0.7", 31755)
    );
    assert_eq!(
        bootstrap_rendezvous("prefill-0", Some(9000), declared),
        ("prefill-0", 9000)
    );
    assert_eq!(
        bootstrap_rendezvous("prefill-0", None, declared),
        ("prefill-0", 31755)
    );
    assert_eq!(
        bootstrap_rendezvous("", Some(9000), declared),
        ("10.0.0.7", 9000)
    );
    assert_eq!(
        bootstrap_rendezvous("prefill-0", None, None),
        ("prefill-0", DEFAULT_BOOTSTRAP_PORT)
    );
}
