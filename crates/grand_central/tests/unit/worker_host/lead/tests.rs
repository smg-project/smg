use super::*;

#[test]
fn a_host_with_no_election_leads_under_its_one_term() {
    assert_eq!(SoleLead::TERM, 0);
    assert_eq!(SoleLead.term(), Some(0));
}
