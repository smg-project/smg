use super::*;
use crate::service::elector::Elector;

#[test]
fn the_lead_names_its_term_while_the_word_and_the_fence_both_hold() {
    let mut elector = Elector::alone("replica", lasting()).expect("the settings are in order");
    let (word, led) = watch::channel(None);
    let lead = ElectedLead::new(elector.fence(), led);
    assert_eq!(lead.term(), None, "no lease and no word");

    elector.tick(Instant::now()).expect("a take");
    let term = elector.fence().term();
    assert_eq!(lead.term(), None, "a term the word has not told");
    word.send_replace(Some(term));
    assert_eq!(lead.term(), Some(term));

    word.send_replace(None);
    assert_eq!(
        lead.term(),
        None,
        "the word turned under a fence that stands"
    );
    word.send_replace(Some(term));
    assert!(elector.release().expect("a release"), "the lease was held");
    assert_eq!(
        lead.term(),
        None,
        "the fence closed under a word that stands"
    );

    elector.tick(Instant::now()).expect("the next take");
    let next = elector.fence().term();
    assert!(next > term, "a take after a release is another term");
    assert_eq!(lead.term(), None, "a term the word has not told yet");
    word.send_replace(Some(next));
    assert_eq!(lead.term(), Some(next));
}
