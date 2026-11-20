//! The lead of a replica that elects.

use std::{sync::Arc, time::Instant};

use tokio::sync::watch;

use crate::{elector::Fence, worker_host::Lead};

/// The lead of a replica that elects: the term its fence admits now, while
/// the role's word says this replica leads.
///
/// The word is the one the serve tells its watches. It reads false until
/// the replica serves and from its stop on, where the fence may still
/// stand. The fence is read by the clock, as before every write a leader
/// makes: past its renew deadline it admits nothing, told or not.
pub(crate) struct ElectedLead {
    fence: Arc<Fence>,
    leads: watch::Receiver<bool>,
}

impl ElectedLead {
    /// The lead under `fence`, by the word `leads` holds.
    pub(crate) fn new(fence: Arc<Fence>, leads: watch::Receiver<bool>) -> Self {
        Self { fence, leads }
    }
}

impl Lead for ElectedLead {
    /// One read of the word and one of the fence, and no wait: the word's
    /// lock is held for the read alone, and its writer holds it for a write
    /// of the word alone.
    fn term(&self) -> Option<u64> {
        if !*self.leads.borrow() {
            return None;
        }
        self.fence.leader_term(Instant::now())
    }
}

/// Election settings under which a fence, once taken, stays open through a
/// case of any length: a case steps its elector by hand and renews nothing.
#[cfg(test)]
pub(super) fn lasting() -> crate::elector::ElectionSettings {
    use std::time::Duration;

    crate::elector::ElectionSettings {
        duration: Duration::from_secs(3 * 3_600),
        renew_period: Duration::from_secs(3_600),
        renew_deadline: Duration::from_secs(2 * 3_600),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elector::Elector;

    #[test]
    fn the_lead_names_its_term_while_the_word_and_the_fence_both_hold() {
        let mut elector = Elector::alone("replica", lasting()).expect("the settings are in order");
        let (word, leads) = watch::channel(false);
        let lead = ElectedLead::new(elector.fence(), leads);
        assert_eq!(lead.term(), None, "no lease and no word");

        elector.tick(Instant::now()).expect("a take");
        let term = elector.fence().term();
        assert_eq!(lead.term(), None, "a term the word has not told");
        word.send_replace(true);
        assert_eq!(lead.term(), Some(term));

        word.send_replace(false);
        assert_eq!(
            lead.term(),
            None,
            "the word turned under a fence that stands"
        );
        word.send_replace(true);
        assert!(elector.release().expect("a release"), "the lease was held");
        assert_eq!(
            lead.term(),
            None,
            "the fence closed under a word that stands"
        );

        elector.tick(Instant::now()).expect("the next take");
        let next = elector.fence().term();
        assert!(next > term, "a take after a release is another term");
        assert_eq!(lead.term(), Some(next));
    }
}
