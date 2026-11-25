//! The lead of a replica that elects.

use std::{sync::Arc, time::Instant};

use tokio::sync::watch;

#[cfg(test)]
use crate::service::elector;
use crate::{service::elector::Fence, worker_host::Lead};

/// The lead of a replica that elects: the term the role's word names, while
/// its fence admits that term now.
///
/// The word is the one the serve tells its watches: the term this replica
/// leads, if it leads one. It names none until the replica serves and from
/// its stop on, where the fence may still stand. The fence is read by the
/// clock, as before every write a leader makes: past its renew deadline it
/// admits nothing, told or not. A term the fence took and the word has not
/// told yet is no lead's: the task of that term starts at its word.
pub(crate) struct ElectedLead {
    fence: Arc<Fence>,
    led: watch::Receiver<Option<u64>>,
}

impl ElectedLead {
    /// The lead under `fence`, by the word `led` holds.
    pub(crate) fn new(fence: Arc<Fence>, led: watch::Receiver<Option<u64>>) -> Self {
        Self { fence, led }
    }
}

impl Lead for ElectedLead {
    /// One read of the word and one of the fence, and no wait: the word's
    /// lock is held for the read alone, and its writer holds it for a write
    /// of the word alone.
    fn term(&self) -> Option<u64> {
        let told = (*self.led.borrow())?;
        (self.fence.leader_term(Instant::now()) == Some(told)).then_some(told)
    }
}

/// Election settings under which a fence, once taken, stays open through a
/// case of any length: a case steps its elector by hand and renews nothing.
#[cfg(test)]
pub(super) fn lasting() -> elector::ElectionSettings {
    use std::time::Duration;

    elector::ElectionSettings {
        duration: Duration::from_secs(3 * 3_600),
        renew_period: Duration::from_secs(3_600),
        renew_deadline: Duration::from_secs(2 * 3_600),
    }
}

#[cfg(test)]
mod tests {
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
}
