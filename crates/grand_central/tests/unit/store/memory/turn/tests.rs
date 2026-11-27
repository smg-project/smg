//! What the cases read of the files' turns: their takes, watches and workers.

use std::sync::atomic::Ordering::SeqCst;

use super::{File, MemoryStore};

impl File {
    /// Takes of this file's turn since the test last cleared them.
    pub(in crate::store::memory) fn takes(&self) -> usize {
        self.takes.load(SeqCst)
    }

    /// Starts this test's count of turn takes again.
    pub(in crate::store::memory) fn clear_takes(&self) {
        self.takes.store(0, SeqCst);
    }
}

impl MemoryStore {
    /// How many files show a watch, for the cases: no door of the store
    /// tells it.
    pub(crate) fn watches(&self) -> usize {
        let files = self.files.iter();
        files.filter(|file| file.shown.watched.load(SeqCst)).count()
    }

    /// How many files show a thread at work, for the deciders' cases: no
    /// door of the store tells it.
    #[cfg(feature = "controller")]
    pub(crate) fn working(&self) -> usize {
        let files = self.files.iter();
        files.filter(|file| file.has_worker()).count()
    }
}
