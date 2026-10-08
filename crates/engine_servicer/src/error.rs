//! Errors a servicer surfaces to its (Python) lifecycle owner.

/// Errors a servicer surfaces to its (Python) lifecycle owner.
#[derive(Debug, thiserror::Error)]
pub enum ServicerError {
    #[error("invalid servicer config: {0}")]
    InvalidConfig(String),
    #[error("servicer failed to start: {0}")]
    Startup(String),
    #[error("servicer did not stop within the timeout")]
    StopTimeout,
    #[error("servicer state is poisoned")]
    Poisoned,
    #[error("servicer thread panicked")]
    ThreadPanicked,
}
