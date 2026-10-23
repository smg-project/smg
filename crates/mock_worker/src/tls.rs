//! The process-level rustls crypto provider.

use rustls::crypto::{ring, CryptoProvider};

/// Installs `ring` as the process-level rustls crypto provider, unless one
/// is installed already.
///
/// The HTTP clients of this crate's binaries take the process provider
/// (reqwest's `rustls-no-provider` feature) and panic at build time when
/// there is none; `ring` is the only backend compiled in. Each binary's
/// `main` calls this first thing, and so does the replay tool's client
/// builder, so its tests have the provider too.
pub fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}
