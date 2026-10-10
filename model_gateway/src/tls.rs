//! The process-level rustls crypto provider.

use rustls::crypto::{ring, CryptoProvider};

/// Installs `ring` as the process-level rustls crypto provider, unless one
/// is installed already.
///
/// `ring` is the only backend compiled into this crate's binaries. The bare
/// rustls builders (the HTTPS listener, the mesh listener) pick it up on
/// their own, but the HTTP client is built with reqwest's
/// `rustls-no-provider` feature and uses whatever provider the process
/// installed, panicking when there is none. Installing is idempotent: a
/// provider installed earlier, by an embedding process or a previous start in
/// the same process, stays. [`crate::server::startup`] calls it first, and
/// so does every place this crate builds an HTTP client (the worker's lazy
/// client, `build_client`, the external discovery step's static client);
/// test code that builds a reqwest client directly calls it itself.
pub fn install_crypto_provider() {
    if CryptoProvider::get_default().is_none() {
        // An error here means another thread installed one in between,
        // which is the outcome wanted.
        let _ = ring::default_provider().install_default();
    }
}
