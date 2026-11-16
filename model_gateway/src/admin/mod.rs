//! The control plane's typed documents: a configuration a running gateway
//! holds in force and answers with its revision and ETag, and replaces only
//! against the ETag the writer read. Beside them, the body, query and path
//! extractors of its management endpoints, which refuse a request in the
//! control plane's own error envelope (`{"error": ..., "code": ...}`).

pub mod document;
pub(crate) mod json;
pub(crate) mod params;
