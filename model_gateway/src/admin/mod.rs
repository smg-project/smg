//! The control plane's typed documents: a configuration a running gateway
//! holds in force and answers with its revision and ETag, and replaces only
//! against the ETag the writer read.

pub mod document;
