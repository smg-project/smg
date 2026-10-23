//! The binary's command line, by concern: `url_port` is what the gateway
//! reads before clap sees it, `value` the parsers of the flags' values,
//! `secrets` the settings a secret can come from.

pub(crate) mod secrets;
pub(crate) mod url_port;
pub(crate) mod value;
