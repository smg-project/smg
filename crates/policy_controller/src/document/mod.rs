//! The policy document: its schema and validator, where a request's routing key
//! comes from, and the calls an operator makes on a document in force.

#[cfg(feature = "operator")]
pub mod operator;
pub mod routing_key;
pub mod spec;
