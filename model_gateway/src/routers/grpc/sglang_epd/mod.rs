//! Explicit managed SGLang EPD. Legacy EPD never enters this executor.

pub mod executor;
pub mod planner;
pub mod registry;
pub mod router;

#[cfg(test)]
mod tests;
