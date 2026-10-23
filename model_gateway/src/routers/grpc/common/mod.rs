//! Shared code for both regular and harmony routers

pub(crate) mod response_collection;
pub(crate) mod response_formatting;
pub(crate) mod responses;
pub(crate) mod stages;

/// Per-choice state of a streamed request, keyed by the choice index (the
/// engine's `0..n`, shifted by the prompt's offset in a batched completion),
/// asked on every chunk the engine sends. The index is a small integer the
/// gateway and the engine derive, never a value a client controls, so the
/// maps hash it with the one-multiply `FxHasher` rather than the standard
/// hasher's keyed SipHash.
pub(crate) type ChoiceMap<V> = rustc_hash::FxHashMap<u32, V>;

/// The choice indices in one state (stopped, completed, tailed); see
/// [`ChoiceMap`].
pub(crate) type ChoiceSet = rustc_hash::FxHashSet<u32>;
