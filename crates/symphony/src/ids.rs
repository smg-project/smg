//! The ids of an output's tool calls.
//!
//! A client keeps its calls and their results by id across a conversation, so an id must not
//! repeat between one turn's calls and the next's. The engine mints them from a nonce drawn when
//! it is made, so no table, assembler or caller chooses one: the output's call `index` is the
//! nonce `index` steps on, written as `call_` and 24 lowercase hex characters, the shape the
//! gateway's other parsers mint, so a client tells no difference between them.

/// The hex characters after `call_`.
const HEX: usize = 24;
/// The ids count within those characters.
const MASK: u128 = (1u128 << (4 * HEX)) - 1;

/// Where one output's call ids count from.
///
/// [`CallIds::fresh`] draws the nonce at random (96 bits), so no two engines share one and an
/// output's ids are distinct from every other output's; within the output, call `index` adds
/// `index` to the nonce, so the ids are distinct from each other and the same for the output
/// however its stream is cut. A time-ordered UUIDv7 would not do for the nonce: its in-process
/// counter steps consecutive values by one within a millisecond, which would make one engine's
/// call 1 the next engine's call 0. [`CallIds::from_nonce`] names the nonce, for a replay that
/// wants the ids reproducible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallIds {
    /// The first call's number.
    base: u128,
}

impl CallIds {
    /// Ids counting from a nonce drawn at random, which no other engine shares.
    pub fn fresh() -> Self {
        Self {
            base: rand::random::<u128>() & MASK,
        }
    }

    /// Ids counting from `nonce`, 24 hex characters; `None` when it is not that.
    pub fn from_nonce(nonce: &str) -> Option<Self> {
        if nonce.len() != HEX || !nonce.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u128::from_str_radix(nonce, 16)
            .ok()
            .map(|base| Self { base })
    }

    /// The starting nonce as 24 lowercase hex characters, for recording alongside a replay's
    /// inputs. Passing it to [`Self::from_nonce`] restores the same ids at every call index.
    /// Keep one nonce per output; reusing it for a new output would reuse that output's ids.
    pub fn nonce(&self) -> String {
        format!("{:0width$x}", self.base, width = HEX)
    }

    /// The id of the output's call `index`.
    pub fn id(&self, index: u32) -> String {
        format!(
            "call_{:0width$x}",
            self.base.wrapping_add(u128::from(index)) & MASK,
            width = HEX
        )
    }
}

/// Whether `id` has the shape the engine mints: `call_` and 24 lowercase hex characters.
#[cfg(test)]
pub(crate) fn is_minted(id: &str) -> bool {
    id.strip_prefix("call_").is_some_and(|hex| {
        hex.len() == HEX
            && hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_takes_the_nonce_its_index_steps_on() {
        let ids = CallIds::from_nonce("0123456789abcdef00000000").expect("24 hex");
        assert_eq!(ids.id(0), "call_0123456789abcdef00000000");
        assert_eq!(ids.id(1), "call_0123456789abcdef00000001");
        assert_eq!(ids.id(255), "call_0123456789abcdef000000ff");
        assert!(is_minted(&ids.id(0)));
    }

    #[test]
    fn fresh_nonces_differ_and_keep_the_shape() {
        let (first, second) = (CallIds::fresh(), CallIds::fresh());
        assert_ne!(first, second);
        assert_ne!(first.id(0), second.id(0));
        assert_ne!(first.id(1), second.id(0));
        for ids in [first, second] {
            assert!(is_minted(&ids.id(0)), "{}", ids.id(0));
            assert!(is_minted(&ids.id(7)), "{}", ids.id(7));
            assert_ne!(ids.id(0), ids.id(1));
        }
    }

    #[test]
    fn a_nonce_is_24_hex_characters_in_either_case() {
        assert!(CallIds::from_nonce("").is_none());
        assert!(CallIds::from_nonce("0123456789abcdef0000000").is_none());
        assert!(CallIds::from_nonce("0123456789abcdef000000000").is_none());
        assert!(CallIds::from_nonce("0123456789abcdef0000000g").is_none());
        assert!(CallIds::from_nonce("call_0123456789abcdef000").is_none());
        let upper = CallIds::from_nonce("0123456789ABCDEF00000000").expect("hex in upper case");
        assert_eq!(upper.id(0), "call_0123456789abcdef00000000");
    }

    #[test]
    fn a_recorded_nonce_restores_ids_including_wraparound() {
        for original in [
            CallIds::fresh(),
            CallIds::from_nonce("000000000000000000000000").expect("zero nonce"),
            CallIds::from_nonce("FFFFFFFFFFFFFFFFFFFFFFFF").expect("maximum nonce"),
        ] {
            let nonce = original.nonce();
            assert_eq!(nonce.len(), HEX);
            assert!(nonce
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
            let restored = CallIds::from_nonce(&nonce).expect("recorded nonce");
            assert_eq!(restored, original);
            for index in [0, 1, 255, u32::MAX] {
                assert_eq!(
                    restored.id(index),
                    original.id(index),
                    "{nonce}, index {index}"
                );
            }
        }
    }

    #[test]
    fn the_count_stays_within_the_24_characters() {
        let ids = CallIds::from_nonce("ffffffffffffffffffffffff").expect("24 hex");
        assert_eq!(ids.id(0), "call_ffffffffffffffffffffffff");
        assert_eq!(ids.id(1), "call_000000000000000000000000");
    }
}
