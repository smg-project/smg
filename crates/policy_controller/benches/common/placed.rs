//! Where a timed view build writes: buffers on a page's start.
//!
//! A row is written field by field, and a few of its stores do not sit on
//! their own size's boundary. A page boundary inside the buffer can split
//! one of them, and a store across two pages costs 6.5 to 7.5 ns: at 8 rows
//! a fifth of the build. Where a buffer's page boundaries fall is the
//! allocator's choice, by what the process allocated before, so one build
//! read two states by the length of the path it was launched from. A timed
//! id names that input: its buffers stand on a page's start, the same on
//! both sides of a comparison.

/// The page the buffers are placed against.
const PAGE: usize = 4096;

/// How often the allocator is asked. The one these benches run under hands
/// out every 16th byte of a page in turn, so 256 asks reach a page's start
/// and the bound leaves four rounds. A buffer large enough for it to map on
/// its own starts 16 bytes into a page and never on one, so a row count far
/// past 512 stops at the assertion below, by design.
const ASKS: usize = 1_024;

/// A buffer of capacity `n` that starts on a page's start. The allocator is
/// asked until it hands such an address; a reject stays allocated until the
/// end, so each ask gets another address. A build is never timed in a buffer
/// that is not placed: when no ask of the bound lands on a page's start the
/// bench stops here.
pub fn page_start_buffer<T>(n: usize) -> Vec<T> {
    let mut rejects = Vec::new();
    let mut buffer = Vec::<T>::with_capacity(n);
    for _ in 1..ASKS {
        if buffer.as_ptr().addr().is_multiple_of(PAGE) {
            break;
        }
        rejects.push(std::mem::replace(&mut buffer, Vec::with_capacity(n)));
    }
    assert!(
        buffer.as_ptr().addr().is_multiple_of(PAGE),
        "no buffer of {n} rows on a page's start in {ASKS} asks: a placed id is not timed elsewhere"
    );
    buffer
}
