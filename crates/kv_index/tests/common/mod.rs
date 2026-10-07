//! Helpers shared by the exactness and concurrency suites: a seeded xorshift generator and the
//! block builders that name a chain's blocks as an engine with chain hashes would.
#![allow(dead_code)]

use kv_index::{request_prefix_hashes, ContentHash, StoredBlock};

/// xorshift64*, enough for a deterministic corpus without pulling a dependency into the test.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn range(&mut self, lo: usize, hi_inclusive: usize) -> usize {
        lo + self.below(hi_inclusive - lo + 1)
    }

    pub fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next() % denominator < numerator
    }
}

pub fn content(stream: u64, position: usize) -> ContentHash {
    kv_index::compute_content_hash(&[
        (stream & 0xffff_ffff) as u32,
        (stream >> 32) as u32,
        position as u32,
    ])
}

/// Blocks of a content sequence as an engine would hash them: the engine hash is the chain hash
/// of the contents so far, which is unique per distinct prefix and shared by every worker that
/// stores the same prefix.
pub fn blocks_of(contents: &[ContentHash]) -> Vec<StoredBlock> {
    contents
        .iter()
        .zip(request_prefix_hashes(contents))
        .map(|(&content_hash, seq_hash)| StoredBlock {
            seq_hash,
            content_hash,
        })
        .collect()
}
