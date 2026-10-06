//! Shared by the integration tests: the ways an output is cut into deltas.

/// The cut offsets to replay `text` with: whole, every two-way split, byte by byte, and thirty
/// seeded plans of one to eight characters; every cut on a character boundary.
pub fn chunkings(text: &str) -> Vec<Vec<usize>> {
    let chars: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    let mut plans = vec![vec![]];
    plans.extend(chars.iter().skip(1).map(|&cut| vec![cut]));
    plans.push(chars.iter().skip(1).copied().collect());
    for seed in 1..=30u64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut cuts = Vec::new();
        let mut at = 0;
        while at < chars.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            at += 1 + (state >> 33) as usize % 8;
            if at < chars.len() {
                cuts.push(chars[at]);
            }
        }
        plans.push(cuts);
    }
    plans
}
