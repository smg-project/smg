//! The two synthetic corpora of the `parallel_encode` and `encode_volume`
//! benches: one generator, so that a corpus, a seed and a byte target give
//! both benches the same text.

const SYLLABLES: &[&str] = &[
    "ka", "ri", "to", "ne", "mu", "sa", "lo", "vi", "dek", "pon", "tra", "sel", "qui", "zam",
    "for", "bli", "ost", "uno", "chi", "wen",
];

const PROSE: &[&str] = &[
    "The harbour wakes before the town does. Fishing boats slip out past the breakwater while the \
     streetlights are still on, and by the time the bakery opens its shutters the first catch is \
     already being sorted on the quay. ",
    "A good map is a story told in one glance: where the rivers bend, which roads were built for \
     carts and which for cars, why the old centre sits on the hill. Reading it slowly rewards you \
     with the reasons behind the shapes. ",
    "清晨的市场里，摊贩们把蔬菜一筐一筐地摆好，水珠还挂在菜叶上。买菜的人不多，大家慢慢地挑选。",
    "fn parse_header(line: &str) -> Option<(&str, &str)> {\n    let (name, value) = line.split_once(':')?;\n    \
     Some((name.trim(), value.trim()))\n}\n",
    "Mountain weather turns in minutes. A clear morning can close into cloud by the time the path \
     reaches the ridge, and the walkers who packed a jacket despite the forecast are the ones \
     still smiling at the hut. ",
];

/// Repeated paragraphs with a running number: the word reuse of ordinary text,
/// which the encoders' piece caches serve well.
fn prose(seed: u64, bytes: usize) -> String {
    let mut text = String::with_capacity(bytes + 256);
    let mut k = seed;
    while text.len() < bytes {
        text.push_str(PROSE[(k % PROSE.len() as u64) as usize]);
        text.push_str(&format!("({k}) "));
        k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1) >> 3;
    }
    text
}

/// Pseudo-words, digits and short CJK runs: a vocabulary wider than a
/// thread's piece-cache share, no special tokens (the cuts come from the
/// pre-tokenizer's pieces alone). The environment variable named by
/// `corpus_var`, set to `prose`, selects [`prose`] instead.
pub(super) fn body(corpus_var: &str, seed: u64, bytes: usize) -> String {
    if std::env::var(corpus_var).is_ok_and(|v| v == "prose") {
        return prose(seed, bytes);
    }
    let mut text = String::with_capacity(bytes + 64);
    let mut k = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = || {
        k = k
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        k >> 33
    };
    let mut in_sentence = 0;
    while text.len() < bytes {
        let roll = next();
        match roll % 11 {
            0 => {
                let n = 3 + (next() % 6) as usize;
                for _ in 0..n {
                    let c = char::from_u32(0x4E00 + (next() % 2000) as u32).unwrap_or('\u{4E2D}');
                    text.push(c);
                }
            }
            1 => text.push_str(&format!("{}", next() % 100_000)),
            _ => {
                let n = 2 + (next() % 3) as usize;
                for _ in 0..n {
                    text.push_str(SYLLABLES[(next() % SYLLABLES.len() as u64) as usize]);
                }
            }
        }
        in_sentence += 1;
        if in_sentence >= 8 + (roll % 7) as usize {
            text.push_str(".\n");
            in_sentence = 0;
        } else {
            text.push(' ');
        }
    }
    text
}
