//! Opt-in: the one-shot decode of real vocabularies against the `tokenizers`
//! crate's own decode. `SMG_DECODE_PARITY_DIRS` = comma-separated
//! `<name>=<directory with tokenizer.json>`; per vocabulary, the ids of a
//! long mixed text (the whole, and windows of it), random id sequences over
//! the vocabulary (byte tokens, specials and pieces mixed), under both
//! special-token settings; one line per vocabulary and a total; the test
//! fails on any difference and does nothing when the variable is unset.
#![expect(
    clippy::print_stdout,
    reason = "the harness reports per vocabulary on purpose"
)]

use llm_tokenizer::create_tokenizer;

const DIRS_ENV: &str = "SMG_DECODE_PARITY_DIRS";

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn the_one_shot_decode_matches_the_tokenizers_decode_on_real_vocabularies() {
    let Ok(list) = std::env::var(DIRS_ENV) else {
        println!("{DIRS_ENV} unset: nothing to compare");
        return;
    };
    let text = concat!(
        "The committee reviewed the quarterly figures and found the margins consistent with the forecast, ",
        "although the logistics costs had risen by a small fraction after the new routing policy took effect in March. ",
        "数据管道在三月份切换到新的路由策略后，物流成本略有上升。 Ünïcödé façade naïve coöperate — “quotes” … 12345 ",
        "<s> </s> [INST] hello [/INST] <0x41> <unk>\n\n  spaces   and\ttabs\n"
    )
    .repeat(40);
    let (mut vocabularies, mut compared, mut mismatches) = (0usize, 0usize, 0usize);
    for spec in list.split(',').filter(|s| !s.is_empty()) {
        let Some((name, dir)) = spec.split_once('=') else {
            continue;
        };
        let ours = match create_tokenizer(dir) {
            Ok(t) => t,
            Err(e) => {
                println!("DECODE-PARITY {name}: does not load: {e}");
                continue;
            }
        };
        let reference = match tokenizers::Tokenizer::from_file(format!("{dir}/tokenizer.json")) {
            Ok(t) => t,
            Err(e) => {
                println!("DECODE-PARITY {name}: tokenizers does not load it: {e}");
                continue;
            }
        };
        vocabularies += 1;
        let (mut here, mut wrong) = (0usize, 0usize);
        let mut cases: Vec<Vec<u32>> = Vec::new();
        let ids = ours
            .encode(&text, false)
            .expect("encode")
            .token_ids()
            .to_vec();
        cases.push(ids.clone());
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..200 {
            let start = (next(&mut state) as usize) % ids.len();
            let len = 1 + (next(&mut state) as usize) % 64;
            cases.push(ids[start..(start + len).min(ids.len())].to_vec());
        }
        let vocab_size = u32::try_from(ours.vocab_size()).unwrap_or(u32::MAX);
        for _ in 0..300 {
            let len = 1 + (next(&mut state) as usize) % 48;
            cases.push(
                (0..len)
                    .map(|_| (next(&mut state) as u32) % vocab_size)
                    .collect(),
            );
        }
        for skip in [false, true] {
            for case in &cases {
                let want = reference.decode(case, skip);
                let got = ours.decode(case, skip);
                here += 1;
                let same = match (&want, &got) {
                    (Ok(w), Ok(g)) => w == g,
                    (Err(_), Err(_)) => true,
                    _ => false,
                };
                if !same {
                    wrong += 1;
                    if wrong <= 3 {
                        println!(
                            "DECODE-PARITY-MISMATCH {name} skip={skip} ids {:?}: tokenizers {:?} vs ours {:?}",
                            &case[..case.len().min(24)],
                            want.as_ref().map(|s| s.chars().take(60).collect::<String>()),
                            got.as_ref().map(|s| s.chars().take(60).collect::<String>())
                        );
                    }
                }
            }
        }
        compared += here;
        mismatches += wrong;
        println!("DECODE-PARITY {name}: compared {here} mismatches {wrong}");
    }
    println!("DECODE-PARITY-TOTAL vocabularies {vocabularies} compared {compared} mismatches {mismatches}");
    assert_eq!(
        mismatches, 0,
        "the one-shot decode differs from the tokenizers decode"
    );
}
