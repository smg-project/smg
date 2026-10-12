use std::sync::Arc;

use tokenizers::{decoders::byte_level::ByteLevel, models::wordlevel::WordLevel};

use crate::{
    mock,
    traits::{Decoder, Encoder},
    Tokenizer,
};

fn byte_fallback_tokenizer() -> Arc<dyn crate::TokenizerTrait> {
    let definition = serde_json::json!({
        "version": "1.0", "added_tokens": [{
            "id": 7, "content": "[EOS]", "special": true, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false
        }],
        "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": {
            "[UNK]": 0, "regular": 1, "<0x74>": 2, "<0xF2>": 3,
            " suffix": 4, "<0xC3>": 5, "<0xA9>": 6, "[EOS]": 7
        }},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "ByteFallback"}, {"type": "Fuse"}
        ]}
    });
    let native =
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&definition).unwrap()).unwrap();
    Arc::new(crate::HuggingFaceTokenizer::from_tokenizer(native))
}

/// A ByteFallback vocabulary with every byte (`<0x00>`..`<0xFF>` = 2..=257),
/// one regular token `x` (1) and a special `[EOS]` (258).
fn byte_fallback_full_tokenizer(
    extra_decoders: Vec<serde_json::Value>,
) -> Arc<dyn crate::TokenizerTrait> {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".into(), serde_json::json!(0));
    vocab.insert("x".into(), serde_json::json!(1));
    for byte in 0u32..=255 {
        vocab.insert(format!("<0x{byte:02X}>"), serde_json::json!(byte + 2));
    }
    vocab.insert("[EOS]".into(), serde_json::json!(258));
    let mut decoders = vec![
        serde_json::json!({"type": "ByteFallback"}),
        serde_json::json!({"type": "Fuse"}),
    ];
    decoders.extend(extra_decoders);
    let definition = serde_json::json!({
        "version": "1.0", "added_tokens": [{
            "id": 258, "content": "[EOS]", "special": true, "single_word": false,
            "lstrip": false, "rstrip": false, "normalized": false
        }],
        "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": vocab},
        "decoder": {"type": "Sequence", "decoders": decoders}
    });
    let native =
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&definition).unwrap()).unwrap();
    Arc::new(crate::HuggingFaceTokenizer::from_tokenizer(native))
}

/// The byte tokens of `text` in the full-byte fixture.
fn byte_ids(text: &str) -> Vec<u32> {
    text.bytes().map(|byte| u32::from(byte) + 2).collect()
}

#[test]
fn byte_fallback_run_streams_each_complete_character() {
    let tokenizer = byte_fallback_full_tokenizer(Vec::new());
    // Nine byte tokens, three CJK characters: each streams as its last byte
    // arrives, nothing waits for the end of the run or of the input.
    let (deltas, batch, recoveries) = stream_all(
        &tokenizer,
        &byte_ids("日本語"),
        crate::StopSequenceConfig::default(),
        true,
    );
    assert_eq!(deltas, vec!["日", "本", "語"]);
    assert_eq!(deltas.concat(), batch);
    assert_eq!(recoveries, 0);
    // An emoji (four bytes) followed by a regular token and more bytes.
    let mut ids = byte_ids("😀");
    ids.push(1);
    ids.extend(byte_ids("é"));
    let (deltas, batch, recoveries) =
        stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
    assert_eq!(deltas, vec!["😀", "x", "é"]);
    assert_eq!(deltas.concat(), batch);
    assert_eq!(recoveries, 0);
    // An incomplete trailing sequence is the only thing held; the end of
    // input releases it as its own replacement characters. Batch decoding
    // would replace the whole run, the streamed character included, so this
    // counts as a recovery.
    let mut ids = byte_ids("日");
    ids.extend(byte_ids("本")[..2].iter());
    let (deltas, batch, recoveries) =
        stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
    assert_eq!(deltas, vec!["日", "\u{FFFD}\u{FFFD}"]);
    assert_eq!(batch, "\u{FFFD}".repeat(5));
    assert_eq!(run_bytes(&deltas.concat()), run_bytes(&batch));
    assert_eq!(recoveries, 1);
}

#[test]
fn byte_fallback_invalid_byte_keeps_streamed_text_and_counts_the_recovery() {
    let tokenizer = byte_fallback_full_tokenizer(Vec::new());
    let before = crate::decode_activity_stats();
    // "日本" streamed, then an invalid byte, then "語" and a regular token.
    // Batch decoding turns the whole run into ten replacement characters; the
    // stream keeps "日本" (six bytes) and sends the remaining four once
    // four bytes are pending (a run that cannot become valid), then "x".
    let mut ids = byte_ids("日本");
    ids.push(0xFF + 2);
    ids.extend(byte_ids("語"));
    ids.push(1);
    let (deltas, batch, recoveries) =
        stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
    assert_eq!(batch, format!("{}x", "\u{FFFD}".repeat(10)));
    assert_eq!(deltas, vec!["日", "本", "\u{FFFD}".repeat(4).as_str(), "x"]);
    assert_eq!(recoveries, 1);
    // The same run cut by the end of input instead of a regular token.
    let mut ids = byte_ids("日本");
    ids.push(0xFF + 2);
    ids.extend(byte_ids("語"));
    let (deltas, batch, recoveries) =
        stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
    assert_eq!(batch, "\u{FFFD}".repeat(10));
    assert_eq!(deltas, vec!["日", "本", "\u{FFFD}".repeat(4).as_str()]);
    assert_eq!(recoveries, 1);
    let after = crate::decode_activity_stats();
    let byte_run = |stats: &[crate::DecodeActivityStats; 2]| {
        stats
            .iter()
            .find(|s| s.kind == "byte_run")
            .map_or(0, |s| s.prefix_mismatches)
    };
    assert!(
        byte_run(&after) >= byte_run(&before) + 2,
        "{before:?} -> {after:?}"
    );
}

#[test]
fn decoder_rewrite_of_streamed_text_emits_the_remainder_without_failing() {
    // A `Replace` step after ByteFallback rewrites "regular suffix" to "X"
    // once the second token arrives: "regular" was streamed, so the stream
    // sends "X" and reports the recovery instead of failing.
    let definition = serde_json::json!({
        "version": "1.0", "added_tokens": [],
        "model": {"type": "WordLevel", "unk_token": "[UNK]", "vocab": {
            "[UNK]": 0, "regular": 1, "<0x74>": 2, "<0xF2>": 3, " suffix": 4
        }},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "ByteFallback"}, {"type": "Fuse"},
            {"type": "Replace", "pattern": {"String": "regular suffix"}, "content": "X"},
            {"type": "Replace", "pattern": {"String": "regular\u{FFFD}"}, "content": "Y"}
        ]}
    });
    let native =
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&definition).unwrap()).unwrap();
    let tokenizer: Arc<dyn crate::TokenizerTrait> =
        Arc::new(crate::HuggingFaceTokenizer::from_tokenizer(native));
    let before = crate::decode_activity_stats();
    let (deltas, batch, recoveries) = stream_all(
        &tokenizer,
        &[1, 4],
        crate::StopSequenceConfig::default(),
        true,
    );
    assert_eq!(batch, "X");
    assert_eq!(deltas, vec!["regular", "X"]);
    assert_eq!(recoveries, 1);
    // The rewrite only shows at the end of input: "regular" + an unfinished
    // byte decodes to "Y"; the end of input sends it rather than failing.
    let (deltas, batch, recoveries) = stream_all(
        &tokenizer,
        &[1, 3],
        crate::StopSequenceConfig::default(),
        true,
    );
    assert_eq!(batch, "Y");
    assert_eq!(deltas, vec!["regular", "Y"]);
    assert_eq!(recoveries, 1);
    let after = crate::decode_activity_stats();
    let decoder = |stats: &[crate::DecodeActivityStats; 2]| {
        stats
            .iter()
            .find(|s| s.kind == "decoder")
            .map_or(0, |s| s.prefix_mismatches)
    };
    assert!(
        decoder(&after) >= decoder(&before) + 2,
        "{before:?} -> {after:?}"
    );
}

#[test]
fn byte_fallback_eager_stop_observation_survives_an_invalid_byte_after_a_character() {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
    // <0xC3> <0xA9> <0xF2> " suffix": "é" is observed, then batch decoding
    // rewrites the run as three replacement characters. The eager observer
    // keeps "é", goes on with "\u{FFFD} suffix", and never fails.
    let tokenizer = byte_fallback_tokenizer();
    let ids = [5, 6, 3, 4];
    for (stops, expected) in [
        (vec!["t"], None),
        (vec!["regular"], None),
        (vec!["suffix"], Some("suffix")),
        (vec!["é"], Some("é")),
        (vec!["\u{FFFD} suf"], Some("\u{FFFD} suf")),
    ] {
        let mut config = StopSequenceConfig::default();
        for stop in &stops {
            config = config.with_stop_sequence(*stop);
        }
        let mut decoder = StopSequenceDecoder::for_stop_matching(tokenizer.clone(), config, true);
        let mut text = String::new();
        let mut stopped = false;
        for &id in &ids {
            match decoder.process_token(id).unwrap() {
                SequenceDecoderOutput::Text(delta) => text.push_str(&delta),
                SequenceDecoderOutput::Held => {}
                SequenceDecoderOutput::StoppedWithText(delta) => {
                    text.push_str(&delta);
                    stopped = true;
                    break;
                }
                SequenceDecoderOutput::Stopped => {
                    stopped = true;
                    break;
                }
            }
        }
        if !stopped {
            if let SequenceDecoderOutput::Text(delta) = decoder.finish().unwrap() {
                text.push_str(&delta);
            }
        }
        assert_eq!(decoder.matched_stop(), expected, "{stops:?}");
        if expected.is_none() {
            assert_eq!(text, "é\u{FFFD} suffix", "{stops:?}");
            assert_eq!(decoder.prefix_mismatches(), 1, "{stops:?}");
        }
    }
}

#[test]
fn byte_fallback_capability_distinguishes_byte_level_and_survives_cache() {
    let mut byte_level = tokenizers::Tokenizer::new(WordLevel::default());
    byte_level.with_decoder(Some(ByteLevel::default()));
    let byte_level: Arc<dyn crate::TokenizerTrait> =
        Arc::new(crate::HuggingFaceTokenizer::from_tokenizer(byte_level));
    for (tokenizer, expected) in [(byte_fallback_tokenizer(), true), (byte_level, false)] {
        assert_eq!(tokenizer.uses_byte_fallback(), expected);
        let cached = crate::CachedTokenizer::new(tokenizer, crate::CacheConfig::default());
        assert_eq!(cached.uses_byte_fallback(), expected);
    }
    assert!(!mock::MockTokenizer::new().uses_byte_fallback());
}

/// Every byte-run fixture of this file at once: the streamed text, the batch
/// text, and how many steps streamed a remainder because batch decoding would
/// have rewritten characters already sent (an invalid byte after them).
fn stream_all(
    tokenizer: &Arc<dyn crate::TokenizerTrait>,
    ids: &[u32],
    config: crate::StopSequenceConfig,
    skip_special_tokens: bool,
) -> (Vec<String>, String, u32) {
    use crate::{SequenceDecoderOutput, StopSequenceDecoder};
    let mut decoder = StopSequenceDecoder::new(tokenizer.clone(), config, skip_special_tokens);
    let mut deltas = Vec::new();
    for &id in ids {
        match decoder.process_token(id).unwrap() {
            SequenceDecoderOutput::Text(delta) => deltas.push(delta),
            SequenceDecoderOutput::Held => {}
            other => panic!("Unexpected stop: {other:?}"),
        }
    }
    if let SequenceDecoderOutput::Text(delta) = decoder.finish().unwrap() {
        deltas.push(delta);
    }
    let batch = tokenizer.decode(ids, skip_special_tokens).unwrap();
    (deltas, batch, decoder.prefix_mismatches())
}

/// Bytes of text accounted as the engine produced them: a replacement
/// character stands for one byte of a collapsed run, any other character for
/// its own UTF-8 bytes. Equal for the streamed and the batch text of one
/// sequence whatever the stream kept.
fn run_bytes(text: &str) -> usize {
    text.chars()
        .map(|c| if c == '\u{FFFD}' { 1 } else { c.len_utf8() })
        .sum()
}

#[test]
fn byte_fallback_streaming_matches_batch_until_an_invalid_byte_follows_sent_text() {
    let tokenizer = byte_fallback_tokenizer();
    // (ids, the streamed deltas, recoveries): `t` (<0x74>) is a complete
    // character and streams at once; a <0xF2> after it makes batch decoding
    // rewrite the whole run as replacement characters, which the stream never
    // retracts: it keeps `t` and sends the rest of the run as batch would.
    for (ids, expected, recoveries) in [
        (vec![2, 3, 4], vec!["t", "\u{FFFD} suffix"], 1),
        (vec![1, 2, 3, 4], vec!["regular", "t", "\u{FFFD} suffix"], 1),
        (vec![2, 2, 4], vec!["t", "t", " suffix"], 0),
        (vec![5, 6, 4], vec!["é", " suffix"], 0),
        (vec![2, 7, 3, 4], vec!["t", "\u{FFFD} suffix"], 1),
        // A skipped special token before an open byte run: the run's own
        // replacement character is kept when the regular token closes it.
        (vec![7, 3, 1], vec!["\u{FFFD}regular"], 0),
    ] {
        let (deltas, batch, seen) =
            stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
        assert_eq!(deltas, expected, "{ids:?}");
        assert_eq!(seen, recoveries, "{ids:?}");
        if recoveries == 0 {
            assert_eq!(deltas.concat(), batch, "{ids:?}");
        } else {
            assert_ne!(deltas.concat(), batch, "{ids:?}");
            assert_eq!(run_bytes(&deltas.concat()), run_bytes(&batch), "{ids:?}");
        }
    }
}

#[test]
fn byte_fallback_eof_preserves_text_and_stops() {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};

    for cached in [false, true] {
        let native = byte_fallback_tokenizer();
        let tokenizer: Arc<dyn crate::TokenizerTrait> = if cached {
            Arc::new(crate::CachedTokenizer::new(
                native,
                crate::CacheConfig::default(),
            ))
        } else {
            native
        };
        // An unfinished run at the end of input is released as batch decoding
        // ends it; a `t` already sent stays, the invalid byte after it becomes
        // its replacement character (one recovery).
        for (ids, expected, recoveries) in [
            (vec![2], "t", 0),
            (vec![2, 3], "t\u{FFFD}", 1),
            (vec![5, 6], "é", 0),
            (vec![1, 2], "regulart", 0),
            (vec![1, 2, 3], "regulart\u{FFFD}", 1),
            (vec![2, 7, 3], "t\u{FFFD}", 1),
            (vec![3], "\u{FFFD}", 0),
            (vec![3, 3], "\u{FFFD}\u{FFFD}", 0),
            (vec![5], "\u{FFFD}", 0),
        ] {
            let (deltas, batch, seen) =
                stream_all(&tokenizer, &ids, StopSequenceConfig::default(), true);
            assert_eq!(deltas.concat(), expected, "{ids:?} cached={cached}");
            assert_eq!(seen, recoveries, "{ids:?} cached={cached}");
            if recoveries == 0 {
                assert_eq!(batch, expected, "{ids:?} cached={cached}");
            }
            let mut decoder =
                StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
            for &id in &ids {
                decoder.process_token(id).unwrap();
            }
            decoder.finish().unwrap();
            assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Held);
        }
        // A string stop inside a complete character stops at once.
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            StopSequenceConfig::default().with_stop_sequence("t"),
            true,
        );
        assert_eq!(
            decoder.process_token(2).unwrap(),
            SequenceDecoderOutput::Stopped
        );
        assert!(decoder.is_stopped());
        assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Held);
        // A string stop that needs the next byte waits for it, then matches.
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            StopSequenceConfig::default().with_stop_sequence("t suffix"),
            true,
        );
        assert_eq!(
            decoder.process_token(2).unwrap(),
            SequenceDecoderOutput::Held
        );
        assert_eq!(
            decoder.process_token(4).unwrap(),
            SequenceDecoderOutput::Stopped
        );
        assert_eq!(decoder.matched_stop(), Some("t suffix"));
    }
}

#[test]
fn byte_fallback_seeded_sequence_does_not_reemit_prompt() {
    let tokenizer = byte_fallback_tokenizer();
    for seed in [vec![1], vec![5]] {
        let mut unchanged = crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed, true);
        assert_eq!(unchanged.finish().unwrap(), "");
    }
    let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer.clone(), vec![1], true);
    assert_eq!(sequence.append_token(2).unwrap(), "t");
    assert_eq!(sequence.append_token(3).unwrap(), "");
    assert_eq!(sequence.append_token(4).unwrap(), "\u{FFFD} suffix");
    assert_eq!(sequence.finish().unwrap(), "");
    assert_eq!(sequence.prefix_mismatches(), 1);

    for seed in [vec![1, 5], vec![1, 5, 7]] {
        let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed, true);
        assert_eq!(sequence.append_token(4).unwrap(), "� suffix");
        assert_eq!(sequence.finish().unwrap(), "");
    }
    let mut sequence =
        crate::Sequence::with_tokens_and_options(tokenizer.clone(), vec![1, 5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "é");
    assert_eq!(sequence.finish().unwrap(), "");

    // An incomplete prefill must not consume generated bytes when they become
    // a valid character; that character streams as soon as it is complete.
    let tokenizer = byte_fallback_tokenizer();
    let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer.clone(), vec![5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "é");
    assert_eq!(sequence.append_token(4).unwrap(), " suffix");
    assert_eq!(sequence.finish().unwrap(), "");
    let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer, vec![5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "é");
    assert_eq!(sequence.finish().unwrap(), "");
}

#[test]
fn byte_fallback_completed_seed_bytes_do_not_reemit_or_change_prefix() {
    let native = byte_fallback_tokenizer();
    for cached in [false, true] {
        let tokenizer: Arc<dyn crate::TokenizerTrait> = if cached {
            Arc::new(crate::CachedTokenizer::new(
                native.clone(),
                crate::CacheConfig::default(),
            ))
        } else {
            native.clone()
        };
        for seed in [vec![2], vec![1, 2], vec![1, 2, 7]] {
            let mut unchanged =
                crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed.clone(), true);
            assert_eq!(unchanged.finish().unwrap(), "");
            for (next, streamed, finished) in [(2, "t", ""), (3, "", "\u{FFFD}")] {
                let mut sequence =
                    crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed.clone(), true);
                assert_eq!(
                    sequence.append_token(next).unwrap(),
                    streamed,
                    "{seed:?} cached={cached}"
                );
                assert_eq!(
                    sequence.finish().unwrap(),
                    finished,
                    "{seed:?} cached={cached}"
                );
                assert_eq!(sequence.prefix_mismatches(), 0, "{seed:?} cached={cached}");
            }
        }
    }
}

#[test]
fn byte_fallback_short_combinations_match_batch_decode() {
    use crate::StopSequenceConfig;
    let tokenizer = byte_fallback_tokenizer();
    for skip_special in [false, true] {
        for a in 1..=7 {
            for b in 1..=7 {
                for c in 1..=7 {
                    let ids = [a, b, c];
                    let (deltas, batch, recoveries) = stream_all(
                        &tokenizer,
                        &ids,
                        StopSequenceConfig::default(),
                        skip_special,
                    );
                    let text = deltas.concat();
                    if recoveries == 0 {
                        assert_eq!(text, batch, "{ids:?} skip_special={skip_special}");
                    } else {
                        // Batch decoding rewrote characters already streamed as
                        // replacement characters (one per byte): the stream kept
                        // them and sent the rest of the run as batch ends it.
                        assert_ne!(text, batch, "{ids:?} skip_special={skip_special}");
                        assert_eq!(
                            run_bytes(&text),
                            run_bytes(&batch),
                            "{ids:?} skip_special={skip_special}"
                        );
                        assert!(
                            batch.contains('\u{FFFD}'),
                            "{ids:?} skip_special={skip_special}: {text:?} vs {batch:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn byte_fallback_token_stop_preserves_pending_text() {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
    let tokenizer = byte_fallback_tokenizer();
    for visible in [false, true] {
        let config = if visible {
            StopSequenceConfig::default().with_visible_stop_token(7)
        } else {
            StopSequenceConfig::default().with_stop_token(7)
        };
        let mut decoder = StopSequenceDecoder::new(tokenizer.clone(), config, false);
        assert_eq!(
            decoder.process_token(2).unwrap(),
            SequenceDecoderOutput::Text("t".to_string())
        );
        // An incomplete run before the stop token is released by it.
        assert_eq!(
            decoder.process_token(3).unwrap(),
            SequenceDecoderOutput::Held
        );
        assert_eq!(
            decoder.process_token(7).unwrap(),
            SequenceDecoderOutput::StoppedWithText(
                if visible { "\u{FFFD}[EOS]" } else { "\u{FFFD}" }.to_string(),
            )
        );
        assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Held);
    }
    let mut decoder = StopSequenceDecoder::new(
        tokenizer,
        StopSequenceConfig::default()
            .with_stop_token(7)
            .with_stop_sequence("t"),
        true,
    );
    assert_eq!(
        decoder.process_token(2).unwrap(),
        SequenceDecoderOutput::Stopped
    );
    assert_eq!(decoder.matched_stop(), Some("t"));
    assert_eq!(
        decoder.process_token(7).unwrap(),
        SequenceDecoderOutput::Stopped
    );
}

/// Every token through a stop decoder with no stops: the text of each step
/// (`None` where it held), the end-of-input text, the recovery count.
fn streamed_steps(
    tokenizer: &Arc<dyn crate::TokenizerTrait>,
    ids: &[u32],
) -> (Vec<Option<String>>, String, u32) {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
    let steps = ids
        .iter()
        .map(|&id| match decoder.process_token(id).unwrap() {
            SequenceDecoderOutput::Text(delta) => Some(delta),
            SequenceDecoderOutput::Held => None,
            other => panic!("unexpected stop: {other:?}"),
        })
        .collect();
    let released = match decoder.finish().unwrap() {
        SequenceDecoderOutput::Text(delta) => delta,
        SequenceDecoderOutput::Held => String::new(),
        other => panic!("unexpected stop: {other:?}"),
    };
    (steps, released, decoder.prefix_mismatches())
}

#[test]
fn byte_fallback_invalid_run_streams_one_replacement_per_byte_as_it_comes() {
    let tokenizer = byte_fallback_full_tokenizer(vec![]);
    // "日本", an invalid byte, "語", a regular token: with four bytes pending
    // the run cannot become valid, so their replacement characters stream at
    // once and every further byte's as it comes, instead of waiting for the
    // regular token; the text at the end is the same as before (日本, four
    // replacement characters, x) and one recovery counts.
    let mut ids = byte_ids("日本");
    ids.push(0xFF + 2);
    ids.extend(byte_ids("語"));
    ids.push(1);
    let (steps, released, recoveries) = streamed_steps(&tokenizer, &ids);
    let r = "\u{FFFD}".to_string();
    assert_eq!(
        steps,
        vec![
            None,
            None,
            Some("日".into()),
            None,
            None,
            Some("本".into()),
            None,
            None,
            None,
            Some(r.repeat(4)),
            Some("x".into()),
        ]
    );
    assert_eq!(released, "");
    assert_eq!(recoveries, 1);
    // An incomplete sequence before the invalid byte belongs to the lost run:
    // its bytes stream as replacement characters with the invalid one and
    // the byte after it (four pending), then one per byte.
    let mut ids = byte_ids("日");
    ids.extend(byte_ids("本")[..2].iter());
    ids.push(0xFF + 2);
    ids.extend(byte_ids("語"));
    let (steps, released, recoveries) = streamed_steps(&tokenizer, &ids);
    assert_eq!(steps[..3], [None, None, Some("日".into())]);
    assert_eq!(steps[3..7], [None, None, None, Some(r.repeat(4))]);
    assert_eq!(steps[7..], [Some(r.clone()), Some(r.clone())]);
    assert_eq!(released, "");
    assert_eq!(recoveries, 1);
    // Runs without an invalid byte are unchanged: each character as it closes.
    let (steps, released, recoveries) = streamed_steps(&tokenizer, &byte_ids("日本語"));
    let texts: Vec<_> = steps.into_iter().flatten().collect();
    assert_eq!(texts, vec!["日", "本", "語"]);
    assert_eq!(released, "");
    assert_eq!(recoveries, 0);
}

#[test]
fn raw_decode_step_over_an_invalid_run_is_bounded_and_does_not_count() {
    // The raw step (what the gateway replays for token accounting) gets the
    // same bound as the client path over an invalid run, and the process-wide
    // counters do not move for it. The counters are shared with the other
    // tests of this binary, so the run repeats: a counting step would add
    // RUNS, concurrent tests a few.
    const RUNS: u64 = 50;
    let tokenizer = byte_fallback_full_tokenizer(vec![]);
    let text: String = "日本語".chars().cycle().take(700).collect();
    let mut run: Vec<u32> = byte_ids(&text)[..2_000].to_vec();
    run[9] = 0xFF + 2;
    run.push(1);
    let before = crate::decode_activity_stats();
    for _ in 0..RUNS {
        let (mut ids, mut prefix, mut prefix_index) = (Vec::new(), String::new(), 0usize);
        let mut streamed = String::new();
        let mut max_window = 0;
        for &id in &run {
            let step = tokenizer
                .decode_step(id, &mut ids, &mut prefix, &mut prefix_index, true)
                .unwrap();
            if let Some(delta) = step {
                streamed.push_str(&delta);
            }
            max_window = max_window.max(ids.len());
        }
        assert!(max_window <= 16, "held window grew to {max_window} ids");
        assert_eq!(streamed.chars().count(), 3 + (2_000 - 9) + 1);
        assert!(streamed.starts_with("日本語\u{FFFD}") && streamed.ends_with("\u{FFFD}x"));
    }
    let after = crate::decode_activity_stats();
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            a.prefix_mismatches - b.prefix_mismatches < RUNS,
            "{}: {} -> {}",
            a.kind,
            b.prefix_mismatches,
            a.prefix_mismatches
        );
    }
}

#[test]
fn byte_fallback_literal_replacement_characters_are_not_a_lost_run() {
    // A literal U+FFFD (`EF BF BD`) is a valid character whose decode ends in
    // the replacement character, so the step holds it; the bytes after it are
    // not a lost run: the stream equals the batch decode and counts nothing.
    let tokenizer = byte_fallback_full_tokenizer(vec![]);
    for text in ["\u{FFFD}é", "\u{FFFD}\u{FFFD}", "\u{FFFD}😀", "é\u{FFFD}日"] {
        let mut ids = byte_ids(text);
        ids.push(1);
        let (deltas, batch, recoveries) =
            stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
        assert_eq!(batch, format!("{text}x"));
        assert_eq!(deltas.concat(), batch, "{text:?}");
        assert_eq!(recoveries, 0, "{text:?}");
    }
    // An invalid byte after a literal U+FFFD: the run is lost from that byte
    // on, and nothing streamed before it is rewritten.
    let mut ids = byte_ids("\u{FFFD}");
    ids.push(0xFF + 2);
    ids.extend(byte_ids("é"));
    ids.push(1);
    let (deltas, batch, recoveries) =
        stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
    assert_eq!(batch, format!("{}x", "\u{FFFD}".repeat(6)));
    assert_eq!(deltas.concat(), batch);
    assert_eq!(recoveries, 0);
}

#[test]
fn byte_fallback_lost_run_after_a_literal_replacement_character_counts_bytes() {
    // "\u{FFFD}A" streamed as byte tokens, then invalid bytes, then a regular
    // token. With fewer than four invalid bytes the run closes through the
    // remainder path, which aligns the streamed text with the batch decode by
    // characters and takes the streamed literal U+FFFD for one byte: two
    // replacement characters more than the run has bytes. From four invalid
    // bytes on, the lost-run path streams one replacement character per byte.
    // Both keep "\u{FFFD}A" and count one recovery; the difference is
    // smg-lab #330.
    let tokenizer = byte_fallback_full_tokenizer(vec![]);
    for (invalid, expected_tail) in [(3, 5), (4, 4), (8, 8)] {
        let mut ids = byte_ids("\u{FFFD}A");
        ids.extend(vec![0xFF + 2; invalid]);
        ids.push(1);
        let (deltas, batch, recoveries) =
            stream_all(&tokenizer, &ids, crate::StopSequenceConfig::default(), true);
        assert_eq!(batch, format!("{}x", "\u{FFFD}".repeat(4 + invalid)));
        assert_eq!(deltas[0], "\u{FFFD}A", "{invalid} invalid bytes");
        assert_eq!(
            deltas.concat(),
            format!("\u{FFFD}A{}x", "\u{FFFD}".repeat(expected_tail)),
            "{invalid} invalid bytes"
        );
        assert_eq!(recoveries, 1, "{invalid} invalid bytes");
    }
}

#[test]
fn byte_fallback_invalid_run_keeps_the_decode_window_bounded() {
    // 2,000 byte tokens with an invalid byte at position 9: the held window
    // stays at a handful of tokens, so the whole run costs a bounded amount
    // of decoder work per token (the per-token work was proportional to the
    // position in the run before, quadratic over the run).
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
    let tokenizer = byte_fallback_full_tokenizer(vec![]);
    let text: String = "日本語".chars().cycle().take(700).collect();
    let mut ids: Vec<u32> = byte_ids(&text)[..2_000].to_vec();
    ids[9] = 0xFF + 2;
    ids.push(1);
    let mut decoder =
        StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
    let mut chars = 0;
    let mut max_window = 0;
    for &id in &ids {
        if let SequenceDecoderOutput::Text(delta) = decoder.process_token(id).unwrap() {
            chars += delta.chars().count();
        }
        max_window = max_window.max(decoder.sequence_window_len());
    }
    if let SequenceDecoderOutput::Text(delta) = decoder.finish().unwrap() {
        chars += delta.chars().count();
    }
    // 日本語 (three characters) streamed before the invalid byte, then one
    // replacement character per remaining byte, then x.
    assert_eq!(chars, 3 + (2_000 - 9) + 1);
    assert!(max_window <= 16, "held window grew to {max_window} tokens");
}

#[test]
fn raw_decode_step_recovers_but_does_not_count() {
    // <0xC3> <0xA9> <0xF2> " suffix" through the raw steps (what the gateway
    // replays for token accounting): the rewritten prefix is recovered, and
    // the process-wide counters do not move for it. The counters are shared
    // with the other tests of this binary, so the raw steps run a thousand
    // times: a counting step would add a thousand, concurrent tests a few.
    const RUNS: u64 = 1_000;
    let tokenizer = byte_fallback_tokenizer();
    let before = crate::decode_activity_stats();
    for eager in [false, true] {
        for _ in 0..RUNS {
            let (mut ids, mut prefix, mut prefix_index) = (Vec::new(), String::new(), 0usize);
            let mut text = String::new();
            for id in [5, 6, 3, 4] {
                let step = if eager {
                    tokenizer.decode_step_eager(id, &mut ids, &mut prefix, &mut prefix_index, true)
                } else {
                    tokenizer.decode_step(id, &mut ids, &mut prefix, &mut prefix_index, true)
                };
                if let Some(delta) = step.unwrap() {
                    text.push_str(&delta);
                }
            }
            assert!(
                text.starts_with("é") && text.ends_with(" suffix"),
                "eager={eager}: {text:?}"
            );
        }
    }
    let after = crate::decode_activity_stats();
    for (b, a) in before.iter().zip(after.iter()) {
        assert!(
            a.prefix_mismatches - b.prefix_mismatches < RUNS,
            "{}: {} -> {}",
            a.kind,
            b.prefix_mismatches,
            a.prefix_mismatches
        );
    }
}

#[test]
fn test_mock_tokenizer_encode() {
    let tokenizer = mock::MockTokenizer::new();
    let encoding = tokenizer.encode("Hello world", false).unwrap();
    let token_ids = encoding.token_ids();
    assert_eq!(token_ids, &[1, 2]); // "Hello" -> 1, "world" -> 2
}

#[test]
fn test_mock_tokenizer_decode() {
    let tokenizer = mock::MockTokenizer::new();
    let text = tokenizer.decode(&[1, 2], false).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_mock_tokenizer_decode_skip_special() {
    let tokenizer = mock::MockTokenizer::new();

    // With special tokens
    let text = tokenizer.decode(&[1000, 1, 2, 999], false).unwrap();
    assert_eq!(text, "<bos> Hello world <eos>");

    // Without special tokens
    let text = tokenizer.decode(&[1000, 1, 2, 999], true).unwrap();
    assert_eq!(text, "Hello world");
}

#[test]
fn test_tokenizer_wrapper() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let encoding = tokenizer.encode("Hello world", false).unwrap();
    assert_eq!(encoding.token_ids(), &[1, 2]);

    let text = tokenizer.decode(&[1, 2], false).unwrap();
    assert_eq!(text, "Hello world");

    assert_eq!(tokenizer.vocab_size(), 14);

    assert_eq!(tokenizer.token_to_id("Hello"), Some(1));
    assert_eq!(tokenizer.token_to_id("unknown"), None);

    assert_eq!(tokenizer.id_to_token(1), Some("Hello".to_string()));
    assert_eq!(tokenizer.id_to_token(9999), None);
}

#[test]
fn test_decode_stream_basic() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Create a decode stream with initial tokens [1, 2] = "Hello world"
    let initial_tokens = vec![1, 2];
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Step with token 3 ("test").
    // DecodeStream computes:
    //   prefix_text = decode([1, 2]) = "Hello world"
    //   new_text    = decode([1, 2, 3]) = "Hello world test"
    // The incremental output is the suffix beyond prefix_text: " test"
    let result = stream.step(3).unwrap();
    assert_eq!(result, Some(" test".to_string()));

    // The stream should now track all three tokens
    assert_eq!(stream.tokens(), &[1, 2, 3]);
}

#[test]
fn test_decode_stream_multiple_steps() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Start with a single token [1] = "Hello"
    let initial_tokens = vec![1];
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Step with token 2 ("world"):
    //   prefix_text = decode([1]) = "Hello"
    //   new_text    = decode([1, 2]) = "Hello world"
    //   incremental = " world"
    let result = stream.step(2).unwrap();
    assert_eq!(result, Some(" world".to_string()));

    // Step with token 3 ("test"):
    //   prefix_text = decode([2]) = "world"
    //   new_text    = decode([2, 3]) = "world test"
    //   incremental = " test"
    let result = stream.step(3).unwrap();
    assert_eq!(result, Some(" test".to_string()));

    assert_eq!(stream.tokens(), &[1, 2, 3]);
}

#[test]
fn test_decode_stream_flush() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let initial_tokens = vec![1]; // "Hello"
    let mut stream = tokenizer.decode_stream(&initial_tokens, false);

    // Both steps produce text, advancing read_offset each time
    let step1 = stream.step(2).unwrap();
    assert_eq!(step1, Some(" world".to_string()));

    let step2 = stream.step(3).unwrap();
    assert_eq!(step2, Some(" test".to_string()));

    // After successful steps that consumed all tokens, read_offset == all_token_ids.len(),
    // so flush has nothing remaining and returns None.
    let flushed = stream.flush().unwrap();
    assert_eq!(flushed, None);
}

#[test]
fn test_special_tokens() {
    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    let special_tokens = tokenizer.get_special_tokens();
    assert_eq!(special_tokens.bos_token, Some("<bos>".to_string()));
    assert_eq!(special_tokens.eos_token, Some("<eos>".to_string()));
    assert_eq!(special_tokens.unk_token, Some("<unk>".to_string()));
    assert!(special_tokens.sep_token.is_none());
    assert!(special_tokens.pad_token.is_none());
}

#[test]
fn test_batch_encode() {
    let tokenizer = mock::MockTokenizer::new();
    let inputs = vec!["Hello", "world", "test"];
    let encodings = tokenizer.encode_batch(&inputs, false).unwrap();

    assert_eq!(encodings.len(), 3);
    assert_eq!(encodings[0].token_ids(), &[1]); // "Hello" -> 1
    assert_eq!(encodings[1].token_ids(), &[2]); // "world" -> 2
    assert_eq!(encodings[2].token_ids(), &[3]); // "test" -> 3
}

#[test]
fn test_thread_safety() {
    use std::thread;

    let mock_tokenizer = Arc::new(mock::MockTokenizer::new());
    let tokenizer = Tokenizer::from_arc(mock_tokenizer);

    // Spawn multiple threads that use the same tokenizer
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let tokenizer_clone = tokenizer.clone();
            thread::spawn(move || {
                let text = "Hello test".to_string();
                let encoding = tokenizer_clone.encode(&text, false).unwrap();
                let decoded = tokenizer_clone.decode(encoding.token_ids(), false).unwrap();
                assert!(decoded.contains("Hello") || decoded.contains("test"));
                i
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Regression test: DecodeStream must not panic when prefix_text.len()
/// falls mid-codepoint in new_text.  This happens with real tokenizers
/// that use byte-fallback — partial byte tokens merge into multi-byte
/// characters when more context arrives, changing the byte length of
/// the prefix portion.
#[test]
fn test_decode_stream_multibyte_char_boundary() {
    use anyhow::Result;

    use crate::{
        stream::DecodeStream,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer as TokenizerTrait},
    };

    /// Mock tokenizer simulating byte-fallback context sensitivity.
    ///
    /// decode([1, 2])   → "abc"  (3 bytes — incomplete byte rendered as ASCII)
    /// decode([1, 2, 3]) → "ab🎉" (6 bytes — merged into 4-byte emoji)
    ///
    /// prefix_text.len() = 3 lands inside the emoji (bytes 2..6 in new_text).
    struct MultiByteTokenizer {
        special_tokens: SpecialTokens,
    }

    impl Encoder for MultiByteTokenizer {
        fn encode(&self, _input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            Ok(Encoding::Plain(vec![]))
        }

        fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
            inputs
                .iter()
                .map(|s| self.encode(s, add_special_tokens))
                .collect()
        }
    }

    impl Decoder for MultiByteTokenizer {
        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            Ok(match token_ids {
                [1, 2] => "abc".into(),
                [1, 2, 3] => "ab\u{1F389}".into(), // "ab🎉"
                _ => String::new(),
            })
        }
    }

    impl TokenizerTrait for MultiByteTokenizer {
        fn vocab_size(&self) -> usize {
            10
        }
        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special_tokens
        }
        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    let tokenizer: Arc<dyn TokenizerTrait> = Arc::new(MultiByteTokenizer {
        special_tokens: SpecialTokens::default(),
    });
    let prompt_tokens = vec![1, 2];
    let mut stream = DecodeStream::new(tokenizer, &prompt_tokens, false);

    // Without the char-boundary fix this panics:
    //   "byte index 3 is not a char boundary; it is inside '🎉' (bytes 2..6)"
    let result = stream.step(3).unwrap();
    assert_eq!(result, Some("\u{1F389}".to_string()));
}
