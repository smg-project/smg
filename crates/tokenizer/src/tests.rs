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

#[test]
fn byte_fallback_streaming_preserves_exact_text() {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};

    let tokenizer = byte_fallback_tokenizer();
    for ids in [
        vec![2, 3, 4],
        vec![1, 2, 3, 4],
        vec![2, 2, 4],
        vec![5, 6, 4],
        vec![2, 7, 3, 4],
    ] {
        let mut decoder =
            StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
        let mut text = String::new();
        for &id in &ids {
            match decoder.process_token(id).unwrap() {
                SequenceDecoderOutput::Text(delta) => text.push_str(&delta),
                SequenceDecoderOutput::Held => {}
                other => panic!("Unexpected stop: {other:?}"),
            }
        }
        if let SequenceDecoderOutput::Text(delta) = decoder.finish().unwrap() {
            text.push_str(&delta);
        }
        assert_eq!(text, tokenizer.decode(&ids, true).unwrap(), "{ids:?}");
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
        for ids in [
            vec![2],
            vec![2, 3],
            vec![5, 6],
            vec![1, 2],
            vec![1, 2, 3],
            vec![2, 7, 3],
        ] {
            let mut decoder =
                StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
            let mut text = String::new();
            for &id in &ids {
                if let SequenceDecoderOutput::Text(delta) = decoder.process_token(id).unwrap() {
                    text.push_str(&delta);
                }
            }
            if let SequenceDecoderOutput::Text(delta) = decoder.finish().unwrap() {
                text.push_str(&delta);
            }
            assert_eq!(
                text,
                tokenizer.decode(&ids, true).unwrap(),
                "{ids:?} cached={cached}"
            );
            assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Held);
        }
        let mut decoder = StopSequenceDecoder::new(
            tokenizer.clone(),
            StopSequenceConfig::default().with_stop_sequence("t"),
            true,
        );
        assert_eq!(
            decoder.process_token(2).unwrap(),
            SequenceDecoderOutput::Held
        );
        assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Stopped);
        assert!(decoder.is_stopped());
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
    assert_eq!(sequence.append_token(2).unwrap(), "");
    assert_eq!(sequence.append_token(3).unwrap(), "");
    assert_eq!(sequence.append_token(4).unwrap(), "�� suffix");
    assert_eq!(sequence.finish().unwrap(), "");

    for seed in [vec![1, 5], vec![1, 5, 7]] {
        let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed, true);
        assert_eq!(sequence.append_token(4).unwrap(), "� suffix");
        assert_eq!(sequence.finish().unwrap(), "");
    }
    let mut sequence =
        crate::Sequence::with_tokens_and_options(tokenizer.clone(), vec![1, 5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "");
    assert_eq!(sequence.finish().unwrap(), "é");

    // An incomplete prefill must not consume generated bytes when they become
    // a valid character; the native streaming decoder emits that character.
    let tokenizer = byte_fallback_tokenizer();
    let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer.clone(), vec![5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "");
    assert_eq!(sequence.append_token(4).unwrap(), "é suffix");
    assert_eq!(sequence.finish().unwrap(), "");
    let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer, vec![5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "");
    assert_eq!(sequence.finish().unwrap(), "é");
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
            for (next, expected) in [(2, "t"), (3, "�")] {
                let mut sequence =
                    crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed.clone(), true);
                assert_eq!(sequence.append_token(next).unwrap(), "");
                assert_eq!(
                    sequence.finish().unwrap(),
                    expected,
                    "{seed:?} cached={cached}"
                );
            }
        }
    }
}

#[test]
fn byte_fallback_short_combinations_match_batch_decode() {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
    let tokenizer = byte_fallback_tokenizer();
    for skip_special in [false, true] {
        for a in 1..=7 {
            for b in 1..=7 {
                for c in 1..=7 {
                    let ids = [a, b, c];
                    let mut decoder = StopSequenceDecoder::new(
                        tokenizer.clone(),
                        StopSequenceConfig::default(),
                        skip_special,
                    );
                    let mut text = String::new();
                    for id in ids {
                        if let SequenceDecoderOutput::Text(delta) =
                            decoder.process_token(id).unwrap()
                        {
                            text.push_str(&delta);
                        }
                    }
                    if let SequenceDecoderOutput::Text(delta) = decoder.finish().unwrap() {
                        text.push_str(&delta);
                    }
                    assert_eq!(
                        text,
                        tokenizer.decode(&ids, skip_special).unwrap(),
                        "{ids:?} skip_special={skip_special}"
                    );
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
            SequenceDecoderOutput::Held
        );
        assert_eq!(
            decoder.process_token(7).unwrap(),
            SequenceDecoderOutput::StoppedWithText(
                if visible { "t[EOS]" } else { "t" }.to_string(),
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
        SequenceDecoderOutput::Held
    );
    assert_eq!(
        decoder.process_token(7).unwrap(),
        SequenceDecoderOutput::Stopped
    );
    assert_eq!(decoder.matched_stop(), Some("t"));
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
