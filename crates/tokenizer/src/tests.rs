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

/// Every token of `ids` through a stop decoder with no stops: the text each
/// token gives, `None` where the decoder holds, and the text the end of the
/// stream releases.
fn streamed(
    tokenizer: &Arc<dyn crate::TokenizerTrait>,
    ids: &[u32],
    skip_special_tokens: bool,
) -> (Vec<Option<String>>, String) {
    use crate::{SequenceDecoderOutput, StopSequenceConfig, StopSequenceDecoder};
    let mut decoder = StopSequenceDecoder::new(
        tokenizer.clone(),
        StopSequenceConfig::default(),
        skip_special_tokens,
    );
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
    (steps, released)
}

/// The streaming rule, spelled out: a run of byte tokens is text as soon as
/// it decodes to complete characters and is held while it does not, as the
/// reference stream does (and the bellwether fixtures pin). ByteFallback
/// decodes a run as one unit, so a run complete when its first bytes went out
/// can turn invalid at a later byte, when the decoder replaces every byte of
/// it; the emitted text stands, and the rest is one replacement character per
/// byte that failed. The end of the stream releases a held run the same way.
#[test]
fn byte_fallback_streaming_emits_a_run_as_soon_as_it_is_complete() {
    let tokenizer = byte_fallback_tokenizer();
    let cases: [(&[u32], &[Option<&str>], &str); 5] = [
        // `t` is a character on its own; the invalid byte after it is held,
        // and the token that closes the run brings one replacement for it.
        (&[2, 3, 4], &[Some("t"), None, Some("� suffix")], ""),
        (
            &[1, 2, 3, 4],
            &[Some("regular"), Some("t"), None, Some("� suffix")],
            "",
        ),
        (&[2, 2, 4], &[Some("t"), Some("t"), Some(" suffix")], ""),
        // Two bytes of one character: held, then the character.
        (&[5, 6, 4], &[None, Some("é"), Some(" suffix")], ""),
        // A skipped special token adds no text and does not close the run.
        (
            &[2, 7, 3, 4],
            &[Some("t"), None, None, Some("� suffix")],
            "",
        ),
    ];
    for (ids, steps, released) in cases {
        let expected: Vec<Option<String>> = steps.iter().map(|s| s.map(str::to_string)).collect();
        assert_eq!(
            streamed(&tokenizer, ids, true),
            (expected, released.to_string()),
            "{ids:?}"
        );
    }
}

#[test]
fn byte_fallback_eof_releases_an_incomplete_run_and_stops() {
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
        for (ids, text) in [
            (vec![2], "t"),
            // The invalid byte held at the end comes out as a replacement.
            (vec![2, 3], "t�"),
            (vec![5, 6], "é"),
            (vec![1, 2], "regulart"),
            (vec![1, 2, 3], "regulart�"),
            (vec![2, 7, 3], "t�"),
        ] {
            let (steps, released) = streamed(&tokenizer, &ids, true);
            let whole: String = steps.into_iter().flatten().chain([released]).collect();
            assert_eq!(whole, text, "{ids:?} cached={cached}");
            let mut decoder =
                StopSequenceDecoder::new(tokenizer.clone(), StopSequenceConfig::default(), true);
            for &id in &ids {
                decoder.process_token(id).unwrap();
            }
            decoder.finish().unwrap();
            assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Held);
        }
        // A complete byte is text at once, so a stop sequence it spells stops
        // at that token.
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
    assert_eq!(sequence.append_token(4).unwrap(), "� suffix");
    assert_eq!(sequence.finish().unwrap(), "");

    // An incomplete byte at the end of the seed is kept as decode context, and
    // comes out as a replacement when a real token closes it.
    for seed in [vec![1, 5], vec![1, 5, 7]] {
        let mut sequence = crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed, true);
        assert_eq!(sequence.append_token(4).unwrap(), "� suffix");
        assert_eq!(sequence.finish().unwrap(), "");
    }
    // ...or as the character it completes.
    let mut sequence =
        crate::Sequence::with_tokens_and_options(tokenizer.clone(), vec![1, 5], true);
    assert_eq!(sequence.append_token(6).unwrap(), "é");
    assert_eq!(sequence.finish().unwrap(), "");

    // An incomplete prefill must not consume generated bytes when they become
    // a valid character; the native streaming decoder emits that character.
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
            // A complete byte after the seed is text at once; an invalid one
            // is held to the end, which releases it as a replacement.
            for (next, appended, released) in [(2, "t", ""), (3, "", "�")] {
                let mut sequence =
                    crate::Sequence::with_tokens_and_options(tokenizer.clone(), seed.clone(), true);
                assert_eq!(
                    sequence.append_token(next).unwrap(),
                    appended,
                    "{seed:?} cached={cached}"
                );
                assert_eq!(
                    sequence.finish().unwrap(),
                    released,
                    "{seed:?} cached={cached}"
                );
            }
        }
    }
}

/// The streaming rule applied to the batch decodes of a sequence's prefixes:
/// a prefix whose decode is complete text (not ending in a replacement
/// character) and longer than what was emitted emits the rest, past the
/// emitted text when the decode still starts with it, else past as many
/// characters; the end of the stream releases the rest of the whole decode the
/// same way.
fn streaming_rule(tokenizer: &Arc<dyn crate::TokenizerTrait>, ids: &[u32], skip: bool) -> String {
    let past = |decoded: &str, emitted: &str| -> String {
        decoded.strip_prefix(emitted).map_or_else(
            || decoded.chars().skip(emitted.chars().count()).collect(),
            str::to_string,
        )
    };
    let mut emitted = String::new();
    for end in 1..=ids.len() {
        let decoded = tokenizer.decode(&ids[..end], skip).unwrap();
        if decoded.len() > emitted.len() && !decoded.ends_with('�') {
            let rest = past(&decoded, &emitted);
            emitted.push_str(&rest);
        }
    }
    let whole = tokenizer.decode(ids, skip).unwrap();
    let rest = past(&whole, &emitted);
    emitted.push_str(&rest);
    emitted
}

#[test]
fn byte_fallback_short_combinations_follow_the_streaming_rule() {
    let tokenizer = byte_fallback_tokenizer();
    let mut unlike_batch = 0;
    for skip_special in [false, true] {
        for a in 1..=7 {
            for b in 1..=7 {
                for c in 1..=7 {
                    let ids = [a, b, c];
                    let (steps, released) = streamed(&tokenizer, &ids, skip_special);
                    let whole: String = steps.into_iter().flatten().chain([released]).collect();
                    let batch = tokenizer.decode(&ids, skip_special).unwrap();
                    let expected = streaming_rule(&tokenizer, &ids, skip_special);
                    assert_eq!(whole, expected, "{ids:?} skip_special={skip_special}");
                    if whole != batch {
                        // Only a run that was complete and turned invalid can
                        // differ from the batch decode, by what was emitted.
                        assert!(
                            batch.chars().all(|c| c == '�' || whole.contains(c)),
                            "{ids:?} skip_special={skip_special}: {whole:?} vs batch {batch:?}"
                        );
                        unlike_batch += 1;
                    }
                }
            }
        }
    }
    assert!(
        unlike_batch > 0,
        "the rule and the batch decode never differed"
    );
}

#[test]
fn byte_fallback_token_stop_closes_the_run_before_it() {
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
        assert_eq!(
            decoder.process_token(7).unwrap(),
            if visible {
                SequenceDecoderOutput::StoppedWithText("[EOS]".to_string())
            } else {
                SequenceDecoderOutput::Stopped
            }
        );
        assert_eq!(decoder.finish().unwrap(), SequenceDecoderOutput::Held);
    }
    // An invalid byte held before the stop token comes out with the stop.
    let mut decoder = StopSequenceDecoder::new(
        tokenizer.clone(),
        StopSequenceConfig::default().with_stop_token(7),
        false,
    );
    assert_eq!(
        decoder.process_token(3).unwrap(),
        SequenceDecoderOutput::Held
    );
    assert_eq!(
        decoder.process_token(7).unwrap(),
        SequenceDecoderOutput::StoppedWithText("�".to_string())
    );
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
