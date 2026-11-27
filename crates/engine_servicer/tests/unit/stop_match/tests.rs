use super::*;

fn byte_fallback_tokenizer() -> Arc<dyn Tokenizer> {
    let definition = serde_json::json!({
        "version":"1.0", "added_tokens":[],
        "model":{"type":"WordLevel", "unk_token":"[UNK]", "vocab":{
            "[UNK]":0, "<0x74>":1, "<0xC3>":2, "<0xA9>":3,
            "<0xF2>":4, " suffix":5
        }},
        "decoder":{"type":"Sequence", "decoders":[
            {"type":"ByteFallback"}, {"type":"Fuse"}
        ]}
    });
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), serde_json::to_vec(&definition).unwrap()).unwrap();
    Arc::new(llm_tokenizer::HuggingFaceTokenizer::from_file(file.path().to_str().unwrap()).unwrap())
}

#[test]
fn byte_fallback_stop_observation_honors_min_tokens() {
    for cached in [false, true] {
        let native = byte_fallback_tokenizer();
        let tokenizer: Arc<dyn Tokenizer> = if cached {
            Arc::new(llm_tokenizer::CachedTokenizer::new(
                native,
                llm_tokenizer::CacheConfig::default(),
            ))
        } else {
            native
        };
        // "t" ends inside min_tokens. Closing the byte run later must
        // not make that old stop eligible.
        let decoder = stop_decoder(Some(&tokenizer), &["t".into()], true).unwrap();
        let mut matcher = StopMatcher::new(decoder, 1);
        assert_eq!(
            matcher.feed(&[1, 2, 3, 5]).unwrap(),
            None,
            "cached={cached}"
        );

        // A stop beginning inside the window but ending past it still
        // matches; the incomplete UTF-8 character crosses the boundary.
        let decoder = stop_decoder(Some(&tokenizer), &["té".into()], true).unwrap();
        let mut matcher = StopMatcher::new(decoder, 1);
        assert_eq!(matcher.feed(&[1, 2]).unwrap(), None);
        assert_eq!(matcher.feed(&[3]).unwrap(), Some("té".into()));
    }
}

#[test]
fn byte_fallback_invalid_byte_after_a_character_does_not_fail_stop_observation() {
    // <0xC3> <0xA9> <0xF2> " suffix": "é" is observed before the invalid
    // byte makes batch decoding rewrite the run; observation goes on with
    // the rest of the run instead of failing, for matching and
    // non-matching stops alike.
    for cached in [false, true] {
        let native = byte_fallback_tokenizer();
        let tokenizer: Arc<dyn Tokenizer> = if cached {
            Arc::new(llm_tokenizer::CachedTokenizer::new(
                native,
                llm_tokenizer::CacheConfig::default(),
            ))
        } else {
            native
        };
        for (stop, expected) in [
            ("t", None),
            ("é", Some("é".to_string())),
            ("suffix", Some("suffix".to_string())),
            ("\u{FFFD} suf", Some("\u{FFFD} suf".to_string())),
        ] {
            let decoder = stop_decoder(Some(&tokenizer), &[stop.into()], true).unwrap();
            let mut matcher = StopMatcher::new(decoder, 0);
            let mut matched = None;
            for &id in &[2, 3, 4, 5] {
                matched = matcher.feed(&[id]).unwrap();
                if matched.is_some() {
                    break;
                }
            }
            assert_eq!(matched, expected, "stop {stop:?} cached={cached}");
        }
    }
}
