use llm_tokenizer::{
    traits::{Decoder, Encoder, Encoding, SpecialTokens},
    MockTokenizer, TiktokenModel, TiktokenTokenizer,
};

use super::*;

/// A tokenizer that reports no vocabulary; every other answer is the mock's.
struct NoVocabulary(MockTokenizer);

impl Encoder for NoVocabulary {
    fn encode(&self, input: &str, add_special_tokens: bool) -> anyhow::Result<Encoding> {
        self.0.encode(input, add_special_tokens)
    }

    fn encode_batch(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> anyhow::Result<Vec<Encoding>> {
        self.0.encode_batch(inputs, add_special_tokens)
    }
}

impl Decoder for NoVocabulary {
    fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.0.decode(token_ids, skip_special_tokens)
    }
}

impl Tokenizer for NoVocabulary {
    fn vocab_size(&self) -> usize {
        0
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.0.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The mock's base vocabulary is its token count; its added tokens sit far above it.
fn tokenizer() -> MockTokenizer {
    MockTokenizer::new().with_tokens(&[("<|added|>", 50_000)])
}

#[test]
fn an_id_inside_the_base_vocabulary_passes_without_a_lookup() {
    let tokenizer = tokenizer();
    let inside: Vec<u32> = (0..tokenizer.vocab_size() as u32).collect();
    assert_eq!(
        reject_unknown_token_id(&tokenizer, &inside, "input_ids", || None),
        Ok(())
    );
}

#[test]
fn an_added_token_above_the_base_vocabulary_passes_by_lookup() {
    let tokenizer = tokenizer();
    assert!(50_000 >= tokenizer.vocab_size() as u32);
    assert_eq!(
        reject_unknown_token_id(&tokenizer, &[1, 50_000, 2], "input_ids", || None),
        Ok(())
    );
}

#[test]
fn the_first_unknown_id_is_named_with_its_position_and_the_bound() {
    let tokenizer = tokenizer();
    let base = tokenizer.vocab_size();
    let unknown = base as u32 + 7;
    let error = reject_unknown_token_id(
        &tokenizer,
        &[1, 2, unknown, unknown + 1],
        "input_ids",
        || None,
    )
    .expect_err("an unknown id is refused");
    assert!(
        error.starts_with(&format!("input_ids[2] = {unknown} ")),
        "{error}"
    );
    assert!(error.contains(&format!("holds {base} tokens")), "{error}");
}

#[test]
fn an_id_the_tokenizer_does_not_know_passes_inside_the_reported_table() {
    let tokenizer = tokenizer();
    let gap = tokenizer.vocab_size() as u32 + 7;
    assert_eq!(
        reject_unknown_token_id(&tokenizer, &[1, gap, 50_000], "input_ids", || Some(2_000)),
        Ok(())
    );
}

#[test]
fn an_id_past_the_reported_table_is_refused_naming_the_table() {
    let tokenizer = tokenizer();
    let error =
        reject_unknown_token_id(&tokenizer, &[1, 1_999, 2_000], "input_ids", || Some(2_000))
            .expect_err("an id past the table is refused");
    assert!(error.starts_with("input_ids[2] = 2000 "), "{error}");
    assert!(error.contains("embedding table holds 2000 rows"), "{error}");
}

#[test]
fn the_table_is_not_read_while_the_tokenizer_knows_every_id() {
    let tokenizer = tokenizer();
    let inside: Vec<u32> = (0..tokenizer.vocab_size() as u32).chain([50_000]).collect();
    let result = reject_unknown_token_id(&tokenizer, &inside, "input_ids", || {
        panic!("the table is read only after an unknown id")
    });
    assert_eq!(result, Ok(()));
}

#[test]
fn a_tokenizer_without_a_vocabulary_is_bounded_by_the_reported_table() {
    let tokenizer = NoVocabulary(MockTokenizer::new());
    assert_eq!(
        reject_unknown_token_id(&tokenizer, &[5, 1_999], "input_ids", || Some(2_000)),
        Ok(())
    );
    let error = reject_unknown_token_id(&tokenizer, &[5, 2_000], "input_ids", || Some(2_000))
        .expect_err("past the table");
    assert!(error.starts_with("input_ids[1] = 2000 "), "{error}");
}

#[test]
fn a_tokenizer_without_a_vocabulary_skips_the_bound() {
    let tokenizer = NoVocabulary(MockTokenizer::new());
    assert_eq!(
        reject_unknown_token_id(&tokenizer, &[u32::MAX], "input_ids", || None),
        Ok(())
    );
}

/// A prompt the tokenizer knows throughout decodes as given, its added token with it.
#[test]
fn the_decodable_ids_of_a_known_prompt_are_the_prompt() {
    let tokenizer = tokenizer();
    assert_eq!(
        decodable_ids(&tokenizer, &[1, 2, 50_000]),
        vec![1, 2, 50_000]
    );
}

/// An id known only to the engine's table leaves the text, the rest keeps its order.
#[test]
fn an_id_known_only_to_the_table_leaves_the_decodable_ids() {
    let tokenizer = tokenizer();
    let gap = tokenizer.vocab_size() as u32 + 7;
    assert_eq!(
        decodable_ids(&tokenizer, &[1, gap, 2, 50_000, gap]),
        vec![1, 2, 50_000]
    );
}

/// The tiktoken decoder refuses an id it lacks; the decodable ids read the text around it.
#[test]
fn a_tiktoken_decoder_reads_the_decodable_ids_of_a_padded_table_prompt() {
    let tokenizer = TiktokenTokenizer::new(TiktokenModel::Cl100kBase).expect("built-in");
    let mut ids = tokenizer
        .encode("Hello world", false)
        .expect("encode")
        .into_token_ids();
    // Past the encoding's special tokens, which sit just above its base vocabulary.
    ids.push(tokenizer.vocab_size() as u32 + 500);

    assert!(
        tokenizer.decode(&ids, false).is_err(),
        "the gap id refuses the decode"
    );
    assert_eq!(
        tokenizer
            .decode(&decodable_ids(&tokenizer, &ids), false)
            .expect("known ids"),
        "Hello world"
    );
}

/// The refusal names the field it was given, so a list of a batch reads as its own.
#[test]
fn the_refusal_names_the_field_it_was_given() {
    let tokenizer = tokenizer();
    let unknown = tokenizer.vocab_size() as u32 + 7;
    let error = reject_unknown_token_id(&tokenizer, &[1, 2, unknown], "prompt[1]", || None)
        .expect_err("an unknown id is refused");
    assert!(
        error.starts_with(&format!("prompt[1][2] = {unknown} ")),
        "{error}"
    );
}

/// A `logit_bias` key at or beyond the vocabulary is refused before any
/// engine wire; the last token id, an empty map and no map pass.
#[test]
fn logit_bias_keys_are_judged_as_a_pretokenized_prompts_ids_are() {
    let tokenizer = tokenizer();
    let base = tokenizer.vocab_size() as u32;
    let bias = |id: u32| Some(HashMap::from([(id.to_string(), -100.0f32)]));
    let no_table = || -> Option<usize> { panic!("the table is read only after an unknown id") };
    // A key inside the base vocabulary, or an added token above it, passes without a table read.
    assert!(validate_logit_bias_token_ids(bias(base - 1).as_ref(), &tokenizer, no_table).is_ok());
    assert!(validate_logit_bias_token_ids(bias(50_000).as_ref(), &tokenizer, no_table).is_ok());
    // A key the tokenizer does not know is refused when no worker reports a table...
    let unknown = base + 7;
    let error =
        validate_logit_bias_token_ids(bias(unknown).as_ref(), &tokenizer, || None).unwrap_err();
    assert!(
        error.starts_with(&format!("Invalid key in 'logit_bias': '{unknown}' ")),
        "{error}"
    );
    assert!(error.contains(&format!("holds {base} tokens")), "{error}");
    // ...passes inside the table the workers report, and is refused past it naming the table.
    assert!(
        validate_logit_bias_token_ids(bias(unknown).as_ref(), &tokenizer, || Some(2_000)).is_ok()
    );
    let error = validate_logit_bias_token_ids(bias(2_000).as_ref(), &tokenizer, || Some(2_000))
        .unwrap_err();
    assert!(error.contains("embedding table holds 2000 rows"), "{error}");
    // Nothing to judge without a map.
    assert!(validate_logit_bias_token_ids(Some(&HashMap::new()), &tokenizer, no_table).is_ok());
    assert!(validate_logit_bias_token_ids(None, &tokenizer, no_table).is_ok());
}
