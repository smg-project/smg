//! The vocabulary bound of a pre-tokenized prompt: an id that can reach no row of the
//! engine's embedding table is refused at the gateway, on `/generate`'s `input_ids` and on
//! `/v1/completions`' prompt of ids alike, before any worker is spoken to.

use llm_tokenizer::traits::Tokenizer;

use crate::worker::WorkerRegistry;

/// The worker label that carries the engine's embedding-table size: `vocab_size` of the
/// engine's model info, kept on every worker that reports it.
const VOCAB_SIZE_LABEL: &str = "vocab_size";

/// The embedding-table size the model's workers report, the smallest among those that report
/// one (an id valid on a larger table is not on a smaller); `None` when no worker reports it.
/// Read only after an id the tokenizer does not know, so a valid prompt never pays for it.
pub(crate) fn reported_embedding_rows(
    worker_registry: &WorkerRegistry,
    model_id: &str,
) -> Option<usize> {
    worker_registry
        .get_by_model(model_id)
        .iter()
        .filter_map(|worker| {
            worker
                .metadata()
                .spec
                .labels
                .get(VOCAB_SIZE_LABEL)
                .and_then(|rows| rows.parse::<usize>().ok())
        })
        .min()
}

/// Whether the tokenizer knows `id`: by range below its base vocabulary (one comparison, the
/// common id), by lookup above it (an added token; the special tokens live there).
fn known_to(tokenizer: &dyn Tokenizer, base_vocab: usize, id: u32) -> bool {
    usize::try_from(id).is_ok_and(|id| id < base_vocab) || tokenizer.id_to_token(id).is_some()
}

/// Refuses the first id that can reach no row of the engine's embedding table, naming its
/// position, its value and the bound it missed.
///
/// An id below the tokenizer's base vocabulary is known by its range alone; one above it is an
/// added token (the special tokens live there) and is resolved by lookup, so the common id costs
/// one comparison and only the rare one a map read. The engines index their embeddings by these
/// ids and the headless lanes run none of the checks an engine's own HTTP front end would, so an
/// id past the table must not leave the gateway: on the engine it is an out-of-range index, not a
/// refused request. The table is padded past the tokenizer's last id on most models and an
/// engine samples over the padded logits, so an id the tokenizer does not know is judged by the
/// table size the model's workers report (`embedding_rows`, read once, only then): below it the
/// id passes, as it would on the engine. When no worker reports a table the tokenizer's ids are
/// the bound, the one case where the gateway is stricter than the engine; a tokenizer that
/// reports no vocabulary either skips the bound rather than refusing every pretokenized prompt.
/// `field` names the request field in the refusal: `input_ids`, `prompt`, or `prompt[b]` for
/// one list of a batch.
pub(crate) fn reject_unknown_token_id(
    tokenizer: &dyn Tokenizer,
    ids: &[u32],
    field: &str,
    embedding_rows: impl FnOnce() -> Option<usize>,
) -> Result<(), String> {
    let base_vocab = tokenizer.vocab_size();
    let known = |id: u32| known_to(tokenizer, base_vocab, id);
    let first_unknown = if base_vocab == 0 {
        // Without a vocabulary every id would be a lookup: the table alone judges them.
        0
    } else {
        match ids.iter().position(|&id| !known(id)) {
            None => return Ok(()),
            Some(index) => index,
        }
    };
    match embedding_rows() {
        Some(rows) => {
            let inside = |id: u32| usize::try_from(id).is_ok_and(|id| id < rows);
            match ids[first_unknown..]
                .iter()
                .position(|&id| !inside(id) && !known(id))
            {
                None => Ok(()),
                Some(offset) => Err(format!(
                    "{field}[{}] = {} is not a token of this model: its embedding table holds \
                     {rows} rows",
                    first_unknown + offset,
                    ids[first_unknown + offset]
                )),
            }
        }
        None if base_vocab == 0 => Ok(()),
        None => Err(format!(
            "{field}[{first_unknown}] = {} is not a token of this model: its vocabulary holds \
             {base_vocab} tokens plus its added tokens",
            ids[first_unknown]
        )),
    }
}

/// The ids of a prompt the bound has passed that the tokenizer can decode, copied for the
/// prompt's text: all of them on a prompt the tokenizer knows throughout (the common one, one
/// comparison per id as in the bound), without the ids known only to the engine's embedding
/// table otherwise. Such an id sits in the padding past the tokenizer's last id, and no decoder
/// has text for it: the byte-level and HF decoders drop it on their own, the tiktoken decoder
/// refuses the whole decode. Filtered here, the text is the same on every decoder and the
/// prompt rides with its ids as given.
pub(crate) fn decodable_ids(tokenizer: &dyn Tokenizer, ids: &[u32]) -> Vec<u32> {
    let base_vocab = tokenizer.vocab_size();
    let known = |id: u32| known_to(tokenizer, base_vocab, id);
    match ids.iter().position(|&id| !known(id)) {
        None => ids.to_vec(),
        Some(first) => {
            let mut kept = Vec::with_capacity(ids.len() - 1);
            kept.extend_from_slice(&ids[..first]);
            kept.extend(ids[first + 1..].iter().copied().filter(|&id| known(id)));
            kept
        }
    }
}

#[cfg(test)]
mod tests {
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
}
