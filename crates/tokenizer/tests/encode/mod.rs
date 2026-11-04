//! Encode and decode integration tests: the encode paths, special tokens, vocabularies and their
//! parity with the reference encoders.

mod configured_special_tokens;
mod encode_path_choice;
mod one_shot_decode_parity;
mod qwen2_vocab_merges;
mod qwen3_asr_bpe_parity;
mod special_tokens_from_config;
mod tiktoken_integration;
mod tokenizer_integration;
mod vllm_encode_parity;
mod vllm_special_tokens_request;
mod vocab_merges_parity;
