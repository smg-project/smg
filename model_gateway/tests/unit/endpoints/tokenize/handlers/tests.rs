use super::*;

#[test]
fn test_get_tokenizer_empty_registry() {
    let workers = WorkerRegistry::new();
    let registry = Arc::new(TokenizerRegistry::new());
    let result = get_tokenizer(&workers, &registry, "any");
    match result {
        Err(e) => assert!(e.contains("No tokenizers available")),
        Ok(_) => panic!("Expected error"),
    }
}
