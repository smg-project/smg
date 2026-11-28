use std::collections::HashSet;

use super::*;

fn written() -> (tempfile::TempDir, Vec<String>) {
    let dir = tempfile::tempdir().unwrap();
    let tokens = write(dir.path(), &["extra_word".to_string()]).unwrap();
    (dir, tokens)
}

#[test]
fn the_directory_holds_the_files_a_tokenizer_loader_reads() {
    let (dir, _) = written();
    let mut names: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert_eq!(
        names,
        [
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
    );
    let tok: Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("tokenizer.json")).unwrap())
            .unwrap();
    assert_eq!(tok["model"]["type"], "BPE");
    assert!(tok["model"]["unk_token"].is_null());
    assert_eq!(tok["normalizer"], json!({"type": "NFC"}));
    let kinds: Vec<&str> = tok["pre_tokenizer"]["pretokenizers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["type"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["Split", "ByteLevel"]);
    assert_eq!(
        tok["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"],
        false
    );
    assert_eq!(tok["decoder"]["type"], "ByteLevel");
    assert_eq!(tok["post_processor"]["type"], "ByteLevel");
    assert!(tok["truncation"].is_null() && tok["padding"].is_null());
    let added: HashSet<&str> = tok["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["content"].as_str().unwrap())
        .collect();
    assert_eq!(added, HashSet::from(["<|start|>", "<|end|>", "<|pad|>"]));
    assert!(tok["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .all(|t| t["special"] == true && t["single_word"] == false && t["lstrip"] == false));
    let cfg: Value = serde_json::from_str(
        &fs::read_to_string(dir.path().join("tokenizer_config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg["eos_token"], "<|end|>");
    assert_eq!(cfg["add_bos_token"], false);
    assert!(cfg["chat_template"].is_string());
    let model_cfg: Value =
        serde_json::from_str(&fs::read_to_string(dir.path().join("config.json")).unwrap()).unwrap();
    let end_id = tok["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["content"] == "<|end|>")
        .unwrap()["id"]
        .clone();
    assert_eq!(model_cfg["eos_token_id"], end_id);
}

#[test]
fn the_byte_alphabet_is_the_byte_level_one() {
    let alphabet = byte_alphabet();
    assert_eq!(alphabet.len(), 256);
    assert_eq!(alphabet.iter().collect::<HashSet<_>>().len(), 256);
    assert_eq!(alphabet[usize::from(b'a')], "a");
    assert_eq!(alphabet[usize::from(b'!')], "!");
    assert_eq!(alphabet[usize::from(b' ')], SPACE);
    assert_eq!(SPACE, "Ġ");
    assert_eq!(alphabet[usize::from(b'\n')], "Ċ");
}

#[test]
fn every_bed_word_is_one_token_after_a_space_and_bare() {
    let (_dir, tokens) = written();
    let vocab: HashSet<&String> = tokens.iter().collect();
    assert_eq!(vocab.len(), tokens.len(), "duplicate ids");
    for word in PROMPT_WORDS
        .split_whitespace()
        .chain(STRUCTURE_WORDS.split_whitespace())
        .chain(["extra_word", "Tools", "Lookup"])
    {
        assert!(vocab.contains(&word.to_string()), "{word}");
        assert!(vocab.contains(&format!("{SPACE}{word}")), "{word}");
    }
    for ch in "{}[]\":,.-_0123456789".chars() {
        assert!(vocab.contains(&ch.to_string()), "{ch}");
    }
    assert!(vocab.contains(&"Ċ".to_string()));
    assert_eq!(
        &tokens[tokens.len() - 3..],
        ["<|start|>", "<|end|>", "<|pad|>"]
    );
}

#[test]
fn merges_are_ranked_spaced_first_and_longest_first() {
    let alphabet = byte_alphabet();
    let words: Vec<String> = ["own", "brown", "the", "then"]
        .iter()
        .map(|w| (*w).to_string())
        .collect();
    let (vocab, merges) = merges_for(&words, &alphabet);
    assert_eq!(merges.iter().collect::<HashSet<_>>().len(), merges.len());
    let first_bare = merges
        .iter()
        .position(|(left, _)| !left.starts_with(SPACE))
        .unwrap();
    assert!(merges[..first_bare]
        .iter()
        .all(|(left, _)| left.starts_with(SPACE)));
    let index = |a: &str, b: &str| merges.iter().position(|m| m.0 == a && m.1 == b).unwrap();
    // brown's chain (longer) ranks before own's, so "brown" never breaks into "br" + "own"
    assert!(index(&format!("{SPACE}b"), "r") < index(&format!("{SPACE}o"), "w"));
    assert!(index("b", "r") < index("o", "w"));
    assert!(vocab.contains(&format!("{SPACE}brown")));
    assert!(vocab.contains(&"then".to_string()));
    assert!(vocab.contains(&format!("{SPACE}the")));
}

#[test]
fn the_vocabulary_is_deterministic_and_capitalize_is_pythons() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(&dir.path().join("a"), &[]).unwrap();
    let b = write(&dir.path().join("b"), &[]).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        fs::read(dir.path().join("a").join("tokenizer.json")).unwrap(),
        fs::read(dir.path().join("b").join("tokenizer.json")).unwrap()
    );
    assert_eq!(capitalize("additionalProperties"), "Additionalproperties");
    assert_eq!(capitalize("lookup_1"), "Lookup_1");
    assert_eq!(capitalize(""), "");
    let words = words_of(&["gamma".to_string()]);
    assert!(words.contains(&"Gamma".to_string()) && words.contains(&"gamma".to_string()));
}

#[test]
fn the_chat_template_renders_roles_tools_and_the_generation_prompt() {
    let mut env = minijinja::Environment::new();
    env.add_template("chat", CHAT_TEMPLATE).unwrap();
    let template = env.get_template("chat").unwrap();
    let messages = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the of and"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "call_1_0", "type": "function", "function": {"name": "lookup_0", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1_0", "content": "{\"b\":0}"},
        {"role": "user", "content": [{"type": "text", "text": "parts"}]},
    ]);
    let tools = json!([{"type": "function", "function": {"name": "lookup_0"}}]);
    let text = template
        .render(minijinja::context! { messages => messages, tools => tools, add_generation_prompt => true })
        .unwrap();
    assert!(
        text.starts_with("<|start|>system\nbe brief\nTools:\n{\"type\":\"function\""),
        "{text}"
    );
    assert!(text.contains("<|start|>tool\n{\"b\":0}<|end|>"), "{text}");
    assert!(
        text.contains("{\"name\":\"lookup_0\",\"arguments\":\"{}\"}"),
        "{text}"
    );
    assert!(
        text.ends_with("<|start|>user\nparts<|end|>\n<|start|>assistant\n"),
        "{text}"
    );
    assert_eq!(text.matches("<|start|>system").count(), 1);
    let two = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the of and"},
    ]);
    let plain = template
        .render(minijinja::context! { messages => two, add_generation_prompt => false })
        .unwrap();
    assert_eq!(
        plain,
        "<|start|>system\nbe brief<|end|>\n<|start|>user\nthe of and<|end|>\n"
    );
}
