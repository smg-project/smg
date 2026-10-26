import json

import pytest
from smg_tokenizer import Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


@pytest.fixture
def checkpoint(tmp_path):
    hf = HFTokenizer(models.BPE(unk_token="[UNK]"))
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    hf.train_from_iterator(
        ["Hello 你好 café assistant user"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", "<s>", "</s>"],
        ),
    )
    hf.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "</s>"}))
    (tmp_path / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ '<s>' + m.role + ':' + m.content + '</s>' }}{% endfor %}"
        "{% if add_generation_prompt %}WRONG{% endif %}"
    )
    return Tokenizer(tmp_path), hf


def test_complete_conversation_preserves_ids_and_byte_offsets(checkpoint):
    smg, hf = checkpoint
    result = smg.render(
        [{"role": "user", "content": "你好 café"}, {"role": "assistant", "content": "Hello"}]
    )
    text = "<s>user:你好 café</s><s>assistant:Hello</s>"
    assert result["text"] == text
    assert result["input_ids"] == hf.encode(text, add_special_tokens=False).ids
    assert result["segmentwise"] is False
    assert len(result["offsets"]) == len(result["input_ids"])
    assert max(b for _, b in result["offsets"]) == len(text.encode())
    assert smg.encode_path in ("v1", "native", "tokenizers")


@pytest.mark.parametrize(
    "key", ["messages", "tools", "add_generation_prompt", "continue_final_message"]
)
def test_complete_conversation_options_are_reserved(checkpoint, key):
    smg, _ = checkpoint
    with pytest.raises(ValueError, match="complete-conversation"):
        smg.render([{"role": "user", "content": "Hello"}], template_kwargs={key: True})
