"""CPU regression checks for the vLLM request produced by the Rust Harmony stages.

    SMG_HARMONY_TEST_FIXTURE=/tmp/harmony.json cargo test -p smg --lib vllm_harmony_wire_constraints
    python model_gateway/tests/harmony_xgrammar.py /tmp/harmony.json /path/to/gpt-oss/tokenizer.json

Requires xgrammar, torch, openai-harmony and transformers. For XGrammar before
0.1.33, pass --vocabulary older. Supply the tokenizer.json from openai/gpt-oss-20b.
"""

import argparse
import json
from pathlib import Path

import openai_harmony as harmony
import xgrammar as xgr
from transformers import PreTrainedTokenizerFast

ANALYSIS = "<|channel|>analysis<|message|>Check the forecast.<|end|><|start|>assistant"
COMMENTARY = "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant"
FINAL = "<|channel|>final<|message|>"
WEATHER = "<|channel|>commentary to=functions.weather<|constrain|>json<|message|>"
PAYLOAD = '{"city":"Paris"}'
EOS = "<|return|>"


def accepts(grammar, suppressed, ids):
    matcher = xgr.GrammarMatcher(grammar)
    mask = xgr.allocate_token_bitmask(1, 201088)
    for token in ids:
        matcher.fill_next_token_bitmask(mask)
        if token in suppressed or not (int(mask[0, token // 32]) & (1 << (token % 32))):
            return False
        assert matcher.accept_token(token)
    return True


def verify(fixtures, tokenizer_path, vocabularies):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), eos_token=EOS)
    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    # vLLM 0.20's Hugging Face adapter passes no request stops to XGrammar.
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=201088))

    def encode(text, markers=True):
        return encoding.encode(
            text, allowed_special="all" if markers else set(), disallowed_special=()
        )

    for vocabulary in vocabularies:
        for choice, wire in fixtures[vocabulary].items():
            grammar = compiler.compile_structural_tag(wire["grammar"])
            suppressed = set(wire["suppressed"])
            for header, enabled in [(FINAL, choice != "required"), (WEATHER, choice != "json")]:
                ids = encode(ANALYSIS + COMMENTARY + header + PAYLOAD + EOS)
                assert accepts(grammar, suppressed, ids) == enabled, (vocabulary, choice, header)
                if not enabled:
                    continue
                parser = harmony.StreamableParser(encoding, harmony.Role.ASSISTANT)
                for token in ids:
                    parser.process(token)
                result = parser.messages[-1]
                assert result.recipient == ("functions.weather" if header == WEATHER else None)
                assert result.channel == ("commentary" if header == WEATHER else "final")
                assert json.loads(result.content[0].text) == {"city": "Paris"}

                # A legitimate call may omit the content type. Its schema still applies.
                assert accepts(
                    grammar,
                    suppressed,
                    encode(header.replace("<|constrain|>json", "") + PAYLOAD + EOS),
                )
                assert not accepts(grammar, suppressed, encode(header + '{"city":123}' + EOS))
                # A user can still put the literal spelling of a marker in valid JSON.
                literal_marker = encode(header) + encode('{"city":"<|call|>"}', markers=False)
                assert accepts(grammar, suppressed, literal_marker + encode(EOS))
                for prefix in ("<|channel|>analysis<|message|>Thinking", header + '{"city":"'):
                    for early_stop in ("<|call|>", "<|endoftext|>", EOS):
                        assert not accepts(grammar, suppressed, encode(prefix + early_stop))

                if vocabulary == "current":
                    spelled = encode(header, markers=False) + encode(PAYLOAD + EOS)
                    assert not accepts(grammar, suppressed, spelled)
                    for marker in ("<|channel|>", "<|message|>"):
                        before, after = header.split(marker)
                        mixed = encode(ANALYSIS + before) + encode(marker, markers=False)
                        assert not accepts(
                            grammar, suppressed, mixed + encode(after + PAYLOAD + EOS)
                        )
                # Older matchers can spell markers as ordinary text. The gateway's
                # byte_spelled_headers_fail_at_the_action_stop test exercises that parser guard.
            wrong_tool = WEATHER.replace("functions.weather", "functions.other")
            assert not accepts(grammar, suppressed, encode(wrong_tool + PAYLOAD + EOS))
            print(f"{vocabulary} {choice}: passed")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("fixture", type=Path)
    args.add_argument("tokenizer", type=Path)
    args.add_argument("--vocabulary", choices=("older", "current", "both"), default="both")
    parsed = args.parse_args()
    vocabularies = ["older", "current"] if parsed.vocabulary == "both" else [parsed.vocabulary]
    verify(json.loads(parsed.fixture.read_text()), parsed.tokenizer, vocabularies)
