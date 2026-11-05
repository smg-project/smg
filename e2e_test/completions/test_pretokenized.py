"""``/v1/completions`` with a prompt of token ids, end to end.

The OpenAI completions API takes ``prompt`` as a string, an array of strings,
an array of token ids, or an array of token-id arrays. A prompt of ids bypasses
the gateway's tokenizer: the ids reach the engine as they were sent, the engine
counts them as the prompt, and the answer opens as the same prompt's answer as
text. The ids come from the gateway's own ``/v1/tokenize`` (no special tokens,
as the text path encodes), so the two requests describe one prompt. On a real
engine the two answers are compared on their first word only: the second
request may run over the engine's prefix cache, a different kernel path whose
logits are not promised identical, so a near-tie further into a greedy run
could flip.

On the mock engine the ids request's answer is filed under the exact ids
(``input_ids=``): the mock serves it only when the prompt it receives has that
key, so a served answer proves the engine saw the ids unchanged, and the
scripted mock reports the prompt count it received.
"""

from __future__ import annotations

import httpx
import pytest

TIMEOUT = 60.0
_TEXT = "The capital of France is"
_OTHER_TEXT = "The capital of Italy is"
_ANSWER = " Paris."
_OTHER_ANSWER = " Rome."


def _tokenize(gateway, model: str, text: str) -> list[int]:
    resp = httpx.post(
        f"{gateway.base_url}/v1/tokenize", json={"model": model, "prompt": text}, timeout=TIMEOUT
    )
    assert resp.status_code == 200, resp.text
    ids = resp.json()["tokens"]
    assert ids and all(isinstance(i, int) for i in ids), ids
    return ids


def _first_word(text: str) -> str:
    words = text.split()
    assert words, repr(text)
    return words[0]


@pytest.mark.engine("sglang", "vllm", "tokenspeed", "mock")
@pytest.mark.gpu(1)
@pytest.mark.model(
    "meta-llama/Llama-3.1-8B-Instruct", mock="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestCompletionPretokenized:
    """``/v1/completions`` with a prompt of token ids: the tokenizer bypass, end to end."""

    def test_a_token_prompt_reaches_the_engine_as_sent_and_answers_like_the_text(
        self, setup_backend, api_client, mock_answers
    ):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)

        mock_answers.enqueue(_ANSWER)
        from_text = api_client.completions.create(
            model=model, prompt=_TEXT, max_tokens=8, temperature=0
        )

        # Filed under the exact ids: the mock serves it only to a prompt with that key.
        mock_answers.enqueue(_ANSWER, input_ids=ids)
        from_ids = api_client.completions.create(
            model=model, prompt=ids, max_tokens=8, temperature=0
        )

        assert from_ids.usage.prompt_tokens == len(ids), from_ids.usage
        assert from_text.usage.prompt_tokens == len(ids), from_text.usage
        assert len(from_ids.choices) == 1, from_ids
        assert from_ids.choices[0].text, from_ids
        if mock_answers.active:
            # The keyed answer is exact: the whole answer matches, and both were served.
            assert from_ids.choices[0].text == from_text.choices[0].text, (from_ids, from_text)
            assert mock_answers.served() == 2
        else:
            # The confident first word; the rest of a greedy run may differ between
            # the full prefill of the text request and the prefix-cached ids request.
            assert _first_word(from_ids.choices[0].text) == _first_word(
                from_text.choices[0].text
            ), (from_ids, from_text)

    def test_a_batch_of_token_prompts_answers_one_choice_per_prompt(
        self, setup_backend, api_client, mock_answers
    ):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)
        other = _tokenize(gateway, model, _OTHER_TEXT)

        mock_answers.enqueue(_ANSWER, input_ids=ids)
        mock_answers.enqueue(_OTHER_ANSWER, input_ids=other)
        response = api_client.completions.create(
            model=model, prompt=[ids, other], max_tokens=8, temperature=0
        )

        assert [choice.index for choice in response.choices] == [0, 1], response
        assert response.usage.prompt_tokens == len(ids) + len(other), response.usage
        assert all(choice.text for choice in response.choices), response
        if mock_answers.active:
            # Each prompt's answer is keyed by its own ids, so the order proves the pairing.
            assert [choice.text for choice in response.choices] == [_ANSWER, _OTHER_ANSWER], (
                response
            )
            assert mock_answers.served() == 2

    def test_an_echoed_token_prompt_opens_with_its_text(
        self, setup_backend, api_client, mock_answers
    ):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)

        mock_answers.enqueue(_ANSWER, input_ids=ids)
        response = api_client.completions.create(
            model=model, prompt=ids, max_tokens=8, temperature=0, echo=True
        )

        text = response.choices[0].text
        assert text.startswith(_TEXT), text
        assert len(text) > len(_TEXT), text
        assert response.usage.prompt_tokens == len(ids), response.usage

    def test_a_streamed_token_prompt_reports_the_prompt_count(
        self, setup_backend, api_client, mock_answers
    ):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)

        mock_answers.enqueue(_ANSWER, input_ids=ids)
        stream = api_client.completions.create(
            model=model,
            prompt=ids,
            max_tokens=8,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )

        texts: list[str] = []
        usage = None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            texts.extend(choice.text for choice in chunk.choices)

        assert "".join(texts), texts
        assert usage is not None, "the stream states its usage"
        assert usage.prompt_tokens == len(ids), usage
