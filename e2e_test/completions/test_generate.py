"""The native generate API (``/generate``) with a pre-tokenized prompt.

A request that carries ``input_ids`` instead of ``text`` bypasses the gateway's
tokenizer: the ids reach the engine as they were sent, the engine counts them
as the prompt, and the answer opens as the same prompt's answer as text. The
ids come from the gateway's own ``/v1/tokenize`` (no special tokens, as the
text path encodes), so the two requests describe one prompt. On a real engine
the two answers are compared on their first token only: the second request may
run over the engine's prefix cache, a different kernel path whose logits are
not promised identical, so a near-tie further into a greedy run could flip.

On the mock engine the ids request's answer is filed under the exact ids
(``input_ids=``): the mock serves it only when the prompt it receives has that
key, so a served answer proves the engine saw the ids unchanged, and the
scripted mock reports the prompt count it received.
"""

from __future__ import annotations

import json

import httpx
import pytest

TIMEOUT = 60.0
_TEXT = "The capital of France is"
_ANSWER = " Paris."


def _generate(gateway, model: str, prompt: dict, *, stream: bool = False) -> httpx.Response:
    body = {
        "model": model,
        **prompt,
        "sampling_params": {"max_new_tokens": 8, "temperature": 0},
        "stream": stream,
    }
    return httpx.post(f"{gateway.base_url}/generate", json=body, timeout=TIMEOUT)


def _tokenize(gateway, model: str, text: str) -> list[int]:
    resp = httpx.post(
        f"{gateway.base_url}/v1/tokenize", json={"model": model, "prompt": text}, timeout=TIMEOUT
    )
    assert resp.status_code == 200, resp.text
    ids = resp.json()["tokens"]
    assert ids and all(isinstance(i, int) for i in ids), ids
    return ids


def _events(resp: httpx.Response) -> list[dict]:
    lines = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    assert lines and lines[-1] == "data: [DONE]", resp.text[:500]
    return [json.loads(line[len("data: ") :]) for line in lines[:-1]]


@pytest.mark.engine("sglang", "vllm", "tokenspeed", "mock")
@pytest.mark.gpu(1)
@pytest.mark.model(
    "meta-llama/Llama-3.1-8B-Instruct", mock="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"
)
@pytest.mark.parametrize("setup_backend", ["grpc"], indirect=True)
class TestGeneratePretokenized:
    """``/generate`` with ``input_ids``: the tokenizer bypass, end to end."""

    def test_input_ids_reach_the_engine_as_sent_and_answer_like_the_text(
        self, setup_backend, mock_answers
    ):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)

        mock_answers.enqueue(_ANSWER)
        from_text = _generate(gateway, model, {"text": _TEXT})
        assert from_text.status_code == 200, from_text.text
        from_text = from_text.json()
        assert isinstance(from_text, dict), from_text

        # Filed under the exact ids: the mock serves it only to a prompt with that key.
        mock_answers.enqueue(_ANSWER, input_ids=ids)
        from_ids = _generate(gateway, model, {"input_ids": ids})
        assert from_ids.status_code == 200, from_ids.text
        from_ids = from_ids.json()
        assert isinstance(from_ids, dict), from_ids

        assert from_ids["meta_info"]["prompt_tokens"] == len(ids), from_ids["meta_info"]
        assert from_text["meta_info"]["prompt_tokens"] == len(ids), from_text["meta_info"]
        assert from_ids["text"], from_ids
        if mock_answers.active:
            # The keyed answer is exact: the whole answer matches, and both were served.
            assert from_ids["output_ids"] == from_text["output_ids"], (from_ids, from_text)
            assert from_ids["text"] == from_text["text"], (from_ids, from_text)
            assert mock_answers.served() == 2
        else:
            # The confident first token; the rest of a greedy run may differ between
            # the full prefill of the text request and the prefix-cached ids request.
            assert from_ids["output_ids"][:1] == from_text["output_ids"][:1], (
                from_ids,
                from_text,
            )

    def test_streamed_input_ids_report_the_prompt_count_on_every_event(
        self, setup_backend, mock_answers
    ):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)

        mock_answers.enqueue(_ANSWER, input_ids=ids)
        resp = _generate(gateway, model, {"input_ids": ids}, stream=True)
        assert resp.status_code == 200, resp.text

        events = _events(resp)
        assert events, resp.text[:500]
        for event in events:
            assert event["meta_info"]["prompt_tokens"] == len(ids), event
        assert "".join(e["text"] for e in events), events

    def test_a_batch_of_id_lists_is_refused(self, setup_backend, mock_answers):
        _backend, model, _client, gateway = setup_backend
        ids = _tokenize(gateway, model, _TEXT)

        resp = _generate(gateway, model, {"input_ids": [ids, ids]})

        assert resp.status_code == 400, resp.text
        assert resp.json()["error"]["code"] == "resolve_input_failed", resp.text
