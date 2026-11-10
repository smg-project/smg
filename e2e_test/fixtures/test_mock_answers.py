"""CPU-only checks for the answers a test states with a logprob per token."""

import importlib
import re
from types import SimpleNamespace

import httpx
import pytest
from fixtures.mock_answers import (
    MockAnswers,
    class_workers,
    clear_class_workers,
    set_class_workers,
)
from infra.constants import WorkerType
from infra.worker import Worker
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import WordLevel

# The module, not the fixture function fixtures.__init__ exports under the same name.
answers_module = importlib.import_module("fixtures.mock_answers")


def _save_model(path, tokenizer):
    tokenizer.save(str(path / "tokenizer.json"))
    return str(path)


@pytest.fixture
def model_dir(tmp_path):
    """A model directory whose byte-level ``tokenizer.json`` knows a few words:
    the shape of the models' tokenizers, every id decoding to its own piece."""
    vocab = {
        "[UNK]": 0,
        "the": 1,
        "Ġcapital": 2,
        "Ġof": 3,
        "Ġfrance": 4,
        "Ġis": 5,
        "Ġparis": 6,
        ".": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return _save_model(tmp_path, tokenizer)


@pytest.fixture
def word_model_dir(tmp_path):
    """A model directory whose ``tokenizer.json`` splits on whitespace, the
    spaces belonging to no token."""
    vocab = {"[UNK]": 0, "the": 1, "capital": 2, "of": 3, "france": 4, "is": 5, "paris": 6, ".": 7}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return _save_model(tmp_path, tokenizer)


def test_an_answer_with_logprobs_states_its_pieces_and_ids_with_one_logprob_each(
    monkeypatch, model_dir
):
    posted = []

    def post(url, json, timeout):
        posted.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": model_dir})
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(workers=[worker])

    answers.enqueue_with_logprobs("the capital of france is paris.", logprob=-0.25, delay_ms=5)

    assert posted == [
        (
            f"{worker.admin_url}/admin/script/all",
            {
                "output_text": ["the", " capital", " of", " france", " is", " paris", "."],
                "output_ids": [1, 2, 3, 4, 5, 6, 7],
                "logprobs": [-0.25] * 7,
                "delay_ms": 5,
            },
        )
    ]
    assert answers.enqueued == 1


def test_the_pieces_carry_the_text_between_the_tokens(monkeypatch, word_model_dir):
    posted = []

    def post(url, json, timeout):
        posted.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(
        answers_module, "get_model_spec", lambda model_id: {"model": word_model_dir}
    )
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(workers=[worker])

    answers.enqueue_with_logprobs("the capital of france is paris.")

    assert posted[0]["output_text"] == ["the ", "capital ", "of ", "france ", "is ", "paris", "."]
    assert posted[0]["output_ids"] == [1, 2, 3, 4, 5, 6, 7]
    assert "".join(posted[0]["output_text"]) == "the capital of france is paris."


def test_an_answer_whose_token_offsets_do_not_partition_the_text_is_refused(monkeypatch):
    class Backwards:
        """A tokenizer whose third token starts before its second."""

        def encode(self, text, add_special_tokens):
            return SimpleNamespace(ids=[1, 2, 3], offsets=[(0, 3), (5, 6), (3, 5)])

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("nothing to post"))
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": "unused"})
    monkeypatch.setattr(answers_module, "_tokenizer", lambda model_dir: Backwards())
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(workers=[worker])

    with pytest.raises(ValueError, match="do not partition the text; state this answer as pieces"):
        answers.enqueue_with_logprobs("Paris.")

    assert answers.enqueued == 0


def test_an_answer_without_tokens_is_refused(monkeypatch, model_dir):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("nothing to post"))
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": model_dir})
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(workers=[worker])

    with pytest.raises(ValueError, match="'': no tokens to state a logprob for"):
        answers.enqueue_with_logprobs("")

    assert answers.enqueued == 0


def test_a_model_directory_without_a_tokenizer_file_is_named(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("nothing to post"))
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": str(tmp_path)})
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(workers=[worker])

    with pytest.raises(FileNotFoundError, match="tokenizer.json: no tokenizer file"):
        answers.enqueue_with_logprobs("Paris.")

    assert answers.enqueued == 0


def test_an_answer_with_logprobs_is_a_no_op_without_a_mock_worker(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("no worker, no post"))
    answers = MockAnswers()

    answers.enqueue_with_logprobs("Paris.")

    assert answers.enqueued == 0


def test_a_scoring_answer_states_its_token_id_logprobs_on_no_tokens(monkeypatch):
    posted = []

    def post(url, json, timeout):
        posted.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", _nothing_served)
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(workers=[worker])
    answers.enqueue_scores([-0.2, -1.6])
    assert posted == [{"output_text": "", "token_id_logprobs": [-0.2, -1.6]}]
    assert answers.enqueued == 1
    # No worker, no post: a real engine scores on its own.
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("no worker, no post"))
    assert MockAnswers().enqueue_scores([-0.2]) is None


def test_a_scoring_answer_on_a_harmony_model_rides_on_no_channel(monkeypatch):
    # An answer of no tokens has no channel: the posted text stays empty where
    # a text answer of this class would ride on the final channel.
    posted = []

    def post(url, json, timeout):
        posted.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", _nothing_served)
    answers = MockAnswers(workers=[_harmony_worker()])
    assert answers.harmony
    answers.enqueue_scores([-0.2, -1.6])
    assert posted == [{"output_text": "", "token_id_logprobs": [-0.2, -1.6]}]


def _nothing_served(url, timeout):
    """A worker that served nothing yet (the reconcile before a filing reads the hits)."""
    return httpx.Response(200, json={"hits": 0}, request=httpx.Request("GET", url))


def _harmony_worker():
    return Worker(
        model_id="openai/gpt-oss-20b", engine="mock", port=100, gpu_ids=[], admin_port=200
    )


def test_a_plain_answer_rides_on_the_final_channel_of_a_harmony_model(monkeypatch):
    posted = []

    def post(url, json, timeout):
        posted.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", _nothing_served)
    answers = MockAnswers(workers=[_harmony_worker()])
    assert answers.harmony

    answers.enqueue("Paris.")
    answers.enqueue(["Pa", "ris."])
    # An answer that states its channels is left as it is.
    answers.enqueue("<|channel|>final<|message|>Rome.<|return|>")
    # An answer cut short (a matched stop sequence, the token limit) leaves
    # the channel open, as an engine does.
    answers.enqueue("1,", finish_reason="stop", matched_stop=",")
    answers.enqueue(["1, 2,", " 3"], finish_reason="length")
    assert [a["output_text"] for a in posted] == [
        "<|channel|>final<|message|>Paris.<|return|>",
        ["<|channel|>final<|message|>", "Pa", "ris.", "<|return|>"],
        "<|channel|>final<|message|>Rome.<|return|>",
        "<|channel|>final<|message|>1,",
        ["<|channel|>final<|message|>", "1, 2,", " 3"],
    ]
    assert answers.enqueued == 5

    # A model that is not a Harmony one takes the text as stated.
    plain = MockAnswers(
        workers=[Worker(model_id="twin", engine="mock", port=101, gpu_ids=[], admin_port=201)]
    )
    assert not plain.harmony
    plain.enqueue("Paris.")
    assert posted[-1]["output_text"] == "Paris."


def test_a_logprob_answer_cut_short_leaves_the_harmony_channel_open(monkeypatch):
    posted = []

    def post(url, json, timeout):
        posted.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    def pieces_by_marker(model, text):
        # The channel markers as whole pieces, the text between them as one
        # piece each: the shape the model's tokenizer gives, without its files.
        pieces = [p for p in re.split(r"(<\|[a-z_]+\|>)", text) if p]
        return list(range(len(pieces))), pieces

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", _nothing_served)
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": "unused"})
    monkeypatch.setattr(answers_module, "_pieces", pieces_by_marker)
    answers = MockAnswers(workers=[_harmony_worker()])

    answers.enqueue_with_logprobs("1, 2, 3.")
    answers.enqueue_with_logprobs("1,", finish_reason="stop", matched_stop=",")
    answers.enqueue_with_logprobs("1, 2,", finish_reason="length")
    assert ["".join(a["output_text"]) for a in posted] == [
        "<|channel|>final<|message|>1, 2, 3.<|return|>",
        "<|channel|>final<|message|>1,",
        "<|channel|>final<|message|>1, 2,",
    ]
    assert all(len(a["logprobs"]) == len(a["output_ids"]) == len(a["output_text"]) for a in posted)
    assert answers.enqueued == 3


def test_the_harmony_channels_are_stated_as_the_model_writes_them(monkeypatch):
    posted = []

    def post(url, json, timeout):
        posted.append(json)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", _nothing_served)
    answers = MockAnswers(workers=[_harmony_worker()])

    answers.enqueue_harmony(analysis="Think.", final="144")
    answers.enqueue_harmony(tool_call=("get_weather", {"location": "Seattle"}))
    answers.enqueue_harmony(analysis="A call.", tool_call=("calculate", '{"expression": "1+1"}'))
    assert [a["output_text"] for a in posted] == [
        "<|channel|>analysis<|message|>Think.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>144<|return|>",
        '<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location": "Seattle"}<|call|>',
        "<|channel|>analysis<|message|>A call.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.calculate <|constrain|>json"
        '<|message|>{"expression": "1+1"}<|call|>',
    ]
    with pytest.raises(ValueError, match="state an analysis"):
        answers.enqueue_harmony()
    with pytest.raises(ValueError, match="not both"):
        answers.enqueue_harmony(tool_call=("calculate", {}), final="4")


def _posting(monkeypatch):
    """Record every answer posted, by admin URL; answer each post with 200."""
    posted = []

    def post(url, json, timeout):
        posted.append((url, json))
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    return posted


def _decodes(count):
    return [
        Worker(
            model_id="twin",
            engine="mock",
            port=100 + i,
            gpu_ids=[],
            worker_type=WorkerType.DECODE,
            admin_port=200 + i,
        )
        for i in range(count)
    ]


def test_an_answer_for_a_prompt_is_filed_on_every_worker_of_the_class(monkeypatch):
    posted = _posting(monkeypatch)
    answers = MockAnswers(workers=_decodes(2))

    answers.enqueue("apricot", prompt_contains="apricot")

    assert [url for url, _ in posted] == [
        "http://127.0.0.1:200/admin/script/all",
        "http://127.0.0.1:201/admin/script/all",
    ]
    assert all(
        body == {"output_text": "apricot", "prompt_contains": "apricot"} for _, body in posted
    )
    assert answers.enqueued == 1, "one answer stated, wherever it is served"


def test_a_class_of_several_workers_takes_no_answer_without_its_prompt(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("nothing to post"))
    answers = MockAnswers(workers=_decodes(3))

    with pytest.raises(ValueError, match="runs 3 mock workers.*prompt_contains"):
        answers.enqueue("apricot")

    assert answers.enqueued == 0


def test_the_answers_served_are_the_workers_hits_since_the_test_began(monkeypatch):
    hits = {"http://127.0.0.1:200": 4, "http://127.0.0.1:201": 1}

    def get(url, timeout):
        base = url.removesuffix("/admin/script/all")
        return httpx.Response(200, json={"hits": hits[base]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    deleted = []
    monkeypatch.setattr(httpx, "delete", lambda url, timeout: deleted.append(url))
    answers = MockAnswers(workers=_decodes(2))
    answers.note_served()
    answers.enqueued = 2

    assert answers.served() == 0
    assert answers.leftovers() == 2, "stated, not yet served"
    hits["http://127.0.0.1:201"] += 2
    assert answers.leftovers() == 0, "served on whichever worker the gateway picked"
    hits["http://127.0.0.1:200"] += 1
    assert answers.leftovers() == -1, "more served than stated: a text matched twice"

    answers.clear()
    assert deleted == [
        "http://127.0.0.1:200/admin/script/all",
        "http://127.0.0.1:201/admin/script/all",
    ]


def test_the_copies_left_on_the_other_workers_are_dropped_once_every_stated_answer_was_served(
    monkeypatch,
):
    hits = {"http://127.0.0.1:200": 0, "http://127.0.0.1:201": 0}

    def get(url, timeout):
        base = url.removesuffix("/admin/script/all")
        return httpx.Response(200, json={"hits": hits[base]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    posted = _posting(monkeypatch)
    deleted = []
    monkeypatch.setattr(httpx, "delete", lambda url, timeout: deleted.append(url))
    answers = MockAnswers(workers=_decodes(2))
    answers.note_served()

    # The first turn's answer, served on one worker: its copy on the other
    # would match the second turn's prompt, which carries the first turn.
    answers.enqueue("OK", prompt_contains="Reply OK.")
    assert deleted == [], "nothing stated before it: nothing to drop"
    hits["http://127.0.0.1:200"] += 1
    answers.enqueue("marigold", prompt_contains="What is the password?")
    assert deleted == [
        "http://127.0.0.1:200/admin/script/all",
        "http://127.0.0.1:201/admin/script/all",
    ], "every stated answer served: the leftover copies go before the next is filed"
    assert [body["output_text"] for _, body in posted] == ["OK", "OK", "marigold", "marigold"]

    # Stated answers still waiting are not dropped by the next filing.
    deleted.clear()
    answers.enqueue("blue", prompt_contains="colour")
    assert deleted == [], "marigold is still waiting"
    assert answers.leftovers() == 2


def test_the_class_workers_are_read_in_every_role_as_a_copy():
    encode = Worker(
        model_id="twin",
        engine="mock",
        port=300,
        gpu_ids=[],
        worker_type=WorkerType.ENCODE,
        admin_port=400,
    )
    decodes = _decodes(2)
    set_class_workers([encode, *decodes])
    try:
        workers = class_workers()
        assert workers == [encode, *decodes], "every role, in the order the backend set"
        workers.clear()
        assert len(class_workers()) == 3, "a copy: the class's list is not the caller's to change"
    finally:
        clear_class_workers()
    assert class_workers() == []
