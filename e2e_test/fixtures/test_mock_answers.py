"""CPU-only checks for the answers a test states with a logprob per token."""

import importlib
from types import SimpleNamespace

import httpx
import pytest
from fixtures.mock_answers import MockAnswers
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
    answers = MockAnswers(worker=worker)

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
    answers = MockAnswers(worker=worker)

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
    answers = MockAnswers(worker=worker)

    with pytest.raises(ValueError, match="do not partition the text; state this answer as pieces"):
        answers.enqueue_with_logprobs("Paris.")

    assert answers.enqueued == 0


def test_an_answer_without_tokens_is_refused(monkeypatch, model_dir):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("nothing to post"))
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": model_dir})
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(worker=worker)

    with pytest.raises(ValueError, match="'': no tokens to state a logprob for"):
        answers.enqueue_with_logprobs("")

    assert answers.enqueued == 0


def test_a_model_directory_without_a_tokenizer_file_is_named(monkeypatch, tmp_path):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("nothing to post"))
    monkeypatch.setattr(answers_module, "get_model_spec", lambda model_id: {"model": str(tmp_path)})
    worker = Worker(model_id="twin", engine="mock", port=1, gpu_ids=[], admin_port=2)
    answers = MockAnswers(worker=worker)

    with pytest.raises(FileNotFoundError, match="tokenizer.json: no tokenizer file"):
        answers.enqueue_with_logprobs("Paris.")

    assert answers.enqueued == 0


def test_an_answer_with_logprobs_is_a_no_op_without_a_mock_worker(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: pytest.fail("no worker, no post"))
    answers = MockAnswers()

    answers.enqueue_with_logprobs("Paris.")

    assert answers.enqueued == 0
