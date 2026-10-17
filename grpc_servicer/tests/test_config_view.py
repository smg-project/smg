"""The lenient reads the engine configuration views share.

Run with: pytest grpc_servicer/tests/test_config_view.py
"""

from types import SimpleNamespace

import pytest
from smg_grpc_servicer.config_view import (
    attr_int,
    attr_list,
    attr_str,
    classification_labels,
    positive_int,
)


@pytest.mark.parametrize("value", [1, 8, 2**40])
def test_a_count_above_zero_is_itself(value):
    assert positive_int(value) == value


@pytest.mark.parametrize("value", [0, -1, True, False, 2.0, "3", None, [4]])
def test_what_is_not_a_count_reads_as_none(value):
    assert positive_int(value) is None


def test_an_absent_config_reads_as_the_defaults():
    for config in (None, SimpleNamespace()):
        assert attr_str(config, "model_type") == ""
        assert attr_int(config, "pad_token_id") == 0
        assert attr_list(config, "architectures") == []


def test_an_empty_value_reads_as_the_default_and_a_value_as_itself():
    empty = SimpleNamespace(model_type=None, pad_token_id=None, architectures=None)
    assert attr_str(empty, "model_type") == ""
    assert attr_int(empty, "pad_token_id") == 0
    assert attr_list(empty, "architectures") == []
    declared = SimpleNamespace(model_type="qwen3", pad_token_id=7, architectures=("A", "B"))
    assert attr_str(declared, "model_type") == "qwen3"
    assert attr_int(declared, "pad_token_id") == 7
    assert attr_list(declared, "architectures") == ["A", "B"]


def test_a_falsy_config_object_reads_as_absent():
    # The views guard on the config's truth, as the servicers always did.
    class Falsy:
        model_type = "set"

        def __bool__(self):
            return False

    assert attr_str(Falsy(), "model_type") == ""


def test_classification_heads_are_reconciled_between_their_spellings():
    assert classification_labels(None) == (None, 0)
    assert classification_labels(SimpleNamespace()) == (None, 0)
    # Only the count: the labels are synthesized the HF way.
    assert classification_labels(SimpleNamespace(num_labels=2)) == (
        {0: "LABEL_0", 1: "LABEL_1"},
        2,
    )
    # Only the labels: the count is theirs.
    labels = {0: "neg", 1: "pos"}
    assert classification_labels(SimpleNamespace(id2label=labels)) == (labels, 2)
    # Both declared: both kept as declared, the count an int.
    assert classification_labels(SimpleNamespace(id2label=labels, num_labels=2)) == (labels, 2)
    # An empty mapping with no count declares no heads.
    assert classification_labels(SimpleNamespace(id2label={}, num_labels=0)) == ({}, 0)
