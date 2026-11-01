"""The web-search backend's no-results answer, as the suites read it off a response's output."""

from __future__ import annotations

import pytest
from infra.web_search import (
    NO_RESULTS_ANSWER,
    is_no_results_answer,
    skip_if_web_search_found_nothing,
)
from openai.types.responses.response_output_item import McpCall, McpListTools


def _mcp_call(error: dict | None = None) -> McpCall:
    return McpCall.model_validate(
        {
            "id": "mcp_1",
            "type": "mcp_call",
            "name": "web_search",
            "arguments": '{"query": "Python programming language", "count": 1}',
            "server_label": "search",
            "status": "completed" if error is None else "failed",
            "error": error,
            "output": None if error else "1. Python (programming language) - Wikipedia",
        }
    )


LIST_TOOLS = McpListTools.model_validate(
    {"id": "mcpl_1", "type": "mcp_list_tools", "server_label": "search", "tools": []}
)

NO_RESULTS = {"type": "mcp_tool_execution_error", "content": NO_RESULTS_ANSWER}
NO_RESULTS_AS_PARTS = {
    "type": "mcp_tool_execution_error",
    "content": [{"type": "text", "text": NO_RESULTS_ANSWER}],
}
RATE_LIMITED = {"type": "mcp_tool_execution_error", "content": "rate limited"}
PROTOCOL_ERROR = {"type": "mcp_protocol_error", "code": -32000, "message": NO_RESULTS_ANSWER}


def test_the_no_results_answer_skips_the_case():
    with pytest.raises(pytest.skip.Exception, match="No web results found"):
        skip_if_web_search_found_nothing([LIST_TOOLS, _mcp_call(NO_RESULTS)])


def test_the_no_results_answer_in_text_parts_skips_the_case():
    with pytest.raises(pytest.skip.Exception):
        skip_if_web_search_found_nothing([_mcp_call(NO_RESULTS_AS_PARTS)])


def test_another_error_content_does_not_skip():
    assert skip_if_web_search_found_nothing([_mcp_call(RATE_LIMITED)]) is None


def test_another_error_type_does_not_skip_even_with_that_text():
    assert skip_if_web_search_found_nothing([_mcp_call(PROTOCOL_ERROR)]) is None


def test_the_no_results_answer_beside_another_error_does_not_skip():
    assert (
        skip_if_web_search_found_nothing([_mcp_call(NO_RESULTS), _mcp_call(RATE_LIMITED)]) is None
    )


def test_a_successful_search_does_not_skip():
    assert skip_if_web_search_found_nothing([LIST_TOOLS, _mcp_call()]) is None


def test_an_output_without_calls_does_not_skip():
    assert skip_if_web_search_found_nothing([LIST_TOOLS]) is None
    assert skip_if_web_search_found_nothing([]) is None
    assert skip_if_web_search_found_nothing(None) is None


def test_the_answer_is_told_by_type_and_exact_text():
    assert is_no_results_answer(_mcp_call(NO_RESULTS).error)
    assert is_no_results_answer(_mcp_call(NO_RESULTS_AS_PARTS).error)
    assert not is_no_results_answer(_mcp_call(RATE_LIMITED).error)
    assert not is_no_results_answer(_mcp_call(PROTOCOL_ERROR).error)
    assert not is_no_results_answer(None)
