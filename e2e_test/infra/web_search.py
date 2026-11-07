"""The live web-search backend behind the suites' MCP cases.

The cases that search the web run through an MCP server in front of a live
search backend, and a search can come back empty for reasons of the backend's
own: it is rate limited, briefly down, or has nothing for the query. The
server reports that as a tool error, and the gateway relays it faithfully as
an ``mcp_call`` item whose ``error`` is an ``mcp_tool_execution_error`` with
the backend's text as its content. A case that asserted a successful search
has nothing to judge the gateway by then, so it skips with that reason
instead of failing; any other error keeps failing it.
"""

from __future__ import annotations

from typing import Any

import pytest

NO_RESULTS_ANSWER = "No web results found"
"""The backend's answer to a search that found nothing, relayed verbatim."""

TOOL_EXECUTION_ERROR = "mcp_tool_execution_error"
"""The ``error.type`` of an ``mcp_call`` whose tool ran and reported an error."""


def _content_text(content: Any) -> str:
    """The text of a tool-execution error's content: a string, or its text parts joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        texts = []
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if isinstance(text, str):
                texts.append(text)
        return "".join(texts)
    return ""


def is_no_results_answer(error: Any) -> bool:
    """Whether an ``mcp_call`` error is the backend's no-results answer and nothing else."""
    error_type = error.get("type") if isinstance(error, dict) else getattr(error, "type", None)
    content = error.get("content") if isinstance(error, dict) else getattr(error, "content", None)
    return (
        error_type == TOOL_EXECUTION_ERROR and _content_text(content).strip() == NO_RESULTS_ANSWER
    )


def skip_if_web_search_found_nothing(output: Any) -> None:
    """Skip the running case when the backend answered its search calls with no results.

    ``output`` is a response's output items (the final response's, for a
    stream). The case skips only when every errored ``mcp_call`` carries the
    no-results answer; any other error, alone or beside it, is left to the
    case's own assertions, which fail on it.
    """
    errors = [
        item.error
        for item in output or ()
        if getattr(item, "type", None) == "mcp_call" and getattr(item, "error", None) is not None
    ]
    if errors and all(is_no_results_answer(error) for error in errors):
        pytest.skip(
            f"the web-search backend answered the MCP call with {NO_RESULTS_ANSWER!r} "
            f"(relayed by the gateway as {TOOL_EXECUTION_ERROR}): nothing to judge the gateway by"
        )
