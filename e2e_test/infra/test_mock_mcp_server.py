"""Exercise the mock's real MCP HTTP transport without a model server."""

import asyncio
import json
import socket

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .mock_mcp_server import IMAGE_GENERATION_PNG_BASE64, MockMcpServer


def test_sdk_discovers_and_calls_the_original_tool():
    async def exchange(server):
        async with streamable_http_client(server.url) as streams:
            async with ClientSession(*streams[:2]) as client:
                initialized = await client.initialize()
                assert initialized.model_dump(by_alias=True)["serverInfo"]["name"] == "smg-e2e-mock"
                tools = (await client.list_tools()).tools
                assert [tool.name for tool in tools] == ["image_generation"]
                assert tools[0].model_dump(by_alias=True)["inputSchema"]["required"] == ["prompt"]
                for arguments in (
                    {"prompt": "a dog"},
                    {"prompt": "a cat", "size": "512x512", "quality": "high", "user": "u"},
                ):
                    result = await client.call_tool("image_generation", arguments)
                    expected = {
                        "result": IMAGE_GENERATION_PNG_BASE64,
                        "revised_prompt": arguments["prompt"],
                        "status": "completed",
                    }
                    assert not result.model_dump(by_alias=True)["isError"]
                    assert len(result.content) == 1
                    assert result.content[0].type == "text"
                    assert json.loads(result.content[0].text) == expected
                    assert result.model_dump(by_alias=True)["structuredContent"] == expected

    with MockMcpServer() as server:
        asyncio.run(exchange(server))
        assert server.call_log == [
            {
                "tool": "image_generation",
                "arguments": {
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "moderation": "auto",
                    "output_format": "png",
                    "user": user,
                },
            }
            for prompt, size, quality, user in (
                ("a dog", "1024x1024", "standard", None),
                ("a cat", "512x512", "high", "u"),
            )
        ]


@pytest.mark.parametrize(
    ("name", "arguments"),
    [("image_generation", {}), ("image_generation", {"prompt": []}), ("unknown", {})],
)
def test_sdk_rejects_bad_calls_without_tool_side_effects(name, arguments):
    async def exchange(server):
        async with streamable_http_client(server.url) as streams:
            async with ClientSession(*streams[:2]) as client:
                await client.initialize()
                result = await client.call_tool(name, arguments)
                assert result.model_dump(by_alias=True)["isError"]

    with MockMcpServer() as server:
        asyncio.run(exchange(server))
        assert server.call_log == []


def test_stop_releases_the_thread_and_socket_and_allows_restart():
    async def exchange(server):
        async with streamable_http_client(server.url) as streams:
            async with ClientSession(*streams[:2]) as client:
                await client.initialize()
                assert (await client.list_tools()).tools[0].name == "image_generation"
                result = await client.call_tool("image_generation", {"prompt": "restart"})
                assert not result.model_dump(by_alias=True)["isError"]
                assert result.model_dump(by_alias=True)["structuredContent"]["result"] == (
                    IMAGE_GENERATION_PNG_BASE64
                )

    server = MockMcpServer()
    for _ in range(2):
        server.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                server.start()
            thread, port = server._thread, server.port
            with socket.create_connection((server.host, port), timeout=1):
                pass
            asyncio.run(exchange(server))
        finally:
            server.stop()
        assert not thread.is_alive()
        assert server.port is None
        with pytest.raises(OSError):
            socket.create_connection((server.host, port), timeout=1)
    server.stop()
    assert len(server.call_log) == 2
