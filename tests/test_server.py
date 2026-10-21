import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from mcp_book.config import Config
from mcp_book.server import create_server

TOOLS = {"list_books", "get_toc", "search", "read_pages", "read_chunk", "get_tables"}


def call(client_target, name: str, arguments: dict):
    async def run():
        async with Client(client_target) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def test_tools_are_registered(indexed_config: Config) -> None:
    async def run():
        async with Client(create_server(indexed_config)) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(run()).tools}
    assert set(tools) == TOOLS
    assert tools["search"].input_schema["properties"]["top_k"]["default"] == 8
    assert "10 pages" in tools["read_pages"].description
    query = tools["search"].input_schema["properties"]["query"]["description"]
    assert "language of the books" in query


def test_every_tool_returns_a_citation(indexed_config: Config) -> None:
    server = create_server(indexed_config)
    search = call(server, "search", {"query": "maionese", "top_k": 2}).structured_content
    chunk_id = search["results"][0]["chunk_id"]

    responses = {
        "list_books": call(server, "list_books", {}).structured_content["books"][0],
        "get_toc": call(server, "get_toc", {"book_id": "manuale-cucina"}).structured_content,
        "search": search["results"][0],
        "read_pages": call(
            server, "read_pages", {"book_id": "manuale-cucina", "start_page": 1, "end_page": 2}
        ).structured_content,
        "read_chunk": call(server, "read_chunk", {"chunk_id": chunk_id}).structured_content,
        "get_tables": call(
            server, "get_tables", {"book_id": "manuale-cucina", "page": 4}
        ).structured_content,
    }

    assert set(responses) == TOOLS
    for name, response in responses.items():
        assert response["citation"].startswith("Manuale di cucina tecnica"), name


def test_errors_are_tool_errors(indexed_config: Config) -> None:
    result = call(create_server(indexed_config), "get_toc", {"book_id": "nessuno"})
    assert result.is_error
    assert "Unknown book_id" in result.content[0].text


def test_stdio_transport(indexed_config: Config) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_book.cli", "serve", "--transport", "stdio"],
        env={
            **os.environ,
            "MCP_BOOK_CONFIG": str(indexed_config.library.index_dir.parent / "config.toml"),
        },
    )
    result = call(parameters, "list_books", {})
    assert result.structured_content["count"] == 1


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_streamable_http_transport(indexed_config: Config) -> None:
    port = free_port()
    config_path = Path(indexed_config.library.index_dir.parent / "config.toml")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mcp_book.cli",
            "-c",
            str(config_path),
            "serve",
            "--transport",
            "http",
            "--port",
            str(port),
        ],
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.2)
        else:
            pytest.fail("HTTP server did not start")
        result = call(f"http://127.0.0.1:{port}/mcp", "search", {"query": "besciamella"})
        assert result.structured_content["results"][0]["section"] == "Besciamella"
    finally:
        process.terminate()
        process.wait(timeout=10)
