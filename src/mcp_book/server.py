from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_book import __version__
from mcp_book.config import Config
from mcp_book.library import MAX_CONTEXT, MAX_TOP_K, Library, LibraryError

INSTRUCTIONS = """\
Personal library of technical cookbooks (PDF). Do not guess what the books say: \
call `search` first, then `read_chunk` or `read_pages` to read around the best hits, \
and `get_tables` for tables (temperatures, ratios, times). Write `search` queries in the \
language the books are written in, whatever language the user speaks: keyword matching \
only works within one language. If the library mixes languages, search once per \
language. Pages are 1-based PDF page indexes; `page_label` is the number printed in the \
book. Quote the `citation` field of every passage you rely on, in the form \
"Title, Chapter, p. N".\
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


def _tool_errors(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapper(*arguments: Any, **keywords: Any) -> dict[str, Any]:
        try:
            return function(*arguments, **keywords)
        except LibraryError as error:
            raise ToolError(str(error)) from error

    return wrapper


def create_server(config: Config, library: Library | None = None) -> MCPServer:
    library = library or Library(config)
    server = MCPServer("mcp-book", version=__version__, instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def list_books() -> dict[str, Any]:
        """List the indexed books with book_id, title, author and page count."""
        return library.list_books()

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def get_toc(book_id: str) -> dict[str, Any]:
        """Table of contents of a book: chapters and sections with their start pages."""
        return library.get_toc(book_id)

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def search(
        query: Annotated[
            str,
            Field(
                description="Question or keywords in the language of the books searched "
                "(translate the user's question if needed)"
            ),
        ],
        book_ids: Annotated[
            list[str] | None, Field(description="Restrict to these books (from list_books)")
        ] = None,
        top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K)] = config.search.top_k,
    ) -> dict[str, Any]:
        """Hybrid keyword + semantic search over all chunks.

        Each result has an excerpt, book, chapter, section, pages, a chunk_id for
        read_chunk and a citation.
        """
        return library.search(query, book_ids, top_k)

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def read_pages(
        book_id: str,
        start_page: Annotated[int, Field(ge=1, description="1-based PDF page index")],
        end_page: Annotated[int, Field(ge=1, description="Inclusive")],
    ) -> dict[str, Any]:
        """Full text of a page range, at most 10 pages per call.

        Longer ranges are cut and return truncated=true with next_start_page.
        """
        return library.read_pages(book_id, start_page, end_page)

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def read_chunk(
        chunk_id: int,
        context: Annotated[
            int, Field(ge=0, le=MAX_CONTEXT, description="Adjacent chunks on each side")
        ] = 1,
    ) -> dict[str, Any]:
        """A chunk from search results plus its neighbours, in reading order."""
        return library.read_chunk(chunk_id, context)

    @server.tool(annotations=READ_ONLY)
    @_tool_errors
    def get_tables(
        book_id: str, page: Annotated[int, Field(ge=1, description="1-based PDF page index")]
    ) -> dict[str, Any]:
        """Tables detected on a page, as Markdown."""
        return library.get_tables(book_id, page)

    return server


def serve(
    config: Config,
    transport: Literal["stdio", "http"] = "stdio",
    host: str | None = None,
    port: int | None = None,
) -> None:
    server = create_server(config)
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run(
            "streamable-http", host=host or config.server.host, port=port or config.server.port
        )
