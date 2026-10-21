from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mcp_book import __version__
from mcp_book.config import Config, load_config
from mcp_book.embeddings import create_embedder
from mcp_book.ingest import ingest_folder
from mcp_book.store import Store


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def command_ingest(config: Config, arguments: argparse.Namespace) -> int:
    folder = Path(arguments.folder).expanduser().resolve() if arguments.folder else None
    folder = folder or config.library.pdf_dir
    if arguments.no_ocr:
        config.ingest.ocr = False
    if arguments.no_tables:
        config.ingest.extract_tables = False

    store = Store(config.database_path)
    try:
        report = ingest_folder(
            folder,
            config,
            store,
            create_embedder(config.embedding),
            rebuild=arguments.rebuild,
            prune=arguments.prune,
            progress=log,
        )
    except (FileNotFoundError, ValueError) as error:
        log(f"Error: {error}")
        return 1
    finally:
        store.close()

    log(
        f"Done: {len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.skipped)} unchanged, {len(report.removed)} removed, "
        f"{len(report.failed)} failed. Index: {config.database_path}"
    )
    for path, error in report.failed:
        log(f"  failed: {path}: {error}")
    return 1 if report.failed else 0


def command_books(config: Config, arguments: argparse.Namespace) -> int:
    store = Store(config.database_path)
    try:
        books = store.list_books()
    finally:
        store.close()
    if not books:
        log("No books indexed yet. Run `mcp-book ingest <folder>`.")
        return 0
    for book in books:
        print(f"{book.id}\t{book.title}\t{book.author or '-'}\t{book.page_count} pages")
    return 0


def command_serve(config: Config, arguments: argparse.Namespace) -> int:
    from mcp_book.server import serve

    if not config.database_path.exists():
        log(f"Warning: no index at {config.database_path}; run `mcp-book ingest` first.")
    serve(config, arguments.transport, arguments.host, arguments.port)
    return 0


def command_search(config: Config, arguments: argparse.Namespace) -> int:
    from mcp_book.library import Library, LibraryError

    library = Library(config)
    try:
        result = library.search(arguments.query, arguments.book or None, arguments.top_k)
    except LibraryError as error:
        log(f"Error: {error}")
        return 1
    finally:
        library.close()
    if result.get("note"):
        log(result["note"])
    for hit in result["results"]:
        print(f"[{hit['chunk_id']}] {hit['citation']}  ({', '.join(hit['matched_by'])})")
        print(f"    {hit['excerpt']}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp-book", description="Search and read a personal PDF library through MCP."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-c", "--config", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index the PDFs in a folder (recursively)")
    ingest.add_argument("folder", nargs="?", help="PDF folder (default: library.pdf_dir)")
    ingest.add_argument("--rebuild", action="store_true", help="drop the index and re-ingest all")
    ingest.add_argument("--prune", action="store_true", help="remove books whose file is gone")
    ingest.add_argument("--no-ocr", action="store_true", help="skip OCR of scanned pages")
    ingest.add_argument("--no-tables", action="store_true", help="skip table extraction")
    ingest.set_defaults(handler=command_ingest)

    books = commands.add_parser("books", help="list indexed books")
    books.set_defaults(handler=command_books)

    search = commands.add_parser("search", help="run a search from the terminal")
    search.add_argument("query")
    search.add_argument("--book", action="append", help="restrict to a book id (repeatable)")
    search.add_argument("-k", "--top-k", type=int, default=None)
    search.set_defaults(handler=command_search)

    serve = commands.add_parser("serve", help="run the MCP server")
    serve.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    serve.add_argument("--host", help="HTTP host (default: server.host)")
    serve.add_argument("--port", type=int, help="HTTP port (default: server.port)")
    serve.set_defaults(handler=command_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        config = load_config(arguments.config)
    except (FileNotFoundError, ValueError) as error:
        log(f"Config error: {error}")
        return 2
    return arguments.handler(config, arguments)


if __name__ == "__main__":
    raise SystemExit(main())
