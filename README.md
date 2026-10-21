# mcp-book

[Italiano](#italiano) · [English](#english)

---

## Italiano

Server MCP che permette a un assistente AI (Claude Desktop, Claude Code, qualsiasi
client MCP) di consultare una libreria personale di libri PDF, pensata per manuali di
cucina tecnici. L'assistente non riceve i libri interi: cerca, poi legge solo i
passaggi che servono, e ogni risposta porta una citazione «Titolo, Capitolo, p. N».

Tutto gira in locale: SQLite (FTS5 + sqlite-vec) per l'indice, un modello di
embedding multilingue locale, Tesseract per le pagine scansionate.

### Come funziona

- **Ingestione**: testo per pagina con PyMuPDF; OCR Tesseract (`ita+eng`) sulle
  pagine senza testo; tabelle con pdfplumber (salvate come righe e Markdown);
  capitoli e sezioni dal sommario del PDF; chunk da ~650 token (max 800) con 100 di
  sovrapposizione, che non attraversano i confini di sezione.
- **Ricerca ibrida**: BM25 (SQLite FTS5) + similarità semantica (`BAAI/bge-m3`,
  sqlite-vec), unite con reciprocal rank fusion.
- **Incrementale**: un PDF non modificato non viene rielaborato.

Dettagli e scelte progettuali in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Installazione

Requisiti: Python 3.11+, [uv](https://docs.astral.sh/uv/), Tesseract con il
pacchetto della lingua italiana.

```bash
# Tesseract
brew install tesseract tesseract-lang                       # macOS
sudo apt install tesseract-ocr tesseract-ocr-ita            # Debian/Ubuntu

git clone https://github.com/p1e7r0/mcp-book.git
cd mcp-book
uv sync
```

Su Linux `uv sync` installa torch con le librerie CUDA (diversi GB). Senza GPU
conviene la versione CPU:

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

In questo caso usa `uv run --no-sync mcp-book ...` (oppure `.venv/bin/mcp-book`):
un `uv run` normale riallinea l'ambiente al lockfile e reinstalla torch con CUDA.
Nella configurazione dei client MCP indica direttamente `/percorso/mcp-book/.venv/bin/mcp-book`
come `command` e `["serve"]` come `args`.

Al primo uso viene scaricato il modello di embedding (`BAAI/bge-m3`, circa 2,3 GB).

### Ingestione

```bash
mkdir -p library          # oppure il percorso in config.toml
cp ~/Libri/*.pdf library/
uv run mcp-book ingest library
uv run mcp-book books     # elenco dei libri indicizzati
```

Opzioni: `--rebuild` (ricostruisce tutto, necessario se cambi modello di
embedding), `--prune` (rimuove i libri il cui file non esiste più), `--no-ocr`,
`--no-tables`. La cartella viene letta ricorsivamente.

Se i metadati del PDF sono sbagliati, crea accanto al PDF un file con lo stesso nome
ed estensione `.toml`:

```toml
# library/la-cucina-tecnica.toml
title = "La cucina tecnica"
author = "Nome Cognome"
```

Prova una ricerca dal terminale:

```bash
uv run mcp-book search "perché la maionese impazzisce"
```

### Configurazione (`config.toml`)

| Sezione | Chiave | Default | Note |
| --- | --- | --- | --- |
| `library` | `pdf_dir` | `library` | percorsi relativi al file di config |
| `library` | `index_dir` | `data` | contiene `library.sqlite` |
| `embedding` | `backend` | `sentence-transformers` | `none` = solo BM25 |
| `embedding` | `model` | `BAAI/bge-m3` | cambiarlo richiede `ingest --rebuild` |
| `embedding` | `device` | `cpu` | `cuda` o `mps` se disponibili |
| `chunking` | `target_tokens` / `max_tokens` / `overlap_tokens` | 650 / 800 / 100 | |
| `ingest` | `ocr`, `ocr_languages`, `ocr_min_chars`, `ocr_dpi` | `true`, `ita+eng`, 25, 300 | |
| `ingest` | `extract_tables` | `true` | |
| `search` | `top_k`, `candidates`, `rrf_k` | 8, 50, 60 | |
| `server` | `host`, `port` | `127.0.0.1`, 8765 | solo trasporto HTTP |

Il file viene cercato in quest'ordine: `--config`, variabile `MCP_BOOK_CONFIG`,
`./config.toml`, `~/.config/mcp-book/config.toml`.

Con i modelli `multilingual-e5` i prefissi `query: `/`passage: ` sono automatici, ma
quei modelli troncano a 512 token: abbassa `max_tokens` di conseguenza.

### Claude Desktop

In `claude_desktop_config.json` (Impostazioni › Sviluppatore › Modifica config):

```json
{
  "mcpServers": {
    "mcp-book": {
      "command": "uv",
      "args": ["--directory", "/percorso/assoluto/mcp-book", "run", "mcp-book", "serve"]
    }
  }
}
```

Se Claude Desktop non trova `uv`, usa il percorso assoluto (`which uv`). Riavvia
Claude Desktop dopo la modifica.

### Claude Code

```bash
claude mcp add --scope user mcp-book -- \
  uv --directory /percorso/assoluto/mcp-book run mcp-book serve
```

Oppure via HTTP, con il server avviato a parte:

```bash
uv run mcp-book serve --transport http          # http://127.0.0.1:8765/mcp
claude mcp add --transport http mcp-book http://127.0.0.1:8765/mcp
```

La prima ricerca semantica carica il modello in memoria e richiede qualche secondo.

### Docker (opzionale)

```bash
docker build -t mcp-book .
docker run --rm -v ~/Libri:/library:ro -v mcp-book-data:/data mcp-book ingest
docker run --rm -p 8765:8765 -v mcp-book-data:/data mcp-book    # HTTP su :8765/mcp
```

Per stdio (Claude Desktop): `"command": "docker", "args": ["run", "-i", "--rm",
"-v", "mcp-book-data:/data", "mcp-book", "serve"]`.

### Strumenti MCP

| Strumento | Cosa restituisce |
| --- | --- |
| `list_books()` | id, titolo, autore, numero di pagine |
| `get_toc(book_id)` | capitoli e sezioni con pagina |
| `search(query, book_ids?, top_k=8)` | estratto, libro, capitolo, sezione, pagine, `chunk_id`, punteggio |
| `read_pages(book_id, start_page, end_page)` | testo completo, max 10 pagine; oltre restituisce `truncated` e `next_start_page` |
| `read_chunk(chunk_id, context=1)` | il chunk e i vicini (0-5 per lato) |
| `get_tables(book_id, page)` | tabelle della pagina in Markdown |

Ogni risposta (o ogni elemento, per gli elenchi) ha un campo `citation`. Le pagine
negli argomenti sono indici del PDF a partire da 1; `page_label` è il numero
stampato nel libro e viene usato nelle citazioni quando il PDF lo contiene.

### Esempi di domande

- «A che temperatura al cuore cuocio un controfiletto sottovuoto? Cita la fonte.»
- «Confronta come i miei libri spiegano la stabilità della maionese.»
- «Nel manuale di pasticceria, qual è il rapporto zucchero/acqua per lo sciroppo?
  Mostrami la tabella.»
- «Riassumi il capitolo sulle salse madri di *La cucina tecnica*.»

### Copyright

I PDF e l'indice (che contiene il testo dei libri) sono esclusi dal repository
tramite `.gitignore`: sono contenuti protetti da copyright, per solo uso personale.
Non pubblicare la cartella `data/` né esporre il server HTTP su una rete pubblica.

### Sviluppo

```bash
uv run pytest          # il PDF di esempio è generato dai test
uv run ruff check . && uv run ruff format --check .
```

I test usano un embedder deterministico (`backend = "hash"`) e non scaricano
modelli; i test OCR vengono saltati se Tesseract non è installato.

---

## English

An MCP server that lets an AI assistant (Claude Desktop, Claude Code, any MCP
client) consult a personal library of PDF books, designed for technical cooking
manuals. The assistant never receives whole books: it searches, then reads only
the passages it needs, and every answer carries a citation "Title, Chapter, p. N".

Everything runs locally: SQLite (FTS5 + sqlite-vec) for the index, a local
multilingual embedding model, Tesseract for scanned pages.

### How it works

- **Ingestion**: per-page text with PyMuPDF; Tesseract OCR (`ita+eng`) on pages
  without text; tables via pdfplumber (stored as rows and Markdown); chapters and
  sections from the PDF outline; ~650-token chunks (max 800) with 100 tokens of
  overlap that never cross a section boundary.
- **Hybrid search**: BM25 (SQLite FTS5) + semantic similarity (`BAAI/bge-m3`,
  sqlite-vec), merged with reciprocal rank fusion.
- **Incremental**: unchanged PDFs are not processed again.

Design notes in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Installation

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and Tesseract with Italian
language data.

```bash
brew install tesseract tesseract-lang                       # macOS
sudo apt install tesseract-ocr tesseract-ocr-ita            # Debian/Ubuntu

git clone https://github.com/p1e7r0/mcp-book.git
cd mcp-book
uv sync
```

On Linux `uv sync` installs torch with CUDA libraries (several GB). Without a GPU,
install the CPU build instead:

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
```

Then use `uv run --no-sync mcp-book ...` (or `.venv/bin/mcp-book`): a plain
`uv run` re-syncs the environment to the lockfile and brings CUDA torch back. In MCP
client configs, use `/path/to/mcp-book/.venv/bin/mcp-book` as `command` and
`["serve"]` as `args`.

The embedding model (`BAAI/bge-m3`, about 2.3 GB) is downloaded on first use.

### Ingestion

```bash
mkdir -p library
cp ~/Books/*.pdf library/
uv run mcp-book ingest library
uv run mcp-book books
uv run mcp-book search "why does mayonnaise split"
```

Flags: `--rebuild` (re-index everything; required after changing the embedding
model), `--prune` (drop books whose file is gone), `--no-ocr`, `--no-tables`.
A `<name>.toml` file next to a PDF (`title = "..."`, `author = "..."`) overrides
bad PDF metadata.

### Configuration

See the table in the Italian section; keys and defaults are the same, and
`config.toml` is commented. Lookup order: `--config`, `MCP_BOOK_CONFIG`,
`./config.toml`, `~/.config/mcp-book/config.toml`. Relative paths resolve from the
config file's directory. `multilingual-e5` models get their `query: `/`passage: `
prefixes automatically but truncate at 512 tokens: lower `max_tokens` if you use one.

### Claude Desktop

```json
{
  "mcpServers": {
    "mcp-book": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mcp-book", "run", "mcp-book", "serve"]
    }
  }
}
```

Use the absolute path of `uv` if Claude Desktop cannot find it, then restart it.

### Claude Code

```bash
claude mcp add --scope user mcp-book -- \
  uv --directory /absolute/path/to/mcp-book run mcp-book serve

# or over streamable HTTP
uv run mcp-book serve --transport http
claude mcp add --transport http mcp-book http://127.0.0.1:8765/mcp
```

### Docker (optional)

```bash
docker build -t mcp-book .
docker run --rm -v ~/Books:/library:ro -v mcp-book-data:/data mcp-book ingest
docker run --rm -p 8765:8765 -v mcp-book-data:/data mcp-book
```

### MCP tools

`list_books()`, `get_toc(book_id)`, `search(query, book_ids?, top_k=8)`,
`read_pages(book_id, start_page, end_page)` (max 10 pages per call),
`read_chunk(chunk_id, context=1)`, `get_tables(book_id, page)`. Every response, or
every item of a list, has a `citation` field. Page arguments are 1-based PDF page
indexes; `page_label` is the printed page number and is used in citations when
the PDF defines it.

### Example prompts

- "What core temperature for a sous-vide beef sirloin? Cite the source."
- "Compare how my books explain mayonnaise stability."
- "Show me the sugar syrup table in the pastry manual."

### Copyright

PDFs and the index (which contains the books' text) are git-ignored: they are
copyrighted and for personal use only. Do not publish `data/` or expose the HTTP
server on a public network.

### Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests use a deterministic embedder (`backend = "hash"`) and download no model; the
OCR tests are skipped when Tesseract is missing.
