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

### Guida passo passo: Mac con Apple Silicon e VS Code

Vale per Mac con chip M1 o successivi e macOS 14 (Sonoma) o più recente: torch non
pubblica più pacchetti per i Mac Intel. Controlla nel Terminale:

```bash
uname -m                   # deve stampare arm64
sw_vers -productVersion    # 14.0 o superiore
```

#### 1. Installa gli strumenti

Se non hai Homebrew, installalo da [brew.sh](https://brew.sh). Poi:

```bash
brew install uv tesseract tesseract-lang
tesseract --list-langs | grep -x ita       # deve stampare: ita
```

`tesseract-lang` contiene tutte le lingue, italiano compreso. Python non va
installato a parte: uv scarica da solo la versione indicata in `.python-version`.

#### 2. Scarica il progetto

```bash
cd ~
git clone https://github.com/p1e7r0/mcp-book.git
cd mcp-book
uv sync
```

Se macOS propone di installare gli strumenti da riga di comando di Xcode (servono a
`git`), accetta e poi ripeti il comando. `uv sync` crea `.venv` con tutte le
dipendenze, compreso torch per Apple Silicon con supporto alla GPU (MPS).

#### 3. Indica dove sono i PDF

Copia i PDF in `~/mcp-book/library/` (`mkdir -p ~/mcp-book/library` la crea), oppure
lasciali dove sono e modifica `config.toml`:

```toml
[library]
pdf_dir = "~/Documents/Libri di cucina"
```

L'indice resta in `~/mcp-book/data/`. Il `device = "auto"` già presente in
`config.toml` usa la GPU del Mac.

#### 4. Indicizza i libri

```bash
cd ~/mcp-book
uv run mcp-book ingest
uv run mcp-book books                                   # libri indicizzati
uv run mcp-book search "perché la maionese impazzisce"  # prova una ricerca
```

- La prima volta scarica il modello `BAAI/bge-m3` (circa 2,3 GB, in
  `~/.cache/huggingface`).
- Le pagine scansionate passano per l'OCR, che è la parte più lenta.
- Se macOS chiede al Terminale l'accesso a Documenti o Scrivania, consenti.
- Quando aggiungi libri rilancia `uv run mcp-book ingest`: elabora solo i file
  nuovi o modificati.

#### 5. Trova i percorsi assoluti

Le app avviate dal Dock non vedono il PATH del Terminale, quindi nelle
configurazioni MCP servono percorsi assoluti:

```bash
which uv        # di solito /opt/homebrew/bin/uv
echo $HOME      # per esempio /Users/mario
```

Negli esempi seguenti sostituisci `/Users/TUO_NOME` con il risultato di `echo $HOME`.

#### 6a. VS Code con GitHub Copilot (modalità agente)

1. Cmd+Shift+P › **MCP: Open User Configuration** per averlo in tutti i progetti
   (per un solo progetto usa invece il file `.vscode/mcp.json` di quel progetto).
2. Incolla, adattando i percorsi:

   ```json
   {
     "servers": {
       "mcp-book": {
         "type": "stdio",
         "command": "/opt/homebrew/bin/uv",
         "args": ["--directory", "/Users/TUO_NOME/mcp-book", "run", "mcp-book", "serve"]
       }
     }
   }
   ```

3. Salva e avvia il server: Cmd+Shift+P › **MCP: List Servers** › mcp-book ›
   **Start Server**. Al primo avvio VS Code chiede se ti fidi del server: conferma.
4. Apri la chat di Copilot, scegli la modalità **Agent**, apri il selettore degli
   strumenti e controlla che i sei strumenti di mcp-book siano attivi.
5. Chiedi, per esempio: «Cerca nei miei libri a che temperatura cuoce il salmone
   sottovuoto e cita la fonte.»

Se qualcosa non va, **MCP: List Servers** › mcp-book › **Show Output** mostra i
log del server.

#### 6b. VS Code con l'estensione Claude Code

L'estensione legge la stessa configurazione MCP della CLI `claude` (se il comando
manca: `curl -fsSL https://claude.ai/install.sh | bash`). Nel terminale integrato di
VS Code:

```bash
claude mcp add --scope user mcp-book -- \
  /opt/homebrew/bin/uv --directory /Users/TUO_NOME/mcp-book run mcp-book serve
claude mcp list            # mcp-book deve risultare connesso
```

`--scope user` lo rende disponibile in tutti i progetti. Poi, nel pannello di
Claude Code, chiedi di cercare nei tuoi libri.

#### 6c. Claude Desktop

Vedi la sezione [Claude Desktop](#claude-desktop): su Mac il file è
`~/Library/Application Support/Claude/claude_desktop_config.json` e `command` deve
essere `/opt/homebrew/bin/uv`.

#### Problemi comuni

| Sintomo | Soluzione |
| --- | --- |
| Il server non parte, `spawn uv ENOENT` | in `command` serve il percorso assoluto di uv |
| Errori `mps` durante l'ingestione | `device = "cpu"` in `config.toml`, oppure lancia con `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run mcp-book ingest` |
| Avviso `sqlite-vec unavailable` | il Python in uso non carica estensioni SQLite: `rm -rf .venv && uv sync --managed-python` |
| Avviso `OCR skipped` | `brew install tesseract tesseract-lang`; se persiste, `export TESSDATA_PREFIX=/opt/homebrew/share/tessdata` prima di `ingest` |
| Errore che chiede `--rebuild` | hai cambiato `model`: `uv run mcp-book ingest --rebuild` |
| La prima ricerca è lenta | il modello viene caricato in memoria alla prima domanda, poi resta caricato |

### Installazione su altri sistemi

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
| `embedding` | `device` | `auto` | `cuda`, poi `mps` (GPU Apple), poi `cpu` |
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

In `claude_desktop_config.json` (Impostazioni › Sviluppatore › Modifica config; su Mac
è `~/Library/Application Support/Claude/claude_desktop_config.json`):

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

### Step by step: Apple Silicon Mac and VS Code

Requires an M1 or later Mac on macOS 14+ (torch no longer ships Intel Mac wheels):
`uname -m` must print `arm64`.

```bash
brew install uv tesseract tesseract-lang
cd ~ && git clone https://github.com/p1e7r0/mcp-book.git && cd mcp-book
uv sync                                    # Python and torch with MPS included
mkdir -p library && cp ~/Books/*.pdf library/   # or set pdf_dir in config.toml
uv run mcp-book ingest                     # first run downloads bge-m3 (~2.3 GB)
uv run mcp-book search "why does mayonnaise split"
which uv; echo $HOME                       # absolute paths for the configs below
```

`device = "auto"` in `config.toml` already uses the Mac GPU (MPS).

**VS Code + GitHub Copilot (agent mode)**: Cmd+Shift+P › **MCP: Open User
Configuration** (or `.vscode/mcp.json` for one project), then:

```json
{
  "servers": {
    "mcp-book": {
      "type": "stdio",
      "command": "/opt/homebrew/bin/uv",
      "args": ["--directory", "/Users/YOUR_NAME/mcp-book", "run", "mcp-book", "serve"]
    }
  }
}
```

Start it from **MCP: List Servers** › mcp-book › **Start Server**, trust it, then
enable its tools in Copilot Chat's **Agent** mode. **Show Output** in the same menu
shows the server logs.

**VS Code + Claude Code extension**: it shares the CLI's MCP config. In the
integrated terminal:

```bash
claude mcp add --scope user mcp-book -- \
  /opt/homebrew/bin/uv --directory /Users/YOUR_NAME/mcp-book run mcp-book serve
claude mcp list
```

Troubleshooting: `spawn uv ENOENT` → use uv's absolute path; `mps` errors → set
`device = "cpu"` or run with `PYTORCH_ENABLE_MPS_FALLBACK=1`; `sqlite-vec
unavailable` → `rm -rf .venv && uv sync --managed-python`; OCR
skipped → `export TESSDATA_PREFIX=/opt/homebrew/share/tessdata`.

### Installation on other systems

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
