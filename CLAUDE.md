# mcp-book: istruzioni operative

Server MCP in Python che indicizza una libreria personale di PDF (manuali di cucina
tecnici) e la espone a un assistente AI: ricerca ibrida, lettura di pagine e chunk,
tabelle, citazioni «Titolo, Capitolo, p. N». Architettura e decisioni in
`docs/ARCHITECTURE.md`; istruzioni utente in `README.md` (italiano + inglese).

Rispondi all'utente in italiano. Codice, identificatori, commenti e commit in inglese.

## Comandi
- `uv sync`: installa (su Mac Apple Silicon torch include MPS; su Linux porta CUDA, vedi README)
- `uv run pytest`: test (nessun download di modelli, il PDF di esempio è generato)
- `uv run ruff check . && uv run ruff format --check .`: lint e formato (line length 100)
- `uv run mcp-book ingest [cartella] [--rebuild] [--prune] [--no-ocr] [--no-tables]`
- `uv run mcp-book books` / `uv run mcp-book search "domanda" [--book ID] [-k N]`
- `uv run mcp-book serve [--transport stdio|http] [--host] [--port]`

## Struttura
`src/mcp_book/`: `config` (config.toml rigido, chiavi sconosciute = errore) · `models` ·
`citation` · `extract` (PyMuPDF, OCR Tesseract, TOC, tabelle pdfplumber) · `margins`
(intestazioni e piè di pagina ripetuti: rimossi e, senza TOC, usati come capitoli) · `chunking`
(sezioni dal TOC, chunk con overlap) · `embeddings` (sentence-transformers, `hash` per i
test, `none`) · `store` (unico SQLite: tabelle, FTS5, vec0) · `search` (BM25 + KNN, RRF) ·
`ingest` (incrementale per SHA-256) · `library` (risposte degli strumenti con citazioni) ·
`server` (MCPServer) · `cli`.

`tests/sample_pdf.py` genera il PDF di prova (TOC, page label romane/arabe, tabella con
bordi, pagina solo immagine per l'OCR); `tests/conftest.py` ha le fixture `sample_pdf`,
`indexed_config`, `library`.

## Insidie
- SDK `mcp` 2.x: `from mcp.server.mcpserver import MCPServer` (FastMCP non esiste più);
  errori per l'assistente con `ToolError`; nei test `from mcp import Client` accetta
  direttamente il server. Verifica le API nel pacchetto installato, non a memoria.
- Con il trasporto stdio lo stdout appartiene al protocollo: log e progressi solo su stderr.
- Argomenti degli strumenti = indici di pagina PDF da 1; le citazioni usano la
  `page_label` stampata quando esiste. Non confonderli.
- `chunks.id` è AUTOINCREMENT di proposito: un `chunk_id` vecchio non deve puntare a
  testo diverso dopo una re-ingestione.
- La tabella `vec_chunks` registra modello e dimensione: cambiare `embedding.model`
  richiede `ingest --rebuild`. Il filtro `book_id IN (...)` su vec0 richiede
  sqlite-vec >= 0.1.9 (partition key).
- Se il modello non si carica, l'ingestione si ferma con un messaggio e la ricerca
  ripiega su BM25 con una `note`: mantenere questo comportamento.
- `device = "auto"` passa `None` a sentence-transformers (ordine: cuda, mps, cpu).
- FTS5 non ha stemmer italiano: `search.stem` usa prefissi senza vocale finale.
- Il Python di sistema di macOS può non caricare estensioni SQLite: usare il Python
  gestito da uv (`uv sync --managed-python`).

## Convenzioni
- Ogni nuova logica con test in `tests/`; nessun test deve scaricare modelli o usare la
  rete: usare `HashEmbedder` / `backend = "hash"`. I test OCR usano `requires_tesseract`.
- Commit in inglese, imperativo, sentence case, senza prefisso; corpo a ~72-80 colonne
  che spiega il perché. Un commit per passo logico.
- **Mai** trailer `Co-Authored-By` né attribuzioni AI nei commit o nelle PR.
- PDF, indice (`data/`, `*.sqlite`) e `library/` non vanno mai committati: contenuti
  protetti da copyright, solo uso personale.
- Prima di dichiarare finito: pytest, ruff check, ruff format --check.
