FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ita tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    HF_HOME=/data/huggingface \
    MCP_BOOK_CONFIG=/app/docker/config.toml \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only torch first: the PyPI wheel for Linux pulls several GB of CUDA libraries.
RUN uv venv /opt/venv \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY docker ./docker
RUN uv pip install .

RUN useradd --create-home --uid 1000 reader && mkdir -p /data && chown reader /data
USER reader

VOLUME ["/data"]
EXPOSE 8765

ENTRYPOINT ["mcp-book"]
CMD ["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
