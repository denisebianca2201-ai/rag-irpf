# Ask the Lion ("Pergunte ao Leão")

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about the Brazilian individual income tax (IRPF), based exclusively on the official Receita Federal **"Perguntas e Respostas"** document. Every answer cites the questions it came from.

Built with Python and LangChain. Generation uses Claude (Anthropic API); embeddings run locally (multilingual E5); the vector store is Chroma; the interfaces are a CLI and a FastAPI service.

> Informational tool only. It does not replace the Receita Federal or a tax professional.

---

## How it works

```
                        indexing (once)
  PDF ──> split per numbered question ──> embeddings (local E5) ──> Chroma (chroma_db/)

                        asking (every question)
  question
     │
     ▼
  [1 input guardrails]  length, control chars, prompt injection, PII redaction
     │
     ▼
  [2 retrieval]         top-K most similar chunks from Chroma (with relevance scores)
     │
     ▼
  [3 relevance gate]    chunks below MIN_RELEVANCE dropped; none left -> refuse, LLM NOT called
     │
     ▼
  [4 LLM (Claude)]      answers only from the retrieved context, cites [Q123]
     │
     ▼
  [5 output guardrails] citation must match a retrieved chunk, prompt-leak check, disclaimer
     │
     ▼
  answer + sources
```

### Indexing
1. `download` fetches the PDF and verifies it really is a PDF.
2. `index` extracts the text with pypdf and splits it **per numbered question** (`NNN — question text`). Each question plus its answer is one chunk, with metadata: question number, page and chapter. Answers longer than `CHUNK_SIZE` are split further, keeping the metadata on every piece.
3. Chunks are embedded locally with `intfloat/multilingual-e5-base` (with the `passage:` / `query:` prefixes E5 requires) and stored in a persistent Chroma collection using cosine distance. Indexing rebuilds the collection from scratch.

### Guardrails (defense in depth)
| Layer | What it does | Refusal reason |
|---|---|---|
| Input | Rejects empty or too-long questions; strips control characters; blocks prompt-injection patterns (English and Portuguese); redacts CPF, CNPJ, e-mail, phone and bank account **before** anything reaches the LLM or the logs | `empty`, `too_long`, `injection` |
| Relevance gate | Drops chunks scoring below `MIN_RELEVANCE`; if none remain the LLM is never called. This blocks off-topic questions | `no_context` |
| Prompt | Retrieved text is delimited and treated as untrusted data; answer only from context; reply `NO_ANSWER` if the context is insufficient; refuse help with tax evasion or falsified declarations | `no_context` |
| Output | The answer must cite at least one retrieved question (`[Q123]`); invented citations are stripped; a hidden canary string detects system-prompt leakage; a fixed disclaimer is appended | `output_check` |
| API | Request body size limit, per-IP rate limit, CORS allow-list, LLM timeout | HTTP 413 / 429 |

Refusals are not errors: they return HTTP 200 with `refused: true` and a `reason` code so clients can handle them. Regex-based injection detection is heuristic and not exhaustive; the other layers cover what it misses.

---

## Project layout

```
src/ask_the_lion/
  config.py           Settings loaded from .env (pydantic-settings)
  download.py         PDF download with validation
  embeddings.py       Local HuggingFace embeddings (+ E5 prefixes)
  ingest.py           PDF -> per-question chunks -> Chroma
  rag.py              Prompt, LLM and the full guarded pipeline (RagPipeline)
  api.py              FastAPI app factory
  cli.py              Typer CLI
  guardrails/
    input.py          Input validation, injection detection, PII redaction
    retrieval.py      Relevance filter
    output.py         Citation / leak checks
    policies.py       Refusal messages, disclaimer, reason codes
tests/                71 offline tests (see "Tests")
data/                 Downloaded PDF (git-ignored)
chroma_db/            Vector index (git-ignored)
```

---

## Setup

Requirements: Python 3.11+, an Anthropic API key with credit, ~2 GB of free disk for the dependencies and the embedding model.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch is enough and much smaller than the default CUDA build
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -e ".[dev,local-embeddings]"

cp .env.example .env
```

Edit `.env` and set at least:

```
ANTHROPIC_API_KEY=sk-ant-...
PDF_URL=https://www.gov.br/receitafederal/pt-br/centrais-de-conteudo/publicacoes/perguntas-e-respostas/dirpf/p-r-irpf-2026-v1-00-2026-04-23.pdf
```

Notes:
- The PDF URL must point to the file itself, **without** the trailing `/view` that the gov.br page adds. The URL changes every tax year; check the Receita Federal site for the new one.
- Get an API key at the Anthropic Console (Settings → API Keys). The account needs credit (Plans & Billing), otherwise requests fail with "credit balance is too low". Each question costs a few cents at most.
- If you only want to run the tests, `pip install -e ".[dev]"` is enough (no torch, no model download).

## Usage

```bash
ask-the-lion download     # saves data/perguntas_respostas.pdf
ask-the-lion index        # builds chroma_db/ (first run downloads the ~1 GB embedding model)
                          # prints: Indexed N chunks

ask-the-lion ask "Quem está obrigado a declarar o imposto de renda?"
ask-the-lion chat         # interactive; empty line or Ctrl-D to quit
```

Run `index` again only when the PDF, the embedding model, `CHUNK_SIZE` or `CHUNK_OVERLAP` change.

### API

```bash
uvicorn --factory ask_the_lion.api:create_app
```

Interactive docs at http://localhost:8000/docs.

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Quem está obrigado a declarar?"}'
```

Response:

```json
{
  "answer": "… [Q1] …\n\n_Informational only, …_",
  "sources": [{"question_number": 1, "page": 23, "excerpt": "…"}],
  "refused": false,
  "reason": null
}
```

When refused: `refused: true`, `sources: []`, and `reason` is one of `empty | too_long | injection | no_context | output_check`. `GET /health` returns `{"status": "ok"}`.

---

## Configuration

All settings come from environment variables or `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Claude model used for generation |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Local embedding model (`…-e5-small` is lighter) |
| `PDF_URL` | – | Direct URL of the official PDF |
| `TOP_K` | `8` | Chunks retrieved per question |
| `MIN_RELEVANCE` | `0.84` | Minimum relevance score (0–1) for a chunk to be used |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `2500` / `250` | Splitting of long answers (needs re-index) |
| `MAX_QUESTION_CHARS` | `1000` | Longer questions are refused |
| `RATE_LIMIT` | `10/minute` | Per-IP limit on `POST /ask` |
| `CORS_ORIGINS` | `[]` | JSON list of allowed origins |

### Calibrating `MIN_RELEVANCE`
E5 scores are high even for unrelated text. With the 2026 document, in-scope questions scored about 0.86–0.90 and off-topic ones about 0.77–0.81, hence 0.84. The margin is small: if legitimate questions get refused with `no_context`, lower it slightly (e.g. 0.83); if off-topic questions get answered, raise it. Re-calibrate if you change the embedding model.

---

## Tests

```bash
pytest
```

71 tests, no network, API key or torch required. They use a deterministic fake embedding model, a fake LLM that records the prompts it receives, and a real temporary Chroma store. They cover:
- per-question splitting, metadata and re-indexing;
- every guardrail (injection strings in both languages, PII redaction, relevance threshold, citation checks, prompt-leak detection);
- the full pipeline, e.g. "off-topic never calls the LLM" and "a CPF never reaches the prompt";
- the API contract, 422/413/429 responses and refusals as HTTP 200;
- PDF download (mocked transport) and the CLI.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Errno 122 Disk quota exceeded` during `pip install` | Per-user quota. Install CPU-only torch (see Setup), use `--no-cache-dir`, run `pip cache purge`. |
| `HTTP 403` on download | Use the direct PDF URL (no `/view`), or download in a browser to `data/perguntas_respostas.pdf` and skip `download`. |
| `No numbered questions found` | The PDF layout changed; adjust `QUESTION_START` in `ingest.py`. |
| `temperature is not supported for claude-sonnet-5-5` | Do not pass a custom temperature; the code already omits it. |
| `credit balance is too low` | Add credit in the Anthropic Console (Plans & Billing). |
| Good questions refused with `no_context` | Lower `MIN_RELEVANCE`. |
| Answers feel truncated | Increase `CHUNK_SIZE` / `TOP_K` and re-run `index`. |

## Limitations
- Answers are only as good as the source PDF and the retrieval; the nearest chunk is not always the most relevant question.
- The question regex assumes the current PDF layout.
- Injection detection is heuristic.
- English-only repository: code, prompts, tests and messages are in English, while the source document and the conversations are in Portuguese. Refusal messages are in English; translate them in `guardrails/policies.py` for a Portuguese-facing deployment.
