# Advanced RAG with Contextual Retrieval — 2026 Edition

A comprehensive, three-part Jupyter notebook series implementing state-of-the-art Retrieval Augmented Generation (RAG) with Anthropic's Contextual Retrieval technique, prompt caching, hybrid search, reranking, and full evaluation — designed for intermediate to advanced practitioners.

---

## Overview

This project implements Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) technique end-to-end, progressively layering each enhancement so you can see the accuracy and cost impact of every addition.

| Method | Pass@5 | Pass@10 | Pass@20 | Cost / 1000 chunks |
|---|---|---|---|---|
| Baseline RAG | 81% | 87% | 90% | ~$0.50 |
| + Contextual Embeddings | 88% | 92% | 94% | ~$0.65 (with caching) |
| + Hybrid Search (BM25) | 89% | 93% | 95% | ~$0.65 |
| + Reranking | 90%+ | 94%+ | 96%+ | ~$0.65 |

**Key cost insight:** Prompt caching reduces contextualisation cost by **60–70%** (from ~$2.40 to ~$0.65 per 1,000 chunks).

---

## What's Implemented

- **Contextual Embeddings** — Claude generates a 50–100 token context for every chunk, situating it within the full document before embedding. Directly implements Anthropic's technique.
- **Prompt Caching** — `cache_control: ephemeral` sends the full document once; all subsequent chunks read from cache at a 90% discount.
- **Hybrid Search** — Combines dense vector search (Voyage AI) with BM25 keyword search via Reciprocal Rank Fusion (RRF).
- **Reranking** — Two-stage retrieval: over-retrieve candidates → rerank with HuggingFace (`ms-marco-MiniLM-L-12-v3`, free) or Cohere (`rerank-english-v3.0`).
- **MCP Integration** — Model Context Protocol client for external web and document retrieval.
- **Evaluation** — Pass@k metrics, visualisation, cost analysis, and method comparison.
- **Complete Pipeline** — `AdvancedRAGPipeline` class with feature toggles for every component.

---

## Notebook Structure

The series is split into three sequential notebooks. Run them in order in a single Jupyter session so variables carry across.

| Notebook | Contents |
|---|---|
| `context_rag_advanced_part1.ipynb` | Setup, document processing, baseline RAG, contextual embeddings, hybrid search |
| `context_rag_advanced_part2.ipynb` | HuggingFace & Cohere rerankers, MCP integration, evaluation framework |
| `context_rag_advanced_part3.ipynb` | Cost analysis, `AdvancedRAGPipeline` class, production best practices |

Each notebook also includes a standalone setup cell so it can be explored independently.

---

## Quick Start

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/sourangshupal/context-rag.git
cd context-rag

# Requires uv — install with: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

Or install from `pyproject.toml` (includes dev extras):

```bash
uv pip install -e ".[dev]"
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_key     # https://openrouter.ai/keys
VOYAGE_API_KEY=your_voyage_key             # https://www.voyageai.com/

# Optional
COHERE_API_KEY=your_cohere_key             # https://cohere.com/ (for Cohere reranker)
HF_TOKEN=your_hf_token                     # https://huggingface.co/settings/tokens (for HF Inference API reranker)
ANTHROPIC_API_KEY=your_anthropic_key       # https://console.anthropic.com/ (direct API, not needed if using OpenRouter)
```

**Minimum required to run:** `OPENROUTER_API_KEY` + `VOYAGE_API_KEY`.

### 4. Add your documents

Place PDF or Markdown files in `data/documents/`:

```
data/documents/
  ├── your_document.pdf
  ├── another_doc.md
  └── ...
```

Sample documents (`machine_learning_intro.md`, `neural_networks.md`) are included to run the notebooks immediately.

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open `context_rag_advanced_part1.ipynb` and run all cells in order.

---

## Configuration

All parameters are set in the configuration cell at the top of Part 1:

```python
# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Models
LLM_MODEL       = "openrouter/anthropic/claude-haiku-4.5"   # for contextual embeddings
EMBEDDING_MODEL = "voyage-2"                                  # Voyage AI
RERANK_MODEL_HF = "ms-marco/MiniLM-L-12-v3"                  # free HuggingFace reranker
RERANK_MODEL_COHERE = "rerank-english-v3.0"                  # Cohere reranker

# Feature toggles
USE_CONTEXTUAL    = True
USE_HYBRID_SEARCH = True
USE_RERANKING     = True
RERANKER_TYPE     = "hf"      # "hf" (free) or "cohere" (paid)
```

---

## Prompt Caching — How It Works

The project uses Anthropic's explicit prompt caching (`cache_control: ephemeral`) via OpenRouter. For every chunk of a document, the full document is sent as a cached prefix:

```
Chunk 1 → writes document to 5-min cache   (1.25× input price)
Chunk 2 → reads document from cache        (0.10× input price — 90% off)
Chunk 3 → reads document from cache        (0.10× input price — 90% off)
...
```

See [`prompt_caching.md`](./prompt_caching.md) for a complete reference on implicit vs explicit caching across all major providers, framework support, and implementation patterns.

---

## Cost Reference (April 2026)

| Component | Model | Price |
|---|---|---|
| LLM (contextual embeddings) | Claude Haiku 4.5 via OpenRouter | ~$0.80 / 1M input tokens |
| Embeddings | Voyage AI `voyage-2` | $0.10 / 1M tokens (200M free) |
| Reranker (free) | HuggingFace ms-marco-MiniLM | $0 |
| Reranker (paid) | Cohere rerank-english-v3.0 | $0.10 / 1K queries |

With prompt caching enabled, effective LLM cost for contextualisation drops to approximately **$0.08–0.10 / 1M cached tokens**.

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Runtime | Python | 3.12 |
| Package manager | uv | latest |
| LLM | Anthropic (via OpenRouter) | `anthropic==0.76.0` |
| Embeddings | Voyage AI | `voyageai==0.3.7` |
| Keyword search | rank-bm25 | `0.2.2` |
| Reranker (free) | Hugging Face Transformers | `transformers>=4.40` |
| Reranker (paid) | Cohere | `cohere==5.20.1` |
| Data | pandas, numpy | pinned in pyproject.toml |

---

## Repository Contents

```
context-rag/
├── context_rag_advanced_part1.ipynb   ← Start here
├── context_rag_advanced_part2.ipynb
├── context_rag_advanced_part3.ipynb
├── prompt_caching.md                  ← Deep-dive reference on prompt caching
├── FINAL_SUMMARY.md                   ← Project completion summary
├── pyproject.toml                     ← Dependencies (uv)
├── requirements.txt                   ← pip-compatible dependency list
├── .env.example                       ← API key template
├── opencode.json                      ← MCP server configuration
└── data/
    └── documents/
        ├── machine_learning_intro.md  ← Sample document
        └── neural_networks.md         ← Sample document
```

---

## References

- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic: Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Claude Cookbook: Contextual Embeddings](https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide)
- [Voyage AI Documentation](https://docs.voyageai.com)
- [Cohere Rerank API](https://docs.cohere.com/reference/rerank)
- [Reciprocal Rank Fusion paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

*Python 3.12 · uv · Anthropic SDK 0.76 · Voyage AI 0.3.7 · Claude Haiku 4.5*
