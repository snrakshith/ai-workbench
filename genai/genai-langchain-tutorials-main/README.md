# GenAI LangChain Tutorials

A hands-on notebook series for building RAG pipelines and GenAI applications with LangChain.

---

## Notebooks

| # | Topic |
|---|-------|
| 01 | Introduction & Fundamentals |
| 02 | Document Loaders |
| 03 | Text Splitting Strategies |
| 04 | Embeddings & Vector Representations |
| 05 | Vector Stores (FAISS, Chroma) |
| — | Simple RAG with LangChain |
| — | Local RAG with Ollama |

---

## Setup

**Prerequisites:** Python 3.11+, OpenAI API key

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Then launch Jupyter:

```bash
jupyter notebook
```

---

## Stack

- **LangChain** — chains, loaders, splitters, retrievers
- **OpenAI** — embeddings + generation
- **FAISS / Chroma** — vector stores
- **RAGAS** — RAG evaluation
- **Ollama** — local LLM option
