# 📚 Advanced RAG Tutorials

> A comprehensive collection of production-ready notebooks covering cutting-edge Retrieval-Augmented Generation (RAG) techniques

---

## 🎯 Overview

This repository provides hands-on tutorials for implementing advanced RAG systems, from foundational techniques to production-scale deployments. Each notebook demonstrates practical implementations using industry-standard frameworks (LangChain, LlamaIndex) and includes side-by-side comparisons to help you choose the right approach for your use case.

**What you'll learn:**
- Query optimization and expansion techniques
- Hybrid retrieval strategies combining dense and sparse methods
- Intelligent reranking and document filtering
- Adaptive retrieval systems that dynamically select strategies
- Production-scale sparse retrieval with Milvus
- Context enrichment for improved answer quality

**Target audience:** ML engineers, data scientists, and developers building RAG applications

---

## ✨ Key Features

- **Multiple Framework Implementations**: Side-by-side LangChain and LlamaIndex examples
- **Production-Ready Code**: Complete, runnable implementations with real datasets
- **Comprehensive Comparisons**: Performance metrics and strategy comparisons
- **Modern Tech Stack**: OpenAI, FAISS, ChromaDB, Milvus, Cohere
- **Best Practices**: Structured outputs, error handling, and evaluation metrics

---

## 🚀 Learning Path

The notebooks follow a logical progression from foundational to advanced techniques:

```
Foundation → Enhancement → Refinement → Adaptation → Production
    ↓            ↓            ↓            ↓            ↓
  HyDe &      Reranking    Query        Adaptive    Sparse
  Fusion                   Transform    Retrieval   Embeddings
```

**Recommended order:**
1. Start with **HyDe** and **Fusion Retrieval** to understand query expansion and hybrid search
2. Master **Reranking** to improve result quality
3. Explore **Query Transformations** for complex query handling
4. Learn **Adaptive Retrieval** for intelligent strategy selection
5. Implement **Context Enrichment** for better answer coherence
6. Deploy **Sparse Embeddings** for production-scale systems

---

## 📖 Notebooks

### 1️⃣ Hypothetical Document Embeddings (HyDe)
**File:** `01_HyDe.ipynb`

Addresses the semantic gap between queries and documents by generating hypothetical answers before retrieval.

**Key Concepts:**
- Query expansion using LLM-generated hypothetical documents
- Retrieval based on hypothetical answers vs. raw queries
- Performance comparison: Standard RAG vs. HyDE RAG

**Implementations:**
- Custom `HyDERetriever` class
- LangChain `HypotheticalDocumentEmbedder`
- LlamaIndex `HyDEQueryTransform`

**Tech Stack:** OpenAI (GPT-4o-mini), FAISS, LangChain, LlamaIndex

---

### 2️⃣ Fusion Retrieval
**File:** `02_fusion_retrieval.ipynb`

Combines vector-based (semantic) and BM25 (keyword) retrieval for robust, production-ready search.

**Key Concepts:**
- Hybrid retrieval: dense (embeddings) + sparse (BM25)
- Score normalization and configurable fusion weights (alpha parameter)
- Reciprocal Rank Fusion (RRF) for combining multiple rankings

**Implementations:**
- Custom `fusion_retrieval()` with normalized score fusion
- LangChain `EnsembleRetriever` with BM25 + vector retriever
- ChromaDB hybrid search integration
- LlamaIndex QueryFusionRetriever with alpha tuning

**Tech Stack:** rank-bm25, sentence-transformers, FAISS, ChromaDB, LangChain

---

### 3️⃣ Reranking Strategies
**File:** `03_reranking.ipynb`

Two-stage retrieval: fast initial fetch followed by intelligent reranking for maximum relevance.

**Key Concepts:**
- LLM-based reranking with structured relevance scores (1-10 scale)
- Cross-Encoder reranking for semantic similarity
- Document compression and filtering
- Commercial reranking APIs (Cohere)

**Implementations:**
- Custom `RatingScore` Pydantic model for LLM scoring
- Cross-Encoder using `ms-marco-MiniLM-L-6-v2`
- LangChain `ContextualCompressionRetriever` with Cross-Encoder compressor
- LlamaIndex `CohereRerank` and `SentenceTransformerRerank` postprocessors

**Tech Stack:** sentence-transformers, Cohere API, LangChain, LlamaIndex

---

### 4️⃣ Query Transformations
**File:** `04_query_transformations.ipynb`

Comprehensive guide to reformulating queries for improved retrieval across multiple strategies.

**Key Concepts:**
- **Query Rewriting**: Clarifying and expanding user queries
- **Step-Back Prompting**: Abstracting to higher-level questions for broader context
- **Sub-Query Decomposition**: Breaking complex questions into 2-4 simpler queries
- **Multi-Query Generation**: Creating query variations for comprehensive coverage
- **Query Routing**: Directing queries to appropriate data sources
- **Contextual Enhancement**: Incorporating conversation history and user profiles
- **Query Translation**: Converting natural language to structured filters

**Implementations:**
- LangChain: `MultiQueryRetriever`, `SelfQueryRetriever`, custom routing
- LlamaIndex: `HyDEQueryTransform`, `StepDecomposeQueryTransform`, `SubQuestionQueryEngine`, `RouterQueryEngine`
- E-commerce search example with metadata filtering

**Tech Stack:** OpenAI API, LangChain, LlamaIndex, scikit-learn

---

### 5️⃣ Adaptive Retrieval
**File:** `05_adaptive_retrieval.ipynb`

Intelligent system that classifies queries and dynamically selects optimal retrieval strategies.

**Key Concepts:**
- Query classification into 4 types: Factual, Analytical, Opinion, Contextual
- Strategy-specific retrieval and ranking pipelines
- LLM-enhanced document ranking at each stage
- Complete `AdaptiveRAG` system

**Query-Specific Strategies:**
- **Factual**: Enhanced query + LLM ranking for precision
- **Analytical**: Sub-query generation + diversity selection
- **Opinion**: Viewpoint identification + opinion-based selection
- **Contextual**: User context incorporation + personalized ranking

**Implementations:**
- `QueryClassifier` with structured output (Pydantic)
- Four custom retrieval strategy classes
- `AdaptiveRetriever` and `PydanticAdaptiveRetriever` orchestrators

**Tech Stack:** FAISS, LangChain, OpenAI API, Pydantic

---

### 6️⃣ Context Enrichment with Window Expansion
**File:** `06_context_enrichment_window.ipynb`

Enhances isolated retrieved chunks by adding surrounding context windows for improved coherence.

**Key Concepts:**
- Chunk-level metadata indexing with chronological indices
- Context window expansion (N chunks before/after)
- Overlap-aware chunk concatenation
- Coherence improvement through neighboring context

**Implementations:**
- `get_chunk_by_index()` for targeted chunk retrieval
- `retrieve_with_context_overlap()` for context-enriched retrieval
- Configurable context window size
- Before/after comparison with real examples

**Tech Stack:** FAISS, LangChain, OpenAI embeddings

---

### 7️⃣ Advanced Sparse Embeddings (BM25 & SPLADE)
**File:** `adv_sparse_embeddings.ipynb`

Production-ready sparse retrieval implementation with Milvus vector database.

**Key Concepts:**
- **BM25**: Statistical keyword matching (3,432 vocabulary terms)
- **SPLADE**: Neural sparse embeddings using BERT (30,522 vocabulary terms)
- **Hybrid Search**: Combining BM25 + SPLADE with Reciprocal Rank Fusion
- **Milvus Integration**: Production-scale sparse vector storage and indexing

**Dataset:** Climate Fever (1,535 climate claims)

**Implementations:**
- BM25 fitting with corpus IDF calculations
- SPLADE neural model (`naver/splade-cocondenser-ensembledistil`)
- Milvus schema with `SPARSE_FLOAT_VECTOR` and `SPARSE_INVERTED_INDEX`
- `perform_search()` for BM25, SPLADE, and hybrid queries
- RRF ranking with k=60 smoothing parameter

**Decision Guide:**
- **BM25**: Fast, exact keyword matches, interpretable
- **SPLADE**: Semantic understanding, synonym handling, expansion
- **Hybrid**: Production-ready, balanced performance, best of both

**Score Ranges:**
- BM25: 0-30 (typical)
- SPLADE: 0-50+ (typical)
- RRF: (0, 0.0167] for k=60

**Tech Stack:** Milvus, pymilvus, SPLADE model, datasets, nltk, BM25

---

## 🛠️ Technologies & Frameworks

| Category | Technologies |
|----------|-------------|
| **LLM Frameworks** | LangChain (LCEL, chains, retrievers), LlamaIndex (query engines, postprocessors) |
| **Vector Databases** | FAISS (in-memory), ChromaDB (embedded), Milvus (production-scale) |
| **Sparse Retrieval** | BM25 (statistical), SPLADE (neural sparse vectors), RRF (rank fusion) |
| **Reranking** | Cross-Encoders (`ms-marco-MiniLM-L-6-v2`), Cohere Rerank API |
| **LLM Providers** | OpenAI (GPT-4o, GPT-4o-mini), Ollama (local), Cohere |
| **Embeddings** | OpenAI embeddings, sentence-transformers, SPLADE |
| **Tools** | Pydantic (structured outputs), pymupdf (PDF processing), deepeval (evaluation) |

---

## 📋 Prerequisites

- **Python 3.10+** (required for LangChain 1.2.0+)
- **OpenAI API Key** (for embeddings and LLM calls)
- **Docker** (optional, for Milvus in `adv_sparse_embeddings.ipynb`)
- **CUDA** (optional, for GPU acceleration with FAISS/Milvus)
- **Hugging Face Access** (for sentence-transformers and datasets)

---

## ⚡ Installation & Setup

### 1. Clone the repository (if applicable)
```bash
git clone <repository-url>
cd advanced-rag-tutorials
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
# Copy the example .env file
cp .env.example .env

# Edit .env and add your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# COHERE_API_KEY=your_cohere_api_key_here (optional)
```

### 4. Quick start example
```python
# Open any notebook in Jupyter
jupyter notebook 01_HyDe.ipynb

# Or use JupyterLab
jupyter lab
```

### 5. For Milvus (sparse embeddings notebook)
```bash
# Start Milvus using Docker
docker compose up -d

# Or use standalone script
bash standalone_embed.sh start
```

---

## 📂 Additional Resources

### Evaluation Framework
The `evaluation/` folder contains notebooks for assessing RAG system performance:
- `evaluation_deep_eval.ipynb` - DeepEval framework integration
- `evaluation_grouse.ipynb` - GROUSE evaluation metrics
- `define_evaluation_metrics.ipynb` - Custom metric definitions

### Utilities
The `utils/` folder provides shared helper functions:
- PDF processing and text extraction
- Embedding utilities
- Common evaluation functions

### Sample Data
The `data/` folder includes:
- `Understanding_Climate_Change.pdf` - Sample document used across tutorials

### Additional Content
The `extras/` folder contains supplementary tutorials on vector indexing and specialized Qdrant implementations.

---

## 🎓 Next Steps

1. **Start with the basics**: Work through notebooks 01-03 to build foundational understanding
2. **Experiment with parameters**: Try different alpha values, reranking thresholds, context window sizes
3. **Evaluate your changes**: Use the evaluation notebooks to measure performance
4. **Build your own RAG system**: Combine techniques from multiple notebooks
5. **Deploy to production**: Use the Milvus sparse embeddings approach for scale

---

## 💡 Tips for Success

- **Run notebooks in order** for the first time to understand progression
- **Read the inline comments** - they explain design decisions and trade-offs
- **Compare framework implementations** (LangChain vs LlamaIndex) to find your preference
- **Monitor API costs** when using OpenAI - consider using smaller models for experimentation
- **Check prerequisites** before running each notebook (some require specific API keys or Docker)

---

**Happy learning! 🚀**


