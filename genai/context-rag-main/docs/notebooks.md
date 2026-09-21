# Advanced RAG with Contextual Retrieval — Complete Feature Reference

All three notebooks build a production-grade RAG system incrementally. Each notebook adds a layer: Part 1 covers the retrieval foundation, Part 2 adds precision and external data, Part 3 ties it all together with cost analysis and production guidance.

---

## Complete Performance Summary (across all parts)

| Pipeline Stage | Pass@5 | Pass@10 | Pass@20 | Cost/1K chunks |
|---|---|---|---|---|
| Baseline RAG | 81% | 87% | 90% | ~$0.50 |
| + Contextual Embeddings | 88% | 92% | 94% | ~$2.40 |
| + Hybrid Search (BM25) | 89% | 93% | 95% | ~$2.40 |
| + HF Reranking | 90%+ | 94%+ | 96%+ | ~$2.40 |
| + Cohere Reranking | 90%+ | 94%+ | 96%+ | ~$3.00 |

---

# Part 1 — Setup, Document Processing & Core Retrieval

## Feature 1: Document Ingestion Pipeline

### What It Does
Loads PDF, Markdown, and plain-text files from `data/documents/` into a uniform document schema. Each document gets a `doc_id`, `original_uuid`, raw `content`, and an empty `chunks` list populated by the next stage.

### Chunking Strategy
Uses a sliding-window splitter:
- **Default chunk size**: 800 characters
- **Default overlap**: 200 characters
- Overlap preserves context across chunk boundaries — critical for queries that span topic transitions

### Why It Matters for Contextual RAG
Chunk quality directly determines retrieval quality. Too small → fragments lose meaning. Too large → embeddings become diluted averages. The 800/200 defaults hit the sweet spot for general prose. Every downstream cost (LLM context calls, embedding API calls) scales linearly with chunk count — chunking decisions have direct cost implications.

---

## Feature 2: Baseline RAG — `VectorDB` Class

### What It Does
Pure semantic search using Voyage AI's `voyage-2` embedding model and cosine similarity. Handles:
- **Batch embedding** via Voyage AI API (128 items per batch)
- **In-memory cosine similarity search** using `numpy.dot`
- **Persistent storage** via `pickle` to `data/baseline_db/vector_db.pkl`
- **Query caching** — repeated queries skip the embedding API call entirely

### Architecture
```
Query → Embed (Voyage AI) → Cosine Similarity vs all stored vectors → Top-K results
```

### Why It Matters
This is the **baseline to beat**. Every subsequent technique is measured against it. Understanding baseline weaknesses motivates each addition:

| Weakness | Root Cause | Fix |
|---|---|---|
| "This reduces latency by 40%" retrieves poorly | Chunk embedded without document context | Contextual Embeddings |
| Exact technical terms missed | Semantic similarity ≠ keyword match | Hybrid BM25 Search |
| Top-10 has irrelevant chunks | No quality re-sorting | Reranking (Part 2) |

### Performance
| Metric | Score |
|---|---|
| Pass@5 | 81% |
| Pass@10 | 87% |
| Pass@20 | 90% |
| Cost per 1000 chunks | ~$0.50 |

---

## Feature 3: Contextual Embeddings — `OpenRouterLLM` + `ContextualVectorDB`

### What It Does
Before embedding each chunk, an LLM (Claude Haiku via OpenRouter) reads the **full source document** and generates a short context description for that chunk. The context is prepended to the chunk text before embedding:

```
[Original chunk content]

[LLM-generated context: "This chunk describes the backpropagation algorithm
within the broader neural network training section..."]
```

The enriched text is then embedded, capturing **both the chunk's content and its role within the document**.

### The Core Problem It Solves
Standard RAG embeds chunks in isolation. A chunk like:

> "This approach reduces latency by 40%"

has no semantic anchor — what approach? Retrieval for "API optimization latency" will miss it. With contextual embedding, the chunk becomes anchored in the document's meaning. Retrieval works correctly.

### OpenRouter Integration
OpenRouter provides a unified OpenAI-compatible endpoint for multiple LLM providers:
```python
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
model = "anthropic/claude-haiku-4-5"  # provider/model format — no openrouter/ prefix
```

### Parallel Processing
`ThreadPoolExecutor` with 5 threads processes chunks concurrently. Each thread makes independent LLM calls. Speedup is near-linear up to API rate limits.

### Token Tracking
LLM call token usage (`prompt_tokens`, `completion_tokens`, cache tokens via `getattr` with safe defaults) is accumulated thread-safely using `threading.Lock`. Feeds cost analysis output.

### Cost Optimization: Prompt Caching
When the same full document is sent as context for each of its chunks, the document prefix hits the provider's prompt cache on chunk 2+:
- **Without caching**: ~$1.02 per million tokens
- **With caching**: ~$0.31 per million tokens (**69% savings**)
- **Net result**: 60–70% overall cost reduction per document

### Performance
| Metric | Score | vs Baseline |
|---|---|---|
| Pass@5 | 88% | +7% absolute |
| Pass@10 | 92% | +5% absolute |
| Failure rate | 8% | 38% reduction |
| Cost per 1000 chunks | ~$2.40 | 4.8× higher |

---

## Feature 4: Hybrid Search — BM25 + Reciprocal Rank Fusion

### What It Does
Combines two fundamentally different retrieval signals:

| Signal | Method | Strength |
|---|---|---|
| Semantic | Vector cosine similarity | Understands meaning and intent |
| Keyword | BM25 (TF-IDF variant) | Finds exact technical terms |

### `BM25Search` Class
Tokenizes all chunk content into an inverted index using `rank-bm25`. At query time: tokenize query → TF-IDF score all docs → return top-K with inverse-rank scores.

### Reciprocal Rank Fusion (RRF)
Combines rankings from both methods without needing score normalization:

```
RRF_score(d) = Σ [ weight × (k / (rank(d) + k)) ]
```

Where `k=60`. Default weights: 80% semantic, 20% BM25.

| Query Type | Semantic | BM25 |
|---|---|---|
| Conceptual questions | 90% | 10% |
| General knowledge | 80% | 20% |
| Technical/domain terms | 70% | 30% |
| Acronyms / exact phrases | 60% | 40% |

### Why It Matters
Semantic search misses exact jargon matches. BM25 misses paraphrases. RRF captures both failure modes at ~50ms added latency and zero additional API cost.

### Performance
| Metric | Score | vs Contextual |
|---|---|---|
| Pass@5 | 89% | +1% |
| Pass@10 | 93% | +1% |
| Cost delta | +$0 | BM25 is free |

---

# Part 2 — Reranking & MCP Integration

## Feature 5: Hugging Face Reranker — `HFReranker` Class

### What It Does
A cross-encoder reranker using `ms-marco/MiniLM-L-12-v3`. Unlike bi-encoders (which embed query and document independently), a cross-encoder sees both query and candidate simultaneously — giving much higher precision but only usable on a small candidate set.

Supports two deployment modes:
- **HF Inference API** (`hf_inference`) — ~200ms latency, no GPU needed, free tier available
- **Local model** (`local`) — ~50ms latency on GPU, no external API calls

### Two-Stage Retrieval Architecture
```
Query
  → Stage 1: Retrieve large candidate set (100 chunks) via vector/hybrid search
  → Stage 2: Rerank top 100 using cross-encoder → return top-K (precision)
```

Stage 1 optimizes **recall** (don't miss relevant chunks). Stage 2 optimizes **precision** (sort the relevant ones to the top). This separation lets each stage do what it does best.

### Why It Matters
Embeddings optimize for rough semantic similarity. Cross-encoders read the full query+document pair and score relevance directly — catching cases where a chunk is semantically adjacent but not actually answering the query. Free cost makes HF reranker the default choice.

### Performance
| Metric | Score | vs Hybrid |
|---|---|---|
| Pass@5 | 90%+ | +1% |
| Pass@10 | 94%+ | +1% |
| Cost | FREE | No change |
| Added latency | ~200ms (API) / ~50ms (local) | per query |

---

## Feature 6: Cohere Reranker — `CohereReranker` Class

### What It Does
State-of-the-art reranking using `rerank-english-v3.0` via Cohere's managed API. Same two-stage architecture as HF reranker.

### HF vs Cohere Comparison
| Factor | HF Reranker | Cohere Reranker |
|---|---|---|
| Cost | FREE | $0.10 per 1K queries |
| Latency | ~200ms (API) / ~50ms (local) | ~150ms |
| Quality | Excellent (BEIR benchmarks) | Excellent (SOTA) |
| Deployment | Self-managed or HF API | Fully managed |
| Best for | High-volume, budget-conscious | Premium precision needs |

### When to Use Cohere
High-stakes retrieval (legal, medical, financial) where marginal accuracy improvement justifies $0.10/1K query cost. For most applications, HF reranker delivers equivalent quality for free.

---

## Feature 7: Two-Stage Retrieval — `two_stage_rerank()`

### What It Does
Unified function combining any vector/hybrid retriever with any reranker:

```python
def two_stage_rerank(db, reranker, query, recall_size=100, top_k=10):
    # Stage 1: Large candidate set
    candidates = db.search(query, k=recall_size)
    # Stage 2: Rerank for precision
    reranked = reranker.rerank(query, candidates, top_k=top_k)
    return reranked
```

### Recall Size Guide
| Recall Size | Use Case |
|---|---|
| 50–100 | Standard — good balance of speed/accuracy |
| 100–200 | High precision — more candidates, better results |
| 200+ | Edge cases — rarely needed, expensive |

---

## Feature 8: MCP Integration — `MCPSearch` Class

### What It Does
Model Context Protocol (MCP) integration pattern for pulling real-time external data into the retrieval pipeline. Augments static vector database results with live information from:
- **Web search** (`search_web`) — via web-search-prime MCP server
- **GitHub search** (`search_github`) — repository code and issues
- **Documentation search** (`search_docs`) — library/framework docs

### `enhanced_search_with_mcp()` Function
Combines internal vector search with MCP results using weighted fusion:
```python
# Internal vector results: weighted at (1 - mcp_weight)
# MCP results: weighted at mcp_weight (default 0.3)
# Combined, sorted by weighted score
```

### Why It Matters
Static document collections go stale. MCP lets RAG systems answer queries that require up-to-date information (latest library versions, recent GitHub issues, live documentation) without rebuilding the vector database. MCP servers are typically free-tier.

### Architecture
```
Query
  → Internal vector search (stored knowledge)
  → MCP web/GitHub/docs search (live knowledge)
  → Weighted score fusion
  → Top-K combined results
```

---

## Feature 9: Evaluation Framework — `evaluate_retrieval()`

### What It Does
Systematic measurement of retrieval quality using the **Pass@k metric**:

```
Pass@k = (queries where at least 1 golden chunk appears in top-k) / total queries
```

Evaluates each pipeline configuration (baseline, contextual, reranked) against a test set of queries with known ground-truth chunks (`golden_chunk_uuids`).

### Why Pass@k
Pass@k measures whether the relevant chunk appears anywhere in the top-k results — not whether it's ranked #1. This matches real RAG behavior: the LLM has all top-k chunks as context and can find the answer as long as it's present.

### Evaluation Results (Part 2)
| Method | Pass@10 | Improvement |
|---|---|---|
| Baseline RAG | 87% | — |
| + Contextual Embeddings | 92% | +5% |
| + Reranking | 94%+ | +2% |

---

# Part 3 — Cost Analysis, Complete Pipeline & Production

## Feature 10: Cost Analysis — `analyze_costs()`

### What It Does
Calculates detailed cost breakdown for each RAG strategy given document count and chunk count. Uses real 2026 pricing:

| Service | Price |
|---|---|
| Voyage AI embeddings | $0.05 / 1M tokens |
| OpenRouter LLM input | $0.30 / 1M tokens |
| OpenRouter LLM output | $0.01 / 1M tokens |
| HF Reranker | FREE |
| Cohere Reranker | $0.10 / 1K queries |

### Two Cost Scenarios Modeled
- **Without prompt caching** (worst case): Full document re-sent for every chunk
- **With prompt caching** (typical): Document cached after first chunk, 90% discount on subsequent reads

### Cost Insight
Prompt caching is the single largest cost lever. Without it, contextual embeddings cost ~3–5× more than baseline. With it, the premium drops to ~2–3×. The `cache_hit_rate` parameter lets you model different cache efficiency scenarios.

---

## Feature 11: `AdvancedRAGPipeline` — Unified End-to-End Class

### What It Does
Single class that assembles the full pipeline with all features. Constructor accepts feature flags:

```python
pipeline = AdvancedRAGPipeline(
    use_contextual=True,   # Contextual embeddings via OpenRouter
    use_hybrid=True,       # BM25 + RRF hybrid search
    use_reranking=True,    # Two-stage reranking
    reranker_type="hf",    # "hf" or "cohere"
    use_mcp=False,         # External MCP data sources
)
pipeline.load(folder_path="data/documents")
results = pipeline.search(query, k=10)
```

### `search()` Method Logic
1. Contextual or baseline vector search
2. BM25 search (if hybrid enabled)
3. RRF fusion (if hybrid enabled)
4. MCP augmentation (if MCP enabled)
5. Reranking (if reranking enabled)
6. Return top-K with full provenance metadata

### Why It Matters
Removes the need to manually wire together 5–6 separate components. Single `search()` call runs the full configured pipeline. Feature flags enable ablation testing — toggle any component on/off to measure its individual contribution.

---

## Feature 12: When to Use Each Method

| Method | Use When | Avoid When |
|---|---|---|
| **Baseline RAG** | Simple docs, budget constrained, fast prototyping | Complex docs, context-dependent queries |
| **Contextual Embeddings** | Long/complex docs, accuracy > cost, domain-specific content | Short docs, budget constraints, quick tests |
| **Hybrid Search** | Technical queries, exact terms/jargon, code/docs | Conceptual questions, general knowledge, cross-language |
| **HF Reranking** | High precision needed, complex/ambiguous queries, free budget | Simple queries, latency-sensitive (<150ms) |
| **Cohere Reranking** | Premium precision, legal/medical, low query volume | High query volume, tight budget |
| **MCP Integration** | Live data needed, rapidly changing sources | Static knowledge bases, offline environments |

---

## Feature 13: Production Deployment Checklist

### Infrastructure
- Managed vector DB (Pinecone, Weaviate, Milvus) instead of pickle
- Load balancer + auto-scaling for query throughput
- CDN for static assets

### Security
- API keys in secrets manager (AWS Secrets Manager, HashiCorp Vault)
- Encryption at rest and in transit
- Rate limiting per user/API key
- Input validation and sanitization

### Reliability
- Retry logic with exponential backoff on all API calls
- Circuit breakers for external providers
- Fallback provider chain (e.g., Cohere fails → HF reranker)
- Health check endpoints
- Blue-green deployments

### Monitoring
- Track query latency (p50, p95, p99)
- Monitor API costs and token usage per request
- Cache hit rate tracking
- Structured logging
- A/B test configurations against Pass@k ground truth

---

## Feature 14: Common Pitfalls & Solutions

| Pitfall | Problem | Solution |
|---|---|---|
| Over-chunking | Too-small chunks lose context | Use 800–1200 chars with 20% overlap |
| Insufficient overlap | Information lost at boundaries | Use 20–25% overlap |
| No prompt caching | 3–5× higher LLM costs | Use `cache_control` header (Anthropic SDK) or OpenRouter automatic caching |
| Too-large recall size | Slow reranking, unnecessary cost | Use 50–100 candidates for most cases |
| No query caching | Repeated queries waste time and money | Cache query embeddings in-memory |
| Not monitoring costs | Surprise billing spikes | Track token usage, set budget alerts |
| Poor error handling | Crashes on API failures | Retries + fallbacks + graceful degradation |
| No evaluation dataset | Can't measure if changes help | Maintain golden query set, track Pass@k over time |
| Hardcoded config | Can't tune for different use cases | Feature flags + environment variables |

---

## Key Design Decisions (All Parts)

### Voyage AI for Embeddings
`voyage-2` outperforms `text-embedding-3-small` on retrieval benchmarks for long-form documents. Separate embedding provider from LLM provider reduces single-vendor dependency.

### OpenRouter Instead of Direct Anthropic API
Single API key and endpoint gives access to multiple LLM providers. Easy model swapping (Claude Haiku → Mistral → GPT-4o-mini) by changing one variable.

### HF Reranker as Default
`ms-marco/MiniLM-L-12-v3` delivers equivalent quality to Cohere at zero cost. For 90%+ of use cases, the free option is the right default.

### Feature Flags Architecture
`USE_CONTEXTUAL`, `USE_HYBRID_SEARCH`, `USE_RERANKING` allow ablation — measure the individual contribution of each technique against your own data before committing to its cost.

### Pickle-Based Persistence
Simple, zero-dependency local persistence for development. In production, replace with a managed vector database for horizontal scaling and concurrent access.

---

## Quick Reference

### Configuration Toggles
```python
USE_CONTEXTUAL = True     # Contextual embeddings via OpenRouter LLM
USE_HYBRID_SEARCH = True  # BM25 + Reciprocal Rank Fusion
USE_RERANKING = True      # Two-stage reranking
RERANKER_TYPE = "hf"      # "hf" (free) or "cohere" ($0.10/1K queries)
```

### Performance Targets
| Configuration | Pass@10 | Cost/1K chunks |
|---|---|---|
| Baseline | 87% | ~$0.50 |
| + Contextual | 92% | ~$2.40 |
| + Hybrid | 93% | ~$2.40 |
| + HF Rerank | 94%+ | ~$2.40 |
| + Cohere Rerank | 94%+ | ~$3.00 |

### API Keys Required
```dotenv
VOYAGE_API_KEY=        # Required for all embeddings
OPENROUTER_API_KEY=    # Required for contextual embeddings
COHERE_API_KEY=        # Required for Cohere reranking
HF_TOKEN=             # Optional — HF free tier without token
```
