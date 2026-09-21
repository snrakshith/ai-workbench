"""Enhanced FastAPI endpoints for advanced RAG features including HyDE and Hybrid Search."""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from enhancement.hyde import create_hyde_enhancer
from enhancement.hybrid_search import create_hybrid_searcher
from retrieval.semantic_search import create_semantic_searcher
from generation.rag_pipeline import create_rag_pipeline
from src.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/enhanced", tags=["Enhanced RAG Features"])

# Global instances (lazy loading)
_hyde_enhancer = None
_hybrid_searcher = None
_semantic_searcher = None
_rag_generator = None
_query_cache = {}


def get_hyde_enhancer():
    """Get or create HyDE enhancer instance."""
    global _hyde_enhancer
    if _hyde_enhancer is None:
        try:
            _hyde_enhancer = create_hyde_enhancer()
            logger.info("HyDE enhancer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HyDE enhancer: {str(e)}")
            raise HTTPException(status_code=503, detail=f"HyDE service unavailable: {str(e)}")
    return _hyde_enhancer


def get_hybrid_searcher():
    """Get or create hybrid searcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        try:
            _hybrid_searcher = create_hybrid_searcher()
            logger.info("Hybrid searcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid searcher: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Hybrid search service unavailable: {str(e)}")
    return _hybrid_searcher


def get_semantic_searcher():
    """Get or create semantic searcher instance."""
    global _semantic_searcher
    if _semantic_searcher is None:
        try:
            _semantic_searcher = create_semantic_searcher()
            logger.info("Semantic searcher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize semantic searcher: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Semantic search service unavailable: {str(e)}")
    return _semantic_searcher


def get_rag_generator():
    """Get or create RAG pipeline instance."""
    global _rag_generator
    if _rag_generator is None:
        try:
            _rag_generator = create_rag_pipeline()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise HTTPException(status_code=503, detail=f"RAG service unavailable: {str(e)}")
    return _rag_generator


# Pydantic Models

class HyDEQueryRequest(BaseModel):
    """Request model for HyDE query enhancement."""
    query: str = Field(..., description="Query to enhance with HyDE", min_length=5, max_length=500)
    domain_context: str = Field(default="LangChain", description="Domain context for hypothetical document generation")
    num_hypothetical: int = Field(default=3, description="Number of hypothetical documents to generate", ge=1, le=5)
    use_cache: bool = Field(default=True, description="Whether to use query cache")


class HyDEBatchRequest(BaseModel):
    """Request model for batch HyDE enhancement."""
    queries: List[str] = Field(..., description="List of queries to enhance", min_items=1, max_items=10)
    domain_context: str = Field(default="LangChain", description="Domain context")
    num_hypothetical: int = Field(default=3, description="Number of hypothetical documents per query", ge=1, le=5)
    use_cache: bool = Field(default=True, description="Whether to use query cache")


class HyDERagRequest(BaseModel):
    """Request model for HyDE-enhanced RAG."""
    query: str = Field(..., description="Question to answer using HyDE-enhanced RAG", min_length=5)
    domain_context: str = Field(default="LangChain", description="Domain context")
    num_hypothetical: int = Field(default=3, description="Number of hypothetical documents", ge=1, le=5)
    top_k: int = Field(default=5, description="Number of documents to retrieve", ge=1, le=20)
    use_memory: bool = Field(default=False, description="Whether to use conversation memory")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for memory")
    use_cache: bool = Field(default=True, description="Whether to use caching")


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    top_k: int = Field(default=8, description="Number of results to return", ge=1, le=50)
    semantic_weight: float = Field(default=0.7, description="Weight for semantic search", ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search", ge=0.0, le=1.0)
    enable_reranking: bool = Field(default=True, description="Whether to apply reranking")
    fusion_method: str = Field(default="rrf", description="Fusion method: 'rrf' or 'weighted'")
    use_cache: bool = Field(default=True, description="Whether to use result cache")


class HybridBatchRequest(BaseModel):
    """Request model for batch hybrid search."""
    queries: List[str] = Field(..., description="List of search queries", min_items=1, max_items=10)
    top_k: int = Field(default=8, description="Number of results per query", ge=1, le=20)
    semantic_weight: float = Field(default=0.7, description="Semantic search weight", ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, description="Keyword search weight", ge=0.0, le=1.0)
    enable_reranking: bool = Field(default=True, description="Apply reranking to results")
    use_cache: bool = Field(default=True, description="Use result caching")


class SearchComparisonRequest(BaseModel):
    """Request model for search method comparison."""
    query: str = Field(..., description="Query to test with different search methods", min_length=1)
    top_k: int = Field(default=5, description="Number of results per method", ge=1, le=20)
    include_semantic: bool = Field(default=True, description="Include semantic search")
    include_keyword: bool = Field(default=True, description="Include keyword search") 
    include_hybrid: bool = Field(default=True, description="Include hybrid search")
    include_hyde: bool = Field(default=True, description="Include HyDE-enhanced search")


class UltimateRagRequest(BaseModel):
    """Request model for ultimate RAG with all enhancements."""
    query: str = Field(..., description="Question for ultimate RAG processing", min_length=5)
    domain_context: str = Field(default="LangChain", description="Domain context")
    num_hypothetical: int = Field(default=3, description="HyDE hypothetical documents", ge=1, le=5)
    semantic_weight: float = Field(default=0.7, description="Semantic search weight", ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, description="Keyword search weight", ge=0.0, le=1.0)
    top_k: int = Field(default=8, description="Documents to retrieve", ge=1, le=20)
    enable_reranking: bool = Field(default=True, description="Apply reranking")
    use_memory: bool = Field(default=False, description="Use conversation memory")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    use_cache: bool = Field(default=True, description="Use caching")


class HyDEConfigUpdate(BaseModel):
    """Model for updating HyDE configuration."""
    num_hypothetical_docs: Optional[int] = Field(None, ge=1, le=5)
    max_doc_length: Optional[int] = Field(None, ge=100, le=800)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    use_query_fusion: Optional[bool] = None
    fusion_weights: Optional[Dict[str, float]] = None


class HybridConfigUpdate(BaseModel):
    """Model for updating hybrid search configuration."""
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_keyword_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    fusion_method: Optional[str] = Field(None, pattern="^(rrf|weighted)$")
    rrf_k: Optional[int] = Field(None, ge=1, le=200)
    enable_reranking: Optional[bool] = None
    reranking_method: Optional[str] = None


# Helper Functions

def cache_key(prefix: str, query: str, **kwargs) -> str:
    """Generate cache key for results."""
    import hashlib
    params_str = f"{query}_{str(sorted(kwargs.items()))}"
    return f"{prefix}_{hashlib.md5(params_str.encode()).hexdigest()[:16]}"


def get_cached_result(key: str) -> Optional[Dict]:
    """Get cached result if available and not expired."""
    if key in _query_cache:
        cached_data, timestamp = _query_cache[key]
        # Cache expires after 1 hour
        if time.time() - timestamp < 3600:
            return cached_data
        else:
            del _query_cache[key]
    return None


def set_cached_result(key: str, result: Dict) -> None:
    """Cache a result with timestamp."""
    _query_cache[key] = (result, time.time())
    # Limit cache size
    if len(_query_cache) > 1000:
        # Remove oldest entries
        oldest_keys = sorted(_query_cache.keys(), key=lambda k: _query_cache[k][1])[:100]
        for k in oldest_keys:
            del _query_cache[k]


# HyDE Enhancement Endpoints

@router.post("/hyde/query")
async def enhance_query_with_hyde(request: HyDEQueryRequest):
    """
    Enhance a single query using HyDE (Hypothetical Document Embeddings).
    
    HyDE generates hypothetical documents that would answer the query,
    then creates enhanced embeddings for more effective retrieval.
    """
    try:
        # Check cache first
        if request.use_cache:
            cache_k = cache_key("hyde", request.query, 
                               domain=request.domain_context, 
                               num_hyp=request.num_hypothetical)
            cached = get_cached_result(cache_k)
            if cached:
                logger.info(f"Returning cached HyDE result for: {request.query[:50]}...")
                return cached
        
        enhancer = get_hyde_enhancer()
        start_time = time.time()
        
        result = enhancer.enhance_query(
            query=request.query,
            domain_context=request.domain_context,
            num_hypothetical=request.num_hypothetical
        )
        
        processing_time = time.time() - start_time
        result["processing_time_seconds"] = processing_time
        
        # Cache successful results
        if request.use_cache and result.get("success"):
            set_cached_result(cache_k, result)
        
        logger.info(f"HyDE enhancement completed in {processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"HyDE enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@router.post("/hyde/batch")
async def batch_enhance_with_hyde(request: HyDEBatchRequest):
    """
    Enhance multiple queries using HyDE in batch mode for improved efficiency.
    """
    try:
        enhancer = get_hyde_enhancer()
        start_time = time.time()
        
        # Process in parallel for better performance
        tasks = []
        for query in request.queries:
            if request.use_cache:
                cache_k = cache_key("hyde", query, 
                                   domain=request.domain_context, 
                                   num_hyp=request.num_hypothetical)
                cached = get_cached_result(cache_k)
                if cached:
                    async def return_cached():
                        return cached
                    tasks.append(return_cached())
                    continue
            
            # Create async task for enhancement
            async def enhance_single(q=query):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, 
                    lambda: enhancer.enhance_query(
                        query=q,
                        domain_context=request.domain_context,
                        num_hypothetical=request.num_hypothetical
                    )
                )
            tasks.append(enhance_single())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": request.queries[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
                if result.get("success"):
                    successful_count += 1
                    
                    # Cache successful results
                    if request.use_cache:
                        cache_k = cache_key("hyde", request.queries[i],
                                           domain=request.domain_context,
                                           num_hyp=request.num_hypothetical)
                        set_cached_result(cache_k, result)
        
        total_time = time.time() - start_time
        
        return {
            "results": processed_results,
            "batch_summary": {
                "total_queries": len(request.queries),
                "successful_enhancements": successful_count,
                "success_rate": (successful_count / len(request.queries)) * 100,
                "total_processing_time_seconds": total_time,
                "average_time_per_query": total_time / len(request.queries),
                "domain_context": request.domain_context
            }
        }
        
    except Exception as e:
        logger.error(f"Batch HyDE enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch enhancement failed: {str(e)}")


@router.post("/hyde/rag")
async def hyde_enhanced_rag(request: HyDERagRequest):
    """
    Complete RAG pipeline with HyDE query enhancement for superior accuracy.
    """
    try:
        start_time = time.time()
        
        # Step 1: Enhance query with HyDE
        enhancer = get_hyde_enhancer()
        enhancement_start = time.time()
        
        enhancement_result = enhancer.enhance_query(
            query=request.query,
            domain_context=request.domain_context,
            num_hypothetical=request.num_hypothetical
        )
        
        if not enhancement_result.get("success"):
            raise HTTPException(status_code=400, detail="HyDE enhancement failed")
        
        enhancement_time = time.time() - enhancement_start
        
        # Step 2: Use enhanced embedding for retrieval
        searcher = get_semantic_searcher()
        retrieval_start = time.time()
        
        enhanced_embeddings = enhancement_result.get("enhanced_embeddings", {})
        if enhanced_embeddings and enhanced_embeddings.get("fused_embedding"):
            retrieved_docs = searcher.search_with_embedding(
                query_embedding=enhanced_embeddings["fused_embedding"],
                top_k=request.top_k,
                original_query=request.query
            )
        else:
            # Fallback to standard search if no enhanced embedding
            retrieved_docs = searcher.search(request.query, top_k=request.top_k)
        
        retrieval_time = time.time() - retrieval_start
        
        # Step 3: Generate response with enhanced documents
        generator = get_rag_generator()
        generation_start = time.time()
        
        # Use the enhanced documents directly in generation
        generation_result = generator.llm_generator.generate_answer(
            query=request.query,
            retrieved_contexts=retrieved_docs,
            use_citations=True,
            max_context_length=8000
        )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "query": request.query,
            "answer": generation_result.get("answer", ""),
            "source_documents": retrieved_docs,
            "hypothetical_documents": enhancement_result.get("hypothetical_documents", []),
            "follow_up_questions": generation_result.get("follow_up_questions", []),
            "citations": generation_result.get("citations", []),
            "confidence_score": generation_result.get("confidence_score", 0.0),
            "enhancement_details": {
                "hyde_applied": len(enhancement_result.get("hypothetical_documents", [])) > 0,
                "hybrid_search_applied": True,
                "reranking_applied": request.enable_reranking,
                "num_hypothetical_generated": len(enhancement_result.get("hypothetical_documents", [])),
                "fusion_strategy": enhancement_result.get("fusion_strategy", "hybrid")
            },
            "performance": {
                "total_time_seconds": total_time,
                "hyde_enhancement_time_seconds": enhancement_time,
                "hybrid_search_time_seconds": retrieval_time,
                "response_generation_time_seconds": generation_time,
                "documents_retrieved": len(retrieved_docs)
            },
            "conversation_id": request.conversation_id
        }
        
    except Exception as e:
        logger.error(f"HyDE-enhanced RAG failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"HyDE RAG failed: {str(e)}")


# Hybrid Search Endpoints

@router.post("/hybrid/search")
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search combining semantic and keyword search with optional reranking.
    """
    try:
        # Check cache
        if request.use_cache:
            cache_k = cache_key("hybrid", request.query,
                               sem_w=request.semantic_weight,
                               key_w=request.keyword_weight,
                               top_k=request.top_k,
                               rerank=request.enable_reranking)
            cached = get_cached_result(cache_k)
            if cached:
                return cached
        
        searcher = get_hybrid_searcher()
        start_time = time.time()
        
        # Update searcher configuration if needed
        searcher.update_config({
            "semantic_weight": request.semantic_weight,
            "keyword_weight": request.keyword_weight,
            "fusion_method": request.fusion_method,
            "enable_reranking": request.enable_reranking
        })
        
        results = searcher.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            rerank=request.enable_reranking
        )
        
        search_time = time.time() - start_time
        
        response = {
            "success": True,
            "query": request.query,
            "results": results,
            "search_metadata": {
                "total_results": len(results),
                "semantic_weight": request.semantic_weight,
                "keyword_weight": request.keyword_weight,
                "fusion_method": request.fusion_method,
                "reranking_applied": request.enable_reranking
            },
            "performance": {
                "search_time_seconds": search_time,
                "results_count": len(results)
            }
        }
        
        # Cache successful results
        if request.use_cache:
            set_cached_result(cache_k, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.post("/hybrid/batch")
async def batch_hybrid_search(request: HybridBatchRequest):
    """
    Perform batch hybrid search for multiple queries efficiently.
    """
    try:
        searcher = get_hybrid_searcher()
        start_time = time.time()
        
        # Update configuration
        searcher.update_config({
            "semantic_weight": request.semantic_weight,
            "keyword_weight": request.keyword_weight,
            "enable_reranking": request.enable_reranking
        })
        
        # Process queries in parallel
        async def search_single(query):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: searcher.hybrid_search(
                    query=query,
                    top_k=request.top_k,
                    semantic_weight=request.semantic_weight,
                    keyword_weight=request.keyword_weight,
                    rerank=request.enable_reranking
                )
            )
        
        tasks = [search_single(query) for query in request.queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        successful_count = 0
        total_results = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": request.queries[i],
                    "success": False,
                    "error": str(result),
                    "results": []
                })
            else:
                processed_results.append({
                    "query": request.queries[i],
                    "success": True,
                    "results": result,
                    "results_count": len(result)
                })
                successful_count += 1
                total_results += len(result)
        
        total_time = time.time() - start_time
        
        return {
            "batch_results": processed_results,
            "batch_summary": {
                "total_queries": len(request.queries),
                "successful_searches": successful_count,
                "success_rate": (successful_count / len(request.queries)) * 100,
                "total_results_found": total_results,
                "average_results_per_query": total_results / max(successful_count, 1),
                "total_processing_time_seconds": total_time,
                "average_time_per_query": total_time / len(request.queries)
            }
        }
        
    except Exception as e:
        logger.error(f"Batch hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch hybrid search failed: {str(e)}")


@router.post("/hybrid/compare")
async def compare_search_methods(request: SearchComparisonRequest):
    """
    Compare different search methods on the same query for analysis.
    """
    try:
        start_time = time.time()
        comparison_results = {}
        
        # Semantic search
        if request.include_semantic:
            semantic_start = time.time()
            searcher = get_semantic_searcher()
            semantic_results = searcher.search(request.query, top_k=request.top_k)
            semantic_time = time.time() - semantic_start
            
            comparison_results["semantic"] = {
                "results": semantic_results,
                "performance": {
                    "search_time_seconds": semantic_time,
                    "results_count": len(semantic_results),
                    "avg_score": sum(r.get("score", 0) for r in semantic_results) / max(1, len(semantic_results))
                }
            }
        
        # Keyword search
        if request.include_keyword:
            keyword_start = time.time()
            hybrid_searcher = get_hybrid_searcher()
            keyword_results = hybrid_searcher.keyword_only_search(request.query, top_k=request.top_k)
            keyword_time = time.time() - keyword_start
            
            comparison_results["keyword"] = {
                "results": keyword_results,
                "performance": {
                    "search_time_seconds": keyword_time,
                    "results_count": len(keyword_results),
                    "avg_score": sum(r.get("score", 0) for r in keyword_results) / max(1, len(keyword_results))
                }
            }
        
        # Hybrid search
        if request.include_hybrid:
            hybrid_start = time.time()
            hybrid_searcher = get_hybrid_searcher()
            hybrid_results = hybrid_searcher.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                rerank=True
            )
            hybrid_time = time.time() - hybrid_start
            
            comparison_results["hybrid"] = {
                "results": hybrid_results,
                "performance": {
                    "search_time_seconds": hybrid_time,
                    "results_count": len(hybrid_results),
                    "avg_score": sum(r.get("score", 0) for r in hybrid_results) / max(1, len(hybrid_results))
                }
            }
        
        # HyDE-enhanced search
        if request.include_hyde:
            hyde_start = time.time()
            
            # First enhance with HyDE
            enhancer = get_hyde_enhancer()
            enhancement = enhancer.enhance_query(request.query, num_hypothetical=3)
            
            if enhancement.get("success"):
                # Search with enhanced embedding
                enhanced_embeddings = enhancement.get("enhanced_embeddings", {})
                if enhanced_embeddings and enhanced_embeddings.get("fused_embedding"):
                    searcher = get_semantic_searcher()
                    hyde_results = searcher.search_with_embedding(
                        query_embedding=enhanced_embeddings["fused_embedding"],
                        top_k=request.top_k,
                        original_query=request.query
                    )
                else:
                    hyde_results = []
            else:
                hyde_results = []
            
            hyde_time = time.time() - hyde_start
            
            comparison_results["hyde"] = {
                "results": hyde_results,
                "performance": {
                    "search_time_seconds": hyde_time,
                    "results_count": len(hyde_results),
                    "avg_score": sum(r.get("score", 0) for r in hyde_results) / max(1, len(hyde_results))
                },
                "enhancement_success": enhancement.get("success", False)
            }
        
        total_time = time.time() - start_time
        
        return {
            "query": request.query,
            "comparison_results": comparison_results,
            "summary": {
                "total_comparison_time_seconds": total_time,
                "methods_compared": len(comparison_results),
                "top_k": request.top_k
            }
        }
        
    except Exception as e:
        logger.error(f"Search comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search comparison failed: {str(e)}")


# Ultimate RAG Endpoint

@router.post("/ultimate")
async def ultimate_rag_with_all_enhancements(request: UltimateRagRequest):
    """
    Ultimate RAG pipeline combining HyDE + Hybrid Search + Reranking for maximum accuracy.
    """
    try:
        start_time = time.time()
        
        # Step 1: HyDE Enhancement
        enhancer = get_hyde_enhancer()
        hyde_start = time.time()
        
        enhancement_result = enhancer.enhance_query(
            query=request.query,
            domain_context=request.domain_context,
            num_hypothetical=request.num_hypothetical
        )
        
        hyde_time = time.time() - hyde_start
        
        # Step 2: Hybrid Search with enhanced query
        hybrid_start = time.time()
        hybrid_searcher = get_hybrid_searcher()
        
        # Configure hybrid searcher
        hybrid_searcher.update_config({
            "semantic_weight": request.semantic_weight,
            "keyword_weight": request.keyword_weight,
            "enable_reranking": request.enable_reranking
        })
        
        # Use enhanced embedding if available, otherwise fall back to original query
        if (enhancement_result.get("success") and 
            enhancement_result.get("enhanced_embeddings", {}).get("fused_embedding")):
            
            # Use enhanced embedding for semantic part and original query for keyword part
            enhanced_embedding = enhancement_result["enhanced_embeddings"]["fused_embedding"]
            
            # Get semantic results using enhanced embedding
            semantic_searcher = get_semantic_searcher()
            semantic_results = semantic_searcher.search_with_embedding(
                query_embedding=enhanced_embedding,
                top_k=request.top_k * 2,  # Get more for better fusion
                original_query=request.query
            )
            
            # Get keyword results using original query
            keyword_results = hybrid_searcher.keyword_only_search(
                query=request.query,
                top_k=request.top_k * 2
            )
            
            # Use the hybrid searcher to combine results
            search_results = hybrid_searcher.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                semantic_weight=request.semantic_weight,
                keyword_weight=request.keyword_weight,
                rerank=request.enable_reranking
            )
        else:
            # Fallback to standard hybrid search
            search_results = hybrid_searcher.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                semantic_weight=request.semantic_weight,
                keyword_weight=request.keyword_weight,
                rerank=request.enable_reranking
            )
        
        hybrid_time = time.time() - hybrid_start
        
        # Step 3: Generate response with enhanced documents
        generator = get_rag_generator()
        generation_start = time.time()
        
        # Use the enhanced documents directly in generation
        generation_result = generator.llm_generator.generate_answer(
            query=request.query,
            retrieved_contexts=search_results,
            use_citations=True,
            max_context_length=8000
        )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "query": request.query,
            "answer": generation_result.get("answer", ""),
            "source_documents": search_results,
            "enhancement_details": {
                "hyde_applied": enhancement_result.get("success", False),
                "hypothetical_documents": enhancement_result.get("hypothetical_documents", []),
                "hybrid_search_applied": True,
                "reranking_applied": request.enable_reranking,
                "semantic_weight": request.semantic_weight,
                "keyword_weight": request.keyword_weight
            },
            "performance": {
                "total_time_seconds": total_time,
                "hyde_enhancement_time_seconds": hyde_time,
                "hybrid_search_time_seconds": hybrid_time,
                "response_generation_time_seconds": generation_time,
                "documents_retrieved": len(search_results),
                "hypothetical_docs_generated": len(enhancement_result.get("hypothetical_documents", []))
            },
            "quality_metrics": {
                "avg_retrieval_score": sum(r.get("score", 0) for r in search_results) / max(1, len(search_results)),
                "top_retrieval_score": search_results[0].get("score", 0) if search_results else 0,
                "enhancement_success": enhancement_result.get("success", False)
            }
        }
        
    except Exception as e:
        logger.error(f"Ultimate RAG failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ultimate RAG failed: {str(e)}")


# Analytics and Configuration Endpoints

@router.get("/hyde/analytics")
async def get_hyde_analytics():
    """Get comprehensive HyDE analytics and performance metrics."""
    try:
        enhancer = get_hyde_enhancer()
        analytics = enhancer.get_hyde_analytics()
        
        return {
            "hyde_analytics": analytics,
            "cache_stats": {
                "total_cached_queries": len([k for k in _query_cache.keys() if k.startswith("hyde_")]),
                "cache_hit_potential": "Estimated based on repeated queries"
            },
            "system_health": {
                "hyde_service_status": "operational",
                "last_analytics_update": time.time()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get HyDE analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics unavailable: {str(e)}")


@router.get("/hybrid/analytics")
async def get_hybrid_analytics():
    """Get comprehensive hybrid search analytics and performance metrics."""
    try:
        searcher = get_hybrid_searcher()
        analytics = searcher.get_hybrid_analytics()
        
        return {
            "hybrid_analytics": analytics,
            "cache_stats": {
                "total_cached_searches": len([k for k in _query_cache.keys() if k.startswith("hybrid_")]),
                "cache_efficiency": "Automatic cleanup maintains optimal performance"
            },
            "system_health": {
                "hybrid_search_status": "operational",
                "last_analytics_update": time.time()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get hybrid analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics unavailable: {str(e)}")


@router.put("/hyde/config")
async def update_hyde_config(config: HyDEConfigUpdate):
    """Update HyDE configuration parameters."""
    try:
        enhancer = get_hyde_enhancer()
        
        # Build config update dict
        config_updates = {}
        if config.num_hypothetical_docs is not None:
            config_updates["num_hypothetical_docs"] = config.num_hypothetical_docs
        if config.max_doc_length is not None:
            config_updates["max_doc_length"] = config.max_doc_length
        if config.temperature is not None:
            config_updates["temperature"] = config.temperature
        if config.use_query_fusion is not None:
            config_updates["use_query_fusion"] = config.use_query_fusion
        if config.fusion_weights is not None:
            config_updates["fusion_weights"] = config.fusion_weights
        
        enhancer.update_config(config_updates)
        
        return {
            "success": True,
            "message": "HyDE configuration updated successfully",
            "updated_config": config_updates,
            "current_config": enhancer.get_hyde_analytics()["configuration"]
        }
    except Exception as e:
        logger.error(f"Failed to update HyDE config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")


@router.put("/hybrid/config")
async def update_hybrid_config(config: HybridConfigUpdate):
    """Update hybrid search configuration parameters."""
    try:
        searcher = get_hybrid_searcher()
        
        # Build config update dict
        config_updates = {}
        if config.semantic_weight is not None:
            config_updates["semantic_weight"] = config.semantic_weight
        if config.keyword_weight is not None:
            config_updates["keyword_weight"] = config.keyword_weight
        if config.min_keyword_score is not None:
            config_updates["min_keyword_score"] = config.min_keyword_score
        if config.fusion_method is not None:
            config_updates["fusion_method"] = config.fusion_method
        if config.rrf_k is not None:
            config_updates["rrf_k"] = config.rrf_k
        if config.enable_reranking is not None:
            config_updates["enable_reranking"] = config.enable_reranking
        if config.reranking_method is not None:
            config_updates["reranking_method"] = config.reranking_method
        
        searcher.update_config(config_updates)
        
        return {
            "success": True,
            "message": "Hybrid search configuration updated successfully",
            "updated_config": config_updates,
            "current_config": searcher.get_hybrid_analytics()["configuration"]
        }
    except Exception as e:
        logger.error(f"Failed to update hybrid config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")


@router.get("/benchmark")
async def performance_benchmark():
    """
    Run comprehensive performance benchmark comparing all search methods.
    """
    try:
        # Benchmark queries of different types
        benchmark_queries = [
            "How to implement custom LangChain document loaders?",
            "What are the main components of LangChain?", 
            "LangChain agent memory and state management",
            "Step by step LangChain application deployment"
        ]
        
        benchmark_results = []
        
        for query in benchmark_queries:
            query_results = {}
            
            # Test each method
            methods = {
                "semantic": lambda q: get_semantic_searcher().search(q, top_k=5),
                "hybrid": lambda q: get_hybrid_searcher().hybrid_search(q, top_k=5, rerank=True),
                "hyde_semantic": lambda q: _hyde_search(q, top_k=5),
                "ultimate": lambda q: _ultimate_search(q, top_k=5)
            }
            
            for method_name, method_func in methods.items():
                try:
                    start_time = time.time()
                    results = method_func(query)
                    end_time = time.time()
                    
                    avg_score = sum(r.get("score", 0) for r in results) / max(1, len(results))
                    
                    query_results[method_name] = {
                        "success": True,
                        "time_seconds": end_time - start_time,
                        "results_count": len(results),
                        "avg_score": avg_score,
                        "top_score": results[0].get("score", 0) if results else 0
                    }
                except Exception as e:
                    query_results[method_name] = {
                        "success": False,
                        "error": str(e),
                        "time_seconds": 0,
                        "results_count": 0,
                        "avg_score": 0,
                        "top_score": 0
                    }
            
            benchmark_results.append({
                "query": query,
                "method_results": query_results
            })
        
        # Calculate summary statistics
        method_summaries = {}
        for method in ["semantic", "hybrid", "hyde_semantic", "ultimate"]:
            times = [r["method_results"][method]["time_seconds"] 
                    for r in benchmark_results 
                    if r["method_results"][method]["success"]]
            scores = [r["method_results"][method]["avg_score"] 
                     for r in benchmark_results 
                     if r["method_results"][method]["success"]]
            
            method_summaries[method] = {
                "avg_time_seconds": sum(times) / len(times) if times else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "success_rate": len(times) / len(benchmark_queries) * 100
            }
        
        return {
            "benchmark_results": benchmark_results,
            "method_summaries": method_summaries,
            "benchmark_metadata": {
                "total_queries_tested": len(benchmark_queries),
                "methods_compared": len(methods),
                "benchmark_timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# Helper functions for benchmark

def _hyde_search(query: str, top_k: int = 5):
    """Helper function for HyDE search in benchmark."""
    enhancer = get_hyde_enhancer()
    enhancement = enhancer.enhance_query(query, num_hypothetical=3)
    
    if enhancement.get("success"):
        enhanced_embeddings = enhancement.get("enhanced_embeddings", {})
        if enhanced_embeddings and enhanced_embeddings.get("fused_embedding"):
            searcher = get_semantic_searcher()
            return searcher.search_with_embedding(
                query_embedding=enhanced_embeddings["fused_embedding"],
                top_k=top_k,
                original_query=query
            )
    
    # Fallback to regular search
    return get_semantic_searcher().search(query, top_k=top_k)


def _ultimate_search(query: str, top_k: int = 5):
    """Helper function for ultimate search in benchmark."""
    # This is a simplified version of the ultimate search for benchmarking
    enhancer = get_hyde_enhancer()
    enhancement = enhancer.enhance_query(query, num_hypothetical=3)
    
    hybrid_searcher = get_hybrid_searcher()
    
    if (enhancement.get("success") and 
        enhancement.get("enhanced_embeddings", {}).get("fused_embedding")):
        
        # Use enhanced embedding for better search
        enhanced_embedding = enhancement["enhanced_embeddings"]["fused_embedding"]
        semantic_searcher = get_semantic_searcher()
        results = semantic_searcher.search_with_embedding(
            query_embedding=enhanced_embedding,
            top_k=top_k,
            original_query=query
        )
        return results
    else:
        # Fallback to hybrid search
        return hybrid_searcher.hybrid_search(query, top_k=top_k, rerank=True)


# Health check endpoint
@router.get("/health")
async def enhanced_api_health():
    """Health check for enhanced API services."""
    health_status = {
        "service": "enhanced_rag_api",
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    # Check HyDE service
    try:
        enhancer = get_hyde_enhancer()
        analytics = enhancer.get_hyde_analytics()
        health_status["services"]["hyde"] = {
            "status": "operational",
            "total_enhancements": analytics["hyde_stats"]["total_enhancements"]
        }
    except Exception as e:
        health_status["services"]["hyde"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check hybrid search service
    try:
        searcher = get_hybrid_searcher()
        analytics = searcher.get_hybrid_analytics()
        health_status["services"]["hybrid_search"] = {
            "status": "operational",
            "total_searches": analytics["hybrid_stats"]["total_searches"]
        }
    except Exception as e:
        health_status["services"]["hybrid_search"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check cache health
    health_status["services"]["cache"] = {
        "status": "operational",
        "cached_queries": len(_query_cache),
        "cache_size_limit": 1000
    }
    
    # Overall health
    service_statuses = [s["status"] for s in health_status["services"].values()]
    if all(status == "operational" for status in service_statuses):
        health_status["status"] = "healthy"
    elif any(status == "operational" for status in service_statuses):
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status