"""FastAPI routes for query enhancement and optimization features."""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from enhancement.hyde import create_hyde_enhancer
from enhancement.hybrid_search import create_hybrid_searcher
from generation.rag_pipeline import create_rag_pipeline
from src.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/enhancement", tags=["Query Enhancement"])


# Pydantic models
class HyDERequest(BaseModel):
    """Request model for HyDE query enhancement."""
    query: str = Field(..., description="Original query to enhance", min_length=5, max_length=500)
    domain_context: str = Field(default="LangChain", description="Domain context for document generation")
    num_hypothetical: Optional[int] = Field(default=3, description="Number of hypothetical documents to generate", ge=1, le=5)


class BatchHyDERequest(BaseModel):
    """Request model for batch HyDE enhancement."""
    queries: List[str] = Field(..., description="List of queries to enhance", min_items=1, max_items=10)
    domain_context: str = Field(default="LangChain", description="Domain context for document generation")
    num_hypothetical: Optional[int] = Field(default=3, description="Number of hypothetical documents per query", ge=1, le=5)


class HyDEConfigUpdate(BaseModel):
    """Model for updating HyDE configuration."""
    num_hypothetical_docs: Optional[int] = Field(None, description="Number of hypothetical documents", ge=1, le=5)
    max_doc_length: Optional[int] = Field(None, description="Maximum document length in words", ge=100, le=800)
    temperature: Optional[float] = Field(None, description="Generation temperature", ge=0.0, le=1.0)
    use_query_fusion: Optional[bool] = Field(None, description="Whether to use query fusion")
    fusion_weights: Optional[Dict[str, float]] = Field(None, description="Fusion weights for original query and hypothetical docs")


class HyDEResponse(BaseModel):
    """Response model for HyDE enhancement."""
    success: bool
    original_query: str
    hypothetical_documents: List[str]
    enhanced_embeddings: Optional[Dict[str, Any]]
    fusion_strategy: Optional[str]
    performance: Dict[str, Any]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str] = None


class BatchHyDEResponse(BaseModel):
    """Response model for batch HyDE enhancement."""
    results: List[Dict[str, Any]]
    batch_summary: Dict[str, Any]


class HyDEAnalytics(BaseModel):
    """Response model for HyDE analytics."""
    hyde_stats: Dict[str, Any]
    configuration: Dict[str, Any]
    success_rate: float
    models: Dict[str, str]


# Initialize HyDE enhancer (lazy loading)
_hyde_enhancer = None


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


@router.post("/hyde", response_model=HyDEResponse)
async def enhance_query_with_hyde(request: HyDERequest):
    """
    Enhance a query using HyDE (Hypothetical Document Embeddings).
    
    HyDE improves retrieval by generating hypothetical documents that would answer 
    the query, then using these documents' embeddings for more effective search.
    """
    try:
        enhancer = get_hyde_enhancer()
        
        result = enhancer.enhance_query(
            query=request.query,
            domain_context=request.domain_context,
            num_hypothetical=request.num_hypothetical
        )
        
        return HyDEResponse(**result)
        
    except Exception as e:
        logger.error(f"HyDE enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@router.post("/hyde/batch", response_model=BatchHyDEResponse)
async def batch_enhance_queries_with_hyde(request: BatchHyDERequest):
    """
    Enhance multiple queries using HyDE in batch mode.
    
    Processes multiple queries efficiently and returns comprehensive results
    with batch-level analytics.
    """
    try:
        enhancer = get_hyde_enhancer()
        
        results = enhancer.batch_enhance_queries(
            queries=request.queries,
            domain_context=request.domain_context
        )
        
        # Calculate batch summary
        successful = sum(1 for r in results if r.get("success", False))
        total_time = sum(r.get("performance", {}).get("total_time_ms", 0) for r in results)
        avg_generation_time = sum(
            r.get("performance", {}).get("generation_time_ms", 0) for r in results
        ) / len(results)
        
        batch_summary = {
            "total_queries": len(request.queries),
            "successful_enhancements": successful,
            "success_rate": (successful / len(request.queries)) * 100,
            "total_batch_time_ms": total_time,
            "average_generation_time_ms": avg_generation_time,
            "domain_context": request.domain_context
        }
        
        return BatchHyDEResponse(
            results=results,
            batch_summary=batch_summary
        )
        
    except Exception as e:
        logger.error(f"Batch HyDE enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch enhancement failed: {str(e)}")


@router.get("/hyde/search/{query}")
async def hyde_enhanced_search(
    query: str,
    top_k: int = Query(default=5, description="Number of results to return", ge=1, le=20),
    domain_context: str = Query(default="LangChain", description="Domain context for HyDE"),
    num_hypothetical: int = Query(default=3, description="Number of hypothetical documents", ge=1, le=5),
    score_threshold: float = Query(default=0.0, description="Minimum similarity score", ge=0.0, le=1.0)
):
    """
    Perform HyDE-enhanced semantic search.
    
    Combines HyDE query enhancement with semantic search for improved retrieval accuracy.
    """
    try:
        from retrieval.semantic_search import create_semantic_searcher
        
        # Enhance query using HyDE
        enhancer = get_hyde_enhancer()
        enhancement_result = enhancer.enhance_query(
            query=query,
            domain_context=domain_context,
            num_hypothetical=num_hypothetical
        )
        
        if not enhancement_result["success"]:
            raise HTTPException(status_code=400, detail="Query enhancement failed")
        
        # Perform search using enhanced embedding
        searcher = create_semantic_searcher()
        enhanced_embeddings = enhancement_result["enhanced_embeddings"]
        
        if not enhanced_embeddings or not enhanced_embeddings.get("fused_embedding"):
            raise HTTPException(status_code=400, detail="No enhanced embedding available")
        
        # Search using the fused embedding
        search_results = searcher.search_with_embedding(
            query_embedding=enhanced_embeddings["fused_embedding"],
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        return {
            "success": True,
            "query": query,
            "enhancement_used": "HyDE",
            "hypothetical_documents_generated": len(enhancement_result["hypothetical_documents"]),
            "search_results": search_results,
            "performance": {
                "enhancement_time_ms": enhancement_result["performance"]["total_time_ms"],
                "search_time_ms": 0,  # Would need to measure this
                "total_results": len(search_results)
            },
            "metadata": {
                "fusion_strategy": enhanced_embeddings["fusion_method"],
                "domain_context": domain_context,
                "search_params": {
                    "top_k": top_k,
                    "score_threshold": score_threshold
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HyDE-enhanced search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")


@router.put("/hyde/config")
async def update_hyde_config(config_update: HyDEConfigUpdate):
    """
    Update HyDE configuration parameters.
    
    Allows fine-tuning of HyDE behavior for different use cases or domains.
    """
    try:
        enhancer = get_hyde_enhancer()
        
        # Convert model to dict, filtering out None values
        updates = {k: v for k, v in config_update.dict().items() if v is not None}
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid configuration updates provided")
        
        enhancer.update_config(updates)
        
        return {
            "success": True,
            "message": "HyDE configuration updated successfully",
            "updated_parameters": list(updates.keys()),
            "new_configuration": enhancer.get_hyde_analytics()["configuration"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HyDE config update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@router.get("/hyde/analytics", response_model=HyDEAnalytics)
async def get_hyde_analytics():
    """
    Get comprehensive HyDE performance analytics.
    
    Provides insights into HyDE usage, success rates, and performance metrics.
    """
    try:
        enhancer = get_hyde_enhancer()
        analytics = enhancer.get_hyde_analytics()
        
        return HyDEAnalytics(**analytics)
        
    except Exception as e:
        logger.error(f"Failed to get HyDE analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@router.get("/hyde/health")
async def hyde_health_check():
    """
    Check HyDE service health.
    
    Verifies that HyDE enhancer is properly initialized and functioning.
    """
    try:
        enhancer = get_hyde_enhancer()
        
        # Test basic functionality
        test_result = enhancer.enhance_query(
            query="What is LangChain?",
            domain_context="LangChain",
            num_hypothetical=1
        )
        
        analytics = enhancer.get_hyde_analytics()
        
        return {
            "status": "healthy" if test_result["success"] else "degraded",
            "service": "HyDE Query Enhancement",
            "model": enhancer.llm_model,
            "embedding_model": enhancer.embedding_model,
            "test_enhancement": "successful" if test_result["success"] else "failed",
            "statistics": {
                "total_enhancements": analytics["hyde_stats"]["total_enhancements"],
                "success_rate": analytics["success_rate"],
                "avg_generation_time_ms": analytics["hyde_stats"]["avg_generation_time"]
            },
            "configuration": analytics["configuration"]
        }
        
    except Exception as e:
        logger.error(f"HyDE health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "HyDE Query Enhancement",
            "error": str(e),
            "message": "HyDE service is not functioning properly"
        }


@router.get("/hyde/demo")
async def hyde_demo():
    """
    Demonstrate HyDE enhancement with example queries.
    
    Shows the difference between original queries and HyDE-enhanced versions.
    """
    demo_queries = [
        "How do I use LangChain agents?",
        "What are the different types of memory in LangChain?",
        "How to implement custom document loaders?"
    ]
    
    try:
        enhancer = get_hyde_enhancer()
        
        demo_results = []
        for query in demo_queries:
            result = enhancer.enhance_query(query, "LangChain", 2)  # Use 2 hypothetical docs for demo
            
            demo_results.append({
                "original_query": query,
                "enhancement_successful": result["success"],
                "hypothetical_documents": result.get("hypothetical_documents", []),
                "generation_time_ms": result.get("performance", {}).get("generation_time_ms", 0),
                "error": result.get("error")
            })
        
        return {
            "demo_name": "HyDE Query Enhancement Demo",
            "description": "Examples showing how HyDE generates hypothetical documents to improve query understanding",
            "results": demo_results,
            "usage_tip": "HyDE works best with specific, detailed questions about technical topics"
        }
        
    except Exception as e:
        logger.error(f"HyDE demo failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


# Hybrid Search Pydantic Models
class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query: str = Field(..., description="Search query", min_length=3, max_length=500)
    top_k: Optional[int] = Field(default=10, description="Number of results to return", ge=1, le=50)
    semantic_weight: Optional[float] = Field(default=0.7, description="Weight for semantic search", ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(default=0.3, description="Weight for keyword search", ge=0.0, le=1.0)
    rerank: Optional[bool] = Field(default=True, description="Whether to apply re-ranking")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")


class HybridConfigUpdate(BaseModel):
    """Model for updating hybrid search configuration."""
    semantic_weight: Optional[float] = Field(None, description="Weight for semantic search", ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(None, description="Weight for keyword search", ge=0.0, le=1.0)
    fusion_method: Optional[str] = Field(None, description="Fusion method", pattern="^(rrf|weighted)$")
    rrf_k: Optional[int] = Field(None, description="RRF k parameter", ge=1, le=100)
    enable_reranking: Optional[bool] = Field(None, description="Whether to enable re-ranking")
    reranking_method: Optional[str] = Field(None, description="Re-ranking method")


class SearchComparisonRequest(BaseModel):
    """Request model for search method comparison."""
    query: str = Field(..., description="Search query", min_length=3, max_length=500)
    top_k: Optional[int] = Field(default=5, description="Number of results per method", ge=1, le=20)
    include_keyword_only: Optional[bool] = Field(default=True, description="Include keyword-only search")
    include_semantic_only: Optional[bool] = Field(default=True, description="Include semantic-only search")
    include_hybrid: Optional[bool] = Field(default=True, description="Include hybrid search")


# Initialize hybrid searcher (lazy loading)
_hybrid_searcher = None


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


@router.post("/hybrid/search")
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search combining semantic and keyword approaches.
    
    Hybrid search improves retrieval by combining the strengths of:
    - Semantic search: Understanding query intent and context
    - Keyword search: Exact term matching and traditional relevance
    """
    try:
        searcher = get_hybrid_searcher()
        
        # Validate weights sum to 1.0
        if abs((request.semantic_weight + request.keyword_weight) - 1.0) > 0.01:
            raise HTTPException(
                status_code=400, 
                detail="Semantic and keyword weights must sum to 1.0"
            )
        
        results = searcher.hybrid_search(
            query=request.query,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            rerank=request.rerank,
            filters=request.filters
        )
        
        return {
            "success": True,
            "query": request.query,
            "search_method": "hybrid",
            "results": results,
            "total_results": len(results),
            "search_params": {
                "semantic_weight": request.semantic_weight,
                "keyword_weight": request.keyword_weight,
                "reranking_applied": request.rerank,
                "top_k": request.top_k
            },
            "metadata": {
                "fusion_method": "reciprocal_rank_fusion",
                "reranking_enabled": request.rerank
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/hybrid/compare")
async def compare_search_methods(request: SearchComparisonRequest):
    """
    Compare different search methods side-by-side.
    
    Shows results from semantic-only, keyword-only, and hybrid search
    to demonstrate the differences and advantages of each approach.
    """
    try:
        searcher = get_hybrid_searcher()
        comparison_results = {}
        
        # Semantic-only search
        if request.include_semantic_only:
            semantic_results = searcher.semantic_only_search(
                query=request.query,
                top_k=request.top_k
            )
            comparison_results["semantic_only"] = {
                "results": semantic_results,
                "count": len(semantic_results),
                "avg_score": sum(r.get("score", 0) for r in semantic_results) / max(1, len(semantic_results)),
                "method": "Semantic search using embeddings"
            }
        
        # Keyword-only search
        if request.include_keyword_only:
            keyword_results = searcher.keyword_only_search(
                query=request.query,
                top_k=request.top_k
            )
            comparison_results["keyword_only"] = {
                "results": keyword_results,
                "count": len(keyword_results),
                "avg_score": sum(r.get("score", 0) for r in keyword_results) / max(1, len(keyword_results)),
                "method": "Keyword search using BM25-like scoring"
            }
        
        # Hybrid search
        if request.include_hybrid:
            hybrid_results = searcher.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                rerank=True
            )
            comparison_results["hybrid"] = {
                "results": hybrid_results,
                "count": len(hybrid_results),
                "avg_score": sum(r.get("score", 0) for r in hybrid_results) / max(1, len(hybrid_results)),
                "method": "Hybrid search with semantic + keyword fusion + re-ranking"
            }
        
        return {
            "success": True,
            "query": request.query,
            "comparison_results": comparison_results,
            "analysis": {
                "methods_compared": len(comparison_results),
                "total_unique_results": len(set(
                    searcher._get_doc_id(doc) 
                    for method_results in comparison_results.values()
                    for doc in method_results["results"]
                )),
                "recommendation": "Hybrid search typically provides the best balance of precision and recall"
            }
        }
        
    except Exception as e:
        logger.error(f"Search comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.put("/hybrid/config")
async def update_hybrid_config(config_update: HybridConfigUpdate):
    """
    Update hybrid search configuration parameters.
    
    Allows fine-tuning of hybrid search behavior for different use cases.
    """
    try:
        searcher = get_hybrid_searcher()
        
        # Convert model to dict, filtering out None values
        updates = {k: v for k, v in config_update.dict().items() if v is not None}
        
        if not updates:
            raise HTTPException(status_code=400, detail="No valid configuration updates provided")
        
        # Validate weight combination if both are provided
        if "semantic_weight" in updates and "keyword_weight" in updates:
            if abs((updates["semantic_weight"] + updates["keyword_weight"]) - 1.0) > 0.01:
                raise HTTPException(
                    status_code=400,
                    detail="Semantic and keyword weights must sum to 1.0"
                )
        
        searcher.update_config(updates)
        
        return {
            "success": True,
            "message": "Hybrid search configuration updated successfully",
            "updated_parameters": list(updates.keys()),
            "new_configuration": searcher.get_hybrid_analytics()["configuration"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid config update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")


@router.get("/hybrid/analytics")
async def get_hybrid_analytics():
    """
    Get comprehensive hybrid search analytics.
    
    Provides insights into hybrid search usage, performance, and effectiveness.
    """
    try:
        searcher = get_hybrid_searcher()
        analytics = searcher.get_hybrid_analytics()
        
        return {
            "success": True,
            "analytics": analytics,
            "insights": {
                "total_searches": analytics["hybrid_stats"]["total_searches"],
                "hybrid_vs_semantic_ratio": (
                    analytics["hybrid_stats"]["hybrid_searches"] /
                    max(1, analytics["hybrid_stats"]["semantic_searches"])
                ),
                "reranking_effectiveness": analytics["performance"]["reranking_usage"],
                "avg_performance": {
                    "total_time_ms": analytics["hybrid_stats"]["avg_search_time"],
                    "breakdown": {
                        "semantic_ms": analytics["hybrid_stats"]["avg_semantic_time"],
                        "keyword_ms": analytics["hybrid_stats"]["avg_keyword_time"],
                        "fusion_ms": analytics["hybrid_stats"]["avg_fusion_time"]
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get hybrid analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")


@router.get("/hybrid/health")
async def hybrid_health_check():
    """
    Check hybrid search service health.
    
    Verifies that all components of hybrid search are functioning properly.
    """
    try:
        searcher = get_hybrid_searcher()
        
        # Test basic functionality
        test_query = "What is LangChain?"
        test_results = searcher.hybrid_search(test_query, top_k=3)
        
        analytics = searcher.get_hybrid_analytics()
        
        return {
            "status": "healthy" if test_results else "degraded",
            "service": "Hybrid Search",
            "components": {
                "semantic_search": "operational",
                "keyword_search": "operational", 
                "result_fusion": "operational",
                "reranking": "operational" if analytics["configuration"]["enable_reranking"] else "disabled"
            },
            "test_search": {
                "query": test_query,
                "results_found": len(test_results),
                "status": "successful" if test_results else "no_results"
            },
            "configuration": analytics["configuration"],
            "performance": analytics["hybrid_stats"]
        }
        
    except Exception as e:
        logger.error(f"Hybrid search health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "Hybrid Search",
            "error": str(e),
            "message": "Hybrid search service is not functioning properly"
        }


@router.get("/hybrid/demo")
async def hybrid_search_demo():
    """
    Demonstrate hybrid search capabilities with example queries.
    
    Shows the advantages of hybrid search over individual methods.
    """
    demo_queries = [
        "How to create LangChain agents?",
        "LangChain memory types and usage",
        "Document loader implementation guide",
        "Best practices for prompt engineering"
    ]
    
    try:
        searcher = get_hybrid_searcher()
        
        demo_results = []
        for query in demo_queries:
            # Get results from all three methods
            semantic_results = searcher.semantic_only_search(query, top_k=3)
            keyword_results = searcher.keyword_only_search(query, top_k=3)
            hybrid_results = searcher.hybrid_search(query, top_k=3)
            
            demo_results.append({
                "query": query,
                "semantic_results": len(semantic_results),
                "keyword_results": len(keyword_results),
                "hybrid_results": len(hybrid_results),
                "semantic_avg_score": sum(r.get("score", 0) for r in semantic_results) / max(1, len(semantic_results)),
                "keyword_avg_score": sum(r.get("score", 0) for r in keyword_results) / max(1, len(keyword_results)),
                "hybrid_avg_score": sum(r.get("score", 0) for r in hybrid_results) / max(1, len(hybrid_results)),
                "top_result_preview": hybrid_results[0]["content"][:150] + "..." if hybrid_results else "No results"
            })
        
        return {
            "demo_name": "Hybrid Search Demonstration",
            "description": "Comparison of semantic, keyword, and hybrid search methods",
            "results": demo_results,
            "summary": {
                "avg_semantic_results": sum(r["semantic_results"] for r in demo_results) / len(demo_results),
                "avg_keyword_results": sum(r["keyword_results"] for r in demo_results) / len(demo_results),
                "avg_hybrid_results": sum(r["hybrid_results"] for r in demo_results) / len(demo_results),
            },
            "insights": [
                "Hybrid search combines strengths of both semantic and keyword approaches",
                "Semantic search excels at understanding query intent",
                "Keyword search ensures important terms are not missed",
                "Re-ranking improves final result quality"
            ]
        }
        
    except Exception as e:
        logger.error(f"Hybrid search demo failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


# Ultimate RAG Models and Endpoint

class UltimateRAGRequest(BaseModel):
    """Request model for the Ultimate RAG endpoint combining HyDE, Hybrid Search, and Memory."""
    query: str = Field(..., description="User's question", min_length=5, max_length=1000)
    user_id: Optional[str] = Field(None, description="User identifier for memory management")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier for context")
    
    # HyDE Configuration
    hyde_config: Optional[Dict[str, Any]] = Field(
        default={"enabled": True, "num_hypotheses": 3},
        description="HyDE configuration for query enhancement"
    )
    
    # Hybrid Search Configuration
    hybrid_config: Optional[Dict[str, Any]] = Field(
        default={"enabled": True, "semantic_weight": 0.7, "keyword_weight": 0.3},
        description="Hybrid search configuration"
    )
    
    # Reranking Configuration
    reranking_config: Optional[Dict[str, Any]] = Field(
        default={"enabled": True, "top_k_rerank": 10},
        description="Cross-encoder reranking configuration"
    )
    
    # Memory Configuration
    memory_config: Optional[Dict[str, Any]] = Field(
        default={"enabled": True, "include_user_profile": True, "max_context_turns": 5},
        description="Memory integration configuration"
    )
    
    # Generation Parameters
    max_context_length: Optional[int] = Field(default=4000, description="Maximum context length")
    use_citations: Optional[bool] = Field(default=True, description="Whether to include citations")


class UltimateRAGResponse(BaseModel):
    """Response model for Ultimate RAG endpoint."""
    success: bool = Field(..., description="Whether the request was successful")
    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Generated answer")
    citations: List[Dict[str, Any]] = Field(default=[], description="Source citations")
    confidence_score: float = Field(..., description="Answer confidence score")
    follow_up_questions: List[str] = Field(default=[], description="Suggested follow-up questions")
    
    # Performance metrics
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    
    # Memory information
    memory_info: Dict[str, Any] = Field(..., description="Memory usage information")
    
    # Enhancement details
    enhancement_details: Dict[str, Any] = Field(..., description="Details about applied enhancements")
    
    # Metadata
    metadata: Dict[str, Any] = Field(..., description="Request metadata")


# Global pipeline instance for Ultimate RAG
_ultimate_rag_pipeline = None

def get_ultimate_rag_pipeline():
    """Get or create the Ultimate RAG pipeline instance."""
    global _ultimate_rag_pipeline
    if _ultimate_rag_pipeline is None:
        _ultimate_rag_pipeline = create_rag_pipeline()
    return _ultimate_rag_pipeline


@router.post("/ultimate", response_model=UltimateRAGResponse)
async def ultimate_rag_query(request: UltimateRAGRequest):
    """
    Ultimate RAG endpoint combining HyDE, Hybrid Search, Memory, and Cross-encoder Reranking.
    
    This endpoint provides the most advanced RAG experience by combining:
    - HyDE (Hypothetical Document Embeddings) for query enhancement
    - Hybrid search (semantic + keyword) for comprehensive retrieval
    - Memory integration for personalized, context-aware responses
    - Cross-encoder reranking for optimal result ordering
    
    Features:
    - Personalized responses based on user profile and conversation history
    - Enhanced query understanding through hypothetical document generation
    - Comprehensive document retrieval using multiple search strategies
    - High-quality result ranking with cross-encoder models
    - Conversation context preservation across interactions
    """
    try:
        logger.info(f"🎯 Ultimate RAG query: {request.query[:50]}...")
        
        # Get the Ultimate RAG pipeline
        pipeline = get_ultimate_rag_pipeline()
        
        # Prepare retrieval parameters with hybrid and HyDE settings
        retrieval_params = {
            "top_k": request.reranking_config.get("top_k_rerank", 10) if request.reranking_config.get("enabled") else 5,
            "max_context_length": request.max_context_length
        }
        
        # Add HyDE enhancement if enabled
        enhanced_query = request.query
        hyde_details = {"enabled": False, "hypotheses_generated": 0}
        
        if request.hyde_config.get("enabled", True):
            try:
                hyde_enhancer = create_hyde_enhancer()
                hyde_result = hyde_enhancer.generate_hypothetical_documents(
                    query=request.query,
                    num_documents=request.hyde_config.get("num_hypotheses", 3)
                )
                if hyde_result["success"]:
                    # Use the enhanced query from HyDE
                    enhanced_query = hyde_result.get("enhanced_query", request.query)
                    hyde_details = {
                        "enabled": True,
                        "hypotheses_generated": len(hyde_result.get("hypothetical_documents", [])),
                        "enhancement_applied": True
                    }
                    logger.info(f"🧠 HyDE enhancement applied: {hyde_details['hypotheses_generated']} hypotheses")
            except Exception as e:
                logger.warning(f"HyDE enhancement failed, continuing without: {e}")
                hyde_details["error"] = str(e)
        
        # Prepare generation parameters
        generation_params = {
            "use_citations": request.use_citations,
            "generate_followup": True
        }
        
        # Execute the RAG query with memory integration
        rag_result = pipeline.query(
            question=enhanced_query,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            retrieval_params=retrieval_params,
            generation_params=generation_params
        )
        
        # Prepare enhancement details
        enhancement_details = {
            "hyde": hyde_details,
            "hybrid_search": {
                "enabled": request.hybrid_config.get("enabled", True),
                "semantic_weight": request.hybrid_config.get("semantic_weight", 0.7),
                "keyword_weight": request.hybrid_config.get("keyword_weight", 0.3)
            },
            "reranking": {
                "enabled": request.reranking_config.get("enabled", True),
                "top_k_rerank": request.reranking_config.get("top_k_rerank", 10)
            },
            "memory_integration": {
                "enabled": bool(request.user_id),
                "user_profile_used": request.memory_config.get("include_user_profile", True),
                "max_context_turns": request.memory_config.get("max_context_turns", 5)
            }
        }
        
        if rag_result["success"]:
            logger.info(f"✅ Ultimate RAG successful: {rag_result['performance']['total_time_ms']:.1f}ms")
            
            return UltimateRAGResponse(
                success=True,
                question=request.query,
                answer=rag_result["answer"],
                citations=rag_result["citations"],
                confidence_score=rag_result["confidence_score"],
                follow_up_questions=rag_result.get("follow_up_questions", []),
                performance=rag_result["performance"],
                memory_info=rag_result.get("memory_info", {
                    "memory_enabled": False,
                    "memories_used": 0,
                    "recent_turns_used": 0,
                    "user_profile_items": 0
                }),
                enhancement_details=enhancement_details,
                metadata={
                    **rag_result["metadata"],
                    "ultimate_rag_version": "1.0",
                    "enhancements_applied": [
                        key for key, config in enhancement_details.items() 
                        if config.get("enabled", False)
                    ]
                }
            )
        else:
            logger.warning(f"⚠️ Ultimate RAG failed: {rag_result.get('error', 'Unknown error')}")
            
            return UltimateRAGResponse(
                success=False,
                question=request.query,
                answer=rag_result.get("answer", "I apologize, but I'm unable to process your question at the moment."),
                citations=[],
                confidence_score=0.0,
                follow_up_questions=[],
                performance=rag_result.get("performance", {
                    "total_time_ms": 0,
                    "retrieval_time_ms": 0,
                    "generation_time_ms": 0,
                    "context_documents": 0
                }),
                memory_info={
                    "memory_enabled": bool(request.user_id),
                    "memories_used": 0,
                    "recent_turns_used": 0,
                    "user_profile_items": 0,
                    "error": rag_result.get("error")
                },
                enhancement_details=enhancement_details,
                metadata={
                    "ultimate_rag_version": "1.0",
                    "error": rag_result.get("error"),
                    "enhancements_attempted": list(enhancement_details.keys())
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Ultimate RAG endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ultimate RAG processing failed: {str(e)}"
        )