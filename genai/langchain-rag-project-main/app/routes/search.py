"""Search and retrieval API routes."""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from retrieval.semantic_search import SemanticSearcher, create_semantic_searcher
from retrieval.vector_store import QdrantVectorStore, load_and_upload_embeddings
sys.path.insert(0, str(project_root / "app"))
from models import (
    SearchRequest, SearchResponse, SearchResult, SearchMetadata, DocumentMetadata,
    MultiQuerySearchRequest, MultiQuerySearchResponse,
    BatchSearchRequest, BatchSearchResponse, BatchSearchResult,
    DocumentSearchRequest, DocumentResponse,
    UploadRequest, UploadResponse,
    SearchAnalytics, CollectionStats,
    ErrorResponse, ExportRequest, ExportFormat
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["search"])

# Global searcher instance (will be initialized on first use)
_searcher: Optional[SemanticSearcher] = None


def get_searcher() -> SemanticSearcher:
    """Get or create semantic searcher instance."""
    global _searcher
    if _searcher is None:
        try:
            _searcher = create_semantic_searcher()
            logger.info("✅ Semantic searcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize searcher: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"Search service unavailable: {str(e)}"
            )
    return _searcher


def convert_to_search_result(result: Dict[str, Any]) -> SearchResult:
    """Convert internal result to API model."""
    return SearchResult(
        id=result["id"],
        score=result["score"],
        content=result["content"],
        metadata=DocumentMetadata(**result["metadata"]),
        search_metadata=SearchMetadata(**result["search_metadata"])
    )


@router.get("/search/health")
async def search_health():
    """Health check for search service."""
    try:
        searcher = get_searcher()
        analytics = searcher.get_search_analytics()
        
        return {
            "status": "healthy",
            "service": "semantic-search",
            "database_connected": bool(analytics.get("collection_stats")),
            "embedding_service_connected": bool(analytics.get("embedding_stats")),
            "collection_info": analytics.get("collection_stats", {}),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search service unhealthy: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents using semantic similarity.
    
    Performs vector similarity search using the query text.
    Supports filtering by document type and module patterns.
    """
    start_time = time.time()
    
    try:
        searcher = get_searcher()
        
        # Prepare filters
        filters = {}
        if request.document_types:
            filters["doc_type"] = [dt.value for dt in request.document_types]
        
        # Perform search
        if request.module_patterns:
            # Use module-filtered search
            results = searcher.search_by_module(
                query=request.query,
                module_patterns=request.module_patterns,
                top_k=request.top_k
            )
        else:
            # Standard semantic search
            results = searcher.search(
                query=request.query,
                top_k=request.top_k,
                filters=filters if filters else None,
                score_threshold=request.score_threshold,
                include_metadata=request.include_metadata
            )
        
        # Convert to API models
        api_results = [convert_to_search_result(result) for result in results]
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=api_results,
            total_results=len(api_results),
            search_time_ms=search_time,
            filters_applied={
                "document_types": [dt.value for dt in request.document_types] if request.document_types else None,
                "module_patterns": request.module_patterns,
                "score_threshold": request.score_threshold
            }
        )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search/multi-query", response_model=MultiQuerySearchResponse)
async def multi_query_search(request: MultiQuerySearchRequest):
    """
    Search using multiple queries with result fusion.
    
    Combines results from multiple queries using Reciprocal Rank Fusion
    or score averaging to improve search quality.
    """
    start_time = time.time()
    
    try:
        searcher = get_searcher()
        
        # Perform multi-query search
        results = searcher.multi_query_search(
            queries=request.queries,
            top_k=request.top_k,
            fusion_method=request.fusion_method
        )
        
        # Convert to API models
        api_results = [convert_to_search_result(result) for result in results]
        
        search_time = (time.time() - start_time) * 1000
        
        return MultiQuerySearchResponse(
            success=True,
            queries=request.queries,
            fusion_method=request.fusion_method,
            results=api_results,
            total_results=len(api_results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Multi-query search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-query search failed: {str(e)}")


@router.post("/search/batch", response_model=BatchSearchResponse)
async def batch_search(request: BatchSearchRequest):
    """
    Perform batch search for multiple queries.
    
    Processes multiple queries independently and returns
    results for each query separately.
    """
    start_time = time.time()
    
    try:
        searcher = get_searcher()
        
        # Perform batch search
        batch_results = searcher.batch_search(
            queries=request.queries,
            top_k=request.top_k
        )
        
        # Convert to API models
        api_batch_results = []
        for batch_result in batch_results:
            api_results = [convert_to_search_result(result) for result in batch_result["results"]]
            api_batch_results.append(
                BatchSearchResult(
                    query_index=batch_result["query_index"],
                    query=batch_result["query"],
                    results=api_results,
                    result_count=len(api_results)
                )
            )
        
        search_time = (time.time() - start_time) * 1000
        
        return BatchSearchResponse(
            success=True,
            total_queries=len(request.queries),
            results=api_batch_results,
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"Batch search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch search failed: {str(e)}")


@router.post("/search/similar", response_model=SearchResponse)
async def find_similar_documents(request: DocumentSearchRequest):
    """
    Find documents similar to a given document.
    
    Uses the embedding of the specified document to find
    other similar documents in the collection.
    """
    start_time = time.time()
    
    try:
        searcher = get_searcher()
        
        # Find similar documents
        results = searcher.get_similar_documents(
            document_id=request.document_id,
            top_k=request.top_k
        )
        
        if not results:
            raise HTTPException(
                status_code=404, 
                detail=f"Document {request.document_id} not found or no similar documents"
            )
        
        # Convert to API models
        api_results = [convert_to_search_result(result) for result in results]
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            success=True,
            query=f"Similar to document {request.document_id}",
            results=api_results,
            total_results=len(api_results),
            search_time_ms=search_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar document search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similar document search failed: {str(e)}")


@router.get("/search/analytics", response_model=SearchAnalytics)
async def get_search_analytics():
    """
    Get search system analytics and statistics.
    
    Returns information about the vector collection,
    embedding system, and search configuration.
    """
    try:
        searcher = get_searcher()
        analytics = searcher.get_search_analytics()
        
        if not analytics:
            raise HTTPException(status_code=503, detail="Analytics unavailable")
        
        return SearchAnalytics(
            collection_stats=CollectionStats(**analytics["collection_stats"]),
            embedding_stats=analytics["embedding_stats"],
            search_config=analytics["search_config"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int):
    """
    Retrieve a specific document by ID.
    
    Returns the full document content, metadata, and embedding.
    """
    try:
        searcher = get_searcher()
        document = searcher.vector_store.get_point(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        return DocumentResponse(
            success=True,
            document=document,
            message=f"Document {document_id} retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")


@router.post("/search/export")
async def export_search_results(
    search_request: SearchRequest,
    export_request: ExportRequest
):
    """
    Export search results in different formats.
    
    Performs a search and exports the results as JSON, Markdown, or CSV.
    """
    try:
        searcher = get_searcher()
        
        # Perform search first
        filters = {}
        if search_request.document_types:
            filters["doc_type"] = [dt.value for dt in search_request.document_types]
        
        results = searcher.search(
            query=search_request.query,
            top_k=search_request.top_k,
            filters=filters if filters else None,
            score_threshold=search_request.score_threshold,
            include_metadata=export_request.include_metadata
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found for export")
        
        # Truncate content if requested
        if export_request.max_content_length:
            for result in results:
                if len(result["content"]) > export_request.max_content_length:
                    result["content"] = result["content"][:export_request.max_content_length] + "..."
        
        # Export in requested format
        exported_data = searcher.export_search_results(results, format=export_request.format.value)
        
        if export_request.format == ExportFormat.json:
            return JSONResponse(
                content={"exported_data": exported_data, "format": "json"},
                media_type="application/json"
            )
        elif export_request.format == ExportFormat.markdown:
            return PlainTextResponse(
                content=exported_data,
                media_type="text/markdown",
                headers={"Content-Disposition": "attachment; filename=search_results.md"}
            )
        else:
            # CSV format
            return PlainTextResponse(
                content=exported_data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=search_results.csv"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Vector management endpoints
@router.post("/vectors/upload", response_model=UploadResponse)
async def upload_vectors(request: UploadRequest, background_tasks: BackgroundTasks):
    """
    Upload embeddings to vector database.
    
    Loads embedding files and uploads them to Qdrant.
    This operation may take several minutes for large datasets.
    """
    start_time = time.time()
    
    try:
        # Reset global searcher to force reinitialization
        global _searcher
        _searcher = None
        
        # Load and upload embeddings
        result = load_and_upload_embeddings()
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))
        
        upload_time = (time.time() - start_time) * 1000
        
        return UploadResponse(
            success=True,
            total_documents=result.get("total_documents"),
            total_uploaded=result.get("total_uploaded"),
            failed_preparations=result.get("failed_preparations"),
            upload_errors=result.get("upload_errors"),
            collection_stats=CollectionStats(**result["vector_store_stats"]) if result.get("vector_store_stats") else None,
            upload_time_ms=upload_time,
            files_loaded=result.get("files_loaded")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector upload failed: {str(e)}")


@router.delete("/vectors/collection")
async def delete_collection():
    """
    Delete the vector collection.
    
    WARNING: This will permanently delete all vectors and cannot be undone.
    """
    try:
        # Reset global searcher
        global _searcher
        _searcher = None
        
        vector_store = QdrantVectorStore()
        success = vector_store.delete_collection()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete collection")
        
        return {
            "success": True,
            "message": f"Collection '{vector_store.collection_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collection deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Collection deletion failed: {str(e)}")