"""Pydantic models for FastAPI requests and responses."""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    markdown = "markdown"
    notebooks = "notebooks"
    python_files = "python_files"
    rst_files = "rst_files"
    readme_files = "readme_files"
    api = "api"
    tutorial = "tutorial"
    guide = "guide"
    example = "example"
    documentation = "documentation"


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=50)
    document_types: Optional[List[DocumentType]] = Field(None, description="Filter by document types")
    module_patterns: Optional[List[str]] = Field(None, description="Filter by module patterns")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score", ge=0.0, le=1.0)
    include_metadata: Optional[bool] = Field(True, description="Include detailed metadata")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()


class MultiQuerySearchRequest(BaseModel):
    """Multi-query search request model."""
    queries: List[str] = Field(..., description="List of search queries", min_items=1, max_items=10)
    top_k: Optional[int] = Field(5, description="Number of final results", ge=1, le=50)
    fusion_method: Optional[str] = Field("rrf", description="Fusion method: 'rrf' or 'average'")
    
    @validator('queries')
    def validate_queries(cls, v):
        cleaned = [q.strip() for q in v if q.strip()]
        if not cleaned:
            raise ValueError('At least one non-empty query is required')
        return cleaned


class DocumentSearchRequest(BaseModel):
    """Document similarity search request."""
    document_id: int = Field(..., description="Document ID for similarity search", ge=0)
    top_k: Optional[int] = Field(5, description="Number of similar documents", ge=1, le=50)


class SearchMetadata(BaseModel):
    """Search result metadata."""
    query: str = Field(..., description="Original search query")
    rank: int = Field(..., description="Result rank in search")
    search_timestamp: str = Field(..., description="Search timestamp")
    similarity_score: float = Field(..., description="Similarity score")
    fusion_score: Optional[float] = Field(None, description="Fusion score if applicable")
    fusion_method: Optional[str] = Field(None, description="Fusion method used")


class DocumentMetadata(BaseModel):
    """Document metadata."""
    doc_type: Optional[str] = Field(None, description="Document type classification")
    source_file: Optional[str] = Field(None, description="Source file path")
    module_path: Optional[str] = Field(None, description="Module path")
    chunk_index: Optional[int] = Field(None, description="Chunk index within document")
    token_count: Optional[int] = Field(None, description="Token count")
    chunk_size: Optional[int] = Field(None, description="Character count")
    file_name: Optional[str] = Field(None, description="File name")
    title: Optional[str] = Field(None, description="Document title")
    url: Optional[str] = Field(None, description="GitHub URL")


class SearchResult(BaseModel):
    """Individual search result."""
    id: int = Field(..., description="Document ID")
    score: float = Field(..., description="Similarity score")
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    search_metadata: SearchMetadata = Field(..., description="Search metadata")


class SearchResponse(BaseModel):
    """Search response model."""
    success: bool = Field(..., description="Request success status")
    query: str = Field(..., description="Original query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Number of results returned")
    search_time_ms: Optional[float] = Field(None, description="Search time in milliseconds")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="Applied filters")


class MultiQuerySearchResponse(BaseModel):
    """Multi-query search response."""
    success: bool = Field(..., description="Request success status")
    queries: List[str] = Field(..., description="Original queries")
    fusion_method: str = Field(..., description="Fusion method used")
    results: List[SearchResult] = Field(..., description="Fused search results")
    total_results: int = Field(..., description="Number of results returned")
    search_time_ms: Optional[float] = Field(None, description="Search time in milliseconds")


class BatchSearchRequest(BaseModel):
    """Batch search request."""
    queries: List[str] = Field(..., description="List of queries", min_items=1, max_items=20)
    top_k: Optional[int] = Field(5, description="Results per query", ge=1, le=20)
    
    @validator('queries')
    def validate_queries(cls, v):
        cleaned = [q.strip() for q in v if q.strip()]
        if not cleaned:
            raise ValueError('At least one non-empty query is required')
        return cleaned


class BatchSearchResult(BaseModel):
    """Individual batch search result."""
    query_index: int = Field(..., description="Query index in batch")
    query: str = Field(..., description="Query text")
    results: List[SearchResult] = Field(..., description="Search results for query")
    result_count: int = Field(..., description="Number of results")


class BatchSearchResponse(BaseModel):
    """Batch search response."""
    success: bool = Field(..., description="Request success status")
    total_queries: int = Field(..., description="Total queries processed")
    results: List[BatchSearchResult] = Field(..., description="Results for each query")
    search_time_ms: Optional[float] = Field(None, description="Total search time in milliseconds")


class CollectionStats(BaseModel):
    """Vector collection statistics."""
    collection_name: str = Field(..., description="Collection name")
    points_count: int = Field(..., description="Number of documents")
    segments_count: int = Field(..., description="Number of segments")
    vector_dimension: int = Field(..., description="Vector dimension")
    distance_metric: str = Field(..., description="Distance metric used")
    status: str = Field(..., description="Collection status")


class SearchAnalytics(BaseModel):
    """Search system analytics."""
    collection_stats: CollectionStats = Field(..., description="Collection statistics")
    embedding_stats: Dict[str, Any] = Field(..., description="Embedding statistics")
    search_config: Dict[str, Any] = Field(..., description="Search configuration")


class UploadRequest(BaseModel):
    """Vector upload request."""
    force_recreate: Optional[bool] = Field(False, description="Recreate collection if exists")
    batch_size: Optional[int] = Field(100, description="Upload batch size", ge=1, le=500)


class UploadResponse(BaseModel):
    """Vector upload response."""
    success: bool = Field(..., description="Upload success status")
    total_documents: Optional[int] = Field(None, description="Total documents processed")
    total_uploaded: Optional[int] = Field(None, description="Successfully uploaded documents")
    failed_preparations: Optional[int] = Field(None, description="Failed to prepare for upload")
    upload_errors: Optional[int] = Field(None, description="Upload errors")
    collection_stats: Optional[CollectionStats] = Field(None, description="Updated collection stats")
    upload_time_ms: Optional[float] = Field(None, description="Upload time in milliseconds")
    files_loaded: Optional[List[str]] = Field(None, description="Loaded embedding files")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")
    collection: Optional[str] = Field(None, description="Active collection")
    database_connected: bool = Field(..., description="Database connection status")
    embedding_service_connected: bool = Field(..., description="Embedding service status")


class DocumentResponse(BaseModel):
    """Single document response."""
    success: bool = Field(..., description="Request success status")
    document: Optional[Dict[str, Any]] = Field(None, description="Document data")
    message: Optional[str] = Field(None, description="Response message")


# Export format enums
class ExportFormat(str, Enum):
    """Supported export formats."""
    json = "json"
    markdown = "markdown"
    csv = "csv"


class ExportRequest(BaseModel):
    """Export request model."""
    format: ExportFormat = Field(..., description="Export format")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in export")
    max_content_length: Optional[int] = Field(None, description="Maximum content length per result")