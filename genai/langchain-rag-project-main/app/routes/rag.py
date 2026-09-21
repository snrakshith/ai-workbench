"""RAG (Retrieval-Augmented Generation) API routes."""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from generation.rag_pipeline import RAGPipeline, create_rag_pipeline
from generation.llm_integration import GeminiLLMGenerator
sys.path.insert(0, str(project_root / "app"))
from models import ErrorResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["rag"])

# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        try:
            _rag_pipeline = create_rag_pipeline()
            logger.info("✅ RAG pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail=f"RAG service unavailable: {str(e)}"
            )
    return _rag_pipeline


# Pydantic models for RAG endpoints
from pydantic import BaseModel, Field, validator
from typing import Union
from enum import Enum


class ConversationMode(str, Enum):
    """Conversation modes."""
    single = "single"
    contextual = "contextual"


class RAGQueryRequest(BaseModel):
    """RAG query request model."""
    question: str = Field(..., description="User's question", min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Conversation identifier for context")
    conversation_mode: Optional[ConversationMode] = Field(ConversationMode.single, description="Conversation mode")
    
    # Retrieval parameters
    retrieval_params: Optional[Dict[str, Any]] = Field(None, description="Custom retrieval parameters")
    top_k: Optional[int] = Field(None, description="Number of documents to retrieve", ge=1, le=20)
    document_types: Optional[List[str]] = Field(None, description="Filter by document types")
    module_patterns: Optional[List[str]] = Field(None, description="Filter by module patterns")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity threshold", ge=0.0, le=1.0)
    
    # Generation parameters
    generation_params: Optional[Dict[str, Any]] = Field(None, description="Custom generation parameters")
    use_citations: Optional[bool] = Field(True, description="Include source citations")
    generate_followup: Optional[bool] = Field(True, description="Generate follow-up questions")
    max_response_tokens: Optional[int] = Field(None, description="Maximum response length", ge=50, le=2000)
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty or whitespace only')
        return v.strip()
    
    def get_retrieval_params(self) -> Dict[str, Any]:
        """Get formatted retrieval parameters."""
        params = self.retrieval_params or {}
        
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.document_types:
            params["document_types"] = self.document_types
        if self.module_patterns:
            params["module_patterns"] = self.module_patterns
        if self.score_threshold is not None:
            params["score_threshold"] = self.score_threshold
            
        return params if params else None
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get formatted generation parameters."""
        params = self.generation_params or {}
        
        if self.use_citations is not None:
            params["use_citations"] = self.use_citations
        if self.generate_followup is not None:
            params["generate_followup"] = self.generate_followup
        if self.max_response_tokens is not None:
            params["max_response_tokens"] = self.max_response_tokens
            
        return params if params else None


class BatchRAGRequest(BaseModel):
    """Batch RAG request model."""
    questions: List[str] = Field(..., description="List of questions", min_items=1, max_items=10)
    conversation_id: Optional[str] = Field(None, description="Base conversation identifier")
    shared_params: Optional[Dict[str, Any]] = Field(None, description="Shared parameters for all queries")
    
    @validator('questions')
    def validate_questions(cls, v):
        cleaned = [q.strip() for q in v if q.strip()]
        if not cleaned:
            raise ValueError('At least one non-empty question is required')
        return cleaned


class RAGCitation(BaseModel):
    """RAG response citation."""
    source_number: int = Field(..., description="Source number in response")
    file_name: str = Field(..., description="Source file name")
    doc_type: str = Field(..., description="Document type")
    module_path: str = Field(..., description="Module path")
    url: Optional[str] = Field(None, description="Source URL")
    similarity_score: float = Field(..., description="Similarity score")
    content_preview: str = Field(..., description="Content preview")


class RAGPerformance(BaseModel):
    """RAG performance metrics."""
    total_time_ms: float = Field(..., description="Total processing time")
    retrieval_time_ms: float = Field(..., description="Document retrieval time") 
    generation_time_ms: float = Field(..., description="Answer generation time")
    context_documents: int = Field(..., description="Number of context documents used")


class RAGMetadata(BaseModel):
    """RAG response metadata."""
    model_used: str = Field(..., description="LLM model used")
    retrieval_strategy: str = Field(..., description="Retrieval strategy")
    timestamp: str = Field(..., description="Response timestamp")
    pipeline_config: Dict[str, Any] = Field(..., description="Pipeline configuration")


class RAGResponse(BaseModel):
    """RAG query response model."""
    success: bool = Field(..., description="Request success status")
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    citations: List[RAGCitation] = Field(..., description="Source citations")
    confidence_score: float = Field(..., description="Answer confidence score", ge=0.0, le=1.0)
    follow_up_questions: List[str] = Field(..., description="Suggested follow-up questions")
    retrieved_documents: int = Field(..., description="Number of retrieved documents")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    performance: RAGPerformance = Field(..., description="Performance metrics")
    metadata: RAGMetadata = Field(..., description="Response metadata")


class BatchRAGResponse(BaseModel):
    """Batch RAG response model."""
    success: bool = Field(..., description="Batch success status")
    total_questions: int = Field(..., description="Total questions processed")
    results: List[Dict[str, Any]] = Field(..., description="Individual results")
    batch_time_ms: float = Field(..., description="Total batch processing time")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    success: bool = Field(..., description="Request success status")
    conversation_id: str = Field(..., description="Conversation identifier")
    history: List[Dict[str, str]] = Field(..., description="Conversation history")
    total_turns: int = Field(..., description="Total conversation turns")


class RAGAnalyticsResponse(BaseModel):
    """RAG analytics response."""
    success: bool = Field(..., description="Request success status")
    pipeline_stats: Dict[str, Any] = Field(..., description="Pipeline statistics")
    retrieval_analytics: Dict[str, Any] = Field(..., description="Retrieval analytics")
    generation_stats: Dict[str, Any] = Field(..., description="Generation statistics")
    configuration: Dict[str, Any] = Field(..., description="Current configuration")
    success_rate: float = Field(..., description="Overall success rate")


class RAGHealthResponse(BaseModel):
    """RAG health check response."""
    overall_health: str = Field(..., description="Overall health status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    last_check: str = Field(..., description="Last health check timestamp")


@router.get("/rag/health", response_model=RAGHealthResponse)
async def rag_health_check():
    """Check RAG system health."""
    try:
        rag_pipeline = get_rag_pipeline()
        health_status = rag_pipeline.health_check()
        
        return RAGHealthResponse(**health_status)
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG health check failed: {str(e)}")


@router.post("/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Process a RAG query with retrieval and generation.
    
    Combines semantic search with LLM generation to provide
    comprehensive answers with citations and follow-up questions.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        # Generate conversation ID if not provided and using contextual mode
        conversation_id = request.conversation_id
        if request.conversation_mode == ConversationMode.contextual and not conversation_id:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
        
        # Process query
        result = rag_pipeline.query(
            question=request.question,
            conversation_id=conversation_id,
            retrieval_params=request.get_retrieval_params(),
            generation_params=request.get_generation_params()
        )
        
        if not result["success"]:
            return RAGResponse(
                success=False,
                question=request.question,
                answer=result.get("answer", "Failed to generate answer"),
                citations=[],
                confidence_score=0.0,
                follow_up_questions=[],
                retrieved_documents=result.get("retrieved_documents", 0),
                conversation_id=conversation_id,
                performance=RAGPerformance(**result.get("performance", {
                    "total_time_ms": 0, "retrieval_time_ms": 0, 
                    "generation_time_ms": 0, "context_documents": 0
                })),
                metadata=RAGMetadata(
                    model_used="unknown",
                    retrieval_strategy="unknown",
                    timestamp=result.get("metadata", {}).get("timestamp", ""),
                    pipeline_config={}
                )
            )
        
        # Format citations
        formatted_citations = []
        for citation in result.get("citations", []):
            formatted_citations.append(RAGCitation(**citation))
        
        return RAGResponse(
            success=True,
            question=result["question"],
            answer=result["answer"],
            citations=formatted_citations,
            confidence_score=result["confidence_score"],
            follow_up_questions=result.get("follow_up_questions", []),
            retrieved_documents=result["retrieved_documents"],
            conversation_id=result.get("conversation_id"),
            performance=RAGPerformance(**result["performance"]),
            metadata=RAGMetadata(**result["metadata"])
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/rag/batch", response_model=BatchRAGResponse)
async def batch_rag_query(request: BatchRAGRequest):
    """
    Process multiple RAG queries in batch.
    
    Efficiently processes multiple questions with optional
    shared configuration and conversation context.
    """
    start_time = time.time()
    
    try:
        rag_pipeline = get_rag_pipeline()
        
        # Process batch
        results = rag_pipeline.batch_query(
            questions=request.questions,
            conversation_id=request.conversation_id,
            batch_params=request.shared_params
        )
        
        batch_time = (time.time() - start_time) * 1000
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.get("success", False)]
        
        summary = {
            "successful_queries": len(successful_results),
            "failed_queries": len(results) - len(successful_results),
            "success_rate": (len(successful_results) / len(results)) * 100 if results else 0,
            "avg_response_time_ms": sum(r.get("performance", {}).get("total_time_ms", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "avg_confidence_score": sum(r.get("confidence_score", 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            "total_documents_retrieved": sum(r.get("retrieved_documents", 0) for r in results)
        }
        
        return BatchRAGResponse(
            success=True,
            total_questions=len(request.questions),
            results=results,
            batch_time_ms=batch_time,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch RAG query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch RAG query failed: {str(e)}")


@router.get("/rag/conversation/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(conversation_id: str, limit: int = 10):
    """
    Get conversation history for a specific conversation.
    
    Returns the recent conversation turns for context
    and continuation of multi-turn conversations.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        
        # In a production system, you'd retrieve conversation-specific history
        # For now, get general history (since we don't have per-conversation storage)
        history = rag_pipeline.llm_generator.get_conversation_history(limit=limit)
        
        return ConversationHistoryResponse(
            success=True,
            conversation_id=conversation_id,
            history=history,
            total_turns=len(history) // 2  # Each turn has user + assistant messages
        )
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")


@router.delete("/rag/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    Clear conversation history for a specific conversation.
    
    Removes conversation context to start fresh or manage memory usage.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        rag_pipeline.clear_conversation_memory(conversation_id=conversation_id)
        
        return {
            "success": True,
            "message": f"Conversation {conversation_id} cleared successfully",
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        logger.error(f"Failed to clear conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")


@router.get("/rag/analytics", response_model=RAGAnalyticsResponse)
async def get_rag_analytics():
    """
    Get comprehensive RAG system analytics.
    
    Returns performance metrics, usage statistics,
    and configuration information for monitoring.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        analytics = rag_pipeline.get_pipeline_analytics()
        
        return RAGAnalyticsResponse(
            success=True,
            pipeline_stats=analytics["pipeline_stats"],
            retrieval_analytics=analytics["retrieval_analytics"],
            generation_stats=analytics["generation_stats"],
            configuration=analytics["configuration"],
            success_rate=analytics["success_rate"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get RAG analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get RAG analytics: {str(e)}")


@router.post("/rag/config")
async def update_rag_config(config_updates: Dict[str, Any]):
    """
    Update RAG pipeline configuration.
    
    Allows dynamic adjustment of retrieval and generation
    parameters without restarting the service.
    """
    try:
        rag_pipeline = get_rag_pipeline()
        rag_pipeline.update_config(config_updates)
        
        return {
            "success": True,
            "message": "RAG configuration updated successfully",
            "updated_config": config_updates,
            "current_config": rag_pipeline.config
        }
        
    except Exception as e:
        logger.error(f"Failed to update RAG config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update RAG config: {str(e)}")


# Additional utility endpoints
@router.post("/rag/test")
async def test_rag_system():
    """
    Test RAG system with a predefined query.
    
    Useful for health checks and system validation.
    """
    test_question = "What is LangChain and what are its main components?"
    
    try:
        rag_pipeline = get_rag_pipeline()
        
        result = rag_pipeline.query(
            question=test_question,
            conversation_id="test_query"
        )
        
        return {
            "success": result["success"],
            "test_question": test_question,
            "answer_preview": result.get("answer", "")[:200] + "..." if result.get("answer", "") else "",
            "confidence_score": result.get("confidence_score", 0),
            "retrieved_documents": result.get("retrieved_documents", 0),
            "performance": result.get("performance", {}),
            "test_passed": result["success"] and result.get("confidence_score", 0) > 0.5
        }
        
    except Exception as e:
        logger.error(f"RAG system test failed: {str(e)}")
        return {
            "success": False,
            "test_question": test_question,
            "error": str(e),
            "test_passed": False
        }


@router.get("/rag/models")
async def get_available_models():
    """Get information about available models and configurations."""
    return {
        "success": True,
        "llm_models": {
            "current": settings.llm_model,
            "available": ["gemini-2.0-flash", "gemini-1.5-pro"],
            "embedding_model": settings.embedding_model
        },
        "configuration_options": {
            "retrieval": {
                "top_k": "Number of documents to retrieve (1-20)",
                "score_threshold": "Minimum similarity threshold (0.0-1.0)",
                "document_types": "Filter by document types",
                "module_patterns": "Filter by module patterns"
            },
            "generation": {
                "use_citations": "Include source citations",
                "generate_followup": "Generate follow-up questions",
                "max_response_tokens": "Maximum response length (50-2000)"
            },
            "conversation": {
                "conversation_mode": "single or contextual",
                "conversation_id": "Identifier for conversation context"
            }
        }
    }