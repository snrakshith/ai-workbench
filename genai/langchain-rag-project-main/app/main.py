"""Main FastAPI application for the LangChain RAG system."""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add src and app to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "app"))

from src.config import settings, config

# Import routes with proper path handling
try:
    from routes.search import router as search_router
    from routes.rag import router as rag_router
    from routes.enhancement import router as enhancement_router
    from enhanced_api import router as enhanced_router
except ImportError:
    # Try relative imports if direct imports fail
    from app.routes.search import router as search_router
    from app.routes.rag import router as rag_router
    from app.routes.enhancement import router as enhancement_router
    from app.enhanced_api import router as enhanced_router

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting LangChain RAG System...")
    logger.info(f"📊 Using collection: {settings.collection_name}")
    logger.info(f"🤖 LLM Model: {settings.llm_model}")
    logger.info(f"📝 Embedding Model: {settings.embedding_model}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down LangChain RAG System...")


# Create FastAPI app
app = FastAPI(
    title=config.get("app", {}).get("name", "LangChain RAG System"),
    description=config.get("app", {}).get("description", "Educational RAG system for LangChain documentation"),
    version=config.get("app", {}).get("version", "1.0.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
static_path = project_root / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include routers
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(enhancement_router)
app.include_router(enhanced_router)


@app.get("/")
async def root():
    """Root endpoint - serve web UI."""
    return RedirectResponse(url="/static/index.html")

@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try to check database connection
        from retrieval.vector_store import QdrantVectorStore
        try:
            vector_store = QdrantVectorStore()
            stats = vector_store.get_collection_stats()
            db_connected = bool(stats.get("points_count", 0) >= 0)
        except Exception:
            db_connected = False
        
        # Try to check embedding service
        from embedding.embedding_generator import GeminiEmbeddingGenerator
        try:
            embedding_gen = GeminiEmbeddingGenerator()
            embedding_connected = True
        except Exception:
            embedding_connected = False
        
        return {
            "status": "healthy",
            "service": "langchain-rag-system", 
            "version": config.get("app", {}).get("version", "1.0.0"),
            "collection": settings.collection_name,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "database_connected": db_connected,
            "embedding_service_connected": embedding_connected,
            "documents_count": stats.get("points_count", 0) if db_connected else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/api/v1/info")
async def system_info():
    """Get system information."""
    return {
        "app_name": config.get("app", {}).get("name", "LangChain RAG System"),
        "version": config.get("app", {}).get("version", "1.0.0"),
        "models": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model
        },
        "config": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k_retrieval": settings.top_k_retrieval,
            "temperature": settings.temperature
        },
        "api_endpoints": {
            "rag_query": "/api/v1/rag/query",
            "rag_batch": "/api/v1/rag/batch", 
            "rag_analytics": "/api/v1/rag/analytics",
            "rag_health": "/api/v1/rag/health",
            "search": "/api/v1/search",
            "multi_query": "/api/v1/search/multi-query",
            "batch_search": "/api/v1/search/batch",
            "similar": "/api/v1/search/similar",
            "search_analytics": "/api/v1/search/analytics",
            "upload": "/api/v1/vectors/upload",
            "hyde_enhancement": "/api/v1/enhancement/hyde",
            "hyde_batch": "/api/v1/enhancement/hyde/batch",
            "hyde_search": "/api/v1/enhancement/hyde/search/{query}",
            "hyde_analytics": "/api/v1/enhancement/hyde/analytics",
            "hybrid_search": "/api/v1/enhancement/hybrid/search",
            "hybrid_compare": "/api/v1/enhancement/hybrid/compare",
            "hybrid_analytics": "/api/v1/enhancement/hybrid/analytics",
            "enhancement_health": "/api/v1/enhancement/hyde/health",
            "enhanced_hyde_query": "/api/v1/enhanced/hyde/query",
            "enhanced_hyde_batch": "/api/v1/enhanced/hyde/batch",
            "enhanced_hyde_rag": "/api/v1/enhanced/hyde/rag",
            "enhanced_hybrid_search": "/api/v1/enhanced/hybrid/search",
            "enhanced_hybrid_batch": "/api/v1/enhanced/hybrid/batch",
            "enhanced_hybrid_compare": "/api/v1/enhanced/hybrid/compare",
            "enhanced_ultimate_rag": "/api/v1/enhanced/ultimate",
            "enhanced_benchmark": "/api/v1/enhanced/benchmark",
            "enhanced_hyde_analytics": "/api/v1/enhanced/hyde/analytics",
            "enhanced_hybrid_analytics": "/api/v1/enhanced/hybrid/analytics",
            "enhanced_health": "/api/v1/enhanced/health",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    try:
        # Check search service
        from retrieval.semantic_search import create_semantic_searcher
        try:
            searcher = create_semantic_searcher()
            search_ready = True
            analytics = searcher.get_search_analytics()
            collection_stats = analytics.get("collection_stats", {})
        except Exception:
            search_ready = False
            collection_stats = {}
        
        return {
            "api_status": "operational",
            "services": {
                "search_service": "ready" if search_ready else "unavailable",
                "vector_database": "connected" if collection_stats else "disconnected",
                "embedding_service": "ready"  # Assume ready if API started
            },
            "collection_info": collection_stats,
            "endpoints_available": len([
                "/api/v1/search", "/api/v1/search/multi-query", "/api/v1/search/batch",
                "/api/v1/search/similar", "/api/v1/search/analytics", "/api/v1/vectors/upload",
                "/api/v1/documents/{id}", "/health", "/api/v1/info", "/api/v1/status"
            ]),
            "message": "RAG system fully operational" if search_ready else "Search service initializing..."
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {
            "api_status": "degraded",
            "error": str(e),
            "message": "Some services may be unavailable"
        }


if __name__ == "__main__":
    print("🚀 Starting LangChain RAG Server...")
    print(f"📍 URL: http://localhost:{settings.api_port}")
    print(f"📚 Docs: http://localhost:{settings.api_port}/docs")
    print("🛑 Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Disable reload to prevent restart loops
        log_level="info"
    )