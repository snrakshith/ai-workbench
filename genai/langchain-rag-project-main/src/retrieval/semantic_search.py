"""Semantic search and retrieval system."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import json

from .vector_store import QdrantVectorStore
from src.embedding.embedding_generator import GeminiEmbeddingGenerator
from src.config import settings

logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Semantic search system using vector similarity."""
    
    def __init__(self, collection_name: str = None):
        self.vector_store = QdrantVectorStore(collection_name=collection_name)
        self.embedding_generator = GeminiEmbeddingGenerator()
        self.collection_name = collection_name or settings.collection_name
        
        # Search parameters
        self.default_top_k = settings.top_k_retrieval
        self.similarity_threshold = 0.7
        
        # Initialize if collection exists
        self._initialize()
    
    def _initialize(self):
        """Initialize the searcher with collection info."""
        try:
            stats = self.vector_store.get_collection_stats()
            if stats:
                logger.info(f"✅ Initialized searcher for collection '{self.collection_name}'")
                logger.info(f"📊 Collection has {stats['points_count']} documents")
            else:
                logger.warning(f"⚠️  Collection '{self.collection_name}' not found or empty")
        except Exception as e:
            logger.error(f"Failed to initialize searcher: {str(e)}")
    
    def search(self, query: str, top_k: int = None, filters: Dict[str, Any] = None,
              score_threshold: float = None, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Perform semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Filter conditions for search
            score_threshold: Minimum similarity score
            include_metadata: Whether to include document metadata
        """
        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Perform vector search
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k or self.default_top_k,
                filter_conditions=filters,
                score_threshold=score_threshold or self.similarity_threshold
            )
            
            # Enhance results with search metadata
            for i, result in enumerate(results):
                result["search_metadata"] = {
                    "query": query,
                    "rank": i + 1,
                    "search_timestamp": datetime.now().isoformat(),
                    "similarity_score": result["score"]
                }
                
                # Fix GitHub URLs (blob/main -> blob/master and remove duplicate langchain/ prefix)
                if "metadata" in result and "url" in result["metadata"]:
                    if "blob/main" in result["metadata"]["url"]:
                        result["metadata"]["url"] = result["metadata"]["url"].replace("blob/main", "blob/master")
                    # Fix duplicate langchain/ prefix in URLs
                    url = result["metadata"]["url"]
                    if "langchain-ai/langchain/blob/master/langchain/" in url:
                        result["metadata"]["url"] = url.replace("blob/master/langchain/", "blob/master/")
                
                if not include_metadata:
                    # Keep only essential metadata
                    essential_keys = ["doc_type", "source_file", "title", "url"]
                    result["metadata"] = {
                        k: v for k, v in result["metadata"].items() 
                        if k in essential_keys
                    }
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def search_with_embedding(self, query_embedding: List[float], top_k: int = None, 
                             filters: Dict[str, Any] = None, score_threshold: float = None,
                             include_metadata: bool = True, original_query: str = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using a pre-computed embedding.
        
        This method is useful for HyDE-enhanced search where the query embedding
        has already been generated through hypothetical document embeddings.
        
        Args:
            query_embedding: Pre-computed query embedding vector
            top_k: Number of results to return
            filters: Filter conditions for search
            score_threshold: Minimum similarity score
            include_metadata: Whether to include document metadata
            original_query: Original query text for metadata (optional)
        """
        try:
            # Perform vector search using the provided embedding
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k or self.default_top_k,
                filter_conditions=filters,
                score_threshold=score_threshold or self.similarity_threshold
            )
            
            # Enhance results with search metadata
            for i, result in enumerate(results):
                result["search_metadata"] = {
                    "query": original_query or "Enhanced embedding query",
                    "rank": i + 1,
                    "search_timestamp": datetime.now().isoformat(),
                    "similarity_score": result["score"],
                    "search_method": "embedding_based"
                }
                
                # Fix GitHub URLs (blob/main -> blob/master and remove duplicate langchain/ prefix)
                if "metadata" in result and "url" in result["metadata"]:
                    if "blob/main" in result["metadata"]["url"]:
                        result["metadata"]["url"] = result["metadata"]["url"].replace("blob/main", "blob/master")
                    # Fix duplicate langchain/ prefix in URLs
                    url = result["metadata"]["url"]
                    if "langchain-ai/langchain/blob/master/langchain/" in url:
                        result["metadata"]["url"] = url.replace("blob/master/langchain/", "blob/master/")
                
                if not include_metadata:
                    # Keep only essential metadata
                    essential_keys = ["doc_type", "source_file", "title", "url"]
                    result["metadata"] = {
                        k: v for k, v in result["metadata"].items() 
                        if k in essential_keys
                    }
            
            query_preview = original_query[:50] + "..." if original_query and len(original_query) > 50 else "Enhanced embedding"
            logger.info(f"Found {len(results)} results for embedding search: {query_preview}")
            return results
            
        except Exception as e:
            logger.error(f"Embedding-based search failed: {str(e)}")
            return []
    
    def search_by_document_type(self, query: str, doc_types: List[str], 
                               top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search within specific document types."""
        results_by_type = {}
        
        for doc_type in doc_types:
            filters = {"doc_type": doc_type}
            results = self.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            results_by_type[doc_type] = results
            
        return results_by_type
    
    def search_by_module(self, query: str, module_patterns: List[str], 
                        top_k: int = None) -> List[Dict[str, Any]]:
        """Search within specific modules or module patterns."""
        # For now, we'll do post-filtering since Qdrant's pattern matching is limited
        # In production, you might want to use Qdrant's more advanced filtering
        
        all_results = self.search(query=query, top_k=top_k * 2 if top_k else None)
        
        filtered_results = []
        for result in all_results:
            module_path = result.get("metadata", {}).get("module_path", "")
            if any(pattern in module_path for pattern in module_patterns):
                filtered_results.append(result)
                
            if top_k and len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def get_similar_documents(self, document_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a given document."""
        try:
            # Get the document
            document = self.vector_store.get_point(document_id)
            if not document:
                logger.error(f"Document {document_id} not found")
                return []
            
            # Search using its embedding
            results = self.vector_store.search_similar(
                query_embedding=document["embedding"],
                top_k=top_k + 1  # +1 because result will include the document itself
            )
            
            # Remove the original document from results
            similar_docs = [r for r in results if r["id"] != document_id][:top_k]
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Failed to find similar documents: {str(e)}")
            return []
    
    def multi_query_search(self, queries: List[str], top_k: int = None, 
                          fusion_method: str = "rrf") -> List[Dict[str, Any]]:
        """
        Search using multiple queries and fuse results.
        
        Args:
            queries: List of query strings
            top_k: Number of final results
            fusion_method: Method for fusing results ("rrf" for Reciprocal Rank Fusion)
        """
        if not queries:
            return []
        
        # Get results for each query
        all_results = []
        for query in queries:
            results = self.search(query=query, top_k=top_k or self.default_top_k)
            all_results.append(results)
        
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(all_results, top_k or self.default_top_k)
        else:
            # Simple score averaging
            return self._average_score_fusion(all_results, top_k or self.default_top_k)
    
    def _reciprocal_rank_fusion(self, result_lists: List[List[Dict[str, Any]]], 
                               top_k: int, k: int = 60) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine multiple result lists."""
        doc_scores = {}
        doc_data = {}
        
        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc["id"]
                rrf_score = 1 / (k + rank + 1)
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_data[doc_id] = doc
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = doc_data[doc_id].copy()
            doc["search_metadata"]["fusion_score"] = rrf_score
            doc["search_metadata"]["fusion_method"] = "reciprocal_rank_fusion"
            fused_results.append(doc)
        
        return fused_results
    
    def _average_score_fusion(self, result_lists: List[List[Dict[str, Any]]], 
                             top_k: int) -> List[Dict[str, Any]]:
        """Fuse results by averaging similarity scores."""
        doc_scores = {}
        doc_counts = {}
        doc_data = {}
        
        for results in result_lists:
            for doc in results:
                doc_id = doc["id"]
                score = doc["score"]
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += score
                    doc_counts[doc_id] += 1
                else:
                    doc_scores[doc_id] = score
                    doc_counts[doc_id] = 1
                    doc_data[doc_id] = doc
        
        # Calculate average scores
        avg_scores = {
            doc_id: doc_scores[doc_id] / doc_counts[doc_id]
            for doc_id in doc_scores
        }
        
        # Sort by average score
        sorted_docs = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        fused_results = []
        for doc_id, avg_score in sorted_docs[:top_k]:
            doc = doc_data[doc_id].copy()
            doc["score"] = avg_score
            doc["search_metadata"]["fusion_score"] = avg_score
            doc["search_metadata"]["fusion_method"] = "average_score"
            fused_results.append(doc)
        
        return fused_results
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search system analytics."""
        try:
            collection_stats = self.vector_store.get_collection_stats()
            embedding_stats = self.embedding_generator.get_embedding_stats()
            
            return {
                "collection_stats": collection_stats,
                "embedding_stats": embedding_stats,
                "search_config": {
                    "default_top_k": self.default_top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "embedding_model": self.embedding_generator.model_name,
                    "vector_dimension": collection_stats.get("vector_dimension", "unknown")
                }
            }
        except Exception as e:
            logger.error(f"Failed to get analytics: {str(e)}")
            return {}
    
    def batch_search(self, queries: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """Perform batch search for multiple queries."""
        results = []
        
        for i, query in enumerate(queries):
            query_results = self.search(query=query, top_k=top_k)
            results.append({
                "query_index": i,
                "query": query,
                "results": query_results,
                "result_count": len(query_results)
            })
        
        return results
    
    def export_search_results(self, results: List[Dict[str, Any]], 
                             format: str = "json") -> str:
        """Export search results to different formats."""
        if format == "json":
            return json.dumps(results, indent=2, ensure_ascii=False)
        elif format == "markdown":
            md_lines = ["# Search Results\\n"]
            for i, result in enumerate(results, 1):
                md_lines.append(f"## Result {i} (Score: {result['score']:.4f})")
                md_lines.append(f"**Source:** {result['metadata'].get('source_file', 'Unknown')}")
                md_lines.append(f"**Module:** {result['metadata'].get('module_path', 'Unknown')}")
                md_lines.append(f"**Type:** {result['metadata'].get('doc_type', 'Unknown')}")
                if result['metadata'].get('url'):
                    md_lines.append(f"**URL:** {result['metadata']['url']}")
                md_lines.append("\\n**Content:**")
                md_lines.append(result['content'])
                md_lines.append("\\n---\\n")
            return "\\n".join(md_lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


def create_semantic_searcher(collection_name: str = None) -> SemanticSearcher:
    """Create and initialize a semantic searcher."""
    return SemanticSearcher(collection_name=collection_name)