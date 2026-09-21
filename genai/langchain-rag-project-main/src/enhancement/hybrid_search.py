"""Hybrid search combining semantic and keyword search with re-ranking."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from collections import Counter
import math

from src.retrieval.semantic_search import SemanticSearcher, create_semantic_searcher
from src.retrieval.vector_store import QdrantVectorStore
from src.config import settings

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Advanced hybrid search combining semantic and keyword-based retrieval."""
    
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or settings.collection_name
        
        # Initialize semantic searcher
        self.semantic_searcher = create_semantic_searcher(collection_name=collection_name)
        self.vector_store = QdrantVectorStore(collection_name=collection_name)
        
        # Hybrid search configuration
        self.config = {
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "min_keyword_score": 0.1,
            "max_results_per_method": 20,
            "fusion_method": "rrf",  # reciprocal rank fusion
            "rrf_k": 60,
            "enable_reranking": True,
            "reranking_method": "cross_encoder",
            "boost_factors": {
                "title_match": 2.0,
                "exact_phrase": 1.5,
                "doc_type_preference": 1.3
            }
        }
        
        # Performance statistics
        self.hybrid_stats = {
            "total_searches": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "avg_search_time": 0,
            "avg_semantic_time": 0,
            "avg_keyword_time": 0,
            "avg_fusion_time": 0,
            "reranking_applied": 0
        }
        
        logger.info(f"✅ Hybrid Searcher initialized for collection: {self.collection_name}")
    
    def hybrid_search(self, query: str, top_k: int = None, 
                     semantic_weight: float = None, keyword_weight: float = None,
                     rerank: bool = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query: Search query
            top_k: Number of final results
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            rerank: Whether to apply re-ranking
            filters: Additional filters for search
        """
        start_time = time.time()
        self.hybrid_stats["total_searches"] += 1
        self.hybrid_stats["hybrid_searches"] += 1
        
        # Use provided weights or defaults
        sem_weight = semantic_weight or self.config["semantic_weight"]
        key_weight = keyword_weight or self.config["keyword_weight"]
        final_top_k = top_k or settings.top_k_retrieval
        apply_rerank = rerank if rerank is not None else self.config["enable_reranking"]
        
        try:
            # Step 1: Semantic search
            semantic_start = time.time()
            semantic_results = self._semantic_search(
                query, 
                top_k=self.config["max_results_per_method"],
                filters=filters
            )
            semantic_time = (time.time() - semantic_start) * 1000
            
            # Step 2: Keyword search
            keyword_start = time.time()
            keyword_results = self._keyword_search(
                query,
                top_k=self.config["max_results_per_method"],
                filters=filters
            )
            keyword_time = (time.time() - keyword_start) * 1000
            
            # Step 3: Fusion
            fusion_start = time.time()
            fused_results = self._fuse_results(
                semantic_results,
                keyword_results,
                semantic_weight=sem_weight,
                keyword_weight=key_weight,
                query=query
            )
            fusion_time = (time.time() - fusion_start) * 1000
            
            # Step 4: Re-ranking (optional)
            if apply_rerank and len(fused_results) > 1:
                rerank_start = time.time()
                fused_results = self._rerank_results(fused_results, query)
                rerank_time = (time.time() - rerank_start) * 1000
                self.hybrid_stats["reranking_applied"] += 1
            else:
                rerank_time = 0
            
            # Step 5: Final selection and metadata enhancement
            final_results = fused_results[:final_top_k]
            
            # Add hybrid search metadata
            total_time = (time.time() - start_time) * 1000
            for i, result in enumerate(final_results):
                result["hybrid_metadata"] = {
                    "final_rank": i + 1,
                    "semantic_contribution": result.get("semantic_score", 0) * sem_weight,
                    "keyword_contribution": result.get("keyword_score", 0) * key_weight,
                    "hybrid_score": result.get("hybrid_score", 0),
                    "reranked": apply_rerank,
                    "search_method": "hybrid",
                    "fusion_method": self.config["fusion_method"]
                }
            
            # Update statistics
            self._update_hybrid_stats(semantic_time, keyword_time, fusion_time + rerank_time, total_time)
            
            logger.info(f"Hybrid search completed: {len(final_results)} results in {total_time:.1f}ms")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            # Fallback to semantic search
            return self._semantic_search(query, top_k=final_top_k, filters=filters)
    
    def _semantic_search(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        self.hybrid_stats["semantic_searches"] += 1
        
        try:
            results = self.semantic_searcher.search(
                query=query,
                top_k=top_k,
                filters=filters,
                include_metadata=True
            )
            
            # Add semantic-specific metadata
            for result in results:
                result["semantic_score"] = result["score"]
                result["search_source"] = "semantic"
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    def _keyword_search(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform keyword-based search using BM25-like scoring."""
        self.hybrid_stats["keyword_searches"] += 1
        
        try:
            # Get all documents for keyword matching
            # In a production system, you'd use a proper full-text search index
            all_docs = self._get_searchable_documents(filters)
            
            if not all_docs:
                return []
            
            # Extract query terms
            query_terms = self._extract_query_terms(query)
            
            # Score documents using BM25-like algorithm
            scored_docs = []
            for doc in all_docs:
                score = self._calculate_keyword_score(doc, query_terms, query)
                if score > self.config["min_keyword_score"]:
                    doc_copy = doc.copy()
                    doc_copy["score"] = score
                    doc_copy["keyword_score"] = score
                    doc_copy["search_source"] = "keyword"
                    scored_docs.append(doc_copy)
            
            # Sort by score and return top results
            scored_docs.sort(key=lambda x: x["score"], reverse=True)
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def _get_searchable_documents(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get documents for keyword search (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In production, you'd use a proper search index
            collection_stats = self.vector_store.get_collection_stats()
            if not collection_stats or collection_stats.get("points_count", 0) == 0:
                return []
            
            # For demo purposes, we'll use a semantic search with very broad query
            # to get a sample of documents for keyword matching
            sample_results = self.semantic_searcher.search(
                query="LangChain documentation guide tutorial",
                top_k=min(500, collection_stats.get("points_count", 100)),
                filters=filters,
                score_threshold=0.0  # Very low threshold to get many results
            )
            
            return sample_results
            
        except Exception as e:
            logger.error(f"Failed to get searchable documents: {str(e)}")
            return []
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from query."""
        # Simple tokenization and term extraction
        # Remove common stop words and normalize
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "about", "into", "through", "during", "before", "after", "above",
            "below", "up", "down", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
            "don", "should", "now", "what", "which", "who", "is", "are", "was", "were"
        }
        
        # Normalize and tokenize
        normalized_query = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = normalized_query.split()
        
        # Filter stop words and short terms
        meaningful_terms = [
            term for term in terms 
            if len(term) > 2 and term not in stop_words
        ]
        
        return meaningful_terms
    
    def _calculate_keyword_score(self, doc: Dict[str, Any], query_terms: List[str], 
                                original_query: str) -> float:
        """Calculate BM25-like keyword score for a document."""
        if not query_terms:
            return 0.0
        
        content = doc.get("content", "").lower()
        metadata = doc.get("metadata", {})
        title = metadata.get("title", "").lower()
        file_name = metadata.get("file_name", "").lower()
        
        # Combine text fields for scoring
        combined_text = f"{content} {title} {file_name}"
        
        # Calculate term frequencies
        words = re.findall(r'\w+', combined_text)
        word_count = len(words)
        term_frequencies = Counter(words)
        
        score = 0.0
        
        # BM25-like scoring parameters
        k1 = 1.2
        b = 0.75
        avgdl = 500  # Assume average document length
        
        for term in query_terms:
            tf = term_frequencies.get(term, 0)
            if tf > 0:
                # BM25 term score
                idf = 1.0  # Simplified, would need document collection stats
                term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (word_count / avgdl)))
                score += term_score
                
                # Boost for title matches
                if term in title:
                    score *= self.config["boost_factors"]["title_match"]
                
                # Boost for filename matches
                if term in file_name:
                    score *= 1.2
        
        # Boost for exact phrase matches
        if len(query_terms) > 1:
            query_phrase = " ".join(query_terms)
            if query_phrase in content:
                score *= self.config["boost_factors"]["exact_phrase"]
        
        # Boost based on document type preferences
        doc_type = metadata.get("doc_type", "")
        if doc_type in ["guide", "tutorial", "api_reference"]:
            score *= self.config["boost_factors"]["doc_type_preference"]
        
        # Normalize score
        max_possible_score = len(query_terms) * 10  # Rough normalization
        normalized_score = min(1.0, score / max_possible_score)
        
        return normalized_score
    
    def _fuse_results(self, semantic_results: List[Dict[str, Any]], 
                     keyword_results: List[Dict[str, Any]],
                     semantic_weight: float, keyword_weight: float,
                     query: str) -> List[Dict[str, Any]]:
        """Fuse semantic and keyword search results."""
        
        if self.config["fusion_method"] == "rrf":
            return self._reciprocal_rank_fusion(semantic_results, keyword_results, query)
        else:
            return self._weighted_score_fusion(
                semantic_results, keyword_results, 
                semantic_weight, keyword_weight, query
            )
    
    def _reciprocal_rank_fusion(self, semantic_results: List[Dict[str, Any]],
                               keyword_results: List[Dict[str, Any]],
                               query: str) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine results."""
        k = self.config["rrf_k"]
        
        # Create document ID mapping
        doc_scores = {}
        doc_data = {}
        
        # Process semantic results
        for rank, doc in enumerate(semantic_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = doc
            doc_data[doc_id]["semantic_rank"] = rank
            doc_data[doc_id]["semantic_score"] = doc.get("score", 0)
        
        # Process keyword results
        for rank, doc in enumerate(keyword_results, 1):
            doc_id = self._get_doc_id(doc)
            rrf_score = 1.0 / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc
            doc_data[doc_id]["keyword_rank"] = rank
            doc_data[doc_id]["keyword_score"] = doc.get("score", 0)
        
        # Sort by RRF score and create final results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            doc = doc_data[doc_id]
            doc["hybrid_score"] = rrf_score
            doc["score"] = rrf_score  # Update main score
            doc["fusion_method"] = "rrf"
            fused_results.append(doc)
        
        return fused_results
    
    def _weighted_score_fusion(self, semantic_results: List[Dict[str, Any]],
                              keyword_results: List[Dict[str, Any]],
                              semantic_weight: float, keyword_weight: float,
                              query: str) -> List[Dict[str, Any]]:
        """Fuse results using weighted score combination."""
        doc_scores = {}
        doc_data = {}
        
        # Normalize scores within each result set
        sem_scores = [doc["score"] for doc in semantic_results] if semantic_results else [0]
        key_scores = [doc["score"] for doc in keyword_results] if keyword_results else [0]
        
        sem_max = max(sem_scores) if sem_scores else 1
        key_max = max(key_scores) if key_scores else 1
        
        # Process semantic results
        for doc in semantic_results:
            doc_id = self._get_doc_id(doc)
            normalized_score = doc["score"] / sem_max if sem_max > 0 else 0
            weighted_score = normalized_score * semantic_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            doc_data[doc_id] = doc
            doc_data[doc_id]["semantic_score"] = doc["score"]
        
        # Process keyword results
        for doc in keyword_results:
            doc_id = self._get_doc_id(doc)
            normalized_score = doc["score"] / key_max if key_max > 0 else 0
            weighted_score = normalized_score * keyword_weight
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + weighted_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc
            doc_data[doc_id]["keyword_score"] = doc["score"]
        
        # Sort by weighted score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for doc_id, weighted_score in sorted_docs:
            doc = doc_data[doc_id]
            doc["hybrid_score"] = weighted_score
            doc["score"] = weighted_score  # Update main score
            doc["fusion_method"] = "weighted"
            fused_results.append(doc)
        
        return fused_results
    
    def _get_doc_id(self, doc: Dict[str, Any]) -> str:
        """Generate a unique document ID for deduplication."""
        # Use content hash or metadata-based ID
        metadata = doc.get("metadata", {})
        file_name = metadata.get("file_name", "")
        chunk_id = str(doc.get("id", ""))
        content_preview = doc.get("content", "")[:50]
        
        return f"{file_name}_{chunk_id}_{hash(content_preview)}"
    
    def _rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Apply re-ranking to improve result quality."""
        
        if self.config["reranking_method"] == "cross_encoder":
            return self._cross_encoder_rerank(results, query)
        else:
            return self._feature_based_rerank(results, query)
    
    def _cross_encoder_rerank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank using cross-encoder approach (simplified)."""
        # This is a simplified version - in production you'd use a proper cross-encoder model
        # For now, we'll use query-document similarity features
        
        reranked_results = []
        
        for doc in results:
            content = doc.get("content", "")
            
            # Calculate re-ranking score based on multiple factors
            rerank_score = doc.get("hybrid_score", 0)
            
            # Query term coverage in document
            query_terms = self._extract_query_terms(query)
            if query_terms:
                term_coverage = sum(1 for term in query_terms if term.lower() in content.lower())
                coverage_score = term_coverage / len(query_terms)
                rerank_score += coverage_score * 0.2
            
            # Document length penalty (prefer medium-length documents)
            content_length = len(content.split())
            if 50 <= content_length <= 500:
                rerank_score += 0.1
            elif content_length < 20:
                rerank_score -= 0.1
            
            # Metadata quality bonus
            metadata = doc.get("metadata", {})
            if metadata.get("title"):
                rerank_score += 0.05
            if metadata.get("doc_type") in ["guide", "tutorial", "api_reference"]:
                rerank_score += 0.1
            
            doc["rerank_score"] = rerank_score
            doc["score"] = rerank_score  # Update main score
            reranked_results.append(doc)
        
        # Sort by re-ranking score
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results
    
    def _feature_based_rerank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Re-rank based on hand-crafted features."""
        # Simplified feature-based re-ranking
        return self._cross_encoder_rerank(results, query)  # Use same logic for now
    
    def _update_hybrid_stats(self, semantic_time: float, keyword_time: float, 
                           fusion_time: float, total_time: float):
        """Update hybrid search statistics."""
        searches = self.hybrid_stats["hybrid_searches"]
        
        # Update averages
        self.hybrid_stats["avg_search_time"] = (
            (self.hybrid_stats["avg_search_time"] * (searches - 1) + total_time) / searches
        )
        self.hybrid_stats["avg_semantic_time"] = (
            (self.hybrid_stats["avg_semantic_time"] * (searches - 1) + semantic_time) / searches
        )
        self.hybrid_stats["avg_keyword_time"] = (
            (self.hybrid_stats["avg_keyword_time"] * (searches - 1) + keyword_time) / searches
        )
        self.hybrid_stats["avg_fusion_time"] = (
            (self.hybrid_stats["avg_fusion_time"] * (searches - 1) + fusion_time) / searches
        )
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update hybrid search configuration."""
        self.config.update(config_updates)
        logger.info(f"Hybrid search configuration updated: {config_updates}")
    
    def get_hybrid_analytics(self) -> Dict[str, Any]:
        """Get comprehensive hybrid search analytics."""
        return {
            "hybrid_stats": self.hybrid_stats,
            "configuration": self.config,
            "performance": {
                "avg_speedup_vs_semantic": (
                    self.hybrid_stats["avg_semantic_time"] / 
                    max(1, self.hybrid_stats["avg_search_time"])
                ) if self.hybrid_stats["avg_search_time"] > 0 else 1,
                "reranking_usage": (
                    self.hybrid_stats["reranking_applied"] /
                    max(1, self.hybrid_stats["hybrid_searches"])
                ) * 100
            }
        }
    
    def semantic_only_search(self, query: str, top_k: int = None, 
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic-only search for comparison."""
        return self._semantic_search(query, top_k or settings.top_k_retrieval, filters)
    
    def keyword_only_search(self, query: str, top_k: int = None,
                          filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform keyword-only search for comparison."""
        return self._keyword_search(query, top_k or settings.top_k_retrieval, filters)


def create_hybrid_searcher(collection_name: str = None) -> HybridSearcher:
    """Create and initialize hybrid searcher."""
    return HybridSearcher(collection_name=collection_name)