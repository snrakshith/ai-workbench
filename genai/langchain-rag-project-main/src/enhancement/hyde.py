"""HyDE (Hypothetical Document Embeddings) implementation for query enhancement."""

import logging
from typing import List, Dict, Any, Optional
import time

import google.genai as genai

from src.config import settings
from src.embedding.embedding_generator import GeminiEmbeddingGenerator

logger = logging.getLogger(__name__)


class HyDEQueryEnhancer:
    """HyDE implementation for improving retrieval through hypothetical document generation."""
    
    def __init__(self, llm_model: str = None, embedding_model: str = None, api_key: str = None):
        self.llm_model = llm_model or settings.llm_model
        self.embedding_model = embedding_model or settings.embedding_model
        self.api_key = api_key or settings.google_api_key
        
        if not self.api_key:
            raise ValueError("Google API key is required for HyDE implementation")
        
        # Initialize Gemini for hypothetical document generation
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize embedding generator for hypothetical documents
        self.embedding_generator = GeminiEmbeddingGenerator(
            model_name=self.embedding_model,
            api_key=self.api_key
        )
        
        # HyDE configuration
        self.config = {
            "num_hypothetical_docs": 3,
            "max_doc_length": 400,
            "temperature": 0.7,
            "use_query_fusion": True,
            "fusion_weights": {
                "original_query": 0.3,
                "hypothetical_docs": 0.7
            }
        }
        
        # Statistics tracking
        self.hyde_stats = {
            "total_enhancements": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0,
            "avg_embedding_time": 0
        }
        
        logger.info(f"✅ HyDE Query Enhancer initialized")
        logger.info(f"🤖 LLM Model: {self.llm_model}")
        logger.info(f"📝 Embedding Model: {self.embedding_model}")
    
    def enhance_query(self, query: str, domain_context: str = "LangChain", 
                     num_hypothetical: int = None) -> Dict[str, Any]:
        """
        Enhance a query using HyDE methodology.
        
        Args:
            query: Original user query
            domain_context: Domain context for document generation
            num_hypothetical: Number of hypothetical documents to generate
        
        Returns:
            Dict containing enhanced query information and embeddings
        """
        start_time = time.time()
        self.hyde_stats["total_enhancements"] += 1
        
        num_docs = num_hypothetical or self.config["num_hypothetical_docs"]
        
        try:
            # Step 1: Generate hypothetical documents
            generation_start = time.time()
            hypothetical_docs = self._generate_hypothetical_documents(
                query, domain_context, num_docs
            )
            generation_time = (time.time() - generation_start) * 1000
            
            if not hypothetical_docs:
                return self._handle_generation_failure(query, generation_time)
            
            # Step 2: Generate embeddings for hypothetical documents
            embedding_start = time.time()
            enhanced_embeddings = self._create_enhanced_embeddings(
                query, hypothetical_docs
            )
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Step 3: Update statistics
            total_time = (time.time() - start_time) * 1000
            self._update_hyde_stats(generation_time, embedding_time, total_time, True)
            
            return {
                "success": True,
                "original_query": query,
                "hypothetical_documents": hypothetical_docs,
                "enhanced_embeddings": enhanced_embeddings,
                "fusion_strategy": "weighted_average",
                "performance": {
                    "total_time_ms": total_time,
                    "generation_time_ms": generation_time,
                    "embedding_time_ms": embedding_time,
                    "hypothetical_docs_generated": len(hypothetical_docs)
                },
                "metadata": {
                    "domain_context": domain_context,
                    "model_used": self.llm_model,
                    "embedding_model": self.embedding_model,
                    "fusion_weights": self.config["fusion_weights"]
                }
            }
            
        except Exception as e:
            logger.error(f"HyDE enhancement failed: {str(e)}")
            self._update_hyde_stats(0, 0, (time.time() - start_time) * 1000, False)
            
            return {
                "success": False,
                "original_query": query,
                "error": str(e),
                "enhanced_embeddings": None,
                "hypothetical_documents": [],
                "performance": {
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "generation_time_ms": 0,
                    "embedding_time_ms": 0,
                    "hypothetical_docs_generated": 0
                }
            }
    
    def _generate_hypothetical_documents(self, query: str, domain_context: str, 
                                       num_docs: int) -> List[str]:
        """Generate hypothetical documents that would answer the query."""
        
        prompt = f"""You are an expert in {domain_context} documentation. Generate {num_docs} different hypothetical document passages that would contain the answer to this question: "{query}"

Each passage should:
1. Be 2-3 paragraphs long ({self.config["max_doc_length"]} words max)
2. Focus on a different aspect or approach to answering the question
3. Include technical details, code examples, or specific implementation guidance
4. Sound like authentic documentation from the {domain_context} library
5. Use realistic class names, method signatures, and parameter details

Generate diverse passages that cover different perspectives on the query. Format as:

PASSAGE 1:
[First hypothetical document passage]

PASSAGE 2:
[Second hypothetical document passage]

PASSAGE 3:
[Third hypothetical document passage]"""

        try:
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=[{"parts": [{"text": prompt}]}],
                config={
                    "max_output_tokens": 3000,  # Increased token limit
                    "temperature": self.config["temperature"],
                    "top_p": 0.9
                }
            )
            
            # Check for response and handle MAX_TOKENS issue
            if not response:
                raise Exception("No response from model")
            
            # Check if response was truncated due to token limit
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and str(candidate.finish_reason) == 'FinishReason.MAX_TOKENS':
                    logger.warning("Response truncated due to MAX_TOKENS limit, trying with shorter prompt")
                    # Try again with a shorter, more focused prompt
                    shorter_prompt = f"""Generate 2 hypothetical LangChain documentation passages about: "{query}"

PASSAGE 1:
[First passage - 1-2 paragraphs]

PASSAGE 2:
[Second passage - 1-2 paragraphs]"""
                    
                    response = self.client.models.generate_content(
                        model=self.llm_model,
                        contents=[{"parts": [{"text": shorter_prompt}]}],
                        config={
                            "max_output_tokens": 1500,
                            "temperature": self.config["temperature"],
                            "top_p": 0.9
                        }
                    )
            
            if not hasattr(response, 'text') or not response.text:
                logger.error(f"Empty response - response: {response}, text_length: {len(response.text) if hasattr(response, 'text') and response.text else 0}")
                raise Exception("Empty response from model")
            
            # Parse the response to extract individual passages
            hypothetical_docs = []
            current_passage = ""
            
            # Split by PASSAGE markers and clean up
            text_parts = response.text.split('PASSAGE ')
            
            for part in text_parts[1:]:  # Skip the first empty part
                # Remove the passage number and extract content
                if ':' in part:
                    content = part.split(':', 1)[1].strip()
                    if len(content.split()) >= 50:  # Minimum 50 words
                        hypothetical_docs.append(content)
            
            # Fallback: if no passages found with markers, try splitting by double newlines
            if not hypothetical_docs:
                logger.warning("No PASSAGE markers found, trying alternative parsing")
                paragraphs = [p.strip() for p in response.text.split('\n\n') if p.strip()]
                for paragraph in paragraphs:
                    if len(paragraph.split()) >= 50:  # Minimum 50 words
                        hypothetical_docs.append(paragraph)
            
            
            logger.info(f"Generated {len(hypothetical_docs)} hypothetical documents for query: {query[:50]}...")
            return hypothetical_docs
            
        except Exception as e:
            logger.error(f"Failed to generate hypothetical documents: {str(e)}")
            return []
    
    def _create_enhanced_embeddings(self, original_query: str, 
                                  hypothetical_docs: List[str]) -> Dict[str, Any]:
        """Create enhanced embeddings using query fusion."""
        
        try:
            # Generate embedding for original query
            original_embedding = self.embedding_generator.generate_embedding(original_query)
            if not original_embedding:
                raise Exception("Failed to generate embedding for original query")
            
            # Generate embeddings for hypothetical documents
            hypothetical_embeddings = []
            for i, doc in enumerate(hypothetical_docs):
                embedding = self.embedding_generator.generate_embedding(doc)
                if embedding:
                    hypothetical_embeddings.append({
                        "document_index": i,
                        "embedding": embedding,
                        "text_preview": doc[:200] + "..." if len(doc) > 200 else doc
                    })
            
            if not hypothetical_embeddings:
                raise Exception("Failed to generate embeddings for hypothetical documents")
            
            # Create fused embedding using weighted average
            if self.config["use_query_fusion"]:
                fused_embedding = self._fuse_embeddings(
                    original_embedding, 
                    [emb["embedding"] for emb in hypothetical_embeddings]
                )
            else:
                # Simple average of hypothetical document embeddings
                fused_embedding = self._average_embeddings([emb["embedding"] for emb in hypothetical_embeddings])
            
            return {
                "original_query_embedding": original_embedding,
                "hypothetical_embeddings": hypothetical_embeddings,
                "fused_embedding": fused_embedding,
                "fusion_method": "weighted_average" if self.config["use_query_fusion"] else "simple_average",
                "embedding_dimension": len(fused_embedding)
            }
            
        except Exception as e:
            logger.error(f"Failed to create enhanced embeddings: {str(e)}")
            return None
    
    def _fuse_embeddings(self, original_embedding: List[float], 
                        hypothetical_embeddings: List[List[float]]) -> List[float]:
        """Fuse original query embedding with hypothetical document embeddings."""
        
        # Get fusion weights
        original_weight = self.config["fusion_weights"]["original_query"]
        hypothetical_weight = self.config["fusion_weights"]["hypothetical_docs"]
        
        # Calculate weighted average of hypothetical embeddings
        avg_hypothetical = self._average_embeddings(hypothetical_embeddings)
        
        # Fuse original and average hypothetical embeddings
        fused_embedding = []
        for i in range(len(original_embedding)):
            fused_value = (
                original_weight * original_embedding[i] + 
                hypothetical_weight * avg_hypothetical[i]
            )
            fused_embedding.append(fused_value)
        
        return fused_embedding
    
    def _average_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate average of multiple embeddings."""
        if not embeddings:
            return []
        
        dimension = len(embeddings[0])
        averaged = [0.0] * dimension
        
        for embedding in embeddings:
            for i in range(dimension):
                averaged[i] += embedding[i]
        
        # Normalize by count
        for i in range(dimension):
            averaged[i] /= len(embeddings)
        
        return averaged
    
    def _handle_generation_failure(self, query: str, generation_time: float) -> Dict[str, Any]:
        """Handle case when hypothetical document generation fails."""
        return {
            "success": False,
            "original_query": query,
            "error": "Failed to generate hypothetical documents",
            "enhanced_embeddings": None,
            "hypothetical_documents": [],
            "performance": {
                "total_time_ms": generation_time,
                "generation_time_ms": generation_time,
                "embedding_time_ms": 0,
                "hypothetical_docs_generated": 0
            }
        }
    
    def _update_hyde_stats(self, generation_time: float, embedding_time: float, 
                          total_time: float, success: bool):
        """Update HyDE statistics."""
        if success:
            self.hyde_stats["successful_generations"] += 1
            
            # Update averages
            successful = self.hyde_stats["successful_generations"]
            self.hyde_stats["avg_generation_time"] = (
                (self.hyde_stats["avg_generation_time"] * (successful - 1) + generation_time) /
                successful
            )
            self.hyde_stats["avg_embedding_time"] = (
                (self.hyde_stats["avg_embedding_time"] * (successful - 1) + embedding_time) /
                successful
            )
        else:
            self.hyde_stats["failed_generations"] += 1
    
    def batch_enhance_queries(self, queries: List[str], domain_context: str = "LangChain") -> List[Dict[str, Any]]:
        """Enhance multiple queries using HyDE."""
        start_time = time.time()
        
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Enhancing query {i+1}/{len(queries)}: {query[:50]}...")
            
            result = self.enhance_query(query, domain_context)
            results.append({
                "batch_index": i,
                "query": query,
                **result
            })
        
        batch_time = (time.time() - start_time) * 1000
        logger.info(f"Completed HyDE batch enhancement: {len(queries)} queries in {batch_time:.1f}ms")
        
        return results
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update HyDE configuration."""
        self.config.update(config_updates)
        logger.info(f"HyDE configuration updated: {config_updates}")
    
    def get_hyde_analytics(self) -> Dict[str, Any]:
        """Get HyDE performance analytics."""
        return {
            "hyde_stats": self.hyde_stats,
            "configuration": self.config,
            "success_rate": (
                self.hyde_stats["successful_generations"] /
                max(1, self.hyde_stats["total_enhancements"])
            ) * 100,
            "models": {
                "llm_model": self.llm_model,
                "embedding_model": self.embedding_model
            }
        }


def create_hyde_enhancer(llm_model: str = None, embedding_model: str = None, 
                        api_key: str = None) -> HyDEQueryEnhancer:
    """Create and initialize HyDE query enhancer."""
    return HyDEQueryEnhancer(llm_model=llm_model, embedding_model=embedding_model, api_key=api_key)