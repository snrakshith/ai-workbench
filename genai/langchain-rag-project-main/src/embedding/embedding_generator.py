"""Embedding generation using Google Gemini models."""

import os
import time
import logging
from typing import List, Dict, Any, Optional
import json
import pickle
from pathlib import Path
import numpy as np
from tqdm import tqdm
import google.genai as genai

from src.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingGenerator:
    """Embedding generator using Google Gemini models."""
    
    def __init__(self, model_name: str = None, api_key: str = None, 
                 batch_size: int = 32, max_retries: int = 3, 
                 output_dimensionality: int = None):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.output_dimensionality = output_dimensionality or settings.embedding_dimension
        
        # Configure Gemini API
        api_key = api_key or settings.google_api_key
        if not api_key:
            raise ValueError("Google API key is required")
        
        self.client = genai.Client(api_key=api_key)
        
        # Test the connection
        self._test_connection()
        
        # Cache for embeddings
        self.cache_dir = Path(settings.embeddings_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embedding_cache.json"
        self.cache = self._load_cache()
    
    def _test_connection(self):
        """Test the Gemini API connection."""
        try:
            # Use the text-embedding-004 model for testing
            config = {}
            if self.output_dimensionality:
                config['output_dimensionality'] = self.output_dimensionality
            
            test_result = self.client.models.embed_content(
                model=self.model_name,
                contents=["test connection"],
                config=config if config else None
            )
            
            if hasattr(test_result, 'embeddings') and test_result.embeddings:
                logger.info(f"✅ Gemini API connection successful")
                logger.info(f"📐 Embedding dimension: {len(test_result.embeddings[0].values)}")
            else:
                raise Exception("No embedding in response")
                
        except Exception as e:
            logger.error(f"❌ Gemini API connection failed: {str(e)}")
            raise
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"📂 Loaded {len(cache)} embeddings from cache")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
        return {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
            logger.debug(f"💾 Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def generate_embedding(self, text: str, retry_count: int = 0) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not text.strip():
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Generate embedding using Gemini
            config = {}
            if self.output_dimensionality:
                config['output_dimensionality'] = self.output_dimensionality
            
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=config if config else None
            )
            
            if hasattr(result, 'embeddings') and result.embeddings:
                embedding = result.embeddings[0].values
                
                # Cache the result
                self.cache[cache_key] = embedding
                
                return embedding
            else:
                logger.warning(f"No embedding in API response")
                return None
                
        except Exception as e:
            if retry_count < self.max_retries:
                wait_time = (2 ** retry_count) + 1  # Exponential backoff
                logger.warning(f"API error (attempt {retry_count + 1}): {str(e)}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self.generate_embedding(text, retry_count + 1)
            else:
                logger.error(f"Failed to generate embedding after {self.max_retries} retries: {str(e)}")
                return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        
        # Process in batches to respect API limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)
                
                # Small delay to respect rate limits
                time.sleep(0.1)
            
            embeddings.extend(batch_embeddings)
            
            # Save cache periodically
            if i % (self.batch_size * 5) == 0:
                self._save_cache()
            
            logger.info(f"Generated embeddings for batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
        
        # Final cache save
        self._save_cache()
        
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks."""
        if not chunks:
            return []
        
        logger.info(f"🚀 Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts for embedding
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings with progress bar
        embeddings = []
        failed_count = 0
        
        with tqdm(total=len(texts), desc="Generating embeddings") as pbar:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = []
                
                for text in batch_texts:
                    embedding = self.generate_embedding(text)
                    batch_embeddings.append(embedding)
                    
                    if embedding is None:
                        failed_count += 1
                    
                    pbar.update(1)
                    time.sleep(0.1)  # Rate limiting
                
                embeddings.extend(batch_embeddings)
                
                # Save cache periodically
                if i % (self.batch_size * 5) == 0:
                    self._save_cache()
        
        # Final cache save
        self._save_cache()
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is not None:
                embedded_chunk = {
                    **chunk,
                    "embedding": embedding,
                    "embedding_model": self.model_name,
                    "embedding_dimension": len(embedding)
                }
                embedded_chunks.append(embedded_chunk)
        
        logger.info(f"✅ Generated {len(embedded_chunks)} embeddings ({failed_count} failed)")
        
        return embedded_chunks
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about generated embeddings."""
        if not self.cache:
            return {"total_embeddings": 0}
        
        # Load a sample embedding to get dimension
        sample_embedding = next(iter(self.cache.values()))
        
        return {
            "total_embeddings": len(self.cache),
            "embedding_dimension": len(sample_embedding),
            "model_name": self.model_name,
            "cache_size_mb": self.cache_file.stat().st_size / 1024 / 1024 if self.cache_file.exists() else 0
        }


class EmbeddingPipeline:
    """Pipeline for generating and managing embeddings."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or settings.embeddings_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = GeminiEmbeddingGenerator()
        self.pipeline_stats = {
            "start_time": None,
            "end_time": None,
            "total_chunks": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "embedding_dimension": None
        }
    
    def process_chunked_documents(self, chunked_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Process chunked documents and generate embeddings."""
        from datetime import datetime
        
        self.pipeline_stats["start_time"] = datetime.now().isoformat()
        
        try:
            embedded_docs = {}
            total_chunks = 0
            successful_embeddings = 0
            
            for doc_type, chunks in chunked_docs.items():
                if not chunks:
                    embedded_docs[doc_type] = []
                    continue
                
                logger.info(f"🔄 Processing {len(chunks)} {doc_type} chunks...")
                
                # Generate embeddings
                embedded_chunks = self.generator.embed_chunks(chunks)
                embedded_docs[doc_type] = embedded_chunks
                
                total_chunks += len(chunks)
                successful_embeddings += len(embedded_chunks)
                
                # Save embedded chunks
                output_file = self.output_dir / f"{doc_type}_embeddings.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(embedded_chunks, f, indent=2)
                
                logger.info(f"💾 Saved {len(embedded_chunks)} embedded {doc_type} chunks")
            
            # Update pipeline stats
            self.pipeline_stats["end_time"] = datetime.now().isoformat()
            self.pipeline_stats["total_chunks"] = total_chunks
            self.pipeline_stats["successful_embeddings"] = successful_embeddings
            self.pipeline_stats["failed_embeddings"] = total_chunks - successful_embeddings
            
            # Get embedding dimension from first successful embedding
            for chunks in embedded_docs.values():
                if chunks and "embedding" in chunks[0]:
                    self.pipeline_stats["embedding_dimension"] = len(chunks[0]["embedding"])
                    break
            
            # Save comprehensive metadata
            self._save_embedding_metadata(embedded_docs)
            
            return {
                "success": True,
                "embedded_documents": embedded_docs,
                "pipeline_stats": self.pipeline_stats,
                "generator_stats": self.generator.get_embedding_stats()
            }
            
        except Exception as e:
            logger.error(f"Embedding pipeline failed: {str(e)}")
            self.pipeline_stats["end_time"] = datetime.now().isoformat()
            return {
                "success": False,
                "error": str(e),
                "pipeline_stats": self.pipeline_stats
            }
    
    def _save_embedding_metadata(self, embedded_docs: Dict[str, List[Dict[str, Any]]]):
        """Save comprehensive embedding metadata."""
        metadata = {
            "pipeline_stats": self.pipeline_stats,
            "generator_stats": self.generator.get_embedding_stats(),
            "document_stats": {},
            "chunk_statistics": {}
        }
        
        # Calculate statistics per document type
        for doc_type, chunks in embedded_docs.items():
            if not chunks:
                continue
                
            chunk_sizes = [chunk["metadata"]["chunk_size"] for chunk in chunks]
            token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
            
            metadata["document_stats"][doc_type] = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "avg_token_count": sum(token_counts) / len(token_counts) if token_counts else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0
            }
        
        # Overall chunk statistics
        all_chunks = [chunk for chunks in embedded_docs.values() for chunk in chunks]
        if all_chunks:
            all_chunk_sizes = [chunk["metadata"]["chunk_size"] for chunk in all_chunks]
            all_token_counts = [chunk["metadata"]["token_count"] for chunk in all_chunks]
            
            metadata["chunk_statistics"] = {
                "total_chunks": len(all_chunks),
                "avg_chunk_size": sum(all_chunk_sizes) / len(all_chunk_sizes),
                "avg_token_count": sum(all_token_counts) / len(all_token_counts),
                "min_chunk_size": min(all_chunk_sizes),
                "max_chunk_size": max(all_chunk_sizes),
                "chunk_size_std": np.std(all_chunk_sizes),
                "token_count_std": np.std(all_token_counts)
            }
        
        # Save metadata
        metadata_file = self.output_dir / "embedding_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved embedding metadata to {metadata_file.name}")
    
    def load_embedded_documents(self, doc_type: str = None) -> List[Dict[str, Any]]:
        """Load embedded documents from disk."""
        if doc_type:
            file_path = self.output_dir / f"{doc_type}_embeddings.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        else:
            # Load all embedded documents
            all_embedded = []
            for file_path in self.output_dir.glob("*_embeddings.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_embedded.extend(json.load(f))
            return all_embedded


def generate_embeddings_for_chunks(chunked_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Convenience function to generate embeddings for chunked documents."""
    pipeline = EmbeddingPipeline()
    return pipeline.process_chunked_documents(chunked_docs)