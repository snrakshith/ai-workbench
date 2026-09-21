"""Vector database management using Qdrant."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, PointStruct,
    Filter, FieldCondition, Range, MatchValue, SearchRequest,
    UpdateCollection, OptimizersConfigDiff, CollectionStatus
)
from qdrant_client.http.exceptions import ResponseHandlingException

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant vector database management."""
    
    def __init__(self, collection_name: str = None, url: str = None, api_key: str = None):
        self.collection_name = collection_name or settings.collection_name
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        
        if not self.url or not self.api_key:
            raise ValueError("Qdrant URL and API key are required")
        
        # Initialize client with timeout settings
        self.client = QdrantClient(
            url=self.url, 
            api_key=self.api_key,
            timeout=60,  # 60 second timeout for operations
            prefer_grpc=False  # Use REST API for better timeout handling
        )
        
        # Test connection
        self._test_connection()
        
        # Collection metadata
        self.collection_info = None
        self.vector_dimension = None
        
    def _test_connection(self):
        """Test Qdrant connection."""
        try:
            collections = self.client.get_collections()
            logger.info(f"✅ Qdrant connection successful, {len(collections.collections)} collections found")
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {str(e)}")
            raise
    
    def create_collection(self, vector_dimension: int = 3072, distance: str = "Cosine", 
                         force_recreate: bool = False) -> bool:
        """Create or recreate the collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if collection_exists and not force_recreate:
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection_info = self.client.get_collection(self.collection_name)
                self.vector_dimension = self.collection_info.config.params.vectors.size
                return True
            
            if collection_exists and force_recreate:
                logger.info(f"Deleting existing collection '{self.collection_name}'")
                self.client.delete_collection(self.collection_name)
            
            # Create collection
            logger.info(f"Creating collection '{self.collection_name}' with dimension {vector_dimension}")
            
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_dimension,
                    distance=distance_map.get(distance, Distance.COSINE)
                ),
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000,
                    memmap_threshold=20000
                )
            )
            
            # Update collection info
            self.collection_info = self.client.get_collection(self.collection_name)
            self.vector_dimension = vector_dimension
            
            logger.info(f"✅ Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if not self.collection_info:
                self.collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "points_count": self.collection_info.points_count,
                "segments_count": self.collection_info.segments_count,
                "vector_dimension": self.collection_info.config.params.vectors.size,
                "distance_metric": str(self.collection_info.config.params.vectors.distance),
                "status": str(self.collection_info.status),
                "optimizer_status": self.collection_info.optimizer_status
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {}
    
    def upload_embeddings(self, embedded_documents: List[Dict[str, Any]], 
                         batch_size: int = 50, show_progress: bool = True, 
                         max_retries: int = 3, retry_delay: float = 2.0) -> Dict[str, Any]:
        """Upload embedded documents to Qdrant."""
        try:
            if not embedded_documents:
                return {"success": False, "error": "No documents to upload"}
            
            # Ensure collection exists
            if not self.collection_info:
                # Detect vector dimension from first embedding
                first_embedding = embedded_documents[0].get('embedding')
                if not first_embedding:
                    return {"success": False, "error": "No embeddings found in documents"}
                
                vector_dim = len(first_embedding)
                if not self.create_collection(vector_dimension=vector_dim):
                    return {"success": False, "error": "Failed to create collection"}
            
            # Prepare points for upload
            points = []
            failed_uploads = []
            
            for i, doc in enumerate(embedded_documents):
                try:
                    embedding = doc.get('embedding')
                    if not embedding:
                        failed_uploads.append(i)
                        continue
                    
                    # Prepare metadata (Qdrant payload)
                    payload = {
                        "content": doc.get("content", ""),
                        "doc_type": doc.get("metadata", {}).get("doc_type", "unknown"),
                        "source_file": doc.get("metadata", {}).get("source_file", ""),
                        "module_path": doc.get("metadata", {}).get("module_path", ""),
                        "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                        "token_count": doc.get("metadata", {}).get("token_count", 0),
                        "chunk_size": doc.get("metadata", {}).get("chunk_size", 0),
                        "file_name": doc.get("metadata", {}).get("file_name", ""),
                        "title": doc.get("metadata", {}).get("title", ""),
                        "url": doc.get("metadata", {}).get("url", ""),
                        "upload_timestamp": datetime.now().isoformat()
                    }
                    
                    point = PointStruct(
                        id=i,
                        vector=embedding,
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare point {i}: {str(e)}")
                    failed_uploads.append(i)
            
            if not points:
                return {"success": False, "error": "No valid points to upload"}
            
            # Upload in batches with retry logic
            total_uploaded = 0
            upload_errors = []
            
            logger.info(f"Uploading {len(points)} points in batches of {batch_size}")
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_number = i // batch_size + 1
                batch_uploaded = False
                
                # Retry logic for each batch
                for attempt in range(max_retries + 1):
                    try:
                        if attempt > 0:
                            logger.info(f"Retrying batch {batch_number}, attempt {attempt + 1}")
                            time.sleep(retry_delay * attempt)  # Exponential backoff
                        
                        operation_info = self.client.upsert(
                            collection_name=self.collection_name,
                            points=batch,
                            wait=True  # Wait for the operation to complete
                        )
                        
                        total_uploaded += len(batch)
                        batch_uploaded = True
                        
                        if show_progress:
                            logger.info(f"✅ Batch {batch_number}/{(len(points) + batch_size - 1) // batch_size} uploaded ({total_uploaded}/{len(points)} points)")
                        
                        # Small delay between successful batches to avoid overwhelming the server
                        time.sleep(0.5)
                        break
                        
                    except Exception as e:
                        if "timeout" in str(e).lower():
                            logger.warning(f"⏰ Batch {batch_number} timed out (attempt {attempt + 1}): {str(e)}")
                        else:
                            logger.error(f"❌ Batch {batch_number} failed (attempt {attempt + 1}): {str(e)}")
                        
                        if attempt == max_retries:
                            error_msg = f"Batch {batch_number} failed after {max_retries + 1} attempts: {str(e)}"
                            logger.error(error_msg)
                            upload_errors.append(error_msg)
                
                if not batch_uploaded:
                    logger.error(f"❌ Skipping batch {batch_number} after all retry attempts failed")
            
            # Update collection info
            self.collection_info = self.client.get_collection(self.collection_name)
            
            result = {
                "success": True,
                "total_documents": len(embedded_documents),
                "total_uploaded": total_uploaded,
                "failed_preparations": len(failed_uploads),
                "upload_errors": len(upload_errors),
                "collection_stats": self.get_collection_stats()
            }
            
            if upload_errors:
                result["upload_error_details"] = upload_errors
                
            logger.info(f"✅ Upload completed: {total_uploaded}/{len(embedded_documents)} points")
            return result
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      filter_conditions: Dict[str, Any] = None, 
                      score_threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        try:
            # Prepare filter if provided
            query_filter = None
            if filter_conditions:
                filter_clauses = []
                
                for field, value in filter_conditions.items():
                    if isinstance(value, str):
                        filter_clauses.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )
                    elif isinstance(value, list):
                        filter_clauses.append(
                            FieldCondition(key=field, match=MatchValue(any=value))
                        )
                    elif isinstance(value, dict) and "range" in value:
                        range_val = value["range"]
                        filter_clauses.append(
                            FieldCondition(
                                key=field, 
                                range=Range(
                                    gte=range_val.get("gte"),
                                    lte=range_val.get("lte")
                                )
                            )
                        )
                
                if filter_clauses:
                    query_filter = Filter(must=filter_clauses)
            
            # Search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload.get("content", ""),
                    "metadata": {
                        "doc_type": hit.payload.get("doc_type"),
                        "source_file": hit.payload.get("source_file"),
                        "module_path": hit.payload.get("module_path"),
                        "chunk_index": hit.payload.get("chunk_index"),
                        "token_count": hit.payload.get("token_count"),
                        "file_name": hit.payload.get("file_name"),
                        "title": hit.payload.get("title"),
                        "url": hit.payload.get("url")
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' deleted")
            self.collection_info = None
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False
    
    def get_point(self, point_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific point by ID."""
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )
            
            if point:
                hit = point[0]
                return {
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "embedding": hit.vector,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "content"}
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve point {point_id}: {str(e)}")
            return None
    
    def update_collection_optimizer(self) -> bool:
        """Trigger collection optimization."""
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=10000,
                    memmap_threshold=10000
                )
            )
            logger.info(f"✅ Collection optimizer updated")
            return True
        except Exception as e:
            logger.error(f"Failed to update optimizer: {str(e)}")
            return False


def load_and_upload_embeddings(embeddings_dir: str = None) -> Dict[str, Any]:
    """Load embeddings from files and upload to Qdrant."""
    embeddings_dir = Path(embeddings_dir or settings.embeddings_path)
    
    if not embeddings_dir.exists():
        return {"success": False, "error": "Embeddings directory not found"}
    
    # Initialize vector store
    vector_store = QdrantVectorStore()
    
    # Load all embedding files
    embedding_files = list(embeddings_dir.glob("*_embeddings.json"))
    
    if not embedding_files:
        return {"success": False, "error": "No embedding files found"}
    
    all_embeddings = []
    for file_path in embedding_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
            all_embeddings.extend(embeddings)
            logger.info(f"Loaded {len(embeddings)} embeddings from {file_path.name}")
    
    if not all_embeddings:
        return {"success": False, "error": "No embeddings found in files"}
    
    # Upload to Qdrant with optimized settings for large datasets
    logger.info(f"Uploading {len(all_embeddings)} embeddings to Qdrant...")
    result = vector_store.upload_embeddings(
        all_embeddings, 
        batch_size=25,  # Smaller batch size for large uploads
        show_progress=True,
        max_retries=5,  # More retries for reliability
        retry_delay=3.0  # Longer delay between retries
    )
    
    return {
        **result,
        "vector_store_stats": vector_store.get_collection_stats(),
        "files_loaded": [f.name for f in embedding_files]
    }