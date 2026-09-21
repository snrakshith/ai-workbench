"""Memory service for storing, retrieving, and managing conversation memories with Mem0."""

import logging
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from mem0 import MemoryClient
from .mem0_config import get_mem0_client, get_mem0_config, is_memory_enabled

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories that can be stored."""
    USER_PROFILE = "user_profile"      # Persistent user preferences and characteristics
    SESSION_CONTEXT = "session"       # Session-specific context (using run_id)
    CONVERSATION = "conversation"      # General conversation history
    PREFERENCE = "preference"          # User preferences and settings
    KNOWLEDGE = "knowledge"            # Learned facts about the user
    FEEDBACK = "feedback"              # User feedback and corrections


class MemoryMetadata:
    """Helper class for managing memory metadata."""
    
    @staticmethod
    def create_metadata(
        memory_type: MemoryType,
        category: str = None,
        importance: str = "medium",
        tags: List[str] = None,
        source: str = "rag_system",
        **kwargs
    ) -> Dict[str, Any]:
        """Create standardized metadata for memories."""
        metadata = {
            "memory_type": memory_type.value,
            "source": source,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            **kwargs
        }
        
        if category:
            metadata["category"] = category
        
        if tags:
            metadata["tags"] = tags
        
        return metadata


class MemoryService:
    """Service for managing conversation memories with Mem0."""
    
    def __init__(self):
        """Initialize the memory service."""
        self.client: Optional[MemoryClient] = get_mem0_client()
        self.config = get_mem0_config()
        self.enabled = is_memory_enabled()
        
        if not self.enabled:
            logger.warning("⚠️ Memory service is disabled. Check Mem0 configuration.")
    
    def add_memory(
        self,
        content: Union[str, List[Dict[str, str]]],
        user_id: str,
        agent_id: str = "langchain_rag",
        run_id: Optional[str] = None,
        memory_type: MemoryType = MemoryType.CONVERSATION,
        category: Optional[str] = None,
        importance: str = "medium",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Add memory to the Mem0 store.
        
        Args:
            content: Memory content (string or list of messages)
            user_id: Unique identifier for the user
            agent_id: Identifier for the agent/system
            run_id: Optional session/conversation identifier
            memory_type: Type of memory being stored
            category: Category for organizing memories
            importance: Importance level (low, medium, high)
            tags: List of tags for the memory
            metadata: Additional metadata
            
        Returns:
            Result of the add operation or None if disabled
        """
        if not self.enabled or not self.client:
            logger.debug("Memory service is disabled")
            return None
        
        try:
            # Create metadata
            mem_metadata = MemoryMetadata.create_metadata(
                memory_type=memory_type,
                category=category,
                importance=importance,
                tags=tags,
                agent_id=agent_id,
                **{k: v for k, v in (metadata or {}).items() if k != 'agent_id'}
            )
            
            # Prepare the add parameters
            add_params = {
                "user_id": user_id,
                "metadata": mem_metadata
            }
            
            if run_id:
                add_params["run_id"] = run_id
            
            # Handle different content types
            if isinstance(content, str):
                # Single string content
                messages = [{"role": "user", "content": content}]
            elif isinstance(content, list):
                # List of messages
                messages = content
            else:
                logger.error(f"Invalid content type: {type(content)}")
                return None
            
            # Add to Mem0
            start_time = time.time()
            result = self.client.add(messages, **add_params)
            end_time = time.time()
            
            logger.info(f"✅ Memory added successfully in {end_time - start_time:.2f}s")
            logger.debug(f"Memory result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to add memory: {e}")
            return None
    
    def search_memory(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            user_id: Unique identifier for the user
            agent_id: Optional agent identifier for filtering
            run_id: Optional session identifier for filtering
            memory_type: Optional memory type filter
            category: Optional category filter
            limit: Maximum number of results
            threshold: Minimum relevance threshold
            
        Returns:
            List of relevant memories
        """
        if not self.enabled or not self.client:
            logger.debug("Memory service is disabled")
            return []
        
        try:
            # Use configuration defaults if not provided
            search_limit = limit or self.config.search_limit
            search_threshold = threshold or self.config.search_threshold
            
            # Build search parameters
            search_params = {
                "user_id": user_id,
                "limit": search_limit
            }
            
            if agent_id:
                search_params["agent_id"] = agent_id
            if run_id:
                search_params["run_id"] = run_id
            
            # Perform search
            start_time = time.time()
            results = self.client.search(query, **search_params)
            end_time = time.time()
            
            if not results or "results" not in results:
                logger.debug("No memories found")
                return []
            
            memories = results["results"]
            
            # Apply additional filtering
            filtered_memories = []
            for memory in memories:
                # Apply threshold filter
                score = memory.get("score", 0)
                if score < search_threshold:
                    continue
                
                # Apply metadata filters
                mem_metadata = memory.get("metadata", {})
                
                if memory_type and mem_metadata.get("memory_type") != memory_type.value:
                    continue
                
                if category and mem_metadata.get("category") != category:
                    continue
                
                filtered_memories.append(memory)
            
            logger.info(f"✅ Found {len(filtered_memories)} relevant memories in {end_time - start_time:.2f}s")
            return filtered_memories
            
        except Exception as e:
            logger.error(f"❌ Failed to search memories: {e}")
            return []
    
    def get_user_history(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.
        
        Args:
            user_id: Unique identifier for the user
            agent_id: Optional agent identifier
            run_id: Optional session identifier
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of user memories
        """
        if not self.enabled or not self.client:
            logger.debug("Memory service is disabled")
            return []
        
        try:
            # Build parameters for get_all
            params = {"user_id": user_id}
            
            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id
            
            # Get all memories
            start_time = time.time()
            memories = self.client.get_all(**params)
            end_time = time.time()
            
            if not memories:
                return []
            
            # Sort by creation time (newest first) and apply limit
            sorted_memories = sorted(
                memories,
                key=lambda x: x.get("created_at", ""),
                reverse=True
            )
            
            if limit:
                sorted_memories = sorted_memories[:limit]
            
            logger.info(f"✅ Retrieved {len(sorted_memories)} memories for user in {end_time - start_time:.2f}s")
            return sorted_memories
            
        except Exception as e:
            logger.error(f"❌ Failed to get user history: {e}")
            return []
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: New content for the memory
            metadata: New or additional metadata
            
        Returns:
            Updated memory or None if failed
        """
        if not self.enabled or not self.client:
            logger.debug("Memory service is disabled")
            return None
        
        try:
            update_data = {}
            
            if content is not None:
                update_data["text"] = content
            
            if metadata is not None:
                # Add timestamp for update
                metadata["updated_at"] = datetime.now().isoformat()
                update_data["metadata"] = metadata
            
            result = self.client.update(memory_id, **update_data)
            logger.info(f"✅ Memory {memory_id} updated successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to update memory {memory_id}: {e}")
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            logger.debug("Memory service is disabled")
            return False
        
        try:
            self.client.delete(memory_id)
            logger.info(f"✅ Memory {memory_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete memory {memory_id}: {e}")
            return False
    
    def clear_user_memories(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None
    ) -> int:
        """
        Clear memories for a user with optional filtering.
        
        Args:
            user_id: Unique identifier for the user
            agent_id: Optional agent identifier filter
            run_id: Optional session identifier filter
            memory_type: Optional memory type filter
            
        Returns:
            Number of memories deleted
        """
        if not self.enabled or not self.client:
            logger.debug("Memory service is disabled")
            return 0
        
        try:
            # Get all memories for the user
            memories = self.get_user_history(user_id, agent_id, run_id)
            
            deleted_count = 0
            for memory in memories:
                # Apply memory type filter if specified
                if memory_type:
                    mem_metadata = memory.get("metadata", {})
                    if mem_metadata.get("memory_type") != memory_type.value:
                        continue
                
                if self.delete_memory(memory["id"]):
                    deleted_count += 1
            
            logger.info(f"✅ Cleared {deleted_count} memories for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Failed to clear memories for user {user_id}: {e}")
            return 0
    
    def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored memories.
        
        Args:
            user_id: Optional user ID to get stats for specific user
            
        Returns:
            Dictionary containing memory statistics
        """
        if not self.enabled or not self.client:
            return {
                "enabled": False,
                "total_memories": 0,
                "message": "Memory service is disabled"
            }
        
        try:
            if user_id:
                memories = self.get_user_history(user_id)
                return {
                    "enabled": True,
                    "user_id": user_id,
                    "total_memories": len(memories),
                    "memory_types": self._count_memory_types(memories),
                    "categories": self._count_categories(memories)
                }
            else:
                return {
                    "enabled": True,
                    "service_status": "operational",
                    "config": self.config.to_dict()
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {
                "enabled": self.enabled,
                "error": str(e)
            }
    
    def _count_memory_types(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count memories by type."""
        type_counts = {}
        for memory in memories:
            metadata = memory.get("metadata", {})
            memory_type = metadata.get("memory_type", "unknown")
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        return type_counts
    
    def _count_categories(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count memories by category."""
        category_counts = {}
        for memory in memories:
            metadata = memory.get("metadata", {})
            category = metadata.get("category", "uncategorized")
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts


# Global memory service instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get the global memory service instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service