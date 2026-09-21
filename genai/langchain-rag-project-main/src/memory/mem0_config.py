"""Mem0 configuration and client initialization."""

import os
import logging
from typing import Optional, Dict, Any
from mem0 import MemoryClient
from src.config import settings

logger = logging.getLogger(__name__)


class Mem0Config:
    """Configuration class for Mem0 integration."""
    
    def __init__(self):
        """Initialize Mem0 configuration."""
        self.api_key = settings.mem0_api_key
        self.search_limit = settings.memory_search_limit
        self.search_threshold = settings.memory_search_threshold
        self.auto_save = settings.memory_auto_save
        self.window_size = settings.memory_window_size
        self.context_weight = settings.memory_context_weight
        self.enabled = settings.memory_enabled
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate Mem0 configuration."""
        if self.enabled and not self.api_key:
            logger.warning("MEM0_API_KEY not set. Memory features will be disabled.")
            self.enabled = False
        
        if self.search_limit < 1 or self.search_limit > 20:
            logger.warning(f"Invalid memory_search_limit: {self.search_limit}. Setting to 5.")
            self.search_limit = 5
        
        if not 0.0 <= self.search_threshold <= 1.0:
            logger.warning(f"Invalid memory_search_threshold: {self.search_threshold}. Setting to 0.7.")
            self.search_threshold = 0.7
        
        if not 0.0 <= self.context_weight <= 1.0:
            logger.warning(f"Invalid memory_context_weight: {self.context_weight}. Setting to 0.3.")
            self.context_weight = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_key": self.api_key,
            "search_limit": self.search_limit,
            "search_threshold": self.search_threshold,
            "auto_save": self.auto_save,
            "window_size": self.window_size,
            "context_weight": self.context_weight,
            "enabled": self.enabled
        }


class Mem0ClientManager:
    """Singleton manager for Mem0 client."""
    
    _instance: Optional['Mem0ClientManager'] = None
    _client: Optional[MemoryClient] = None
    _config: Optional[Mem0Config] = None
    
    def __new__(cls) -> 'Mem0ClientManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Mem0 client manager."""
        if self._config is None:
            self._config = Mem0Config()
            if self._config.enabled:
                self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Mem0 client."""
        try:
            if self._config.api_key:
                self._client = MemoryClient(api_key=self._config.api_key)
                logger.info("✅ Mem0 client initialized successfully")
            else:
                logger.warning("⚠️ Mem0 API key not provided. Memory features disabled.")
                self._config.enabled = False
        except Exception as e:
            logger.error(f"❌ Failed to initialize Mem0 client: {e}")
            self._config.enabled = False
    
    @property
    def client(self) -> Optional[MemoryClient]:
        """Get the Mem0 client."""
        if not self._config.enabled:
            return None
        return self._client
    
    @property
    def config(self) -> Mem0Config:
        """Get the Mem0 configuration."""
        return self._config
    
    @property
    def is_enabled(self) -> bool:
        """Check if memory features are enabled."""
        return self._config.enabled and self._client is not None
    
    def test_connection(self) -> bool:
        """Test the Mem0 connection."""
        if not self.is_enabled:
            return False
        
        try:
            # Try to get memories for a test user to verify connection
            self._client.search("test", user_id="connection_test", limit=1)
            logger.info("✅ Mem0 connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Mem0 connection test failed: {e}")
            return False
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get health information for the memory service."""
        return {
            "enabled": self.is_enabled,
            "client_initialized": self._client is not None,
            "config": self._config.to_dict() if self._config else None,
            "connection_status": "connected" if self.test_connection() else "disconnected"
        }


# Global instance
mem0_manager = Mem0ClientManager()


def get_mem0_client() -> Optional[MemoryClient]:
    """Get the global Mem0 client instance."""
    return mem0_manager.client


def get_mem0_config() -> Mem0Config:
    """Get the global Mem0 configuration."""
    return mem0_manager.config


def is_memory_enabled() -> bool:
    """Check if memory features are enabled."""
    return mem0_manager.is_enabled