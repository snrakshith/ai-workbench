"""Conversation memory manager for handling multi-turn dialogues and context management."""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .memory_service import MemoryService, MemoryType, get_memory_service
from .mem0_config import get_mem0_config

logger = logging.getLogger(__name__)


class ConversationTurn:
    """Represents a single conversation turn (user input + assistant response)."""
    
    def __init__(
        self,
        user_input: str,
        assistant_response: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a conversation turn."""
        self.user_input = user_input
        self.assistant_response = assistant_response
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.turn_id = self._generate_turn_id()
    
    def _generate_turn_id(self) -> str:
        """Generate a unique ID for this turn."""
        content = f"{self.user_input}{self.assistant_response}{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to message format for Mem0."""
        return [
            {"role": "user", "content": self.user_input},
            {"role": "assistant", "content": self.assistant_response}
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "assistant_response": self.assistant_response,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class ConversationContext:
    """Manages conversation context and memory integration."""
    
    def __init__(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        agent_id: str = "langchain_rag",
        memory_service: Optional[MemoryService] = None
    ):
        """Initialize conversation context."""
        self.user_id = user_id
        self.conversation_id = conversation_id or self._generate_conversation_id()
        self.agent_id = agent_id
        self.memory_service = memory_service or get_memory_service()
        self.config = get_mem0_config()
        
        # Track conversation state
        self.turns: List[ConversationTurn] = []
        self.session_metadata: Dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id
        }
        
        logger.info(f"🎯 Conversation context initialized for user {user_id}, conversation {self.conversation_id}")
    
    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_hash = hashlib.md5(self.user_id.encode()).hexdigest()[:8]
        return f"conv_{timestamp}_{user_hash}"
    
    def add_turn(
        self,
        user_input: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn and store it in memory.
        
        Args:
            user_input: User's input/question
            assistant_response: Assistant's response
            metadata: Additional metadata for this turn
            
        Returns:
            The created conversation turn
        """
        # Create the turn
        turn = ConversationTurn(
            user_input=user_input,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )
        
        # Add to local history
        self.turns.append(turn)
        
        # Store in memory if enabled
        if self.memory_service.enabled:
            self._store_turn_in_memory(turn)
        
        # Update session metadata
        self.session_metadata.update({
            "last_turn_at": turn.timestamp.isoformat(),
            "total_turns": len(self.turns)
        })
        
        logger.debug(f"Added conversation turn {turn.turn_id} for conversation {self.conversation_id}")
        return turn
    
    def _store_turn_in_memory(self, turn: ConversationTurn) -> None:
        """Store a comprehensive conversation turn in Mem0 with all context."""
        try:
            # Create optimized metadata for the turn (keep under 2000 chars for Mem0)
            basic_metadata = {
                "conversation_id": self.conversation_id,
                "turn_id": turn.turn_id,
                "turn_number": len(self.turns),
                "timestamp": turn.timestamp.isoformat(),
                "retrieved_documents": turn.metadata.get("retrieved_documents", 0),
                "confidence_score": turn.metadata.get("confidence_score", 0),
                "response_time_ms": turn.metadata.get("response_time_ms", 0),
                "model_used": turn.metadata.get("model_used", "unknown"),
                "citations_count": turn.metadata.get("citations_count", 0)
            }
            
            # 1. Store the complete conversation turn with full context
            full_conversation_content = f"""
CONVERSATION TURN #{len(self.turns)}
Timestamp: {turn.timestamp.isoformat()}
Turn ID: {turn.turn_id}

USER QUESTION:
{turn.user_input}

ASSISTANT RESPONSE:
{turn.assistant_response}

METADATA:
- Retrieved Documents: {turn.metadata.get('retrieved_documents', 0)}
- Confidence Score: {turn.metadata.get('confidence_score', 'N/A')}
- Response Time: {turn.metadata.get('response_time_ms', 0)}ms
- Model Used: {turn.metadata.get('model_used', 'Unknown')}
- Citations Count: {turn.metadata.get('citations_count', 0)}

RETRIEVED CONTEXT SUMMARY:
{self._format_document_context_for_content(turn.metadata.get('document_context', []))}

CITATIONS:
{self._format_citations_for_content(turn.metadata.get('citations', []))}
"""
            
            self.memory_service.add_memory(
                content=full_conversation_content,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.conversation_id,
                memory_type=MemoryType.CONVERSATION,
                category="full_dialogue",
                importance="high",
                tags=["conversation", "dialogue", "turn", "complete_context"],
                metadata=basic_metadata
            )
            
            # 2. Store just the user question separately for better searchability
            self.memory_service.add_memory(
                content=f"User asked: {turn.user_input}",
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.conversation_id,
                memory_type=MemoryType.CONVERSATION,
                category="user_query",
                importance="medium",
                tags=["question", "user_input", "query"],
                metadata={**basic_metadata, "query_type": "user_question"}
            )
            
            # 3. Store the assistant response separately
            self.memory_service.add_memory(
                content=f"Assistant responded: {turn.assistant_response[:500]}..." if len(turn.assistant_response) > 500 else f"Assistant responded: {turn.assistant_response}",
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.conversation_id,
                memory_type=MemoryType.CONVERSATION,
                category="assistant_response",
                importance="medium",
                tags=["response", "assistant_output", "answer"],
                metadata={**basic_metadata, "response_type": "assistant_answer"}
            )
            
            # 4. Store retrieved document context separately if available
            if turn.metadata.get("document_context"):
                doc_context_content = "RETRIEVED DOCUMENT CONTEXT:\n"
                for doc in turn.metadata["document_context"]:
                    doc_context_content += f"\nRank {doc['rank']} (Score: {doc['score']:.3f}):\n"
                    doc_context_content += f"File: {doc['file_name']} ({doc['doc_type']})\n"
                    doc_context_content += f"Content: {doc['content_preview']}\n"
                
                self.memory_service.add_memory(
                    content=doc_context_content,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    run_id=self.conversation_id,
                    memory_type=MemoryType.KNOWLEDGE,
                    category="document_context",
                    importance="medium",
                    tags=["context", "documents", "retrieval", "sources"],
                    metadata={**basic_metadata, "context_type": "retrieved_documents"}
                )
            
            # 5. Store citations separately if available
            if turn.metadata.get("citations"):
                citations_content = "CITATIONS AND SOURCES:\n"
                for citation in turn.metadata["citations"]:
                    citations_content += f"- {citation}\n"
                
                self.memory_service.add_memory(
                    content=citations_content,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    run_id=self.conversation_id,
                    memory_type=MemoryType.KNOWLEDGE,
                    category="citations",
                    importance="low",
                    tags=["citations", "sources", "references"],
                    metadata={**basic_metadata, "content_type": "citations"}
                )
            
            logger.info(f"💾 Stored comprehensive conversation turn {turn.turn_id} with full context, documents, and citations")
            
        except Exception as e:
            logger.error(f"Failed to store turn in memory: {e}")
            # Try a simpler storage as fallback
            try:
                simple_content = f"Q: {turn.user_input}\nA: {turn.assistant_response[:200]}..."
                self.memory_service.add_memory(
                    content=simple_content,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    run_id=self.conversation_id,
                    memory_type=MemoryType.CONVERSATION,
                    category="simple_dialogue",
                    importance="medium",
                    tags=["conversation", "fallback"],
                    metadata={"conversation_id": self.conversation_id, "turn_id": turn.turn_id}
                )
                logger.info(f"💾 Stored simplified conversation turn {turn.turn_id}")
            except Exception as fallback_error:
                logger.error(f"Even fallback storage failed: {fallback_error}")
    
    def _format_document_context_for_content(self, document_context: List[Dict]) -> str:
        """Format document context for storage in content (not metadata)."""
        if not document_context:
            return "No document context available"
        
        formatted = []
        for doc in document_context[:3]:  # Top 3 documents
            formatted.append(f"• {doc.get('file_name', 'Unknown')} (Score: {doc.get('score', 0):.3f})")
            formatted.append(f"  Preview: {doc.get('content_preview', '')[:100]}...")
        
        return "\n".join(formatted)
    
    def _format_citations_for_content(self, citations: List) -> str:
        """Format citations for storage in content (not metadata)."""
        if not citations:
            return "No citations available"
        
        formatted = []
        for i, citation in enumerate(citations[:3], 1):  # Top 3 citations
            if isinstance(citation, dict):
                formatted.append(f"{i}. {citation.get('file_name', 'Unknown source')}")
            else:
                formatted.append(f"{i}. {str(citation)[:100]}...")
        
        return "\n".join(formatted)
    
    def get_relevant_context(
        self,
        current_query: str,
        include_session: bool = True,
        include_user_profile: bool = True,
        max_context_turns: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context for the current query.
        
        Args:
            current_query: The current user query
            include_session: Whether to include session-specific memories
            include_user_profile: Whether to include user profile memories
            max_context_turns: Maximum number of recent turns to include
            
        Returns:
            Dictionary containing relevant context information
        """
        context = {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "recent_turns": [],
            "relevant_memories": [],
            "user_profile": [],
            "context_summary": ""
        }
        
        if not self.memory_service.enabled:
            return context
        
        try:
            # Get recent conversation turns
            max_turns = max_context_turns or self.config.window_size
            context["recent_turns"] = self._get_recent_turns(max_turns)
            
            # Search for relevant memories from current session
            if include_session:
                session_memories = self.memory_service.search_memory(
                    query=current_query,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    run_id=self.conversation_id,
                    limit=3
                )
                context["relevant_memories"].extend(session_memories)
            
            # Search for relevant user profile memories
            if include_user_profile:
                profile_memories = self.memory_service.search_memory(
                    query=current_query,
                    user_id=self.user_id,
                    memory_type=MemoryType.USER_PROFILE,
                    limit=2
                )
                context["user_profile"].extend(profile_memories)
                
                # Also get general conversation memories
                general_memories = self.memory_service.search_memory(
                    query=current_query,
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    limit=3
                )
                # Filter out current session memories to avoid duplicates
                filtered_general = [
                    mem for mem in general_memories
                    if mem.get("metadata", {}).get("run_id") != self.conversation_id
                ]
                context["relevant_memories"].extend(filtered_general[:2])
            
            # Create context summary
            context["context_summary"] = self._create_context_summary(context)
            
            logger.debug(f"Retrieved context with {len(context['relevant_memories'])} memories and {len(context['recent_turns'])} recent turns")
            
        except Exception as e:
            logger.error(f"Failed to get relevant context: {e}")
        
        return context
    
    def _get_recent_turns(self, max_turns: int) -> List[Dict[str, Any]]:
        """Get recent conversation turns."""
        recent_turns = self.turns[-max_turns:] if max_turns > 0 else self.turns
        return [turn.to_dict() for turn in recent_turns]
    
    def _create_context_summary(self, context: Dict[str, Any]) -> str:
        """Create a text summary of the conversation context."""
        summary_parts = []
        
        # Add user profile context
        if context["user_profile"]:
            profile_items = [mem.get("memory", "") for mem in context["user_profile"]]
            summary_parts.append(f"User Profile: {'; '.join(profile_items[:2])}")
        
        # Add relevant memories
        if context["relevant_memories"]:
            memory_items = [mem.get("memory", "") for mem in context["relevant_memories"]]
            summary_parts.append(f"Relevant Context: {'; '.join(memory_items[:3])}")
        
        # Add recent conversation
        if context["recent_turns"]:
            recent_count = len(context["recent_turns"])
            summary_parts.append(f"Recent conversation history: {recent_count} turns")
        
        return "; ".join(summary_parts)
    
    def extract_user_preferences(
        self,
        user_input: str,
        assistant_response: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and store user preferences from the conversation.
        
        Args:
            user_input: User's input
            assistant_response: Assistant's response
            
        Returns:
            Extracted preferences if any
        """
        if not self.memory_service.enabled:
            return None
        
        try:
            # Simple keyword-based preference extraction
            # In a production system, you might use NLP models for this
            preference_keywords = [
                "i like", "i prefer", "i love", "i hate", "i dislike",
                "i want", "i need", "my favorite", "i always", "i never"
            ]
            
            user_lower = user_input.lower()
            extracted_prefs = []
            
            for keyword in preference_keywords:
                if keyword in user_lower:
                    # Extract the preference
                    start_idx = user_lower.find(keyword)
                    preference_text = user_input[start_idx:].split('.')[0]
                    extracted_prefs.append(preference_text.strip())
            
            # Store preferences as user profile memories
            if extracted_prefs:
                for pref in extracted_prefs:
                    self.memory_service.add_memory(
                        content=pref,
                        user_id=self.user_id,
                        agent_id=self.agent_id,
                        memory_type=MemoryType.USER_PROFILE,
                        category="preference",
                        importance="high",
                        tags=["preference", "profile"],
                        metadata={
                            "conversation_id": self.conversation_id,
                            "extracted_from": "conversation",
                            "extraction_method": "keyword_based"
                        }
                    )
                
                logger.info(f"Extracted {len(extracted_prefs)} preferences for user {self.user_id}")
                return {"preferences": extracted_prefs}
        
        except Exception as e:
            logger.error(f"Failed to extract user preferences: {e}")
        
        return None
    
    def format_context_for_llm(self, context: Dict[str, Any]) -> str:
        """
        Format context for inclusion in LLM prompt.
        
        Args:
            context: Context dictionary from get_relevant_context
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        # Add user profile information
        if context.get("user_profile"):
            profile_text = "\\n".join([
                f"- {mem.get('memory', '')}"
                for mem in context["user_profile"][:2]
            ])
            context_parts.append(f"User Profile:\\n{profile_text}")
        
        # Add relevant memories
        if context.get("relevant_memories"):
            memories_text = "\\n".join([
                f"- {mem.get('memory', '')}"
                for mem in context["relevant_memories"][:3]
            ])
            context_parts.append(f"Relevant Past Context:\\n{memories_text}")
        
        # Add recent conversation turns
        if context.get("recent_turns"):
            recent_turns = context["recent_turns"][-3:]  # Last 3 turns
            turns_text = "\\n".join([
                f"User: {turn['user_input'][:100]}...\\nAssistant: {turn['assistant_response'][:100]}..."
                if len(turn['user_input']) > 100 else
                f"User: {turn['user_input']}\\nAssistant: {turn['assistant_response'][:100]}..."
                for turn in recent_turns
            ])
            context_parts.append(f"Recent Conversation:\\n{turns_text}")
        
        return "\\n\\n".join(context_parts) if context_parts else "No relevant context available."
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "total_turns": len(self.turns),
            "started_at": self.session_metadata.get("started_at"),
            "last_turn_at": self.session_metadata.get("last_turn_at"),
            "memory_enabled": self.memory_service.enabled
        }


class ConversationManager:
    """Manager for handling multiple conversations and their contexts."""
    
    def __init__(self, memory_service: Optional[MemoryService] = None):
        """Initialize the conversation manager."""
        self.memory_service = memory_service or get_memory_service()
        self.active_conversations: Dict[str, ConversationContext] = {}
        logger.info("🎯 Conversation Manager initialized")
    
    def get_or_create_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        agent_id: str = "langchain_rag"
    ) -> ConversationContext:
        """
        Get an existing conversation or create a new one.
        
        Args:
            user_id: Unique identifier for the user
            conversation_id: Optional conversation identifier
            agent_id: Agent identifier
            
        Returns:
            ConversationContext instance
        """
        # Create conversation key
        conv_key = f"{user_id}_{conversation_id}" if conversation_id else f"{user_id}_default"
        
        # Check if conversation exists
        if conv_key in self.active_conversations:
            return self.active_conversations[conv_key]
        
        # Create new conversation
        conversation = ConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
            memory_service=self.memory_service
        )
        
        self.active_conversations[conv_key] = conversation
        logger.info(f"Created new conversation: {conversation.conversation_id}")
        
        return conversation
    
    def end_conversation(self, user_id: str, conversation_id: Optional[str] = None) -> None:
        """End and clean up a conversation."""
        conv_key = f"{user_id}_{conversation_id}" if conversation_id else f"{user_id}_default"
        
        if conv_key in self.active_conversations:
            conversation = self.active_conversations[conv_key]
            
            # Store conversation summary if enabled
            if self.memory_service.enabled:
                summary = conversation.get_conversation_summary()
                self.memory_service.add_memory(
                    content=f"Conversation ended with {summary['total_turns']} turns",
                    user_id=user_id,
                    agent_id=conversation.agent_id,
                    run_id=conversation.conversation_id,
                    memory_type=MemoryType.SESSION_CONTEXT,
                    category="session_end",
                    metadata=summary
                )
            
            # Remove from active conversations
            del self.active_conversations[conv_key]
            logger.info(f"Ended conversation: {conversation.conversation_id}")
    
    def get_active_conversations_count(self) -> int:
        """Get the number of active conversations."""
        return len(self.active_conversations)
    
    def cleanup_inactive_conversations(self, max_age_hours: int = 24) -> int:
        """Clean up conversations that have been inactive."""
        # This would typically check timestamps and remove old conversations
        # For now, just return the count
        return 0


# Global conversation manager instance
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager instance."""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager