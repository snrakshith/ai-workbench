"""Complete RAG pipeline combining retrieval and generation."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

from .llm_integration import GeminiLLMGenerator, create_llm_generator
from .prompt_templates import analyze_context_type, get_context_metadata
from src.retrieval.semantic_search import SemanticSearcher, create_semantic_searcher
from src.memory.conversation_manager import get_conversation_manager, ConversationContext
from src.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for question answering."""
    
    def __init__(self, collection_name: str = None, llm_model: str = None):
        self.collection_name = collection_name or settings.collection_name
        self.llm_model = llm_model or settings.llm_model
        
        # Initialize components with adaptive prompting enabled
        self.searcher = create_semantic_searcher(collection_name=self.collection_name)
        self.llm_generator = create_llm_generator(model_name=self.llm_model, use_adaptive_prompts=True)
        
        # Initialize memory components
        self.conversation_manager = get_conversation_manager()
        self.memory_enabled = settings.memory_enabled
        
        # Pipeline configuration
        self.config = {
            "retrieval": {
                "default_top_k": settings.top_k_retrieval,
                "min_similarity_threshold": 0.6,
                "max_context_length": 4000,
                "enable_reranking": True
            },
            "generation": {
                "use_citations": True,
                "include_confidence": True,
                "generate_followup": True,
                "max_response_tokens": settings.max_tokens
            },
            "conversation": {
                "enable_memory": True,
                "context_turns": 3,
                "memory_decay": False
            }
        }
        
        # Pipeline statistics
        self.pipeline_stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "avg_response_time": 0,
            "avg_retrieval_time": 0,
            "avg_generation_time": 0,
            "avg_context_docs": 0,
            "quality_scores": []
        }
        
        logger.info(f"✅ RAG Pipeline initialized")
        logger.info(f"📊 Collection: {self.collection_name}")
        logger.info(f"🤖 LLM Model: {self.llm_model}")
    
    def query(self, question: str, conversation_id: str = None, 
              user_id: str = None,
              retrieval_params: Dict[str, Any] = None,
              generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete RAG query processing.
        
        Args:
            question: User's question
            conversation_id: Optional conversation identifier for context
            user_id: User identifier for memory management
            retrieval_params: Custom retrieval parameters
            generation_params: Custom generation parameters
        """
        start_time = time.time()
        self.pipeline_stats["total_queries"] += 1
        
        try:
            # Step 1: Get or create conversation context for memory integration
            conversation_context = None
            memory_context = {}
            if self.memory_enabled and user_id:
                conversation_context = self.conversation_manager.get_or_create_conversation(
                    user_id=user_id,
                    conversation_id=conversation_id
                )
                # Retrieve relevant memory context
                memory_context = conversation_context.get_relevant_context(
                    current_query=question,
                    include_session=True,
                    include_user_profile=True,
                    max_context_turns=self.config["conversation"]["context_turns"]
                )
                logger.info(f"🧠 Retrieved memory context: {len(memory_context.get('relevant_memories', []))} memories, {len(memory_context.get('recent_turns', []))} recent turns")
            
            # Step 2: Retrieve relevant documents
            retrieval_start = time.time()
            retrieved_docs = self._retrieve_documents(question, retrieval_params)
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Check if we should fallback to direct LLM generation
            should_fallback = self._should_fallback_to_direct_llm(retrieved_docs, question)
            if should_fallback:
                return self._handle_no_context_found(question, retrieval_time, conversation_context, memory_context)
            
            # Step 3: Generate answer using LLM with memory context
            generation_start = time.time()
            
            # Prepare memory-enhanced context for LLM
            memory_enhanced_context = self._prepare_memory_enhanced_context(
                retrieved_docs=retrieved_docs,
                memory_context=memory_context,
                question=question
            )
            
            generation_result = self.llm_generator.generate_answer(
                query=question,
                retrieved_contexts=memory_enhanced_context["documents"],
                conversation_history=memory_enhanced_context.get("conversation_summary"),
                memory_context=memory_enhanced_context.get("memory_summary"),
                use_citations=generation_params.get("use_citations", self.config["generation"]["use_citations"]) if generation_params else self.config["generation"]["use_citations"],
                max_context_length=retrieval_params.get("max_context_length", self.config["retrieval"]["max_context_length"]) if retrieval_params else self.config["retrieval"]["max_context_length"]
            )
            
            generation_time = (time.time() - generation_start) * 1000
            
            if not generation_result["success"]:
                return self._handle_generation_failure(question, retrieved_docs, generation_result, retrieval_time, generation_time)
            
            # Step 4: Post-process and enhance response
            enhanced_response = self._enhance_response(
                question=question,
                generation_result=generation_result,
                retrieved_docs=retrieved_docs,
                conversation_id=conversation_id,
                generation_params=generation_params
            )
            
            # Step 5: Store conversation turn in memory
            if self.memory_enabled and conversation_context:
                try:
                    # Prepare comprehensive metadata including document context
                    document_context = []
                    for i, doc in enumerate(retrieved_docs[:3]):  # Top 3 documents
                        doc_info = {
                            "rank": i + 1,
                            "score": doc.get("score", 0),
                            "file_name": doc.get("metadata", {}).get("file_name", "Unknown"),
                            "doc_type": doc.get("metadata", {}).get("doc_type", "unknown"),
                            "content_preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                        }
                        document_context.append(doc_info)
                    
                    # Optimize metadata to stay under 2000 character limit for Mem0
                    comprehensive_metadata = {
                        "retrieved_documents": len(retrieved_docs),
                        "confidence_score": enhanced_response["confidence_score"],
                        "response_time_ms": (time.time() - start_time) * 1000,
                        "model_used": self.llm_model,
                        "citations_count": len(enhanced_response["citations"]),
                        "retrieval_strategy": "memory_enhanced_semantic_search" if self.memory_enabled and user_id else "semantic_search",
                        # Store detailed info separately to avoid metadata size limit
                        "document_context": document_context,
                        "citations": enhanced_response["citations"][:3],  # Limit to top 3 citations
                        "follow_up_questions": enhanced_response.get("follow_up_questions", [])[:2]  # Limit to 2 questions
                    }
                    
                    conversation_turn = conversation_context.add_turn(
                        user_input=question,
                        assistant_response=enhanced_response["answer"],
                        metadata=comprehensive_metadata
                    )
                    
                    # Extract user preferences from the conversation
                    conversation_context.extract_user_preferences(
                        user_input=question,
                        assistant_response=enhanced_response["answer"]
                    )
                    
                    logger.debug(f"💾 Stored conversation turn: {conversation_turn.turn_id}")
                except Exception as e:
                    logger.warning(f"Failed to store conversation turn: {e}")
            
            # Step 6: Update statistics and conversation history
            total_time = (time.time() - start_time) * 1000
            self._update_pipeline_stats(retrieval_time, generation_time, total_time, len(retrieved_docs))
            
            # Step 7: Build final response
            final_response = {
                "success": True,
                "question": question,
                "answer": enhanced_response["answer"],
                "citations": enhanced_response["citations"],
                "confidence_score": enhanced_response["confidence_score"],
                "follow_up_questions": enhanced_response.get("follow_up_questions", []),
                "retrieved_documents": len(retrieved_docs),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "performance": {
                    "total_time_ms": total_time,
                    "retrieval_time_ms": retrieval_time,
                    "generation_time_ms": generation_time,
                    "context_documents": len(retrieved_docs)
                },
                "memory_info": {
                    "memory_enabled": self.memory_enabled,
                    "memories_used": len(memory_context.get("relevant_memories", [])) if memory_context else 0,
                    "recent_turns_used": len(memory_context.get("recent_turns", [])) if memory_context else 0,
                    "user_profile_items": len(memory_context.get("user_profile", [])) if memory_context else 0,
                    "context_summary": memory_context.get("context_summary", "") if memory_context else ""
                },
                "metadata": {
                    "model_used": self.llm_model,
                    "retrieval_strategy": "memory_enhanced_semantic_search" if self.memory_enabled and user_id else "semantic_search",
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_config": self._get_current_config()
                }
            }
            
            self.pipeline_stats["successful_responses"] += 1
            return final_response
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {str(e)}")
            self.pipeline_stats["failed_responses"] += 1
            
            total_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "question": question,
                "error": str(e),
                "answer": "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.",
                "citations": [],
                "confidence_score": 0.0,
                "retrieved_documents": 0,
                "conversation_id": conversation_id,
                "sources": [],
                "followup_questions": [],
                "performance": {
                    "total_time_ms": total_time,
                    "retrieval_time_ms": 0,
                    "generation_time_ms": 0,
                    "context_documents": 0
                },
                "metadata": {
                    "model_used": self.llm_model,
                    "retrieval_strategy": "semantic_search", 
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_config": self._get_current_config()
                }
            }
    
    def _retrieve_documents(self, question: str, retrieval_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the question."""
        params = retrieval_params or {}
        
        # Use configured or provided parameters
        top_k = params.get("top_k", self.config["retrieval"]["default_top_k"])
        score_threshold = params.get("score_threshold", self.config["retrieval"]["min_similarity_threshold"])
        document_types = params.get("document_types", None)
        module_patterns = params.get("module_patterns", None)
        
        try:
            if module_patterns:
                # Module-specific search
                retrieved_docs = self.searcher.search_by_module(
                    query=question,
                    module_patterns=module_patterns,
                    top_k=top_k
                )
            elif document_types:
                # Document type filtered search
                type_results = self.searcher.search_by_document_type(
                    query=question,
                    doc_types=document_types,
                    top_k=top_k
                )
                # Flatten results and sort by score
                retrieved_docs = []
                for doc_type, docs in type_results.items():
                    retrieved_docs.extend(docs)
                retrieved_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
                retrieved_docs = retrieved_docs[:top_k]
            else:
                # Standard semantic search
                retrieved_docs = self.searcher.search(
                    query=question,
                    top_k=top_k,
                    score_threshold=score_threshold
                )
            
            # Apply additional filtering if needed
            if score_threshold and score_threshold > self.config["retrieval"]["min_similarity_threshold"]:
                retrieved_docs = [doc for doc in retrieved_docs if doc.get("score", 0) >= score_threshold]
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {question[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return []
    
    def _enhance_response(self, question: str, generation_result: Dict[str, Any],
                         retrieved_docs: List[Dict[str, Any]], conversation_id: str = None,
                         generation_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance the generated response with additional features and context analysis."""
        enhanced = {
            "answer": generation_result["answer"],
            "citations": generation_result["citations"],
            "confidence_score": generation_result["confidence_score"]
        }
        
        # Add context analysis metadata if retrieved docs are available
        if retrieved_docs:
            try:
                # Combine all retrieved content for analysis
                combined_context = "\n\n".join([
                    doc.get("content", "") for doc in retrieved_docs[:3]  # Top 3 docs
                ])
                
                # Analyze context type and metadata
                context_type = analyze_context_type(combined_context)
                context_metadata = get_context_metadata(combined_context)
                
                enhanced["context_analysis"] = {
                    "primary_content_type": context_type.value,
                    "content_metadata": context_metadata,
                    "adaptive_prompting_used": True
                }
                
                logger.debug(f"Context analysis: {context_type.value}, complexity: {context_metadata.get('estimated_complexity', 'unknown')}")
                
            except Exception as e:
                logger.warning(f"Context analysis failed: {str(e)}")
                enhanced["context_analysis"] = {
                    "primary_content_type": "unknown",
                    "content_metadata": {},
                    "adaptive_prompting_used": False,
                    "analysis_error": str(e)
                }
        
        # Generate follow-up questions if enabled
        if (generation_params and generation_params.get("generate_followup", True)) or \
           (not generation_params and self.config["generation"]["generate_followup"]):
            try:
                follow_ups = self.llm_generator.generate_follow_up_questions(
                    query=question,
                    answer=enhanced["answer"],
                    retrieved_contexts=retrieved_docs
                )
                enhanced["follow_up_questions"] = follow_ups
            except Exception as e:
                logger.warning(f"Failed to generate follow-up questions: {str(e)}")
                enhanced["follow_up_questions"] = []
        
        # Add response structure analysis
        enhanced["response_structure"] = self._analyze_response_structure(enhanced["answer"])
        
        return enhanced
    
    def _analyze_response_structure(self, answer: str) -> Dict[str, Any]:
        """Analyze the structure of the generated response."""
        try:
            structure = {
                "has_code_blocks": "```" in answer,
                "has_bullet_points": any(line.strip().startswith(("- ", "* ", "• ")) for line in answer.split("\n")),
                "has_numbered_lists": any(line.strip().match(r"^\d+\.") for line in answer.split("\n") if hasattr(line.strip(), 'match')),
                "has_headers": any(line.strip().startswith("#") for line in answer.split("\n")),
                "word_count": len(answer.split()),
                "line_count": len([line for line in answer.split("\n") if line.strip()]),
                "estimated_reading_time_minutes": max(1, len(answer.split()) // 200),
                "contains_technical_terms": any(term in answer.lower() for term in [
                    "langchain", "api", "function", "class", "method", "parameter", "import", "library"
                ])
            }
            
            # Check for structured format compliance
            structured_indicators = [
                structure["has_code_blocks"],
                structure["has_bullet_points"] or structure["has_numbered_lists"],
                structure["has_headers"]
            ]
            structure["follows_structured_format"] = sum(structured_indicators) >= 2
            
            return structure
            
        except Exception as e:
            logger.warning(f"Response structure analysis failed: {str(e)}")
            return {
                "analysis_error": str(e),
                "word_count": len(answer.split()) if answer else 0
            }
    
    def _prepare_memory_enhanced_context(self, retrieved_docs: List[Dict[str, Any]], 
                                       memory_context: Dict[str, Any], 
                                       question: str) -> Dict[str, Any]:
        """
        Prepare memory-enhanced context by combining retrieved documents with memory context.
        
        Args:
            retrieved_docs: Documents retrieved from vector search
            memory_context: Context from conversation memory
            question: Current user question
            
        Returns:
            Enhanced context dictionary with documents and memory summaries
        """
        # Weight documents based on memory context
        context_weight = settings.memory_context_weight
        enhanced_docs = []
        
        try:
            # If we have memory context, enhance document ranking
            if memory_context and memory_context.get("relevant_memories"):
                memory_keywords = []
                for memory in memory_context["relevant_memories"]:
                    memory_content = memory.get("memory", "").lower()
                    memory_keywords.extend(memory_content.split()[:10])  # Take first 10 words
                
                # Boost document scores if they align with memory context
                for doc in retrieved_docs:
                    original_score = doc.get("score", 0)
                    doc_content = doc.get("content", "").lower()
                    
                    # Calculate memory relevance boost
                    memory_boost = 0
                    for keyword in memory_keywords:
                        if keyword in doc_content:
                            memory_boost += 0.05  # Small boost per matching keyword
                    
                    # Apply weighted combination
                    enhanced_score = (original_score * (1 - context_weight)) + (memory_boost * context_weight)
                    doc["enhanced_score"] = enhanced_score
                    enhanced_docs.append(doc)
                
                # Re-sort by enhanced score
                enhanced_docs.sort(key=lambda x: x.get("enhanced_score", 0), reverse=True)
            else:
                enhanced_docs = retrieved_docs
            
            # Prepare memory summaries for LLM context
            memory_summary = ""
            conversation_summary = ""
            
            if memory_context:
                # Create formatted memory context for LLM
                memory_summary = memory_context.get("context_summary", "")
                
                # Format recent conversation turns
                recent_turns = memory_context.get("recent_turns", [])
                if recent_turns:
                    turn_summaries = []
                    for turn in recent_turns[-3:]:  # Last 3 turns
                        turn_summaries.append(
                            f"User: {turn['user_input'][:100]}...\n"
                            f"Assistant: {turn['assistant_response'][:100]}..."
                        )
                    conversation_summary = "\n---\n".join(turn_summaries)
            
            return {
                "documents": enhanced_docs,
                "memory_summary": memory_summary,
                "conversation_summary": conversation_summary,
                "memory_boost_applied": bool(memory_context and memory_context.get("relevant_memories"))
            }
            
        except Exception as e:
            logger.warning(f"Failed to prepare memory-enhanced context: {e}")
            # Fallback to original documents
            return {
                "documents": retrieved_docs,
                "memory_summary": "",
                "conversation_summary": "",
                "memory_boost_applied": False
            }
    
    def _should_fallback_to_direct_llm(self, retrieved_docs: List[Dict[str, Any]], question: str) -> bool:
        """
        Determine if we should fallback to direct LLM generation.
        
        Args:
            retrieved_docs: Retrieved documents from search
            question: User's question
            
        Returns:
            True if should fallback to direct LLM, False otherwise
        """
        # No documents retrieved at all
        if not retrieved_docs:
            return True
        
        # Check if all documents have very low similarity scores
        high_similarity_threshold = 0.85  # Increased for stricter filtering
        medium_similarity_threshold = 0.7  # Increased for more fallback
        
        # Get the highest score
        max_score = max(doc.get("score", 0) for doc in retrieved_docs) if retrieved_docs else 0
        
        # Log similarity scores for debugging
        scores = [doc.get("score", 0) for doc in retrieved_docs]
        logger.info(f"Query similarity check: '{question[:30]}...', max score: {max_score:.3f}")
        
        # If the best match is very poor, fallback to direct LLM
        if max_score < medium_similarity_threshold:
            logger.info(f"Low similarity scores (max: {max_score:.3f}), falling back to direct LLM")
            return True
            
        # Check for very specific technical questions that might not be in LangChain docs
        non_langchain_keywords = [
            'fibonacci', 'factorial', 'prime number', 'sorting algorithm', 
            'binary search', 'data structure', 'algorithm', 'math', 'mathematics',
            'sql', 'database design', 'web scraping', 'gui', 'tkinter', 'flask routing',
            'django models', 'numpy array', 'pandas dataframe', 'matplotlib plot',
            'tensorflow model', 'pytorch tensor', 'opencv image', 'selenium automation'
        ]
        
        question_lower = question.lower()
        detected_keyword = None
        for keyword in non_langchain_keywords:
            if keyword in question_lower:
                detected_keyword = keyword
                break
                
        if detected_keyword:
            # For clearly non-LangChain queries, be more aggressive with fallback
            lower_threshold_for_non_langchain = 0.8  # Even stricter for non-LangChain
            if max_score < lower_threshold_for_non_langchain:
                logger.info(f"Non-LangChain technical query detected ('{detected_keyword}') with similarity ({max_score:.3f}), falling back to direct LLM")
                return True
        
        # Otherwise, use RAG with available context
        return False

    def _handle_no_context_found(self, question: str, retrieval_time: float, 
                                  conversation_context: ConversationContext = None, 
                                  memory_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle case when no relevant context is found - fallback to direct LLM generation."""
        try:
            logger.info(f"No relevant context found for query: {question[:50]}... Falling back to direct LLM generation")
            
            # Generate direct LLM response without RAG context
            generation_start = time.time()
            fallback_result = self.llm_generator.generate_direct_answer(
                query=question,
                include_disclaimer=True  # Indicate this is not from documentation
            )
            generation_time = (time.time() - generation_start) * 1000
            
            if fallback_result["success"]:
                # Update statistics
                total_time = retrieval_time + generation_time
                self.pipeline_stats["successful_responses"] += 1
                self.pipeline_stats["avg_response_time"] = (
                    (self.pipeline_stats["avg_response_time"] * (self.pipeline_stats["total_queries"] - 1) + total_time) /
                    self.pipeline_stats["total_queries"]
                )
                
                return {
                    "success": True,
                    "question": question,
                    "answer": fallback_result["answer"],
                    "citations": [],  # No citations for direct LLM response
                    "confidence_score": 0.7,  # Medium confidence for non-RAG response
                    "follow_up_questions": fallback_result.get("follow_up_questions", []),
                    "retrieved_documents": 0,
                    "conversation_id": fallback_result.get("conversation_id"),
                    "performance": {
                        "total_time_ms": total_time,
                        "retrieval_time_ms": retrieval_time,
                        "generation_time_ms": generation_time,
                        "context_documents": 0
                    },
                    "metadata": {
                        "model_used": self.llm_model,
                        "retrieval_strategy": "direct_llm_fallback",
                        "timestamp": datetime.now().isoformat(),
                        "no_context_found": True,
                        "pipeline_config": {
                            "fallback_enabled": True,
                            "retrieval_failed": True
                        }
                    }
                }
            else:
                # If direct LLM also fails, return error response
                logger.warning(f"Direct LLM generation also failed: {fallback_result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "question": question,
                    "answer": "I apologize, but I'm unable to provide an answer to your question at the moment. Please try rephrasing your question or try again later.",
                    "citations": [],
                    "confidence_score": 0.0,
                    "follow_up_questions": [
                        "Could you try rephrasing your question?",
                        "Is there a specific aspect you'd like to focus on?",
                        "Would you like to try a different question?"
                    ],
                    "retrieved_documents": 0,
                    "performance": {
                        "total_time_ms": retrieval_time + generation_time,
                        "retrieval_time_ms": retrieval_time,
                        "generation_time_ms": generation_time,
                        "context_documents": 0
                    },
                    "metadata": {
                        "no_context_found": True,
                        "direct_generation_failed": True,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Fallback LLM generation failed: {str(e)}")
            self.pipeline_stats["failed_responses"] += 1
            
            return {
                "success": False,
                "question": question,
                "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "citations": [],
                "confidence_score": 0.0,
                "follow_up_questions": [],
                "retrieved_documents": 0,
                "performance": {
                    "total_time_ms": retrieval_time,
                    "retrieval_time_ms": retrieval_time,
                    "generation_time_ms": 0,
                    "context_documents": 0
                },
                "metadata": {
                    "error": str(e),
                    "no_context_found": True,
                    "fallback_failed": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _handle_generation_failure(self, question: str, retrieved_docs: List[Dict[str, Any]],
                                  generation_result: Dict[str, Any], retrieval_time: float,
                                  generation_time: float) -> Dict[str, Any]:
        """Handle LLM generation failure."""
        return {
            "success": False,
            "question": question,
            "error": generation_result.get("error", "Unknown generation error"),
            "answer": "I found relevant information but encountered an issue generating the response. Please try again.",
            "citations": [],
            "confidence_score": 0.0,
            "retrieved_documents": len(retrieved_docs),
            "performance": {
                "total_time_ms": retrieval_time + generation_time,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": generation_time,
                "context_documents": len(retrieved_docs)
            }
        }
    
    def _update_pipeline_stats(self, retrieval_time: float, generation_time: float,
                              total_time: float, context_docs: int):
        """Update pipeline statistics."""
        # Update averages
        successful = self.pipeline_stats["successful_responses"]
        
        self.pipeline_stats["avg_response_time"] = (
            (self.pipeline_stats["avg_response_time"] * successful + total_time) / 
            (successful + 1)
        )
        
        self.pipeline_stats["avg_retrieval_time"] = (
            (self.pipeline_stats["avg_retrieval_time"] * successful + retrieval_time) / 
            (successful + 1)
        )
        
        self.pipeline_stats["avg_generation_time"] = (
            (self.pipeline_stats["avg_generation_time"] * successful + generation_time) / 
            (successful + 1)
        )
        
        self.pipeline_stats["avg_context_docs"] = (
            (self.pipeline_stats["avg_context_docs"] * successful + context_docs) / 
            (successful + 1)
        )
    
    def _get_current_config(self) -> Dict[str, Any]:
        """Get current pipeline configuration."""
        return {
            "retrieval_top_k": self.config["retrieval"]["default_top_k"],
            "similarity_threshold": self.config["retrieval"]["min_similarity_threshold"],
            "use_citations": self.config["generation"]["use_citations"],
            "enable_memory": self.config["conversation"]["enable_memory"],
            "max_context_length": self.config["retrieval"]["max_context_length"]
        }
    
    def batch_query(self, questions: List[str], conversation_id: str = None,
                   batch_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process multiple questions in batch."""
        start_time = time.time()
        
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Processing batch query {i+1}/{len(questions)}: {question[:50]}...")
            
            result = self.query(
                question=question,
                conversation_id=f"{conversation_id}_batch_{i}" if conversation_id else None,
                retrieval_params=batch_params.get("retrieval_params") if batch_params else None,
                generation_params=batch_params.get("generation_params") if batch_params else None
            )
            
            results.append({
                "batch_index": i,
                "question": question,
                **result
            })
        
        batch_time = (time.time() - start_time) * 1000
        
        logger.info(f"Completed batch processing: {len(questions)} questions in {batch_time:.1f}ms")
        
        return results
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update pipeline configuration."""
        for section, updates in config_updates.items():
            if section in self.config:
                self.config[section].update(updates)
        logger.info(f"Pipeline configuration updated: {config_updates}")
    
    def get_pipeline_analytics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline analytics."""
        # Get component statistics
        searcher_analytics = self.searcher.get_search_analytics()
        llm_stats = self.llm_generator.get_generation_stats()
        
        return {
            "pipeline_stats": self.pipeline_stats,
            "retrieval_analytics": searcher_analytics,
            "generation_stats": llm_stats,
            "configuration": self.config,
            "success_rate": (
                self.pipeline_stats["successful_responses"] / 
                max(1, self.pipeline_stats["total_queries"])
            ) * 100,
            "avg_quality_score": (
                sum(self.pipeline_stats["quality_scores"]) / 
                max(1, len(self.pipeline_stats["quality_scores"]))
            ) if self.pipeline_stats["quality_scores"] else 0
        }
    
    def clear_conversation_memory(self, conversation_id: str = None):
        """Clear conversation memory."""
        if conversation_id:
            # In a production system, you'd clear specific conversation
            # For now, clear all since we don't have conversation-specific storage
            logger.info(f"Clearing memory for conversation: {conversation_id}")
        
        self.llm_generator.clear_conversation_history()
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all pipeline components."""
        health_status = {
            "overall_health": "healthy",
            "components": {},
            "last_check": datetime.now().isoformat()
        }
        
        # Check searcher health
        try:
            search_analytics = self.searcher.get_search_analytics()
            health_status["components"]["retrieval"] = {
                "status": "healthy",
                "documents": search_analytics.get("collection_stats", {}).get("points_count", 0),
                "last_search": "available"
            }
        except Exception as e:
            health_status["components"]["retrieval"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_health"] = "degraded"
        
        # Check LLM health
        try:
            llm_stats = self.llm_generator.get_generation_stats()
            health_status["components"]["generation"] = {
                "status": "healthy",
                "model": self.llm_model,
                "success_rate": llm_stats.get("success_rate", 0)
            }
        except Exception as e:
            health_status["components"]["generation"] = {
                "status": "unhealthy", 
                "error": str(e)
            }
            health_status["overall_health"] = "degraded"
        
        return health_status


def create_rag_pipeline(collection_name: str = None, llm_model: str = None) -> RAGPipeline:
    """Create and initialize RAG pipeline."""
    return RAGPipeline(collection_name=collection_name, llm_model=llm_model)