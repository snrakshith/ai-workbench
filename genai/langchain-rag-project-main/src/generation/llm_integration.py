"""LLM integration for RAG answer generation using Google Gemini."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime
import re

import google.genai as genai
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate

from src.config import settings
from src.generation.prompt_templates import (
    get_adaptive_prompt,
    get_direct_llm_prompt, 
    analyze_context_type,
    get_context_metadata
)

logger = logging.getLogger(__name__)


class GeminiLLMGenerator:
    """LLM integration using Google Gemini for RAG answer generation with adaptive prompting."""
    
    def __init__(self, model_name: str = None, api_key: str = None, use_adaptive_prompts: bool = True):
        self.model_name = model_name or settings.llm_model
        self.api_key = api_key or settings.google_api_key
        self.use_adaptive_prompts = use_adaptive_prompts
        
        if not self.api_key:
            raise ValueError("Google API key is required for LLM integration")
        
        # Initialize Gemini client with the correct pattern
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize prompt templates
        self.adaptive_prompt = get_adaptive_prompt()
        self.direct_prompt = get_direct_llm_prompt()
        
        # Generation parameters
        self.generation_config = {
            "max_output_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_p": 0.9,
            "top_k": 40
        }
        
        # Test connection
        self._test_connection()
        
        # Conversation history for context
        self.conversation_history = []
        
        # Response statistics
        self.response_stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "total_tokens_used": 0,
            "adaptive_prompts_used": 0,
            "direct_prompts_used": 0
        }
    
    def _test_connection(self):
        """Test LLM connection."""
        try:
            # Use the new google-genai client pattern
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": "Hello! Please respond with 'Connection successful'"}]}],
                config=self.generation_config
            )
            
            if response and hasattr(response, 'text') and response.text:
                logger.info(f"✅ Gemini LLM connection successful")
                logger.info(f"🤖 Model: {self.model_name}")
            else:
                logger.info(f"✅ Gemini LLM client initialized successfully")
                logger.info(f"🤖 Model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"❌ Gemini LLM connection failed: {str(e)}")
            # Don't raise the exception during initialization, just log it
            logger.warning(f"⚠️ Will attempt to use LLM anyway")
    
    def generate_answer(self, query: str, retrieved_contexts: List[Dict[str, Any]], 
                       conversation_history: List[Dict[str, str]] = None,
                       memory_context: str = None,
                       use_citations: bool = True, max_context_length: int = 4000) -> Dict[str, Any]:
        """
        Generate an answer using retrieved contexts.
        
        Args:
            query: User's question
            retrieved_contexts: List of retrieved document chunks
            conversation_history: Previous conversation turns
            memory_context: Memory context from Mem0 (user profile, preferences, etc.)
            use_citations: Whether to include source citations
            max_context_length: Maximum context length in characters
        """
        start_time = time.time()
        self.response_stats["total_requests"] += 1
        
        try:
            # Prepare context from retrieved documents
            context_text = self._prepare_context(retrieved_contexts, max_context_length)
            
            # Build prompt
            prompt = self._build_rag_prompt(
                query=query,
                context=context_text,
                retrieved_contexts=retrieved_contexts,
                conversation_history=conversation_history,
                memory_context=memory_context,
                use_citations=use_citations
            )
            
            # Generate response using new API pattern
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}]}],
                config=self.generation_config
            )
            
            if not response.text:
                raise Exception("Empty response from model")
            
            # Process response
            processed_response = self._process_response(
                response.text,
                retrieved_contexts,
                use_citations
            )
            
            # Update statistics
            response_time = (time.time() - start_time) * 1000
            self.response_stats["successful_responses"] += 1
            self.response_stats["avg_response_time"] = (
                (self.response_stats["avg_response_time"] * (self.response_stats["successful_responses"] - 1) + response_time) /
                self.response_stats["successful_responses"]
            )
            
            # Estimate token usage (rough approximation)
            estimated_tokens = len(prompt.split()) + len(response.text.split())
            self.response_stats["total_tokens_used"] += estimated_tokens
            
            return {
                "success": True,
                "answer": processed_response["answer"],
                "citations": processed_response["citations"],
                "confidence_score": processed_response["confidence_score"],
                "context_used": len(retrieved_contexts),
                "response_time_ms": response_time,
                "estimated_tokens": estimated_tokens,
                "model_used": self.model_name,
                "generation_metadata": {
                    "temperature": settings.temperature,
                    "max_tokens": settings.max_tokens,
                    "context_length": len(context_text),
                    "prompt_length": len(prompt)
                }
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            self.response_stats["failed_responses"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "answer": "I apologize, but I encountered an error while generating the answer. Please try again.",
                "citations": [],
                "confidence_score": 0.0,
                "context_used": 0,
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    def _prepare_context(self, retrieved_contexts: List[Dict[str, Any]], 
                        max_length: int) -> str:
        """Prepare context text from retrieved documents."""
        if not retrieved_contexts:
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for i, context in enumerate(retrieved_contexts):
            # Format context with metadata
            content = context.get("content", "")
            metadata = context.get("metadata", {})
            
            source_info = f"Source {i+1}"
            if metadata.get("file_name"):
                source_info += f" ({metadata['file_name']})"
            if metadata.get("doc_type"):
                source_info += f" [{metadata['doc_type']}]"
            
            formatted_context = f"\n--- {source_info} ---\n{content}\n"
            
            # Check length limit
            if current_length + len(formatted_context) > max_length:
                # Try to fit partial content
                remaining_space = max_length - current_length - len(f"\n--- {source_info} ---\n...\n")
                if remaining_space > 100:  # Only include if meaningful content fits
                    partial_content = content[:remaining_space] + "..."
                    formatted_context = f"\n--- {source_info} ---\n{partial_content}\n"
                    context_parts.append(formatted_context)
                break
            
            context_parts.append(formatted_context)
            current_length += len(formatted_context)
        
        return "".join(context_parts) if context_parts else "No relevant context available."
    
    def _build_rag_prompt(self, query: str, context: str, 
                         retrieved_contexts: List[Dict[str, Any]],
                         conversation_history: List[Dict[str, str]] = None,
                         memory_context: str = None,
                         use_citations: bool = True) -> str:
        """Build the RAG prompt for the LLM using adaptive prompting."""
        
        if self.use_adaptive_prompts and context.strip() != "No relevant context found.":
            # Use adaptive prompting system
            self.response_stats["adaptive_prompts_used"] += 1
            
            # Analyze context type and get metadata
            context_type = analyze_context_type(context)
            context_metadata = get_context_metadata(context)
            
            # Log context analysis for debugging
            logger.debug(f"Context type detected: {context_type.value}")
            logger.debug(f"Context metadata: {context_metadata}")
            
            # Use adaptive prompt template with memory context
            enhanced_context = context
            if memory_context:
                enhanced_context = f"MEMORY CONTEXT:\n{memory_context}\n\nDOCUMENTATION CONTEXT:\n{context}"
            
            prompt = self.adaptive_prompt.format(
                context=enhanced_context,
                question=query
            )
            
            return prompt
        else:
            # Fallback to original prompt system
            return self._build_legacy_rag_prompt(query, context, retrieved_contexts, conversation_history, memory_context, use_citations)
    
    def _build_legacy_rag_prompt(self, query: str, context: str, 
                                retrieved_contexts: List[Dict[str, Any]],
                                conversation_history: List[Dict[str, str]] = None,
                                memory_context: str = None,
                                use_citations: bool = True) -> str:
        """Build the original RAG prompt for backward compatibility."""
        
        # System instructions
        system_prompt = """You are an expert assistant specializing in LangChain, a framework for building applications with large language models. Your role is to provide accurate, helpful, and detailed answers based on the provided documentation context.

IMPORTANT GUIDELINES:
1. **Accuracy**: Base your answers strictly on the provided context from LangChain documentation
2. **Specificity**: Include specific module names, class names, function signatures, and parameters when relevant
3. **Code Examples**: Provide practical, working code examples whenever possible
4. **Clarity**: Structure responses with clear headings, bullet points, or numbered lists
5. **Completeness**: Address all parts of the user's question when the context allows
6. **Limitations**: If the context doesn't contain sufficient information, clearly state this
7. **Citations**: Reference specific sources when making claims or providing examples"""

        if use_citations:
            system_prompt += """
8. **Source References**: Include source references in your answer using [Source N] format where N is the source number"""

        # Add conversation history if provided
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                conversation_context += f"{role.upper()}: {content}\n"
            conversation_context += "\n"

        # Add memory context if available
        memory_section = ""
        if memory_context and memory_context.strip():
            memory_section = f"\nMEMORY CONTEXT (User profile and past conversations):\n{memory_context}\n"
        
        # Build the complete prompt
        prompt = f"""{system_prompt}

CONTEXT FROM LANGCHAIN DOCUMENTATION:
{context}
{memory_section}
{conversation_context}
USER QUESTION: {query}

Please provide a comprehensive answer based on the context above. Consider the user's profile and past interactions when personalizing your response. If you reference information from the context, indicate which source it came from."""

        return prompt
    
    def _process_response(self, response_text: str, retrieved_contexts: List[Dict[str, Any]],
                         use_citations: bool) -> Dict[str, Any]:
        """Process the LLM response to extract answer and citations."""
        
        # Clean up response
        answer = response_text.strip()
        
        # Extract citations if enabled
        citations = []
        confidence_score = 0.8  # Default confidence
        
        if use_citations and retrieved_contexts:
            # Look for source references in the response
            source_pattern = r'\[Source\s+(\d+)\]'
            mentioned_sources = set()
            
            for match in re.finditer(source_pattern, answer):
                source_num = int(match.group(1)) - 1  # Convert to 0-based index
                if 0 <= source_num < len(retrieved_contexts):
                    mentioned_sources.add(source_num)
            
            # Create citations for mentioned sources
            for source_idx in mentioned_sources:
                context = retrieved_contexts[source_idx]
                metadata = context.get("metadata", {})
                
                citation = {
                    "source_number": source_idx + 1,
                    "file_name": metadata.get("file_name", "Unknown"),
                    "doc_type": metadata.get("doc_type", "unknown"),
                    "module_path": metadata.get("module_path", ""),
                    "url": metadata.get("url", ""),
                    "similarity_score": context.get("score", 0.0),
                    "content_preview": context.get("content", "")[:200] + "..." if len(context.get("content", "")) > 200 else context.get("content", "")
                }
                citations.append(citation)
            
            # Adjust confidence based on citation quality
            if citations:
                avg_similarity = sum(c["similarity_score"] for c in citations) / len(citations)
                confidence_score = min(0.95, 0.6 + (avg_similarity * 0.35))
        
        # Additional confidence scoring based on response characteristics
        if len(answer) < 50:
            confidence_score *= 0.8  # Penalize very short answers
        elif "I don't know" in answer or "not enough information" in answer.lower():
            confidence_score *= 0.5  # Lower confidence for uncertain answers
        elif any(keyword in answer.lower() for keyword in ["langchain", "import", "class", "function"]):
            confidence_score = min(0.95, confidence_score + 0.1)  # Boost for technical content
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence_score": round(confidence_score, 3)
        }
    
    def generate_follow_up_questions(self, query: str, answer: str, 
                                   retrieved_contexts: List[Dict[str, Any]]) -> List[str]:
        """Generate relevant follow-up questions based on the query and answer."""
        try:
            context_topics = []
            for context in retrieved_contexts:
                metadata = context.get("metadata", {})
                if metadata.get("module_path"):
                    context_topics.append(metadata["module_path"])
                if metadata.get("doc_type"):
                    context_topics.append(metadata["doc_type"])
            
            topics_text = ", ".join(set(context_topics[:10]))  # Limit to avoid long prompts
            
            prompt = f"""Based on the following question and answer about LangChain, generate 3 relevant follow-up questions that a user might want to ask next.

Original Question: {query}
Answer: {answer[:500]}...
Related Topics: {topics_text}

Generate 3 specific, actionable follow-up questions that would help the user learn more about LangChain. Focus on practical implementation details, related features, or common use cases.

Format as a simple numbered list:
1. [Question 1]
2. [Question 2] 
3. [Question 3]"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}]}],
                config={
                    "max_output_tokens": 200,
                    "temperature": 0.7
                }
            )
            
            # Handle different response structures
            response_text = ""
            if hasattr(response, 'text') and response.text:
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        response_text = response.candidates[0].content.parts[0].text
            elif isinstance(response, dict):
                if 'candidates' in response and response['candidates']:
                    candidate = response['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        response_text = candidate['content']['parts'][0]['text']
            
            if response_text:
                # Parse numbered list
                questions = []
                for line in response_text.strip().split('\n'):
                    if re.match(r'^\d+\.', line.strip()):
                        question = re.sub(r'^\d+\.\s*', '', line.strip())
                        if question and len(question) > 10:
                            questions.append(question)
                
                return questions[:3]  # Limit to 3 questions
                
        except Exception as e:
            logger.warning(f"Failed to generate follow-up questions: {str(e)}")
            # Return predefined fallback questions based on query keywords
            return self._get_fallback_followup_questions(query, answer)
        
        return []
    
    def _get_fallback_followup_questions(self, query: str, answer: str) -> List[str]:
        """Generate fallback follow-up questions when API fails."""
        query_lower = query.lower()
        
        # Topic-specific follow-up questions
        if 'agent' in query_lower:
            return [
                "What are the different types of LangChain agents available?",
                "How do I add custom tools to a LangChain agent?",
                "What's the difference between LangGraph and traditional LangChain agents?"
            ]
        elif 'document' in query_lower or 'loader' in query_lower:
            return [
                "What are the most commonly used document loaders in LangChain?",
                "How do I handle different file formats with document loaders?",
                "What's the best way to chunk large documents after loading?"
            ]
        elif 'runnable' in query_lower:
            return [
                "How do I chain multiple Runnables together?",
                "What are the performance benefits of using Runnables?",
                "How do I handle errors and retries with Runnables?"
            ]
        elif 'vector' in query_lower or 'embedding' in query_lower:
            return [
                "Which vector stores work best with LangChain?",
                "How do I optimize embedding performance for large datasets?",
                "What are the best practices for vector similarity search?"
            ]
        elif 'rag' in query_lower or 'retrieval' in query_lower:
            return [
                "How do I improve RAG response quality?",
                "What are advanced RAG techniques beyond basic retrieval?",
                "How do I evaluate RAG system performance?"
            ]
        else:
            # Generic LangChain questions
            return [
                "What are the core components I should learn next in LangChain?",
                "How can I optimize the performance of my LangChain application?",
                "What are some common LangChain patterns and best practices?"
            ]
    
    def generate_direct_answer(self, query: str, include_disclaimer: bool = True) -> Dict[str, Any]:
        """
        Generate direct answer without RAG context (fallback when no relevant docs found).
        
        Args:
            query: User's question
            include_disclaimer: Whether to include disclaimer about not using documentation
            
        Returns:
            Dictionary with generation result
        """
        start_time = time.time()
        self.response_stats["total_requests"] += 1
        self.response_stats["direct_prompts_used"] += 1
        
        try:
            if self.use_adaptive_prompts:
                # Use adaptive direct prompt template
                prompt = self.direct_prompt.format(question=query)
                logger.debug("Using adaptive direct prompt template")
            else:
                # Fallback to original direct prompt system
                system_prompt = """You are a helpful AI assistant with expertise in Python programming, machine learning, and software development. 
                
Since no specific documentation was found to answer this question, provide a helpful, accurate response based on your general knowledge. 
Be informative, practical, and include code examples when appropriate."""
                
                # Add disclaimer if requested
                if include_disclaimer:
                    system_prompt += """

IMPORTANT: This response is based on general knowledge and not from specific documentation. 
For the most current and detailed information, please refer to the official documentation of the relevant tools or frameworks."""
                
                # Create the prompt
                prompt = f"""{system_prompt}

User Question: {query}

Please provide a comprehensive and helpful answer."""
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": prompt}]}],
                config=self.generation_config
            )
            
            if response and hasattr(response, 'text') and response.text:
                answer = response.text.strip()
                
                # Generate follow-up questions
                try:
                    follow_up_questions = self._generate_follow_up_for_direct_answer(query, answer)
                except Exception:
                    follow_up_questions = [
                        "Would you like more specific details about any aspect?",
                        "Do you need help with implementation?",
                        "Are there related topics you'd like to explore?"
                    ]
                
                # Update statistics
                response_time = (time.time() - start_time) * 1000
                self.response_stats["successful_responses"] += 1
                self.response_stats["total_response_time"] += response_time
                self.response_stats["avg_response_time"] = (
                    self.response_stats["total_response_time"] / 
                    self.response_stats["successful_responses"]
                )
                
                # Estimate tokens (rough approximation)
                estimated_tokens = len(answer.split()) * 1.3
                self.response_stats["total_tokens_used"] += estimated_tokens
                
                logger.info(f"✅ Direct LLM answer generated successfully ({response_time:.0f}ms)")
                
                return {
                    "success": True,
                    "answer": answer,
                    "follow_up_questions": follow_up_questions,
                    "response_time_ms": response_time,
                    "tokens_used": int(estimated_tokens),
                    "model_used": self.model_name,
                    "generation_type": "direct_llm_fallback"
                }
            else:
                raise Exception("Empty response from Gemini API")
                
        except Exception as e:
            self.response_stats["failed_responses"] += 1
            response_time = (time.time() - start_time) * 1000
            error_msg = str(e)
            logger.error(f"❌ Direct LLM generation failed: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "answer": "",
                "follow_up_questions": [],
                "response_time_ms": response_time,
                "tokens_used": 0,
                "model_used": self.model_name,
                "generation_type": "direct_llm_fallback"
            }
    
    def _generate_follow_up_for_direct_answer(self, query: str, answer: str) -> List[str]:
        """Generate follow-up questions for direct LLM answers."""
        try:
            follow_up_prompt = f"""Based on this Q&A exchange, generate 3 relevant follow-up questions that would help the user learn more or get additional practical help.

Question: {query}
Answer: {answer[:500]}...

Generate 3 follow-up questions as a simple list, one per line, without numbering or bullet points."""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"parts": [{"text": follow_up_prompt}]}],
                config={
                    **self.generation_config,
                    "max_output_tokens": 200,
                    "temperature": 0.7
                }
            )
            
            if response and hasattr(response, 'text') and response.text:
                questions = [
                    q.strip() 
                    for q in response.text.strip().split('\n') 
                    if q.strip() and not q.strip().startswith(('1.', '2.', '3.', '-', '•'))
                ][:3]
                
                return questions if questions else self._get_fallback_follow_ups(query)
            else:
                return self._get_fallback_follow_ups(query)
                
        except Exception as e:
            logger.warning(f"Failed to generate follow-up questions for direct answer: {e}")
            return self._get_fallback_follow_ups(query)
    
    def _get_fallback_follow_ups(self, query: str) -> List[str]:
        """Get fallback follow-up questions based on query type."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['code', 'example', 'implement', 'build', 'create']):
            return [
                "Would you like to see more specific code examples?",
                "Do you need help with error handling or best practices?",
                "Are there specific libraries or frameworks you'd like to use?"
            ]
        elif any(word in query_lower for word in ['how', 'what', 'why', 'explain']):
            return [
                "Would you like more detailed explanation of any aspect?",
                "Are there specific use cases you're working on?",
                "Do you need help with practical implementation?"
            ]
        else:
            return [
                "Would you like more specific details about this topic?",
                "Do you have a particular use case in mind?",
                "Are there related concepts you'd like to explore?"
            ]

    def update_conversation_history(self, query: str, answer: str):
        """Update conversation history for context."""
        self.conversation_history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        self.conversation_history.append({
            "role": "assistant", 
            "content": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 exchanges (20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent conversation history."""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get LLM generation statistics including adaptive prompting metrics."""
        total_requests = self.response_stats["total_requests"]
        successful_responses = self.response_stats["successful_responses"]
        
        stats = {
            **self.response_stats,
            "model_name": self.model_name,
            "adaptive_prompts_enabled": self.use_adaptive_prompts,
            "success_rate": (
                successful_responses / max(1, total_requests)
            ) * 100,
            "avg_tokens_per_request": (
                self.response_stats["total_tokens_used"] / 
                max(1, successful_responses)
            )
        }
        
        # Add adaptive prompting metrics if enabled
        if self.use_adaptive_prompts:
            stats.update({
                "adaptive_prompt_usage_rate": (
                    self.response_stats["adaptive_prompts_used"] / max(1, total_requests)
                ) * 100,
                "direct_prompt_usage_rate": (
                    self.response_stats["direct_prompts_used"] / max(1, total_requests)
                ) * 100
            })
        
        return stats


def create_llm_generator(model_name: str = None, api_key: str = None, use_adaptive_prompts: bool = True) -> GeminiLLMGenerator:
    """Create and initialize LLM generator with optional adaptive prompting."""
    return GeminiLLMGenerator(model_name=model_name, api_key=api_key, use_adaptive_prompts=use_adaptive_prompts)