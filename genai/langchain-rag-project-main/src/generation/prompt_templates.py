#!/usr/bin/env python3
"""
Enhanced prompt templates for technical RAG systems.

This module implements the Adaptive Multi-Type Prompt system that automatically
adapts response format based on the type of content retrieved and question asked.
"""

from langchain.prompts import PromptTemplate
from typing import Dict, List, Any, Optional
import re
from enum import Enum


class ContentType(Enum):
    """Content types for adaptive prompting."""
    API_DOCUMENTATION = "api_documentation"
    TUTORIAL_GUIDE = "tutorial_guide"
    CODE_EXAMPLES = "code_examples"
    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    MIXED = "mixed"


class AdaptivePromptTemplates:
    """Collection of adaptive prompt templates for technical RAG systems."""
    
    def __init__(self):
        """Initialize the adaptive prompt templates."""
        self.adaptive_technical_prompt = self._create_adaptive_technical_prompt()
        self.direct_llm_prompt = self._create_direct_llm_prompt()
        self.context_analysis_prompt = self._create_context_analysis_prompt()
    
    def _create_adaptive_technical_prompt(self) -> PromptTemplate:
        """Create the main adaptive technical prompt template."""
        template = """You are a comprehensive technical assistant. Analyze the context to determine the best response approach for this technical question.

CONTEXT:
{context}

QUESTION: {question}

CONTEXT ANALYSIS:
First, identify what types of content are in the context:
- [ ] API Documentation
- [ ] Tutorial/Guide content  
- [ ] Code examples/snippets
- [ ] Configuration files
- [ ] Troubleshooting guides

ADAPTIVE RESPONSE RULES:
📚 **If mostly documentation**: Focus on official specs, parameters, and usage
🎓 **If mostly tutorials**: Provide step-by-step educational explanations
💻 **If mostly code**: Give working examples with detailed explanations
⚙️ **If configuration-heavy**: Include setup steps and configuration options
🔧 **If troubleshooting content**: Emphasize problem-solving and debugging

TECHNICAL RESPONSE REQUIREMENTS:
✅ Always include working code examples when relevant
✅ Cite specific sections/files from the context
✅ Mention version compatibility if specified
✅ Include both basic and advanced usage patterns
✅ Add error handling and edge cases
✅ Provide testing/validation steps

RESPONSE FORMAT:
**Quick Answer**: 3-4-sentence direct response

**Implementation**:
```language
// Complete, runnable code with comments
```

**Explanation**: 
- Step-by-step breakdown
- Key concepts and reasoning
- Integration points and dependencies

**Additional Context**:
- Best practices from the documentation
- Common pitfalls and solutions
- Related functionality and next steps

[Sources: Reference specific docs/files from context]

ANSWER:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _create_direct_llm_prompt(self) -> PromptTemplate:
        """Create prompt template for direct LLM queries (fallback)."""
        template = """You are a helpful technical assistant. The user has asked a programming or technical question that doesn't relate to the specific documentation in your knowledge base.

QUESTION: {question}

Please provide a comprehensive answer that includes:

**Quick Answer**: Direct response to the question

**Implementation** (if applicable):
```language
// Complete, working code example with comments
```

**Explanation**:
- Clear step-by-step breakdown
- Key concepts and best practices
- Common use cases and variations

**Additional Notes**:
- Important considerations and gotchas
- Alternative approaches if relevant
- Testing and validation tips

ANSWER:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["question"]
        )
    
    def _create_context_analysis_prompt(self) -> PromptTemplate:
        """Create prompt template for analyzing context type."""
        template = """Analyze the following context and determine the primary content type:

CONTEXT:
{context}

Based on the content, identify the primary type:
- API_DOCUMENTATION: Official API docs, function signatures, parameters
- TUTORIAL_GUIDE: Step-by-step tutorials, how-to guides, learning materials
- CODE_EXAMPLES: Code snippets, examples, implementations
- CONFIGURATION: Config files, setup instructions, environment settings
- TROUBLESHOOTING: Error handling, debugging, problem-solving guides
- MIXED: Multiple content types present

Return only the content type (e.g., "API_DOCUMENTATION"):"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context"]
        )


class ContextAnalyzer:
    """Utility class for analyzing context content types."""
    
    def __init__(self):
        """Initialize the context analyzer."""
        self.api_keywords = [
            'api', 'endpoint', 'parameter', 'response', 'request', 
            'method', 'function', 'class', 'module', 'library'
        ]
        self.tutorial_keywords = [
            'tutorial', 'guide', 'step', 'how to', 'getting started',
            'introduction', 'example', 'walkthrough', 'lesson'
        ]
        self.code_keywords = [
            'def ', 'class ', 'import ', 'from ', '```', 'code',
            'example', 'snippet', 'implementation'
        ]
        self.config_keywords = [
            'config', 'settings', 'environment', 'setup', 'installation',
            'requirements', '.env', 'yaml', 'json', 'toml'
        ]
        self.troubleshooting_keywords = [
            'error', 'exception', 'problem', 'issue', 'debug',
            'troubleshoot', 'fix', 'solve', 'warning', 'fail'
        ]
    
    def analyze_content_type(self, context: str) -> ContentType:
        """
        Analyze context content and determine the primary type.
        
        Args:
            context: The context string to analyze
            
        Returns:
            ContentType enum value
        """
        if not context or not context.strip():
            return ContentType.MIXED
        
        context_lower = context.lower()
        
        # Count keyword occurrences
        scores = {
            ContentType.API_DOCUMENTATION: self._count_keywords(context_lower, self.api_keywords),
            ContentType.TUTORIAL_GUIDE: self._count_keywords(context_lower, self.tutorial_keywords),
            ContentType.CODE_EXAMPLES: self._count_keywords(context_lower, self.code_keywords),
            ContentType.CONFIGURATION: self._count_keywords(context_lower, self.config_keywords),
            ContentType.TROUBLESHOOTING: self._count_keywords(context_lower, self.troubleshooting_keywords)
        }
        
        # Find the type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return ContentType.MIXED
        
        # Check for mixed content (multiple high scores)
        high_score_types = [t for t, s in scores.items() if s >= max_score * 0.7]
        if len(high_score_types) > 1:
            return ContentType.MIXED
        
        return max(scores, key=scores.get)
    
    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords in text."""
        count = 0
        for keyword in keywords:
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
        return count
    
    def get_content_metadata(self, context: str) -> Dict[str, Any]:
        """
        Get detailed metadata about the context content.
        
        Args:
            context: The context string to analyze
            
        Returns:
            Dictionary with content analysis metadata
        """
        content_type = self.analyze_content_type(context)
        
        metadata = {
            'primary_type': content_type.value,
            'word_count': len(context.split()) if context else 0,
            'has_code_blocks': '```' in context,
            'has_links': 'http' in context.lower() or '[' in context,
            'estimated_complexity': self._estimate_complexity(context)
        }
        
        return metadata
    
    def _estimate_complexity(self, context: str) -> str:
        """Estimate complexity level of the content."""
        if not context:
            return "unknown"
        
        word_count = len(context.split())
        code_blocks = context.count('```')
        technical_terms = len(re.findall(r'\b(?:class|function|method|api|parameter|configuration)\b', context.lower()))
        
        if word_count < 100 and code_blocks == 0:
            return "basic"
        elif word_count < 500 and technical_terms < 5:
            return "intermediate"
        else:
            return "advanced"


# Singleton instances for easy import
adaptive_prompts = AdaptivePromptTemplates()
context_analyzer = ContextAnalyzer()


def get_adaptive_prompt() -> PromptTemplate:
    """Get the adaptive technical prompt template."""
    return adaptive_prompts.adaptive_technical_prompt


def get_direct_llm_prompt() -> PromptTemplate:
    """Get the direct LLM prompt template."""
    return adaptive_prompts.direct_llm_prompt


def get_context_analysis_prompt() -> PromptTemplate:
    """Get the context analysis prompt template."""
    return adaptive_prompts.context_analysis_prompt


def analyze_context_type(context: str) -> ContentType:
    """Analyze and return the primary content type."""
    return context_analyzer.analyze_content_type(context)


def get_context_metadata(context: str) -> Dict[str, Any]:
    """Get detailed metadata about context content."""
    return context_analyzer.get_content_metadata(context)