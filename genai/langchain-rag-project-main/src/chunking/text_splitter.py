"""Advanced text chunking strategies for LangChain documentation."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import settings

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Base class for document chunkers."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
        
    @abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any], 
                     chunk_index: int, start_char: int = 0) -> Dict[str, Any]:
        """Create a standardized chunk with metadata."""
        return {
            "content": content,
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_size": len(content),
                "token_count": self.count_tokens(content),
                "start_char": start_char
            }
        }


class SemanticHierarchicalChunker(BaseChunker):
    """Semantic chunking that respects document structure and preserves context."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, 
                 min_chunk_size: int = 100, max_chunk_size: int = 1500):
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Separators for recursive splitting (in order of priority)
        self.separators = [
            "\n\n# ",      # Main headers
            "\n\n## ",     # Sub headers
            "\n\n### ",    # Sub-sub headers
            "\n\n#### ",   # Minor headers
            "\n\n",        # Double newlines (paragraph breaks)
            "\n",          # Single newlines
            ". ",          # Sentences
            " ",           # Words
            ""             # Characters
        ]
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text using semantic and hierarchical approach."""
        if not text.strip():
            return []
            
        metadata = metadata or {}
        doc_type = metadata.get("doc_type", "")
        
        # Choose splitting strategy based on document type
        if doc_type == "notebook":
            return self._split_notebook(text, metadata)
        elif doc_type == "code":
            return self._split_code_file(text, metadata)
        else:
            return self._split_markdown(text, metadata)
    
    def _split_markdown(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split markdown documents preserving structure."""
        chunks = []
        
        # First, try to split by major sections
        sections = self._split_by_headers(text)
        
        for section_index, (header, content) in enumerate(sections):
            if not content.strip():
                continue
                
            section_metadata = {
                **metadata,
                "section_header": header,
                "section_index": section_index
            }
            
            # If section is small enough, keep as one chunk
            if self.count_tokens(content) <= self.chunk_size:
                chunk = self._create_chunk(content, section_metadata, len(chunks))
                chunks.append(chunk)
            else:
                # Split large sections recursively
                section_chunks = self._recursive_split(content, section_metadata)
                chunks.extend(section_chunks)
        
        # If no sections found, fall back to recursive splitting
        if not chunks:
            chunks = self._recursive_split(text, metadata)
        
        return chunks
    
    def _split_notebook(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split notebook content preserving cell boundaries."""
        chunks = []
        
        # Split by code blocks (notebook cells)
        cell_pattern = r'```python\n(.*?)\n```'
        parts = re.split(cell_pattern, text, flags=re.DOTALL)
        
        current_chunk = ""
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            # Check if adding this part would exceed chunk size
            potential_chunk = current_chunk + part
            
            if self.count_tokens(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_metadata = {**metadata, "cell_type": "mixed"}
                    chunk = self._create_chunk(current_chunk, chunk_metadata, len(chunks))
                    chunks.append(chunk)
                current_chunk = part
        
        # Add final chunk
        if current_chunk.strip():
            chunk_metadata = {**metadata, "cell_type": "mixed"}
            chunk = self._create_chunk(current_chunk, chunk_metadata, len(chunks))
            chunks.append(chunk)
        
        return chunks if chunks else self._recursive_split(text, metadata)
    
    def _split_code_file(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split code files preserving function/class boundaries."""
        chunks = []
        
        # Try to split by functions and classes
        function_pattern = r'((?:def|class|async def)\s+\w+.*?:(?:\n.*?)*?)(?=\ndef|\nclass|\nasync def|\Z)'
        matches = re.finditer(function_pattern, text, re.DOTALL | re.MULTILINE)
        
        last_end = 0
        for match in matches:
            start, end = match.span()
            
            # Add any content between functions
            if start > last_end:
                between_content = text[last_end:start].strip()
                if between_content:
                    chunk_metadata = {**metadata, "code_type": "module_level"}
                    chunk = self._create_chunk(between_content, chunk_metadata, len(chunks))
                    chunks.append(chunk)
            
            # Add the function/class
            func_content = match.group(1)
            if self.count_tokens(func_content) <= self.max_chunk_size:
                chunk_metadata = {**metadata, "code_type": "function_or_class"}
                chunk = self._create_chunk(func_content, chunk_metadata, len(chunks))
                chunks.append(chunk)
            else:
                # Split large functions
                sub_chunks = self._recursive_split(func_content, 
                                                 {**metadata, "code_type": "large_function"})
                chunks.extend(sub_chunks)
            
            last_end = end
        
        # Add remaining content
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                chunk_metadata = {**metadata, "code_type": "module_level"}
                chunk = self._create_chunk(remaining, chunk_metadata, len(chunks))
                chunks.append(chunk)
        
        return chunks if chunks else self._recursive_split(text, metadata)
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        """Split text by markdown headers."""
        sections = []
        
        # Find all headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        headers = list(re.finditer(header_pattern, text, re.MULTILINE))
        
        if not headers:
            return [("", text)]
        
        for i, header_match in enumerate(headers):
            header_level = len(header_match.group(1))
            header_text = header_match.group(2)
            header_start = header_match.start()
            
            # Find the end of this section
            section_end = len(text)
            for j in range(i + 1, len(headers)):
                next_header = headers[j]
                next_level = len(next_header.group(1))
                
                # End section when we hit a header of equal or higher level
                if next_level <= header_level:
                    section_end = next_header.start()
                    break
            
            # Extract section content
            section_content = text[header_start:section_end].strip()
            sections.append((header_text, section_content))
        
        return sections
    
    def _recursive_split(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recursively split text using LangChain's splitter."""
        chunks = []
        
        # Use LangChain's recursive splitter as fallback
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=self.count_tokens
        )
        
        # Preserve code blocks
        text_chunks = self._split_preserving_code_blocks(text, splitter)
        
        for i, chunk_text in enumerate(text_chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk = self._create_chunk(chunk_text, metadata, len(chunks))
                chunks.append(chunk)
        
        return chunks
    
    def _split_preserving_code_blocks(self, text: str, splitter) -> List[str]:
        """Split text while preserving code blocks intact."""
        # Find code blocks
        code_block_pattern = r'```[\w]*\n(.*?)\n```'
        code_blocks = list(re.finditer(code_block_pattern, text, re.DOTALL))
        
        if not code_blocks:
            return splitter.split_text(text)
        
        chunks = []
        last_end = 0
        
        for code_match in code_blocks:
            start, end = code_match.span()
            
            # Split text before code block
            if start > last_end:
                before_text = text[last_end:start]
                chunks.extend(splitter.split_text(before_text))
            
            # Add code block as single chunk (if not too large)
            code_block = code_match.group(0)
            if self.count_tokens(code_block) <= self.max_chunk_size:
                chunks.append(code_block)
            else:
                # If code block is too large, split it but try to preserve structure
                chunks.extend(splitter.split_text(code_block))
            
            last_end = end
        
        # Split remaining text
        if last_end < len(text):
            remaining = text[last_end:]
            chunks.extend(splitter.split_text(remaining))
        
        return chunks


class DocumentChunker:
    """Main document chunking orchestrator."""
    
    def __init__(self, strategy: str = "semantic_hierarchical"):
        self.strategy = strategy
        
        if strategy == "semantic_hierarchical":
            self.chunker = SemanticHierarchicalChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document."""
        content = document["content"]
        metadata = document["metadata"]
        
        if not content.strip():
            logger.warning(f"Empty content in document: {metadata.get('source_file', 'unknown')}")
            return []
        
        try:
            chunks = self.chunker.split_text(content, metadata)
            
            # Add document-level metadata to each chunk
            for chunk in chunks:
                chunk["metadata"]["document_id"] = metadata.get("content_hash", "")
                chunk["metadata"]["source_document"] = metadata.get("source_file", "")
            
            logger.debug(f"Split document into {len(chunks)} chunks: {metadata.get('source_file', 'unknown')}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {metadata.get('source_file', 'unknown')}: {str(e)}")
            return []
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for i, document in enumerate(documents):
            chunks = self.chunk_document(document)
            
            # Add global chunk index
            for chunk in chunks:
                chunk["metadata"]["global_chunk_index"] = len(all_chunks)
                all_chunks.append(chunk)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Chunked {i + 1}/{len(documents)} documents, {len(all_chunks)} total chunks")
        
        logger.info(f"Completed chunking: {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks


def chunk_langchain_documents(documents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Chunk all LangChain documents by type."""
    chunker = DocumentChunker()
    chunked_docs = {}
    
    for doc_type, docs in documents.items():
        if not docs:
            continue
            
        logger.info(f"Chunking {len(docs)} {doc_type} documents...")
        chunks = chunker.chunk_documents(docs)
        chunked_docs[doc_type] = chunks
        
        # Log statistics
        if chunks:
            avg_chunk_size = sum(chunk["metadata"]["chunk_size"] for chunk in chunks) / len(chunks)
            avg_tokens = sum(chunk["metadata"]["token_count"] for chunk in chunks) / len(chunks)
            logger.info(f"  {doc_type}: {len(chunks)} chunks, avg size: {avg_chunk_size:.0f} chars, {avg_tokens:.0f} tokens")
    
    return chunked_docs