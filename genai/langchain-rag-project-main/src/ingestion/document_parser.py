"""Multi-format document parser for LangChain documentation."""

import json
import re
import ast
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

import pandas as pd
from bs4 import BeautifulSoup
import nbformat

logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Document metadata structure."""
    
    def __init__(self, source_file: Path):
        self.source_file = str(source_file)
        self.file_name = source_file.name
        self.file_size = source_file.stat().st_size if source_file.exists() else 0
        self.last_modified = datetime.fromtimestamp(
            source_file.stat().st_mtime
        ).isoformat() if source_file.exists() else None
        self.doc_type = self._determine_doc_type(source_file)
        self.module_path = self._extract_module_path(source_file)
        self.title = ""
        self.section_hierarchy = []
        self.code_snippets = []
        self.url = self._generate_github_url(source_file)
        self.content_hash = ""
        
    def _determine_doc_type(self, source_file: Path) -> str:
        """Determine document type based on file path and content."""
        file_path_lower = str(source_file).lower()
        
        if "api" in file_path_lower or "reference" in file_path_lower:
            return "api"
        elif "tutorial" in file_path_lower or "getting_started" in file_path_lower:
            return "tutorial"
        elif "how_to" in file_path_lower or "guide" in file_path_lower:
            return "guide"
        elif "example" in file_path_lower or "cookbook" in file_path_lower:
            return "example"
        elif source_file.suffix == ".ipynb":
            return "notebook"
        elif source_file.suffix == ".py":
            return "code"
        else:
            return "documentation"
    
    def _extract_module_path(self, source_file: Path) -> str:
        """Extract Python module path from file path."""
        path_parts = source_file.parts
        
        # Find langchain in the path
        try:
            langchain_idx = next(
                i for i, part in enumerate(path_parts) 
                if "langchain" in part.lower()
            )
            module_parts = path_parts[langchain_idx:]
            
            # Remove file extension
            if module_parts:
                last_part = module_parts[-1]
                if "." in last_part:
                    module_parts = module_parts[:-1] + (last_part.split('.')[0],)
            
            return ".".join(module_parts)
        except (StopIteration, IndexError):
            return str(source_file.stem)
    
    def _generate_github_url(self, source_file: Path) -> str:
        """Generate GitHub URL for the source file."""
        path_str = str(source_file)
        if "langchain" in path_str:
            # Extract relative path from langchain directory
            parts = Path(path_str).parts
            try:
                langchain_idx = next(
                    i for i, part in enumerate(parts) 
                    if part == "langchain"
                )
                # Skip 'langchain' directory to get correct path
                relative_path = "/".join(parts[langchain_idx + 1:])
                return f"https://github.com/langchain-ai/langchain/blob/master/{relative_path}"
            except StopIteration:
                pass
        
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source_file": self.source_file,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "last_modified": self.last_modified,
            "doc_type": self.doc_type,
            "module_path": self.module_path,
            "title": self.title,
            "section_hierarchy": self.section_hierarchy,
            "code_snippets": self.code_snippets,
            "url": self.url,
            "content_hash": self.content_hash
        }


class MultiFormatParser:
    """Parser for multiple document formats."""
    
    def __init__(self):
        self.supported_formats = {'.md', '.mdx', '.rst', '.py', '.ipynb', '.txt'}
    
    def parse_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a document and extract content and metadata."""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
            
        if file_path.suffix not in self.supported_formats:
            logger.debug(f"Unsupported format: {file_path.suffix}")
            return None
        
        try:
            metadata = DocumentMetadata(file_path)
            
            # Parse content based on file type
            if file_path.suffix in {'.md', '.mdx'}:
                content = self._parse_markdown(file_path, metadata)
            elif file_path.suffix == '.rst':
                content = self._parse_rst(file_path, metadata)
            elif file_path.suffix == '.py':
                content = self._parse_python(file_path, metadata)
            elif file_path.suffix == '.ipynb':
                content = self._parse_notebook(file_path, metadata)
            else:
                content = self._parse_text(file_path, metadata)
            
            if not content or not content.strip():
                logger.debug(f"No content extracted from {file_path}")
                return None
            
            # Generate content hash
            metadata.content_hash = hashlib.md5(content.encode()).hexdigest()
            
            return {
                "content": content,
                "metadata": metadata.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return None
    
    def _parse_markdown(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Parse Markdown files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract title from first header
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata.title = title_match.group(1).strip()
        
        # Extract section hierarchy
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata.section_hierarchy = [
            {"level": len(h[0]), "title": h[1].strip()} 
            for h in headers
        ]
        
        # Extract code blocks
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        metadata.code_snippets = [
            {"language": lang or "text", "code": code.strip()}
            for lang, code in code_blocks
        ]
        
        return content
    
    def _parse_rst(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Parse ReStructuredText files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract title (RST uses underlines)
        lines = content.split('\n')
        for i, line in enumerate(lines[1:], 1):
            if line.strip() and all(c in '=-~^' for c in line.strip()):
                if i > 0 and lines[i-1].strip():
                    metadata.title = lines[i-1].strip()
                    break
        
        # Extract code blocks
        code_blocks = re.findall(r'\.\. code-block::\s*(\w+)?\n\n(.*?)(?=\n\S|\Z)', 
                                content, re.DOTALL)
        metadata.code_snippets = [
            {"language": lang or "text", "code": code.strip()}
            for lang, code in code_blocks
        ]
        
        return content
    
    def _parse_python(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Parse Python files and extract docstrings."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            # Parse AST to extract docstrings
            tree = ast.parse(content)
            docstrings = []
            
            # Extract module docstring
            if (ast.get_docstring(tree)):
                docstrings.append(f"Module: {ast.get_docstring(tree)}")
            
            # Extract class and function docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('_'):
                        continue  # Skip private functions
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings.append(f"Function {node.name}: {docstring}")
                
                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstrings.append(f"Class {node.name}: {docstring}")
            
            if docstrings:
                metadata.title = f"Python Module: {file_path.stem}"
                return "\n\n".join(docstrings)
            
        except SyntaxError:
            logger.warning(f"Syntax error in Python file: {file_path}")
        
        # If AST parsing fails, return the raw content
        return content
    
    def _parse_notebook(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Parse Jupyter notebooks."""
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        content_parts = []
        code_snippets = []
        
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                content_parts.append(cell.source)
                
                # Extract title from first markdown cell
                if not metadata.title and cell.source.strip():
                    first_line = cell.source.split('\n')[0]
                    if first_line.startswith('#'):
                        metadata.title = first_line.lstrip('#').strip()
                        
            elif cell.cell_type == 'code':
                if cell.source.strip():
                    code_snippets.append({
                        "language": "python",
                        "code": cell.source
                    })
                    content_parts.append(f"```python\n{cell.source}\n```")
        
        metadata.code_snippets = code_snippets
        
        if not metadata.title:
            metadata.title = f"Notebook: {file_path.stem}"
        
        return "\n\n".join(content_parts)
    
    def _parse_text(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Parse plain text files."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Use first non-empty line as title
        lines = content.split('\n')
        for line in lines:
            if line.strip():
                metadata.title = line.strip()
                break
        
        return content
    
    def parse_multiple_documents(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """Parse multiple documents."""
        documents = []
        
        for file_path in file_paths:
            doc = self.parse_document(file_path)
            if doc:
                documents.append(doc)
                if len(documents) % 100 == 0:
                    logger.info(f"Parsed {len(documents)} documents...")
        
        logger.info(f"Successfully parsed {len(documents)} out of {len(file_paths)} documents")
        return documents


def parse_langchain_documents(doc_paths: Dict[str, List[Path]]) -> Dict[str, List[Dict[str, Any]]]:
    """Parse all LangChain documentation files."""
    parser = MultiFormatParser()
    parsed_docs = {}
    
    for doc_type, paths in doc_paths.items():
        logger.info(f"Parsing {len(paths)} {doc_type} files...")
        parsed_docs[doc_type] = parser.parse_multiple_documents(paths)
    
    return parsed_docs