"""Unstructured.io-based document parser for enhanced document processing."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

from unstructured.partition.auto import partition

# Try to import element types for classification
try:
    from unstructured.documents.elements import Element, Text, Title, NarrativeText, ListItem, Table
    ELEMENT_TYPES_AVAILABLE = True
except ImportError:
    ELEMENT_TYPES_AVAILABLE = False
    logger.warning("Unstructured element types not available, using basic text extraction")

logger = logging.getLogger(__name__)


class EnhancedDocumentMetadata:
    """Enhanced document metadata structure with Unstructured.io capabilities."""
    
    def __init__(self, source_file: Path, elements: List[Element] = None):
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
        self.tables = []
        self.list_items = []
        self.narrative_text_count = 0
        self.element_counts = {}
        self.url = self._generate_github_url(source_file)
        self.content_hash = ""
        
        if elements:
            self._extract_enhanced_metadata(elements)
        
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
    
    def _extract_enhanced_metadata(self, elements):
        """Extract enhanced metadata from Unstructured elements."""
        self.element_counts = {}
        titles = []
        current_hierarchy = []
        
        for element in elements:
            element_type = type(element).__name__
            self.element_counts[element_type] = self.element_counts.get(element_type, 0) + 1
            
            # Check if element types are available for classification
            if ELEMENT_TYPES_AVAILABLE:
                # Extract titles and hierarchy
                if isinstance(element, Title):
                    title_text = element.text.strip()
                    titles.append(title_text)
                    
                    # Determine hierarchy level from metadata if available
                    metadata = getattr(element, 'metadata', {})
                    level = 1
                    
                    # Try to infer level from context or formatting
                    if hasattr(element, 'category_depth'):
                        level = getattr(element, 'category_depth', 1)
                    
                    hierarchy_item = {
                        "level": level,
                        "title": title_text
                    }
                    current_hierarchy.append(hierarchy_item)
                    
                    # Set main title from first title element
                    if not self.title and title_text:
                        self.title = title_text
                
                # Count narrative text
                elif isinstance(element, NarrativeText):
                    self.narrative_text_count += 1
                
                # Extract tables
                elif isinstance(element, Table):
                    if hasattr(element, 'text') and element.text:
                        self.tables.append(element.text)
                
                # Extract list items
                elif isinstance(element, ListItem):
                    if hasattr(element, 'text') and element.text:
                        self.list_items.append(element.text)
            
            # Extract code snippets from any element
            if hasattr(element, 'text') and element.text:
                text = element.text.strip()
                # Check if this looks like code (simple heuristic)
                if (text.count('\n') > 2 and 
                    any(keyword in text for keyword in ['def ', 'class ', 'import ', 'from ', '```'])):
                    
                    # Try to determine language
                    language = "text"
                    if "```python" in text or "def " in text or "import " in text:
                        language = "python"
                    elif "```javascript" in text or "```js" in text:
                        language = "javascript"
                    elif "```bash" in text or "```sh" in text:
                        language = "bash"
                    
                    self.code_snippets.append({
                        "language": language,
                        "code": text
                    })
        
        self.section_hierarchy = current_hierarchy
        
        # If no title was found from Title elements, try first element with text
        if not self.title and elements:
            for element in elements:
                if hasattr(element, 'text') and element.text.strip():
                    first_line = element.text.split('\n')[0].strip()
                    if len(first_line) < 100:  # Reasonable title length
                        self.title = first_line
                        break
    
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
            "tables": self.tables,
            "list_items": self.list_items,
            "narrative_text_count": self.narrative_text_count,
            "element_counts": self.element_counts,
            "url": self.url,
            "content_hash": self.content_hash
        }


class UnstructuredDocumentParser:
    """Enhanced document parser using Unstructured.io."""
    
    def __init__(self):
        self.supported_formats = {'.md', '.mdx', '.rst', '.py', '.ipynb', '.txt', '.pdf', '.docx', '.html'}
    
    def parse_document(self, file_path: Path, extract_tables: bool = True, 
                      extract_images: bool = False) -> Optional[Dict[str, Any]]:
        """Parse a document using Unstructured.io and extract enhanced content and metadata."""
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
            
        if file_path.suffix not in self.supported_formats:
            logger.debug(f"Unsupported format: {file_path.suffix}")
            return None
        
        try:
            # Use format-specific partition functions for better results
            elements = self._partition_document(file_path, extract_tables, extract_images)
            
            if not elements:
                logger.debug(f"No elements extracted from {file_path}")
                return None
            
            # Extract content from elements
            content = self._elements_to_content(elements)
            
            if not content or not content.strip():
                logger.debug(f"No content extracted from {file_path}")
                return None
            
            # Create enhanced metadata
            metadata = EnhancedDocumentMetadata(file_path, elements)
            
            # Generate content hash
            metadata.content_hash = hashlib.md5(content.encode()).hexdigest()
            
            return {
                "content": content,
                "metadata": metadata.to_dict(),
                "elements": [self._element_to_dict(el) for el in elements[:50]]  # First 50 elements for analysis
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return None
    
    def _partition_document(self, file_path: Path, extract_tables: bool, 
                          extract_images: bool) -> list:
        """Partition document using Unstructured.io auto-partition."""
        
        # Special handling for Jupyter notebooks - skip unstructured and use nbformat
        if file_path.suffix == '.ipynb':
            return self._parse_notebook_with_nbformat(file_path)
        
        try:
            # Use auto-partition which handles all formats automatically
            elements = partition(
                filename=str(file_path),
                include_metadata=True,
                metadata_filename=file_path.name
            )
            
            return elements
            
        except Exception as e:
            logger.warning(f"Partition failed for {file_path}, trying with basic parameters: {e}")
            try:
                # Fallback with minimal parameters
                elements = partition(filename=str(file_path))
                return elements
            except Exception as e2:
                logger.error(f"All partition attempts failed for {file_path}: {e2}")
                return []
    
    def _parse_notebook_with_nbformat(self, file_path: Path) -> list:
        """Parse Jupyter notebook using nbformat as fallback for corrupted files."""
        try:
            import nbformat
            from unstructured.documents.elements import Text, Title
            
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            elements = []
            
            for cell in notebook.cells:
                if cell.cell_type == 'markdown' and cell.source.strip():
                    # Create Title element for headers, Text for other markdown
                    lines = cell.source.split('\n')
                    if lines[0].strip().startswith('#'):
                        # This is a header
                        header_text = lines[0].strip().lstrip('#').strip()
                        elements.append(Title(text=header_text))
                        
                        # Add remaining content as Text if any
                        remaining_content = '\n'.join(lines[1:]).strip()
                        if remaining_content:
                            elements.append(Text(text=remaining_content))
                    else:
                        # Regular markdown content
                        elements.append(Text(text=cell.source))
                        
                elif cell.cell_type == 'code' and cell.source.strip():
                    # Add code as text with special marking
                    code_text = f"```python\n{cell.source}\n```"
                    elements.append(Text(text=code_text))
            
            logger.debug(f"Successfully parsed notebook {file_path.name} using nbformat fallback")
            return elements
            
        except Exception as e:
            logger.warning(f"Failed to parse notebook {file_path} with nbformat fallback: {e}")
            return []
    
    def _elements_to_content(self, elements: list) -> str:
        """Convert elements to text content while preserving structure."""
        content_parts = []
        
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text = element.text.strip()
                if not text:
                    continue
                
                # Add formatting based on element type if available
                if ELEMENT_TYPES_AVAILABLE:
                    if isinstance(element, Title):
                        # Add markdown-style headers
                        level = getattr(element, 'category_depth', 1)
                        header_prefix = "#" * min(level, 6)
                        content_parts.append(f"{header_prefix} {text}")
                    elif isinstance(element, ListItem):
                        content_parts.append(f"• {text}")
                    elif isinstance(element, Table):
                        content_parts.append(f"\n{text}\n")
                    else:
                        content_parts.append(text)
                else:
                    # Basic text extraction without type checking
                    content_parts.append(text)
        
        return "\n\n".join(content_parts)
    
    def _element_to_dict(self, element) -> Dict[str, Any]:
        """Convert element to dictionary for analysis."""
        element_dict = {
            "type": type(element).__name__,
            "text": getattr(element, 'text', '')[:500],  # First 500 chars
        }
        
        # Add metadata if available
        if hasattr(element, 'metadata'):
            element_dict["metadata"] = getattr(element, 'metadata', {})
        
        # Add category information
        if hasattr(element, 'category'):
            element_dict["category"] = getattr(element, 'category', '')
        
        return element_dict
    
    def parse_multiple_documents(self, file_paths: List[Path], 
                               extract_tables: bool = True,
                               extract_images: bool = False) -> List[Dict[str, Any]]:
        """Parse multiple documents with enhanced features."""
        documents = []
        
        for file_path in file_paths:
            doc = self.parse_document(file_path, extract_tables, extract_images)
            if doc:
                documents.append(doc)
                if len(documents) % 50 == 0:
                    logger.info(f"Parsed {len(documents)} documents with Unstructured.io...")
        
        logger.info(f"Successfully parsed {len(documents)} out of {len(file_paths)} documents using Unstructured.io")
        return documents


def parse_langchain_documents_unstructured(doc_paths: Dict[str, List[Path]], 
                                         extract_tables: bool = True,
                                         extract_images: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """Parse all LangChain documentation files using Unstructured.io."""
    parser = UnstructuredDocumentParser()
    parsed_docs = {}
    
    for doc_type, paths in doc_paths.items():
        logger.info(f"Parsing {len(paths)} {doc_type} files with Unstructured.io...")
        parsed_docs[doc_type] = parser.parse_multiple_documents(
            paths, extract_tables, extract_images
        )
    
    return parsed_docs


def compare_parsing_methods(file_paths: List[Path], sample_size: int = 10) -> Dict[str, Any]:
    """Compare custom parser vs Unstructured.io parser on a sample of documents."""
    from .document_parser import MultiFormatParser
    
    # Sample files for comparison
    sample_paths = file_paths[:sample_size] if len(file_paths) > sample_size else file_paths
    
    custom_parser = MultiFormatParser()
    unstructured_parser = UnstructuredDocumentParser()
    
    results = {
        "custom_parser": {"successful": 0, "failed": 0, "docs": []},
        "unstructured_parser": {"successful": 0, "failed": 0, "docs": []},
        "comparison": []
    }
    
    for file_path in sample_paths:
        logger.info(f"Comparing parsers on: {file_path.name}")
        
        # Custom parser
        custom_result = custom_parser.parse_document(file_path)
        if custom_result:
            results["custom_parser"]["successful"] += 1
            results["custom_parser"]["docs"].append(custom_result)
        else:
            results["custom_parser"]["failed"] += 1
        
        # Unstructured parser
        unstructured_result = unstructured_parser.parse_document(file_path)
        if unstructured_result:
            results["unstructured_parser"]["successful"] += 1
            results["unstructured_parser"]["docs"].append(unstructured_result)
        else:
            results["unstructured_parser"]["failed"] += 1
        
        # Compare results
        comparison = {
            "file": str(file_path),
            "custom_success": custom_result is not None,
            "unstructured_success": unstructured_result is not None,
            "custom_content_length": len(custom_result["content"]) if custom_result else 0,
            "unstructured_content_length": len(unstructured_result["content"]) if unstructured_result else 0,
        }
        
        if custom_result and unstructured_result:
            comparison["content_length_diff"] = (
                len(unstructured_result["content"]) - len(custom_result["content"])
            )
            comparison["unstructured_elements"] = len(unstructured_result.get("elements", []))
            comparison["custom_code_snippets"] = len(custom_result["metadata"].get("code_snippets", []))
            comparison["unstructured_code_snippets"] = len(unstructured_result["metadata"].get("code_snippets", []))
        
        results["comparison"].append(comparison)
    
    return results