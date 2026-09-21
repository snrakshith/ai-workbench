"""Main data ingestion pipeline for LangChain documentation."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from .repo_cloner import LangChainRepoManager, clone_langchain_docs
from .document_parser import parse_langchain_documents
from .unstructured_parser import parse_langchain_documents_unstructured, compare_parsing_methods
from src.config import settings

logger = logging.getLogger(__name__)


def serialize_for_json(obj):
    """Custom serializer to handle non-JSON serializable objects."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (Path)):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


class DataIngestionPipeline:
    """Main pipeline for ingesting LangChain documentation."""
    
    def __init__(self, output_dir: str = None, use_unstructured: bool = True):
        self.output_dir = Path(output_dir or settings.processed_data_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_unstructured = use_unstructured
        
        self.repo_manager = LangChainRepoManager()
        self.pipeline_metadata = {
            "start_time": None,
            "end_time": None,
            "total_documents": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "document_types": {},
            "parser_used": "unstructured" if use_unstructured else "custom"
        }
    
    def run_full_pipeline(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Run the complete data ingestion pipeline."""
        self.pipeline_metadata["start_time"] = datetime.now().isoformat()
        
        try:
            # Step 1: Clone/update repository
            logger.info("🔄 Starting data ingestion pipeline...")
            clone_result = self._clone_repository(force_refresh)
            if not clone_result["success"]:
                return clone_result
            
            # Step 2: Parse documents
            logger.info("📄 Parsing documents...")
            parse_result = self._parse_documents()
            if not parse_result["success"]:
                return parse_result
            
            # Step 3: Save processed data
            logger.info("💾 Saving processed data...")
            save_result = self._save_processed_data(parse_result["documents"])
            
            # Step 4: Generate statistics
            logger.info("📊 Generating statistics...")
            stats = self._generate_statistics(parse_result["documents"])
            
            self.pipeline_metadata["end_time"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "pipeline_metadata": self.pipeline_metadata,
                "repository_stats": clone_result["stats"],
                "parsing_stats": parse_result["stats"],
                "document_stats": stats,
                "output_files": save_result["files"]
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.pipeline_metadata["end_time"] = datetime.now().isoformat()
            return {
                "success": False,
                "error": str(e),
                "pipeline_metadata": self.pipeline_metadata
            }
    
    def _clone_repository(self, force_refresh: bool) -> Dict[str, Any]:
        """Clone or update the LangChain repository."""
        try:
            result = clone_langchain_docs(force_refresh=force_refresh)
            if result["success"]:
                logger.info(f"✅ Repository ready: {result['stats']['total_files']} files found")
            return result
        except Exception as e:
            logger.error(f"Repository cloning failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _parse_documents(self) -> Dict[str, Any]:
        """Parse all documentation files."""
        try:
            # Get document paths
            doc_paths = self.repo_manager.get_documentation_paths()
            
            # Parse documents using selected parser
            if self.use_unstructured:
                logger.info("Using Unstructured.io parser for enhanced document processing...")
                parsed_docs = parse_langchain_documents_unstructured(
                    doc_paths, 
                    extract_tables=True, 
                    extract_images=False
                )
            else:
                logger.info("Using custom parser...")
                parsed_docs = parse_langchain_documents(doc_paths)
            
            # Calculate statistics
            total_docs = sum(len(docs) for docs in parsed_docs.values())
            total_attempted = sum(len(paths) for paths in doc_paths.values())
            
            stats = {
                "total_attempted": total_attempted,
                "total_successful": total_docs,
                "success_rate": total_docs / total_attempted if total_attempted > 0 else 0,
                "by_type": {
                    doc_type: len(docs) 
                    for doc_type, docs in parsed_docs.items()
                }
            }
            
            self.pipeline_metadata["total_documents"] = total_attempted
            self.pipeline_metadata["successful_parses"] = total_docs
            self.pipeline_metadata["failed_parses"] = total_attempted - total_docs
            self.pipeline_metadata["document_types"] = stats["by_type"]
            
            logger.info(f"✅ Parsed {total_docs}/{total_attempted} documents ({stats['success_rate']:.1%})")
            
            return {
                "success": True,
                "documents": parsed_docs,
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Document parsing failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _save_processed_data(self, documents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Save processed documents to files."""
        saved_files = []
        
        try:
            # Save each document type to separate JSON files
            for doc_type, docs in documents.items():
                if not docs:
                    continue
                    
                output_file = self.output_dir / f"{doc_type}_documents.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(docs, f, indent=2, ensure_ascii=False, default=serialize_for_json)
                
                saved_files.append(str(output_file))
                logger.info(f"💾 Saved {len(docs)} {doc_type} documents to {output_file.name}")
            
            # Save combined metadata
            all_metadata = []
            for doc_type, docs in documents.items():
                for doc in docs:
                    metadata = doc["metadata"].copy()
                    metadata["content_length"] = len(doc["content"])
                    all_metadata.append(metadata)
            
            if all_metadata:
                metadata_file = self.output_dir / "document_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(all_metadata, f, indent=2, ensure_ascii=False, default=serialize_for_json)
                saved_files.append(str(metadata_file))
                
                # Also save as CSV for easy analysis
                df = pd.DataFrame(all_metadata)
                csv_file = self.output_dir / "document_metadata.csv"
                df.to_csv(csv_file, index=False)
                saved_files.append(str(csv_file))
                
                logger.info(f"💾 Saved metadata for {len(all_metadata)} documents")
            
            # Save pipeline metadata
            pipeline_file = self.output_dir / "pipeline_metadata.json"
            with open(pipeline_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_metadata, f, indent=2, ensure_ascii=False, default=serialize_for_json)
            saved_files.append(str(pipeline_file))
            
            return {"success": True, "files": saved_files}
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_statistics(self, documents: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate detailed statistics about the processed documents."""
        stats = {
            "total_documents": 0,
            "total_content_length": 0,
            "by_type": {},
            "by_doc_type": {},
            "content_length_stats": {},
            "top_modules": [],
            "file_size_distribution": {}
        }
        
        all_docs = []
        for doc_type, docs in documents.items():
            all_docs.extend(docs)
            
            type_stats = {
                "count": len(docs),
                "avg_content_length": 0,
                "total_content_length": 0
            }
            
            if docs:
                content_lengths = [len(doc["content"]) for doc in docs]
                type_stats["avg_content_length"] = sum(content_lengths) / len(content_lengths)
                type_stats["total_content_length"] = sum(content_lengths)
            
            stats["by_type"][doc_type] = type_stats
        
        if all_docs:
            stats["total_documents"] = len(all_docs)
            stats["total_content_length"] = sum(len(doc["content"]) for doc in all_docs)
            
            # Statistics by document type (api, tutorial, etc.)
            doc_types = {}
            for doc in all_docs:
                doc_type = doc["metadata"]["doc_type"]
                if doc_type not in doc_types:
                    doc_types[doc_type] = []
                doc_types[doc_type].append(len(doc["content"]))
            
            stats["by_doc_type"] = {
                dt: {
                    "count": len(lengths),
                    "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                    "total_length": sum(lengths)
                }
                for dt, lengths in doc_types.items()
            }
            
            # Content length distribution
            content_lengths = [len(doc["content"]) for doc in all_docs]
            content_lengths.sort()
            
            stats["content_length_stats"] = {
                "min": min(content_lengths),
                "max": max(content_lengths),
                "median": content_lengths[len(content_lengths) // 2],
                "avg": sum(content_lengths) / len(content_lengths),
                "percentile_95": content_lengths[int(len(content_lengths) * 0.95)]
            }
            
            # Top modules by content
            module_stats = {}
            for doc in all_docs:
                module = doc["metadata"]["module_path"]
                if module not in module_stats:
                    module_stats[module] = {"count": 0, "total_length": 0}
                module_stats[module]["count"] += 1
                module_stats[module]["total_length"] += len(doc["content"])
            
            stats["top_modules"] = sorted(
                [
                    {"module": module, **stat_data}
                    for module, stat_data in module_stats.items()
                ],
                key=lambda x: x["total_length"],
                reverse=True
            )[:20]
        
        return stats
    
    def run_parser_comparison(self, sample_size: int = 10) -> Dict[str, Any]:
        """Compare custom parser vs Unstructured.io parser on a sample of documents."""
        try:
            # Get sample of document paths
            doc_paths = self.repo_manager.get_documentation_paths()
            all_paths = []
            for paths in doc_paths.values():
                all_paths.extend(paths)
            
            logger.info(f"Running parser comparison on {min(sample_size, len(all_paths))} documents...")
            comparison_results = compare_parsing_methods(all_paths, sample_size)
            
            # Save comparison results
            comparison_file = self.output_dir / "parser_comparison.json"
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Parser comparison saved to {comparison_file}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Parser comparison failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def load_processed_documents(self, doc_type: str = None) -> List[Dict[str, Any]]:
        """Load previously processed documents."""
        if doc_type:
            file_path = self.output_dir / f"{doc_type}_documents.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        else:
            # Load all documents
            all_docs = []
            for file_path in self.output_dir.glob("*_documents.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_docs.extend(json.load(f))
            return all_docs


def run_ingestion_pipeline(force_refresh: bool = False, output_dir: str = None, 
                          use_unstructured: bool = True) -> Dict[str, Any]:
    """Convenience function to run the full ingestion pipeline."""
    pipeline = DataIngestionPipeline(output_dir=output_dir, use_unstructured=use_unstructured)
    return pipeline.run_full_pipeline(force_refresh=force_refresh)


def run_parser_comparison_pipeline(sample_size: int = 10, output_dir: str = None) -> Dict[str, Any]:
    """Convenience function to run parser comparison."""
    pipeline = DataIngestionPipeline(output_dir=output_dir)
    return pipeline.run_parser_comparison(sample_size=sample_size)