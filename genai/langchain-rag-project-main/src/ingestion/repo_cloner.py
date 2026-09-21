"""Repository cloning and documentation extraction utilities."""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import shutil

from src.config import settings

logger = logging.getLogger(__name__)


class LangChainRepoManager:
    """Manages cloning and updating the LangChain repository."""
    
    def __init__(self, local_path: str = None, repo_url: str = None):
        self.repo_url = repo_url or settings.langchain_repo_url
        self.local_path = Path(local_path or settings.langchain_local_path)
        
    def clone_repository(self, force_refresh: bool = False) -> bool:
        """Clone the LangChain repository."""
        try:
            # Check if directory exists
            if self.local_path.exists():
                if force_refresh:
                    logger.info(f"Removing existing repository at {self.local_path}")
                    shutil.rmtree(self.local_path)
                else:
                    logger.info(f"Repository already exists at {self.local_path}")
                    return self._update_repository()
            
            # Create parent directory if it doesn't exist
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clone repository
            logger.info(f"Cloning LangChain repository from {self.repo_url}")
            result = subprocess.run([
                "git", "clone", "--depth", "1", self.repo_url, str(self.local_path)
            ], capture_output=True, text=True, check=True)
            
            logger.info(f"Successfully cloned repository to {self.local_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error cloning repository: {str(e)}")
            return False
    
    def _update_repository(self) -> bool:
        """Update existing repository."""
        try:
            if not (self.local_path / ".git").exists():
                logger.warning("Not a git repository, will re-clone")
                return self.clone_repository(force_refresh=True)
                
            logger.info("Updating existing repository")
            result = subprocess.run([
                "git", "-C", str(self.local_path), "pull", "origin", "main"
            ], capture_output=True, text=True, check=True)
            
            logger.info("Repository updated successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to update repository: {e.stderr}")
            logger.info("Will re-clone repository")
            return self.clone_repository(force_refresh=True)
    
    def get_documentation_paths(self) -> Dict[str, List[Path]]:
        """Get paths to different types of documentation files."""
        if not self.local_path.exists():
            raise FileNotFoundError(f"Repository not found at {self.local_path}")
        
        doc_paths = {
            "markdown": [],
            "notebooks": [],
            "python_files": [],
            "rst_files": [],
            "readme_files": []
        }
        
        # Define documentation directories to search
        doc_directories = [
            "docs",
            "cookbook", 
            "templates",
            "libs/langchain/langchain",
            "libs/community/langchain_community",
            "libs/experimental/langchain_experimental",
            "libs/partners",
        ]
        
        for doc_dir in doc_directories:
            doc_path = self.local_path / doc_dir
            if doc_path.exists():
                # Find markdown files
                doc_paths["markdown"].extend(doc_path.rglob("*.md"))
                doc_paths["markdown"].extend(doc_path.rglob("*.mdx"))
                
                # Find notebooks
                doc_paths["notebooks"].extend(doc_path.rglob("*.ipynb"))
                
                # Find Python files with docstrings
                doc_paths["python_files"].extend(doc_path.rglob("*.py"))
                
                # Find RST files
                doc_paths["rst_files"].extend(doc_path.rglob("*.rst"))
        
        # Find README files across the entire repository
        doc_paths["readme_files"].extend(self.local_path.rglob("README*"))
        doc_paths["readme_files"].extend(self.local_path.rglob("readme*"))
        
        # Filter out unwanted directories
        excluded_dirs = {".git", "__pycache__", "node_modules", ".pytest_cache", "build", "dist"}
        
        for file_type in doc_paths:
            doc_paths[file_type] = [
                path for path in doc_paths[file_type]
                if not any(excluded in path.parts for excluded in excluded_dirs)
            ]
        
        return doc_paths
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get statistics about the cloned repository."""
        if not self.local_path.exists():
            return {"error": "Repository not found"}
        
        doc_paths = self.get_documentation_paths()
        
        stats = {
            "repository_path": str(self.local_path),
            "last_updated": datetime.now().isoformat(),
            "file_counts": {
                file_type: len(paths) 
                for file_type, paths in doc_paths.items()
            },
            "total_files": sum(len(paths) for paths in doc_paths.values())
        }
        
        # Get git information if available
        try:
            # Get last commit hash
            result = subprocess.run([
                "git", "-C", str(self.local_path), "rev-parse", "HEAD"
            ], capture_output=True, text=True, check=True)
            stats["commit_hash"] = result.stdout.strip()
            
            # Get last commit date
            result = subprocess.run([
                "git", "-C", str(self.local_path), "log", "-1", "--format=%cd", "--date=iso"
            ], capture_output=True, text=True, check=True)
            stats["last_commit_date"] = result.stdout.strip()
            
        except subprocess.CalledProcessError:
            logger.warning("Could not retrieve git information")
        
        return stats


def clone_langchain_docs(force_refresh: bool = False) -> Dict[str, Any]:
    """Convenience function to clone LangChain documentation."""
    manager = LangChainRepoManager()
    
    success = manager.clone_repository(force_refresh=force_refresh)
    if not success:
        return {"success": False, "error": "Failed to clone repository"}
    
    stats = manager.get_repository_stats()
    doc_paths = manager.get_documentation_paths()
    
    return {
        "success": True,
        "stats": stats,
        "documentation_paths": {
            file_type: [str(path) for path in paths]
            for file_type, paths in doc_paths.items()
        }
    }