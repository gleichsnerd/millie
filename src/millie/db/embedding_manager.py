"""Manager for discovering and running Milvus embedders."""
import hashlib
import json
import os
import glob
import importlib.util
import ast
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Set, Type, TypeVar
import uuid

from millie.orm.milvus_model import MilvusModel

from .milvus_embedder import _EMBEDDERS

class EmbeddingManager:
    """Manages embedding generation for Milvus collections."""
    
    def __init__(self, cwd: str = os.getcwd()):
        """Initialize the embedding manager.
        
        Args:
            cwd: Working directory to scan for embedder files. Defaults to current directory.
        """
        self.cwd = cwd
        
    def _has_embedder_decorator(self, file_path: str) -> bool:
        """Check if a file contains any functions with the milvus_embedder decorator.
        
        Args:
            file_path: Path to the Python file to check
            
        Returns:
            True if the file contains an embedder, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read(), filename=file_path)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name) and decorator.id == 'milvus_embedder':
                            return True
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name) and decorator.func.id == 'milvus_embedder':
                                return True
            return False
        except Exception:
            return False
        
    def discover_embedders(self) -> List[Callable]:
        """Find all embedder functions in the codebase.
        
        Returns:
            List of discovered embedder functions
        """
        # Clear existing embedders before discovery
        _EMBEDDERS.clear()
        
        # Add current working directory and src directory to Python path
        import sys
        if self.cwd not in sys.path:
            sys.path.insert(0, self.cwd)
            
        # Also add the parent directory to handle package imports
        parent_dir = os.path.dirname(self.cwd)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        # Find all Python files recursively
        for file_path in glob.glob(os.path.join(self.cwd, "**/*.py"), recursive=True):
            if not os.path.isfile(file_path):
                continue
                
            # Skip files in venv directories
            if "venv" in file_path or "site-packages" in file_path:
                continue
                
            # Only process files that have the milvus_embedder decorator
            if not self._has_embedder_decorator(file_path):
                continue
                
            try:
                # Import the module
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module  # Register the module in sys.modules
                spec.loader.exec_module(module)
                
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                
        return list(_EMBEDDERS.values())
        
    def run_embedders(self) -> Dict[str, Any]:
        """Run all discovered embedders.
        
        Returns:
            Dictionary mapping embedder names to their results
        """
        # First discover all embedder functions
        embedders = self.discover_embedders()
        if not embedders:
            print("No embedders found.")
            return {}
            
        results = {}
        
        # Run each embedder function
        for embedder in embedders:
            try:
                print(f"Running embedder: {embedder.__name__}")
                embedder()  # Just run the function, it handles its own DB operations
                results[embedder.__name__] = {
                    "status": "success"
                }
                print(f"✅ {embedder.__name__} completed successfully")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                results[embedder.__name__] = {
                    "status": "error",
                    "error": error_msg
                }
                print(f"❌ {embedder.__name__}: {error_msg}")
                
        return results 
    

    @staticmethod
    def get_value_hash(value: str) -> str:
        """Generate a hash for a value.
        
        Args:
            value: Text to hash
            
        Returns:
            SHA-256 hash of the description
        """
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def load_embeddings_file(file_path: Path) -> Dict[str, list]:
        """Load embeddings from a JSON file.
        
        Args:
            file_path: Path to embeddings file
            
        Returns:
            Dictionary mapping hashes to embeddings
        """
        path = Path(file_path)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
    
    @staticmethod
    def save_embeddings_file(file_path: Path, embeddings: Dict[str, list]) -> None:
        """Save embeddings to a JSON file.
        
        Args:
            file_path: Path to embeddings file
            embeddings: Dictionary mapping hashes to embeddings
        """
        with open(file_path, 'w') as f:
            json.dump(embeddings, f, indent=2)
    
    @staticmethod
    def create_model_from_data(model_cls: Type[MilvusModel], data: Dict[str, Any], field: str, source_file: Optional[Path] = None) -> MilvusModel:
        """Create a model instance from loaded data.
        
        Args:
            model_cls: The model class to create an instance of
            data: Dictionary containing model data
            source_file: Optional path to source file for loading embeddings
            
        Returns:
            An instance of the specified model class
        """
        # Ensure we have an ID
        data['id'] = data.get('id', str(uuid.uuid4()))
        
        # Try to load cached embedding if we have a source file
        value = data.get(field, '')
        if source_file and value:
            embeddings_file = source_file.with_suffix('.embeddings.json')
            embeddings = EmbeddingManager.load_embeddings_file(embeddings_file)
            hash_value = EmbeddingManager.get_value_hash(value)
            
            if hash_value in embeddings:
                data['embedding'] = embeddings[hash_value]
        
        # Set default embedding if not present
        if 'embedding' not in data:
            data['embedding'] = [0.1] * 1536  # Default 1536-dim embedding
        
        return model_cls(**data)
    
    @staticmethod
    def process_file(entities: List[Dict[str, Any]], model_class: Type[MilvusModel], source_file: Path, field: str, embedding_generator: Callable) -> None:
        """Process a single source file and update its embeddings.
        
        Args:
            source_file: Path to rule YAML file
            embedding_generator: Function to generate embeddings from a description
        """
        # Get embeddings file path
        embeddings_file = source_file.with_suffix('.embeddings.json')
        current_embeddings = EmbeddingManager.load_embeddings_file(embeddings_file)
        
        # Track which hashes we need
        needed_hashes: Set[str] = set()
        
        # Load rules from this file
        models = [EmbeddingManager.create_model_from_data(model_class, data, field, source_file) for data in entities]
        
        # Process each rule
        for model in models:
            value = getattr(model, field, None)
            if value:
                hash_value = EmbeddingManager.get_value_hash(value)
                needed_hashes.add(hash_value)
                
                if hash_value not in current_embeddings:
                    print(f"Generating embedding for model: {model}")
                    embedding = embedding_generator(value)
                    current_embeddings[hash_value] = embedding
        
        # Remove unused hashes
        unused_hashes = set(current_embeddings.keys()) - needed_hashes
        for hash_value in unused_hashes:
            del current_embeddings[hash_value]
        
        # Save updated embeddings
        EmbeddingManager.save_embeddings_file(embeddings_file, current_embeddings)
        print(f"Updated embeddings for {source_file.name}")