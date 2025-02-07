"""Manager for discovering and running Milvus seeders."""
import os
import glob
import importlib.util
import ast
from typing import Callable, List, Dict, Any, Union, Type
from collections import defaultdict

from .milvus_seeder import _SEEDERS, milvus_seeder
from .session import MilvusSession
from ..orm.milvus_model import MilvusModel

class SeedManager:
    """Manages seeding of Milvus collections."""
    
    def __init__(self, cwd: str = None):
        """Initialize the seed manager.
        
        Args:
            cwd: Working directory to scan for seed files. Defaults to current directory.
        """
        self.cwd = cwd or os.getcwd()
        
    def _has_seeder_decorator(self, file_path: str) -> bool:
        """Check if a file contains any functions with the milvus_seeder decorator.
        
        Args:
            file_path: Path to the Python file to check
            
        Returns:
            True if the file contains a seeder, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read(), filename=file_path)
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name) and decorator.id == 'milvus_seeder':
                            print(f"Found seeder function {node.name} in {file_path}")
                            return True
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name) and decorator.func.id == 'milvus_seeder':
                                print(f"Found seeder function {node.name} in {file_path}")
                                return True
            return False
        except Exception as e:
            print(f"Error checking for seeder in {file_path}: {e}")
            return False
        
    def discover_seeders(self) -> List[Callable]:
        """Find all seeder functions in the codebase.
        
        Returns:
            List of discovered seeder functions
        """
        # Clear existing seeders before discovery
        _SEEDERS.clear()
        print(f"\nScanning for seeders in {self.cwd}")
        
        # Add current working directory and src directory to Python path
        import sys
        if self.cwd not in sys.path:
            sys.path.insert(0, self.cwd)
            print(f"Added {self.cwd} to Python path")
            
        # Also add the parent directory to handle package imports
        parent_dir = os.path.dirname(self.cwd)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            print(f"Added {parent_dir} to Python path")
            
        # Find all Python files recursively
        python_files = list(glob.glob(os.path.join(self.cwd, "**/*.py"), recursive=True))
        print(f"\nFound {len(python_files)} Python files to scan")
        
        # Clear sys.modules of any previously imported seeder modules
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('seeder_') or 'seeders' in module_name.lower():
                del sys.modules[module_name]
        
        for file_path in python_files:
            if not os.path.isfile(file_path):
                continue
                
            # Skip files in venv directories
            if "venv" in file_path or "site-packages" in file_path:
                continue
                
            print(f"\nChecking {file_path}")
            # Only process files that have the milvus_seeder decorator
            if not self._has_seeder_decorator(file_path):
                continue
                
            try:
                # Import the module
                module_name = f"seeder_{os.path.splitext(os.path.basename(file_path))[0]}"
                print(f"Importing module {module_name} from {file_path}")
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module  # Register the module in sys.modules
                spec.loader.exec_module(module)
                
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                
        seeders = list(_SEEDERS.values())
        print(f"\nDiscovered {len(seeders)} seeders: {[s.__name__ for s in seeders]}")
        return seeders
        
    def run_seeders(self) -> Dict[str, Any]:
        """Run all discovered seeders.
        
        Returns:
            Dictionary mapping seeder names to their results
        """
        # First discover all seeder functions
        seeders = self.discover_seeders()
        if not seeders:
            print("No seeders found.")
            return {}
            
        results = {}
        session = MilvusSession()
        
        # Run each seeder function and collect entities by collection
        entities_by_collection = defaultdict(list)
        
        # First run all seeders and collect their entities
        for seeder in seeders:
            try:
                print(f"\nRunning seeder: {seeder.__name__}")
                seeded_entities = seeder()
                
                # Skip if seeder returned None
                if seeded_entities is None:
                    results[seeder.__name__] = {
                        "status": "success",
                        "count": 0
                    }
                    print(f"✅ {seeder.__name__} completed successfully (no entities)")
                    continue
                
                # Handle both single entities and lists
                if not isinstance(seeded_entities, list):
                    seeded_entities = [seeded_entities]
                
                # Group entities by their collection
                for entity in seeded_entities:
                    if not isinstance(entity, MilvusModel):
                        raise TypeError(f"Seeder {seeder.__name__} returned invalid entity type: {type(entity)}")
                    collection_name = entity.__class__.collection_name()
                    print(f"Adding entity to collection {collection_name}")
                    entities_by_collection[collection_name].append(entity)
                    
                results[seeder.__name__] = {
                    "status": "success",
                    "count": len(seeded_entities)
                }
                print(f"✅ {seeder.__name__} completed successfully")
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                results[seeder.__name__] = {
                    "status": "error",
                    "error": error_msg
                }
                print(f"❌ {seeder.__name__}: {error_msg}")
                continue
        
        print(f"\nProcessing entities for {len(entities_by_collection)} collections: {list(entities_by_collection.keys())}")
        
        # Now upsert all entities into their respective collections
        for collection_name, entities in entities_by_collection.items():
            try:
                print(f"\nProcessing collection {collection_name} with {len(entities)} entities")
                # Get the model class from the first entity
                model_class = entities[0].__class__
                
                # Check if collection exists
                if not session.collection_exists(model_class):
                    raise Exception(f"Collection {collection_name} does not exist; please create and run a migration first")
                
                # Get collection and prepare entities for upsert
                collection = session.get_milvus_collection(model_class)
                
                # Group entities by their IDs for upsert
                entity_dicts = [entity.to_dict() for entity in entities]
                ids = [entity["id"] for entity in entity_dicts]
                
                # Delete existing entities with these IDs
                if ids:
                    expr = 'id in ["' + '","'.join(ids) + '"]'
                    collection.delete(expr)
                
                # Insert all entities
                collection.insert(entity_dicts)
                    
                print(f"✅ Upserted {len(entities)} entities into {collection_name}")
                
            except Exception as e:
                error_msg = f"Error upserting into {collection_name}: {str(e)}"
                results[f"upsert_{collection_name}"] = {
                    "status": "error",
                    "error": error_msg
                }
                print(f"❌ {error_msg}")
                
        return results 