import os
from pathlib import Path
from datetime import datetime
import json
from typing import List, Type, Callable, Dict, Any, Union
import glob
import importlib.util
import inspect
import ast
from collections import defaultdict

from millie.db.migration_builder import MigrationBuilder
from millie.db.schema_differ import SchemaDiffer
from millie.db.schema_history import SchemaHistory
from millie.db.schema import Schema, SchemaField
from millie.orm.milvus_model import MilvusModel, MilvusModelMeta
from millie.db.migration import Migration
from .session import MilvusSession

class MigrationManager:
    """Manages migrations for Milvus collections."""
    
    def __init__(self, cwd: str = None, schema_dir: str = os.getenv('MILLIE_SCHEMA_DIR', 'schema')):
        """Initialize the migration manager.
        
        Args:
            cwd: Working directory to scan for migration files. Defaults to current directory.
            schema_dir: Directory to store schema history files. Defaults to 'schema' under cwd.
        """
        self.cwd = cwd or os.getcwd()
        self.schema_dir = schema_dir or os.path.join(self.cwd, 'schema')
        self.migrations_dir = os.path.join(self.schema_dir, 'migrations')
        self.history_dir = os.path.join(self.schema_dir, 'history')
        
        # Create required directories if they don't exist
        os.makedirs(self.schema_dir, exist_ok=True)
        os.makedirs(self.migrations_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        
    def _find_all_models(self) -> List[Type[MilvusModel]]:
        """Find all model classes in the codebase.
        
        Returns:
            List of discovered model classes
        """
        # Add current working directory to Python path
        import sys
        if self.cwd not in sys.path:
            sys.path.insert(0, self.cwd)
            
        # Clear existing models before discovery
        MilvusModelMeta._models.clear()
            
        # Find all Python files in the specified directory
        model_glob = os.getenv('MILLIE_MODEL_GLOB')
        files = []
        if model_glob:
            # Use the model_glob as a glob pattern
            files = glob.glob(os.path.join(self.cwd, model_glob))
        else:
            files = glob.glob(os.path.join(self.cwd, "**/*.py"), recursive=True)
            
        for file_path in files:
            if not os.path.isfile(file_path):
                continue
                
            # Skip files in venv directories and test files
            if "venv" in file_path or "site-packages" in file_path:
                continue
                
            try:
                # Import the module
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {str(e)}")
                
        # Return all registered models
        return MilvusModelMeta.get_all_models()
        
    def _get_model_by_name(self, model_name: str) -> Type[MilvusModel]:
        """Get model class by name."""
        return MilvusModelMeta.get_model(model_name)
    
    def detect_changes(self, save_schema: bool = False):
        """Detect schema changes for all @MilvusModel classes"""
        models = self._find_all_models()
        changes = {}
        for model in models:
            model_changes = self.detect_changes_for_model(model, save_schema)
            if model_changes["added"] or model_changes["removed"] or model_changes["modified"]:
                changes[model.__name__] = model_changes

        return changes

    def detect_changes_for_model(self, model_cls: Type[MilvusModel], save_schema: bool = False):
        """Detect schema changes."""
        history = SchemaHistory(self.history_dir, self.migrations_dir)
        differ = SchemaDiffer()
        
        # Get current and new schemas using model class
        current_schema = history.get_schema_from_history(model_cls)
        new_schema = Schema.from_model(model_cls)
        
        # If no current schema exists, treat all fields as added
        if current_schema is None:
            initial_schema = history.build_initial_schema(model_cls)
            added_fields = initial_schema.fields
            history.save_model_schema(initial_schema) if save_schema else None
            return {
                "initial": True,
                "added": added_fields,
                "removed": [],
                "modified": []
            }
        
        # Compare schemas to detect changes
        changes = differ.diff_schemas(current_schema, new_schema)
        
        # Only save the new schema if there are actual changes
        if save_schema and (changes["added"] or changes["removed"] or changes["modified"]):
            history.save_model_schema(new_schema)

            
        return changes
        
    def generate_migration(self, name: str) -> str:
        """Generate a migration file.
        
        Args:
            name: Name of the migration
            
        Returns:
            Path to the generated migration file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        migration_name = f"{timestamp}_{name}"
        migration_path = os.path.join(self.migrations_dir, f"{migration_name}.py")
        
        # Get changes for all models
        migration_changes = self.detect_changes(save_schema=True)
        
        # Generate migration content for each model with changes
        builder = MigrationBuilder()
        up_codes = []
        down_codes = []
        
        for model_name, model_changes in migration_changes.items():
            model_cls = self._get_model_by_name(model_name)
            
            if model_cls:
                up_code, down_code = builder.generate_migration_code(model_cls, model_changes)
                up_codes.append(up_code)
                down_codes.append(down_code)
        
        # Combine all migration code
        combined_up = "\n\n".join(up_codes) if up_codes else "        pass"
        combined_down = "\n\n".join(down_codes) if down_codes else "        pass"
        
        # Generate the final migration content
        content = builder.generate_migration_file_content(name, migration_name, combined_up, combined_down)
        
        # Write migration file
        with open(migration_path, 'w') as f:
            f.write(content)
            
        return migration_path
    
    def create_empty_migration_file(self, name: str) -> str:
        """Generate a new migration file."""
        builder = MigrationBuilder(self.migrations_dir)
        return builder.generate_migration(name)
    
    def run_migrations(self) -> List[str]:
        """Run all pending migrations."""
        applied = []
        migrations = self._get_pending_migrations()
        
        for migration in migrations:
            try:
                self._apply_migration(migration)
                applied.append(migration)
            except Exception as e:
                raise Exception(f"Failed to apply migration {migration}: {str(e)}")
        
        return applied
    
    def _get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        if not os.path.exists(self.migrations_dir):
            return []
        
        migrations = []
        for filename in sorted(os.listdir(self.migrations_dir)):
            if filename.endswith('.py') and not filename.startswith('__'):
                migrations.append(os.path.join(self.migrations_dir, filename))
        
        return migrations
    
    def _apply_migration(self, migration_file: str):
        """Apply a single migration."""
        # Import the migration module
        module_name = os.path.splitext(os.path.basename(migration_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, migration_file)
        if not spec or not spec.loader:
            raise Exception(f"Could not load migration {migration_file}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find the migration class (it should be the only class that inherits from Migration)
        migration_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Migration) and obj != Migration:
                migration_class = obj
                break
                
        if not migration_class:
            raise Exception(f"No migration class found in {migration_file}")
            
        # Create an instance and apply the migration
        migration = migration_class()
        migration.apply()

    
