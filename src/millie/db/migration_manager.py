import os
from pathlib import Path
from datetime import datetime
import json
from typing import List, Type
import glob
import importlib.util
import inspect

from millie.db.migration_builder import MigrationBuilder
from millie.db.schema_differ import SchemaDiffer
from millie.db.schema_history import SchemaHistory
from millie.orm.base_model import BaseModel
from millie.schema.schema import Schema

class MigrationManager:
    """Manages database migrations."""
    
    def __init__(
        self, 
        schema_dir: str = os.getenv(
            'MILLIE_SCHEMA_DIR', 
            os.path.join(str(Path(__file__).parent),'schema')
        )
    ):
        """Initialize the migration manager."""
        self.schema_dir = schema_dir
        os.makedirs(self.schema_dir, exist_ok=True)
        self.migrations_dir = os.path.join(self.schema_dir, 'migrations')
        os.makedirs(self.migrations_dir, exist_ok=True)
        self.history_dir = os.path.join(self.schema_dir, 'history')
        os.makedirs(self.history_dir, exist_ok=True)
        
    def detect_changes(self):
        """Detect schema changes for all @MilvusModel classes"""
        models = self._find_all_models()
        changes = {}
        for model in models:
            model_changes = self.detect_changes_for_model(model)
            if model_changes["added"] or model_changes["removed"] or model_changes["modified"]:
                changes[model.__name__] = model_changes

        return changes

    def detect_changes_for_model(self, model_cls: Type[BaseModel]):
        """Detect schema changes."""
        history = SchemaHistory(self.history_dir, self.migrations_dir)
        differ = SchemaDiffer()
        # Get current and new schemas using model class
        current_schema = history.get_schema_from_history(model_cls)
        new_schema = Schema.from_model(model_cls)
        
        # Find changes using SchemaDiffer
        changes = differ.find_schema_changes(current_schema, new_schema)
        return changes
        
    def generate_migration(self, name: str, model_cls: Type[BaseModel] = None) -> str:
        """Generate a new migration file using schema change detection"""
        # Generate migration file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        migration_name = f"{timestamp}_{name}"
        
        if model_cls:
            # Single model mode
            changes = {model_cls.__name__: self.detect_changes_for_model(model_cls)}
        else:
            # Multi-model mode
            changes = self.detect_changes()
        
        if len(changes) > 0:
            # Create builder and generate migration code
            builder = MigrationBuilder()
            
            # Process changes for each model
            up_codes = []
            down_codes = []
            for model_name, model_changes in changes.items():
                if not model_changes["added"] and not model_changes["removed"] and not model_changes["modified"]:
                    continue
                    
                model = model_cls if model_cls and model_cls.__name__ == model_name else self._get_model_by_name(model_name)
                if model:
                    up_code, down_code = builder.generate_migration_code(model, model_changes)
                    up_codes.append(up_code)
                    down_codes.append(down_code)
            
            if not up_codes:  # No actual changes to migrate
                return None
            
            # Combine all changes into one migration
            migration_content = builder.generate_migration_file_content(
                name, 
                migration_name, 
                '\n'.join(up_codes), 
                '\n'.join(down_codes)
            )

            # Save migration file
            migration_path = os.path.join(self.migrations_dir, f"{migration_name}.py")
            with open(migration_path, 'w') as f:
                f.write(migration_content)

            # Update schema history for each model
            history = SchemaHistory(self.history_dir, self.migrations_dir)
            for model_name in changes.keys():
                model = model_cls if model_cls and model_cls.__name__ == model_name else self._get_model_by_name(model_name)
                if model:
                    new_schema = Schema.from_model(model)
                    history.save_model_schema(new_schema)
            
            return migration_path
        else:
            return None
    
    def create_empty_migration_file(self, name: str) -> str:
        """Generate a new migration file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{name}.json"
        filepath = os.path.join(self.migrations_dir, filename)
        
        migration_template = {
            "version": timestamp,
            "name": name,
            "up": [],
            "down": []
        }
        
        with open(filepath, 'w') as f:
            json.dump(migration_template, f, indent=2)
        
        return filepath
    
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
            if filename.endswith('.json'):
                migrations.append(os.path.join(self.migrations_dir, filename))
        
        return migrations
    
    def _apply_migration(self, migration_file: str):
        """Apply a single migration."""
        with open(migration_file, 'r') as f:
            migration = json.load(f)
        
        # Apply each operation in the migration
        for operation in migration['up']:
            self._apply_operation(operation)
        
    def _apply_operation(self, operation: dict):
        """Apply a single migration operation."""
        # TODO: Implement actual migration operations
        # This will depend on what kind of migrations we need to support
        pass 

    def _find_all_models(self):
        """Find all models decorated with @MilvusModel using MILLIE_MODEL_GLOB"""
        models = []
        model_paths = os.getenv('MILLIE_MODEL_GLOB', '**/*.py')
        
        # Convert relative paths to absolute
        if not os.path.isabs(model_paths):
            model_paths = os.path.join(os.getcwd(), model_paths)
            
        # Find all Python files matching the glob pattern
        for file_path in glob.glob(model_paths, recursive=True):
            if not os.path.isfile(file_path):
                continue
                
            try:
                # Import the module
                module_name = os.path.splitext(os.path.basename(file_path))[0]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find all classes in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj):
                        # Check if class inherits from BaseModel and isn't BaseModel itself
                        if issubclass(obj, BaseModel) and obj != BaseModel:
                            models.append(obj)
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
                continue
        
        return models
    
    def _get_model_by_name(self, model_name: str) -> Type[BaseModel]:
        """Get model class by name."""
        # First try the models we found
        models = self._find_all_models()
        for model in models:
            if model.__name__ == model_name:
                return model
        
        return None
    
