import os
import json
import glob
import logging
import importlib.util
import inspect
from pathlib import Path
import re
from typing import Dict, List, Literal, Type, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from pymilvus import FieldSchema, DataType

from millie.orm.milvus_model import MilvusModel
from millie.orm.milvus_model import MilvusModel
from millie.db.schema import Schema, SchemaField

load_dotenv()

class SchemaHistory:
    """Tracks schema history for Milvus collections."""
    
    def __init__(self, history_dir: str, migrations_dir: str):
        """Initialize schema history.
        
        Args:
            history_dir: Directory to store schema history files
            migrations_dir: Directory containing migration files
        """
        self.history_dir = history_dir
        self.migrations_dir = migrations_dir
        os.makedirs(history_dir, exist_ok=True)

    def get_model_schema_filename(self, model_cls: Type[MilvusModel]) -> str:
        """Get the path to a model's schema file."""
        model_name = model_cls.__name__
        # Strip off any decorator-added prefixes
        if model_name.startswith('Combined'):
            model_name = model_name[8:]  # Remove 'Combined' prefix
        return os.path.join(self.history_dir, f"{model_name}.json")

    def get_schema_from_history(self, model_cls: Type[MilvusModel]) -> Optional[Schema]:
        """Get the current schema for a model from history.
        
        Args:
            model_cls: Model class to get schema for
            
        Returns:
            Current schema or None if no history exists
        """
        # For initial schema, return None to trigger schema creation
        history_file = self.get_model_schema_filename(model_cls)
        if not os.path.exists(history_file):
            return None
            
        # Load schema from history file
        with open(history_file, 'r') as f:
            data = json.load(f)
            return Schema.from_dict(data)

    def save_model_schema(self, schema: Schema):
        """Save schema history for a specific model."""
        # Strip off any decorator-added prefixes
        schema_name = schema.name
        
        history_file = os.path.join(self.history_dir, f"{schema_name}.json")
        
        # Load existing data to preserve version if it exists
        current_version = 0
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                existing_data = json.load(f)
                current_version = existing_data.get("version", 0)
        
        # Create new data with version
        data = schema.to_dict()
        data["updated_at"] = datetime.now().isoformat()
        data["version"] = current_version + 1
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Update the schema object with the new version
        schema.version = data["version"]

    def build_model_schema_from_migrations(self, model_class: Type[MilvusModel]) -> Schema:
        """Build schema representation from migrations."""
        # Start with empty schema
        schema = Schema(
            name=model_class.__name__.replace('Combined', ''),
            collection_name=model_class.collection_name(),
            fields=[],
            is_migration_collection=getattr(model_class, 'is_migration_collection', False)
        )
        
        # Apply all migrations in order
        migrations = self.get_migrations()
        for migration_file in migrations:
            schema = self.apply_migration_to_schema(schema, migration_file)
        
        return schema
    
    def build_initial_schema(self, model_cls: Type[MilvusModel]) -> Schema:
        schema_def = model_cls.schema()
        if not schema_def or not hasattr(schema_def, 'fields'):
            return Schema(
                name=model_cls.__name__,
                collection_name=model_cls.collection_name(),
                fields=[],
                is_migration_collection=getattr(model_cls, 'is_migration_collection', False)
            )
        
        # For initial migration, include all fields
        added_fields = []
        for field_schema in schema_def.fields:
            added_fields.append(SchemaField.from_field_schema(field_schema))
        
        # Save the initial schema to history
        initial_schema = Schema(
            name=model_cls.__name__,
            collection_name=model_cls.collection_name(),
            fields=added_fields,
            is_migration_collection=getattr(model_cls, 'is_migration_collection', False)
        )

        return initial_schema

    def get_migrations(self) -> List[str]:
        """Get list of migration files in order."""
        if not os.path.exists(self.migrations_dir):
            return []
        
        migrations = []
        for filename in sorted(os.listdir(self.migrations_dir)):
            if filename.endswith('.py') and not filename.startswith('__'):
                migrations.append(os.path.join(self.migrations_dir, filename))
        
        return migrations

    def apply_migration_to_schema(self, schema: Schema, migration_file: str) -> Schema:
        """Apply a migration to a schema without executing it."""
        try:
            # Import migration module
            module_name = os.path.splitext(os.path.basename(migration_file))[0]
            spec = importlib.util.spec_from_file_location(module_name, migration_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find migration class
            migration_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr.__name__.startswith('Migration_'):
                    migration_class = attr
                    break
            
            if not migration_class:
                return schema
            
            # Get the up() method source code
            up_method = getattr(migration_class, 'up')
            source = inspect.getsource(up_method)
            
            # Parse the migration code to find schema changes
            lines = [line.strip() for line in re.split(r'[\(\),\[\]\n]', source) if line.strip()][1:]

            add_fields = []
            drop_fields = []
            current_field = []
            is_altering_schema = False
            is_building_field = False
            mode: Optional[Literal['add', 'drop']] = None

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                line = line.replace("'", '"')
                i += 1
                
                if 'alter_schema' in line:
                    is_altering_schema = True
                    # Reset state when starting a new alter_schema
                    if current_field and mode == 'add':
                        field_def = "FieldSchema(" + ", ".join(current_field) + ")"
                        add_fields.append(field_def)
                    current_field = []
                    is_building_field = False
                    mode = None
                elif not is_altering_schema:
                    continue
                # Handle add_fields
                elif 'add_fields=' in line:
                    mode = 'add'
                    current_field = []
                elif 'drop_fields=' in line:
                    mode = 'drop'
                    current_field = []
                elif 'FieldSchema' in line:
                    # Start collecting a new field
                    if current_field:  # If we have a field in progress, save it
                        if mode == 'add':
                            field_def = "FieldSchema(" + ", ".join(current_field) + ")"
                            add_fields.append(field_def)
                        elif mode == 'drop':
                            drop_fields.extend(current_field)
                    current_field = []
                    is_building_field = True
                elif is_building_field and mode == 'add' and self._is_field_schema_parameter(line):
                    current_field.append(line)
                elif mode == 'drop' and '"' in line:
                    # Extract field name from quoted string
                    field_name = line.strip().strip('"')
                    if field_name:
                        drop_fields.append(field_name)
            
            # Add the last field if we have one
            if current_field:
                if mode == 'add':
                    field_def = "FieldSchema(" + ", ".join(current_field) + ")"
                    add_fields.append(field_def)

            # Process drop fields first
            for field_name in drop_fields:
                schema.fields = [f for f in schema.fields if f.name != field_name]

            # Then process add fields
            for field_str in add_fields:
                field = self._parse_field_schema(field_str)
                if field:
                    schema.fields.append(field)
                else:
                    raise Exception(f"Failed to parse field schema: {field_str}")
                
            return schema
        except Exception as e:
            print(f"Error applying migration {migration_file}: {e}")
            return schema
        
    def _is_field_schema_parameter(self, line: str) -> bool:
        valid_parameters = ['name=', 'dtype=', 'max_length=', 'dim=', 'is_primary=', 'is_partition_key=', 'is_clustering_key=', 'default_value=', 'element_type=', 'mmap_enabled=']
        under_test = line.strip().split('=')[0] + '='
        return under_test in valid_parameters

    def _parse_field_schema(self, field_def: str) -> Optional[SchemaField]:
        """Parse a FieldSchema definition string into a SchemaField object."""
        try:
            # Extract field parameters
            params = {}
            
            # Get name
            name_start = field_def.index('name="') + 6
            name_end = field_def.index('"', name_start)
            params['name'] = field_def[name_start:name_end]
            
            # Get dtype
            dtype_start = field_def.index('DataType.') + 9
            dtype_end = field_def.find(',', dtype_start)
            if dtype_end == -1:  # No comma found, look for closing parenthesis
                dtype_end = field_def.find(')', dtype_start)
            params['dtype'] = field_def[dtype_start:dtype_end].strip()
            
            # Get max_length if present
            if 'max_length=' in field_def:
                max_len_start = field_def.index('max_length=') + 10
                max_len_end = field_def.find(',', max_len_start)
                if max_len_end == -1:  # No comma found, look for closing parenthesis
                    max_len_end = field_def.find(')', max_len_start)
                max_len_str = field_def[max_len_start:max_len_end].strip().split('=')[-1]  # Take value after any =
                params['max_length'] = int(max_len_str)
            
            # Get dim if present
            if 'dim=' in field_def:
                dim_start = field_def.index('dim=') + 4
                dim_end = field_def.find(',', dim_start)
                if dim_end == -1:  # No comma found, look for closing parenthesis
                    dim_end = field_def.find(')', dim_start)
                dim_str = field_def[dim_start:dim_end].strip().split('=')[-1]  # Take value after any =
                params['dim'] = int(dim_str)
            
            # Get is_primary if present
            params['is_primary'] = 'is_primary=True' in field_def
            
            return SchemaField(**params)
        except Exception as e:
            print(f"Error parsing field schema: {e}")
            return None

    def update_model_schema(self, model_class: Type[MilvusModel]) -> Schema:
        """Update schema for a single model."""
        current = self.get_schema_from_history(model_class)
        new_schema = self.build_model_schema_from_migrations(model_class)
        
        if current.to_dict() != new_schema.to_dict():
            self.save_model_schema(new_schema)
            
        return new_schema

    def schema_changed(self, model_class: Type[MilvusModel]) -> bool:
        """Check if a model's schema has changed from history."""
        current = self.get_schema_from_history(model_class)
        new_schema = self.build_model_schema_from_migrations(model_class)
        
        return current.to_dict() != new_schema.to_dict()

    def save_schema_to_history(self, schema: Schema):
        """Save schema to history.
        
        Args:
            schema: Schema to save
        """
        history_file = os.path.join(self.history_dir, f"{schema.collection_name}.json")
        with open(history_file, 'w') as f:
            json.dump(schema.to_dict(), f, indent=2)
