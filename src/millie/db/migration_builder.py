import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Type, Any
from millie.orm.milvus_model import MilvusModel
from millie.db.schema import SchemaField

class MigrationBuilder:
    """Handles the creation of migration files."""
    
    def __init__(self, migrations_dir: str = None):
        schema_dir = os.getenv('MILLIE_SCHEMA_DIR', str(Path(__file__).parent / 'schema'))
        self.migrations_dir = migrations_dir or os.path.join(schema_dir, 'migrations')
        os.makedirs(self.migrations_dir, exist_ok=True)
    
    def generate_migration(self, name: str) -> str:
        """Generate a new migration file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{name}.py"
        filepath = os.path.join(self.migrations_dir, filename)
        
        migration_template = self.generate_migration_file_content(name, timestamp, "        pass", "        pass")
        
        with open(filepath, 'w') as f:
            f.write(migration_template)
        
        return filepath 

    @staticmethod
    def generate_migration_code(model_class: Type[MilvusModel], changes: Dict[str, List]) -> Tuple[str, str]:
        """Generate upgrade and downgrade code for schema changes."""
        collection_name = model_class.collection_name()
        
        # For initial migration, create collection with all fields
        if not changes["modified"] and not changes["removed"] and changes["added"]:
            # Build field schemas
            field_schemas = []
            for field in changes["added"]:
                field_schema = f'FieldSchema(name="{field.name}", dtype=DataType.{field.dtype}'
                if field.max_length:
                    field_schema += f', max_length={field.max_length}'
                if field.dim:
                    field_schema += f', dim={field.dim}'
                if field.is_primary:
                    field_schema += ', is_primary=True'
                field_schema += ')'
                field_schemas.append(field_schema)
            
            # Create collection with all fields
            fields_str = ',\n            '.join(field_schemas)
            upgrade_lines = [
                f'        fields = [\n            {fields_str}\n        ]',
                f'        collection = self.ensure_collection("{collection_name}", fields)'
            ]
            # Drop collection in down()
            downgrade_lines = [
                f'        collection = Collection("{collection_name}")',
                '        collection.drop()'
            ]
        else:
            # For subsequent migrations, use alter_schema
            upgrade_lines = [f'        collection = Collection("{collection_name}")']
            downgrade_lines = [f'        collection = Collection("{collection_name}")']
            
            # Handle added fields
            for field in changes["added"]:
                field_schema = f'FieldSchema(name="{field.name}", dtype=DataType.{field.dtype}'
                if field.max_length:
                    field_schema += f', max_length={field.max_length}'
                if field.dim:
                    field_schema += f', dim={field.dim}'
                if field.is_primary:
                    field_schema += ', is_primary=True'
                field_schema += ')'
                
                upgrade_lines.append(f'        collection.alter_schema(add_fields=[{field_schema}])')
                downgrade_lines.append(f'        collection.alter_schema(drop_fields=["{field.name}"])')
            
            # Handle removed fields
            for field in changes["removed"]:
                field_schema = f'FieldSchema(name="{field.name}", dtype=DataType.{field.dtype}'
                if field.max_length:
                    field_schema += f', max_length={field.max_length}'
                if field.dim:
                    field_schema += f', dim={field.dim}'
                if field.is_primary:
                    field_schema += ', is_primary=True'
                field_schema += ')'
                
                upgrade_lines.append(f'        collection.alter_schema(drop_fields=["{field.name}"])')
                downgrade_lines.append(f'        collection.alter_schema(add_fields=[{field_schema}])')
            
            # Handle modified fields
            for old_field, new_field in changes["modified"]:
                # For modified fields, we need to drop and recreate since Milvus doesn't support direct modification
                old_schema = f'FieldSchema(name="{old_field.name}", dtype=DataType.{old_field.dtype}'
                if old_field.max_length:
                    old_schema += f', max_length={old_field.max_length}'
                if old_field.dim:
                    old_schema += f', dim={old_field.dim}'
                if old_field.is_primary:
                    old_schema += ', is_primary=True'
                old_schema += ')'
                
                new_schema = f'FieldSchema(name="{new_field.name}", dtype=DataType.{new_field.dtype}'
                if new_field.max_length:
                    new_schema += f', max_length={new_field.max_length}'
                if new_field.dim:
                    new_schema += f', dim={new_field.dim}'
                if new_field.is_primary:
                    new_schema += ', is_primary=True'
                new_schema += ')'
                
                upgrade_lines.extend([
                    f'        collection.alter_schema(drop_fields=["{old_field.name}"])',
                    f'        collection.alter_schema(add_fields=[{new_schema}])'
                ])
                downgrade_lines.extend([
                    f'        collection.alter_schema(drop_fields=["{new_field.name}"])',
                    f'        collection.alter_schema(add_fields=[{old_schema}])'
                ])
        
        return (
            '\n'.join(upgrade_lines),
            '\n'.join(downgrade_lines)
        )
    
    @staticmethod
    def generate_migration_file_content(name: str, migration_name: str, up_code: str, down_code: str) -> str:
        """Generate the content of a migration file."""
        return f'''"""
{name}

Revision ID: {migration_name}
Created at: {datetime.now().isoformat()}
"""
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema
from millie.db.migration import Migration

class Migration_{migration_name}(Migration):
    """Migration for {name}."""

    def up(self):
        """Upgrade to this version."""
{up_code}

    def down(self):
        """Downgrade from this version."""
{down_code}
'''

    def build_migration(self, name: str, model_cls: Type[MilvusModel] = None, changes: Dict[str, List] = None) -> str:
        """Build migration file content.
        
        Args:
            name: Name of the migration
            model_cls: Optional model class to generate migration for
            changes: Optional dict of schema changes
            
        Returns:
            String content of the migration file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if model_cls and changes:
            # For initial migration, create collection with all fields
            if not changes["modified"] and not changes["removed"] and changes["added"]:
                # Build field schemas
                field_schemas = []
                for field in changes["added"]:
                    field_schema = f'FieldSchema(name="{field.name}", dtype=DataType.{field.dtype}'
                    if field.max_length:
                        field_schema += f', max_length={field.max_length}'
                    if field.dim:
                        field_schema += f', dim={field.dim}'
                    if field.is_primary:
                        field_schema += ', is_primary=True'
                    field_schema += ')'
                    field_schemas.append(field_schema)
                
                # Create collection with all fields
                fields_str = ',\n            '.join(field_schemas)
                up_code = f'''        fields = [
            {fields_str}
        ]
        collection = Collection(name="{model_cls.collection_name()}", schema=CollectionSchema(fields=fields))'''
                down_code = f'''        collection = Collection("{model_cls.collection_name()}")
        collection.drop()'''
            else:
                # For subsequent migrations, use alter_schema
                up_code, down_code = self.generate_migration_code(model_cls, changes)
            
            content = self.generate_migration_file_content(name, timestamp, up_code, down_code)
        else:
            content = self.generate_migration_file_content(name, timestamp, "        pass", "        pass")
        
        return content
        
    def _field_to_schema_str(self, field: SchemaField) -> str:
        """Convert a field to its FieldSchema string representation."""
        args = [f'name="{field.name}"', f'dtype=DataType.{field.dtype}']
        
        if field.max_length is not None:
            args.append(f'max_length={field.max_length}')
        if field.dim is not None:
            args.append(f'dim={field.dim}')
        if field.is_primary:
            args.append('is_primary=True')
            
        return f'FieldSchema({", ".join(args)})' 