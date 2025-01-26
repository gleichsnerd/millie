import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Type
from dotenv import load_dotenv
from millie.orm.base_model import BaseModel

load_dotenv()

class MigrationBuilder:
    """Handles the creation of migration files."""
    
    def __init__(self, migrations_dir: str = None):
        schema_dir = os.getenv('MILLIE_SCHEMA_DIR', str(Path(__file__).parent / 'schema'))
        self.migrations_dir = migrations_dir or os.path.join(schema_dir, 'migrations')
        os.makedirs(self.migrations_dir, exist_ok=True)
    
    def generate_migration(self, name: str) -> str:
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

    @staticmethod
    def generate_migration_code(model_class: Type[BaseModel], changes: Dict[str, List]) -> Tuple[str, str]:
        """Generate upgrade and downgrade code for schema changes."""
        collection_name = model_class.collection_name()
        
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
from pymilvus import Collection, FieldSchema, DataType
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