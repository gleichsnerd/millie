import logging
from typing import List, Dict, Set, Type, Tuple, Any
from millie.orm.base_model import BaseModel
from millie.schema.schema import Schema


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaDiffer:
    """Compares schema versions to detect changes."""
    
    @staticmethod
    def find_schema_changes(old_schema: Schema, new_schema: Schema) -> Dict[str, List]:
        """Compare two schemas and return differences."""
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        changes = {
            "added": [],
            "removed": [],
            "modified": []
        }
        
        # Find added and modified fields
        for name, field in new_fields.items():
            if name not in old_fields:
                changes["added"].append(field)
            elif SchemaDiffer._field_changed(old_fields[name], field):
                changes["modified"].append((old_fields[name], field))
        
        # Find removed fields
        for name in old_fields:
            if name not in new_fields:
                changes["removed"].append(old_fields[name])
        
        return changes
    
    @staticmethod
    def _field_changed(old_field: Dict, new_field: Dict) -> bool:
        """Check if field definition has changed."""
        # Get field attributes
        old_max_length = old_field.max_length
        new_max_length = new_field.max_length
        old_dim = old_field.dim
        new_dim = new_field.dim
        
        # Convert string representations to numbers or None
        if isinstance(old_max_length, str):
            old_max_length = None if old_max_length in ["-1", "None"] else int(old_max_length)
        if isinstance(new_max_length, str):
            new_max_length = None if new_max_length in ["-1", "None"] else int(new_max_length)
        if isinstance(old_dim, str):
            old_dim = None if old_dim in ["-1", "None"] else int(old_dim)
        if isinstance(new_dim, str):
            new_dim = None if new_dim in ["-1", "None"] else int(new_dim)
        
        # For VARCHAR fields, ensure max_length is a positive integer
        if old_field.dtype == "VARCHAR":
            old_max_length = old_max_length if old_max_length and old_max_length > 0 else 100
        else:
            old_max_length = None
            
        if new_field.dtype == "VARCHAR":
            new_max_length = new_max_length if new_max_length and new_max_length > 0 else 100
        else:
            new_max_length = None
            
        # For vector fields, ensure dim is a positive integer
        vector_types = ["FLOAT_VECTOR", "BINARY_VECTOR"]
        if old_field.dtype in vector_types:
            old_dim = old_dim if old_dim and old_dim > 0 else 1536
        else:
            old_dim = None
            
        if new_field.dtype in vector_types:
            new_dim = new_dim if new_dim and new_dim > 0 else 1536
        else:
            new_dim = None
        
        # Compare values
        max_length_changed = old_max_length != new_max_length
        dim_changed = old_dim != new_dim
        
        return (
            old_field.dtype != new_field.dtype or
            max_length_changed or
            dim_changed
        )
    
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
