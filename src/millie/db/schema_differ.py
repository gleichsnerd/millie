import logging
from typing import List, Dict, Set, Type, Tuple, Any
from millie.orm.milvus_model import MilvusModel
from millie.db.schema import Schema, SchemaField


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaDiffer:
    """Utility class to find differences between schemas."""

    def diff_schemas(self, old_schema: Schema | None, new_schema: Schema) -> Dict[str, List[SchemaField]]:
        """Compare two schemas and return the differences."""
        # Handle initial schema case
        if old_schema is None:
            # For initial schema, include all fields
            return {
                "initial": True,
                "added": new_schema.fields,
                "removed": [],
                "modified": []
            }
        
        old_fields = {f.name: f for f in old_schema.fields}
        new_fields = {f.name: f for f in new_schema.fields}
        
        added = []
        removed = []
        modified = []
        
        # Check for added fields
        for name, field in new_fields.items():
            if name not in old_fields:
                added.append(field)
        
        # Check for removed fields
        for name, field in old_fields.items():
            if name not in new_fields:
                removed.append(field)
        
        # Check for modified fields
        for name, old_field in old_fields.items():
            if name in new_fields:
                new_field = new_fields[name]
                if self._is_field_modified(old_field, new_field):
                    modified.append((old_field, new_field))
        
        # Log changes for debugging
        if added or removed or modified:
            logger.debug(f"Schema changes detected:")
            if added:
                logger.debug(f"Added fields: {[f.name for f in added]}")
            if removed:
                logger.debug(f"Removed fields: {[f.name for f in removed]}")
            if modified:
                logger.debug(f"Modified fields: {[(old.name, new.name) for old, new in modified]}")
        
        return {
            "added": added,
            "removed": removed,
            "modified": modified
        }

    def _is_field_modified(self, old_field: SchemaField, new_field: SchemaField) -> bool:
        """Check if a field has been modified."""
        return (
            old_field.dtype != new_field.dtype or
            old_field.max_length != new_field.max_length or
            old_field.dim != new_field.dim or
            old_field.is_primary != new_field.is_primary
        )

    @staticmethod
    def generate_migration_code(model_class: Type[MilvusModel], changes: Dict[str, List]) -> Tuple[str, str]:
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
