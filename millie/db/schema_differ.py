import logging
from typing import List, Dict, Set, Type, Tuple
from millie.orm import MilvusModel


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchemaDiffer:
    """Compares schema versions to generate migrations."""
    
    @staticmethod
    def find_schema_changes(old_schema: Dict, new_schema: Dict) -> Dict[str, List]:
        """Compare two schemas and return differences."""
        old_fields = {f["name"]: f for f in old_schema["fields"]}
        new_fields = {f["name"]: f for f in new_schema["fields"]}
        
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
        old_max_length = old_field.get("max_length")
        new_max_length = new_field.get("max_length")
        old_dim = old_field.get("dim")
        new_dim = new_field.get("dim")
        
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
        if old_field["dtype"] == "VARCHAR":
            old_max_length = old_max_length if old_max_length and old_max_length > 0 else 100
        else:
            old_max_length = None
            
        if new_field["dtype"] == "VARCHAR":
            new_max_length = new_max_length if new_max_length and new_max_length > 0 else 100
        else:
            new_max_length = None
            
        # For vector fields, ensure dim is a positive integer
        vector_types = ["FLOAT_VECTOR", "BINARY_VECTOR"]
        if old_field["dtype"] in vector_types:
            old_dim = old_dim if old_dim and old_dim > 0 else 1536
        else:
            old_dim = None
            
        if new_field["dtype"] in vector_types:
            new_dim = new_dim if new_dim and new_dim > 0 else 1536
        else:
            new_dim = None
        
        # Compare values
        max_length_changed = old_max_length != new_max_length
        dim_changed = old_dim != new_dim
        
        return (
            old_field["dtype"] != new_field["dtype"] or
            max_length_changed or
            dim_changed
        )
    
    @staticmethod
    def generate_migration_code(model_class: Type[MilvusModel], changes: Dict[str, List]) -> Tuple[str, str]:
        """Generate upgrade and downgrade code for schema changes."""
        collection_name = model_class.collection_name()
        
        upgrade_lines = [f'    collection = Collection("{collection_name}")']
        downgrade_lines = [f'    collection = Collection("{collection_name}")']
        
        # Handle added fields
        if changes["added"]:
            add_fields = []
            for f in changes["added"]:
                field_str = f'            FieldSchema("{f["name"]}", DataType.{f["dtype"]}'
                if f.get("max_length") is not None and f.get("max_length") != -1:
                    field_str += f", max_length={f["max_length"]}"
                if f.get("dim") is not None and f.get("dim") != -1:
                    field_str += f", dim={f["dim"]}"
                if hasattr(f, "is_primary") and f.get("is_primary"):
                    field_str += ", is_primary=True"
                field_str += ")"
                add_fields.append(field_str)
            
            upgrade_lines.append("    collection.alter(add_fields=[")
            upgrade_lines.extend(add_fields)
            upgrade_lines.append("    ])")
            
            # Downgrade removes added fields
            downgrade_lines.append(
                f'    collection.alter(drop_fields=[{", ".join(repr(f["name"]) for f in changes["added"])}])'
            )
        
        # Handle removed fields
        if changes["removed"]:
            upgrade_lines.append(
                f'    collection.alter(drop_fields=[{", ".join(repr(f["name"]) for f in changes["removed"])}])'
            )
            
            # Downgrade adds back removed fields
            add_fields = [
                f'            FieldSchema("{f["name"]}", DataType.{f["dtype"]}'
                f'{f", max_length={f["max_length"]}" if f.get("max_length") else ""}'
                f'{f", dim={f["dim"]}" if f.get("dim") else ""})'
                for f in changes["removed"]
            ]
            downgrade_lines.append("    collection.alter(add_fields=[")
            downgrade_lines.extend(add_fields)
            downgrade_lines.append("    ])")
        
        # Handle modified fields
        if changes["modified"]:
            for old_field, new_field in changes["modified"]:
                upgrade_lines.extend([
                    f'    # Modify {old_field["name"]}',
                    f'    collection.alter(drop_fields=["{old_field["name"]}"])',
                    "    collection.alter(add_fields=[",
                    f'        FieldSchema("{new_field["name"]}", DataType.{new_field["dtype"]}'
                    f'{f", max_length={new_field["max_length"]}" if new_field.get("max_length") else ""}'
                    f'{f", dim={new_field["dim"]}" if new_field.get("dim") else ""})',
                    "    ])"
                ])
                
                downgrade_lines.extend([
                    f'    # Restore original {old_field["name"]}',
                    f'    collection.alter(drop_fields=["{new_field["name"]}"])',
                    "    collection.alter(add_fields=[",
                    f'        FieldSchema("{old_field["name"]}", DataType.{old_field["dtype"]}'
                    f'{f", max_length={old_field["max_length"]}" if old_field.get("max_length") else ""}'
                    f'{f", dim={old_field["dim"]}" if old_field.get("dim") else ""})',
                    "    ])"
                ])
        
        return "\n".join(upgrade_lines), "\n".join(downgrade_lines)
