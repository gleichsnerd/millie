from dataclasses import dataclass, fields
from .base_model import BaseModel

def MilvusModel(is_migration_collection=False):
    """Decorator to convert a class into a MilvusModel dataclass."""
    
    def decorator(cls):
        # Create a new dataclass that inherits from BaseModel
        # Use kw_only=True to enforce keyword-only arguments
        new_cls = dataclass(kw_only=True)(cls)

        # Create a new class that inherits from both new_cls and BaseModel
        class CombinedModel(new_cls, BaseModel):
            pass

        # Copy over any additional properties or methods from BaseModel
        for field in fields(BaseModel):
            setattr(CombinedModel, field.name, field.default)

        # Optionally handle is_migration_collection if needed
        CombinedModel.is_migration_collection = is_migration_collection

        return CombinedModel
    
    return decorator