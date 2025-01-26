from dataclasses import dataclass, fields, field
import pytest
from typeguard import TypeCheckError
from millie.orm.milvus_model import MilvusModel
from millie.orm.base_model import BaseModel
from typing import Optional, List

@MilvusModel()
class SimpleModel:
    name: str
    age: int
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None
    
    def get_full_name(self) -> str:
        return f"{self.name} (age: {self.age})"

@MilvusModel(is_migration_collection=True)
class MigrationModel:
    name: str
    version: str

@MilvusModel()
class ChildModel(SimpleModel):
    extra_field: str

def test_model_inheritance():
    """Test that decorated models inherit from BaseModel."""
    assert issubclass(SimpleModel, BaseModel)

def test_dataclass_creation():
    """Test that decorated models are converted to dataclasses."""
    # Should not raise TypeError
    model = SimpleModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25
    )
    assert model.name == "test"
    assert model.age == 25

def test_base_model_fields():
    """Test that BaseModel fields are preserved."""
    model_fields = {f.name for f in fields(SimpleModel)}
    assert "id" in model_fields
    assert "embedding" in model_fields
    assert "metadata" in model_fields

def test_original_fields():
    """Test that original class fields are preserved."""
    model_fields = {f.name for f in fields(SimpleModel)}
    assert "name" in model_fields
    assert "age" in model_fields

def test_migration_flag():
    """Test that is_migration_collection flag is set correctly."""
    assert not hasattr(SimpleModel, "is_migration_collection") or not SimpleModel.is_migration_collection
    assert hasattr(MigrationModel, "is_migration_collection")
    assert MigrationModel.is_migration_collection

def test_class_name_preservation():
    """Test that original class name is preserved."""
    assert SimpleModel.__name__ == "SimpleModel"
    assert MigrationModel.__name__ == "MigrationModel"

def test_model_functionality():
    """Test that decorated models retain BaseModel functionality."""
    model = SimpleModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25,
        metadata={"key": "value"}
    )
    
    # Test collection name
    assert SimpleModel.collection_name() == "simple"
    
    # Test to_dict
    data = model.to_dict()
    assert data["id"] == "123"
    assert data["name"] == "test"
    assert "metadata" in data
    
    # Test from_dict
    restored = SimpleModel.from_dict(data)
    assert restored.id == model.id
    assert restored.name == model.name
    assert restored.age == model.age

def test_field_defaults():
    """Test that field defaults work correctly."""
    model = SimpleModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25
    )
    assert model.tags == []  # default_factory
    assert model.description is None  # default value

def test_method_preservation():
    """Test that methods from the original class are preserved."""
    model = SimpleModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25
    )
    assert model.get_full_name() == "test (age: 25)"

def test_inheritance_chain():
    """Test that inheritance from decorated models works."""
    model = ChildModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25,
        extra_field="extra"
    )
    assert isinstance(model, BaseModel)
    assert isinstance(model, SimpleModel)
    assert model.get_full_name() == "test (age: 25)"
    assert model.extra_field == "extra"

def test_missing_required_fields():
    """Test that missing required fields raise an error."""
    with pytest.raises(TypeError):
        SimpleModel(
            id="123",
            embedding=[0.1, 0.2]
            # missing name and age
        )

def test_invalid_field_type():
    """Test that invalid field types raise an error."""
    with pytest.raises(TypeCheckError):
        SimpleModel(
            id="123",
            embedding=[0.1, 0.2],
            name="test",
            age="not an integer"  # wrong type
        ) 
