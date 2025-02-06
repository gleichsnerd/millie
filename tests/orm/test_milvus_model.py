from dataclasses import fields, field as dataclass_field
import pytest
from typeguard import TypeCheckError
from millie.orm.milvus_model import MilvusModel
from millie.orm.decorators import MillieMigrationModel
from typing import Optional, List, Dict, Any
from pymilvus import DataType, FieldSchema

class SimpleModel(MilvusModel):
    """Simple test model."""
    id: str
    name: str
    age: int
    tags: List[str] = dataclass_field(default_factory=list)
    description: Optional[str] = None
    
    def get_full_name(self) -> str:
        return f"{self.name} (age: {self.age})"
    
    @classmethod
    def collection_name(cls) -> str:
        return "simple"
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="age", dtype=DataType.INT64),
                FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
        }

@MillieMigrationModel
class MigrationModel(MilvusModel):
    """Migration test model."""
    id: str
    name: str
    version: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "migrations"
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
        }

class ChildModel(SimpleModel):
    """Child test model."""
    extra_field: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "child"
    
    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {
            "fields": [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="age", dtype=DataType.INT64),
                FieldSchema(name="extra_field", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
        }
    
    def get_full_name(self) -> str:
        return f"{self.name} (age: {self.age})"

def test_model_inheritance():
    """Test that models properly inherit from MilvusModel."""
    assert issubclass(SimpleModel, MilvusModel)

def test_dataclass_creation():
    """Test that models are converted to dataclasses."""
    # Should not raise TypeError
    model = SimpleModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25
    )
    assert model.name == "test"
    assert model.age == 25

def test_milvus_model_fields():
    """Test that MilvusModel fields are preserved."""
    model_fields = {f.name for f in fields(SimpleModel)}
    assert "id" in model_fields
    assert "embedding" in model_fields
    assert "metadata" in model_fields

def test_original_fields():
    """Test that original class fields are preserved."""
    model_fields = {f.name for f in fields(SimpleModel)}
    assert "name" in model_fields
    assert "age" in model_fields

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
    assert isinstance(model, MilvusModel)
    assert issubclass(ChildModel, SimpleModel)
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

def test_migration_model():
    """Test that migration models are properly marked."""
    assert MigrationModel.is_migration_collection == True
    assert SimpleModel.is_migration_collection == False

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

def test_class_name_preservation():
    """Test that original class name is preserved."""
    assert SimpleModel.__name__ == "SimpleModel"
    assert MigrationModel.__name__ == "MigrationModel"

def test_model_functionality():
    """Test that models retain MilvusModel functionality."""
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
