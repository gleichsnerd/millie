from dataclasses import dataclass
from datetime import datetime
import pytest
from millie.orm.milvus_model import MilvusModel

class TestModel(MilvusModel):
    name: str
    age: int
    extra_data: dict = None

    @classmethod
    def collection_name(cls) -> str:
        return "test"

    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                {"name": "id", "type": "str", "is_primary": True},
                {"name": "name", "type": "str"},
                {"name": "age", "type": "int"},
                {"name": "embedding", "type": "float_vector", "dim": 2},
                {"name": "metadata", "type": "json"},
                {"name": "extra_data", "type": "json"}
            ]
        }

def test_collection_name():
    """Test collection name generation."""
    assert TestModel.collection_name() == "test"

def test_post_init_metadata_json():
    """Test metadata JSON parsing in post_init."""
    model = TestModel(
        id="123",
        embedding=[0.1, 0.2],
        metadata='{"key": "value"}',
        name="test",
        age=25
    )
    assert model.metadata == {"key": "value"}

def test_post_init_invalid_metadata():
    """Test handling of invalid metadata JSON."""
    model = TestModel(
        id="123",
        embedding=[0.1, 0.2],
        metadata='invalid json',
        name="test",
        age=25
    )
    assert model.metadata == {}

def test_to_dict_basic():
    """Test basic dictionary conversion."""
    model = TestModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25
    )
    data = model.to_dict()
    assert data["id"] == "123"
    assert data["embedding"] == [0.1, 0.2]
    assert data["name"] == "test"
    assert data["age"] == 25
    assert "metadata" not in data

def test_to_dict_with_metadata():
    """Test dictionary conversion with metadata."""
    model = TestModel(
        id="123",
        embedding=[0.1, 0.2],
        metadata={"key": "value"},
        name="test",
        age=25
    )
    data = model.to_dict()
    assert "metadata" in data
    assert isinstance(data["metadata"], str)
    assert "key" in data["metadata"]

def test_from_dict_basic():
    """Test model creation from dictionary."""
    data = {
        "id": "123",
        "embedding": [0.1, 0.2],
        "name": "test",
        "age": 25
    }
    model = TestModel.from_dict(data)
    assert model.id == "123"
    assert model.embedding == [0.1, 0.2]
    assert model.name == "test"
    assert model.age == 25
    assert model.metadata == {}

def test_from_dict_with_metadata():
    """Test model creation from dictionary with metadata."""
    data = {
        "id": "123",
        "embedding": [0.1, 0.2],
        "metadata": '{"key": "value"}',
        "name": "test",
        "age": 25
    }
    model = TestModel.from_dict(data)
    assert model.metadata == {"key": "value"}

def test_serialize_complex_type():
    """Test serialization of complex data types."""
    model = TestModel(
        id="123",
        embedding=[0.1, 0.2],
        name="test",
        age=25,
        extra_data={
            "date": datetime(2024, 1, 1),
            "nested": {"key": "value"},
            "list": [1, 2, {"inner": "value"}]
        }
    )
    data = model.to_dict()
    assert "extra_data" in data
    # The datetime should be serialized to ISO format
    assert "2024-01-01" in data["extra_data"]

def test_json_serialization():
    """Test JSON serialization and deserialization."""
    original = TestModel(
        id="123",
        embedding=[0.1, 0.2],
        metadata={"meta": "data"},
        name="test",
        age=25,
        extra_data={"key": "value"}
    )
    
    # Serialize to JSON
    json_str = original.serialize_for_json()
    
    # Deserialize back to model
    restored = TestModel.deserialize_from_json(json_str)
    
    # Verify all data is preserved
    assert restored.id == original.id
    assert restored.embedding == original.embedding
    assert restored.metadata == original.metadata
    assert restored.name == original.name
    assert restored.age == original.age
    assert restored.extra_data == original.extra_data 