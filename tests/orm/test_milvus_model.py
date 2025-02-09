"""Tests for MilvusModel functionality."""
from dataclasses import fields
import json
from datetime import datetime
import re
import uuid
import pytest
from unittest.mock import Mock, patch
from pymilvus import Collection, DataType, Hit
from typeguard import TypeCheckError
from millie.orm.milvus_model import MilvusModel, MODEL_REGISTRY
from millie.orm.decorators import MillieMigrationModel
from millie.orm.fields import milvus_field
from typing import Optional, List, ClassVar, Dict, Any

# ============================================================================
# Test Models
# ============================================================================

class TestModel(MilvusModel):
    """Test model for basic functionality."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    name: str = milvus_field(DataType.VARCHAR, max_length=50)
    age: int = milvus_field(DataType.INT64)
    extra_data: dict = milvus_field(DataType.JSON, default_factory=dict)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test"

class SimpleModel(MilvusModel):
    """Simple test model with defaults."""
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    age: int = milvus_field(DataType.INT64)
    tags: List[str] = milvus_field(DataType.ARRAY, default_factory=list)
    description: Optional[str] = milvus_field(DataType.VARCHAR, max_length=500, default=None)
    
    def get_full_name(self) -> str:
        return f"{self.name} (age: {self.age})"
    
    @classmethod
    def collection_name(cls) -> str:
        return "simple"

@MillieMigrationModel
class MigrationModel(MilvusModel):
    """Migration test model."""
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    version: str = milvus_field(DataType.VARCHAR, max_length=50)
    
    @classmethod
    def collection_name(cls) -> str:
        return "migrations"

class ChildModel(SimpleModel):
    """Child test model for inheritance testing."""
    extra_field: str = milvus_field(DataType.VARCHAR, max_length=100)
    
    @classmethod
    def collection_name(cls) -> str:
        return "child"

class ComplexModel(MilvusModel):
    """Model for testing complex type serialization."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
    metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
    nested_data: dict = milvus_field(DataType.JSON)
    
    @classmethod
    def collection_name(cls) -> str:
        return "complex"

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear the model registry before each test."""
    MODEL_REGISTRY.clear()
    yield
    MODEL_REGISTRY.clear()

@pytest.fixture
def mock_collection():
    """Create a mock Milvus collection."""
    with patch("millie.orm.milvus_model.Collection") as mock:
        collection = Mock(spec=Collection)
        mock.return_value = collection
        yield collection

@pytest.fixture
def mock_connection():
    """Mock MilvusConnection."""
    with patch('millie.orm.milvus_model.MilvusConnection') as mock:
        mock_collection = Mock(spec=Collection)
        mock.get_collection.return_value = mock_collection
        yield mock

@pytest.fixture
def test_model():
    """Create a test model instance."""
    return TestModel(
        id="123",
        name="Test Model",
        age=25,
        embedding=[0.1] * 128,
        metadata={"key": "value"}
    )

# ============================================================================
# Core MilvusModel Tests
# ============================================================================

def test_required_fields(test_model):
    """Test that MilvusModel requires id, embedding, and metadata fields."""
    assert hasattr(test_model, 'id')
    assert hasattr(test_model, 'embedding')
    assert hasattr(test_model, 'metadata')

def test_field_defaults():
    """Test default values for MilvusModel fields."""
    model = TestModel(
        id="123",
        name="test",
        age=25
    )
    assert model.embedding is None  # Optional by default
    assert model.metadata == {}  # Default empty dict
    assert model.extra_data == {}  # Default from field

def test_metadata_validation():
    """Test metadata field validation."""
    # Valid metadata
    model = TestModel(
        id="123",
        name="test",
        age=25,
        metadata={"key": "value"}
    )
    assert model.metadata == {"key": "value"}
    
    # Invalid metadata
    with pytest.raises(TypeCheckError, match=re.escape("str is not an instance of typing.Dict[str, typing.Any]")):
        TestModel(
            id="123",
            name="test",
            age=25,
            metadata="invalid"
        )

def test_embedding_validation():
    """Test embedding field validation."""
    # Valid embedding
    model = TestModel(
        id="123",
        name="test",
        age=25,
        embedding=[0.1, 0.2]
    )
    assert model.embedding == [0.1, 0.2]
    
    # Invalid embedding
    with pytest.raises(TypeCheckError):
        TestModel(
            id="123",
            name="test",
            age=25,
            embedding="invalid"
        )

def test_model_registration():
    """Test automatic model registration."""
    # Clear registry first
    MODEL_REGISTRY.clear()
    
    # Create a new model class
    class RegisterTestModel(MilvusModel):
        id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
        embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
        metadata: Dict[str, Any] = milvus_field(DataType.JSON, default_factory=dict)
        
        @classmethod
        def collection_name(cls) -> str:
            return "register_test"
    
    assert "RegisterTestModel" in MODEL_REGISTRY
    assert MODEL_REGISTRY["RegisterTestModel"] == RegisterTestModel

def test_schema_generation():
    """Test schema generation from model."""
    schema = TestModel.schema()
    
    # Check schema properties
    assert schema.name == "TestModel"
    assert schema.collection_name == "test"
    assert not schema.is_migration_collection
    
    # Check fields
    field_dict = {f.name: f for f in schema.fields}
    
    # Check id field
    assert "id" in field_dict
    assert field_dict["id"].dtype == DataType.VARCHAR
    assert field_dict["id"].max_length == 100
    assert field_dict["id"].is_primary
    
    # Check embedding field
    assert "embedding" in field_dict
    assert field_dict["embedding"].dtype == DataType.FLOAT_VECTOR
    assert field_dict["embedding"].dim == 1536
    
    # Check metadata field
    assert "metadata" in field_dict
    assert field_dict["metadata"].dtype == DataType.JSON

# ============================================================================
# Inheritance Tests
# ============================================================================

def test_model_inheritance():
    """Test model inheritance and field propagation."""
    # Test inheritance chain
    assert issubclass(SimpleModel, MilvusModel)
    assert issubclass(ChildModel, SimpleModel)
    
    # Test field inheritance
    model = ChildModel(
        id="123",
        name="test",
        age=25,
        extra_field="extra"
    )
    
    # Check inherited fields
    assert model.id == "123"
    assert model.name == "test"
    assert model.age == 25
    assert model.metadata == {}
    assert model.embedding is None
    assert model.tags == []
    assert model.description is None
    
    # Check new fields
    assert model.extra_field == "extra"
    
    # Check inherited methods
    assert model.get_full_name() == "test (age: 25)"
    assert ChildModel.collection_name() == "child"

def test_inheritance_schema():
    """Test schema generation in inherited models."""
    parent_schema = SimpleModel.schema()
    child_schema = ChildModel.schema()
    
    # Child should have all parent fields plus its own
    parent_fields = {f.name for f in parent_schema.fields}
    child_fields = {f.name for f in child_schema.fields}
    
    assert parent_fields.issubset(child_fields)
    assert "extra_field" in child_fields
    assert len(child_fields) == len(parent_fields) + 1

def test_inheritance_defaults():
    """Test default values in inherited models."""
    model = ChildModel(
        name="test",
        age=25,
        extra_field="extra"
    )
    
    # Check inherited defaults
    assert isinstance(model.id, str)  # UUID default
    assert model.tags == []  # default_factory
    assert model.description is None  # default value
    assert model.metadata == {}  # default_factory from MilvusModel
    assert model.embedding is None  # Optional from MilvusModel

def test_migration_model_inheritance():
    """Test migration model inheritance and decoration."""
    assert MigrationModel.is_migration_collection == True
    assert SimpleModel.is_migration_collection == False
    assert ChildModel.is_migration_collection == False
    
    model = MigrationModel(
        name="test",
        version="1.0"
    )
    
    # Check fields and defaults
    assert isinstance(model.id, str)
    assert model.name == "test"
    assert model.version == "1.0"
    assert model.metadata == {}
    assert model.embedding is None

# ============================================================================
# Complex Type Tests
# ============================================================================

def test_complex_type_serialization():
    """Test serialization of complex nested types."""
    nested_data = {
        "datetime": datetime(2024, 1, 1),
        "list": [
            {"inner": datetime(2024, 1, 2)},
            {"inner": [1, 2, {"deep": datetime(2024, 1, 3)}]}
        ],
        "dict": {
            "inner": {
                "date": datetime(2024, 1, 4),
                "list": [datetime(2024, 1, 5)]
            }
        }
    }
    
    model = ComplexModel(
        id="123",
        nested_data=nested_data
    )
    
    data = model.to_dict()
    serialized_data = json.loads(data["nested_data"])
    
    # Check datetime serialization at different levels
    assert serialized_data["datetime"] == "2024-01-01T00:00:00"
    assert serialized_data["list"][0]["inner"] == "2024-01-02T00:00:00"
    assert serialized_data["list"][1]["inner"][2]["deep"] == "2024-01-03T00:00:00"
    assert serialized_data["dict"]["inner"]["date"] == "2024-01-04T00:00:00"
    assert serialized_data["dict"]["inner"]["list"][0] == "2024-01-05T00:00:00"

def test_complex_type_edge_cases():
    """Test serialization of edge cases in complex types."""
    edge_cases = {
        "none": None,
        "empty_dict": {},
        "empty_list": [],
        "mixed_list": [1, None, {}, []],
        "nested_empty": {"dict": {}, "list": []},
        "special_chars": {"key\n\t": "value\n\t"},
        "unicode": {"🔑": "值"}
    }
    
    model = ComplexModel(
        id="123",
        nested_data=edge_cases
    )
    
    data = model.to_dict()
    serialized_data = json.loads(data["nested_data"])
    
    # Check edge cases are preserved
    assert serialized_data["none"] is None
    assert serialized_data["empty_dict"] == {}
    assert serialized_data["empty_list"] == []
    assert serialized_data["mixed_list"] == [1, None, {}, []]
    assert serialized_data["nested_empty"] == {"dict": {}, "list": []}
    assert serialized_data["special_chars"] == {"key\n\t": "value\n\t"}
    assert serialized_data["unicode"] == {"🔑": "值"}

def test_complex_type_deserialization():
    """Test deserialization of complex types."""
    json_data = {
        "id": "123",
        "nested_data": json.dumps({
            "datetime": "2024-01-01T00:00:00",
            "list": [1, "2024-01-02T00:00:00", {"key": "value"}],
            "none": None,
            "empty": {},
            "unicode": {"🔑": "值"}
        })
    }
    
    model = ComplexModel.from_dict(json_data)
    
    # Check deserialized data
    assert model.id == "123"
    assert isinstance(model.nested_data, dict)
    assert model.nested_data["datetime"] == "2024-01-01T00:00:00"
    assert model.nested_data["list"] == [1, "2024-01-02T00:00:00", {"key": "value"}]
    assert model.nested_data["none"] is None
    assert model.nested_data["empty"] == {}
    assert model.nested_data["unicode"] == {"🔑": "值"}

def test_complex_type_validation():
    """Test validation of complex types."""
    # Test invalid nested data type
    with pytest.raises(TypeCheckError):
        ComplexModel(
            id="123",
            nested_data="not a dict"
        )

# ============================================================================
# Database Operation Tests
# ============================================================================

def test_collection_operations(mock_connection):
    """Test collection operations like load, unload, save, and delete."""
    collection = mock_connection.get_collection.return_value
    
    model = TestModel(
        id="123",
        name="test",
        age=25
    )
    
    # Test save operation
    model.save()
    collection.delete.assert_called_once_with('id == "123"')
    collection.insert.assert_called_once()
    
    # Reset mock calls
    collection.delete.reset_mock()
    collection.insert.reset_mock()
    
    # Test load operation
    TestModel.load()
    collection.load.assert_called_once()
    
    # Test unload operation
    TestModel.unload()
    collection.release.assert_called_once()
    
    # Test delete operation
    model.delete()
    collection.delete.assert_called_once_with('id == "123"')

def test_query_operations(mock_connection):
    """Test query operations."""
    collection = mock_connection.get_collection.return_value
    
    # Mock query results
    mock_results = [
        {"id": "1", "name": "Test 1", "age": 25},
        {"id": "2", "name": "Test 2", "age": 30}
    ]
    collection.query.return_value = mock_results
    
    # Test get_all
    results = TestModel.get_all(
        offset=10,
        limit=5,
        output_fields=["id", "name"],
        order_by="age",
        order_desc=True
    )
    collection.query.assert_called_with(
        expr="",
        output_fields=["id", "name"],
        offset=10,
        limit=5,
        order_by="age DESC"
    )
    
    # Test get_by_id
    result = TestModel.get_by_id("test-123")
    collection.query.assert_called_with(
        expr='id == "test-123"',
        output_fields=['*']
    )
    
    # Test filter
    results = TestModel.filter(age=25, name="Test 1")
    collection.query.assert_called_with(
        expr='age == 25 && name == "Test 1"',
        output_fields=['*']
    )

def test_search_operations(mock_connection):
    """Test vector search operations."""
    collection = mock_connection.get_collection.return_value
    
    # Mock search results
    mock_results = Mock()
    mock_results.ids = ["1", "2"]
    mock_results.distances = [0.1, 0.2]
    mock_results.fields = [
        {"id": "1", "name": "Test 1", "age": 25},
        {"id": "2", "name": "Test 2", "age": 30}
    ]
    mock_results.__iter__ = lambda self: iter(self.fields)
    collection.search.return_value = [mock_results]
    
    # Test similarity search
    query_embedding = [0.1] * 128
    results = TestModel.search_by_similarity(
        query_embedding,
        limit=10,
        expr='age > 20',
        metric_type="IP",
        search_params={"nprobe": 20},
        output_fields=["id", "name"]
    )
    
    collection.search.assert_called_with(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 20}},
        limit=10,
        expr='age > 20',
        output_fields=["id", "name"]
    )

def test_bulk_operations(mock_connection):
    """Test bulk insert and upsert operations."""
    collection = mock_connection.get_collection.return_value
    
    # Create test models
    models = [
        TestModel(id=f"test_{i}", name=f"Test {i}", age=i)
        for i in range(5)
    ]
    
    # Test bulk insert
    assert TestModel.bulk_insert(models, batch_size=2) is True
    assert collection.insert.call_count == 3  # 2 batches of 2 and 1 batch of 1
    
    # Reset mock
    collection.insert.reset_mock()
    collection.delete.reset_mock()
    
    # Test bulk upsert
    assert TestModel.bulk_upsert(models, batch_size=2) is True
    assert collection.delete.call_count == 3  # Same batching
    assert collection.insert.call_count == 3  # Same batching
