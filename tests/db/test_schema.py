from millie.db.schema import Schema, SchemaField
from pymilvus import DataType
from pymilvus import FieldSchema
from millie.orm.fields import milvus_field
from millie.orm.milvus_model import MilvusModel
from typing import List

def test_schema_creation():
    """Test basic schema creation and field management."""
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=[],
        is_migration_collection=False
    )
    
    # Test basic properties
    assert schema.name == "TestModel"
    assert schema.collection_name == "test_collection"
    assert len(schema.fields) == 0
    assert not schema.is_migration_collection

def test_schema_with_fields():
    """Test schema with fields."""
    fields = [
        SchemaField(name="id", dtype="INT64", is_primary=True),
        SchemaField(name="name", dtype="VARCHAR", max_length=128),
        SchemaField(name="embedding", dtype="FLOAT_VECTOR", dim=128)
    ]
    
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=fields,
        is_migration_collection=False
    )
    
    # Test fields were added correctly
    assert len(schema.fields) == 3
    assert schema.fields[0].name == "id"
    assert schema.fields[0].is_primary
    assert schema.fields[1].max_length == 128
    assert schema.fields[2].dim == 128

def test_schema_to_dict():
    """Test schema serialization to dict."""
    fields = [
        SchemaField(name="id", dtype="INT64", is_primary=True),
        SchemaField(name="name", dtype="VARCHAR", max_length=128)
    ]
    
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=fields,
        is_migration_collection=False
    )
    
    data = schema.to_dict()
    
    # Test serialization
    assert data["name"] == "TestModel"
    assert data["collection_name"] == "test_collection"
    assert len(data["schema"]["fields"]) == 2
    assert data["schema"]["fields"][0]["name"] == "id"
    assert data["schema"]["fields"][0]["is_primary"]
    assert data["schema"]["fields"][1]["max_length"] == 128

def test_schema_from_dict():
    """Test schema deserialization from dict."""
    data = {
        "name": "TestModel",
        "collection_name": "test_collection",
        "schema": {
            "fields": [
                {
                    "name": "id",
                    "dtype": "INT64",
                    "is_primary": True
                },
                {
                    "name": "name",
                    "dtype": "VARCHAR",
                    "max_length": 128
                }
            ]
        },
        "is_migration_collection": False
    }
    
    schema = Schema.from_dict(data)
    
    # Test deserialization
    assert schema.name == "TestModel"
    assert schema.collection_name == "test_collection"
    assert len(schema.fields) == 2
    assert schema.fields[0].name == "id"
    assert schema.fields[0].is_primary
    assert schema.fields[1].max_length == 128 

def test_schema_field_edge_cases():
    """Test edge cases for SchemaField."""
    # Test optional parameters
    field = SchemaField(name="test", dtype="VARCHAR")
    assert field.max_length is None
    assert field.dim is None
    assert not field.is_primary
    
    # Test zero values
    field = SchemaField(name="test", dtype="VARCHAR", max_length=0, dim=0)
    assert field.max_length == 0
    assert field.dim == 0
    
    # Test serialization of None values
    data = field.to_dict()
    assert "max_length" in data  # Should include even if 0
    assert "dim" in data  # Should include even if 0
    assert "is_primary" not in data  # Should not include if False

def test_schema_field_from_dict_edge_cases():
    """Test edge cases for SchemaField deserialization."""
    # Test with missing optional fields
    data = {
        "name": "test",
        "dtype": "VARCHAR"
    }
    field = SchemaField.from_dict(data)
    assert field.name == "test"
    assert field.dtype == "VARCHAR"
    assert field.max_length is None
    assert field.dim is None
    assert not field.is_primary
    
    # Test with explicit None values
    data = {
        "name": "test",
        "dtype": "VARCHAR",
        "max_length": None,
        "dim": None,
        "is_primary": False
    }
    field = SchemaField.from_dict(data)
    assert field.max_length is None
    assert field.dim is None

def test_schema_duplicate_fields():
    """Test handling of duplicate fields in schema."""
    fields = [
        SchemaField(name="id", dtype="INT64", is_primary=True),
        SchemaField(name="id", dtype="VARCHAR")  # Duplicate name
    ]
    
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=fields,
        is_migration_collection=False
    )
    
    # Both fields should be present (no automatic deduplication)
    assert len(schema.fields) == 2
    assert schema.fields[0].dtype == "INT64"
    assert schema.fields[1].dtype == "VARCHAR"

def test_schema_empty_fields():
    """Test schema with empty or None fields."""
    # Test with empty list
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=[],
        is_migration_collection=False
    )
    assert len(schema.fields) == 0
    
    # Test serialization of empty fields
    data = schema.to_dict()
    assert len(data["schema"]["fields"]) == 0

def test_schema_special_characters():
    """Test schema with special characters in names."""
    schema = Schema(
        name="Test-Model_123",
        collection_name="test.collection@123",
        fields=[
            SchemaField(name="field-name_123", dtype="VARCHAR")
        ],
        is_migration_collection=False
    )
    
    # Test serialization and deserialization preserves special characters
    data = schema.to_dict()
    loaded = Schema.from_dict(data)
    
    assert loaded.name == "Test-Model_123"
    assert loaded.collection_name == "test.collection@123"
    assert loaded.fields[0].name == "field-name_123" 

def test_schema_field_from_field_schema():
    """Test creating SchemaField from FieldSchema."""
    # Test basic field
    field_schema = FieldSchema(name="test", dtype=DataType.INT64)
    schema_field = SchemaField.from_field_schema(field_schema)
    assert schema_field.name == "test"
    assert schema_field.dtype == "INT64"
    assert not schema_field.is_primary
    
    # Test field with max_length
    field_schema = FieldSchema(name="str_field", dtype=DataType.VARCHAR, max_length=100)
    schema_field = SchemaField.from_field_schema(field_schema)
    assert schema_field.max_length == 100
    
    # Test vector field
    field_schema = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    schema_field = SchemaField.from_field_schema(field_schema)
    assert schema_field.dim == 128
    
    # Test primary key field
    field_schema = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36)
    schema_field = SchemaField.from_field_schema(field_schema)
    assert schema_field.is_primary

def test_schema_field_from_dict():
    """Test creating SchemaField from dictionary."""
    # Test basic field
    data = {"name": "test", "dtype": "INT64"}
    schema_field = SchemaField.from_dict(data)
    assert schema_field.name == "test"
    assert schema_field.dtype == "INT64"
    assert not schema_field.is_primary
    
    # Test field with all attributes
    data = {
        "name": "vector",
        "dtype": "FLOAT_VECTOR",
        "dim": 128,
        "is_primary": True
    }
    schema_field = SchemaField.from_dict(data)
    assert schema_field.name == "vector"
    assert schema_field.dtype == "FLOAT_VECTOR"
    assert schema_field.dim == 128
    assert schema_field.is_primary

def test_schema_field_to_dict():
    """Test converting SchemaField to dictionary."""
    # Test basic field
    schema_field = SchemaField(name="test", dtype="INT64")
    data = schema_field.to_dict()
    assert data == {"name": "test", "dtype": "INT64"}
    
    # Test field with all attributes
    schema_field = SchemaField(
        name="vector",
        dtype="FLOAT_VECTOR",
        dim=128,
        max_length=100,
        is_primary=True
    )
    data = schema_field.to_dict()
    assert data == {
        "name": "vector",
        "dtype": "FLOAT_VECTOR",
        "dim": 128,
        "max_length": 100,
        "is_primary": True
    }

def test_schema_field_to_field_schema():
    """Test converting SchemaField to FieldSchema."""
    # Test basic field
    schema_field = SchemaField(name="test", dtype="INT64")
    field_schema = schema_field.to_field_schema()
    assert field_schema.name == "test"
    assert field_schema.dtype == DataType.INT64
    
    # Test field with all attributes
    schema_field = SchemaField(
        name="vector",
        dtype="FLOAT_VECTOR",
        dim=128,
        max_length=100,
        is_primary=True
    )
    field_schema = schema_field.to_field_schema()
    assert field_schema.name == "vector"
    assert field_schema.dtype == DataType.FLOAT_VECTOR
    assert field_schema.dim == 128
    assert field_schema.max_length == 100
    assert field_schema.is_primary

# Test model for Schema tests
class TestModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True)
    name: str =  milvus_field(DataType.VARCHAR, max_length=100)
    age: int
    vector: List[float]
    
    @classmethod
    def collection_name(cls) -> str:
        return "test"
    
def test_schema_from_model():
    """Test creating Schema from model."""
    schema = Schema.from_model(TestModel)
    assert schema.name == "TestModel"
    assert schema.collection_name == "test"
    assert len(schema.fields) == 2
    
    # Check fields
    id_field = schema.get_field("id")
    assert id_field and id_field.is_primary
    assert id_field.dtype == "VARCHAR"
    
    name_field = schema.get_field("name")
    assert name_field and name_field.max_length == 100
    assert name_field.dtype == "VARCHAR"

def test_schema_from_dict():
    """Test creating Schema from dictionary."""
    data = {
        "name": "TestSchema",
        "collection_name": "test_collection",
        "schema": {
            "fields": [
                {
                    "name": "id",
                    "dtype": "VARCHAR",
                    "max_length": 36,
                    "is_primary": True
                },
                {
                    "name": "vector",
                    "dtype": "FLOAT_VECTOR",
                    "dim": 128
                }
            ]
        },
        "is_migration_collection": True,
        "version": 2
    }
    
    schema = Schema.from_dict(data)
    assert schema.name == "TestSchema"
    assert schema.collection_name == "test_collection"
    assert schema.is_migration_collection
    assert schema.version == 2
    assert len(schema.fields) == 2

def test_schema_to_dict():
    """Test converting Schema to dictionary."""
    schema = Schema(
        name="TestSchema",
        collection_name="test_collection",
        fields=[
            SchemaField(name="id", dtype="VARCHAR", max_length=36, is_primary=True),
            SchemaField(name="vector", dtype="FLOAT_VECTOR", dim=128)
        ],
        is_migration_collection=True
    )
    schema.version = 2
    
    data = schema.to_dict()
    assert data["name"] == "TestSchema"
    assert data["collection_name"] == "test_collection"
    assert data["is_migration_collection"]
    assert data["version"] == 2
    assert len(data["schema"]["fields"]) == 2

def test_schema_get_field():
    """Test getting fields from Schema."""
    schema = Schema(
        name="TestSchema",
        collection_name="test_collection",
        fields=[
            SchemaField(name="id", dtype="VARCHAR", max_length=36, is_primary=True),
            SchemaField(name="vector", dtype="FLOAT_VECTOR", dim=128)
        ]
    )
    
    # Test existing field
    field = schema.get_field("id")
    assert field and field.name == "id"
    assert field.is_primary
    
    # Test non-existent field
    field = schema.get_field("nonexistent")
    assert field is None
