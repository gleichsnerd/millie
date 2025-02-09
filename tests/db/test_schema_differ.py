"""Tests for the SchemaDiffer class."""
import pytest
from pymilvus import DataType
from millie.db.schema_differ import SchemaDiffer
from millie.db.schema import Schema, SchemaField
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field

class TestModel(MilvusModel):
    """Test model for schema diffing."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    name: str = milvus_field(DataType.VARCHAR, max_length=200)
    age: int = milvus_field(DataType.INT64)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test_model"

@pytest.fixture
def differ():
    """Create a SchemaDiffer instance."""
    return SchemaDiffer()

@pytest.fixture
def base_schema():
    """Create a base schema for testing."""
    return Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=200),
            SchemaField("age", "INT64")
        ]
    )

def test_initial_schema(differ):
    """Test diffing when there is no old schema (initial migration)."""
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=200)
        ]
    )
    
    changes = differ.diff_schemas(None, new_schema)
    
    assert changes["initial"] is True
    assert len(changes["added"]) == 2
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0
    assert [f.name for f in changes["added"]] == ["id", "name"]

def test_add_field(differ, base_schema):
    """Test detecting added fields."""
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            *base_schema.fields,
            SchemaField("email", "VARCHAR", max_length=200)
        ]
    )
    
    changes = differ.diff_schemas(base_schema, new_schema)
    
    assert len(changes["added"]) == 1
    assert changes["added"][0].name == "email"
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0

def test_remove_field(differ, base_schema):
    """Test detecting removed fields."""
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=200)
        ]
    )
    
    changes = differ.diff_schemas(base_schema, new_schema)
    
    assert len(changes["removed"]) == 1
    assert changes["removed"][0].name == "age"
    assert len(changes["added"]) == 0
    assert len(changes["modified"]) == 0

def test_modify_field(differ, base_schema):
    """Test detecting modified fields."""
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=300),  # Changed max_length
            SchemaField("age", "INT64")
        ]
    )
    
    changes = differ.diff_schemas(base_schema, new_schema)
    
    assert len(changes["modified"]) == 1
    old_field, new_field = changes["modified"][0]
    assert old_field.name == "name"
    assert old_field.max_length == 200
    assert new_field.name == "name"
    assert new_field.max_length == 300
    assert len(changes["added"]) == 0
    assert len(changes["removed"]) == 0

def test_multiple_changes(differ, base_schema):
    """Test detecting multiple types of changes at once."""
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=300),  # Modified
            SchemaField("email", "VARCHAR", max_length=200)  # Added
            # age removed
        ]
    )
    
    changes = differ.diff_schemas(base_schema, new_schema)
    
    assert len(changes["added"]) == 1
    assert changes["added"][0].name == "email"
    
    assert len(changes["removed"]) == 1
    assert changes["removed"][0].name == "age"
    
    assert len(changes["modified"]) == 1
    old_field, new_field = changes["modified"][0]
    assert old_field.name == "name"
    assert new_field.max_length == 300

def test_field_modification_types(differ):
    """Test detecting different types of field modifications."""
    old_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("f1", "VARCHAR", max_length=100),
            SchemaField("f2", "INT64"),
            SchemaField("f3", "FLOAT_VECTOR", dim=128),
            SchemaField("f4", "VARCHAR", is_primary=False)
        ]
    )
    
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("f1", "VARCHAR", max_length=200),  # Changed max_length
            SchemaField("f2", "INT32"),  # Changed type
            SchemaField("f3", "FLOAT_VECTOR", dim=256),  # Changed dimension
            SchemaField("f4", "VARCHAR", is_primary=True)  # Changed primary key status
        ]
    )
    
    changes = differ.diff_schemas(old_schema, new_schema)
    
    assert len(changes["modified"]) == 4
    modified_fields = {old.name: (old, new) for old, new in changes["modified"]}
    
    # Check max_length change
    old_f1, new_f1 = modified_fields["f1"]
    assert old_f1.max_length == 100
    assert new_f1.max_length == 200
    
    # Check type change
    old_f2, new_f2 = modified_fields["f2"]
    assert old_f2.dtype == "INT64"
    assert new_f2.dtype == "INT32"
    
    # Check dimension change
    old_f3, new_f3 = modified_fields["f3"]
    assert old_f3.dim == 128
    assert new_f3.dim == 256
    
    # Check primary key change
    old_f4, new_f4 = modified_fields["f4"]
    assert not old_f4.is_primary
    assert new_f4.is_primary

def test_no_changes(differ, base_schema):
    """Test when there are no changes between schemas."""
    new_schema = Schema(
        name="TestModel",
        collection_name="test_model",
        fields=[
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=200),
            SchemaField("age", "INT64")
        ]
    )
    
    changes = differ.diff_schemas(base_schema, new_schema)
    
    assert len(changes["added"]) == 0
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0

def test_generate_migration_code(differ):
    """Test generating migration code from schema changes."""
    changes = {
        "added": [SchemaField("email", "VARCHAR", max_length=200)],
        "removed": [SchemaField("age", "INT64")],
        "modified": [
            (
                SchemaField("name", "VARCHAR", max_length=200),
                SchemaField("name", "VARCHAR", max_length=300)
            )
        ]
    }
    
    up_code, down_code = differ.generate_migration_code(TestModel, changes)
    
    # Check upgrade code
    assert 'collection = Collection("test_model")' in up_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=200)])' in up_code
    assert 'collection.alter_schema(drop_fields=["age"])' in up_code
    assert 'collection.alter_schema(drop_fields=["name"])' in up_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=300)])' in up_code
    
    # Check downgrade code
    assert 'collection = Collection("test_model")' in down_code
    assert 'collection.alter_schema(drop_fields=["email"])' in down_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="age", dtype=DataType.INT64)])' in down_code
    assert 'collection.alter_schema(drop_fields=["name"])' in down_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200)])' in down_code

def test_generate_migration_code_with_vector_field(differ):
    """Test generating migration code for vector fields."""
    changes = {
        "added": [
            SchemaField("embedding", "FLOAT_VECTOR", dim=128)
        ],
        "removed": [],
        "modified": []
    }
    
    up_code, down_code = differ.generate_migration_code(TestModel, changes)
    
    # Check upgrade code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)])' in up_code
    
    # Check downgrade code
    assert 'collection.alter_schema(drop_fields=["embedding"])' in down_code 