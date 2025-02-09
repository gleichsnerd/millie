"""Tests for the MigrationBuilder class."""
import os
import pytest
from unittest.mock import patch
from pymilvus import DataType

from millie.db.migration_builder import MigrationBuilder
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
from millie.db.schema import SchemaField

@pytest.fixture
def temp_migrations_dir(tmp_path):
    """Create a temporary directory for migrations."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    return str(migrations_dir)

@pytest.fixture
def builder(temp_migrations_dir):
    """Create a MigrationBuilder instance."""
    return MigrationBuilder(migrations_dir=temp_migrations_dir)

def test_init_default_directory():
    """Test initialization with default directory."""
    with patch('os.makedirs') as mock_makedirs:
        builder = MigrationBuilder()
        assert 'schema/migrations' in builder.migrations_dir
        mock_makedirs.assert_called_once()

def test_init_custom_directory(temp_migrations_dir):
    """Test initialization with custom directory."""
    builder = MigrationBuilder(migrations_dir=temp_migrations_dir)
    assert builder.migrations_dir == temp_migrations_dir
    assert os.path.exists(temp_migrations_dir)

def test_generate_empty_migration(builder, temp_migrations_dir):
    """Test generating an empty migration file."""
    migration_path = builder.generate_migration("test_migration")
    assert os.path.exists(migration_path)
    
    with open(migration_path) as f:
        content = f.read()
        assert "class Migration_" in content
        assert "test_migration" in content
        assert "def up(self):" in content
        assert "def down(self):" in content
        assert "pass" in content

class TestModel(MilvusModel):
    """Test model for migration generation."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    name: str = milvus_field(DataType.VARCHAR, max_length=200)
    age: int = milvus_field(DataType.INT64)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test_model"

def test_generate_initial_migration(builder):
    """Test generating an initial migration with fields."""
    changes = {
        "initial": True,
        "added": [
            SchemaField("id", "VARCHAR", max_length=100, is_primary=True),
            SchemaField("name", "VARCHAR", max_length=200),
            SchemaField("age", "INT64")
        ],
        "removed": [],
        "modified": []
    }
    
    up_code, down_code = builder.generate_migration_code(TestModel, changes)
    
    # Check upgrade code
    assert 'fields = [' in up_code
    assert 'FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True)' in up_code
    assert 'FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200)' in up_code
    assert 'FieldSchema(name="age", dtype=DataType.INT64)' in up_code
    assert 'collection = self.ensure_collection("test_model", fields)' in up_code
    
    # Check downgrade code
    assert 'collection = Collection("test_model")' in down_code
    assert 'collection.drop()' in down_code

def test_generate_add_field_migration(builder):
    """Test generating a migration that adds a field."""
    changes = {
        "initial": False,
        "added": [
            SchemaField("email", "VARCHAR", max_length=200)
        ],
        "removed": [],
        "modified": []
    }
    
    up_code, down_code = builder.generate_migration_code(TestModel, changes)
    
    # Check upgrade code
    assert 'collection = Collection("test_model")' in up_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=200)])' in up_code
    
    # Check downgrade code
    assert 'collection.alter_schema(drop_fields=["email"])' in down_code

def test_generate_remove_field_migration(builder):
    """Test generating a migration that removes a field."""
    changes = {
        "initial": False,
        "added": [],
        "removed": [
            SchemaField("age", "INT64")
        ],
        "modified": []
    }
    
    up_code, down_code = builder.generate_migration_code(TestModel, changes)
    
    # Check upgrade code
    assert 'collection.alter_schema(drop_fields=["age"])' in up_code
    
    # Check downgrade code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="age", dtype=DataType.INT64)])' in down_code

def test_generate_modify_field_migration(builder):
    """Test generating a migration that modifies a field."""
    changes = {
        "initial": False,
        "added": [],
        "removed": [],
        "modified": [
            (
                SchemaField("name", "VARCHAR", max_length=200),
                SchemaField("name", "VARCHAR", max_length=300)
            )
        ]
    }
    
    up_code, down_code = builder.generate_migration_code(TestModel, changes)
    
    # Check upgrade code
    assert 'collection.alter_schema(drop_fields=["name"])' in up_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=300)])' in up_code
    
    # Check downgrade code
    assert 'collection.alter_schema(drop_fields=["name"])' in down_code
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200)])' in down_code

def test_field_to_schema_str(builder):
    """Test converting a field to its FieldSchema string representation."""
    # Test basic field
    field = SchemaField("test", "VARCHAR")
    schema_str = builder._field_to_schema_str(field)
    assert schema_str == 'FieldSchema(name="test", dtype=DataType.VARCHAR)'
    
    # Test field with max_length
    field = SchemaField("test", "VARCHAR", max_length=100)
    schema_str = builder._field_to_schema_str(field)
    assert schema_str == 'FieldSchema(name="test", dtype=DataType.VARCHAR, max_length=100)'
    
    # Test field with dimension
    field = SchemaField("test", "FLOAT_VECTOR", dim=128)
    schema_str = builder._field_to_schema_str(field)
    assert schema_str == 'FieldSchema(name="test", dtype=DataType.FLOAT_VECTOR, dim=128)'
    
    # Test primary key field
    field = SchemaField("test", "VARCHAR", is_primary=True)
    schema_str = builder._field_to_schema_str(field)
    assert schema_str == 'FieldSchema(name="test", dtype=DataType.VARCHAR, is_primary=True)'

def test_build_migration_with_changes(builder):
    """Test building a complete migration file with changes."""
    changes = {
        "initial": False,
        "added": [SchemaField("email", "VARCHAR", max_length=200)],
        "removed": [SchemaField("age", "INT64")],
        "modified": []
    }
    
    content = builder.build_migration("add_email_remove_age", TestModel, changes)
    
    assert "add_email_remove_age" in content
    assert "Migration_" in content
    assert 'collection.alter_schema(add_fields=[FieldSchema(name="email", dtype=DataType.VARCHAR, max_length=200)])' in content
    assert 'collection.alter_schema(drop_fields=["age"])' in content

def test_build_migration_without_changes(builder):
    """Test building an empty migration file."""
    content = builder.build_migration("empty_migration")
    
    assert "empty_migration" in content
    assert "Migration_" in content
    assert "        pass" in content 