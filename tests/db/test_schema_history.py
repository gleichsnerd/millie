import os
import pytest
from pathlib import Path
from millie.db.schema_history import SchemaHistory
from millie.schema.schema import Schema, SchemaField
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel

@pytest.fixture
def temp_history_dir(tmp_path):
    """Create a temporary schema directory."""
    history_dir = tmp_path / "schema"
    history_dir.mkdir()
    return history_dir

@pytest.fixture
def temp_migrations_dir(tmp_path):
    """Create a temporary migrations directory."""
    migrations_dir = tmp_path / "schema" / "migrations"
    migrations_dir.mkdir(parents=True)
    return migrations_dir

@pytest.fixture
def schema_history(temp_history_dir, temp_migrations_dir):
    """Create a SchemaHistory instance with temporary directories."""
    return SchemaHistory(str(temp_history_dir), str(temp_migrations_dir))

def test_schema_history_initialization(schema_history, temp_history_dir, temp_migrations_dir):
    """Test SchemaHistory initialization."""
    assert schema_history.history_dir == str(temp_history_dir)
    assert schema_history.migrations_dir == str(temp_migrations_dir)

def test_get_model_schema_filename(schema_history):
    """Test getting model schema filename."""
    class TestModel(BaseModel):
        pass
    
    filename = schema_history.get_model_schema_filename(TestModel)
    assert filename.endswith("TestModel.json")
    
    # Test with Combined prefix
    class CombinedTestModel(BaseModel):
        pass
    
    filename = schema_history.get_model_schema_filename(CombinedTestModel)
    assert filename.endswith("TestModel.json")  # Should strip Combined prefix

def test_load_model_schema_empty(schema_history):
    """Test loading schema when no file exists."""
    class TestModel(BaseModel):
        @classmethod
        def collection_name(cls):
            return "test_collection"
    
    schema = schema_history.get_schema_from_history(TestModel)
    assert schema is not None
    assert schema.collection_name == "test_collection"
    assert len(schema.fields) == 0

def test_save_and_load_model_schema(schema_history):
    """Test saving and loading a schema."""
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=[
            SchemaField(name="id", dtype="INT64", is_primary=True),
            SchemaField(name="name", dtype="VARCHAR", max_length=128)
        ],
        is_migration_collection=False
    )
    
    # Save schema
    schema_history.save_model_schema(schema)
    
    # Create a model class to load the schema
    class TestModel(BaseModel):
        @classmethod
        def collection_name(cls):
            return "test_collection"
    
    # Load schema
    loaded_schema = schema_history.get_schema_from_history(TestModel)
    
    # Verify loaded schema matches saved schema
    assert loaded_schema is not None
    assert loaded_schema.collection_name == schema.collection_name
    assert len(loaded_schema.fields) == 2
    assert loaded_schema.fields[0].name == "id"
    assert loaded_schema.fields[1].name == "name"

def test_parse_field_schema(schema_history):
    """Test parsing FieldSchema from migration code."""
    # Test basic field
    field = schema_history._parse_field_schema('FieldSchema(name="test", dtype=DataType.VARCHAR)')
    assert field.name == "test"
    assert field.dtype == "VARCHAR"
    assert not field.is_primary
    assert field.max_length is None
    assert field.dim is None
    
    # Test field with all parameters
    field = schema_history._parse_field_schema('FieldSchema(name="test", dtype=DataType.FLOAT_VECTOR, dim=128, max_length=256, is_primary=True)')
    assert field.name == "test"
    assert field.dtype == "FLOAT_VECTOR"
    assert field.is_primary
    assert field.max_length == 256
    assert field.dim == 128
    
    # Test invalid field definition
    field = schema_history._parse_field_schema('invalid field def')
    assert field is None

@pytest.fixture
def sample_migration(temp_migrations_dir):
    """Create a sample migration file."""
    migration_content = '''
from pymilvus import FieldSchema, DataType

class Migration_20240125_test:
    def up(self):
        self.alter_schema(add_fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128)
        ])
        
        self.alter_schema(drop_fields=["old_field"])
        
    def down(self):
        pass
'''
    migration_file = temp_migrations_dir / "20240125_000000_test.py"
    migration_file.write_text(migration_content.lstrip())
    return str(migration_file)

def test_apply_migration_to_schema(schema_history, sample_migration):
    """Test applying a migration to a schema."""
    # Create initial schema with old_field
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=[
            SchemaField(name="old_field", dtype="VARCHAR", max_length=64)
        ],
        is_migration_collection=False
    )
    
    # Apply migration
    updated_schema = schema_history.apply_migration_to_schema(schema, sample_migration)
    
    # Verify fields were added and removed correctly
    assert len(updated_schema.fields) == 2
    assert updated_schema.fields[0].name == "id"
    assert updated_schema.fields[0].is_primary
    assert updated_schema.fields[1].name == "name"
    assert updated_schema.fields[1].max_length == 128
    assert not any(f.name == "old_field" for f in updated_schema.fields)

def test_build_model_schema_with_migrations(schema_history, sample_migration):
    """Test building schema from multiple migrations."""
    # Create a second migration
    second_migration = '''
from pymilvus import FieldSchema, DataType

class Migration_20240126_test:
    def up(self):
        self.alter_schema(add_fields=[
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
        ])
    def down(self):
        pass
'''
    second_file = Path(schema_history.migrations_dir) / "20240126_000000_test.py"
    second_file.write_text(second_migration.lstrip())
    
    # Create model class
    class TestModel(BaseModel):
        @classmethod
        def collection_name(cls):
            return "test_collection"
    
    # Build schema from migrations
    schema = schema_history.build_model_schema_from_migrations(TestModel)
    
    # Verify all fields were added in order
    assert schema is not None
    assert schema.collection_name == "test_collection"
    assert len(schema.fields) == 3  # id and name from first migration + embedding from second
    field_names = {f.name for f in schema.fields}
    assert "id" in field_names
    assert "name" in field_names
    assert "embedding" in field_names
    
    # Verify field properties
    embedding_field = next(f for f in schema.fields if f.name == "embedding")
    assert embedding_field.dtype == "FLOAT_VECTOR"
    assert embedding_field.dim == 128

def test_edge_cases_schema_history(schema_history, temp_migrations_dir):
    """Test edge cases in schema history."""
    # Test empty migrations directory
    schema = Schema(name="Test", collection_name="test", fields=[], is_migration_collection=False)
    assert len(schema_history.get_migrations()) == 0
    
    # Test invalid migration file (no migration class)
    invalid_migration = '''
def some_function():
    pass
'''
    invalid_file = temp_migrations_dir / "invalid.py"
    invalid_file.write_text(invalid_migration.lstrip())
    
    # Should not raise error and return unchanged schema
    result = schema_history.apply_migration_to_schema(schema, str(invalid_file))
    assert result.to_dict() == schema.to_dict()
    
    # Test migration with syntax error
    bad_migration = '''
from pymilvus import FieldSchema, DataType

class Migration_20240127_test:
    def up(self):
        self.alter_schema(add_fields=[
            FieldSchema(name="test", dtype=DataType.VARCHAR)
        ]
    def down(self):
        pass
'''
    bad_file = temp_migrations_dir / "bad.py"
    bad_file.write_text(bad_migration.lstrip())
    
    # Should handle error gracefully
    result = schema_history.apply_migration_to_schema(schema, str(bad_file))
    assert result.to_dict() == schema.to_dict()

def test_schema_versioning(schema_history):
    """Test schema versioning during save/load."""
    schema = Schema(
        name="TestModel",
        collection_name="test_collection",
        fields=[SchemaField(name="test", dtype="VARCHAR")],
        is_migration_collection=False
    )
    
    # Save schema first time
    schema_history.save_model_schema(schema)
    
    # Load and verify version
    class TestModel(BaseModel):
        @classmethod
        def collection_name(cls):
            return "test_collection"
    
    loaded = schema_history.get_schema_from_history(TestModel)
    assert loaded.version == 1
    
    # Save again and verify version increment
    schema_history.save_model_schema(loaded)
    loaded = schema_history.get_schema_from_history(TestModel)
    assert loaded.version == 2 