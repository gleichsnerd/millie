import os
import pytest
from pymilvus import FieldSchema, DataType

from millie.db.migration_manager import MigrationManager
from millie.orm.milvus_model import MilvusModel
from millie.orm.milvus_model import MilvusModel

# Test Models
class SimpleModel(MilvusModel):
    field1: str
    field2: int

class TestModel(SimpleModel):
    @classmethod
    def collection_name(cls):
        return "test_collection"
    
    @classmethod
    def schema(cls):
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=50)
            ]
        }

class MixinClass:
    pass

class MultiInheritanceModel(MilvusModel, MixinClass):
    pass

# Fixtures
@pytest.fixture
def migrations_dir(tmp_path):
    """Create a temporary migrations directory."""
    migrations = tmp_path / "schema" / "migrations"
    migrations.mkdir(parents=True)
    return str(migrations)

@pytest.fixture
def schema_dir(tmp_path):
    """Create a temporary schema directory."""
    schema = tmp_path / "schema"
    schema.mkdir(parents=True)
    return str(schema)

@pytest.fixture
def manager(schema_dir, tmp_path):
    """Create a MigrationManager instance."""
    # Create a test models file in the temp directory
    models_file = tmp_path / "test_models.py"
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class TestModel(MilvusModel):
    id: str
    name: str

    @classmethod
    def collection_name(cls) -> str:
        return "test_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=50)
            ]
        }
''')
    
    # Set MILLIE_MODEL_GLOB to point to our test models file
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    
    manager = MigrationManager(schema_dir=schema_dir)
    
    yield manager
    
    # Clean up environment after test
    if 'MILLIE_MODEL_GLOB' in os.environ:
        del os.environ['MILLIE_MODEL_GLOB']

@pytest.fixture
def test_model(manager):
    """Get the test model class from our test environment."""
    models = manager._find_all_models()
    return next(model for model in models if model.__name__ == 'TestModel')

# Model Discovery Tests
def test_find_all_models(manager):
    """Test finding all models that inherit from MilvusModel."""
    models = manager._find_all_models()
    assert len(models) == 1
    assert models[0].__name__ == "TestModel"

# Schema Change Detection Tests
def test_detect_changes_initial(manager, test_model):
    """Test detecting changes for initial schema."""
    changes = manager.detect_changes_for_model(test_model)
    
    assert len(changes["added"]) == 2
    assert changes["added"][0].name == "id"
    assert changes["added"][1].name == "name"
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0

def test_detect_changes_no_changes(manager, test_model):
    """Test detecting no changes when schema is unchanged."""
    migration_path = manager.generate_migration("initial")
    assert os.path.exists(migration_path)

    changes = manager.detect_changes_for_model(test_model)
    assert len(changes["added"]) == 0
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0

# Migration Generation Tests
def test_generate_migration_file(manager, test_model):
    """Test generating a migration file."""
    migration_path = manager.generate_migration("test_migration")
    
    assert os.path.exists(migration_path)
    assert migration_path.endswith(".py")
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'class Migration_' in content
        assert 'def up(self)' in content
        assert 'def down(self)' in content
        assert 'fields = [' in content
        assert 'FieldSchema(name="id"' in content
        assert 'FieldSchema(name="name"' in content

def test_migration_with_field_changes(manager, tmp_path):
    """Test migration generation with field changes."""
    # Create initial model file
    models_file = tmp_path / "change_models.py"
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class ChangeModel(MilvusModel):
    id: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "change_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True)
            ]
        }
''')
    
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    models = manager._find_all_models()
    change_model = next(m for m in models if m.__name__ == 'ChangeModel')
    
    changes = manager.detect_changes_for_model(change_model)
    migration_path = manager.generate_migration("initial")
    
    # Update model with new field
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class ChangeModel(MilvusModel):
    id: str
    description: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "change_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("description", DataType.VARCHAR, max_length=200)
            ]
        }
''')
    
    # Clear model cache and reload
    models = manager._find_all_models()
    change_model = next(m for m in models if m.__name__ == 'ChangeModel')
    
    changes = manager.detect_changes_for_model(change_model)
    assert len(changes["added"]) == 1
    assert changes["added"][0].name == "description"
    
    migration_path = manager.generate_migration("add_description")
    assert os.path.exists(migration_path)
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'FieldSchema(name="description"' in content
        assert 'max_length=200' in content

def test_migration_with_field_removal(manager, tmp_path):
    """Test migration generation when removing fields."""
    # Create initial model file
    models_file = tmp_path / "remove_models.py"
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class RemoveModel(MilvusModel):
    id: str
    temp: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "remove_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema("temp", DataType.VARCHAR, max_length=50)
            ]
        }
''')
    
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    models = manager._find_all_models()
    remove_model = next(m for m in models if m.__name__ == 'RemoveModel')
    
    changes = manager.detect_changes_for_model(remove_model)
    migration_path = manager.generate_migration("initial")
    
    # Update model removing field
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class RemoveModel(MilvusModel):
    id: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "remove_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True)
            ]
        }
''')
    
    # Clear model cache and reload
    models = manager._find_all_models()
    remove_model = next(m for m in models if m.__name__ == 'RemoveModel')
    
    changes = manager.detect_changes_for_model(remove_model)
    assert len(changes["removed"]) == 1
    assert changes["removed"][0].name == "temp"
    
    migration_path = manager.generate_migration("remove_temp")
    assert os.path.exists(migration_path)
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'drop_fields=["temp"]' in content

def test_field_modifications(manager, tmp_path):
    """Test migration generation for various field modifications."""
    # Create initial model file
    models_file = tmp_path / "modify_models.py"
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class ModifyModel(MilvusModel):
    id: str
    name: str
    vector: list
    
    @classmethod
    def collection_name(cls) -> str:
        return "modify_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=100),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
            ]
        }
''')
    
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    models = manager._find_all_models()
    modify_model = next(m for m in models if m.__name__ == 'ModifyModel')
    
    changes = manager.detect_changes_for_model(modify_model)
    migration_path = manager.generate_migration("initial")
    
    # Update model with modified fields
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel

class ModifyModel(MilvusModel):
    id: str
    name: int
    vector: list
    
    @classmethod
    def collection_name(cls) -> str:
        return "modify_collection"
    
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=50, is_primary=True),
                FieldSchema("name", DataType.INT64),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=256)
            ]
        }
''')
    
    # Clear model cache and reload
    models = manager._find_all_models()
    modify_model = next(m for m in models if m.__name__ == 'ModifyModel')
    
    changes = manager.detect_changes_for_model(modify_model)
    assert len(changes["modified"]) == 3
    
    migration_path = manager.generate_migration("modify_fields")
    assert os.path.exists(migration_path)
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'max_length=50' in content
        assert 'DataType.INT64' in content
        assert 'dim=256' in content 