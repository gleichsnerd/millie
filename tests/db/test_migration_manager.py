import os
import pytest
from dataclasses import dataclass
from pymilvus import FieldSchema, DataType

from millie.db.migration_manager import MigrationManager
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel

# Test Models
@dataclass
class SimpleModel:
    field1: str
    field2: int

@MilvusModel()
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

class MultiInheritanceModel(BaseModel, MixinClass):
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
def manager(schema_dir):
    """Create a MigrationManager instance."""
    return MigrationManager(schema_dir=schema_dir)

@pytest.fixture
def test_module(tmp_path):
    """Create a temporary module for model discovery testing."""
    module_path = tmp_path / "test_models.py"
    module_content = '''
from dataclasses import dataclass
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel

@dataclass
class SimpleModel:
    field1: str
    field2: int

@MilvusModel()
class TestModel(SimpleModel):
    pass

class MixinClass:
    pass

class MultiInheritanceModel(BaseModel, MixinClass):
    pass

class NonBaseModel:
    pass
'''
    module_path.write_text(module_content)
    return module_path

@pytest.fixture
def test_model():
    """Create a test model for schema change detection."""
    @MilvusModel()
    class TestModel:
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
    return TestModel

# Model Discovery Tests
def test_find_all_models(test_module):
    """Test finding all models that inherit from BaseModel."""
    os.environ['MILLIE_MODEL_GLOB'] = str(test_module)
    manager = MigrationManager()
    
    models = manager._find_all_models()
    
    assert len(models) == 2
    model_names = {model.__name__ for model in models}
    assert "TestModel" in model_names
    assert "MultiInheritanceModel" in model_names
    
    del os.environ['MILLIE_MODEL_GLOB']

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
    changes = manager.detect_changes_for_model(test_model)
    migration_path = manager.generate_migration("initial", test_model)
    
    changes = manager.detect_changes_for_model(test_model)
    assert len(changes["added"]) == 0
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0

# Migration Generation Tests
def test_generate_migration_file(manager, test_model):
    """Test generating a migration file."""
    migration_path = manager.generate_migration("test_migration", test_model)
    
    assert os.path.exists(migration_path)
    assert migration_path.endswith(".py")
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'class Migration_' in content
        assert 'def up(self)' in content
        assert 'def down(self)' in content
        assert 'collection.alter_schema(add_fields=[' in content
        assert 'FieldSchema(name="id"' in content
        assert 'FieldSchema(name="name"' in content

def test_migration_with_field_changes(manager):
    """Test migration generation with field changes."""
    @MilvusModel()
    class ChangeModel:
        id: str
        
        @classmethod
        def collection_name(cls):
            return "change_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True)
                ]
            }
    
    changes = manager.detect_changes_for_model(ChangeModel)
    migration_path = manager.generate_migration("initial", ChangeModel)
    
    @MilvusModel()
    class ChangeModel:
        id: str
        description: str
        
        @classmethod
        def collection_name(cls):
            return "change_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema("description", DataType.VARCHAR, max_length=200)
                ]
            }
    
    changes = manager.detect_changes_for_model(ChangeModel)
    assert len(changes["added"]) == 1
    assert changes["added"][0].name == "description"
    
    migration_path = manager.generate_migration("add_description", ChangeModel)
    assert os.path.exists(migration_path)
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'FieldSchema(name="description"' in content
        assert 'max_length=200' in content

def test_migration_with_field_removal(manager):
    """Test migration generation when removing fields."""
    @MilvusModel()
    class RemoveModel:
        id: str
        temp: str
        
        @classmethod
        def collection_name(cls):
            return "remove_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema("temp", DataType.VARCHAR, max_length=50)
                ]
            }
    
    changes = manager.detect_changes_for_model(RemoveModel)
    migration_path = manager.generate_migration("initial", RemoveModel)
    
    @MilvusModel()
    class RemoveModel:
        id: str
        
        @classmethod
        def collection_name(cls):
            return "remove_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True)
                ]
            }
    
    changes = manager.detect_changes_for_model(RemoveModel)
    assert len(changes["removed"]) == 1
    assert changes["removed"][0].name == "temp"
    
    migration_path = manager.generate_migration("remove_temp", RemoveModel)
    assert os.path.exists(migration_path)
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'drop_fields=["temp"]' in content

def test_field_modifications(manager):
    """Test migration generation for various field modifications."""
    @MilvusModel()
    class ModifyModel:
        id: str
        name: str
        vector: list
        
        @classmethod
        def collection_name(cls):
            return "modify_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                    FieldSchema("name", DataType.VARCHAR, max_length=100),
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=128)
                ]
            }
    
    changes = manager.detect_changes_for_model(ModifyModel)
    migration_path = manager.generate_migration("initial", ModifyModel)
    
    @MilvusModel()
    class ModifyModel:
        id: str
        name: int
        vector: list
        
        @classmethod
        def collection_name(cls):
            return "modify_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=50, is_primary=True),
                    FieldSchema("name", DataType.INT64),
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=256)
                ]
            }
    
    changes = manager.detect_changes_for_model(ModifyModel)
    assert len(changes["modified"]) == 3
    
    migration_path = manager.generate_migration("modify_fields", ModifyModel)
    assert os.path.exists(migration_path)
    
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'max_length=50' in content
        assert 'DataType.INT64' in content
        assert 'dim=256' in content 