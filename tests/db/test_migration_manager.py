import os
from pymilvus import DataType
import pytest
from millie.orm.fields import milvus_field

from millie.db.migration_manager import MigrationManager
from millie.orm.milvus_model import MODEL_REGISTRY, MilvusModel

# Test Models
class SimpleModel(MilvusModel):
    field1: str
    field2: int

class TestModel(MilvusModel):
    """Test model for migration manager tests."""
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    age: int = milvus_field(DataType.INT64)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test_collection"

class MixinClass:
    pass

class MultiInheritanceModel(MilvusModel, MixinClass):
    pass

@pytest.fixture(autouse=True)
def clear_model_registry():
    """Clear the model registry before each test."""
    MODEL_REGISTRY.clear()
    yield
    MODEL_REGISTRY.clear()

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
    os.environ['MILLIE_SCHEMA_DIR'] = str(schema)
    return str(schema)

@pytest.fixture
def manager(schema_dir, tmp_path):
    """Create a MigrationManager instance."""
    # Create a test models file in the temp directory
    models_file = tmp_path / "test_models.py"
    models_file.write_text('''
from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class TestModel(MilvusModel):
    """Test model for migration manager tests."""
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    age: int = milvus_field(DataType.INT64)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test_collection"
    
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
    model_names = [model.__name__ for model in models]
    assert "TestModel" in model_names


# Schema Change Detection Tests
def test_detect_changes_initial(manager, test_model):
    """Test detecting changes for initial schema."""
    changes = manager.detect_changes_for_model(test_model)
    
    assert len(changes["added"]) == 3
    assert changes["added"][0].name == "id"
    assert changes["added"][1].name == "name"
    assert changes["added"][2].name == "age"
    
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
    models_file.write_text('''from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class ChangeModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def collection_name(cls) -> str:
        return "change_collection"
''')
    
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    manager.detect_changes(save_schema=True)
    
    # Update model with new fields
    models_file.write_text('''from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class ChangeModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    age: int = milvus_field(DataType.INT64)

    @classmethod
    def collection_name(cls) -> str:
        return "change_collection"
''')
    
    # Detect changes
    changes = manager.detect_changes()
    assert len(changes) == 1
    assert "ChangeModel" in changes
    model_changes = changes["ChangeModel"]
    assert model_changes["added"] or model_changes["removed"] or model_changes["modified"]
    assert len(model_changes["added"]) == 2
    assert any(f.name == "name" for f in model_changes["added"])
    assert any(f.name == "age" for f in model_changes["added"])

def test_migration_with_field_removal(manager, tmp_path):
    """Test migration generation when removing fields."""
    # Create initial model file
    models_file = tmp_path / "remove_models.py"
    models_file.write_text('''from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class RemoveModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    temp: str = milvus_field(DataType.VARCHAR, max_length=50)

    @classmethod
    def collection_name(cls) -> str:
        return "remove_collection"
''')
    
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    manager.detect_changes(save_schema=True)
    
    # Update model to remove field
    models_file.write_text('''from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class RemoveModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def collection_name(cls) -> str:
        return "remove_collection"
''')
    
    # Detect changes
    changes = manager.detect_changes()
    assert len(changes) == 1
    assert "RemoveModel" in changes
    model_changes = changes["RemoveModel"]
    assert model_changes["added"] or model_changes["removed"] or model_changes["modified"]
    assert len(model_changes["removed"]) == 1
    assert model_changes["removed"][0].name == "temp"

def test_field_modifications(manager, tmp_path):
    """Test migration generation for various field modifications."""
    # Create initial model file
    models_file = tmp_path / "modify_models.py"
    models_file.write_text('''from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class ModifyModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    vector: list = milvus_field(DataType.FLOAT_VECTOR, dim=128)

    @classmethod
    def collection_name(cls) -> str:
        return "modify_collection"
''')
    
    os.environ['MILLIE_MODEL_GLOB'] = str(models_file)
    manager.detect_changes(save_schema=True)
    
    # Update model with modified fields
    models_file.write_text('''from pymilvus import DataType, FieldSchema
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
import uuid

class ModifyModel(MilvusModel):
    id: str = milvus_field(DataType.VARCHAR, max_length=36, is_primary=True, default_factory=lambda: str(uuid.uuid4()))
    name: str = milvus_field(DataType.VARCHAR, max_length=200)  # Changed max_length
    vector: list = milvus_field(DataType.FLOAT_VECTOR, dim=256)  # Changed dim

    @classmethod
    def collection_name(cls) -> str:
        return "modify_collection"
''')
    
    # Detect changes
    changes = manager.detect_changes()
    assert len(changes) == 1
    assert "ModifyModel" in changes
    model_changes = changes["ModifyModel"]
    assert model_changes["added"] or model_changes["removed"] or model_changes["modified"]
    assert len(model_changes["modified"]) == 2
    modified_fields = [field.name for old_field, field in model_changes["modified"]]
    assert "name" in modified_fields
    assert "vector" in modified_fields 