import os
import pytest
from pathlib import Path
from datetime import datetime
from pymilvus import FieldSchema, DataType

from millie.db.migration_manager import MigrationManager
from millie.schema.schema import Schema, SchemaField
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel

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
def test_model():
    """Create a test model class."""
    @MilvusModel()
    class TestModel:
        id: str
        name: str
        
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
    # First create initial schema
    changes = manager.detect_changes_for_model(test_model)
    migration_path = manager.generate_migration("initial", test_model)
    
    # Then check for changes again
    changes = manager.detect_changes_for_model(test_model)
    assert len(changes["added"]) == 0
    assert len(changes["removed"]) == 0
    assert len(changes["modified"]) == 0

def test_generate_migration_file(manager, test_model):
    """Test generating a migration file."""
    migration_path = manager.generate_migration("test_migration", test_model)
    
    assert os.path.exists(migration_path)
    assert migration_path.endswith(".py")
    
    # Check file content
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
    # Create initial model
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
    
    # Create initial migration
    changes = manager.detect_changes_for_model(ChangeModel)
    migration_path = manager.generate_migration("initial", ChangeModel)
    
    # Modify model to add a field
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
    
    # Create migration for changes
    changes = manager.detect_changes_for_model(ChangeModel)
    assert len(changes["added"]) == 1
    assert changes["added"][0].name == "description"
    
    migration_path = manager.generate_migration("add_description", ChangeModel)
    assert os.path.exists(migration_path)
    
    # Check migration content
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'FieldSchema(name="description"' in content
        assert 'max_length=200' in content

def test_migration_with_field_removal(manager):
    """Test migration generation when removing fields."""
    # Create initial model with two fields
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
    
    # Create initial migration
    changes = manager.detect_changes_for_model(RemoveModel)
    migration_path = manager.generate_migration("initial", RemoveModel)
    
    # Modify model to remove temp field
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
    
    # Create migration for changes
    changes = manager.detect_changes_for_model(RemoveModel)
    assert len(changes["removed"]) == 1
    assert changes["removed"][0].name == "temp"
    
    migration_path = manager.generate_migration("remove_temp", RemoveModel)
    assert os.path.exists(migration_path)
    
    # Check migration content
    with open(migration_path, 'r') as f:
        content = f.read()
        assert 'drop_fields=["temp"]' in content

def test_field_modifications(manager):
    """Test migration generation for various field modifications."""
    # Create initial model with fields having specific constraints
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
    
    # Create initial migration
    changes = manager.detect_changes_for_model(ModifyModel)
    migration_path = manager.generate_migration("initial", ModifyModel)
    
    # Modify model with various field changes
    @MilvusModel()
    class ModifyModel:
        id: str  # Change primary key length
        name: int  # Change type from VARCHAR to INT64
        vector: list  # Change vector dimension
        
        @classmethod
        def collection_name(cls):
            return "modify_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=50, is_primary=True),  # Changed length
                    FieldSchema("name", DataType.INT64),  # Changed type
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=256)  # Changed dimension
                ]
            }
    
    # Create migration for changes
    changes = manager.detect_changes_for_model(ModifyModel)
    
    # Verify field modifications were detected
    assert len(changes["modified"]) == 3
    
    # Check specific modifications - modified fields are (old_field, new_field) tuples
    modified_fields = {field_tuple[1].name: field_tuple[1] for field_tuple in changes["modified"]}
    
    # Check id field changes
    assert "id" in modified_fields
    assert modified_fields["id"].max_length == 50
    assert modified_fields["id"].is_primary
    
    # Check name field changes
    assert "name" in modified_fields
    assert modified_fields["name"].dtype == "INT64"
    assert modified_fields["name"].max_length is None
    
    # Check vector field changes
    assert "vector" in modified_fields
    assert modified_fields["vector"].dtype == "FLOAT_VECTOR"
    assert modified_fields["vector"].dim == 256
    
    # Generate and verify migration file
    migration_path = manager.generate_migration("modify_fields", ModifyModel)
    assert os.path.exists(migration_path)
    
    # Check migration content
    with open(migration_path, 'r') as f:
        content = f.read()
        # Verify field modifications in migration code
        assert 'max_length=50' in content
        assert 'DataType.INT64' in content
        assert 'dim=256' in content 

def test_multi_model_migration(manager):
    """Test migration generation with multiple models changing at once."""
    # Create initial models
    @MilvusModel()
    class FirstModel:
        id: str
        name: str
        
        @classmethod
        def collection_name(cls):
            return "first_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                    FieldSchema("name", DataType.VARCHAR, max_length=100)
                ]
            }
    
    @MilvusModel()
    class SecondModel:
        id: str
        score: float
        
        @classmethod
        def collection_name(cls):
            return "second_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                    FieldSchema("score", DataType.DOUBLE)
                ]
            }
    
    # Create initial migrations
    manager.generate_migration("first_initial", FirstModel)
    manager.generate_migration("second_initial", SecondModel)
    
    # Modify both models
    @MilvusModel()
    class FirstModel:
        id: str
        name: str
        description: str  # Added field
        
        @classmethod
        def collection_name(cls):
            return "first_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                    FieldSchema("name", DataType.VARCHAR, max_length=100),
                    FieldSchema("description", DataType.VARCHAR, max_length=500)
                ]
            }
    
    @MilvusModel()
    class SecondModel:
        id: str  # Kept
        # score removed
        tags: list  # Added vector field
        
        @classmethod
        def collection_name(cls):
            return "second_collection"
        
        @classmethod
        def schema(cls):
            return {
                "fields": [
                    FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                    FieldSchema("tags", DataType.FLOAT_VECTOR, dim=128)
                ]
            }
    
    # Mock finding both models
    manager._find_all_models = lambda: [FirstModel, SecondModel]
    
    # Create combined migration
    changes = manager.detect_changes()
    
    # Verify changes for FirstModel
    assert "FirstModel" in changes
    assert len(changes["FirstModel"]["added"]) == 1
    assert changes["FirstModel"]["added"][0].name == "description"
    assert len(changes["FirstModel"]["removed"]) == 0
    assert len(changes["FirstModel"]["modified"]) == 0
    
    # Verify changes for SecondModel
    assert "SecondModel" in changes
    assert len(changes["SecondModel"]["added"]) == 1
    assert changes["SecondModel"]["added"][0].name == "tags"
    assert len(changes["SecondModel"]["removed"]) == 1
    assert changes["SecondModel"]["removed"][0].name == "score"
    assert len(changes["SecondModel"]["modified"]) == 0
    
    # Generate combined migration
    migration_path = manager.generate_migration("multi_model_changes")
    assert os.path.exists(migration_path)
    
    # Check migration content
    with open(migration_path, 'r') as f:
        content = f.read()
        # Verify FirstModel changes
        assert 'collection = Collection("first_collection")' in content
        assert 'FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=500)' in content
        
        # Verify SecondModel changes
        assert 'collection = Collection("second_collection")' in content
        assert 'drop_fields=["score"]' in content
        assert 'FieldSchema(name="tags", dtype=DataType.FLOAT_VECTOR, dim=128)' in content 

def test_model_detection_edge_cases(manager):
    """Test detection of various model types and edge cases."""
    from millie.orm.base_model import BaseModel
    from millie.orm.milvus_model import MilvusModel
    from pymilvus import DataType, FieldSchema
    import os
    from pathlib import Path
    
    # Create a temporary module
    module_path = Path(manager.schema_dir) / "test_models.py"
    module_content = '''
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel
from pymilvus import DataType, FieldSchema

# Regular Milvus model - should be detected
@MilvusModel()
class ValidModel:
    id: str
    name: str
    
    @classmethod
    def collection_name(cls):
        return "valid_collection"
    
    @classmethod
    def schema(cls):
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=100)
            ]
        }

# Model that inherits from BaseModel but isn't decorated - should be detected
class NonDecoratedModel(BaseModel):
    id: str
    name: str
    
    @classmethod
    def collection_name(cls):
        return "non_decorated_collection"
    
    @classmethod
    def schema(cls):
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=100)
            ]
        }

# Model with multiple inheritance - should be detected
class MixinClass:
    pass
    
class MultiInheritanceModel(BaseModel, MixinClass):
    id: str
    name: str
    
    @classmethod
    def collection_name(cls):
        return "multi_inheritance_collection"
    
    @classmethod
    def schema(cls):
        return {
            "fields": [
                FieldSchema("id", DataType.VARCHAR, max_length=36, is_primary=True),
                FieldSchema("name", DataType.VARCHAR, max_length=100)
            ]
        }
'''
    module_path.write_text(module_content)

    # Set MILLIE_MODEL_GLOB to point to our test file
    os.environ['MILLIE_MODEL_GLOB'] = str(module_path)
    
    try:
        # Test model detection
        models = manager._find_all_models()
        
        # All models that inherit from BaseModel should be detected
        assert len(models) == 3
        model_names = {model.__name__ for model in models}
        assert "ValidModel" in model_names
        assert "NonDecoratedModel" in model_names
        assert "MultiInheritanceModel" in model_names
        
    finally:
        # Clean up
        del os.environ['MILLIE_MODEL_GLOB'] 
