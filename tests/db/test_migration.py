"""Tests for the Migration base class."""
import pytest
from unittest.mock import Mock, patch
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema
from millie.db.migration import Migration

class TestMigration(Migration):
    """Test migration class that implements abstract methods."""
    def __init__(self, up_mock=None, down_mock=None):
        super().__init__()
        self._up_mock = up_mock or Mock()
        self._down_mock = down_mock or Mock()
    
    def up(self):
        """Test implementation of up."""
        self._up_mock()
    
    def down(self):
        """Test implementation of down."""
        self._down_mock()

@pytest.fixture
def migration():
    """Create a test migration instance."""
    return TestMigration()

@pytest.fixture
def mock_collection():
    """Create a mock Collection."""
    with patch('millie.db.migration.Collection') as mock:
        yield mock

@pytest.fixture
def test_fields():
    """Create test fields with a primary key."""
    return [
        FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema("test_field", DataType.VARCHAR, max_length=100),
        FieldSchema("vector_field", DataType.FLOAT_VECTOR, dim=128)
    ]

def test_migration_init():
    """Test migration initialization."""
    migration = TestMigration()
    assert migration.logger is not None
    assert migration.logger.name == "millie.db.migration.TestMigration"

def test_migration_apply_success(migration):
    """Test successful migration apply."""
    migration.apply()
    migration._up_mock.assert_called_once()

def test_migration_apply_failure(migration):
    """Test failed migration apply."""
    migration._up_mock.side_effect = Exception("Test error")
    with pytest.raises(Exception, match="Test error"):
        migration.apply()

def test_migration_rollback_success(migration):
    """Test successful migration rollback."""
    migration.rollback()
    migration._down_mock.assert_called_once()

def test_migration_rollback_failure(migration):
    """Test failed migration rollback."""
    migration._down_mock.side_effect = Exception("Test error")
    with pytest.raises(Exception, match="Test error"):
        migration.rollback()

def test_ensure_collection_exists(mock_collection, test_fields):
    """Test ensuring a collection that already exists."""
    migration = TestMigration()
    
    # Configure mock to simulate existing collection
    mock_collection.return_value = Mock(spec=Collection)
    
    result = migration.ensure_collection("test_collection", test_fields)
    assert result == mock_collection.return_value
    mock_collection.assert_called_once_with("test_collection")

def test_ensure_collection_create_new(mock_collection, test_fields):
    """Test creating a new collection."""
    migration = TestMigration()
    
    # First call raises exception (collection doesn't exist), second creates it
    collection_mock = Mock(spec=Collection)
    mock_collection.side_effect = [Exception(), collection_mock]
    
    result = migration.ensure_collection("test_collection", test_fields)
    assert result == collection_mock
    
    # Verify collection creation with schema
    mock_collection.assert_called_with("test_collection", schema=CollectionSchema(fields=test_fields))
    
    # Verify index creation for vector field
    result.create_index.assert_called_once_with(
        field_name="vector_field",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )

def test_ensure_collection_no_fields():
    """Test ensuring a collection with no fields."""
    migration = TestMigration()
    with pytest.raises(ValueError, match="No fields provided for collection creation"):
        migration.ensure_collection("test_collection", [])

def test_ensure_collection_multiple_vector_fields(mock_collection):
    """Test creating a collection with multiple vector fields."""
    migration = TestMigration()
    fields = [
        FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema("vector1", DataType.FLOAT_VECTOR, dim=128),
        FieldSchema("vector2", DataType.FLOAT_VECTOR, dim=256)
    ]
    
    # Mock collection creation
    collection_mock = Mock(spec=Collection)
    mock_collection.side_effect = [Exception(), collection_mock]
    
    result = migration.ensure_collection("test_collection", fields)
    
    # Verify index creation for both vector fields
    assert result.create_index.call_count == 2
    result.create_index.assert_any_call(
        field_name="vector1",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )
    result.create_index.assert_any_call(
        field_name="vector2",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
    )

def test_ensure_collection_creation_failure(mock_collection, test_fields):
    """Test handling collection creation failure."""
    migration = TestMigration()
    
    # Both attempts to create collection fail
    mock_collection.side_effect = Exception("Failed to create collection")
    
    with pytest.raises(Exception, match="Failed to create collection"):
        migration.ensure_collection("test_collection", test_fields) 