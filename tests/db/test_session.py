"""Tests for the MilvusSession class."""
import pytest
from unittest.mock import patch, MagicMock
from pymilvus import Collection, CollectionSchema, utility
from millie.db.session import MilvusSession
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
from pymilvus import DataType

class TestModel(MilvusModel):
    """Test model for session operations."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    embedding: list = milvus_field(DataType.FLOAT_VECTOR, dim=3)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test_model"

@pytest.fixture
def mock_collection():
    """Mock Milvus collection."""
    collection = MagicMock(spec=Collection)
    collection.load = MagicMock()
    collection.release = MagicMock()
    collection.create_index = MagicMock()
    return collection

@pytest.fixture
def mock_utility():
    """Mock pymilvus utility functions."""
    with patch('millie.db.session.utility') as mock:
        mock.has_collection = MagicMock(return_value=True)
        mock.list_collections = MagicMock(return_value=['test_collection'])
        mock.drop_collection = MagicMock()
        yield mock

@pytest.fixture
def session(mock_utility):
    """Create a MilvusSession instance."""
    with patch('millie.db.session.MilvusConnection') as mock_conn:
        mock_conn.get_collection = MagicMock()
        session = MilvusSession(host='localhost', port=19530)
        yield session

def test_init_session():
    """Test session initialization."""
    with patch('millie.db.session.MilvusConnection') as mock_conn:
        session = MilvusSession(host='test-host', port=1234, db_name='test-db')
        mock_conn.assert_called_once_with('test-host', 1234, 'test-db')

def test_collection_exists(session, mock_utility):
    """Test checking if collection exists."""
    assert session.collection_exists(TestModel) is True
    mock_utility.has_collection.assert_called_once_with('test_model')

def test_get_collection_existing(session, mock_collection, mock_utility):
    """Test getting an existing collection."""
    with patch('millie.db.session.MilvusConnection.get_collection', return_value=mock_collection) as mock_get:
        collection = session.get_milvus_collection(TestModel)
        assert collection == mock_collection
        mock_utility.has_collection.assert_called_once_with('test_model')
        mock_get.assert_called_once_with('test_model')

def test_get_collection_new(session, mock_collection, mock_utility):
    """Test creating a new collection."""
    mock_utility.has_collection.return_value = False
    with patch('millie.db.session.Collection') as mock_coll:
        mock_coll.return_value = mock_collection
        with patch('millie.db.session.MilvusConnection.get_collection', return_value=mock_collection) as mock_get:
            collection = session.get_milvus_collection(TestModel)
            mock_coll.assert_called_once()
            mock_collection.create_index.assert_called_once()
            mock_get.assert_called_once_with('test_model')

def test_drop_all_collections(session, mock_utility):
    """Test dropping all collections."""
    session.drop_all_collections()
    mock_utility.list_collections.assert_called_once()
    mock_utility.drop_collection.assert_called_once_with('test_collection')

def test_drop_all_collections_empty(session, mock_utility):
    """Test dropping collections when none exist."""
    mock_utility.list_collections.return_value = []
    session.drop_all_collections()
    mock_utility.list_collections.assert_called_once()
    mock_utility.drop_collection.assert_not_called()

def test_load_collection(session, mock_collection):
    """Test loading a collection."""
    with patch('millie.db.session.MilvusConnection.get_collection', return_value=mock_collection) as mock_get:
        session.load_collection(TestModel)
        mock_collection.load.assert_called_once()
        mock_get.assert_called_once_with('test_model')

def test_unload_collection(session):
    """Test unloading a collection."""
    with patch('millie.db.session.Collection') as mock_coll:
        mock_collection = MagicMock()
        mock_coll.return_value = mock_collection
        
        session.unload_collection(TestModel)
        mock_collection.release.assert_called_once()

def test_init_collection_success(session, mock_collection):
    """Test successful collection initialization."""
    with patch('millie.db.session.MilvusConnection.get_collection', return_value=mock_collection) as mock_get:
        session.init_collection(TestModel)
        mock_get.assert_called_once_with('test_model')

def test_init_collection_error(session, mock_collection):
    """Test collection initialization with error."""
    with patch('millie.db.session.MilvusConnection.get_collection', side_effect=Exception("Test error")):
        with pytest.raises(Exception) as exc_info:
            session.init_collection(TestModel)
        assert str(exc_info.value) == "Test error"

def test_collection_method(session, mock_collection):
    """Test the collection method."""
    with patch('millie.db.session.MilvusConnection.get_collection', return_value=mock_collection) as mock_get:
        collection = session.collection(TestModel)
        assert collection == mock_collection
        mock_get.assert_called_once_with('test_model') 