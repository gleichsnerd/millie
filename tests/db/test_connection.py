"""Tests for the MilvusConnection class."""
import os
import pytest
from unittest.mock import patch, Mock, call
from pymilvus import Collection
from millie.db.connection import MilvusConnection

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test."""
    MilvusConnection._instance = None
    MilvusConnection._collections = {}
    yield

@pytest.fixture
def mock_connections():
    """Mock pymilvus connections module."""
    with patch('millie.db.connection.connections') as mock:
        yield mock

@pytest.fixture
def mock_collection():
    """Mock pymilvus Collection class."""
    with patch('millie.db.connection.Collection') as mock:
        yield mock

@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict(os.environ, {
        'MILVUS_HOST': 'test-host',
        'MILVUS_PORT': '19531',
        'MILVUS_DB_NAME': 'test-db'
    }, clear=True), \
    patch('millie.db.connection.load_dotenv') as mock_load:
        yield

@pytest.fixture
def mock_connection():
    """Create a mocked connection with all dependencies."""
    with patch('millie.db.connection.connections') as mock_conn, \
         patch('millie.db.connection.Collection') as mock_coll, \
         patch('millie.db.connection.load_dotenv'):
        yield MilvusConnection()

def test_singleton_pattern(mock_connections):
    """Test that MilvusConnection is a singleton."""
    with patch('millie.db.connection.load_dotenv'):
        conn1 = MilvusConnection()
        conn2 = MilvusConnection()
        assert conn1 is conn2
        
        # Verify connect was only called once
        mock_connections.connect.assert_called_once()

def test_init_with_defaults(mock_connections):
    """Test initialization with default values."""
    with patch.dict(os.environ, clear=True), \
         patch('millie.db.connection.load_dotenv'):
        conn = MilvusConnection()
        assert conn.host == 'localhost'
        assert conn.port == 19530
        assert conn.db_name == 'default'
        
        mock_connections.connect.assert_called_once_with(
            alias="default",
            host='localhost',
            port='19530',
            db_name='default'
        )

def test_init_with_env_vars(mock_env_vars, mock_connections):
    """Test initialization with environment variables."""
    conn = MilvusConnection()
    assert conn.host == 'test-host'
    assert conn.port == 19531
    assert conn.db_name == 'test-db'
    
    mock_connections.connect.assert_called_once_with(
        alias="default",
        host='test-host',
        port='19531',
        db_name='test-db'
    )

def test_init_with_params(mock_connections):
    """Test initialization with explicit parameters."""
    with patch('millie.db.connection.load_dotenv'):
        conn = MilvusConnection(host='custom-host', port=19532, db_name='custom-db')
        assert conn.host == 'custom-host'
        assert conn.port == 19532
        assert conn.db_name == 'custom-db'
        
        mock_connections.connect.assert_called_once_with(
            alias="default",
            host='custom-host',
            port='19532',
            db_name='custom-db'
        )

def test_init_params_override_env_vars(mock_env_vars, mock_connections):
    """Test that explicit parameters override environment variables."""
    conn = MilvusConnection(host='override-host', port=19533, db_name='override-db')
    assert conn.host == 'override-host'
    assert conn.port == 19533
    assert conn.db_name == 'override-db'
    
    mock_connections.connect.assert_called_once_with(
        alias="default",
        host='override-host',
        port='19533',
        db_name='override-db'
    )

def test_connection_error(mock_connections):
    """Test handling of connection errors."""
    with patch('millie.db.connection.load_dotenv'):
        mock_connections.connect.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            MilvusConnection()

def test_close_connection(mock_connection, mock_connections):
    """Test closing the connection."""
    conn = mock_connection
    
    # Create some mock collections
    collection1 = Mock(spec=Collection)
    collection2 = Mock(spec=Collection)
    MilvusConnection._collections = {
        'collection1': collection1,
        'collection2': collection2
    }
    
    conn.close()
    
    # Verify collections were released
    collection1.release.assert_called_once()
    collection2.release.assert_called_once()
    
    # Verify connection was closed
    mock_connections.disconnect.assert_called_once_with("default")

def test_get_collection_cached(mock_connection, mock_collection):
    """Test getting a cached collection."""
    conn = mock_connection
    mock_coll = Mock(spec=Collection)
    mock_collection.return_value = mock_coll
    
    # First call should create new collection
    collection1 = conn.get_collection("test_collection")
    mock_collection.assert_called_once_with("test_collection")
    assert collection1 is mock_coll
    
    # Second call should return cached collection
    collection2 = conn.get_collection("test_collection")
    assert mock_collection.call_count == 1  # No additional Collection creation
    assert collection2 is mock_coll

def test_get_collection_different_names(mock_connection, mock_collection):
    """Test getting different collections."""
    conn = mock_connection
    
    # Create mock collections
    mock_collection1 = Mock(spec=Collection)
    mock_collection2 = Mock(spec=Collection)
    mock_collection.side_effect = [mock_collection1, mock_collection2]
    
    # Get two different collections
    collection1 = conn.get_collection("collection1")
    collection2 = conn.get_collection("collection2")
    
    assert collection1 is mock_collection1
    assert collection2 is mock_collection2
    assert mock_collection.call_count == 2
    mock_collection.assert_has_calls([
        call("collection1"),
        call("collection2")
    ])

def test_remove_collection(mock_connection):
    """Test removing a collection from cache."""
    conn = mock_connection
    
    # Add mock collections to cache
    mock_collection1 = Mock(spec=Collection)
    mock_collection2 = Mock(spec=Collection)
    MilvusConnection._collections = {
        'collection1': mock_collection1,
        'collection2': mock_collection2
    }
    
    # Remove one collection
    conn.remove_collection('collection1')
    
    assert 'collection1' not in MilvusConnection._collections
    assert 'collection2' in MilvusConnection._collections
    assert MilvusConnection._collections['collection2'] is mock_collection2

def test_remove_nonexistent_collection(mock_connection):
    """Test removing a collection that isn't in the cache."""
    conn = mock_connection
    
    # Add a mock collection to cache
    mock_collection = Mock(spec=Collection)
    MilvusConnection._collections = {'collection1': mock_collection}
    
    # Remove a collection that doesn't exist
    conn.remove_collection('nonexistent')
    
    # Verify the existing collection wasn't affected
    assert 'collection1' in MilvusConnection._collections
    assert MilvusConnection._collections['collection1'] is mock_collection 