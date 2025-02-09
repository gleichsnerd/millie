"""Tests for the EmbeddingManager class."""
import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from typeguard import TypeCheckError
from millie.db.embedding_manager import EmbeddingManager
from millie.orm.milvus_model import MilvusModel
from millie.orm.fields import milvus_field
from pymilvus import DataType

class TestModel(MilvusModel):
    """Test model for embedding operations."""
    id: str = milvus_field(DataType.VARCHAR, max_length=100, is_primary=True)
    name: str = milvus_field(DataType.VARCHAR, max_length=100)
    description: str = milvus_field(DataType.VARCHAR, max_length=1000)
    embedding: list = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
    
    @classmethod
    def collection_name(cls) -> str:
        return "test_model"

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)

@pytest.fixture
def embedding_manager(temp_dir):
    """Create an EmbeddingManager instance for testing."""
    return EmbeddingManager(cwd=temp_dir)

@pytest.fixture
def test_embedder_file(temp_dir):
    """Create a test file with an embedder function."""
    file_content = '''
from millie.db.milvus_embedder import milvus_embedder

@milvus_embedder
def test_embedder():
    return "test"
'''
    file_path = os.path.join(temp_dir, 'test_embedder.py')
    with open(file_path, 'w') as f:
        f.write(file_content)
    return file_path

def test_has_embedder_decorator(embedding_manager, test_embedder_file):
    """Test detection of milvus_embedder decorator."""
    assert embedding_manager._has_embedder_decorator(test_embedder_file) is True
    
    # Test with file without decorator
    no_decorator_file = os.path.join(embedding_manager.cwd, 'no_decorator.py')
    with open(no_decorator_file, 'w') as f:
        f.write('def regular_function():\n    pass')
    assert embedding_manager._has_embedder_decorator(no_decorator_file) is False

def test_has_embedder_decorator_with_invalid_file(embedding_manager):
    """Test decorator detection with invalid file."""
    # Test with non-existent file
    assert embedding_manager._has_embedder_decorator("nonexistent.py") is False
    
    # Test with invalid Python syntax
    invalid_file = os.path.join(embedding_manager.cwd, 'invalid.py')
    with open(invalid_file, 'w') as f:
        f.write('this is not python code')
    assert embedding_manager._has_embedder_decorator(invalid_file) is False

def test_discover_embedders(embedding_manager, test_embedder_file):
    """Test discovering embedder functions."""
    embedders = embedding_manager.discover_embedders()
    assert len(embedders) == 1
    assert embedders[0].__name__ == 'test_embedder'

def test_discover_embedders_with_error(embedding_manager):
    """Test discovering embedders with syntax error."""
    # Create file with syntax error
    error_file = os.path.join(embedding_manager.cwd, 'error.py')
    with open(error_file, 'w') as f:
        f.write('@milvus_embedder\ndef broken_function():\n    return invalid_var')
    
    embedders = embedding_manager.discover_embedders()
    assert len(embedders) == 0

def test_discover_embedders_with_import_error(embedding_manager):
    """Test discovering embedders with import error."""
    # Create file that imports a non-existent module
    error_file = os.path.join(embedding_manager.cwd, 'import_error.py')
    with open(error_file, 'w') as f:
        f.write('''
from nonexistent_module import something
from millie.db.milvus_embedder import milvus_embedder

@milvus_embedder
def error_embedder():
    pass
''')
    
    embedders = embedding_manager.discover_embedders()
    assert len(embedders) == 0

def test_discover_embedders_with_invalid_spec(embedding_manager):
    """Test discovering embedders with invalid module spec."""
    with patch('importlib.util.spec_from_file_location', return_value=None):
        embedders = embedding_manager.discover_embedders()
        assert len(embedders) == 0

def test_run_embedders(embedding_manager, test_embedder_file):
    """Test running embedder functions."""
    results = embedding_manager.run_embedders()
    assert len(results) == 1
    assert 'test_embedder' in results
    assert results['test_embedder']['status'] == 'success'

def test_run_embedders_with_error(embedding_manager):
    """Test running embedders that raise exceptions."""
    error_file = os.path.join(embedding_manager.cwd, 'error_embedder.py')
    with open(error_file, 'w') as f:
        f.write('''
from millie.db.milvus_embedder import milvus_embedder

@milvus_embedder
def error_embedder():
    raise ValueError("Test error")
''')
    
    results = embedding_manager.run_embedders()
    assert len(results) == 1
    assert 'error_embedder' in results
    assert results['error_embedder']['status'] == 'error'
    assert 'Test error' in results['error_embedder']['error']

def test_run_embedders_no_embedders(embedding_manager):
    """Test running embedders when none are found."""
    results = embedding_manager.run_embedders()
    assert results == {}

def test_get_value_hash():
    """Test generating hash for a value."""
    value = "test string"
    hash1 = EmbeddingManager.get_value_hash(value)
    hash2 = EmbeddingManager.get_value_hash(value)
    
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 hash length
    assert hash1 == hash2  # Same input should produce same hash

def test_load_embeddings_file(temp_dir):
    """Test loading embeddings from file."""
    file_path = Path(temp_dir) / "test.embeddings.json"
    test_data = {"hash1": [1.0, 2.0], "hash2": [3.0, 4.0]}
    
    # Test loading non-existent file
    assert EmbeddingManager.load_embeddings_file(file_path) == {}
    
    # Test loading existing file
    with open(file_path, 'w') as f:
        json.dump(test_data, f)
    
    loaded_data = EmbeddingManager.load_embeddings_file(file_path)
    assert loaded_data == test_data

def test_save_embeddings_file(temp_dir):
    """Test saving embeddings to file."""
    file_path = Path(temp_dir) / "test.embeddings.json"
    test_data = {"hash1": [1.0, 2.0], "hash2": [3.0, 4.0]}
    
    EmbeddingManager.save_embeddings_file(file_path, test_data)
    
    with open(file_path, 'r') as f:
        saved_data = json.load(f)
    assert saved_data == test_data

def test_create_model_from_data():
    """Test creating model instance from data."""
    data = {
        "name": "test",
        "description": "test description"
    }
    
    # Test without source file
    model = EmbeddingManager.create_model_from_data(TestModel, data, "description")
    assert isinstance(model, TestModel)
    assert model.name == "test"
    assert model.description == "test description"
    assert len(model.embedding) == 1536
    assert model.id is not None

def test_create_model_from_data_with_embeddings(temp_dir):
    """Test creating model with cached embeddings."""
    source_file = Path(temp_dir) / "test.yaml"
    embeddings_file = source_file.with_suffix('.embeddings.json')
    
    data = {
        "name": "test",
        "description": "test description"
    }
    
    # Create embeddings file
    description_hash = EmbeddingManager.get_value_hash(data["description"])
    test_embedding = [0.5] * 1536
    embeddings_data = {description_hash: test_embedding}
    
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings_data, f)
    
    # Create model with source file
    model = EmbeddingManager.create_model_from_data(TestModel, data, "description", source_file)
    assert model.embedding == test_embedding

def test_create_model_from_data_missing_field(temp_dir):
    """Test creating model when field is missing."""
    data = {"name": "test"}  # No description field
    
    with pytest.raises(TypeCheckError, match="NoneType is not an instance of <class 'str'>"):
        EmbeddingManager.create_model_from_data(TestModel, data, "description")

def test_process_file(temp_dir):
    """Test processing file and updating embeddings."""
    source_file = Path(temp_dir) / "test.yaml"
    entities = [
        {"name": "test1", "description": "description1"},
        {"name": "test2", "description": "description2"}
    ]
    
    def mock_embedding_generator(text: str) -> list:
        return [float(ord(c)) for c in text[:1536]]
    
    EmbeddingManager.process_file(entities, TestModel, source_file, "description", mock_embedding_generator)
    
    # Verify embeddings file was created
    embeddings_file = source_file.with_suffix('.embeddings.json')
    assert embeddings_file.exists()
    
    # Load and verify embeddings
    with open(embeddings_file, 'r') as f:
        embeddings = json.load(f)
    
    assert len(embeddings) == 2
    for entity in entities:
        hash_value = EmbeddingManager.get_value_hash(entity["description"])
        assert hash_value in embeddings
        assert isinstance(embeddings[hash_value], list)

def test_process_file_removes_unused(temp_dir):
    """Test that process_file removes unused embeddings."""
    source_file = Path(temp_dir) / "test.yaml"
    embeddings_file = source_file.with_suffix('.embeddings.json')
    
    # Create initial embeddings file with unused hash
    initial_embeddings = {
        "unused_hash": [0.1] * 1536,
        EmbeddingManager.get_value_hash("description1"): [0.2] * 1536
    }
    with open(embeddings_file, 'w') as f:
        json.dump(initial_embeddings, f)
    
    # Process file with only one entity
    entities = [{"name": "test1", "description": "description1"}]
    
    def mock_embedding_generator(text: str) -> list:
        return [0.2] * 1536
    
    EmbeddingManager.process_file(entities, TestModel, source_file, "description", mock_embedding_generator)
    
    # Verify unused hash was removed
    with open(embeddings_file, 'r') as f:
        final_embeddings = json.load(f)
    
    assert "unused_hash" not in final_embeddings
    assert len(final_embeddings) == 1
    assert EmbeddingManager.get_value_hash("description1") in final_embeddings

def test_process_file_missing_field(temp_dir):
    """Test processing file with missing fields."""
    source_file = Path(temp_dir) / "test.yaml"
    entities = [
        {"name": "test1"},  # No description field
        {"name": "test2", "description": "description2"}
    ]
    
    def mock_embedding_generator(text: str) -> list:
        return [float(ord(c)) for c in text[:1536]]
    
    with pytest.raises(TypeCheckError, match="NoneType is not an instance of <class 'str'>"):
        EmbeddingManager.process_file(entities, TestModel, source_file, "description", mock_embedding_generator) 