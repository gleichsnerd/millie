"""Tests for the SeedManager class."""
import os
import pytest
from unittest.mock import patch, MagicMock
from millie.db.seed_manager import SeedManager
from millie.db.milvus_seeder import milvus_seeder, _SEEDERS

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return str(tmp_path)

@pytest.fixture
def seed_manager(temp_dir):
    """Create a SeedManager instance for testing."""
    return SeedManager(cwd=temp_dir)

@pytest.fixture
def mock_glob():
    """Mock glob.glob to return test files."""
    with patch('glob.glob') as mock:
        mock.return_value = [
            'test_seeds/seed_one.py',
            'test_seeds/seed_two.py'
        ]
        yield mock

@pytest.fixture
def mock_import():
    """Mock importlib to simulate module imports."""
    with patch('importlib.util.spec_from_file_location') as mock_spec:
        mock_loader = MagicMock()
        mock_spec.return_value = MagicMock(loader=mock_loader)
        yield mock_spec

def test_discover_seeders_empty(seed_manager, mock_glob):
    """Test discovering seeders when no files are found."""
    mock_glob.return_value = []
    seeders = seed_manager.discover_seeders()
    assert len(seeders) == 0

def test_discover_seeders_with_files(seed_manager, temp_dir):
    """Test discovering seeders from Python files."""
    _SEEDERS.clear()
    
    # Create test seeders in temporary directory
    seeder_code = '''
from millie.db.milvus_seeder import milvus_seeder

@milvus_seeder
def seed_one():
    return "one"

@milvus_seeder
def seed_two():
    return "two"
'''
    with open(os.path.join(temp_dir, 'seeders.py'), 'w') as f:
        f.write(seeder_code)
    
    seeders = seed_manager.discover_seeders()
    assert len(seeders) == 2

def test_run_seeders_success(seed_manager, temp_dir):
    """Test running seeders that complete successfully."""
    _SEEDERS.clear()

    # Create test seeder in temporary directory
    seeder_code = '''
from millie.db.milvus_seeder import milvus_seeder
from millie.orm.milvus_model import MilvusModel

class TestModel(MilvusModel):
    name: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "test"
        
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                {"name": "id", "type": "str", "is_primary": True},
                {"name": "name", "type": "str"}
            ]
        }

@milvus_seeder
def successful_seeder():
    return TestModel(id="1", name="test")
'''
    with open(os.path.join(temp_dir, 'success_seeder.py'), 'w') as f:
        f.write(seeder_code)

    results = seed_manager.run_seeders()
    assert "successful_seeder" in results
    assert results["successful_seeder"]["status"] == "success"
    assert results["successful_seeder"]["count"] == 1

def test_run_seeders_with_error(seed_manager, temp_dir):
    """Test running seeders where some fail."""
    _SEEDERS.clear()

    # Create test seeder in temporary directory
    seeder_code = '''
from millie.db.milvus_seeder import milvus_seeder

@milvus_seeder
def failing_seeder():
    raise ValueError("Test error")
'''
    with open(os.path.join(temp_dir, 'failing_seeder.py'), 'w') as f:
        f.write(seeder_code)

    results = seed_manager.run_seeders()
    assert "failing_seeder" in results
    assert results["failing_seeder"]["status"] == "error"
    assert results["failing_seeder"]["error"] == "Error: Test error"

def test_run_seeders_mixed_results(seed_manager, temp_dir):
    """Test running a mix of successful and failing seeders."""
    _SEEDERS.clear()

    # Create test seeders in temporary directory
    seeder_code = '''
from millie.db.milvus_seeder import milvus_seeder
from millie.orm.milvus_model import MilvusModel

class TestModel(MilvusModel):
    name: str
    
    @classmethod
    def collection_name(cls) -> str:
        return "test"
        
    @classmethod
    def schema(cls) -> dict:
        return {
            "fields": [
                {"name": "id", "type": "str", "is_primary": True},
                {"name": "name", "type": "str"}
            ]
        }

@milvus_seeder
def success_one():
    return TestModel(id="1", name="test1")

@milvus_seeder
def failing_one():
    raise ValueError("Error1")

@milvus_seeder
def success_two():
    return TestModel(id="2", name="test2")
'''
    with open(os.path.join(temp_dir, 'mixed_seeders.py'), 'w') as f:
        f.write(seeder_code)

    results = seed_manager.run_seeders()
    assert len(results) == 4  # 3 seeders + 1 upsert result
    assert results["success_one"]["status"] == "success"
    assert results["success_one"]["count"] == 1
    assert results["failing_one"]["status"] == "error"
    assert results["failing_one"]["error"] == "Error: Error1"
    assert results["success_two"]["status"] == "success"
    assert results["success_two"]["count"] == 1
    assert results["upsert_test"]["status"] == "error"
    assert "Collection test does not exist" in results["upsert_test"]["error"]

def test_custom_working_directory(mock_glob):
    """Test using a custom working directory."""
    custom_dir = "/custom/path"
    manager = SeedManager(cwd=custom_dir)
    manager.discover_seeders()
    
    # Verify glob was called with custom directory
    mock_glob.assert_called_once_with(os.path.join(custom_dir, "**/*.py"), recursive=True)

def test_import_error_handling(seed_manager, mock_glob, mock_import):
    """Test handling of import errors."""
    mock_import.side_effect = ImportError("Test import error")
    
    # Should not raise an exception
    seeders = seed_manager.discover_seeders()
    assert len(seeders) == 0 