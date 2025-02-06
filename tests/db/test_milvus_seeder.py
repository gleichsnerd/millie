"""Tests for the milvus_seeder decorator."""
import pytest
from millie.db.milvus_seeder import milvus_seeder, _SEEDERS

def test_seeder_registration():
    """Test that decorated functions are registered in _SEEDERS."""
    # Clear any existing seeders
    _SEEDERS.clear()
    
    @milvus_seeder
    def test_seeder():
        return "test"
        
    assert "test_seeder" in _SEEDERS
    assert _SEEDERS["test_seeder"] == test_seeder
    
def test_seeder_execution():
    """Test that decorated functions can be executed normally."""
    @milvus_seeder
    def test_seeder():
        return "test_value"
        
    result = test_seeder()
    assert result == "test_value"
    
def test_seeder_preserves_metadata():
    """Test that decorator preserves function metadata."""
    @milvus_seeder
    def test_seeder():
        """Test docstring."""
        pass
        
    assert test_seeder.__name__ == "test_seeder"
    assert test_seeder.__doc__ == "Test docstring."
    
def test_multiple_seeders():
    """Test that multiple seeders can be registered."""
    _SEEDERS.clear()
    
    @milvus_seeder
    def seeder_one():
        return 1
        
    @milvus_seeder
    def seeder_two():
        return 2
        
    assert len(_SEEDERS) == 2
    assert "seeder_one" in _SEEDERS
    assert "seeder_two" in _SEEDERS
    assert _SEEDERS["seeder_one"]() == 1
    assert _SEEDERS["seeder_two"]() == 2 