"""Tests for the embeddings CLI commands."""
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from millie.cli.embeddings.manager import embeddings

@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()

@pytest.fixture
def mock_embedding_manager():
    """Mock the EmbeddingManager class."""
    with patch('millie.cli.embeddings.manager.EmbeddingManager') as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance

def test_update_embeddings_success(runner, mock_embedding_manager):
    """Test successful embedding update."""
    # Mock successful results
    mock_embedding_manager.run_embedders.return_value = {
        "test_embedder": {
            "status": "success",
            "count": 1
        }
    }
    
    result = runner.invoke(embeddings, ['update'])
    assert result.exit_code == 0
    assert "✅ All embeddings updated successfully!" in result.output
    mock_embedding_manager.run_embedders.assert_called_once()

def test_update_embeddings_with_errors(runner, mock_embedding_manager):
    """Test embedding update with errors."""
    # Mock results with errors
    mock_embedding_manager.run_embedders.return_value = {
        "test_embedder": {
            "status": "error",
            "error": "Test error"
        }
    }
    
    result = runner.invoke(embeddings, ['update'])
    assert result.exit_code == 1
    assert "❌ Some embedders failed:" in result.output
    assert "test_embedder: Test error" in result.output
    mock_embedding_manager.run_embedders.assert_called_once()

def test_update_embeddings_exception(runner, mock_embedding_manager):
    """Test embedding update with exception."""
    # Mock an exception
    mock_embedding_manager.run_embedders.side_effect = Exception("Test exception")
    
    result = runner.invoke(embeddings, ['update'])
    assert result.exit_code == 1
    assert "❌ Error updating embeddings: Test exception" in result.output
    mock_embedding_manager.run_embedders.assert_called_once()

def test_update_embeddings_mixed_results(runner, mock_embedding_manager):
    """Test embedding update with mixed results."""
    # Mock mixed results (some success, some errors)
    mock_embedding_manager.run_embedders.return_value = {
        "success_embedder": {
            "status": "success",
            "count": 1
        },
        "error_embedder": {
            "status": "error",
            "error": "Test error"
        }
    }
    
    result = runner.invoke(embeddings, ['update'])
    assert result.exit_code == 1
    assert "❌ Some embedders failed:" in result.output
    assert "error_embedder: Test error" in result.output
    mock_embedding_manager.run_embedders.assert_called_once()

def test_embeddings_help(runner):
    """Test embeddings help command."""
    result = runner.invoke(embeddings, ['--help'])
    assert result.exit_code == 0
    assert "Manage embeddings for Milvus collections" in result.output
    assert "update  Update embeddings for all collections" in result.output

def test_update_help(runner):
    """Test update command help."""
    result = runner.invoke(embeddings, ['update', '--help'])
    assert result.exit_code == 0
    assert "Update embeddings for all collections" in result.output 