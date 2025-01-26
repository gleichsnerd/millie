import os
import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from millie.cli.migrate.manager import migrate, check_migration_table
from millie.db.session import MilvusSession
from millie.db.migration_history import MigrationHistoryModel
from millie.schema.schema import Schema, SchemaField
from millie.db.schema_history import SchemaHistory

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_session():
    """Create a mock MilvusSession."""
    with patch('millie.cli.migrate.manager.MilvusSession') as mock:
        session_instance = MagicMock()
        mock.return_value = session_instance
        yield session_instance

@pytest.fixture
def mock_migration_manager():
    """Create a mock MigrationManager."""
    with patch('millie.cli.migrate.manager.MigrationManager') as mock:
        manager_instance = MagicMock()
        mock.return_value = manager_instance
        yield manager_instance

@pytest.fixture
def mock_schema_history():
    """Create a mock SchemaHistory."""
    with patch('millie.cli.migrate.manager.SchemaHistory') as mock:
        history_instance = MagicMock()
        mock.return_value = history_instance
        yield history_instance

def test_check_migration_table(mock_session):
    """Test checking if migration table exists."""
    # Test when table exists
    mock_session.collection_exists.return_value = True
    assert check_migration_table() is True

    # Test when table doesn't exist
    mock_session.collection_exists.return_value = False
    assert check_migration_table() is False

def test_init_command_table_exists(cli_runner, mock_session):
    """Test init command when migration table already exists."""
    mock_session.collection_exists.return_value = True
    
    result = cli_runner.invoke(migrate, ['init'])
    assert result.exit_code == 0
    assert "✅ Migration history table already exists." in result.output
    mock_session.init_collection.assert_not_called()

def test_init_command_create_table(cli_runner, mock_session):
    """Test init command when migration table needs to be created."""
    mock_session.collection_exists.return_value = False
    
    result = cli_runner.invoke(migrate, ['init'])
    assert result.exit_code == 0
    assert "🤖Creating migration history table" in result.output
    assert "✅ Migration history table created." in result.output
    mock_session.init_collection.assert_called_once_with(MigrationHistoryModel)

def test_create_command_no_changes(cli_runner, mock_migration_manager):
    """Test create command when no schema changes are detected."""
    mock_migration_manager.detect_changes.return_value = {}
    
    result = cli_runner.invoke(migrate, ['create', 'test_migration'])
    assert result.exit_code == 0
    assert "No schema changes detected." in result.output
    mock_migration_manager.generate_migration.assert_not_called()

def test_create_command_with_changes(cli_runner, mock_migration_manager):
    """Test create command when schema changes are detected."""
    # Mock schema changes
    changes = {
        "TestModel": {
            "added": [
                SchemaField(name="new_field", dtype="VARCHAR"),
            ],
            "removed": [
                SchemaField(name="old_field", dtype="INT64"),
            ],
            "modified": [
                (
                    SchemaField(name="modified_field", dtype="VARCHAR"),
                    SchemaField(name="modified_field", dtype="INT64")
                ),
            ]
        }
    }
    mock_migration_manager.detect_changes.return_value = changes
    mock_migration_manager.generate_migration.return_value = "migrations/20240126_test_migration.py"
    
    result = cli_runner.invoke(migrate, ['create', 'test_migration'])
    assert result.exit_code == 0
    assert "Changes detected in TestModel" in result.output
    assert "+ new_field (VARCHAR)" in result.output
    assert "- old_field (INT64)" in result.output
    assert "~ modified_field: VARCHAR -> INT64" in result.output
    mock_migration_manager.generate_migration.assert_called_once_with('test_migration')

def test_create_command_prompts_for_name(cli_runner, mock_migration_manager):
    """Test create command prompts for name when not provided."""
    mock_migration_manager.detect_changes.return_value = {"TestModel": {"added": [], "removed": [], "modified": []}}
    
    result = cli_runner.invoke(migrate, ['create'], input='test_migration\n')
    assert "Enter a name for the migration" in result.output
    mock_migration_manager.generate_migration.assert_called_once_with('test_migration')

def test_rebuild_command_success(cli_runner, mock_migration_manager, mock_schema_history):
    """Test rebuild command when schema rebuild succeeds."""
    # Mock models to rebuild
    test_model = MagicMock()
    test_model.__name__ = "TestModel"
    mock_migration_manager._find_all_models.return_value = [test_model]
    
    # Mock schema building
    test_schema = Schema(name="TestModel", collection_name="test_collection", fields=[])
    mock_schema_history.build_model_schema_from_migrations.return_value = test_schema
    
    result = cli_runner.invoke(migrate, ['schema-history', 'rebuild'])
    assert result.exit_code == 0
    assert "🤖 Rebuilding schema history from migrations..." in result.output
    assert "Rebuilding schema for TestModel..." in result.output
    assert "✅ Schema rebuilt for TestModel" in result.output
    assert "✅ Successfully rebuilt all schema history files" in result.output
    
    mock_schema_history.build_model_schema_from_migrations.assert_called_once_with(test_model)
    mock_schema_history.save_model_schema.assert_called_once_with(test_schema)

def test_rebuild_command_failure(cli_runner, mock_migration_manager, mock_schema_history):
    """Test rebuild command when schema rebuild fails."""
    mock_migration_manager._find_all_models.side_effect = Exception("Failed to find models")
    
    result = cli_runner.invoke(migrate, ['schema-history', 'rebuild'])
    assert result.exit_code == 1
    assert "❌ Error rebuilding schema history: Failed to find models" in result.output 
