import os
import pytest
import click
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from millie.cli.router import cli, add_millie_commands
from millie.db.migration_history import MigrationHistoryModel
from millie.schema.schema import Schema, SchemaField
from millie.db.schema_history import SchemaHistory
from tests.util import click_skip_py310

@pytest.fixture
def test_cli():
    """Create a fresh CLI for each test."""
    test_cli = click.Group(name='cli')
    add_millie_commands(test_cli)
    return test_cli

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner."""
    return CliRunner()

@pytest.fixture
def mock_session():
    """Create a mock MilvusSession."""
    session_mock = MagicMock()
    session_mock.connect.return_value = True
    session_mock.collection_exists.return_value = True
    session_mock.init_collection.return_value = None
    session_mock.unload_collection.return_value = None
    
    with patch('millie.db.session.MilvusSession', return_value=session_mock) as mock:
        with patch('millie.cli.migrate.manager.MilvusSession', return_value=session_mock):
            yield mock

@pytest.fixture
def mock_migration_manager():
    """Create a mock MigrationManager."""
    with patch('millie.db.migration_manager.MigrationManager') as mock:
        mock.return_value = MagicMock()
        mock.return_value._find_all_models.return_value = []
        mock.return_value.detect_changes.return_value = {}
        mock.return_value.generate_migration.return_value = "test_migration.py"
        yield mock

@pytest.fixture
def mock_schema_history():
    """Create a mock SchemaHistory."""
    with patch('millie.db.schema_history.SchemaHistory') as mock:
        mock.return_value = MagicMock()
        mock.return_value.build_model_schema_from_migrations.return_value = {}
        mock.return_value.save_model_schema.return_value = None
        yield mock

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Set required environment variables."""
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")
    monkeypatch.setenv("MILVUS_DB_NAME", "millie_test")
    monkeypatch.setenv("MILLIE_SCHEMA_DIR", "schema")

@click_skip_py310()
def test_init_command_table_exists(test_cli, cli_runner, mock_session):
    """Test initializing migrations when table exists."""
    result = cli_runner.invoke(test_cli, ["migrate", "init"])
    assert result.exit_code == 0
    assert "✅ Migration history table already exists" in result.output

@click_skip_py310()
def test_init_command_create_table(test_cli, cli_runner, mock_session):
    """Test initializing migrations when table doesn't exist."""
    mock_session.return_value.collection_exists.return_value = False
    result = cli_runner.invoke(test_cli, ["migrate", "init"])
    assert result.exit_code == 0
    assert "✅ Migration history table created" in result.output

@click_skip_py310()
def test_create_command_no_changes(test_cli, cli_runner, mock_migration_manager):
    """Test creating a migration when there are no changes."""
    mock_migration_manager.return_value.detect_changes.return_value = {}
    result = cli_runner.invoke(test_cli, ["migrate", "create"], input="test\n")
    assert result.exit_code == 0
    assert "No schema changes detected" in result.output

@click_skip_py310()
def test_create_command_with_changes(test_cli, cli_runner, mock_migration_manager):
    """Test creating a migration when there are changes."""
    mock_migration_manager.return_value.detect_changes.return_value = {
        "TestModel": {
            "added": [SchemaField(name="new_field", dtype="VARCHAR")],
            "removed": [],
            "modified": []
        }
    }
    mock_migration_manager.return_value.generate_migration.return_value = "test_migration.py"
    
    result = cli_runner.invoke(test_cli, ["migrate", "create", "--name", "test"])
    assert result.exit_code == 0
    assert "✅ Created migration" in result.output

@click_skip_py310()
def test_create_command_prompts_for_name(test_cli, cli_runner, mock_migration_manager):
    """Test creating a migration prompts for name if not provided."""
    mock_migration_manager.return_value.detect_changes.return_value = {
        "TestModel": {
            "added": [SchemaField(name="new_field", dtype="VARCHAR")],
            "removed": [],
            "modified": []
        }
    }
    mock_migration_manager.return_value.generate_migration.return_value = "test_migration.py"
    
    result = cli_runner.invoke(test_cli, ["migrate", "create"], input="test\n")
    assert result.exit_code == 0
    assert "Enter a name for the migration" in result.output
    assert "✅ Created migration" in result.output

@click_skip_py310()
def test_schema_history_rebuild_success(test_cli, cli_runner, mock_migration_manager, mock_schema_history):
    """Test rebuilding schema history successfully."""
    mock_migration_manager.return_value._find_all_models.return_value = [MagicMock()]
    result = cli_runner.invoke(test_cli, ["migrate", "schema-history", "rebuild"])
    assert result.exit_code == 0
    assert "✅ Successfully rebuilt all schema history files" in result.output

@click_skip_py310()
def test_schema_history_rebuild_failure(test_cli, cli_runner, mock_migration_manager, mock_schema_history):
    """Test rebuilding schema history with failure."""
    mock_migration_manager.return_value._find_all_models.return_value = [MagicMock()]
    mock_schema_history.return_value.build_model_schema_from_migrations.side_effect = Exception("Test error")
    result = cli_runner.invoke(test_cli, ["migrate", "schema-history", "rebuild"])
    assert result.exit_code == 1
    assert "❌ Error rebuilding schema history: Test error" in result.output

@click_skip_py310()
def test_init_command_db_connection_error(test_cli, cli_runner, mock_session):
    """Test initialization when database connection fails."""
    mock_session.return_value.connect.side_effect = Exception("Connection failed")
    result = cli_runner.invoke(test_cli, ["migrate", "init"])
    assert result.exit_code == 1
    assert "❌ Error connecting to database" in result.output 