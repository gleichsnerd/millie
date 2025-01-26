import click

from millie.cli.router import add_millie_commands

def test_add_millie_commands():
    """Test that add_millie_commands adds all expected commands."""
    test_cli = click.Group(name='cli')
    add_millie_commands(test_cli)
    
    # Verify all expected commands are added
    assert 'migrate' in test_cli.commands
    assert 'milvus' in test_cli.commands
    assert 'attu' in test_cli.commands
    assert 'db' in test_cli.commands
    
    # Verify migrate command is a group with subcommands
    migrate_cmd = test_cli.commands['migrate']
    assert isinstance(migrate_cmd, click.Group)
    assert 'init' in migrate_cmd.commands
    assert 'create' in migrate_cmd.commands
    assert 'schema-history' in migrate_cmd.commands
