import os
from pathlib import Path
from datetime import datetime
import json
from typing import List, Optional

from millie.db.schema_differ import SchemaDiffer

class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, schema_dir: str = os.getenv('MILLIE_SCHEMA_DIR', os.path.join(str(Path(__file__).parent),'schema'))):
        self.schema_dir = schema_dir
        os.makedirs(self.schema_dir, exist_ok=True)
        self.migrations_dir = os.path.join(self.schema_dir, 'migrations')
        os.makedirs(self.migrations_dir, exist_ok=True)
        self.history_dir = os.path.join(self.schema_dir, 'history')
        os.makedirs(self.history_dir, exist_ok=True)

    def generate_migration(self, name: str) -> str:
        """Generate a new migration file using schema change detection"""
        schema_differ = SchemaDiffer()
        schema_differ.detect_schema_changes()
        return self.create_migration_file(name)
    
    def create_migration_file(self, name: str) -> str:
        """Generate a new migration file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{name}.json"
        filepath = os.path.join(self.migrations_dir, filename)
        
        migration_template = {
            "version": timestamp,
            "name": name,
            "up": [],
            "down": []
        }
        
        with open(filepath, 'w') as f:
            json.dump(migration_template, f, indent=2)
        
        return filepath
    
    def run_migrations(self) -> List[str]:
        """Run all pending migrations."""
        applied = []
        migrations = self._get_pending_migrations()
        
        for migration in migrations:
            try:
                self._apply_migration(migration)
                applied.append(migration)
            except Exception as e:
                raise Exception(f"Failed to apply migration {migration}: {str(e)}")
        
        return applied
    
    def _get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations."""
        if not os.path.exists(self.migrations_dir):
            return []
        
        migrations = []
        for filename in sorted(os.listdir(self.migrations_dir)):
            if filename.endswith('.json'):
                migrations.append(os.path.join(self.migrations_dir, filename))
        
        return migrations
    
    def _apply_migration(self, migration_file: str):
        """Apply a single migration."""
        with open(migration_file, 'r') as f:
            migration = json.load(f)
        
        # Apply each operation in the migration
        for operation in migration['up']:
            self._apply_operation(operation)
    
    def _apply_operation(self, operation: dict):
        """Apply a single migration operation."""
        # TODO: Implement actual migration operations
        # This will depend on what kind of migrations we need to support
        pass 