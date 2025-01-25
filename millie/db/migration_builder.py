import os
from pathlib import Path
from datetime import datetime
import json
from typing import List

class MigrationBuilder:
    """Handles the creation of migration files."""
    
    def __init__(self, migrations_dir: str = None):
        self.migrations_dir = migrations_dir or str(Path(__file__).parent / 'migrations')
        os.makedirs(self.migrations_dir, exist_ok=True)
    
    def generate_migration(self, name: str) -> str:
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