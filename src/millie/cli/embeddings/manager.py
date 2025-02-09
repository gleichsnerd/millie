#!/usr/bin/env python3
import sys
import click
from click import echo

from millie.db.embedding_manager import EmbeddingManager

@click.group()
def embeddings():
    """
    Manage embeddings for Milvus collections.
    
    Commands for managing embeddings:
    - update: Update embeddings for all collections using registered embedders
    """
    pass

@embeddings.command()
def update():
    """Update embeddings for all collections using registered embedders."""
    try:
        manager = EmbeddingManager()
        results = manager.run_embedders()
        
        # Check for any errors
        has_errors = any(
            result.get("status") == "error" 
            for result in results.values()
        )
        
        if has_errors:
            echo("❌ Some embedders failed:")
            for name, result in results.items():
                if result.get("status") == "error":
                    echo(f"  - {name}: {result.get('error')}")
            sys.exit(1)
        else:
            echo("✅ All embeddings updated successfully!")
            sys.exit(0)
            
    except Exception as e:
        echo(f"❌ Error updating embeddings: {str(e)}", err=True)
        sys.exit(1) 