"""
add_test_field

Revision ID: 20250126_013214_add_test_field
Created at: 2025-01-26T01:32:14.062389
"""
from pymilvus import Collection, FieldSchema, DataType
from millie.db.migration import Migration

class Migration_20250126_013214_add_test_field(Migration):
    """Migration for add_test_field."""

    def up(self):
        """Upgrade to this version."""
        collection = Collection("rules")
        collection.alter_schema(add_fields=[FieldSchema(name="test_field", dtype=DataType.VARCHAR, max_length=1000)])

    def down(self):
        """Downgrade from this version."""
        collection = Collection("rules")
        collection.alter_schema(drop_fields=["test_field"])
