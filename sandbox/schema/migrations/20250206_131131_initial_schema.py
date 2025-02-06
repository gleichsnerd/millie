"""
initial_schema

Revision ID: 20250206_131131_initial_schema
Created at: 2025-02-06T13:11:31.404509
"""
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema
from millie.db.migration import Migration

class Migration_20250206_131131_initial_schema(Migration):
    """Migration for initial_schema."""

    def up(self):
        """Upgrade to this version."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        collection = self.ensure_collection("rules", fields)

    def down(self):
        """Downgrade from this version."""
        collection = Collection("rules")
        collection.drop()
