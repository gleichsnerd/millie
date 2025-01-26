"""
initial_schema

Revision ID: 20250125_145815_initial_schema
Created at: 2025-01-25T14:58:15.672007
"""
from pymilvus import Collection, FieldSchema, DataType
from millie.db.migration import Migration

class Migration_20250125_145815_initial_schema(Migration):
    """Migration for initial_schema."""

    def up(self):
        """Upgrade to this version."""
        collection = Collection("rules")
        collection.alter_schema(add_fields=[FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True)])
        collection.alter_schema(add_fields=[FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50)])
        collection.alter_schema(add_fields=[FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=50)])
        collection.alter_schema(add_fields=[FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000)])
        collection.alter_schema(add_fields=[FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)])
        collection.alter_schema(add_fields=[FieldSchema(name="metadata", dtype=DataType.JSON)])
        collection.alter_schema(add_fields=[FieldSchema(name="priority", dtype=DataType.INT64)])

    def down(self):
        """Downgrade from this version."""
        collection = Collection("rules")
        collection.alter_schema(drop_fields=["id"])
        collection.alter_schema(drop_fields=["type"])
        collection.alter_schema(drop_fields=["section"])
        collection.alter_schema(drop_fields=["description"])
        collection.alter_schema(drop_fields=["embedding"])
        collection.alter_schema(drop_fields=["metadata"])
        collection.alter_schema(drop_fields=["priority"])
