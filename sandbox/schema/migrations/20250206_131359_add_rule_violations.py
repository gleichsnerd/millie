"""
add_rule_violations

Revision ID: 20250206_131359_add_rule_violations
Created at: 2025-02-06T13:13:59.044945
"""
from pymilvus import Collection, FieldSchema, DataType, CollectionSchema
from millie.db.migration import Migration

class Migration_20250206_131359_add_rule_violations(Migration):
    """Migration for add_rule_violations."""

    def up(self):
        """Upgrade to this version."""
        collection = Collection("rules")
        collection.alter_schema(add_fields=[FieldSchema(name="priority", dtype=DataType.INT64)])

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="rule_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="penalty", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        collection = self.ensure_collection("rule_violations", fields)

    def down(self):
        """Downgrade from this version."""
        collection = Collection("rules")
        collection.alter_schema(drop_fields=["priority"])

        collection = Collection("rule_violations")
        collection.drop()
