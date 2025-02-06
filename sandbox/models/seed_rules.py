"""Example seed function for RuleViolation model."""
from millie.db.milvus_seeder import milvus_seeder
from sandbox.models.rule_violation import RuleViolationModel

@milvus_seeder
def seed_rule_violations():
    """Seed example rule violations into Milvus."""
    # Create some example rule violations
    violations = [
        RuleViolationModel(
            id="rule_1",
            embedding=[0.1] * 1536,  # Example 1536-dim embedding
            rule_id="R001",
            penalty="critical",
            description="Use of hardcoded credentials",
            metadata={
                "framework": "general",
                "category": "security"
            }
        ),
        RuleViolationModel(
            id="rule_2",
            embedding=[0.2] * 1536,  # Example 1536-dim embedding
            rule_id="R002",
            penalty="warning",
            description="Inefficient database query pattern",
            metadata={
                "framework": "django",
                "category": "performance"
            }
        ),
        RuleViolationModel(
            id="rule_3",
            embedding=[0.3] * 1536,  # Example 1536-dim embedding
            rule_id="R003",
            penalty="info",
            description="Missing error handling",
            metadata={
                "framework": "general",
                "category": "reliability"
            }
        )
    ]
    
    return violations
   