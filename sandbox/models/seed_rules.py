"""Example seed function for RuleViolation model."""
from millie.db.milvus_seeder import milvus_seeder
from sandbox.models.rule import RuleModel
from sandbox.models.rule_violation import RuleViolationModel

@milvus_seeder
def seed_rules():
    """Seed example rules into Milvus."""
    # Create some example rules
    rules = [
        RuleModel(
            id="rule_1",
            embedding=[0.1] * 1536,  # Example 1536-dim embedding
            name="Use of hardcoded credentials",
            type="security",
            description="Use of hardcoded credentials",
            section="security",
            metadata={
                "framework": "general",
                "category": "security"
            }
        ),
        RuleModel(
            id="rule_2",
            embedding=[0.2] * 1536,  # Example 1536-dim embedding
            name="Inefficient database query pattern",
            type="performance",
            description="Inefficient database query pattern",
            section="performance",
            metadata={
                "framework": "django",
                "category": "performance"
            }
        ),
        RuleModel(
            id="rule_3",
            embedding=[0.3] * 1536,  # Example 1536-dim embedding
            name="Missing error handling",
            type="reliability",
            description="Missing error handling",
            section="reliability",
            metadata={
                "framework": "general",
                "category": "reliability"
            }
        )
    ]
    
    return rules
   