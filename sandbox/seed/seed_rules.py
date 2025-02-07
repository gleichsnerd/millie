"""Example seed function for RuleViolation model."""
from millie.db.milvus_seeder import milvus_seeder
from millie.embedders.openai import generate_embedding_text_embedding_3_small
from sandbox.models.rule import RuleModel

@milvus_seeder
def seed_rules():
    """Seed example rules into Milvus."""
    # Create some example rules
    rules = [
        RuleModel(
            id="rule_1",
            name="Use of hardcoded credentials",
            type="security",
            description="Use of hardcoded credentials",
            section="security",
            metadata={
                "framework": "general",
                "category": "security"
            },
            embedding=generate_embedding_text_embedding_3_small("Use of hardcoded credentials")
        ),
        RuleModel(
            id="rule_2",
            name="Inefficient database query pattern",
            type="performance",
            description="Inefficient database query pattern",
            section="performance",
            metadata={
                "framework": "django",
                "category": "performance"
            },
            embedding=generate_embedding_text_embedding_3_small("Inefficient database query pattern")
        ),
        RuleModel(
            id="rule_3",
            name="Missing error handling",
            type="reliability",
            description="Missing error handling",
            section="reliability",
            metadata={
                "framework": "general",
                "category": "reliability"
            },
            embedding=generate_embedding_text_embedding_3_small("Missing error handling")
        )
    ]
    
    return rules
   