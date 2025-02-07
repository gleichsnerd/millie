"""Examples of querying models from Milvus collections."""
import os
import sys
from typing import List, Optional

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from millie.embedders.openai import generate_embedding_text_embedding_3_small
from sandbox.models.rule import RuleModel
from sandbox.models.rule_violation import RuleViolationModel

def get_rule_by_id(rule_id: str) -> Optional[RuleModel]:
    """Get a single rule by its ID.
    
    Args:
        rule_id: The ID of the rule to fetch
        
    Returns:
        The rule if found, None otherwise
    """
    return RuleModel.get_by_id(rule_id)

def search_rules_by_similarity(query_text: str, limit: int = 5) -> List[RuleModel]:
    """Search for rules similar to the query text.
    
    Args:
        query_text: The text to search for
        limit: Maximum number of results to return
        
    Returns:
        List of rules sorted by similarity
    """
    # Generate embedding for query text
    query_embedding = generate_embedding_text_embedding_3_small(query_text)
    
    return RuleModel.search_by_similarity(
        query_embedding=query_embedding,
        limit=limit
    )

def filter_rules_by_type(rule_type: str) -> List[RuleModel]:
    """Get all rules of a specific type.
    
    Args:
        rule_type: The type of rules to fetch
        
    Returns:
        List of matching rules
    """
    return RuleModel.filter(type=rule_type)

def get_rule_violations_by_rule(rule_id: str) -> List[RuleViolationModel]:
    """Get all violations for a specific rule.
    
    Args:
        rule_id: The ID of the rule to get violations for
        
    Returns:
        List of rule violations
    """
    return RuleViolationModel.filter(rule_id=rule_id)

def search_rules_hybrid(
    query_text: str,
    rule_type: Optional[str] = None,
    limit: int = 5
) -> List[RuleModel]:
    """Search for rules using both vector similarity and filters.
    
    Args:
        query_text: The text to search for
        rule_type: Optional type to filter by
        limit: Maximum number of results to return
        
    Returns:
        List of rules sorted by similarity
    """
    # Generate embedding for query text
    query_embedding = generate_embedding_text_embedding_3_small(query_text)
    
    # Build expression for type filter
    expr = f'type == "{rule_type}"' if rule_type else None
    
    return RuleModel.search_by_similarity(
        query_embedding=query_embedding,
        limit=limit,
        expr=expr
    )

if __name__ == "__main__":
    # Example usage
    print("\nFetching rule by ID:")
    rule = get_rule_by_id("rule_1")
    if rule:
        print(f"Found rule: {rule.id}")
    
    print("\nSearching rules by similarity:")
    similar_rules = search_rules_by_similarity("security credentials")
    for rule in similar_rules:
        print(f"Similar rule: {rule.id}")
    
    print("\nFiltering rules by type:")
    security_rules = filter_rules_by_type("security")
    for rule in security_rules:
        print(f"Security rule: {rule.id}")
    
    print("\nGetting violations for a rule:")
    violations = get_rule_violations_by_rule("rule_1")
    for violation in violations:
        print(f"Violation: {violation.description}")
    
    print("\nHybrid search:")
    hybrid_results = search_rules_hybrid("security credentials", rule_type="security")
    for rule in hybrid_results:
        print(f"Hybrid result: {rule.id}") 