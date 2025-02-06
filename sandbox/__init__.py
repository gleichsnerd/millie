"""Sandbox package for testing and development."""
from sandbox.models import RuleModel, RuleViolationModel

__all__ = ['RuleModel', 'RuleViolationModel']

def get_rule_model():
    from sandbox.models import get_rule_model
    return get_rule_model()

def get_rule_violation_model():
    from sandbox.models import get_rule_violation_model
    return get_rule_violation_model()

# Import models for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sandbox.models import RuleModel, RuleViolationModel 