import sys
import pytest

def click_skip_py310():
    """Decorator to skip tests if the Python version is less than the specified minimum version."""
    def decorator(func):
        return pytest.mark.skipif(sys.version_info < (3, 11), reason="Click MultiCommand nesting is failing in tests for Python 3.10")(func)
    return decorator
