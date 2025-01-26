from dataclasses import dataclass
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel
from millie.db.migration_manager import MigrationManager
import os

@dataclass
class SimpleModel:
    field1: str
    field2: int

@MilvusModel()  # Need to call the decorator function
class TestModel(SimpleModel):
    pass

class MixinClass:
    pass

# This should be detected since it inherits from BaseModel
class MultiInheritanceModel(BaseModel, MixinClass):
    pass

class NonBaseModel:
    pass

def test_find_all_models(tmp_path):
    # Create a temporary module
    module_path = tmp_path / "test_models.py"
    module_content = '''
from dataclasses import dataclass
from millie.orm.base_model import BaseModel
from millie.orm.milvus_model import MilvusModel

@dataclass
class SimpleModel:
    field1: str
    field2: int

@MilvusModel()  # Need to call the decorator function
class TestModel(SimpleModel):
    pass

class MixinClass:
    pass

# This should be detected since it inherits from BaseModel
class MultiInheritanceModel(BaseModel, MixinClass):
    pass

class NonBaseModel:
    pass
'''
    module_path.write_text(module_content)

    # Set MILLIE_MODEL_GLOB to point to our test file
    os.environ['MILLIE_MODEL_GLOB'] = str(module_path)

    # Create MigrationManager instance
    manager = MigrationManager()
    
    # Find models
    models = manager._find_all_models()

    # Should find both TestModel and MultiInheritanceModel since both inherit from BaseModel
    assert len(models) == 2
    model_names = {model.__name__ for model in models}
    assert "TestModel" in model_names
    assert "MultiInheritanceModel" in model_names

    # Clean up
    del os.environ['MILLIE_MODEL_GLOB'] 