from dataclasses import field as dataclass_field, InitVar
from typing import Any, Callable, TypeAlias, TypeVar, overload, Union
from pymilvus import DataType

T = TypeVar('T')

# Type aliases to help the type checker understand field assignments
FieldType: TypeAlias = Union[T, InitVar[T]]  # This helps type checker understand these become parameters

class MilvusFieldInfo:
    """Stores Milvus field configuration."""
    def __init__(self, data_type: DataType, **kwargs):
        self.data_type = data_type
        self.kwargs = kwargs

@overload
def milvus_field(data_type: DataType, *, default: T, **kwargs) -> FieldType[T]: ...

@overload
def milvus_field(data_type: DataType, *, default_factory: Callable[[], T], **kwargs) -> FieldType[T]: ...

@overload
def milvus_field(data_type: DataType, *, default: Callable[..., T], **kwargs) -> FieldType[T]: ...

@overload
def milvus_field(data_type: DataType, **kwargs) -> Any: ...

def milvus_field(data_type: DataType, *, default=None, default_factory=None, **kwargs):
    """Create a field with Milvus configuration.
    
    Usage:
    field: int = milvus_field(DataType.INT64)
    field: str = milvus_field(DataType.VARCHAR, max_length=100)
    field: Dict = milvus_field(DataType.JSON, default_factory=dict)
    timestamp: datetime = milvus_field(DataType.VARCHAR, max_length=30, default=datetime.now)
    embedding: Optional[List[float]] = milvus_field(DataType.FLOAT_VECTOR, dim=1536)
    
    These fields become parameters in the model's __init__ method:
    model = MyModel(
        field="value",
        timestamp=datetime.now(),
        embedding=[1.0, 2.0, 3.0]
    )
    
    Args:
        data_type: The Milvus DataType for this field
        default: Default value or callable for the field
        default_factory: Callable that returns a default value
        **kwargs: Additional arguments to pass to FieldSchema
        
    Returns:
        Field definition with Milvus metadata
    """
    if default_factory is not None:
        return dataclass_field(
            default_factory=default_factory,
            metadata={'milvus': MilvusFieldInfo(data_type, **kwargs)}
        )
    if callable(default) and not isinstance(default, type):
        return dataclass_field(
            default_factory=default,
            metadata={'milvus': MilvusFieldInfo(data_type, **kwargs)}
        )
    return dataclass_field(
        default=default,
        metadata={'milvus': MilvusFieldInfo(data_type, **kwargs)}
    )
