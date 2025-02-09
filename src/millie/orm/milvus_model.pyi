from typing import TypeVar, Type, Dict, Any, List, Optional, ClassVar, Generic, Union
from abc import ABC
from datetime import datetime
from dataclasses import Field
from pymilvus import DataType, Collection, Hit

T = TypeVar('T', bound='MilvusModel')

class MilvusModel(ABC, Generic[T]):
    is_migration_collection: ClassVar[bool]
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]

    def __init__(self, **kwargs: Any) -> None: ...
    
    def __post_init__(self) -> None: ...
    
    @classmethod
    def __class_getitem__(cls, key: Any) -> Type['MilvusModel']: ...
    
    @classmethod
    def collection_name(cls) -> str: ...
    
    @classmethod
    def schema(cls) -> Dict[str, Any]: ...
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T: ...
    
    @classmethod
    def get_all_models(cls) -> List[Type['MilvusModel']]: ...
    
    @classmethod
    def get_model(cls, name: str) -> Optional[Type['MilvusModel']]: ...
    
    def serialize_for_json(self) -> str: ...
    
    @classmethod
    def deserialize_from_json(cls: Type[T], json_str: str) -> T: ...
    
    @classmethod
    def load(cls) -> None: ...
    
    @classmethod
    def unload(cls) -> None: ...
    
    def save(self) -> bool: ...
    
    @classmethod
    def bulk_insert(cls: Type[T], models: List[T], batch_size: int = 100) -> bool: ...
    
    @classmethod
    def bulk_upsert(cls: Type[T], models: List[T], batch_size: int = 100) -> bool: ...
    
    def delete(self) -> bool: ...
    
    @classmethod
    def delete_many(cls, expr: str) -> bool: ...
    
    @classmethod
    def get_all(
        cls: Type[T],
        offset: int = 0,
        limit: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> List[T]: ...
    
    @classmethod
    def get_by_id(cls: Type[T], id: str, output_fields: Optional[List[str]] = None) -> Optional[T]: ...
    
    @classmethod
    def filter(cls: Type[T], output_fields: Optional[List[str]] = None, **kwargs) -> List[T]: ...
    
    @classmethod
    def search_by_similarity(
        cls: Type[T],
        query_embedding: List[float],
        limit: int = 5,
        expr: Optional[str] = None,
        metric_type: str = "L2",
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[T]: ... 