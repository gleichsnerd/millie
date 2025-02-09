"""RAG operations manager."""
import os
from typing import List
import logging

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

def generate_embedding_text_embedding_3_small(text: str) -> List[float]:
    """Generate an embedding for a text string.
    
    Args:
        text: Text to generate embedding for
    Returns:
        List of floats representing the embedding
    Raises:
        ImportError: If OpenAI package is not installed
        ValueError: If OPENAI_API_KEY is not set
    """
    if not HAS_OPENAI:
        raise ImportError(
            "OpenAI package is not installed. "
            "Please install it with 'pip install millie[openai]'"
        )
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding

def generate_embedding_text_embedding_3_large(text: str) -> List[float]:
    """Generate an embedding for a text string using text-embedding-3-large.
    
    Args:
        text: Text to generate embedding for
    Returns:
        List of floats representing the embedding
    Raises:
        ImportError: If OpenAI package is not installed
        ValueError: If OPENAI_API_KEY is not set
    """
    if not HAS_OPENAI:
        raise ImportError(
            "OpenAI package is not installed. "
            "Please install it with 'pip install millie[openai]'"
        )
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    
    return response.data[0].embedding

def generate_embedding_text_embedding_ada_002(text: str) -> List[float]:
    """Generate an embedding for a text string using text-embedding-ada-002.
    
    Args:
        text: Text to generate embedding for
    Returns:
        List of floats representing the embedding
    Raises:
        ImportError: If OpenAI package is not installed
        ValueError: If OPENAI_API_KEY is not set
    """
    if not HAS_OPENAI:
        raise ImportError(
            "OpenAI package is not installed. "
            "Please install it with 'pip install millie[openai]'"
        )
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        model="text-embedding-ada-002", 
        input=text
    )
    
    return response.data[0].embedding
