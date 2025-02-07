"""RAG operations manager."""
import os
from typing import List, Optional
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

def generate_embedding_text_embedding_3_small(text: str) -> List[float]:
    """Generate an embedding for a text string.
    
    Args:
        text: Text to generate embedding for
    Returns:
        List of floats representing the embedding
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    
    return response.data[0].embedding
