"""Sentence Transformers embedder implementation."""
import logging
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)

def generate_embedding_all_MiniLM_L6_v2(text: str) -> List[float]:
    """Generate an embedding for a text string using all-MiniLM-L6-v2.
    
    Args:
        text: Text to generate embedding for
    Returns:
        List of floats representing the embedding
    Raises:
        ImportError: If sentence-transformers package is not installed
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError(
            "sentence-transformers package is not installed. "
            "Please install it with 'pip install millie[sentence-transformers]'"
        )
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding.tolist()